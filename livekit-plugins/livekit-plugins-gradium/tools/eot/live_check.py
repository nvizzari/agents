"""Live end-to-end check of `STTExperimental` against production Gradium, without a room.

    python tools/eot/live_check.py --per-cohort 10 --other fr ja de --concurrency 6

Each dataset turn is pushed through `STTExperimental(...).stream()` in 20 ms frames at
real-time pacing, followed by `--tail` seconds of digital silence, then `end_input()`.
Every `SpeechEvent` is recorded with its wall time. Scenarios, assigned round-robin:

- `plain`: nothing else happens.
- `reconnect`: halfway through the audio, `update_options(keywords=...)` changes a setup
  option, which drains the connection and opens a new one.
- `manual`: at the first hold that lasts at least 0.4 s, `stream.flush()` is called, as
  `AgentSession.commit_user_turn` does; a FINAL_TRANSCRIPT must answer it.
- `end_input`: `end_input()` is called halfway through the audio, mid-speech.

Checks, per stream:

- event order: each user turn is START_OF_SPEECH, then INTERIM / PREFLIGHT events, then
  FINAL_TRANSCRIPT (absent only when no word was heard), then END_OF_SPEECH. A reconnect
  emits a FINAL mid-turn and the turn goes on: the framework appends consecutive finals;
- a turn that ends right after a preflight (no interim since) has a final equal to it;
- every timestamp lies on the caller's timeline: between `start_time_offset` and the
  offset plus the audio pushed so far; END_OF_SPEECH ends where the final just before it
  ends;
- the scenario's own expectation (two connections; an answered flush; a clean end).

It also measures, at real-time pacing, the wall time from each `flush` to its `flushed`
(the simulator's `--flush-rtt`), and when END_OF_SPEECH reached the caller relative to
the annotated end-of-turn silence.

Output: `data/live/<run>/events.jsonl`, one line per turn, and a summary on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiohttp  # noqa: E402
import numpy as np  # noqa: E402
from collect import language_hint  # noqa: E402
from common import DATA_DIR, import_harness, iter_rows, load_api_key  # noqa: E402

from livekit import rtc  # noqa: E402
from livekit.agents import APIConnectOptions  # noqa: E402
from livekit.agents.stt import SpeechEventType  # noqa: E402
from livekit.plugins.gradium.experimental import signals  # noqa: E402
from livekit.plugins.gradium.experimental.machine import SendFlush  # noqa: E402
from livekit.plugins.gradium.experimental.stt import (  # noqa: E402
    SAMPLE_RATE,
    STTExperimental,
)

FRAME_MS = 20
SCENARIOS = ("plain", "reconnect", "manual", "end_input")
OFFSET = 100.0  # start_time_offset: the caller's timeline began this long ago
EPS = 0.05
E = SpeechEventType


@dataclass
class Recorded:
    wall: float
    type: str
    text: str = ""
    start: float | None = None
    end: float | None = None


@dataclass
class TurnRun:
    id: str
    language: str
    cohort: str
    scenario: str
    audio_s: float
    eot_start: float | None
    events: list[Recorded] = field(default_factory=list)
    flushes: list[dict[str, Any]] = field(default_factory=list)
    readies: int = 0
    action_at: float | None = None
    """Wall time (seconds since the first frame) of the scenario's action."""
    error: str | None = None
    problems: list[str] = field(default_factory=list)


def pick_turns(per_cohort: int, others: list[str], max_s: float) -> list[dict[str, Any]]:
    """English turns per cohort plus a few of other languages, fixed across runs."""
    cols = ["id", "audio", "cohort", "language", "silence_spans"]
    rows: list[dict[str, Any]] = []
    en = list(iter_rows("en", ["id", "cohort", "silence_spans"]))
    rng = random.Random("live_check")
    by_cohort: dict[str, list[str]] = {}
    for r in en:
        by_cohort.setdefault(str(r["cohort"]), []).append(str(r["id"]))
    want = set()
    for ids in by_cohort.values():
        want.update(rng.sample(sorted(ids), min(len(ids), per_cohort * 3)))
    taken: dict[str, int] = {}
    for r in iter_rows("en", cols):
        c = str(r["cohort"])
        if r["id"] in want and taken.get(c, 0) < per_cohort and _duration(r) <= max_s:
            rows.append(r)
            taken[c] = taken.get(c, 0) + 1
    for lang in others:
        for r in iter_rows(lang, cols):
            if _duration(r) <= max_s:
                rows.append(r)
                break
    return rows


def _duration(row: dict[str, Any]) -> float:
    spans = row["silence_spans"]
    return float(spans[-1]["end"]) if spans else 0.0


def _pcm(row: dict[str, Any]) -> tuple[np.ndarray, float]:
    import_harness()
    from eot_harness.streaming_stt import prepare_pcm16_audio

    pcm, audio_s = prepare_pcm16_audio(row, sample_rate=SAMPLE_RATE)
    return np.frombuffer(pcm, dtype=np.int16), audio_s


def _frame(samples: np.ndarray) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=len(samples),
    )


async def run_turn(
    row: dict[str, Any],
    scenario: str,
    http: aiohttp.ClientSession,
    key: str,
    args: argparse.Namespace,
) -> TurnRun:
    audio, audio_s = _pcm(row)
    spans = row["silence_spans"]
    eot = [
        s
        for i, s in enumerate(spans)
        if (s.get("label") or ("eot" if i == len(spans) - 1 else "hold")) == "eot"
    ]
    run = TurnRun(
        id=str(row["id"]),
        language=str(row["language"]),
        cohort=str(row["cohort"]),
        scenario=scenario,
        audio_s=audio_s,
        eot_start=float(eot[0]["start"]) if eot else None,
    )
    provider = STTExperimental(
        api_key=key,
        http_session=http,
        eagerness=args.eagerness,
        overrides=_parse_overrides(args.set),
        language=language_hint(str(row["language"])),
    )
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=3, timeout=10))
    stream.start_time_offset = OFFSET
    t0 = time.monotonic()

    def now() -> float:
        return time.monotonic() - t0

    # instrumentation: flush timing and connections, without changing behaviour
    machine = stream.machine
    original_on_signal = machine.on_signal
    original_apply = stream._apply
    pending: dict[int, dict[str, Any]] = {}

    def on_signal(sig: signals.Signal) -> Any:
        if isinstance(sig, signals.Flushed) and sig.flush_id in pending:
            pending.pop(sig.flush_id)["reply_wall"] = now()
        elif isinstance(sig, signals.Ready):
            run.readies += 1
        return original_on_signal(sig)

    async def apply(ws: Any, actions: Any) -> None:
        for a in actions:
            if isinstance(a, SendFlush):
                rec = {
                    "id": a.flush_id,
                    "sent_wall": now(),
                    "audio": machine.audio_time,
                    "state": machine.state.value,
                }
                run.flushes.append(rec)
                pending[a.flush_id] = rec
        await original_apply(ws, actions)

    machine.on_signal = on_signal  # type: ignore[method-assign]
    stream._apply = apply  # type: ignore[method-assign]

    async def consume() -> None:
        async for ev in stream:
            alt = ev.alternatives[0] if ev.alternatives else None
            run.events.append(
                Recorded(
                    wall=now(),
                    type=ev.type.value,
                    text=alt.text if alt else "",
                    start=alt.start_time if alt else None,
                    end=alt.end_time if alt else None,
                )
            )

    consumer = asyncio.create_task(consume())
    n = SAMPLE_RATE * FRAME_MS // 1000
    tail = np.zeros(int(args.tail * SAMPLE_RATE), dtype=np.int16)
    samples = np.concatenate([audio, tail])
    action_sample = len(audio) // 2
    if scenario == "manual":
        holds = [s for s in spans[:-1] if float(s["end"]) - float(s["start"]) >= 0.4]
        action_sample = (
            int((float(holds[0]["start"]) + 0.3) * SAMPLE_RATE) if holds else len(audio) // 2
        )
    try:
        acted = False
        for i in range(0, len(samples), n):
            target = i / SAMPLE_RATE
            delay = target - now()
            if delay > 0:
                await asyncio.sleep(delay)
            if not acted and i >= action_sample and scenario != "plain":
                acted = True
                run.action_at = now()
                if scenario == "reconnect":
                    provider.update_options(keywords=["LiveKit"])
                elif scenario == "manual":
                    stream.flush()
                elif scenario == "end_input":
                    stream.end_input()
                    break
            stream.push_frame(_frame(samples[i : i + n]))
        else:
            stream.end_input()
        await asyncio.wait_for(consumer, timeout=args.tail + 20)
    except Exception as e:  # noqa: BLE001
        run.error = repr(e)
    finally:
        await stream.aclose()
    run.problems = check(run, pushed_s=len(samples) / SAMPLE_RATE)
    return run


def _parse_overrides(items: list[str]) -> Any:
    from common import parse_overrides

    return parse_overrides(items) or None


def check(run: TurnRun, *, pushed_s: float) -> list[str]:
    """Every property the framework relies on; returns what failed."""
    problems: list[str] = []
    if run.error:
        problems.append(f"error: {run.error}")
    open_turn = False
    words_seen = False
    last: Recorded | None = None
    final: Recorded | None = None
    for ev in run.events:
        if ev.type == E.START_OF_SPEECH.value:
            if open_turn:
                problems.append("START_OF_SPEECH inside an open turn")
            open_turn, words_seen, final, last = True, False, None, None
        elif ev.type in (E.INTERIM_TRANSCRIPT.value, E.PREFLIGHT_TRANSCRIPT.value):
            if not open_turn:
                problems.append(f"{ev.type} outside a turn")
            # after a reconnect, a turn goes on past a final: the framework appends finals
            final = None
            words_seen = True
            last = ev
        elif ev.type == E.FINAL_TRANSCRIPT.value:
            if not open_turn and not (run.scenario == "manual" and ev.text == ""):
                problems.append("FINAL_TRANSCRIPT outside a turn")
            if last is not None and last.type == E.PREFLIGHT_TRANSCRIPT.value:
                if last.text != ev.text:
                    problems.append(f"final {ev.text!r} differs from preflight {last.text!r}")
            final = last = ev
        elif ev.type == E.END_OF_SPEECH.value:
            if not open_turn:
                problems.append("END_OF_SPEECH outside a turn")
            if words_seen and final is None:
                problems.append("END_OF_SPEECH without a final after words")
            if last is final and final is not None and ev.end is not None and final.end is not None:
                # A turn carried over a reconnect ends at the new connection's start when
                # no word follows: the old word times are on the old connection's clock.
                carried = run.scenario == "reconnect" and run.readies > 1 and not ev.text
                if abs(ev.end - final.end) > 1e-6 and not (carried and ev.end >= final.end):
                    problems.append("END_OF_SPEECH end differs from the final's")
            open_turn = False
            last = None
        for t in (ev.start, ev.end):
            if t is not None and not (OFFSET - EPS <= t <= OFFSET + pushed_s + EPS):
                problems.append(f"{ev.type} time {t:.3f} off the caller's timeline")
    if open_turn:
        problems.append("stream ended with a turn open")
    heard = any(ev.type == E.INTERIM_TRANSCRIPT.value for ev in run.events)
    if heard and not any(ev.type == E.FINAL_TRANSCRIPT.value and ev.text for ev in run.events):
        problems.append("words heard but no final transcript")
    if run.scenario == "reconnect" and run.readies < 2:
        problems.append(f"expected a reconnect, saw {run.readies} connection(s)")
    if run.scenario == "manual" and run.action_at is not None:
        answered = [
            ev
            for ev in run.events
            if ev.type == E.FINAL_TRANSCRIPT.value and ev.wall >= run.action_at
        ]
        if not answered:
            problems.append("manual flush got no final")
    return problems


async def main_async(args: argparse.Namespace) -> None:
    rows = pick_turns(args.per_cohort, args.other, args.max_s)
    key = load_api_key()
    sem = asyncio.Semaphore(args.concurrency)
    out_dir = DATA_DIR / "live" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[TurnRun] = []
    async with aiohttp.ClientSession() as http:

        async def one(i: int, row: dict[str, Any]) -> None:
            async with sem:
                runs.append(await run_turn(row, SCENARIOS[i % len(SCENARIOS)], http, key, args))

        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    with (out_dir / "events.jsonl").open("w") as f:
        for r in runs:
            f.write(json.dumps(asdict(r)) + "\n")
    report(runs, args)


def report(runs: list[TurnRun], args: argparse.Namespace) -> None:
    print(f"{len(runs)} turns, eagerness {args.eagerness}, overrides {args.set or 'none'}")
    for scenario in SCENARIOS:
        sub = [r for r in runs if r.scenario == scenario]
        bad = [r for r in sub if r.problems]
        print(f"  {scenario:10} {len(sub) - len(bad)}/{len(sub)} pass")
        for r in bad:
            print(f"    {r.id} ({r.language}): {'; '.join(sorted(set(r.problems)))}")
    rtt = [f["reply_wall"] - f["sent_wall"] for r in runs for f in r.flushes if "reply_wall" in f]
    unanswered = sum(1 for r in runs for f in r.flushes if "reply_wall" not in f)
    if rtt:
        a = np.asarray(rtt) * 1000
        print(
            f"flush -> flushed wall time: n={len(a)}, p50 {np.median(a):.0f} ms, "
            f"p90 {np.quantile(a, 0.9):.0f} ms, max {a.max():.0f} ms; {unanswered} unanswered"
        )
    pre = sum(1 for r in runs for e in r.events if e.type == E.PREFLIGHT_TRANSCRIPT.value)
    print(f"preflights: {pre}")
    lat = []
    for r in runs:
        if r.scenario != "plain" or r.eot_start is None:
            continue
        ends = [e.wall for e in r.events if e.type == E.END_OF_SPEECH.value]
        after = [w - r.eot_start for w in ends if w >= r.eot_start]
        if after:
            lat.append(after[0])
    if lat:
        a = np.asarray(lat) * 1000
        print(
            f"plain turns: END_OF_SPEECH after the annotated end-of-turn silence start, wall "
            f"time: n={len(a)}, p50 {np.median(a):.0f} ms, p90 {np.quantile(a, 0.9):.0f} ms"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--per-cohort", type=int, default=10)
    ap.add_argument("--other", nargs="*", default=["fr", "ja", "de"])
    ap.add_argument("--max-s", type=float, default=14.0)
    ap.add_argument("--tail", type=float, default=2.5, help="silence pushed after the audio")
    ap.add_argument("--eagerness", default="balanced")
    ap.add_argument("--set", nargs="*", default=[], metavar="KNOB=VALUE")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--run", default="check")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
