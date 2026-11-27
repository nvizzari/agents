"""Replay eot-bench turns through production Gradium STT and record every message.

    python tools/eot/collect.py --mode passive --lang ar de ...
    python tools/eot/collect.py --mode flush --lang en ... --weights path/to/passive.json
    python tools/eot/collect.py --mode machine --preset balanced --lang en --tag m-balanced
    python tools/eot/collect.py --mode passive --noise babble --snr 10 --subset 500 --lang en

Modes:

- `passive`: audio is sent as fast as the socket takes it, as the eot-bench adapter does.
  The probabilities are deterministic, so pacing does not change them.
- `flush`: audio is sent in lockstep with the steps, never more than `--lead` seconds
  ahead of the last step received, which is how far a live client is ahead after one
  round trip. A passive scorer (`--weights`) runs on the stream; when `EagerPolicy` fires,
  a `flush` is sent and no more audio goes out until `flushed` comes back. The flush
  therefore lands at the same audio position on every run.
- `machine`: same lockstep pacing, but the plugin's own `TurnMachine` (with the profile
  given by `--preset`, `--scorer` and `--set`) decides, and every
  `SendFlush` it returns is sent for real: eager flushes and commit flushes alike. This
  is the only way to measure a profile with preflight on, because the steps after a
  flush differ from the passive stream. `simulate.py` replays these recordings through
  the same machine and checks that it sends the same flushes at the same steps.

`--noise pink|babble --snr DB` mixes background noise into the audio first (see
`noise.py`); `--subset N` replays a fixed random sample of N turns per language, the same
on every run.

Output: `data/raw/<tag>/<lang>.jsonl` (the tag defaults to the mode, plus the noise
condition), one turn per line, appended and resumable:

    {"id", "language", "cohort", "audio_s", "mode", "settings", "ready", "messages"}

`messages` holds the server messages in arrival order (step, text, end_text, flushed),
stripped of fields nothing reads, plus `{"type": "flush_sent", "flush_id": n}` where the
client sent a flush. `common.Driver` turns them back into scorer calls.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import itertools
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiohttp  # noqa: E402
import noise  # noqa: E402
import numpy as np  # noqa: E402
from common import (  # noqa: E402
    FLUSH_SENT,
    LANGUAGES,
    Driver,
    import_harness,
    iter_rows,
    load_api_key,
    parse_overrides,
    raw_path,
)

from livekit.plugins.gradium.experimental.machine import SendFlush, TurnMachine  # noqa: E402
from livekit.plugins.gradium.experimental.profiles import TurnProfile, preset  # noqa: E402
from livekit.plugins.gradium.experimental.scorer import (  # noqa: E402
    LearnedScorer,
    Weights,
    load_scorer,
    load_weights,
)
from livekit.plugins.gradium.experimental.signals import Step, parse  # noqa: E402

URL = "wss://api.gradium.ai/api/speech/asr"
SAMPLE_RATE = 24000
FRAME_S = 0.08
FRAME_BYTES = int(SAMPLE_RATE * FRAME_S) * 2
# Benchmark languages the server has no hint for: their turns are sent with "any". This
# describes the recordings under `data/`, not the server; when the server gains one of
# these, remove it here and collect that language again (README, "Adding a language").
ANY_HINT_LANGUAGES = frozenset({"id", "tr", "zh"})
# The model delay the plugin's presets use; collections send it explicitly so a change of
# the server default cannot silently change the training data.
DEFAULT_DELAY_IN_FRAMES = 10
# Lead used by every lockstep replay: one round trip of a live client.
DEFAULT_LEAD = 0.16
_KEEP = {
    "step": ("type", "total_duration_s", "vad"),
    "text": ("type", "text", "start_s"),
    "end_text": ("type", "stop_s"),
    "flushed": ("type", "flush_id"),
}


def language_hint(lang: str) -> str:
    return "any" if lang in ANY_HINT_LANGUAGES else lang


def compact(msg: dict[str, Any]) -> dict[str, Any] | None:
    keep = _KEEP.get(msg.get("type", ""))
    if keep is None:
        return None
    out = {k: msg[k] for k in keep if k in msg}
    if "vad" in out:
        out["vad"] = [{"inactivity_prob": v["inactivity_prob"]} for v in out["vad"]]
    return out


@dataclass(frozen=True)
class EagerSettings:
    pause_threshold: float
    """Score at which a pause opens (the silence clock starts)."""
    eager_threshold: float
    """Score at which the turn has probably ended and a flush is sent."""
    min_silence: float
    """Pause age, in audio seconds, before a flush may be sent."""
    lead: float
    """Audio sent ahead of the last step received."""


class EagerPolicy:
    """When a live client would flush, decided from the passive score alone.

    A pause opens at the first step scoring at least `pause_threshold` and closes when the
    score falls below it. Inside a pause, the first step at least `min_silence` after it
    opened that scores at least `eager_threshold` triggers the flush. After a flush the
    pause stays open until the speaker resumes (a word starting after the flush), so each
    pause gets at most one flush. Nothing is flushed before the first word: the state
    machine opens a turn on a word, so leading silence is not a pause.
    """

    def __init__(self, scorer: LearnedScorer, settings: EagerSettings) -> None:
        self._scorer = scorer
        self._s = settings
        self._opened_at: float | None = None
        self._heard = False

    def on_message(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == "text":
            self._heard = True

    def should_flush(self, step: Step, score: float) -> bool:
        if not self._heard or self._scorer.flushed:
            return False
        s = self._s
        if score < s.pause_threshold:
            self._opened_at = None
            return False
        if self._opened_at is None:
            self._opened_at = step.audio_time
        return score >= s.eager_threshold and step.audio_time - self._opened_at >= (
            s.min_silence - 1e-6
        )


class LockstepPolicy(Protocol):
    """Decides, message by message, which flushes a lockstep replay sends."""

    def on_message(self, data: dict[str, Any]) -> Sequence[int]:
        """Consume one server message; return the ids of the flushes to send now."""
        ...


class EagerFlushPolicy:
    """`EagerPolicy` on the passive score: the `flush` collection mode."""

    def __init__(self, weights: Weights, settings: EagerSettings) -> None:
        scorer = LearnedScorer(weights)
        self._policy = EagerPolicy(scorer, settings)
        self._driver = Driver(scorer)
        self._next_id = 1

    def on_message(self, data: dict[str, Any]) -> Sequence[int]:
        if data.get("type") == "flushed":
            return ()
        step = self._driver.feed(data)
        self._policy.on_message(data)
        if step is None or not self._policy.should_flush(step, self._driver.score):
            return ()
        flush_id = self._next_id
        self._next_id += 1
        self._driver.feed({"type": FLUSH_SENT, "flush_id": flush_id})
        return (flush_id,)


class MachinePolicy:
    """The plugin's `TurnMachine`: the `machine` collection mode."""

    def __init__(self, profile: TurnProfile) -> None:
        self._profile = profile
        self.machine: TurnMachine | None = None

    def on_message(self, data: dict[str, Any]) -> Sequence[int]:
        sig = parse(data)
        if sig is None:
            return ()
        if self.machine is None:
            if data.get("type") != "ready":
                raise RuntimeError("first message is not `ready`")
            # the scorer the plugin would install for this server model
            scorer = load_scorer(self._profile.weights, model=str(data.get("model_ext") or ""))
            self.machine = TurnMachine(self._profile, scorer)
        actions = self.machine.on_signal(sig)
        return [a.flush_id for a in actions if isinstance(a, SendFlush)]


class Replay:
    """One turn over one websocket."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, pcm: bytes) -> None:
        self.ws = ws
        self.frames = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
        self.messages: list[dict[str, Any]] = []
        self.ready: dict[str, Any] | None = None

    async def _send_frame(self, frame: bytes) -> None:
        await self.ws.send_str(
            json.dumps({"type": "audio", "audio": base64.b64encode(frame).decode()})
        )

    async def _recv(self) -> dict[str, Any] | None:
        """Next server message; None at end of stream. Raises on errors."""
        msg = await self.ws.receive(timeout=30)
        if msg.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError(f"socket closed early: {msg.type!r}")
        data: dict[str, Any] = json.loads(msg.data)
        type_ = data.get("type")
        if type_ == "error":
            raise RuntimeError(f"server error: {data.get('message')!r} code={data.get('code')}")
        if type_ == "end_of_stream":
            return None
        if type_ == "ready":
            self.ready = {k: v for k, v in data.items() if k != "client_req_id"}
        kept = compact(data)
        if kept is not None:
            self.messages.append(kept)
        return data

    async def passive(self) -> None:
        async def send() -> None:
            for frame in self.frames:
                await self._send_frame(frame)
            await self.ws.send_str(json.dumps({"type": "end_of_stream"}))

        sender = asyncio.create_task(send())
        try:
            while await self._recv() is not None:
                pass
        finally:
            sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)

    async def lockstep(self, policy: LockstepPolicy, lead: float) -> None:
        """Send audio at most `lead` seconds ahead of the last step; pause on each flush.

        Audio stops while a flush is unanswered, so a flush always lands at the same
        audio position and a replay is reproducible.
        """
        audio_time = 0.0
        sent = 0
        eos_sent = False
        waiting: set[int] = set()

        async def top_up() -> None:
            nonlocal sent, eos_sent
            while sent < len(self.frames) and sent * FRAME_S < audio_time + lead - 1e-6:
                await self._send_frame(self.frames[sent])
                sent += 1
            if sent == len(self.frames) and not eos_sent:
                await self.ws.send_str(json.dumps({"type": "end_of_stream"}))
                eos_sent = True

        await top_up()
        while (data := await self._recv()) is not None:
            if data.get("type") == "flushed":
                waiting.discard(int(data.get("flush_id", -1)))
            elif data.get("type") == "step":
                audio_time = float(data.get("total_duration_s") or 0.0)
            for flush_id in policy.on_message(data):
                marker = {"type": FLUSH_SENT, "flush_id": flush_id, "sent_s": sent * FRAME_S}
                if eos_sent:
                    # The server answers no flush after end_of_stream. The decision is
                    # still recorded, so a replay of this run sends the same flushes.
                    marker["after_eos"] = True
                else:
                    waiting.add(flush_id)
                    await self.ws.send_str(json.dumps({"type": "flush", "flush_id": flush_id}))
                self.messages.append(marker)
            if not waiting:
                await top_up()


@dataclass(frozen=True)
class Job:
    """What to replay and how; one per collection run."""

    mode: str
    weights: Weights | None = None
    settings: EagerSettings | None = None
    profile: TurnProfile | None = None
    lead: float = DEFAULT_LEAD
    delay_in_frames: int = DEFAULT_DELAY_IN_FRAMES
    noise: str | None = None
    snr_db: float | None = None

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.settings is not None:
            out.update(asdict(self.settings))
        if self.weights is not None and self.mode == "flush":
            out["weights"] = self.weights.name
        if self.profile is not None:
            out["profile"] = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in dataclasses.asdict(self.profile).items()
            }
            out["lead"] = self.lead
        if self.noise is not None:
            out["noise"] = {"kind": self.noise, "snr_db": self.snr_db}
        return out


def load_audio(
    row: dict[str, Any], job: Job, babble: noise.BabbleBed | None
) -> tuple[bytes, float]:
    """The turn's PCM, byte-identical to an eot-bench run unless noise is mixed in."""
    import_harness()
    from eot_harness.streaming_stt import pcm16le_bytes, prepare_pcm16_audio

    pcm, audio_s = prepare_pcm16_audio(row, sample_rate=SAMPLE_RATE)
    if job.noise is None:
        return pcm, audio_s
    assert job.snr_db is not None
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    bed = noise.noise_for(job.noise, str(row["id"]), len(audio), babble=babble)
    mixed = noise.mix(
        audio, row["silence_spans"], noise=bed, snr_db=job.snr_db, sample_rate=SAMPLE_RATE
    )
    return pcm16le_bytes(mixed), audio_s


async def replay_turn(
    http: aiohttp.ClientSession,
    key: str,
    row: dict[str, Any],
    job: Job,
    babble: noise.BabbleBed | None = None,
) -> dict[str, Any]:
    pcm, audio_s = load_audio(row, job, babble)
    lang = str(row["language"])
    json_config: dict[str, Any] = {
        "language": language_hint(lang),
        "delay_in_frames": (
            job.profile.delay_in_frames if job.profile is not None else job.delay_in_frames
        ),
    }
    setup = {
        "type": "setup",
        "model_name": "default",
        "input_format": "pcm",
        "json_config": json_config,
    }
    async with http.ws_connect(URL, headers={"x-api-key": key}, heartbeat=None) as ws:
        await ws.send_str(json.dumps(setup))
        replay = Replay(ws, pcm)
        if job.mode == "passive":
            await replay.passive()
        elif job.mode == "flush":
            assert job.weights is not None and job.settings is not None
            await replay.lockstep(EagerFlushPolicy(job.weights, job.settings), job.settings.lead)
        else:
            assert job.profile is not None
            await replay.lockstep(MachinePolicy(job.profile), job.lead)
    return {
        "id": row["id"],
        "language": lang,
        "cohort": row.get("cohort"),
        "audio_s": audio_s,
        "mode": job.mode,
        "settings": {"language_hint": language_hint(lang), **job.describe()},
        "ready": replay.ready,
        "messages": replay.messages,
    }


def subset_ids(lang: str, n: int, *, exclude: frozenset[str] = frozenset()) -> list[str]:
    """A fixed random sample of `n` turn ids of `lang`: same on every run."""
    ids = sorted(str(r["id"]) for r in iter_rows(lang, ["id"]) if str(r["id"]) not in exclude)
    return sorted(random.Random(f"subset/{lang}").sample(ids, min(n, len(ids))))


def babble_bed(lang: str, exclude: set[str]) -> noise.BabbleBed:
    """Babble from turns of `lang` that are not replayed in this run."""
    import_harness()
    from eot_harness.streaming_stt import prepare_pcm16_audio

    pool: list[np.ndarray] = []
    for row in iter_rows(lang, ["id", "audio"]):
        if str(row["id"]) in exclude:
            continue
        pcm, _ = prepare_pcm16_audio(row, sample_rate=SAMPLE_RATE)
        pool.append(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0)
        if len(pool) == noise.pool_size():
            break
    return noise.BabbleBed(pool, SAMPLE_RATE, seed=noise.bed_seed(lang))


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["id"] for line in f}


async def collect_language(
    lang: str, job: Job, *, tag: str, concurrency: int, limit: int, subset: int
) -> None:
    out = raw_path(tag, lang)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(out)
    key = load_api_key()
    sem = asyncio.Semaphore(concurrency)
    started = time.time()
    n_ok = n_failed = 0
    wanted = set(subset_ids(lang, subset)) if subset else None
    columns = ["id", "audio", "cohort", "language", "silence_spans"]
    rows = (
        r
        for r in iter_rows(lang, columns)
        if r["id"] not in done and (wanted is None or r["id"] in wanted)
    )
    todo = list(itertools.islice(rows, limit or None))
    print(f"[{lang}/{tag}] {len(done)} done, {len(todo)} to go", file=sys.stderr, flush=True)
    babble = None
    if job.noise == "babble" and todo:
        replayed = wanted if wanted is not None else {str(r["id"]) for r in todo} | done
        babble = babble_bed(lang, replayed)

    with out.open("a") as f:
        async with aiohttp.ClientSession() as http:

            async def one(row: dict[str, Any]) -> None:
                nonlocal n_ok, n_failed
                async with sem:
                    for attempt in range(6):
                        try:
                            res = await asyncio.wait_for(
                                replay_turn(http, key, row, job, babble), timeout=300
                            )
                            break
                        except Exception as e:  # noqa: BLE001
                            # 1008 is the per-key session limit: back off and retry
                            print(f"retry {row['id']} #{attempt}: {e!r}", file=sys.stderr)
                            await asyncio.sleep(2 * (attempt + 1))
                    else:
                        n_failed += 1
                        print(f"FAILED {row['id']}", file=sys.stderr)
                        return
                f.write(json.dumps(res, separators=(",", ":")) + "\n")
                f.flush()
                n_ok += 1
                if n_ok % 100 == 0:
                    print(
                        f"[{lang}/{tag}] {n_ok}/{len(todo)} {time.time() - started:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )

            await asyncio.gather(*(one(r) for r in todo))
    print(
        f"[{lang}/{tag}] finished {n_ok}/{len(todo)}, {n_failed} failed, "
        f"{time.time() - started:.0f}s",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", choices=("passive", "flush", "machine"), required=True)
    ap.add_argument("--lang", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    ap.add_argument("--concurrency", type=int, default=6, help="the key allows 12 sessions")
    ap.add_argument("--limit", type=int, default=0, help="turns per language, 0 for all")
    ap.add_argument("--subset", type=int, default=0, help="fixed random sample per language")
    ap.add_argument("--tag", help="output directory under data/raw (default: from the mode)")
    ap.add_argument("--noise", choices=noise.KINDS)
    ap.add_argument("--snr", type=float, help="speech-to-noise ratio in dB, with --noise")
    ap.add_argument("--weights", help="flush mode: weights whose passive set drives the policy")
    ap.add_argument("--pause-threshold", type=float, default=0.2)
    ap.add_argument("--eager-threshold", type=float, default=0.5)
    ap.add_argument("--min-silence", type=float, default=0.3)
    ap.add_argument("--lead", type=float, default=DEFAULT_LEAD)
    ap.add_argument(
        "--delay-in-frames",
        type=int,
        default=DEFAULT_DELAY_IN_FRAMES,
        help="model delay sent in the setup (machine mode: the profile's)",
    )
    ap.add_argument("--preset", default="balanced", help="machine mode: eagerness")
    ap.add_argument("--scorer", default="learned", choices=("learned", "fallback"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KNOB=VALUE")
    args = ap.parse_args()

    if (args.noise is None) != (args.snr is None):
        ap.error("--noise and --snr go together")
    job = Job(
        mode=args.mode,
        lead=args.lead,
        delay_in_frames=args.delay_in_frames,
        noise=args.noise,
        snr_db=args.snr,
    )
    if args.mode == "flush":
        if not args.weights:
            ap.error("--mode flush needs --weights")
        job = dataclasses.replace(
            job,
            weights=load_weights(args.weights),
            settings=EagerSettings(
                pause_threshold=args.pause_threshold,
                eager_threshold=args.eager_threshold,
                min_silence=args.min_silence,
                lead=args.lead,
            ),
        )
    elif args.mode == "machine":
        profile = preset(args.preset, parse_overrides(args.set), scorer=args.scorer)
        job = dataclasses.replace(job, profile=profile)
    tag = args.tag or args.mode + (f"-{args.noise}{args.snr:g}" if args.noise else "")
    for lang in args.lang:
        asyncio.run(
            collect_language(
                lang,
                job,
                tag=tag,
                concurrency=args.concurrency,
                limit=args.limit,
                subset=args.subset,
            )
        )


if __name__ == "__main__":
    main()
