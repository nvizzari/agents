"""Offline simulator: recorded Gradium streams through `signals -> scorer -> TurnMachine`.

    python tools/eot/simulate.py lag                      # PAUSED entry vs silence onset
    python tools/eot/simulate.py tune --scorer learned    # calibrate presets, non-English
    python tools/eot/simulate.py report --scorer learned  # every preset, English held out
    python tools/eot/simulate.py replay --tag m-balanced  # a live `collect.py --mode machine` run
    python tools/eot/simulate.py noise                    # quiet presets on noisy audio

The machine is the plugin's own class, fed the recorded messages in arrival order. Its
only clock is the audio time of the steps, so a replay reproduces every decision the
plugin would take on that stream.

What is exact and what is not:

- Passive recordings with `preflight=False`: every decision up to the first commit of a
  turn is exact. A commit sends a flush; the simulator answers it after the next step
  (as production does under lockstep pacing) with the recorded words that start before
  the flush point, moved forward in the stream. After a false cut-off, the real stream
  would read higher until the speaker resumes; the machine is IDLE or COMMITTED then,
  so this barely matters, but those spans are not exact.
- Passive recordings with `preflight=True`: the decision to enter EAGER is exact, but
  the steps after the eager flush come from the passive stream, not a flushed one, and
  the `post_flush` weights score them. Use these numbers for exploring only.
- Recordings from `collect.py --mode machine`: the flushes were really sent, so for the
  profile they were recorded with, every decision is exact. With another profile the
  replay is exact until the machine would send a flush the recording does not have (or
  skip one it has); such turns are counted as diverged.

Latencies are in audio seconds from the annotated start of the end-of-turn silence:

- decision: the step at which the machine commits (PAUSED or EAGER -> COMMITTED);
- delivery: when END_OF_SPEECH would reach the framework. The commit waits for the
  reply to its flush unless an eager flush was already answered; a reply is modelled as
  arriving `--flush-rtt` seconds of wall time after its flush was sent (measured at
  real-time pacing by `live_check.py`). This is the one modelled quantity.

False cut-offs, two kinds:

- in a hold: a hold span (0.2 to 5 s, the harness filter) inside which the machine
  commits. `cut_rate` is the harness's false cut-off rate.
- in speech: a commit outside every annotated silence. The harness cannot see these. The
  learned score was trained on silence spans only and is unreliable inside speech, so
  they are not rare. `cut_incl` counts each one as one more cut-off over the same hold
  denominator; presets are tuned against `cut_incl`, so `cut_rate` is always lower.

An end-of-turn span with no commit before the recording ends (the server sends about
0.9 s of steps past the audio) is "never committed" when a turn was open, and its
latency is imputed as the pause start plus `max_silence`, the earliest the machine could
have committed had the silence gone on. When the turn had already been committed earlier
and no word reopened it, the span is "ended early": it is a false cut-off already
counted above and is left out of the latency statistics.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import multiprocessing
import random
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from common import (  # noqa: E402
    DATA_DIR,
    FLUSH_SENT,
    LANGUAGES,
    iter_rows,
    parse_overrides,
    raw_path,
    read_records,
)

from livekit.plugins.gradium.experimental.machine import (  # noqa: E402
    EndOfSpeech,
    Preflight,
    SendFlush,
    StartOfSpeech,
    TurnMachine,
    TurnState,
)
from livekit.plugins.gradium.experimental.profiles import (  # noqa: E402
    TurnProfile,
    preset,
)
from livekit.plugins.gradium.experimental.scorer import (  # noqa: E402
    FallbackScorer,
    FeatureWindow,
    LearnedScorer,
    Scorer,
    Weights,
    load_weights,
)
from livekit.plugins.gradium.experimental.signals import (  # noqa: E402
    Flushed,
    Signal,
    Step,
    Word,
    WordEnd,
    parse,
)

EPS = 1e-6
NON_EN = tuple(lang for lang in LANGUAGES if lang != "en")
MIN_HOLD, MAX_HOLD = 0.2, 5.0  # harness filter on hold spans
MIN_SPAN = 0.1  # harness minimum silence span
# Wall time from a flush to its `flushed`, at real-time pacing. Default from the plan's
# finding 4 (316 ms); `live_check.py` measures it and the reports name the value used.
DEFAULT_FLUSH_RTT = 0.32
DEFAULT_LEAD = 0.16
FRAMEWORK_PREFLIGHT_LIMIT = 3  # PreemptiveGenerationOptions.max_retries default
EAGERNESS = ("responsive", "balanced", "prudent")
BUDGETS = {"responsive": 0.10, "balanced": 0.05, "prudent": 0.02}
RESULTS_DIR = DATA_DIR / "sim"


# --- recorded turns -----------------------------------------------------------------


@dataclass
class Turn:
    id: str
    language: str
    cohort: str
    audio_s: float
    spans: list[dict[str, Any]]
    # parsed stream: Signal, or ("flush", id) where the recording sent a flush
    stream: list[Signal | tuple[str, int]]
    lockstep: bool
    """The recording sent flushes for real (`--mode machine` or `flush`)."""
    source: str = ""
    """The recording's tag: noisy variants of a turn share its id, not its stream."""


def load_spans(langs: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for lang in langs:
        for r in iter_rows(lang, ["id", "silence_spans"]):
            spans = r["silence_spans"]
            out[r["id"]] = [
                {
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "label": s.get("label") or ("eot" if i == len(spans) - 1 else "hold"),
                }
                for i, s in enumerate(spans)
            ]
    return out


def load_turns(
    tag: str, langs: Iterable[str], spans: dict[str, list[dict[str, Any]]] | None = None
) -> list[Turn]:
    langs = [lang for lang in langs if raw_path(tag, lang).exists()]
    spans = spans if spans is not None else load_spans(langs)
    turns = []
    for rec in read_records(tag, langs):
        stream: list[Signal | tuple[str, int]] = []
        lockstep = False
        for msg in rec["messages"]:
            if msg.get("type") == FLUSH_SENT:
                stream.append(("flush", int(msg["flush_id"])))
                lockstep = True
                continue
            sig = parse(msg)
            if sig is not None:
                stream.append(sig)
        turns.append(
            Turn(
                id=rec["id"],
                language=rec["language"],
                cohort=str(rec.get("cohort")),
                audio_s=float(rec["audio_s"]),
                spans=spans[rec["id"]],
                stream=stream,
                lockstep=lockstep or rec.get("mode") in ("flush", "machine"),
                source=tag,
            )
        )
    return turns


# --- scorers --------------------------------------------------------------------------


class CachedScorer:
    """`LearnedScorer` that serves cached scores while the stream is untouched.

    Until the machine sends a flush or ends a turn, the learned score depends only on
    the recorded stream, so it is computed once per (turn, weights, warm-up) and shared
    by every profile. On the first flush or turn end, the feature window is rebuilt by
    replaying the calls so far (window only: no tree is evaluated for past steps) and
    scoring continues exactly as `LearnedScorer.on_step` does. Scores are bit-identical
    to running `LearnedScorer` throughout; `simulate.py check` verifies it.
    """

    def __init__(self, weights: Weights, cache: list[float]) -> None:
        self._weights = weights
        self._cache = cache
        self._k = 0
        self._log: list[tuple[Any, float | None]] = []
        self._window: FeatureWindow | None = None

    @property
    def name(self) -> str:
        return self._weights.name

    def _go_live(self) -> FeatureWindow:
        if self._window is None:
            window = FeatureWindow()
            for obj, at in self._log:
                if at is None:
                    window.on_step(obj)
                else:
                    window.on_word(obj, at)
            self._window = window
            self._log = []
        return self._window

    def reset(self) -> None:
        self._go_live().reset()

    def on_step(self, step: Step) -> float:
        window = self._window
        if window is not None:
            vector = window.on_step(step)
            w = self._weights
            return (w.post_flush if window.flushed else w.passive).predict(vector)
        self._log.append((step, None))
        score = self._cache[self._k]
        self._k += 1
        return score

    def on_word(self, word: Word, audio_time: float) -> None:
        if self._window is not None:
            self._window.on_word(word, audio_time)
        else:
            self._log.append((word, audio_time))

    def on_flush(self) -> None:
        self._go_live().on_flush()

    def on_turn_end(self) -> None:
        self._go_live().on_turn_end()


def pristine_scores(turn: Turn, weights: Weights, warmup_steps: int) -> list[float]:
    """Learned scores the machine would read on `turn` before any flush or turn end."""
    scorer = LearnedScorer(weights)
    out: list[float] = []
    steps = 0
    audio_time = 0.0
    for sig in turn.stream:
        if isinstance(sig, Step):
            audio_time = sig.audio_time
            steps += 1
            if steps > warmup_steps:
                out.append(scorer.on_step(sig))
        elif isinstance(sig, Word):
            scorer.on_word(sig, audio_time)
    return out


class ScorerFactory:
    """Builds the scorer for a profile, with the score cache for the learned one."""

    def __init__(self, weights: Weights | None) -> None:
        self.weights = weights
        self._cache: dict[tuple[str, str, int], list[float]] = {}

    @property
    def kind(self) -> str:
        return "learned" if self.weights is not None else "fallback"

    def warm(self, turns: Iterable[Turn], warmup_steps: int) -> None:
        """Fill the cache before forking workers, so they share it."""
        if self.weights is None:
            return
        for t in turns:
            if not t.lockstep and (t.source, t.id, warmup_steps) not in self._cache:
                self._cache[(t.source, t.id, warmup_steps)] = pristine_scores(
                    t, self.weights, warmup_steps
                )

    def __call__(self, turn: Turn, profile: TurnProfile) -> Scorer:
        if self.weights is None:
            return FallbackScorer()
        if turn.lockstep:
            return LearnedScorer(self.weights)
        key = (turn.source, turn.id, profile.warmup_steps)
        cache = self._cache.get(key)
        if cache is None:
            cache = self._cache[key] = pristine_scores(turn, self.weights, profile.warmup_steps)
        return CachedScorer(self.weights, cache)


# --- one turn -------------------------------------------------------------------------


@dataclass
class Trace:
    """What the machine did on one recorded turn, in audio seconds."""

    pauses: list[float] = field(default_factory=list)
    """PAUSED entries."""
    pause_ends: list[float] = field(default_factory=list)
    """When each pause in `pauses` ended (resume or commit); inf while still open."""
    commits: list[tuple[float, float, bool]] = field(default_factory=list)
    """(commit time, time the flush serving it was sent, preflight emitted in this pause)."""
    eagers: list[float] = field(default_factory=list)
    preflights: list[float] = field(default_factory=list)
    """Eager flush time behind each emitted preflight."""
    wasted: int = 0
    """Preflights followed by a resume."""
    opens: list[float] = field(default_factory=list)
    """When each turn opened (START_OF_SPEECH)."""
    max_preflights_per_turn: int = 0
    open_pause: float | None = None
    """PAUSED entry of the pause still open when the recording ended."""
    last_step: float = 0.0
    diverged_at: float | None = None


def simulate(
    turn: Turn,
    profile: TurnProfile,
    scorer: Scorer,
    *,
    lead: float = DEFAULT_LEAD,
    stop_after: float | None = None,
) -> Trace:
    """Drive the plugin's `TurnMachine` over one recorded turn.

    With `stop_after`, the replay ends at the first commit at or after that audio time:
    no metric reads anything later, and skipping the rest saves most of the scoring.
    """
    machine = TurnMachine(profile, scorer)
    tr = Trace()
    stream = turn.stream
    consumed: set[int] = set()
    pending: list[tuple[int, float]] = []  # passive replay: flushes awaiting a reply
    expected_flushes: list[int] = []  # lockstep replay: flushes the machine just sent
    pause_flush_t: float | None = None
    preflight_this_pause = False
    preflights_this_turn = 0
    now = 0.0

    def feed(sig: Signal) -> None:
        nonlocal pause_flush_t, preflight_this_pause, preflights_this_turn
        prev = machine.state
        actions = machine.on_signal(sig)
        new = machine.state
        is_step = isinstance(sig, Step)
        if prev in (TurnState.PAUSED, TurnState.EAGER) and new not in (
            TurnState.PAUSED,
            TurnState.EAGER,
        ):
            tr.pause_ends.append(now)
        if is_step and prev is TurnState.SPEAKING and new is TurnState.PAUSED:
            tr.pauses.append(now)
            pause_flush_t = None
            preflight_this_pause = False
        if prev is TurnState.PAUSED and new is TurnState.EAGER:
            tr.eagers.append(now)
            pause_flush_t = now
        for a in actions:
            if isinstance(a, SendFlush):
                if turn.lockstep:
                    expected_flushes.append(a.flush_id)
                else:
                    pending.append((a.flush_id, now))
            elif isinstance(a, StartOfSpeech):
                tr.opens.append(now)
            elif isinstance(a, Preflight):
                tr.preflights.append(pause_flush_t if pause_flush_t is not None else now)
                preflight_this_pause = True
                preflights_this_turn += 1
                tr.max_preflights_per_turn = max(tr.max_preflights_per_turn, preflights_this_turn)
        ended = any(isinstance(a, EndOfSpeech) for a in actions)
        if (
            is_step
            and prev in (TurnState.PAUSED, TurnState.EAGER)
            and (new is TurnState.COMMITTED or ended)
        ):
            flush_t = pause_flush_t if pause_flush_t is not None else now
            tr.commits.append((now, flush_t, preflight_this_pause))
        if prev is TurnState.EAGER and new is TurnState.SPEAKING:
            if preflight_this_pause:
                tr.wasted += 1
            pause_flush_t = None
            preflight_this_pause = False
        if ended:
            preflights_this_turn = 0
            pause_flush_t = None
            preflight_this_pause = False

    for i, item in enumerate(stream):
        if i in consumed:
            continue
        if isinstance(item, tuple):
            # lockstep recording: the flush the recorded machine sent here
            if expected_flushes and expected_flushes[0] == item[1]:
                expected_flushes.pop(0)
            elif tr.diverged_at is None:
                tr.diverged_at = now
            continue
        if expected_flushes and tr.diverged_at is None:
            # the machine sent a flush the recording does not have
            tr.diverged_at = now
        expected_flushes.clear()
        if isinstance(item, Step):
            now = tr.last_step = item.audio_time
        feed(item)
        if stop_after is not None and tr.commits and tr.commits[-1][0] >= stop_after - EPS:
            return tr
        if isinstance(item, Step) and pending:
            replies = [p for p in pending if p[1] < now - EPS]
            if replies:
                pending[:] = [p for p in pending if p[1] >= now - EPS]
                for flush_id, sent_at in replies:
                    _deliver_held_words(stream, i, sent_at + lead, consumed, feed)
                    feed(Flushed(flush_id))
    if machine.state in (TurnState.PAUSED, TurnState.EAGER):
        tr.open_pause = machine_pause_start(tr)
    return tr


def machine_pause_start(tr: Trace) -> float | None:
    return tr.pauses[-1] if tr.pauses else None


def _deliver_held_words(
    stream: Sequence[Signal | tuple[str, int]],
    i: int,
    cutoff: float,
    consumed: set[int],
    feed: Callable[[Signal], None],
) -> None:
    """What a flush returns: recorded words starting before `cutoff`, moved forward."""
    for j in range(i + 1, len(stream)):
        if j in consumed:
            continue
        sig = stream[j]
        if not isinstance(sig, Word):
            continue
        if sig.start_time >= cutoff:
            return
        consumed.add(j)
        feed(sig)
        if j + 1 < len(stream) and isinstance(stream[j + 1], WordEnd):
            consumed.add(j + 1)
            feed(stream[j + 1])  # type: ignore[arg-type]


# --- metrics --------------------------------------------------------------------------


@dataclass
class TurnScore:
    holds: int = 0
    cut: int = 0
    eot: bool = False
    decision: float = math.nan
    delivery: float = math.nan
    never: bool = False
    """The turn was open through the end-of-turn span and never committed."""
    ended_early: bool = False
    """The turn was committed before its end-of-turn span and no word reopened it."""
    preflight_lead: float | None = None
    """Delivery minus preflight arrival, when the turn ended on an adopted preflight."""
    speech_commits: int = 0
    wasted: int = 0
    preflights: int = 0
    over_limit: bool = False
    after_cut: bool = False
    """A false cut-off happened earlier in the turn: later spans are approximate."""
    diverged: bool = False
    pause_lag: list[tuple[str, float]] = field(default_factory=list)


def score_turn(turn: Turn, tr: Trace, profile: TurnProfile, flush_rtt: float) -> TurnScore:
    ts = TurnScore(
        wasted=tr.wasted,
        preflights=len(tr.preflights),
        over_limit=tr.max_preflights_per_turn > FRAMEWORK_PREFLIGHT_LIMIT,
        diverged=tr.diverged_at is not None,
    )
    commit_times = [c[0] for c in tr.commits]
    in_span = [False] * len(commit_times)
    for span in turn.spans:
        start, end, label = span["start"], span["end"], span["label"]
        dur = end - start
        if dur < MIN_SPAN - EPS:
            continue
        for k, c in enumerate(commit_times):
            if start - EPS <= c < end - EPS or (label == "eot" and c >= start - EPS):
                in_span[k] = True
        if label == "hold":
            if not (MIN_HOLD - EPS <= dur <= MAX_HOLD + EPS):
                continue
            ts.holds += 1
            if any(start - EPS <= c < end - EPS for c in commit_times):
                ts.cut += 1
        else:
            ts.eot = True
            # the turn was already ended once: what follows is not exact (module docstring)
            ts.after_cut = any(c < start - EPS for c in commit_times)
            after = [c for c in tr.commits if c[0] >= start - EPS]
            if after:
                c, flush_t, with_preflight = after[0]
                ts.decision = c - start
                ts.delivery = max(c, flush_t + flush_rtt) - start
                if with_preflight:
                    ts.preflight_lead = max(c, flush_t + flush_rtt) - (flush_t + flush_rtt)
            elif tr.commits and not any(o > tr.commits[-1][0] for o in tr.opens):
                ts.ended_early = True
            else:
                ts.never = True
                base = tr.open_pause if tr.open_pause is not None else tr.last_step
                ts.decision = max(base + profile.max_silence, tr.last_step) - start
                ts.delivery = ts.decision + flush_rtt
        if dur >= MIN_HOLD - EPS:
            # first PAUSED entry that covers the span's start, or enters inside it
            onset = _pause_onset(tr, start, end if label == "hold" else math.inf)
            if onset is not None:
                ts.pause_lag.append((label, onset - start))
    ts.speech_commits = sum(1 for flag in in_span if not flag)
    return ts


def _pause_onset(tr: Trace, start: float, end: float) -> float | None:
    """PAUSED entry for the silence at `start`: a pause still open at `start` (negative
    lag), else the first one entered inside the span."""
    ends = tr.pause_ends + [math.inf] * (len(tr.pauses) - len(tr.pause_ends))
    for p, e in zip(tr.pauses, ends, strict=True):
        if p < start - EPS and e >= start - EPS:
            return p
        if start - EPS <= p < end - EPS:
            return p
    return None


@dataclass
class Summary:
    turns: int
    holds: int
    eots: int
    cut_rate: float
    """Hold spans committed in / hold spans: the harness's false cut-off rate."""
    cut_incl: float
    """Same, counting each commit inside annotated speech as one more cut-off: what the
    presets are tuned against."""
    turns_cut: float
    """Turns with at least one false cut-off of either kind."""
    decision_mean: float
    decision_p50: float
    decision_p90: float
    delivery_mean: float
    delivery_p50: float
    delivery_p90: float
    never: float
    ended_early: float
    speech_commits: float
    """Commits outside any annotated silence, per turn."""
    preflights: float
    """Preflights per turn."""
    wasted: float
    """Wasted preflights per turn."""
    turns_wasted: float
    over_limit: float
    lead_p50: float
    lead_mean: float
    adopted: float
    """Share of end-of-turn commits that ended on a preflight."""
    after_cut: float
    diverged: float

    def row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def summarize(scores: list[TurnScore]) -> Summary:
    holds = sum(s.holds for s in scores)
    cut = sum(s.cut for s in scores)
    eots = [s for s in scores if s.eot]
    timed = [s for s in eots if not s.ended_early]
    dec = np.asarray([s.decision for s in timed])
    dlv = np.asarray([s.delivery for s in timed])
    speech = sum(s.speech_commits for s in scores)
    leads = np.asarray([s.preflight_lead for s in eots if s.preflight_lead is not None])
    n = max(len(scores), 1)
    return Summary(
        turns=len(scores),
        holds=holds,
        eots=len(eots),
        cut_rate=cut / holds if holds else 0.0,
        cut_incl=(cut + speech) / holds if holds else 0.0,
        turns_cut=sum(1 for s in scores if s.cut or s.speech_commits) / n,
        decision_mean=float(dec.mean()) if len(dec) else math.nan,
        decision_p50=float(np.median(dec)) if len(dec) else math.nan,
        decision_p90=float(np.quantile(dec, 0.9)) if len(dec) else math.nan,
        delivery_mean=float(dlv.mean()) if len(dlv) else math.nan,
        delivery_p50=float(np.median(dlv)) if len(dlv) else math.nan,
        delivery_p90=float(np.quantile(dlv, 0.9)) if len(dlv) else math.nan,
        never=sum(1 for s in eots if s.never) / max(len(eots), 1),
        ended_early=sum(1 for s in eots if s.ended_early) / max(len(eots), 1),
        speech_commits=speech / n,
        preflights=sum(s.preflights for s in scores) / n,
        wasted=sum(s.wasted for s in scores) / n,
        turns_wasted=sum(1 for s in scores if s.wasted) / n,
        over_limit=sum(1 for s in scores if s.over_limit) / n,
        lead_p50=float(np.median(leads)) if len(leads) else math.nan,
        lead_mean=float(leads.mean()) if len(leads) else math.nan,
        adopted=len(leads) / max(len(eots), 1),
        after_cut=sum(1 for s in scores if s.after_cut) / n,
        diverged=sum(1 for s in scores if s.diverged) / n,
    )


def run(
    turns: Sequence[Turn],
    profile: TurnProfile,
    factory: ScorerFactory,
    *,
    flush_rtt: float = DEFAULT_FLUSH_RTT,
    lead: float = DEFAULT_LEAD,
) -> list[TurnScore]:
    return [
        score_turn(
            t,
            simulate(t, profile, factory(t, profile), lead=lead, stop_after=eot_start(t)),
            profile,
            flush_rtt,
        )
        for t in turns
    ]


def eot_start(turn: Turn) -> float | None:
    starts = [s["start"] for s in turn.spans if s["label"] == "eot"]
    return min(starts) if starts else None


# --- calibration ----------------------------------------------------------------------


def _profile(base: TurnProfile, knobs: dict[str, Any]) -> TurnProfile | None:
    try:
        return dataclasses.replace(base, **knobs)
    except ValueError:
        return None


_WORKER: dict[str, Any] = {}


def _evaluate_in_worker(profile: TurnProfile) -> Summary:
    return summarize(run(_WORKER["turns"], profile, _WORKER["factory"]))


def calibrate(
    turns: Sequence[Turn],
    base: TurnProfile,
    budget: float,
    grid: dict[str, Sequence[Any]],
    factory: ScorerFactory,
    *,
    rounds: int = 3,
    workers: int = 8,
    joint: dict[str, Sequence[Any]] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[TurnProfile, Summary]:
    """Coordinate descent: lowest mean decision latency with `cut_incl` <= `budget`.

    Each round sweeps one knob at a time over its grid, keeping the others fixed, and
    keeps the best feasible value. The whole machine runs for each candidate, so every
    interaction between knobs (hysteresis, timeouts, resumes) is in the number.
    Candidates of one sweep run in parallel worker processes. `joint` first tries
    every combination of those knobs' values, so the descent does not start in a local
    optimum (one knob at a time cannot trade a later timeout for a lower threshold).
    """
    factory.warm(turns, base.warmup_steps)
    _WORKER.update(turns=turns, factory=factory)
    seen: dict[tuple[Any, ...], Summary] = {}

    def key(p: TurnProfile) -> tuple[Any, ...]:
        return tuple(dataclasses.astuple(p))

    def better(a: Summary, b: Summary | None) -> bool:
        if b is None:
            return True
        fa, fb = a.cut_incl <= budget + EPS, b.cut_incl <= budget + EPS
        if fa != fb:
            return fa
        if not fa:
            return a.cut_incl < b.cut_incl
        return a.decision_mean < b.decision_mean

    with multiprocessing.get_context("fork").Pool(workers) as pool:

        def evaluate(cands: list[TurnProfile]) -> list[Summary]:
            todo = [c for c in cands if key(c) not in seen]
            for c, s in zip(todo, pool.map(_evaluate_in_worker, todo), strict=True):
                seen[key(c)] = s
            return [seen[key(c)] for c in cands]

        best = base
        (best_s,) = evaluate([base])
        if joint:
            combos = [
                c
                for values in itertools.product(*joint.values())
                if (c := _profile(base, dict(zip(joint, values, strict=True)))) is not None
            ]
            for cand, s in zip(combos, evaluate(combos), strict=True):
                if better(s, best_s):
                    best, best_s = cand, s
            log(
                f"  joint {[getattr(best, k) for k in joint]}: cut {best_s.cut_incl:.2%}, "
                f"decision {best_s.decision_mean * 1000:.0f} ms ({len(seen)} profiles run)"
            )
        for r in range(rounds):
            changed = False
            for knob, values in grid.items():
                cands = [c for v in values if (c := _profile(best, {knob: v})) is not None]
                for cand, s in zip(cands, evaluate(cands), strict=True):
                    if cand != best and better(s, best_s):
                        best, best_s, changed = cand, s, True
                log(
                    f"  round {r} {knob}={getattr(best, knob)}: cut {best_s.cut_incl:.2%}, "
                    f"decision {best_s.decision_mean * 1000:.0f} ms ({len(seen)} profiles run)"
                )
            if not changed:
                break
    return best, best_s


JOINT_GRID: dict[str, list[Any]] = {
    "commit_threshold": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99],
    "max_silence": [0.8, 1.04, 1.28, 1.52, 1.76, 2.0, 2.4, 2.8, 3.2, 3.6],
    "min_silence": [0.0, 0.16, 0.32, 0.56],
}
TUNE_GRID: dict[str, list[Any]] = {
    # the learned score saturates near 1 inside speech, so the top of the range is fine
    "commit_threshold": [round(x, 2) for x in np.arange(0.30, 0.97, 0.02)]
    + [0.97, 0.98, 0.985, 0.99, 0.995, 0.998],
    "min_silence": [round(x, 2) for x in np.arange(0.0, 1.21, 0.08)],
    "max_silence": [round(x, 2) for x in np.arange(0.64, 4.01, 0.16)],
    "pause_threshold": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6],
    "resume_threshold": [0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
    "confirm_steps": [1, 2, 3],
}


# --- commands -------------------------------------------------------------------------


def _weights_for(kind: str) -> Weights | None:
    return load_weights("default") if kind == "learned" else None


def _fmt(s: Summary) -> str:
    return (
        f"cut {s.cut_rate * 100:.1f}% / incl. speech {s.cut_incl * 100:.1f}% (turns "
        f"{s.turns_cut * 100:.1f}%) | decision "
        f"{s.decision_mean * 1000:.0f} / {s.decision_p50 * 1000:.0f} / "
        f"{s.decision_p90 * 1000:.0f} ms | delivery {s.delivery_mean * 1000:.0f} / "
        f"{s.delivery_p50 * 1000:.0f} / {s.delivery_p90 * 1000:.0f} ms | never "
        f"{s.never * 100:.1f}%, early {s.ended_early * 100:.1f}%"
        + (
            f" | preflights/turn {s.preflights:.2f}, wasted/turn {s.wasted:.3f} "
            f"(turns {s.turns_wasted * 100:.1f}%), adopted {s.adopted * 100:.0f}%, "
            f"lead p50 {s.lead_p50 * 1000:.0f} ms, over limit {s.over_limit * 100:.2f}%"
            if s.preflights
            else ""
        )
        + (f" | diverged {s.diverged * 100:.1f}%" if s.diverged else "")
    )


def _dump(name: str, payload: Any) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def _base_profile(args: argparse.Namespace, eagerness: str = "balanced") -> TurnProfile:
    return preset(
        eagerness,  # type: ignore[arg-type]
        parse_overrides(args.set) if getattr(args, "set", None) else None,  # type: ignore[arg-type]
        scorer=args.scorer,
    )


def cmd_lag(args: argparse.Namespace) -> None:
    factory = ScorerFactory(_weights_for(args.scorer))
    profile = _base_profile(args, args.preset)
    for name, langs in (("non-English", NON_EN), ("English", ("en",))):
        turns = load_turns(args.tag, langs)
        scores = run(turns, profile, factory)
        out = {}
        for label in ("hold", "eot"):
            lags = np.asarray([lag for s in scores for lbl, lag in s.pause_lag if lbl == label])
            n_spans = sum(
                1
                for t in turns
                for sp in t.spans
                if sp["label"] == label and sp["end"] - sp["start"] >= MIN_HOLD - EPS
            )
            q = np.quantile(lags, [0.1, 0.5, 0.9]) if len(lags) else [math.nan] * 3
            out[label] = {
                "spans": n_spans,
                "paused": len(lags),
                "mean": float(lags.mean()) if len(lags) else math.nan,
                "p10": float(q[0]),
                "p50": float(q[1]),
                "p90": float(q[2]),
            }
            print(
                f"{name} {label}: PAUSED entered in {len(lags)}/{n_spans} spans >= 0.2 s; "
                f"lag mean {out[label]['mean'] * 1000:.0f} ms, p10 {q[0] * 1000:.0f}, "
                f"p50 {q[1] * 1000:.0f}, p90 {q[2] * 1000:.0f} ms "
                f"(pause_threshold {profile.pause_threshold}, {args.scorer})"
            )
        _dump(f"lag_{args.scorer}_{name}.json", {"profile": dataclasses.asdict(profile), **out})


# Knobs that never move a flush: with preflight on, the eager flush is decided by the
# pause and eager knobs alone, so a lockstep recording can be replayed exactly while these
# change (until a commit lands at another step and the next turn diverges; such turns
# are counted).
POST_FLUSH_GRID: dict[str, list[Any]] = {
    "commit_threshold": TUNE_GRID["commit_threshold"],
    "confirm_timeout": [0.16, 0.32, 0.48, 0.64, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    "max_silence": TUNE_GRID["max_silence"],
    "confirm_steps": [1, 2, 3],
}


def _recorded_profile(tag: str, langs: Sequence[str]) -> TurnProfile:
    for lang in langs:
        path = raw_path(tag, lang)
        if path.exists():
            with path.open() as f:
                return TurnProfile(**json.loads(f.readline())["settings"]["profile"])
    raise SystemExit(f"no recording under data/raw/{tag}")


def cmd_tune(args: argparse.Namespace) -> None:
    langs = args.lang or NON_EN
    # several recordings, comma-separated, are pooled (e.g. every noise condition)
    turns = [t for tag in args.tag.split(",") for t in load_turns(tag, langs)]
    print(f"tuning on {len(turns)} turns ({args.tag}, {', '.join(langs)})")
    out: dict[str, Any] = {}
    if args.from_recording:
        # preflight on: the knobs after the eager flush, replayed on a live recording
        base = _recorded_profile(args.tag, langs)
        factory = ScorerFactory(_weights_for("learned" if base.weights else "fallback"))
        eagerness = args.eagerness[0] if args.eagerness else "balanced"
        print(f"{eagerness}: from {base}")
        budget = args.budget if args.budget is not None else BUDGETS[eagerness]
        best, s = calibrate(turns, base, budget, POST_FLUSH_GRID, factory, rounds=args.rounds)
        print(f"  -> {_fmt(s)}")
        out[eagerness] = {"profile": dataclasses.asdict(best), "tuning": s.row()}
        path = _dump(f"tune_{args.tag}.json", out)
        print(f"wrote {path}")
        return
    factory = ScorerFactory(_weights_for(args.scorer))
    grid = {k: v for k, v in TUNE_GRID.items() if k not in (args.freeze or [])}
    for eagerness in args.eagerness or EAGERNESS:
        base = _base_profile(args, eagerness)
        base = dataclasses.replace(base, preflight=False)
        print(f"{eagerness}: budget {BUDGETS[eagerness]:.0%}")
        best, s = calibrate(
            turns,
            base,
            BUDGETS[eagerness],
            grid,
            factory,
            rounds=args.rounds,
            joint=JOINT_GRID if args.joint else None,
        )
        print(f"  -> {_fmt(s)}")
        out[eagerness] = {"profile": dataclasses.asdict(best), "tuning": s.row()}
    name = "noise" if "," in args.tag else args.tag
    path = _dump(f"tune_{args.scorer}_{name}.json", out)
    print(f"wrote {path}")


def cmd_report(args: argparse.Namespace) -> None:
    factory = ScorerFactory(_weights_for(args.scorer))
    sets = {"en": load_turns(args.tag, ["en"])}
    if not args.english_only:
        sets["non-en"] = load_turns(args.tag, NON_EN)
    if args.same_turns_as:
        keep = {t.id for t in load_turns(args.same_turns_as, LANGUAGES)}
        sets = {k: [t for t in v if t.id in keep] for k, v in sets.items()}
        sets = {k: v for k, v in sets.items() if v}
    out: dict[str, Any] = {}
    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned else None
    for eagerness in args.eagerness or EAGERNESS:
        profile = (
            TurnProfile(**tuned[eagerness]["profile"]) if tuned else _base_profile(args, eagerness)
        )
        if args.preflight is not None:
            profile = dataclasses.replace(profile, preflight=args.preflight == "on")
        out[eagerness] = {"profile": dataclasses.asdict(profile)}
        for name, turns in sets.items():
            scores = run(turns, profile, factory, flush_rtt=args.flush_rtt)
            s = summarize(scores)
            out[eagerness][name] = s.row()
            print(f"{eagerness:10} {name:6} {_fmt(s)}")
            if name == "en" and args.cohorts:
                for cohort in ("general", "accented", "short_turns"):
                    sub = [sc for t, sc in zip(turns, scores, strict=True) if t.cohort == cohort]
                    cs = summarize(sub)
                    out[eagerness][f"en/{cohort}"] = cs.row()
                    print(f"{'':10} {cohort:6} {_fmt(cs)}")
    pf = "" if args.preflight is None else f"_preflight-{args.preflight}"
    pf += "_tuned" if args.tuned else ""
    pf += f"_on-{args.same_turns_as}" if args.same_turns_as else ""
    path = _dump(f"report_{args.scorer}_{args.tag}{pf}.json", out)
    print(f"wrote {path} (flush rtt {args.flush_rtt * 1000:.0f} ms)")


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay a `--mode machine` recording with the profile it was recorded with."""
    langs = args.lang or LANGUAGES
    turns = load_turns(args.tag, langs)
    if not turns:
        raise SystemExit(f"no recording under data/raw/{args.tag}")
    with raw_path(args.tag, turns[0].language).open() as f:
        recorded = json.loads(f.readline())["settings"]["profile"]
    profile = TurnProfile(**recorded)
    if args.set:
        profile = dataclasses.replace(profile, **parse_overrides(args.set))
    factory = ScorerFactory(_weights_for("learned" if profile.weights else "fallback"))
    out: dict[str, Any] = {"profile": dataclasses.asdict(profile)}
    for name, keep in (
        ("en", lambda t: t.language == "en"),
        ("non-en", lambda t: t.language != "en"),
    ):
        sub = [t for t in turns if keep(t)]
        if not sub:
            continue
        s = summarize(run(sub, profile, factory, flush_rtt=args.flush_rtt))
        out[name] = s.row()
        print(f"{args.tag} {name:6} {_fmt(s)}")
    suffix = "_" + "_".join(args.set) if args.set else ""
    _dump(f"replay_{args.tag}{suffix}.json", out)


def cmd_noise(args: argparse.Namespace) -> None:
    factory = ScorerFactory(_weights_for(args.scorer))
    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned else None
    conditions = ["passive"] + [
        f"passive-{k}{snr}" for k in ("pink", "babble") for snr in (15, 10, 5)
    ]
    lang_sets = {"en": ["en"], "non-en": list(NON_EN)}
    out: dict[str, Any] = {}
    for set_name, langs in lang_sets.items():
        noisy_ids = {t.id for t in load_turns("passive-pink10", langs)}
        if not noisy_ids:
            continue
        for cond in conditions:
            turns = [t for t in load_turns(cond, langs) if t.id in noisy_ids]
            for eagerness in args.eagerness or EAGERNESS:
                profile = (
                    TurnProfile(**tuned[eagerness]["profile"])
                    if tuned
                    else preset(eagerness, scorer=args.scorer)  # type: ignore[arg-type]
                )
                if args.preflight is not None:
                    profile = dataclasses.replace(profile, preflight=args.preflight == "on")
                s = summarize(run(turns, profile, factory, flush_rtt=args.flush_rtt))
                out.setdefault(set_name, {}).setdefault(cond, {})[eagerness] = s.row()
                print(f"{set_name} {cond:16} {eagerness:10} {_fmt(s)}")
    _dump(f"noise_{args.scorer}{'_tuned' if tuned else ''}.json", out)


def cmd_check(args: argparse.Namespace) -> None:
    """The score cache changes nothing: same traces as a plain `LearnedScorer`."""
    weights = load_weights("default")
    turns = random.Random(0).sample(load_turns("passive", ["en", "fr", "ja"]), 300)
    profiles = [
        dataclasses.replace(preset(e, scorer="learned"), preflight=pf)
        for e in EAGERNESS
        for pf in (False, True)
    ]
    factory = ScorerFactory(weights)
    mismatches = 0
    for p in profiles:
        for t in turns:
            a = simulate(t, p, factory(t, p))
            b = simulate(t, p, LearnedScorer(weights))
            mismatches += a != b
    print(f"{len(profiles) * len(turns)} replays, {mismatches} differ between cached and plain")
    if mismatches:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scorer", choices=("learned", "fallback"), default="learned")
    ap.add_argument("--tag", default="passive", help="recording under data/raw; `tune` pools a,b,c")
    ap.add_argument("--lang", nargs="+", choices=LANGUAGES)
    ap.add_argument("--flush-rtt", type=float, default=DEFAULT_FLUSH_RTT)
    ap.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE", help="repeatable")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lag = sub.add_parser("lag")
    lag.add_argument("--preset", default="balanced")
    tune = sub.add_parser("tune")
    tune.add_argument("--eagerness", nargs="+", choices=EAGERNESS)
    tune.add_argument("--rounds", type=int, default=3)
    tune.add_argument("--freeze", nargs="*", help="knobs to leave at the preset value")
    tune.add_argument("--joint", action="store_true", help="start from a joint grid search")
    tune.add_argument("--budget", type=float, help="with --from-recording: cut_incl target")
    tune.add_argument(
        "--from-recording",
        action="store_true",
        help="preflight on: start from the profile a `--mode machine` run recorded",
    )
    rep = sub.add_parser("report")
    rep.add_argument("--eagerness", nargs="+", choices=EAGERNESS)
    rep.add_argument("--english-only", action="store_true")
    rep.add_argument("--cohorts", action="store_true")
    rep.add_argument("--preflight", choices=("on", "off"))
    rep.add_argument("--tuned", help="profiles from a `tune` output instead of the presets")
    rep.add_argument("--same-turns-as", metavar="TAG", help="only the turns of that recording")
    sub.add_parser("replay")
    nz = sub.add_parser("noise")
    nz.add_argument("--eagerness", nargs="+", choices=EAGERNESS)
    nz.add_argument("--preflight", choices=("on", "off"))
    nz.add_argument("--tuned", help="profiles from a `tune` output instead of the presets")
    sub.add_parser("check")
    args = ap.parse_args()
    {
        "lag": cmd_lag,
        "tune": cmd_tune,
        "report": cmd_report,
        "replay": cmd_replay,
        "noise": cmd_noise,
        "check": cmd_check,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
