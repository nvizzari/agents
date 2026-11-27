"""Turn-taking knobs and the named presets built from them.

A preset is addressed by eagerness: how much false cut-off risk buys latency. Any knob
can then be overridden:

    profile = preset("balanced", {"max_silence": 2.5})

There is no noise setting. A `noisy` variant was calibrated on noise-mixed replays and
did no better than the next stricter eagerness (see the plan's step 4 results), so in
noise, step down one eagerness level.

This module has no I/O.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

Eagerness = Literal["responsive", "balanced", "prudent"]
ScorerKind = Literal["learned", "fallback"]
"""Which scorer a profile's thresholds were calibrated for. The two scores live on
different scales (the learned score rises earlier and higher in a pause), so a threshold
that suits one is wrong for the other."""


@dataclass(frozen=True)
class TurnProfile:
    """Every knob of the turn state machine, validated at construction.

    Scores are end-of-turn scores from the scorer, in [0, 1], higher when the turn has
    ended. Times are seconds of audio. The thresholds must keep this order, or some
    transitions could never fire, or fire in a loop:
    `start_threshold <= resume_threshold < pause_threshold <= eager_threshold <=
    commit_threshold`.
    """

    eager_threshold: float
    """Score needed to enter `EAGER` (flush, then preflight). Lower: earlier preflight,
    more wasted LLM calls."""
    commit_threshold: float
    """Score needed to end the turn: post-flush in `EAGER`, passive in `PAUSED` when no
    eager entry is allowed. Lower: faster, more cut-offs."""
    min_silence: float
    """Silence, counted from entry to `PAUSED`, required before `EAGER` or a commit."""
    max_silence: float
    """Silence, counted from entry to `PAUSED`, after which the turn ends regardless of
    score."""
    confirm_steps: int
    """Consecutive steps at or above `commit_threshold` needed to commit."""
    confirm_timeout: float
    """Longest stay in `EAGER` before committing."""
    max_eager_per_turn: int
    """Cap on `EAGER` entries (flushes and preflights) per turn. 0 disables `EAGER`."""
    pause_threshold: float
    """Score at which `SPEAKING` becomes `PAUSED`."""
    resume_threshold: float
    """Score below which `PAUSED` or `EAGER` goes back to `SPEAKING`."""
    start_threshold: float | None
    """Score at or below which `IDLE` opens a turn before any word arrives (faster
    barge-in, more false starts on noise). None waits for the first word."""
    warmup_steps: int
    """Steps ignored at the start of each connection."""
    delay_in_frames: int
    """Model delay, in 80 ms steps: transcript latency against transcript accuracy. Sent
    in the setup message."""
    preflight: bool
    """Emit `PREFLIGHT_TRANSCRIPT` at `EAGER`. False goes from `PAUSED` straight to
    `COMMITTED`."""
    weights: str | Path | None
    """Name or path of the scorer weight set. None forces the `min(p2, p3)` fallback."""

    def __post_init__(self) -> None:
        for name in (
            "eager_threshold",
            "commit_threshold",
            "pause_threshold",
            "resume_threshold",
            "start_threshold",
        ):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        for name in ("min_silence", "max_silence", "confirm_timeout"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be a non-negative number, got {value!r}")

        if not self.resume_threshold < self.pause_threshold:
            raise ValueError(
                "resume_threshold must be below pause_threshold, or a single step could "
                f"both pause and resume (got {self.resume_threshold} >= {self.pause_threshold})"
            )
        if not self.pause_threshold <= self.eager_threshold <= self.commit_threshold:
            raise ValueError(
                "thresholds must satisfy pause_threshold <= eager_threshold <= "
                f"commit_threshold (got {self.pause_threshold}, {self.eager_threshold}, "
                f"{self.commit_threshold})"
            )
        if self.start_threshold is not None and self.start_threshold > self.resume_threshold:
            raise ValueError(
                "start_threshold must not exceed resume_threshold, or a turn could open on "
                f"a score that already reads as a pause (got {self.start_threshold} > "
                f"{self.resume_threshold})"
            )
        if not self.max_silence > self.min_silence:
            raise ValueError(
                f"max_silence ({self.max_silence}) must exceed min_silence ({self.min_silence})"
            )
        if self.confirm_timeout <= 0:
            raise ValueError("confirm_timeout must be positive")
        if self.confirm_steps < 1:
            raise ValueError(f"confirm_steps must be at least 1, got {self.confirm_steps}")
        if self.max_eager_per_turn < 0:
            raise ValueError(f"max_eager_per_turn must be >= 0, got {self.max_eager_per_turn}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        # the server decides which delays it serves and rejects the others at setup
        if self.delay_in_frames < 0:
            raise ValueError(f"delay_in_frames must be >= 0, got {self.delay_in_frames}")


class TurnOverrides(TypedDict, total=False):
    """Knobs to set on top of a preset. Same names and meaning as `TurnProfile`."""

    eager_threshold: float
    commit_threshold: float
    min_silence: float
    max_silence: float
    confirm_steps: int
    confirm_timeout: float
    max_eager_per_turn: int
    pause_threshold: float
    resume_threshold: float
    start_threshold: float | None
    warmup_steps: int
    delay_in_frames: int
    preflight: bool
    weights: str | Path | None


@dataclass(frozen=True)
class _Table:
    eagerness: dict[Eagerness, TurnOverrides]
    shared: TurnOverrides


# Calibrated by `tools/eot/simulate.py tune --joint` on recorded production streams of
# the 13 non-English eot-bench languages (server model 933b5345@150), so English numbers
# are held out. Each eagerness has the lowest mean decision latency whose false cut-offs
# (commits inside hold spans of 0.2 to 5 s, plus commits inside annotated speech, per
# hold span) stay within its budget: 10%, 5%, 2%. The preflight knobs (`eager_threshold`,
# `confirm_timeout`, and the commit threshold after a flush) were tuned on live
# recordings in which the flushes were really sent (`collect.py --mode machine`). Results
# and commands: STT_EXPERIMENTAL_PLAN.md, "Results: simulator and presets (step 4)".
# Thresholds are tied to the scorer and the server model: re-calibrate when either
# changes, never copy them across.
_LEARNED = _Table(
    eagerness={
        "responsive": {
            "pause_threshold": 0.3,
            "resume_threshold": 0.2,
            "min_silence": 0.0,
            "max_silence": 1.92,
            "commit_threshold": 0.86,
            "preflight": True,
            "eager_threshold": 0.75,
            "confirm_timeout": 0.64,
            "max_eager_per_turn": 3,
        },
        "balanced": {
            "pause_threshold": 0.3,
            "resume_threshold": 0.1,
            "min_silence": 0.24,
            "max_silence": 3.68,
            "commit_threshold": 0.94,
            "preflight": True,
            "eager_threshold": 0.75,
            "confirm_timeout": 3.0,
            "max_eager_per_turn": 2,
        },
        "prudent": {
            "pause_threshold": 0.2,
            "resume_threshold": 0.15,
            "min_silence": 0.0,
            "max_silence": 3.36,
            "commit_threshold": 0.98,
            "preflight": True,
            "eager_threshold": 0.9,
            "confirm_timeout": 2.0,
            "max_eager_per_turn": 1,
        },
    },
    shared={
        "confirm_steps": 1,
        "delay_in_frames": 10,
        "start_threshold": None,
        "warmup_steps": 0,
        "weights": "default",
    },
)

# The `min(p2, p3)` fallback, calibrated the same way with preflight off. It runs when
# the weights are missing or were trained for another server model. Preflight stays off:
# there is no post-flush model for this score, and a flush raises every later step.
_FALLBACK = _Table(
    eagerness={
        "responsive": {
            "pause_threshold": 0.2,
            "resume_threshold": 0.08,
            "min_silence": 0.0,
            "max_silence": 0.64,
            "commit_threshold": 0.4,
            "eager_threshold": 0.4,
        },
        "balanced": {
            "pause_threshold": 0.3,
            "resume_threshold": 0.1,
            "min_silence": 0.2,
            "max_silence": 1.68,
            "commit_threshold": 0.54,
            "eager_threshold": 0.54,
        },
        "prudent": {
            "pause_threshold": 0.3,
            "resume_threshold": 0.1,
            "min_silence": 0.16,
            "max_silence": 1.76,
            "commit_threshold": 0.8,
            "eager_threshold": 0.8,
        },
    },
    shared={
        "confirm_steps": 1,
        "delay_in_frames": 10,
        "start_threshold": None,
        # The model needs about a second of audio before its probabilities mean anything:
        # at the start of a connection they read as "silent" whatever the audio holds, and
        # `min(p2, p3)` would pause or commit on them. The first 12 steps (0.96 s) of each
        # connection are therefore not scored. The learned scorer needs none (its presets
        # use 0): it was trained on these first steps too.
        "warmup_steps": 12,
        "weights": None,
        # With preflight off the machine never enters EAGER, so `eager_threshold` above
        # changes nothing. It is still set, equal to `commit_threshold`, because a profile
        # needs a value that keeps the order `pause <= eager <= commit`, and so that a
        # user who overrides `preflight=True` gets a preflight at the commit score, not
        # at an untested lower one.
        "preflight": False,
        "confirm_timeout": 1.0,
        "max_eager_per_turn": 0,
    },
)
_TABLES: dict[ScorerKind, _Table] = {"learned": _LEARNED, "fallback": _FALLBACK}


def preset(
    eagerness: Eagerness = "balanced",
    overrides: TurnOverrides | None = None,
    *,
    scorer: ScorerKind = "learned",
) -> TurnProfile:
    """Build the profile for a named preset, with `overrides` applied on top.

    Args:
        eagerness: False cut-off budget: "responsive" (10%), "balanced" (5%) or
            "prudent" (2%).
        overrides: Knobs set on top of the preset.
        scorer: The scorer the thresholds are for. The stream builds both profiles and
            uses the one matching the scorer it could load, so a missing weights file or
            a server model change also switches to the fallback's thresholds.

    Raises:
        ValueError: An unknown preset name, or a combination of knobs that breaks the
            ordering or ranges documented on `TurnProfile`.
    """
    if scorer not in _TABLES:
        raise ValueError(f"unknown scorer kind {scorer!r}, expected one of {list(_TABLES)}")
    table = _TABLES[scorer]
    if eagerness not in table.eagerness:
        raise ValueError(
            f"unknown eagerness {eagerness!r}, expected one of {list(table.eagerness)}"
        )
    knobs: dict[str, object] = {
        **table.shared,
        **table.eagerness[eagerness],
    }
    if overrides:
        unknown = set(overrides) - {f.name for f in dataclasses.fields(TurnProfile)}
        if unknown:
            raise ValueError(f"unknown turn knobs: {sorted(unknown)}")
        knobs.update(overrides)
    return TurnProfile(**knobs)  # type: ignore[arg-type]
