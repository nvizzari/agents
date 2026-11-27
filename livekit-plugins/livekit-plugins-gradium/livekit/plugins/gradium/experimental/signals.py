"""Typed view of the Gradium STT websocket messages.

Everything downstream (scorer, state machine, benchmark adapter) consumes these types
and never touches the wire format. This module has no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

# Horizons, in seconds, of the inactivity probabilities carried by each step.
HORIZONS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
# Audio covered by one step.
STEP_DURATION = 0.08


@dataclass(frozen=True)
class Ready:
    """The session is set up."""

    delay_in_frames: int
    """Steps the transcript lags behind the audio."""
    frame_size: int
    """Samples per step."""
    sample_rate: int
    model: str
    """Server model version (`model_ext`); scorer weights are tied to it."""
    request_id: str


@dataclass(frozen=True)
class Step:
    """Inactivity probabilities after the model consumed `audio_time` seconds of audio.

    The probabilities are not delayed by the model delay: they describe the audio up to
    `audio_time`, while words for that audio arrive `delay_in_frames` steps later.
    """

    audio_time: float
    """Seconds of audio processed (`total_duration_s`). Always positive."""
    probs: tuple[float, float, float, float]
    """Probability that the speaker stays silent for each of `HORIZONS`."""


@dataclass(frozen=True)
class Word:
    """A transcribed segment, usually one word."""

    text: str
    start_time: float
    """Audio time, in seconds, at which the segment starts."""


@dataclass(frozen=True)
class WordEnd:
    """End time of the segment carried by the previous `Word`."""

    stop_time: float


@dataclass(frozen=True)
class Flushed:
    """The server returned every word held by the model delay at the matching flush."""

    flush_id: int


@dataclass(frozen=True)
class EndOfStream:
    """The server has consumed all audio and sent all words."""


@dataclass(frozen=True)
class ServerError:
    """The server reports a failure and closes the socket next."""

    message: str
    code: int | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class ServerLog:
    level: str
    message: str


Signal: TypeAlias = Ready | Step | Word | WordEnd | Flushed | EndOfStream | ServerError | ServerLog


def parse(data: dict[str, Any]) -> Signal | None:
    """Convert one decoded websocket message, or return None when it carries nothing.

    Warm-up steps are dropped here: the server opens every stream with steps that report
    `total_duration_s == 0`, sent before any audio is processed. Their probabilities
    describe no audio, and counting them would place every later step late.

    Raises:
        ValueError: A known message type is missing a required field.
    """
    type_ = data.get("type")
    try:
        if type_ in ("step", "vad"):
            audio_time = float(data.get("total_duration_s") or 0.0)
            if audio_time <= 0:
                return None
            probs = tuple(float(v["inactivity_prob"]) for v in data["vad"])
            if len(probs) != len(HORIZONS):
                raise ValueError(f"expected {len(HORIZONS)} horizons, got {len(probs)}")
            return Step(audio_time=audio_time, probs=probs)  # type: ignore[arg-type]
        if type_ == "text":
            return Word(text=str(data["text"]), start_time=float(data["start_s"]))
        if type_ == "end_text":
            return WordEnd(stop_time=float(data["stop_s"]))
        if type_ == "flushed":
            return Flushed(flush_id=int(data["flush_id"]))
        if type_ == "ready":
            return Ready(
                delay_in_frames=int(data["delay_in_frames"]),
                frame_size=int(data["frame_size"]),
                sample_rate=int(data["sample_rate"]),
                model=str(data.get("model_ext") or data.get("model_name") or ""),
                request_id=str(data.get("request_id") or ""),
            )
        if type_ == "end_of_stream":
            return EndOfStream()
        if type_ == "error":
            code = data.get("code")
            return ServerError(
                message=str(data.get("message") or "unknown error"),
                code=int(code) if isinstance(code, int) else None,
                raw=data,
            )
        if type_ == "log":
            return ServerLog(
                level=str(data.get("level") or ""), message=str(data.get("message") or "")
            )
    except (KeyError, TypeError) as e:
        raise ValueError(f"malformed Gradium `{type_}` message: {e!r}") from e
    return None
