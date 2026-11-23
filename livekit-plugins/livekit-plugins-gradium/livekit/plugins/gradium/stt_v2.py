# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradium STT whose end of turn combines a local VAD with Gradium's semantic VAD.

The local VAD (Silero) says *when* the user went quiet or started again; it runs on the
audio as it arrives, with no network round trip and no model delay. Gradium's per-step
inactivity probabilities say *whether* a pause ends the turn. The turn ends once the pause
has lasted `min_silence_duration` and Gradium's probability is above a threshold that falls the
longer the pause lasts, or once it has lasted `max_silence_duration`.

Every time below is in seconds of audio since the stream started ("audio time"): the local
VAD reports the audio it processed, and a Gradium `step` describes the audio up to its
`total_duration_s` (measured: after the user resumes, the buckets react within one or two
steps whatever the model delay), so both line up without correction.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import enum
import json
import os
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, get_args

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
    vad as agents_vad,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .api import ws_headers
from .log import log_server_message, logger
from .models import LogLevel
from .stt import SUPPORTED_SAMPLE_RATE, _api_error

EagerPreflightTrigger: TypeAlias = Literal["eager"]

DELAYS_IN_FRAMES = (7, 8, 10, 12, 14, 16, 20, 24, 36, 48)
"""The `delay_in_frames` values Gradium accepts."""
_FRAME_S = 0.08
_DEFAULT_DELAY_IN_FRAMES = 10
# A word that ends this long after the local VAD placed the start of a pause was spoken
# during that pause: the local VAD missed it, and the user is talking again. A word's
# `stop_s` lands within ~0.05 s of its real end and the VAD places a pause up to ~0.1 s
# late, so the words that end the turn stay well inside this margin.
_WORD_RESUME_MARGIN_S = 0.25
# How long, beyond its model delay, Gradium takes to send a word: measured 1.0 s at delay
# 7 and 1.2 s at delay 10 from the start of the word, at p50.
_WORD_LAG_S = 0.6
# How long to wait for a `flushed` reply before ending the turn without it: several times
# the measured 0.2-0.35 s round trip.
_FLUSH_REPLY_TIMEOUT_S = 1.5
# How often the stream checks that audio still reaches the local VAD while a turn is open.
_STALL_CHECK_S = 0.1
# Audio pushed to the local VAD but not processed yet, beyond its window and resampling.
_VAD_BACKLOG_S = 0.1


@dataclass
class STTV2Options:
    language: LanguageCode
    temperature: float | None
    keywords: list[str] | None
    keyword_boost: float | None
    log_level: LogLevel | None
    delay_in_frames: int | None
    speech_threshold: float
    silence_threshold: float
    min_speech_duration: float
    min_silence_duration: float
    max_silence_duration: float
    eot_horizon: float
    eot_threshold_start: float
    eot_threshold_end: float
    eager_horizon: float
    eager_threshold: float | None
    preflight_trigger: EagerPreflightTrigger | None


class STTV2(stt.STT):
    def __init__(
        self,
        *,
        vad: agents_vad.VAD,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_name: str = "default",
        http_session: aiohttp.ClientSession | None = None,
        language: str = "en",
        temperature: float | None = None,
        keywords: list[str] | None = None,
        keyword_boost: float | None = None,
        log_level: LogLevel | None = None,
        delay_in_frames: int | None = 7,
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.35,
        min_speech_duration: float = 0.15,
        min_silence_duration: float = 0.1,
        max_silence_duration: float = 2.5,
        eot_horizon: float = 3.0,
        eot_threshold_start: float = 0.5,
        eot_threshold_end: float = 0.4,
        eager_horizon: float = 2.0,
        eager_threshold: float | None = None,
        preflight_trigger: EagerPreflightTrigger | None = None,
    ) -> None:
        """Create a Gradium STT that ends turns from a local VAD and Gradium's semantic VAD.

        Use it with `AgentSession(turn_detection="stt", ...)`: the STT owns the end of turn.

        How a turn ends:

        1. The local `vad` hears speech for `min_speech_duration`: the turn starts.
        2. Its speech probability falls below `silence_threshold`: a pause starts. It ends
           when the probability stays above `speech_threshold` for `min_speech_duration`
           again, or when Gradium sends a word that ended during the pause.
        3. Once the pause has lasted `min_silence_duration`, each Gradium step is checked:
           - Eager: when the `eager_horizon` probability reaches `eager_threshold`, the STT
             flushes, which makes Gradium send the words its model delay still holds, and
             previews them as a `PREFLIGHT_TRANSCRIPT` so the session can start the LLM.
           - Commit: when the `eot_horizon` probability reaches a threshold that falls from
             `eot_threshold_start` (at `min_silence_duration`) to `eot_threshold_end` (at
             `max_silence_duration`), the turn ends: `FINAL_TRANSCRIPT` then
             `END_OF_SPEECH`, as soon as the flush is answered.
        4. After `max_silence_duration` of pause the turn ends whatever Gradium says.

        If the user starts again before the commit, the pending end is dropped, and a stale
        preview is superseded by an `INTERIM_TRANSCRIPT` so the session drops its early
        LLM request.

        Parameter sets that work together, with the other options at their defaults. They
        were measured live against Gradium (delay 7, Silero) on 45 synthetic clips of one to
        three sentences, 18 of them with a scripted mid-sentence pause of 0.6 s or 1.5 s.
        Times are seconds after the user stopped, p50 / p90; every premature end was in a
        1.5 s pause:

        - Responsive: `min_silence_duration=0.1, max_silence_duration=2.0,
          eot_threshold_start=0.4, eot_threshold_end=0.4`. Final 0.78 / 0.98; 12 premature
          ends in 90 runs.
        - Balanced, the defaults: `min_silence_duration=0.1, max_silence_duration=2.5,
          eot_threshold_start=0.5, eot_threshold_end=0.4`. Final 0.81 / 1.09; 6 premature
          ends in 90 runs.
        - Patient, speculative: `min_silence_duration=0.3, max_silence_duration=3.0,
          eot_threshold_start=0.95, eot_threshold_end=0.6, eager_threshold=0.5`. Final
          1.21 / 1.55, but the preview starts the LLM at 0.84 / 1.10; 1 premature end in 45
          runs. Costs the LLM requests of the previews (52 in those 45 runs). Without the
          eager preview the same thresholds give a final at 1.56 / 2.07.

        For comparison, `STT` with its defaults: final 1.97 / 2.38, no premature end.

        Args:
            vad: The local VAD, such as `silero.VAD.load()`. Only its per-window speech
                probability is used; its own speech and silence durations are ignored.
                The STT runs its own stream of it, besides the session's.
            delay_in_frames: Gradium's model delay, in 80 ms frames: one of 7, 8, 10, 12,
                14, 16, 20, 24, 36, 48. Lower sends words sooner, and so makes the flush
                and the final transcript faster; higher can transcribe more accurately.
                None uses the server's default (10).
            speech_threshold: Local VAD probability at or above which a window is speech.
            silence_threshold: Local VAD probability below which a window is silence.
                Between the two, the VAD keeps its state: the gap is the hysteresis that
                stops a probability hovering around one threshold from flapping.
            min_speech_duration: Seconds of continuous speech that start a turn, or end a
                pause. Higher ignores coughs and clicks, but reacts later to the user
                starting again.
            min_silence_duration: Seconds of pause before the turn may end. The floor of
                the latency.
            max_silence_duration: Seconds of pause after which the turn ends whatever
                Gradium says. The ceiling of the latency.
            eot_horizon: Which of Gradium's inactivity horizons, in seconds, decides the
                end of turn; the hosted model sends 0.5, 1, 2 and 3. A longer horizon asks
                "will the user stay quiet for that long", so it is slower to rise and cuts
                fewer mid-sentence pauses. A horizon the server does not send fails the
                stream with an error naming the ones it sends.
            eot_threshold_start: `eot_horizon` probability that ends the turn at
                `min_silence_duration`. Raise it to cut fewer pauses, lower it to end sooner.
            eot_threshold_end: `eot_horizon` probability that ends the turn at
                `max_silence_duration`; the threshold falls in a straight line in between.
            eager_horizon: Which Gradium horizon, in seconds, triggers the eager flush and
                preview.
            eager_threshold: `eager_horizon` probability that triggers them, once the pause
                lasted `min_silence_duration`. None flushes only at the commit, so the final comes
                one flush round trip (about 0.25 s) after the decision; the local VAD can
                still cancel the commit during that time if the user starts again, which
                measurably avoids premature turns. An eager flush answered before the
                decision gives up that window: pair it with a stricter commit threshold.
            preflight_trigger: `"eager"` previews the words as a `PREFLIGHT_TRANSCRIPT`
                when the eager flush is answered, so it needs `eager_threshold`; None, the
                default, never previews.
        """
        if preflight_trigger is not None and preflight_trigger not in get_args(
            EagerPreflightTrigger
        ):
            raise ValueError(
                f"unknown preflight_trigger {preflight_trigger!r}, "
                f"expected one of {get_args(EagerPreflightTrigger)} or None"
            )
        if delay_in_frames is not None and delay_in_frames not in DELAYS_IN_FRAMES:
            raise ValueError(f"delay_in_frames must be one of {DELAYS_IN_FRAMES} or None")
        for name, value in (
            ("speech_threshold", speech_threshold),
            ("silence_threshold", silence_threshold),
            ("eot_threshold_start", eot_threshold_start),
            ("eot_threshold_end", eot_threshold_end),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if eager_threshold is not None and not 0.0 <= eager_threshold <= 1.0:
            raise ValueError(f"eager_threshold must be within [0, 1], got {eager_threshold}")
        if silence_threshold > speech_threshold:
            raise ValueError("silence_threshold must not be above speech_threshold")
        if min_speech_duration < 0:
            raise ValueError("min_speech_duration must be >= 0")
        if not 0.0 <= min_silence_duration < max_silence_duration:
            raise ValueError("expected 0 <= min_silence_duration < max_silence_duration")
        for name, horizon in (("eot_horizon", eot_horizon), ("eager_horizon", eager_horizon)):
            if horizon <= 0:
                raise ValueError(f"{name} must be a positive number of seconds, got {horizon}")
        if preflight_trigger == "eager" and eager_threshold is None:
            raise ValueError("preflight_trigger='eager' needs an eager_threshold")
        if (
            eager_threshold is not None
            and eager_horizon == eot_horizon
            and eager_threshold >= eot_threshold_start
        ):
            # the commit threshold only falls from eot_threshold_start: on the same horizon,
            # the commit is always decided first and the eager stage never runs
            raise ValueError(
                f"eager_threshold ({eager_threshold}) on the commit's horizon must be below "
                f"eot_threshold_start ({eot_threshold_start})"
            )

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
                manual_flush=True,
            ),
        )

        api_key = api_key or os.environ.get("GRADIUM_API_KEY")
        if not api_key:
            raise ValueError(
                "Gradium API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `GRADIUM_API_KEY` environment variable"
            )
        self._api_key = api_key
        self._model_endpoint = (
            model_endpoint
            or os.environ.get("GRADIUM_MODEL_ENDPOINT")
            or "wss://api.gradium.ai/api/speech/asr"
        )
        self._model_name = model_name
        self._vad = vad
        self._opts = STTV2Options(
            language=LanguageCode(language),
            temperature=temperature,
            keywords=list(keywords) if keywords is not None else None,
            keyword_boost=keyword_boost,
            log_level=log_level,
            delay_in_frames=delay_in_frames,
            speech_threshold=speech_threshold,
            silence_threshold=silence_threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            max_silence_duration=max_silence_duration,
            eot_horizon=eot_horizon,
            eot_threshold_start=eot_threshold_start,
            eot_threshold_end=eot_threshold_end,
            eager_horizon=eager_horizon,
            eager_threshold=eager_threshold,
            preflight_trigger=preflight_trigger,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStreamV2]()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "Gradium"

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Gradium STT only supports streaming")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStreamV2:
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.language = LanguageCode(language)
        stream = SpeechStreamV2(
            stt=self,
            opts=opts,
            vad=self._vad,
            conn_options=conn_options,
            api_key=self._api_key,
            model_endpoint=self._model_endpoint,
            model_name=self._model_name,
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream


class _Pause(enum.Enum):
    """Where the current pause stands; every stage but LISTENING has a flush id.

    LISTENING -> PREVIEW_SENT -> PREVIEWED: the eager flush and its reply.
    LISTENING, PREVIEW_SENT or PREVIEWED -> ENDING -> CONFIRMING -> commit: the end of
    the turn is decided, its flush answered, and it commits once the decision still holds.
    Any stage -> LISTENING when the user talks again.
    """

    LISTENING = enum.auto()
    """No flush for this pause."""
    PREVIEW_SENT = enum.auto()
    """The eager flush is in flight."""
    PREVIEWED = enum.auto()
    """The eager flush is answered: every word spoken before it has arrived."""
    ENDING = enum.auto()
    """The end of the turn is decided; its flush is in flight."""
    CONFIRMING = enum.auto()
    """The end of the turn is decided and its flush answered."""


class _TurnTracker:
    """The end-of-turn state machine; synchronous and free of I/O, so it can be replayed.

    Feed it the local VAD's windows (`on_vad`) and Gradium's messages (`on_step`,
    `on_text`, `on_end_text`, `on_flushed`), all in audio time. It emits speech events
    through `emit` and asks for a flush through `request_flush`, which returns the id the
    matching `flushed` will carry.
    """

    def __init__(
        self,
        opts: STTV2Options,
        *,
        emit: Callable[[stt.SpeechEvent], None],
        request_flush: Callable[[], int],
    ) -> None:
        self._opts = opts
        self._emit = emit
        self._request_flush = request_flush
        self.word_lag_s = _DEFAULT_DELAY_IN_FRAMES * _FRAME_S + _WORD_LAG_S

        # local VAD
        self._vad_t = 0.0
        self._vad_wall: float | None = None
        self._speech_run = 0.0
        self._speech_window_t = 0.0  # end of the last window at or above speech_threshold
        self._speaking = False

        # Gradium's latest step
        self._probs: Mapping[float, float] | None = None
        self._probs_t = 0.0

        # the turn
        self._turn_open = False
        self._words: list[str] = []
        self._pause = _Pause.LISTENING
        self._flush_id: int | None = None
        self._preview_text = ""
        self._end_reason = ""  # why the end of the turn was decided, for the logs
        # Three times, each with one use:
        # - the latest audio time known to hold speech (a speech window of the local VAD or
        #   the end of a word): the current pause is measured from it;
        self._speech_seen_t = 0.0
        # - the end of the turn's last word, on the stream's clock, for the final's
        #   `end_time`; None until the last word's end arrives;
        self._word_end_t: float | None = None
        # - since when speech has produced no word: the noise guard's clock.
        self._wordless_since = 0.0

    @property
    def turn_open(self) -> bool:
        return self._turn_open

    @property
    def pending_flush(self) -> int | None:
        """The id of this pause's flush while its reply is awaited."""
        if self._pause in (_Pause.PREVIEW_SENT, _Pause.ENDING):
            return self._flush_id
        return None

    @property
    def vad_time(self) -> tuple[float, float | None]:
        """The audio time of the last local VAD window, and the wall-clock time it came."""
        return self._vad_t, self._vad_wall

    def on_audio_stalled(self) -> None:
        """No audio has come for a while (track ended, muted): end the open turn.

        Time only moves with the audio, so without this a turn whose audio stops would
        never end.
        """
        if not self._turn_open:
            return
        self._speaking = False
        self._speech_run = 0.0
        self._end_turn("audio_stalled")

    def set_delay_in_frames(self, delay: int) -> None:
        self.word_lag_s = delay * _FRAME_S + _WORD_LAG_S

    def on_vad(self, t: float, prob: float, *, wall: float) -> None:
        """A local VAD window ending at audio time `t`, processed at wall-clock `wall`."""
        dt = max(t - self._vad_t, 0.0)
        self._vad_t, self._vad_wall = t, wall
        if prob >= self._opts.speech_threshold:
            self._speech_run += dt
            self._speech_window_t = t
        elif prob < self._opts.silence_threshold:
            self._speech_run = 0.0

        if self._speaking:
            if prob < self._opts.silence_threshold:
                self._speaking = False
            else:
                self._speech_seen_t = t
        elif self._speech_run >= self._opts.min_speech_duration and self._speech_run > 0:
            self._speaking = True
            self._speech_seen_t = t
            self._on_speech(onset=t - self._speech_run)
        self._evaluate()

    def on_step(self, t: float, probs: Mapping[float, float]) -> None:
        """A Gradium step: inactivity probability by horizon, for the audio up to `t`."""
        self._probs, self._probs_t = probs, t
        self._evaluate()

    def on_text(self, text: str) -> None:
        if not self._turn_open:
            self._open_turn(onset=None)
            # the local VAD missed this speech; until the word's end arrives, its arrival
            # is the only evidence of when it was spoken
            self._speech_seen_t = max(self._speech_seen_t, self._vad_t)
        elif self._pause in (_Pause.PREVIEWED, _Pause.CONFIRMING):
            # the flush returned every word the model held when it was sent: a word after
            # its reply was spoken after the pause started
            self._resume("word after the flush reply")
        self._words.append(text)
        self._word_end_t = None
        self._emit_text(stt.SpeechEventType.INTERIM_TRANSCRIPT, " ".join(self._words))

    def on_end_text(self, stop_s: float, *, stream_time: float | None = None) -> None:
        """The end of the word Gradium sent last, in audio time.

        `stream_time` is the same instant on the stream's clock, the one `end_time` uses;
        it differs from audio time after a reconnect.
        """
        if not self._turn_open:
            return  # a late end of the last turn's last word
        self._word_end_t = stop_s if stream_time is None else stream_time
        self._wordless_since = max(self._wordless_since, stop_s)
        if stop_s <= self._speech_seen_t:
            return
        if self._turn_open and not self._speaking:
            if stop_s > self._speech_seen_t + _WORD_RESUME_MARGIN_S:
                self._resume("word ended during the pause")
        self._speech_seen_t = stop_s

    def on_flushed(self, flush_id: int) -> None:
        if flush_id != self._flush_id:
            return  # the flush of a pause the user has since ended
        if self._pause is _Pause.PREVIEW_SENT:
            self._pause = _Pause.PREVIEWED
        elif self._pause is _Pause.ENDING:
            self._pause = _Pause.CONFIRMING
        else:
            return
        self._evaluate()
        if self._pause is _Pause.PREVIEWED and self._opts.preflight_trigger == "eager":
            text = " ".join(self._words)
            if text and text != self._preview_text:
                self._preview_text = text
                self._emit_text(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, text)

    def on_flush_lost(self) -> None:
        """The pending flush will never be answered (reconnect): send a new one if needed."""
        if self._pause is _Pause.PREVIEW_SENT:
            self._pause, self._flush_id = _Pause.LISTENING, None
        elif self._pause is _Pause.ENDING:
            self._flush_id = self._request_flush()

    def commit(self, reason: str = "") -> None:
        """End the open turn now with the words received so far.

        `reason` names what ended it, for the logs; by default, why the end was decided.
        """
        if not self._turn_open:
            return
        text = " ".join(self._words)
        logger.debug(
            "turn ended: %s",
            reason or self._end_reason,
            extra={"reason": reason or self._end_reason, "words": len(self._words)},
        )
        end_t = self._word_end_t
        alternatives = [
            stt.SpeechData(
                text=text, language=self._opts.language, end_time=end_t if end_t else 0.0
            )
        ]
        self._emit(
            stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=alternatives)
        )
        self._emit(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                alternatives=alternatives if end_t is not None else [],
                speech_end_time=self._wall_of(self._speech_seen_t),
            )
        )
        self._turn_open = False
        self._words = []
        self._word_end_t = None
        self._reset_pause()

    def _open_turn(self, onset: float | None) -> None:
        self._turn_open = True
        self._wordless_since = onset if onset is not None else self._vad_t
        self._reset_pause()
        self._emit(
            stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH,
                speech_start_time=self._wall_of(onset) if onset is not None else None,
            )
        )

    def _on_speech(self, onset: float) -> None:
        if not self._turn_open:
            self._open_turn(onset)
        else:
            # the words of the resumed speech are a model delay away: the noise guard
            # must not count the pause, or it ends the turn mid-sentence after a long one
            self._wordless_since = max(self._wordless_since, onset)
            self._resume("local VAD heard speech")

    def _resume(self, cause: str) -> None:
        """The user is talking again: the pause did not end the turn."""
        if self._pause is not _Pause.LISTENING:
            logger.debug(
                "user resumed (%s), dropping the %s pause",
                cause,
                self._pause.name,
                extra={"cause": cause, "flush_id": self._flush_id},
            )
        stale_preview = bool(self._preview_text)
        self._reset_pause()
        if stale_preview and self._words:
            # an interim after a preview is how the session learns the preview is stale
            self._emit_text(stt.SpeechEventType.INTERIM_TRANSCRIPT, " ".join(self._words))

    def _reset_pause(self) -> None:
        self._pause = _Pause.LISTENING
        self._flush_id = None
        self._preview_text = ""
        self._end_reason = ""

    def _eot_threshold(self, silence: float) -> float:
        o = self._opts
        frac = (silence - o.min_silence_duration) / (
            o.max_silence_duration - o.min_silence_duration
        )
        frac = min(max(frac, 0.0), 1.0)
        return o.eot_threshold_start + (o.eot_threshold_end - o.eot_threshold_start) * frac

    def _evaluate(self) -> None:
        if not self._turn_open:
            return
        o = self._opts
        now = max(self._vad_t, self._probs_t)
        if self._speaking:
            # The local VAD hears speech but Gradium has transcribed nothing for far longer
            # than its delay: background noise, which would otherwise hold the turn open.
            if now - self._wordless_since >= o.max_silence_duration + self.word_lag_s:
                self._end_turn("no_words", wordless=now - self._wordless_since)
            elif self._pause is not _Pause.LISTENING:
                # only the guard decides while the user speaks, and the words its flush
                # brought back show the speech was real: drop the decision
                self._reset_pause()
            return

        if self._speech_run > 0 and now - self._speech_window_t <= o.min_speech_duration:
            # speech may be resuming, and neither the local VAD nor Gradium's buckets can
            # tell yet: decide nothing until it is confirmed or dies out
            return

        match self._pause:
            case _Pause.CONFIRMING:
                self.commit()
                return
            case _Pause.ENDING:
                return

        silence = now - self._speech_seen_t
        if silence < o.min_silence_duration:
            return
        # a step from before the pause still describes speech
        probs = self._probs if self._probs_t >= self._speech_seen_t else {}
        eot_prob = probs.get(o.eot_horizon) if probs else None
        eager_prob = probs.get(o.eager_horizon) if probs else None
        threshold = self._eot_threshold(silence)
        if silence >= o.max_silence_duration:
            self._end_turn("max_silence_duration", silence=silence)
        elif eot_prob is not None and eot_prob >= threshold:
            self._end_turn("gradium", silence=silence, probability=eot_prob, threshold=threshold)
        elif (
            self._pause is _Pause.LISTENING
            and eager_prob is not None
            and o.eager_threshold is not None
            and eager_prob >= o.eager_threshold
        ):
            self._flush_id = self._request_flush()
            self._pause = _Pause.PREVIEW_SENT
            logger.debug(
                "eager flush",
                extra={"silence": silence, "probability": eager_prob, "flush_id": self._flush_id},
            )

    def _end_turn(self, reason: str, **details: float) -> None:
        """The turn should end: commit once a flush has returned every held word."""
        if self._pause in (_Pause.LISTENING, _Pause.PREVIEW_SENT, _Pause.PREVIEWED):
            self._end_reason = reason
            logger.debug("end of turn decided: %s", reason, extra={"reason": reason, **details})
        match self._pause:
            case _Pause.LISTENING:
                self._flush_id = self._request_flush()
                self._pause = _Pause.ENDING
            case _Pause.PREVIEW_SENT:
                self._pause = _Pause.ENDING  # its reply will do
            case _Pause.PREVIEWED | _Pause.CONFIRMING:
                self.commit()
            case _Pause.ENDING:
                pass

    def _emit_text(self, type_: stt.SpeechEventType, text: str) -> None:
        self._emit(
            stt.SpeechEvent(
                type=type_,
                alternatives=[stt.SpeechData(text=text, language=self._opts.language)],
            )
        )

    def _wall_of(self, t: float) -> float | None:
        if self._vad_wall is None:
            return None
        return self._vad_wall - (self._vad_t - t)


def _horizons(step: dict[str, Any]) -> dict[float, float]:
    """A step's inactivity probabilities by horizon in seconds; empty for a warm-up step."""
    if step.get("total_duration_s", 0.0) <= 0:
        return {}  # sent before any audio was processed: a warm-up estimate
    return {float(v["horizon_s"]): float(v["inactivity_prob"]) for v in step.get("vad", [])}


class SpeechStreamV2(stt.SpeechStream):
    @dataclass
    class _TurnFlush(stt.SpeechStream._FlushSentinel):
        """A flush the end-of-turn logic asked for, behind the audio already queued."""

        flush_id: int

    def __init__(
        self,
        *,
        stt: STTV2,
        opts: STTV2Options,
        vad: agents_vad.VAD,
        conn_options: APIConnectOptions,
        api_key: str,
        model_endpoint: str,
        model_name: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SUPPORTED_SAMPLE_RATE)
        self._opts = opts
        self._vad = vad
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name
        self._session = http_session
        # ids are never reused, so a reply from an earlier connection cannot match
        self._last_flush_id = 0
        # seconds of audio taken from the input over every connection so far
        self._audio_pushed_s = 0.0
        # the turn flush awaiting its reply, and when to stop waiting, in event-loop time
        self._flush_deadline: tuple[int, float] | None = None
        self._flush_requested = asyncio.Event()
        # after a flush timed out and ended the turn, its words may still come: they
        # belong to that turn, so drop them until its reply, or until this event-loop time
        self._timed_out_flush: tuple[int, float] | None = None
        # Gradium's id for the current connection, on every event for support and traces
        self._request_id = ""
        self._tracker = _TurnTracker(opts, emit=self._emit, request_flush=self._request_turn_flush)

    def _emit(self, event: stt.SpeechEvent) -> None:
        event.request_id = self._request_id
        self._event_ch.send_nowait(event)

    def _next_flush_id(self) -> int:
        self._last_flush_id += 1
        return self._last_flush_id

    def _request_turn_flush(self) -> int:
        flush_id = self._next_flush_id()
        if self._input_ch.closed:
            # no flush can follow end_of_stream: the server's own end_of_stream ends the turn
            return flush_id
        deadline = asyncio.get_running_loop().time() + _FLUSH_REPLY_TIMEOUT_S
        self._flush_deadline = (flush_id, deadline)
        self._flush_requested.set()
        self._input_ch.send_nowait(self._TurnFlush(flush_id))
        return flush_id

    async def _run(self) -> None:
        # the base class calls _run again after a retryable error: a new connection and a
        # new local VAD stream, both starting at the audio consumed so far
        offset = self._audio_pushed_s
        if self._tracker.pending_flush is not None:
            # its reply was due on the connection that failed
            self._flush_deadline = None
            self._tracker.on_flush_lost()
        vad_stream = self._vad.stream()
        vad_task = asyncio.create_task(self._vad_task(vad_stream, offset))
        ws_task = asyncio.create_task(self._run_ws(vad_stream, offset))
        watchdog = asyncio.create_task(self._stall_watchdog())
        flush_timer = asyncio.create_task(self._flush_timer())
        try:
            done, _ = await asyncio.wait([vad_task, ws_task], return_when=asyncio.FIRST_COMPLETED)
            if ws_task not in done:
                vad_task.result()  # raises the VAD's error
                raise RuntimeError("the local VAD stream ended before the input")
            ws_task.result()
        except APIError as e:
            # The turn survives a reconnect, so a transient failure must not end it while
            # the user is talking; only when the stream gives up are its words final.
            if not e.retryable or self._num_retries >= self._conn_options.max_retry:
                self._tracker.commit("stream failed")
            raise
        finally:
            await utils.aio.cancel_and_wait(ws_task, vad_task, watchdog, flush_timer)
            await vad_stream.aclose()

    async def _flush_timer(self) -> None:
        """Go on without a turn flush's reply once it is `_FLUSH_REPLY_TIMEOUT_S` late."""
        loop = asyncio.get_running_loop()
        while True:
            self._flush_requested.clear()
            if self._flush_deadline is None:
                await self._flush_requested.wait()
                continue
            flush_id, deadline = self._flush_deadline
            if (delay := deadline - loop.time()) > 0:
                # a newer flush replaces this one and wakes the timer
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._flush_requested.wait(), delay)
                continue
            self._flush_deadline = None
            if self._tracker.pending_flush != flush_id:
                continue
            logger.warning(
                "Gradium did not answer flush %d within %.1fs, going on without it",
                flush_id,
                _FLUSH_REPLY_TIMEOUT_S,
            )
            was_open = self._tracker.turn_open
            self._tracker.on_flushed(flush_id)
            if was_open and not self._tracker.turn_open:
                self._timed_out_flush = (flush_id, loop.time() + _FLUSH_REPLY_TIMEOUT_S)

    async def _stall_watchdog(self) -> None:
        """End the open turn once no audio has reached the local VAD for `max_silence_duration`."""
        tracker = self._tracker
        while True:
            await asyncio.sleep(_STALL_CHECK_S)
            vad_t, vad_wall = tracker.vad_time
            if not tracker.turn_open or vad_wall is None or self._input_ch.closed:
                continue
            if time.time() - vad_wall < self._opts.max_silence_duration:
                continue
            if not self._input_ch.empty() or self._audio_pushed_s - vad_t > _VAD_BACKLOG_S:
                continue  # the audio is late (backlog, reconnect), not missing
            logger.debug("no audio for %.1fs, ending the turn", time.time() - vad_wall)
            tracker.on_audio_stalled()

    async def _vad_task(self, vad_stream: agents_vad.VADStream, offset: float) -> None:
        async for ev in vad_stream:
            if ev.type == agents_vad.VADEventType.INFERENCE_DONE:
                self._tracker.on_vad(ev.timestamp + offset, ev.probability, wall=time.time())

    async def _run_ws(self, vad_stream: agents_vad.VADStream, offset: float) -> None:
        closing_ws = False
        # ids of the flushes the caller asked for, whose reply must always yield a final
        manual_flushes: set[int] = set()
        tracker = self._tracker

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=SUPPORTED_SAMPLE_RATE, num_channels=1, samples_per_channel=1920
            )
            async for data in self._input_ch:
                flush_id: int | None = None
                frames: list[rtc.AudioFrame]
                if isinstance(data, self._TurnFlush):
                    # the flush covers the audio sent so far, partial frame included
                    frames = audio_bstream.flush()
                    flush_id = data.flush_id
                elif isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                    flush_id = self._next_flush_id()
                    manual_flushes.add(flush_id)
                else:
                    vad_stream.push_frame(data)
                    self._audio_pushed_s += data.samples_per_channel / data.sample_rate
                    frames = audio_bstream.write(data.data.tobytes())

                try:
                    for frame in frames:
                        msg = {"type": "audio", "audio": base64.b64encode(frame.data).decode()}
                        await ws.send_str(json.dumps(msg))
                    if flush_id is not None:
                        await ws.send_str(json.dumps({"type": "flush", "flush_id": flush_id}))
                except (aiohttp.ClientError, ConnectionError):
                    return  # the server closed first; recv_task raises the retryable error

            closing_ws = True
            try:
                await ws.send_str(json.dumps({"type": "end_of_stream"}))
            except (aiohttp.ClientError, ConnectionError):
                return

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            monitor = utils.aio.WebSocketMonitor(
                ws, liveness_timeout=self._conn_options.liveness_timeout
            )
            while True:
                try:
                    msg = await asyncio.wait_for(monitor.receive(), timeout=5.0)
                except asyncio.TimeoutError:
                    if closing_ws:
                        tracker.commit("no end_of_stream reply")
                        return
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        tracker.commit("connection closed after the input")
                        return
                    raise APIStatusError(
                        "Gradium connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("Unexpected Gradium message type: %s", msg.type)
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.exception("Failed to decode message from Gradium")
                    continue
                if not isinstance(data, dict):
                    logger.error("Unexpected Gradium message: %s", msg.data)
                    continue

                type_ = data.get("type", "")
                if type_ == "error":
                    raise _api_error(data)
                if type_ == "step":
                    self._check_horizons(data)
                if type_ == "end_of_stream":
                    if closing_ws:
                        tracker.commit("end_of_stream")
                        return
                    raise APIStatusError(
                        "Gradium ended the session while input was still open",
                        status_code=ws.close_code or -1,
                        body=data,
                    )
                try:
                    self._handle_message(type_, data, offset, manual_flushes)
                except Exception:
                    logger.exception("Failed to process message from Gradium")

        ws = await self._connect_ws()
        try:
            self._input_ch.restart_provider_clock()
            tasks = [asyncio.create_task(send_task(ws)), asyncio.create_task(recv_task(ws))]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            await ws.close()

    def _check_horizons(self, step: dict[str, Any]) -> None:
        """Fail the stream, without retrying, on a step lacking a horizon the options use."""
        sent = _horizons(step)
        if not sent:
            return  # a warm-up step
        needed = {self._opts.eot_horizon}
        if self._opts.eager_threshold is not None:
            needed.add(self._opts.eager_horizon)
        if missing := needed - sent.keys():
            raise APIError(
                f"Gradium sent no inactivity horizon {sorted(missing)}; "
                f"it sends {sorted(sent)}: set eot_horizon and eager_horizon to one of them",
                body=step,
                retryable=False,
            )

    def _handle_message(
        self, type_: str, data: dict[str, Any], offset: float, manual_flushes: set[int]
    ) -> None:
        tracker = self._tracker
        if type_ in ("text", "end_text") and self._timed_out_flush is not None:
            if asyncio.get_running_loop().time() < self._timed_out_flush[1]:
                return  # held words of a turn that ended without them
            self._timed_out_flush = None
        if type_ == "step":
            if not (probs := _horizons(data)):
                return
            tracker.on_step(data["total_duration_s"] + offset, probs)
            self._mark_provider_progress()
        elif type_ == "text":
            tracker.on_text(data["text"])
            self._mark_provider_progress()
        elif type_ == "end_text":
            # Gradium's clock restarts with each connection, as does `start_time_offset`,
            # the stream time of the connection's first frame
            stop_s = data["stop_s"]
            tracker.on_end_text(stop_s + offset, stream_time=stop_s + self.start_time_offset)
            self._mark_provider_progress()
        elif type_ == "flushed":
            flush_id = data.get("flush_id")
            if not isinstance(flush_id, int):
                return
            if self._flush_deadline is not None and self._flush_deadline[0] == flush_id:
                self._flush_deadline = None
            if self._timed_out_flush is not None and self._timed_out_flush[0] == flush_id:
                self._timed_out_flush = None
            if flush_id in manual_flushes:
                manual_flushes.discard(flush_id)
                if tracker.turn_open:
                    tracker.commit("session flush")
                else:
                    # nothing was heard, but the caller's flush still needs a final
                    self._emit(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[stt.SpeechData(text="", language=self._opts.language)],
                        )
                    )
            else:
                tracker.on_flushed(flush_id)
        elif type_ == "ready":
            if isinstance(request_id := data.get("request_id"), str):
                self._request_id = request_id
            if isinstance(delay := data.get("delay_in_frames"), int) and delay > 0:
                tracker.set_delay_in_frames(delay)
        elif type_ == "log":
            log_server_message(data)
        else:
            logger.warning("Unknown message type from Gradium: %s", type_)

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        ws = await self._session.ws_connect(
            self._model_endpoint, headers=ws_headers(self._api_key), autoping=False
        )
        json_config: dict[str, Any] = {"language": self._opts.language.language}
        if self._opts.delay_in_frames is not None:
            json_config["delay_in_frames"] = self._opts.delay_in_frames
        if self._opts.temperature is not None:
            json_config["temp"] = self._opts.temperature
        if self._opts.keywords:
            keywords: dict[str, Any] = {"words": self._opts.keywords}
            if self._opts.keyword_boost is not None:
                keywords["boost"] = self._opts.keyword_boost
            json_config["keywords"] = keywords
        setup: dict[str, Any] = {
            "type": "setup",
            "model_name": self._model_name,
            "input_format": "pcm",
            "json_config": json_config,
        }
        if self._opts.log_level is not None:
            setup["log_level"] = self._opts.log_level
        await ws.send_str(json.dumps(setup))
        return ws
