# Copyright 2026 LiveKit, Inc.
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

"""`STTExperimental`: Gradium speech-to-text that owns end-of-turn detection.

The stream only does I/O: it connects, sends audio in 80 ms frames, parses server
messages with `signals.parse`, feeds them to `machine.TurnMachine` and turns the
machine's actions into `SpeechEvent`s and `flush` messages. Every turn decision lives in
the machine. Use it with `turn_detection="stt"`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import os
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ..api import ws_headers
from ..log import log_server_message, logger
from ..models import LogLevel
from ..tts import _close_error
from . import signals
from .machine import (
    Action,
    EndOfSpeech,
    Final,
    Interim,
    Preflight,
    SendFlush,
    StartOfSpeech,
    TurnMachine,
)
from .profiles import Eagerness, ScorerKind, TurnOverrides, TurnProfile, preset
from .scorer import FallbackScorer, LearnedScorer, Scorer, load_scorer

SAMPLE_RATE = 24000
# Samples per audio message: one 80 ms model step.
SAMPLES_PER_FRAME = 1920
# Language hint that asks the server to transcribe any language.
ANY_LANGUAGE = "any"
DEFAULT_GRADIUM_API = "api.gradium.ai"
# Path of the streaming speech-to-text websocket on a Gradium API host:
# https://docs.gradium.ai/api-reference/endpoint/stt-websocket
_STT_PATH = "/api/speech/asr"
# Longest wait, after the client's end_of_stream, for the server to return its last
# words and its own end_of_stream.
_DRAIN_TIMEOUT = 5.0


@dataclass(frozen=True)
class _Settings:
    """Everything a stream is configured with; rebuilt, never mutated, on update."""

    eagerness: Eagerness
    overrides: TurnOverrides
    language: str
    keywords: tuple[str, ...]
    keyword_boost: float | None
    temperature: float | None
    log_level: LogLevel | None
    profiles: dict[ScorerKind, TurnProfile] = dataclasses.field(init=False)
    """The preset resolved for each scorer, overrides applied to both."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profiles",
            {
                kind: preset(self.eagerness, self.overrides, scorer=kind)
                for kind in ("learned", "fallback")
            },
        )

    @property
    def weights(self) -> str | Path | None:
        """The weight set to load; None asks for the fallback scorer."""
        return self.profiles["learned"].weights

    @property
    def profile(self) -> TurnProfile:
        """The profile used when the requested scorer loads."""
        return self.profiles["learned" if self.weights is not None else "fallback"]

    def profile_for(self, scorer: Scorer) -> TurnProfile:
        """The profile whose thresholds were calibrated for `scorer`."""
        return self.profiles[_scorer_kind(scorer)]

    @property
    def delay_in_frames(self) -> int:
        return self.profile.delay_in_frames

    def setup_key(self) -> tuple[Any, ...]:
        """The options sent in the setup message: changing one needs a new connection."""
        return (
            self.language,
            self.keywords,
            self.keyword_boost,
            self.temperature,
            self.log_level,
            self.delay_in_frames,
        )

    def updated(
        self,
        *,
        eagerness: NotGivenOr[Eagerness],
        overrides: NotGivenOr[TurnOverrides],
        weights: NotGivenOr[str | Path | None],
        delay_in_frames: NotGivenOr[int],
        language: NotGivenOr[str],
        keywords: NotGivenOr[list[str] | None],
        keyword_boost: NotGivenOr[float | None],
        temperature: NotGivenOr[float | None],
        log_level: NotGivenOr[LogLevel | None],
    ) -> _Settings:
        merged: TurnOverrides = {**self.overrides, **(overrides if is_given(overrides) else {})}
        if is_given(weights):
            merged["weights"] = weights
        if is_given(delay_in_frames):
            merged["delay_in_frames"] = delay_in_frames
        return _Settings(
            eagerness=eagerness if is_given(eagerness) else self.eagerness,
            overrides=merged,
            language=language if is_given(language) else self.language,
            keywords=tuple(keywords or ()) if is_given(keywords) else self.keywords,
            keyword_boost=keyword_boost if is_given(keyword_boost) else self.keyword_boost,
            temperature=temperature if is_given(temperature) else self.temperature,
            log_level=log_level if is_given(log_level) else self.log_level,
        )


def _scorer_kind(scorer: Scorer) -> ScorerKind:
    return "learned" if isinstance(scorer, LearnedScorer) else "fallback"


def _stt_url(gradium_api: str) -> str:
    """The streaming STT websocket URL on the API host `gradium_api`."""
    if not gradium_api or "/" in gradium_api:
        raise ValueError(
            f"gradium_api must be a host such as {DEFAULT_GRADIUM_API!r}, without scheme "
            f"or path, got {gradium_api!r}"
        )
    return f"wss://{gradium_api}{_STT_PATH}"


class STTExperimental(stt.STT):
    def __init__(
        self,
        *,
        eagerness: Eagerness = "balanced",
        overrides: TurnOverrides | None = None,
        weights: NotGivenOr[str | Path | None] = NOT_GIVEN,
        language: str = "en",
        keywords: list[str] | None = None,
        keyword_boost: float | None = None,
        temperature: float | None = None,
        delay_in_frames: NotGivenOr[int] = NOT_GIVEN,
        log_level: LogLevel | None = None,
        api_key: str | None = None,
        gradium_api: str | None = None,
        model_name: str = "default",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Gradium STT whose turn detection is a client-side state machine.

        Use with `turn_detection="stt"`. The turn ends on Gradium's inactivity
        probabilities fused by a scorer; a likely end first sends a `flush` and emits a
        `PREFLIGHT_TRANSCRIPT` so the session can start the LLM early.

        Args:
            eagerness: How much false cut-off risk buys latency: "responsive" (10%
                budget), "balanced" (5%) or "prudent" (2%). In background noise, pick
                one level stricter than in quiet.
            overrides: Turn knobs to set on top of the preset; see `TurnProfile`. They
                apply whichever scorer runs (learned, or the fallback).
            weights: Name or path of the scorer weight set; None forces the
                `min(p2, p3)` fallback. Defaults to the preset's.
            language: Language hint, sent to the server as given; "any" transcribes
                any language. The server rejects a hint it does not know, which fails
                the stream.
            keywords: Words or phrases to favor in the transcript.
            keyword_boost: How strongly to favor the keywords; the server defaults to 2.
            temperature: Decoding temperature; the server default when None.
            delay_in_frames: Model delay in 80 ms steps; defaults to the preset's. The
                server rejects a delay it does not serve, which fails the stream.
            log_level: Lowest level of the server's log messages to receive and log;
                None asks for none.
            api_key: Defaults to the `GRADIUM_API_KEY` environment variable.
            gradium_api: Host of the Gradium API, without scheme or path. Defaults to
                the `GRADIUM_API` environment variable, then `api.gradium.ai`. The
                plugin connects to `wss://<gradium_api>/api/speech/asr`; see
                https://docs.gradium.ai/api-reference/endpoint/stt-websocket.
            model_name: Server model name sent in the setup message.
            http_session: Session used for the websocket; the job's by default.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
                manual_flush=True,
            )
        )
        api_key = api_key or os.environ.get("GRADIUM_API_KEY")
        if not api_key:
            raise ValueError("Gradium API key is required: pass `api_key` or set `GRADIUM_API_KEY`")
        self._api_key = api_key
        self._url = _stt_url(gradium_api or os.environ.get("GRADIUM_API") or DEFAULT_GRADIUM_API)
        self._model_name = model_name
        self._settings = _Settings(
            eagerness=eagerness,
            overrides={},
            language=language,
            keywords=(),
            keyword_boost=None,
            temperature=None,
            log_level=None,
        ).updated(
            eagerness=NOT_GIVEN,
            overrides=overrides or {},
            weights=weights,
            delay_in_frames=delay_in_frames,
            language=NOT_GIVEN,
            keywords=keywords,
            keyword_boost=keyword_boost,
            temperature=temperature,
            log_level=log_level,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "Gradium"

    @property
    def profile(self) -> TurnProfile:
        """The turn knobs for the requested scorer.

        A stream switches to the fallback scorer's profile when the weights cannot be
        used (missing file, or trained for another server model); see `SpeechStream`.
        """
        return self._settings.profile

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("STTExperimental only supports streaming")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        settings = self._settings
        if is_given(language):
            settings = dataclasses.replace(settings, language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            settings=settings,
            api_key=self._api_key,
            url=self._url,
            model_name=self._model_name,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        eagerness: NotGivenOr[Eagerness] = NOT_GIVEN,
        overrides: NotGivenOr[TurnOverrides] = NOT_GIVEN,
        weights: NotGivenOr[str | Path | None] = NOT_GIVEN,
        delay_in_frames: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[str] | None] = NOT_GIVEN,
        keyword_boost: NotGivenOr[float | None] = NOT_GIVEN,
        temperature: NotGivenOr[float | None] = NOT_GIVEN,
        log_level: NotGivenOr[LogLevel | None] = NOT_GIVEN,
    ) -> None:
        """Update new and running streams.

        Turn knobs apply from the next step without reconnecting; `overrides` merge into
        the ones already set. `weights` reloads the scorer in place. Options sent in the
        setup message (`language`, `keywords`, `keyword_boost`, `temperature`,
        `delay_in_frames`, `log_level`) reconnect each running stream once the audio
        already queued has been sent, so no audio is lost.

        Raises:
            ValueError: The resulting knobs are invalid; nothing is changed.
        """
        kwargs: dict[str, Any] = {
            "eagerness": eagerness,
            "overrides": overrides,
            "weights": weights,
            "delay_in_frames": delay_in_frames,
            "language": language,
            "keywords": keywords,
            "keyword_boost": keyword_boost,
            "temperature": temperature,
            "log_level": log_level,
        }
        self._settings = self._settings.updated(**kwargs)
        for stream in self._streams:
            stream.update_options(**kwargs)


@dataclass
class _Reconnect(stt.SpeechStream._FlushSentinel):
    """Queued behind pending audio: end the connection and open one with new options."""

    generation: int


@dataclass
class _Connection:
    generation: int
    closing: bool = False
    """The client sent end_of_stream: a close or the server's end_of_stream is expected."""
    reconnect: bool = False


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STTExperimental,
        conn_options: APIConnectOptions,
        settings: _Settings,
        api_key: str,
        url: str,
        model_name: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._settings = settings
        self._api_key = api_key
        self._url = url
        self._model_name = model_name
        self._session = http_session
        # until `ready` names the server model, the fallback scorer and its thresholds
        fallback = FallbackScorer()
        self._machine = TurnMachine(settings.profile_for(fallback), fallback)
        self._request_id = ""
        self._server_model: str | None = None
        self._scorer_key: tuple[str | Path | None, str | None] = (None, None)
        self._scorer_task: asyncio.Task[None] | None = None
        # bumped by each update that needs a new connection
        self._generation = 0
        self._malformed_logged = False

    @property
    def machine(self) -> TurnMachine:
        """The turn state machine, for debugging: current state and last score."""
        return self._machine

    def update_options(
        self,
        *,
        eagerness: NotGivenOr[Eagerness] = NOT_GIVEN,
        overrides: NotGivenOr[TurnOverrides] = NOT_GIVEN,
        weights: NotGivenOr[str | Path | None] = NOT_GIVEN,
        delay_in_frames: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[str] | None] = NOT_GIVEN,
        keyword_boost: NotGivenOr[float | None] = NOT_GIVEN,
        temperature: NotGivenOr[float | None] = NOT_GIVEN,
        log_level: NotGivenOr[LogLevel | None] = NOT_GIVEN,
    ) -> None:
        """Same as `STTExperimental.update_options`, for this stream only."""
        old = self._settings
        new = old.updated(
            eagerness=eagerness,
            overrides=overrides,
            weights=weights,
            delay_in_frames=delay_in_frames,
            language=language,
            keywords=keywords,
            keyword_boost=keyword_boost,
            temperature=temperature,
            log_level=log_level,
        )
        self._settings = new
        self._machine.profile = new.profile_for(self._machine.scorer)
        if new.weights != old.weights and self._server_model is not None:
            if self._scorer_task is not None:
                self._scorer_task.cancel()
            self._scorer_task = asyncio.create_task(self._install_scorer(self._server_model))
        if new.setup_key() != old.setup_key():
            self._generation += 1
            with contextlib.suppress(utils.aio.ChanClosed):
                # behind the audio already queued, which the current connection sends
                self._input_ch.send_nowait(_Reconnect(self._generation))

    async def aclose(self) -> None:
        if self._scorer_task is not None:
            await utils.aio.cancel_and_wait(self._scorer_task)
        await super().aclose()

    async def _install_scorer(self, model: str | None) -> None:
        """Load the scorer for the current weights and `model`, unless already loaded.

        The machine's thresholds follow the scorer that actually loaded: when the
        weights are unusable, the fallback runs with the fallback's calibrated profile.
        """
        weights = self._settings.weights
        key = (weights, model)
        if key == self._scorer_key:
            return
        scorer: Scorer = (
            FallbackScorer()
            if weights is None
            # reads the weights file: keep it off the event loop
            else await asyncio.to_thread(load_scorer, weights, model=model)
        )
        if weights != self._settings.weights:
            return  # superseded by a newer update while loading
        self._scorer_key = key
        self._machine.set_scorer(scorer)
        self._machine.profile = self._settings.profile_for(scorer)
        logger.debug("Gradium scorer %s for model %s", scorer.name, model)

    # connection

    async def _main_task(self) -> None:
        try:
            await super()._main_task()
        except Exception:
            # A retry can inherit the open turn, but a replacement stream cannot.
            self._emit(self._machine.finalize())
            raise

    async def _run(self) -> None:
        while True:
            conn = _Connection(generation=self._generation)
            ws = await self._connect_ws()
            self._input_ch.restart_provider_clock()
            ended = False
            try:
                await self._run_connection(ws, conn)
                ended = not conn.reconnect
            finally:
                # Whatever ended the connection, no received word stays buffered. Only the
                # end of input ends the user's turn: after a reconnect or a failure the
                # base class retries, and the turn continues on the next connection.
                self._emit(self._machine.finalize(keep_turn=not ended))
                with contextlib.suppress(Exception):
                    await ws.close()
            if not conn.reconnect:
                return

    async def _run_connection(self, ws: aiohttp.ClientWebSocketResponse, conn: _Connection) -> None:
        send_task = asyncio.create_task(self._send_loop(ws, conn))
        recv_task = asyncio.create_task(self._recv_loop(ws, conn))
        try:
            done, _ = await asyncio.wait(
                (send_task, recv_task), return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                recv_task.result()
                return
            send_task.result()
            # Sending stopped: end of input, a reconnect, or a failed write. The server
            # answers end_of_stream with its last words; after a failed write, the close
            # that follows carries the real reason.
            done, _ = await asyncio.wait((recv_task,), timeout=_DRAIN_TIMEOUT)
            if recv_task in done:
                recv_task.result()
            elif conn.closing:
                logger.warning("Gradium did not end the session in time, closing it")
            else:
                raise APIConnectionError("Gradium connection stalled after a failed write")
        finally:
            await utils.aio.gracefully_cancel(send_task, recv_task)

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        settings = self._settings
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._url, headers=ws_headers(self._api_key), autoping=False
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("failed to connect to Gradium") from None
        except aiohttp.ClientResponseError as e:
            # the request info holds the API key header: do not chain or format it
            raise APIStatusError(
                f"failed to connect to Gradium: {e.message}", status_code=e.status, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(f"failed to connect to Gradium ({type(e).__name__})") from None

        json_config: dict[str, Any] = {
            "delay_in_frames": settings.delay_in_frames,
            "language": settings.language,
        }
        if settings.temperature is not None:
            json_config["temp"] = settings.temperature
        if settings.keywords:
            keywords: dict[str, Any] = {"words": list(settings.keywords)}
            if settings.keyword_boost is not None:
                keywords["boost"] = settings.keyword_boost
            json_config["keywords"] = keywords
        setup: dict[str, Any] = {
            "type": "setup",
            "model_name": self._model_name,
            "input_format": "pcm",
            "json_config": json_config,
        }
        if settings.log_level is not None:
            setup["log_level"] = settings.log_level
        if not await _send(ws, setup):
            with contextlib.suppress(Exception):
                await ws.close()
            raise APIConnectionError("Gradium closed the connection during setup")
        return ws

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse, conn: _Connection) -> None:
        bstream = utils.audio.AudioByteStream(
            sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=SAMPLES_PER_FRAME
        )
        async for item in self._input_ch:
            if isinstance(item, _Reconnect):
                if item.generation <= conn.generation:
                    continue  # this connection already carries those options
                conn.reconnect = True
                break
            if isinstance(item, self._FlushSentinel):
                if not await _send_audio(ws, bstream.flush()):
                    return
                if self._input_ch.closed and self._input_ch.empty():
                    # the flush `end_input()` queues before closing: end_of_stream below
                    # returns the last words, and nobody waits for a manual answer
                    continue
                await self._apply(ws, self._machine.request_flush())
                continue
            if not await _send_audio(ws, bstream.write(item.data.tobytes())):
                return
        # end of input, or a reconnect: send what is buffered and ask the server to
        # return its last words
        if not await _send_audio(ws, bstream.flush()):
            return
        conn.closing = True
        await _send(ws, {"type": "end_of_stream"})

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse, conn: _Connection) -> None:
        monitor = utils.aio.WebSocketMonitor(
            ws, liveness_timeout=self._conn_options.liveness_timeout
        )
        while True:
            msg = await monitor.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                if conn.closing:
                    return
                raise _close_error(
                    "Gradium connection closed unexpectedly",
                    code=ws.close_code,
                    body=f"{msg.data=} {msg.extra=}",
                    reason=msg.extra,
                )
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("Gradium connection lost") from ws.exception()
            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.warning("unexpected Gradium message type %s", msg.type)
                continue

            try:
                data = json.loads(msg.data)
                if not isinstance(data, dict):
                    raise ValueError("not a JSON object")
                sig = signals.parse(data)
            except ValueError as e:  # json.JSONDecodeError is a ValueError
                self._log_malformed(e)
                continue
            if sig is None:
                continue

            if isinstance(sig, signals.ServerLog):
                log_server_message(data)
                continue
            if isinstance(sig, signals.Ready):
                self._request_id = sig.request_id
                self._server_model = sig.model
                await self._install_scorer(sig.model)

            if not isinstance(sig, (signals.EndOfStream, signals.ServerError)):
                # the connection's end is handled once, by `_run`, which knows whether the
                # user's turn continues on another connection
                await self._apply(ws, self._machine.on_signal(sig))

            if isinstance(sig, (signals.Step, signals.Word, signals.WordEnd, signals.Flushed)):
                self._mark_provider_progress()
            elif isinstance(sig, signals.EndOfStream):
                if conn.closing:
                    return
                # A duration limit, for instance. Raise so the base class reconnects.
                raise APIStatusError(
                    "Gradium ended the session while input was still open",
                    status_code=ws.close_code or -1,
                    body=data,
                )
            elif isinstance(sig, signals.ServerError):
                raise _close_error(
                    f"Gradium returned error: {sig.message}", code=sig.code, body=sig.raw
                )

    def _log_malformed(self, error: ValueError) -> None:
        # one warning per stream: a server bug would otherwise log 12 times a second
        if not self._malformed_logged:
            self._malformed_logged = True
            logger.warning("skipping malformed Gradium message: %s", error)
        else:
            logger.debug("skipping malformed Gradium message: %s", error)

    # actions

    async def _apply(self, ws: aiohttp.ClientWebSocketResponse, actions: Sequence[Action]) -> None:
        for action in actions:
            if isinstance(action, SendFlush):
                # a failed write surfaces as the close the receive loop reads next
                await _send(ws, {"type": "flush", "flush_id": action.flush_id})
            else:
                self._emit((action,))

    def _emit(self, actions: Sequence[Action]) -> None:
        offset = self.start_time_offset
        hint = self._settings.language
        language = LanguageCode("" if hint == ANY_LANGUAGE else hint)
        for action in actions:
            if isinstance(action, StartOfSpeech):
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH, request_id=self._request_id
                )
            elif isinstance(action, SendFlush):
                raise AssertionError("finalize must not send a flush")
            else:
                event = stt.SpeechEvent(
                    type=_EVENT_TYPES[type(action)],
                    request_id=self._request_id,
                    alternatives=[
                        stt.SpeechData(
                            language=language,
                            text=action.text,
                            start_time=action.start_time + offset,
                            end_time=action.end_time + offset,
                        )
                    ],
                )
            with contextlib.suppress(utils.aio.ChanClosed):
                self._event_ch.send_nowait(event)


_EVENT_TYPES: dict[type[Action], stt.SpeechEventType] = {
    Interim: stt.SpeechEventType.INTERIM_TRANSCRIPT,
    Preflight: stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
    Final: stt.SpeechEventType.FINAL_TRANSCRIPT,
    EndOfSpeech: stt.SpeechEventType.END_OF_SPEECH,
}


async def _send(ws: aiohttp.ClientWebSocketResponse, payload: dict[str, Any]) -> bool:
    """Send one JSON message; False when the socket is already closing."""
    try:
        await ws.send_str(json.dumps(payload))
    except (aiohttp.ClientError, ConnectionError):
        return False
    return True


async def _send_audio(ws: aiohttp.ClientWebSocketResponse, frames: list[rtc.AudioFrame]) -> bool:
    for frame in frames:
        audio = base64.b64encode(frame.data).decode("ascii")
        if not await _send(ws, {"type": "audio", "audio": audio}):
            return False
    return True
