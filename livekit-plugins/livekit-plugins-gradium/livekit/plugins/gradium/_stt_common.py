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

"""What `STT` and `STTExperimental` share: Gradium's STT websocket protocol and the STT plumbing.

Both classes speak the same protocol and differ only in how they end a turn, so the
protocol lives here once: a fix to it applies to both.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import aiohttp

from livekit.agents import APIConnectOptions, APIStatusError, LanguageCode, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from .api import ws_headers
from .log import logger
from .models import LogLevel
from .tts import _close_error

SUPPORTED_SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
"""Gradium processes audio in frames of this many samples."""
FRAME_S = FRAME_SAMPLES / SUPPORTED_SAMPLE_RATE
"""80 ms: the duration of a frame, of a model-delay unit and of a `step`."""
DEFAULT_DELAY_IN_FRAMES = 10
"""Gradium's default model delay, used until the server's ready message arrives."""
DEFAULT_ENDPOINT = "wss://api.gradium.ai/api/speech/asr"


class GradiumSTTBase(stt.STT):
    """Credentials, endpoint, HTTP session and identity common to Gradium's STTs."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model_endpoint: str | None,
        model_name: str,
        http_session: aiohttp.ClientSession | None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,  # only chunk start times are available
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
            model_endpoint or os.environ.get("GRADIUM_MODEL_ENDPOINT") or DEFAULT_ENDPOINT
        )
        self._model_name = model_name
        self._session = http_session

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


async def connect(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    api_key: str,
    model_name: str,
    language: LanguageCode,
    temperature: float | None,
    keywords: list[str] | None,
    keyword_boost: float | None,
    log_level: LogLevel | None,
    delay_in_frames: int | None = None,
) -> aiohttp.ClientWebSocketResponse:
    """Open the websocket and send the setup message, which must come first."""
    ws = await session.ws_connect(endpoint, headers=ws_headers(api_key), autoping=False)
    json_config: dict[str, Any] = {"language": language.language}
    if delay_in_frames is not None:
        json_config["delay_in_frames"] = delay_in_frames
    if temperature is not None:
        json_config["temp"] = temperature
    if keywords:
        keywords_config: dict[str, Any] = {"words": keywords}
        if keyword_boost is not None:
            keywords_config["boost"] = keyword_boost
        json_config["keywords"] = keywords_config
    setup: dict[str, Any] = {
        "type": "setup",
        "model_name": model_name,
        "input_format": "pcm",
        "json_config": json_config,
    }
    if log_level is not None:
        setup["log_level"] = log_level
    await ws.send_str(json.dumps(setup))
    return ws


class SocketClosed(Exception):
    """The websocket closed; `error` is what to raise when that was not expected."""

    def __init__(self, error: APIStatusError) -> None:
        super().__init__(str(error))
        self.error = error


async def receive_message(
    ws: aiohttp.ClientWebSocketResponse,
    monitor: utils.aio.WebSocketMonitor,
    *,
    timeout: float,
) -> dict[str, Any] | None:
    """The next JSON object Gradium sent, or None when none came within `timeout`.

    Frames that are not a JSON object are logged and skipped.

    Raises:
        SocketClosed: the socket closed.
        APIStatusError: Gradium sent an `error`; it closes the socket right after.
    """
    while True:
        try:
            msg = await asyncio.wait_for(monitor.receive(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        ):
            raise SocketClosed(
                APIStatusError(
                    "Gradium connection closed unexpectedly",
                    status_code=ws.close_code or -1,
                    body=f"{msg.data=} {msg.extra=}",
                )
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
        if data.get("type") == "error":
            raise api_error(data)
        return data


def api_error(msg_data: dict[str, Any]) -> APIStatusError:
    """Build the error for a `{"type": "error"}` frame.

    The server sends it right before closing the socket, so the message is the only
    place the actual failure reason (unsupported language, no credits, ...) shows up.
    Its `code` is the websocket close code that follows; client errors are not retried.
    """
    message = msg_data.get("message") or "unknown error"
    return _close_error(
        f"Gradium returned error: {message}", code=msg_data.get("code"), body=msg_data
    )


def session_ended_error(ws: aiohttp.ClientWebSocketResponse, data: Any) -> APIStatusError:
    """The server sent `end_of_stream` while audio was still coming (a duration limit,
    for instance) and will close the socket: a retryable error, so the base class
    reconnects instead of the next audio write failing with a raw connection error."""
    return APIStatusError(
        "Gradium ended the session while input was still open",
        status_code=ws.close_code or -1,
        body=data,
    )
