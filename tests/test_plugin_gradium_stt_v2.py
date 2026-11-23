"""Tests for the end of turn of Gradium's STTV2: a local VAD plus Gradium's semantic VAD.

Most tests drive `_TurnTracker`, the synchronous state machine, with a scripted timeline of
local VAD windows and Gradium messages; the last ones go through the real stream with a
scripted VAD and websocket.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.agents.vad import VAD, VADCapabilities, VADEvent, VADEventType, VADStream
from livekit.plugins.gradium import STTV2
from livekit.plugins.gradium.stt_v2 import _TurnTracker

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

SAMPLE_RATE = 24000
VAD_WINDOW_S = 0.032  # Silero's window at 16 kHz
STEP_S = 0.08
SPEECH = 0.9
SILENCE = 0.05
# Gradium probabilities, one per horizon (0.5, 1, 2, 3 s)
ENDED = [0.99, 0.99, 0.99, 0.99]
UNSURE = [0.6, 0.4, 0.2, 0.1]


class _ScriptedVAD(VAD):
    """A local VAD whose speech probability for every window is `prob`, set by the test."""

    def __init__(self) -> None:
        super().__init__(capabilities=VADCapabilities(update_interval=VAD_WINDOW_S))
        self.prob = SILENCE

    def stream(self) -> VADStream:
        return _ScriptedVADStream(self)


class _ScriptedVADStream(VADStream):
    def __init__(self, vad: _ScriptedVAD) -> None:
        self._scripted = vad
        super().__init__(vad)

    async def _main_task(self) -> None:
        t = 0.0
        async for frame in self._input_ch:
            if not isinstance(frame, rtc.AudioFrame):
                continue
            t += frame.samples_per_channel / frame.sample_rate
            self._event_ch.send_nowait(
                VADEvent(
                    type=VADEventType.INFERENCE_DONE,
                    samples_index=0,
                    timestamp=t,
                    speech_duration=0.0,
                    silence_duration=0.0,
                    probability=self._scripted.prob,
                )
            )


def _stt(**overrides: Any) -> STTV2:
    return STTV2(vad=_ScriptedVAD(), api_key="test-key", **overrides)


class _Timeline:
    """A `_TurnTracker` with the defaults of `STTV2`, fed a scripted conversation.

    `t` is the audio time; each helper advances it. Gradium steps come every 80 ms during
    `silence`, with the given probabilities.
    """

    def __init__(self, **overrides: Any) -> None:
        self.events: list[SpeechEvent] = []
        self.flushes: list[int] = []
        self.t = 0.0
        self.tracker = _TurnTracker(
            _stt(**overrides)._opts, emit=self.events.append, request_flush=self._flush
        )

    def _flush(self) -> int:
        self.flushes.append(len(self.flushes) + 1)
        return self.flushes[-1]

    def speech(self, duration: float, prob: float = SPEECH) -> None:
        self._advance(duration, prob, steps=None)

    def silence(self, duration: float, steps: list[float] | None = None) -> None:
        self._advance(duration, SILENCE, steps)

    def _advance(self, duration: float, prob: float, steps: list[float] | None) -> None:
        end = self.t + duration
        next_step = self.t + STEP_S
        while self.t + VAD_WINDOW_S <= end + 1e-9:
            self.t += VAD_WINDOW_S
            self.tracker.on_vad(self.t, prob, wall=self.t)
            if steps is not None and self.t >= next_step:
                self.tracker.on_step(self.t, steps)
                next_step += STEP_S

    def word(self, text: str, stop_s: float | None = None) -> None:
        self.tracker.on_text(text)
        self.tracker.on_end_text(self.t if stop_s is None else stop_s)

    def reply(self, flush_id: int | None = None) -> None:
        self.tracker.on_flushed(self.flushes[-1] if flush_id is None else flush_id)

    def types(self, *types: SpeechEventType) -> list[SpeechEventType]:
        return [e.type for e in self.events if not types or e.type in types]

    def texts(self, type_: SpeechEventType) -> list[str]:
        return [e.alternatives[0].text for e in self.events if e.type == type_]

    def finals(self) -> list[str]:
        return self.texts(SpeechEventType.FINAL_TRANSCRIPT)


def _spoken(**overrides: Any) -> _Timeline:
    """A timeline where the user said "hello there" and just went quiet."""
    tl = _Timeline(**overrides)
    tl.speech(0.5)
    tl.word("hello")
    tl.word("there")
    return tl


def test_turn_ends_on_the_flush_reply_once_gradium_is_confident():
    # The session must get the final transcript with every word the model delay held, so the
    # turn ends on the flush reply, not on the decision; and it must not wait past a short
    # pause when Gradium is sure the user is done.
    tl = _spoken()
    tl.silence(0.2, steps=ENDED)
    assert len(tl.flushes) == 1
    assert SpeechEventType.END_OF_SPEECH not in tl.types()

    tl.word("friend")  # held by the model delay, sent before the reply
    tl.reply()
    assert tl.finals() == ["hello there friend"]
    assert tl.types()[-2:] == [SpeechEventType.FINAL_TRANSCRIPT, SpeechEventType.END_OF_SPEECH]


def test_final_end_time_is_unknown_until_the_last_word_ends():
    # The session takes the final's `end_time` as when the user stopped speaking; the end
    # of an earlier word would place it too early.
    tl = _spoken()
    tl.tracker.on_text("friend")  # its end_text not in yet
    tl.silence(0.2, steps=ENDED)
    tl.reply()
    final = next(e for e in tl.events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    assert final.alternatives[0].end_time == 0.0


def test_late_word_end_does_not_leak_into_the_next_turn():
    # An `end_text` that arrives after its turn ended belongs to that turn, not the next.
    tl = _spoken()
    tl.silence(0.2, steps=ENDED)
    tl.reply()
    tl.tracker.on_end_text(0.6)
    tl.speech(0.5)
    tl.tracker.on_text("next")
    tl.silence(0.2, steps=ENDED)
    tl.reply()
    finals = [e for e in tl.events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert finals[-1].alternatives[0].end_time == 0.0


def test_unsure_gradium_holds_the_turn_until_max_silence():
    # A mid-sentence pause (Gradium unsure) must not end the turn; `max_silence` caps how
    # long the user can be kept waiting.
    tl = _spoken(max_silence=1.5)
    tl.silence(1.4, steps=UNSURE)
    assert tl.flushes == []

    tl.silence(0.2, steps=UNSURE)
    assert len(tl.flushes) == 1
    tl.reply()
    assert tl.finals() == ["hello there"]


def test_commit_threshold_falls_as_the_pause_lasts():
    # The threshold slides from `eot_threshold_start` to `eot_threshold_end`: a probability
    # too low to end the turn early ends it later in the pause.
    tl = _spoken(min_silence=0.1, max_silence=2.1, eot_threshold_start=0.9, eot_threshold_end=0.1)
    probs = [0.0, 0.0, 0.0, 0.5]
    tl.silence(0.9, steps=probs)  # threshold still above 0.5
    assert tl.flushes == []
    tl.silence(0.4, steps=probs)  # threshold reaches 0.5 at 1.1 s of pause
    assert len(tl.flushes) == 1


def test_user_resuming_before_the_reply_keeps_the_turn_open():
    # Speech heard by the local VAD between the decision and the flush reply means the
    # pause was mid-sentence: the reply must not end the turn, and the next pause flushes
    # again rather than reuse the stale reply.
    tl = _spoken()
    tl.silence(0.2, steps=ENDED)
    tl.speech(0.2)
    tl.reply(1)
    assert SpeechEventType.FINAL_TRANSCRIPT not in tl.types()

    tl.word("friend")
    tl.silence(0.2, steps=ENDED)
    assert tl.flushes == [1, 2]
    tl.reply(2)
    assert tl.finals() == ["hello there friend"]


def test_reply_during_resumed_speech_not_yet_confirmed_does_not_end_the_turn():
    # Speech confirmation takes `min_speech_duration`: a flush reply that lands inside it
    # must wait for the verdict instead of committing on a pause that is over.
    tl = _spoken()
    tl.silence(0.2, steps=ENDED)
    tl.speech(0.064)  # resuming, not confirmed yet
    tl.reply()
    assert SpeechEventType.FINAL_TRANSCRIPT not in tl.types()

    tl.speech(0.15)
    tl.word("friend")
    tl.silence(0.2, steps=ENDED)
    tl.reply()
    assert tl.finals() == ["hello there friend"]


def test_short_noise_and_hovering_probability_do_not_end_a_pause():
    # Hysteresis: a click shorter than `min_speech_duration` does not resume the turn, and a
    # probability between the two thresholds does not flap the speech state.
    tl = _spoken()
    tl.silence(0.2, steps=ENDED)
    tl.speech(0.064)  # shorter than min_speech_duration
    tl.speech(0.2, prob=0.4)  # between silence_threshold and speech_threshold
    tl.reply()
    assert tl.finals() == ["hello there"]


def test_word_spoken_during_the_pause_resumes_the_turn():
    # If the local VAD misses quiet speech, a word that ended well after the pause started
    # still proves the user is talking: the pending end is dropped.
    tl = _spoken()
    pause_start = tl.t
    tl.silence(0.5, steps=ENDED)
    tl.word("again", stop_s=pause_start + 0.4)
    tl.reply()
    assert SpeechEventType.FINAL_TRANSCRIPT not in tl.types()


def test_last_word_ending_just_after_the_vad_pause_does_not_resume():
    # The local VAD places a pause up to ~0.1 s after the last word really ended, and
    # Gradium's `stop_s` is ~0.05 s off: the turn's own last word must not count as new
    # speech, or every turn would wait for a second pause.
    tl = _spoken()
    pause_start = tl.t
    tl.silence(0.2, steps=ENDED)
    tl.word("friend", stop_s=pause_start + 0.15)
    tl.reply()
    assert tl.finals() == ["hello there friend"]


def test_eager_flush_previews_the_words_and_a_resume_supersedes_the_preview():
    # The eager stage lets the session start the LLM before the turn ends; when the user
    # resumes, an interim after the preview is what tells the session the preview is stale.
    tl = _spoken(eot_threshold_start=0.95, eot_threshold_end=0.95, eager_threshold=0.5)
    tl.silence(0.2, steps=[0.0, 0.0, 0.6, 0.0])
    tl.reply()
    assert tl.texts(SpeechEventType.PREFLIGHT_TRANSCRIPT) == ["hello there"]
    assert SpeechEventType.FINAL_TRANSCRIPT not in tl.types()

    tl.speech(0.2)
    assert tl.types()[-1] == SpeechEventType.INTERIM_TRANSCRIPT


def test_words_after_the_flush_reply_resume_the_turn():
    # The reply carries every word spoken before the flush, so a later word is new speech.
    tl = _spoken(eot_threshold_start=0.95, eot_threshold_end=0.95, eager_threshold=0.5)
    tl.silence(0.2, steps=[0.0, 0.0, 0.6, 0.0])
    tl.reply()
    tl.word("and", stop_s=tl.t)
    tl.silence(2.5, steps=[0.0, 0.0, 0.6, 0.0])
    assert tl.flushes == [1, 2]


def test_steps_from_before_the_pause_do_not_end_the_turn():
    # A step describes the audio up to its time: one from before the pause, still in flight
    # when the pause starts, says nothing about the pause.
    tl = _Timeline()
    tl.speech(0.5)
    tl.word("hello")
    tl.tracker.on_step(0.4, ENDED)
    tl.silence(0.2)  # no new step
    assert tl.flushes == []


def test_noise_heard_as_speech_without_words_ends_the_turn():
    # Background noise that the local VAD calls speech would hold a turn open forever;
    # with no word from Gradium long past its delay, the turn ends anyway.
    tl = _Timeline(max_silence=1.0)
    tl.speech(1.0 + tl.tracker.word_lag_s + 0.1)
    assert len(tl.flushes) == 1
    tl.reply()
    assert tl.types(SpeechEventType.END_OF_SPEECH) == [SpeechEventType.END_OF_SPEECH]


def test_noise_guard_does_not_count_the_pause_before_resumed_speech():
    # After a long hesitation, the words of the resumed speech arrive a model delay after
    # it starts; the guard must wait for them from the resumption, not from the last word
    # before the pause, or it cuts the user's sentence in half.
    tl = _spoken(max_silence=2.5)
    tl.silence(2.2, steps=UNSURE)
    tl.speech(2.0)  # no word transcribed yet
    assert tl.flushes == []


def test_noise_guard_drops_its_decision_once_words_arrive():
    # The guard's flush often brings back the very words it was waiting for: the user is
    # talking, so neither a preview nor the next dip between two words may end the turn.
    tl = _Timeline(max_silence=1.0)
    tl.speech(1.0 + tl.tracker.word_lag_s + 0.1)
    assert len(tl.flushes) == 1

    tl.word("I", stop_s=tl.t)
    tl.reply()
    tl.speech(0.032, prob=0.1)  # a stop consonant
    assert (
        tl.types(
            SpeechEventType.PREFLIGHT_TRANSCRIPT,
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
        )
        == []
    )


def test_turn_opened_by_a_word_the_vad_missed_does_not_end_at_once():
    # When the local VAD heard nothing, the word's arrival is the only evidence of speech:
    # the pause is measured from it, not from the last speech the VAD heard.
    tl = _Timeline(max_silence=1.0)
    tl.silence(3.0)
    tl.word("psst", stop_s=0.0)  # its end not known yet: stop_s from an old clock
    tl.silence(0.5)
    assert tl.flushes == []


@pytest.mark.parametrize(
    "options",
    [
        {"silence_threshold": 0.6, "speech_threshold": 0.5},
        {"min_silence": 1.0, "max_silence": 1.0},
        {"eot_bucket": 4},
        {"eot_threshold_start": 1.5},
        {"delay_in_frames": 9},
        {"preflight_trigger": "vad_trip"},
    ],
)
def test_inconsistent_options_are_rejected(options):
    # A typo or an impossible combination must fail at construction, not silently change
    # when turns end.
    with pytest.raises(ValueError):
        _stt(**options)


class _QueueWebSocket:
    """Server side driven by the test: `server()` queues a message for the client."""

    def __init__(self, *, auto_eos: bool = True) -> None:
        self.sent: list[dict[str, Any]] = []
        self.close_code: int | None = None
        self._auto_eos = auto_eos
        self._inbox: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

    def server(self, msg: dict[str, Any]) -> None:
        self._inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(msg), ""))

    def server_close(self) -> None:
        self._inbox.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1011, ""))

    async def send_str(self, data: str) -> None:
        msg = json.loads(data)
        self.sent.append(msg)
        if msg["type"] == "end_of_stream" and self._auto_eos:
            self.server({"type": "end_of_stream"})

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        return await self._inbox.get()

    async def ping(self) -> None:
        pass

    async def close(self) -> bool:
        return True

    def sent_of(self, type_: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m["type"] == type_]


class _Session:
    """Hands out one scripted socket per connection."""

    def __init__(self, sockets: list[_QueueWebSocket]) -> None:
        self._sockets = sockets
        self.connects = 0

    async def ws_connect(self, url: str, **kwargs: Any) -> _QueueWebSocket:
        self.connects += 1
        return self._sockets[self.connects - 1]


async def _until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def _step(total_duration_s: float, prob: float) -> dict[str, Any]:
    return {
        "type": "step",
        "total_duration_s": total_duration_s,
        "vad": [{"horizon_s": h, "inactivity_prob": prob} for h in (0.5, 1, 2, 3)],
    }


class _Stream:
    """A real `SpeechStreamV2` over scripted sockets (one per connection) and a scripted VAD.

    `ws` is the socket of the current connection; `speak` and `quiet` push real-time audio
    the VAD calls speech or silence.
    """

    def __init__(
        self,
        *,
        connections: int = 1,
        max_retry: int = 0,
        auto_eos: bool = True,
        **overrides: Any,
    ) -> None:
        self.sockets = [_QueueWebSocket(auto_eos=auto_eos) for _ in range(connections)]
        self.session = _Session(self.sockets)
        self.vad = _ScriptedVAD()
        stt = STTV2(vad=self.vad, api_key="test-key", http_session=self.session, **overrides)  # type: ignore[arg-type]
        self.stream = stt.stream(
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.1, timeout=5)
        )
        self.events: list[SpeechEvent] = []
        self.error: BaseException | None = None
        self._task = asyncio.create_task(self._collect())

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        with contextlib.suppress(RuntimeError):  # the input may be closed already
            self.stream.end_input()
        await asyncio.wait_for(self._task, 5)
        await self.stream.aclose()

    async def _collect(self) -> None:
        try:
            async for ev in self.stream:
                self.events.append(ev)
        except APIStatusError as e:
            self.error = e

    @property
    def ws(self) -> _QueueWebSocket:
        return self.sockets[max(self.session.connects, 1) - 1]

    async def speak(self, seconds: float) -> None:
        await self._push(seconds, SPEECH)

    async def quiet(self, seconds: float) -> None:
        await self._push(seconds, SILENCE)

    async def _push(self, seconds: float, prob: float) -> None:
        self.vad.prob = prob
        for _ in range(round(seconds / 0.02)):
            self.stream.push_frame(rtc.AudioFrame.create(SAMPLE_RATE, 1, SAMPLE_RATE // 50))
            await asyncio.sleep(0.02)

    def word(self, text: str, stop_s: float) -> None:
        """Gradium sends a word and its end, on the current connection's clock."""
        self.ws.server({"type": "text", "text": text, "start_s": 0.0})
        self.ws.server({"type": "end_text", "stop_s": stop_s})

    async def flush_sent(self) -> int:
        await _until(lambda: bool(self.ws.sent_of("flush")))
        return int(self.ws.sent_of("flush")[-1]["flush_id"])

    async def pause_ends_turn(self) -> int:
        """After "hello": a pause Gradium is sure of, up to the flush; returns its id."""
        await self.speak(0.4)
        self.word("hello", stop_s=0.4)
        await self.quiet(0.3)
        self.ws.server(_step(0.7, 0.99))
        return await self.flush_sent()

    async def wait_for(self, type_: SpeechEventType) -> None:
        await _until(lambda: type_ in self.types())

    def types(self) -> list[SpeechEventType]:
        return [e.type for e in self.events]

    def finals(self) -> list[SpeechEvent]:
        return [e for e in self.events if e.type == SpeechEventType.FINAL_TRANSCRIPT]

    def final_texts(self) -> list[str]:
        return [e.alternatives[0].text for e in self.finals()]


async def test_stream_feeds_the_vad_and_ends_the_turn_on_gradium_flush_reply():
    # End to end through the stream: the audio reaches the local VAD, its silence plus
    # Gradium's step makes the stream send a flush, and the final comes on the reply.
    async with _Stream() as s:
        await s.speak(0.4)
        s.word("hello", stop_s=0.4)
        await s.quiet(0.3)
        s.ws.server(_step(0.7, 0.99))
        flush_id = await s.flush_sent()
        assert s.finals() == []

        s.ws.server({"type": "flushed", "flush_id": flush_id})
        await s.wait_for(SpeechEventType.END_OF_SPEECH)
        assert s.final_texts() == ["hello"]


async def test_setup_asks_for_the_configured_model_delay():
    # The model delay sets how late words arrive, and so the latency of every turn.
    async with _Stream(delay_in_frames=10) as s:
        await _until(lambda: bool(s.ws.sent))
        assert s.ws.sent[0]["json_config"]["delay_in_frames"] == 10


async def test_after_a_reconnect_gradium_and_the_vad_stay_on_the_stream_clock():
    # A reconnect restarts Gradium's clock and the local VAD stream at 0. Both must be
    # re-anchored to the audio already consumed, or the pause of the open turn is measured
    # against the wrong times and never ends on the new connection's steps.
    async with _Stream(connections=2, max_retry=1, max_silence=5.0) as s:
        await s.speak(0.4)
        s.word("hello", stop_s=0.4)
        await s.wait_for(SpeechEventType.INTERIM_TRANSCRIPT)
        s.ws.server_close()
        await _until(lambda: s.session.connects == 2)

        await s.quiet(0.3)
        s.ws.server(_step(0.3, 0.99))  # 0.3 s into the new connection: 0.7 s of audio
        s.ws.server({"type": "flushed", "flush_id": await s.flush_sent()})
        await s.wait_for(SpeechEventType.END_OF_SPEECH)
        # the word ended 0.4 s into the stream, on the first connection's clock
        assert s.finals()[0].alternatives[0].end_time == pytest.approx(0.4, abs=0.05)


async def test_transient_server_error_does_not_end_the_turn():
    # The stream reconnects after a retryable error while the user may still be talking:
    # ending the turn there would make the agent answer half a sentence.
    async with _Stream(connections=2, max_retry=1) as s:
        await s.speak(0.4)
        s.word("I", stop_s=0.4)
        await s.wait_for(SpeechEventType.INTERIM_TRANSCRIPT)
        s.ws.server({"type": "error", "message": "worker failure", "code": 1011})
        await s.speak(0.3)
        s.word("want", stop_s=0.3)
        await s.quiet(0.3)
        s.ws.server(_step(0.6, 0.99))
        s.ws.server({"type": "flushed", "flush_id": await s.flush_sent()})
        await s.wait_for(SpeechEventType.END_OF_SPEECH)
        assert s.final_texts() == ["I want"]


async def test_words_are_final_when_the_stream_gives_up():
    # When no retry is left, the words already transcribed are all the user will get:
    # they must still reach the session as a final.
    async with _Stream() as s:
        await s.speak(0.4)
        s.ws.server({"type": "text", "text": "hello", "start_s": 0.0})
        s.ws.server({"type": "error", "message": "worker failure", "code": 1011})
        await _until(lambda: s.error is not None)
        assert s.final_texts() == ["hello"]


async def test_turn_ends_when_the_audio_stops():
    # A track that ends or a muted microphone stops the audio mid-turn; time only moves
    # with the audio, so without a watchdog the turn would stay open forever.
    async with _Stream(max_silence=1.0) as s:
        await s.speak(0.4)
        s.word("hello", stop_s=0.4)
        await s.wait_for(SpeechEventType.INTERIM_TRANSCRIPT)
        # no more audio
        s.ws.server({"type": "flushed", "flush_id": await s.flush_sent()})
        await s.wait_for(SpeechEventType.END_OF_SPEECH)
        assert s.final_texts() == ["hello"]


async def test_unanswered_flush_ends_the_turn_on_time_and_its_late_words_are_dropped():
    # A lost `flushed` must not hold the user's turn for long; and when the held words
    # come after all, they belong to the turn already ended, not to a new one.
    async with _Stream() as s:
        flush_id = await s.pause_ends_turn()
        loop = asyncio.get_running_loop()
        sent_at = loop.time()
        await s.wait_for(SpeechEventType.END_OF_SPEECH)
        assert loop.time() - sent_at == pytest.approx(1.5, abs=0.15)

        s.word("late", stop_s=0.7)
        s.ws.server({"type": "flushed", "flush_id": flush_id})
        await asyncio.sleep(0.1)
        assert s.final_texts() == ["hello"]
        assert s.types().count(SpeechEventType.START_OF_SPEECH) == 1


async def test_flush_pending_at_a_reconnect_is_sent_again():
    # Its reply was due on the connection that failed: without a new flush on the new
    # connection, the turn would wait for the flush timeout.
    async with _Stream(connections=2, max_retry=1) as s:
        await s.pause_ends_turn()
        s.ws.server_close()
        await _until(lambda: s.session.connects == 2)
        s.ws.server({"type": "flushed", "flush_id": await s.flush_sent()})
        await s.wait_for(SpeechEventType.END_OF_SPEECH)
        assert s.final_texts() == ["hello"]


async def test_session_flush_ends_the_turn_on_its_reply():
    # The session flushes the STT on its own turn decisions and waits for a final.
    async with _Stream() as s:
        await s.speak(0.4)
        s.word("hello", stop_s=0.4)
        await s.wait_for(SpeechEventType.INTERIM_TRANSCRIPT)
        s.stream.flush()
        s.ws.server({"type": "flushed", "flush_id": await s.flush_sent()})
        await s.wait_for(SpeechEventType.END_OF_SPEECH)
        assert s.final_texts() == ["hello"]


async def test_session_flush_without_speech_gets_an_empty_final():
    # Nothing was heard, but the session still waits for an answer to its flush.
    async with _Stream() as s:
        s.stream.flush()
        s.ws.server({"type": "flushed", "flush_id": await s.flush_sent()})
        await s.wait_for(SpeechEventType.FINAL_TRANSCRIPT)
        assert s.final_texts() == [""]


async def test_pause_decided_after_the_input_ended_is_left_to_the_server():
    # No flush can follow end_of_stream: the server's own end_of_stream ends the turn.
    async with _Stream(auto_eos=False) as s:
        await s.speak(0.4)
        s.word("hello", stop_s=0.4)
        await s.quiet(0.06)
        s.stream.end_input()
        await _until(lambda: bool(s.ws.sent_of("end_of_stream")))
        sent = len(s.ws.sent)  # end_input's own flush, then end_of_stream
        s.ws.server(_step(0.7, 0.99))
        await asyncio.sleep(0.1)
        assert len(s.ws.sent) == sent

        s.ws.server({"type": "end_of_stream"})
        await s.wait_for(SpeechEventType.END_OF_SPEECH)
        assert s.final_texts() == ["hello"]
