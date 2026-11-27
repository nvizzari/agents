"""Tests for `STTExperimental`: its turn state machine, presets and websocket stream.

Machine tests drive `TurnMachine` directly with scripted scores (the fake scorer returns
the first horizon probability as the score). Stream tests run the real stream against a
scripted fake Gradium server.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins.gradium.experimental import machine as m, signals
from livekit.plugins.gradium.experimental.profiles import TurnProfile, preset

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

# ---------------------------------------------------------------------------
# machine helpers

# A profile pinned for the tests, independent of the provisional preset values.
PROFILE = TurnProfile(
    eager_threshold=0.5,
    commit_threshold=0.8,
    min_silence=0.16,
    max_silence=2.0,
    confirm_steps=1,
    confirm_timeout=1.0,
    max_eager_per_turn=2,
    pause_threshold=0.3,
    resume_threshold=0.1,
    start_threshold=None,
    warmup_steps=0,
    delay_in_frames=10,
    preflight=True,
    weights=None,
)
# Scores named by what they do under PROFILE.
SPEECH = 0.0  # below resume_threshold: talking
HOLD = 0.4  # paused, below eager_threshold
LIKELY = 0.6  # enters EAGER, below commit_threshold
END = 0.9  # commits

A = TypeVar("A", bound=m.Action)


class FakeScorer:
    """Returns the first horizon probability as the score and counts the calls."""

    name = "fake"

    def __init__(self) -> None:
        self.steps = 0
        self.flushes = 0
        self.turn_ends = 0

    def reset(self) -> None:
        pass

    def on_step(self, step: signals.Step) -> float:
        self.steps += 1
        return step.probs[0]

    def on_word(self, word: signals.Word, audio_time: float) -> None:
        pass

    def on_flush(self) -> None:
        self.flushes += 1

    def on_turn_end(self) -> None:
        self.turn_ends += 1


class Driver:
    """Feeds a machine one 80 ms step at a time and records every action it returns."""

    def __init__(self, **knobs: Any) -> None:
        self.scorer = FakeScorer()
        self.machine = m.TurnMachine(dataclasses.replace(PROFILE, **knobs), self.scorer)
        self.t = 0.0
        self.actions: list[m.Action] = []

    def _record(self, actions: Sequence[m.Action]) -> None:
        self.actions.extend(actions)

    def steps(self, score: float, n: int = 1) -> Driver:
        for _ in range(n):
            self.t = round(self.t + signals.STEP_DURATION, 6)
            self._record(self.machine.on_signal(signals.Step(self.t, (score,) * 4)))
        return self

    def seconds(self, score: float, duration: float) -> Driver:
        return self.steps(score, round(duration / signals.STEP_DURATION))

    def word(self, text: str, start: float | None = None, stop: float | None = None) -> Driver:
        """A word starting at `start` (default: now), as it arrives from the server."""
        start = self.t if start is None else start
        self._record(self.machine.on_signal(signals.Word(text, start)))
        if stop is not None:
            self._record(self.machine.on_signal(signals.WordEnd(stop)))
        return self

    def speak(self, *words: str) -> Driver:
        """Speech scores, then the words as if just transcribed."""
        self.steps(SPEECH, 2)
        for w in words:
            self.word(w)
        return self

    def flushed(self, flush_id: int | None = None) -> Driver:
        """Answer `flush_id`, by default the last flush sent."""
        if flush_id is None:
            flush_id = self.of(m.SendFlush)[-1].flush_id
        self._record(self.machine.on_signal(signals.Flushed(flush_id)))
        return self

    def request_flush(self) -> Driver:
        self._record(self.machine.request_flush())
        return self

    def finalize(self) -> Driver:
        self._record(self.machine.finalize())
        return self

    def finalize_keeping_turn(self) -> Driver:
        self._record(self.machine.finalize(keep_turn=True))
        return self

    def on(self, sig: signals.Signal) -> Driver:
        self._record(self.machine.on_signal(sig))
        return self

    def of(self, cls: type[A]) -> list[A]:
        return [a for a in self.actions if isinstance(a, cls)]

    def texts(self, cls: type[m.Interim | m.Preflight | m.Final]) -> list[str]:
        return [a.text for a in self.of(cls)]

    def clear(self) -> Driver:
        self.actions.clear()
        return self

    @property
    def state(self) -> m.TurnState:
        return self.machine.state


def _until_eager(d: Driver) -> Driver:
    """Pause long enough at an eager score to enter EAGER under PROFILE."""
    return d.steps(HOLD).seconds(LIKELY, PROFILE.min_silence)


# ---------------------------------------------------------------------------
# machine


def test_hold_then_resume_ends_no_turn() -> None:
    # A mid-sentence hold that never looks like an end must not end the turn or flush.
    d = Driver().speak("so", "I", "think").seconds(HOLD, 1.0)
    assert d.state is m.TurnState.PAUSED
    d.steps(SPEECH).word("that")
    assert d.state is m.TurnState.SPEAKING
    assert not d.of(m.Final) and not d.of(m.EndOfSpeech) and not d.of(m.SendFlush)
    assert d.texts(m.Interim)[-1] == "so I think that"


def test_eager_then_resume_emits_a_differing_interim_and_no_final() -> None:
    # After a preflight, resumed speech must produce an interim that differs from the
    # preflight (so the framework cancels the preemptive generation) and no final.
    d = _until_eager(Driver().speak("book", "a", "table"))
    assert d.state is m.TurnState.EAGER
    d.flushed()
    assert d.texts(m.Preflight) == ["book a table"]
    d.steps(SPEECH).word("for", start=d.t)
    assert d.state is m.TurnState.SPEAKING
    assert d.texts(m.Interim)[-1] == "book a table for"
    assert not d.of(m.Final) and not d.of(m.EndOfSpeech)


def test_committed_turn_final_equals_preflight_and_carries_word_end() -> None:
    # The preemptive generation is adopted only when the final equals the preflight, and
    # the framework measures its endpointing delay from END_OF_SPEECH's end time.
    d = Driver().steps(SPEECH, 2).word("hello", 0.1, 0.4).word("there", 0.5, 0.83)
    _until_eager(d).flushed().steps(END)
    assert d.texts(m.Preflight) == d.texts(m.Final) == ["hello there"]
    (eos,) = d.of(m.EndOfSpeech)
    assert eos.end_time == pytest.approx(0.83)
    assert len(d.of(m.SendFlush)) == 1
    assert d.state is m.TurnState.IDLE
    assert d.scorer.turn_ends == 1


@pytest.mark.parametrize("server_answers", [True, False])
def test_max_silence_always_ends_the_turn(server_answers: bool) -> None:
    # A score stuck below every threshold, or a server that never answers the flush,
    # must not hold the turn open: max_silence commits and the turn is emitted.
    d = Driver().speak("hmm").seconds(HOLD, PROFILE.max_silence + signals.STEP_DURATION)
    assert d.state is m.TurnState.COMMITTED
    assert len(d.of(m.SendFlush)) == 1
    if server_answers:
        d.flushed()
    else:
        d.seconds(HOLD, m.FLUSH_REPLY_TIMEOUT)
    assert d.texts(m.Final) == ["hmm"]
    assert len(d.of(m.EndOfSpeech)) == 1
    assert d.state is m.TurnState.IDLE


def test_finalize_mid_pause_emits_every_received_word() -> None:
    # A reconnect, server end or end of input must not lose words already received,
    # including the delayed tail that arrives during the pause.
    d = Driver().speak("one", "two").seconds(HOLD, 0.4)
    d.word("three", start=d.t - 0.6)
    d.finalize()
    assert d.texts(m.Final) == ["one two three"]
    assert len(d.of(m.EndOfSpeech)) == 1
    assert d.state is m.TurnState.IDLE


def test_max_eager_per_turn_bounds_flushes_and_preflights() -> None:
    # Every wasted preflight spends one of the framework's preemptive generation slots;
    # the cap bounds them, and the turn can still end once the cap is reached.
    d = Driver().speak("a")
    for i in range(4):
        _until_eager(d)
        if d.state is m.TurnState.EAGER:
            d.flushed()
        d.steps(SPEECH).word(f"w{i}", start=d.t)
    assert len(d.of(m.SendFlush)) == len(d.of(m.Preflight)) == PROFILE.max_eager_per_turn
    d.steps(HOLD).seconds(END, PROFILE.min_silence + signals.STEP_DURATION).flushed()
    assert d.texts(m.Final) == ["a w0 w1 w2 w3"]


def test_preflight_disabled_commits_from_pause_with_a_single_flush() -> None:
    # preflight=False never emits PREFLIGHT_TRANSCRIPT; the commit still flushes once so
    # the final includes the words held by the model delay.
    d = Driver(preflight=False).speak("yes").steps(HOLD).seconds(END, 0.3)
    assert d.state is m.TurnState.COMMITTED
    d.word("please", start=d.t - 0.8).flushed()
    assert not d.of(m.Preflight)
    assert len(d.of(m.SendFlush)) == 1
    assert d.texts(m.Final) == ["yes please"]


@pytest.mark.parametrize(
    ("words", "expected"),
    [((), [""]), (("call", "me"), ["call me"])],
)
def test_manual_flush_is_always_answered_with_a_final(
    words: tuple[str, ...], expected: list[str]
) -> None:
    # commit_user_turn waits for a FINAL_TRANSCRIPT after its flush; it must get one,
    # empty when nothing was heard, and an open turn must end with it.
    d = Driver()
    if words:
        d.speak(*words)
    d.request_flush().flushed()
    assert d.texts(m.Final) == expected
    assert len(d.of(m.EndOfSpeech)) == (1 if words else 0)
    assert d.state is m.TurnState.IDLE


def test_warmup_steps_cannot_open_a_turn_and_start_threshold_opens_one_before_words() -> None:
    # The head has no context during warm-up, so its scores are neither scored nor
    # acted on; after it, a low score opens the turn before the first word (barge-in).
    d = Driver(warmup_steps=3, start_threshold=0.05).steps(SPEECH, 3)
    assert d.scorer.steps == 0 and d.machine.last_score is None
    assert not d.of(m.StartOfSpeech)
    d.steps(SPEECH)
    assert len(d.of(m.StartOfSpeech)) == 1
    assert d.state is m.TurnState.SPEAKING
    assert not d.of(m.Interim)


def test_delayed_words_do_not_end_a_pause() -> None:
    # Words arrive ~1.2 s after they start, so the tail of the speech before a pause
    # arrives during it. It must join the turn without counting as a resume.
    d = _until_eager(Driver().speak("what", "time"))
    d.word("is", start=d.t - 1.0).word("it", start=d.t - 0.9)
    assert d.state is m.TurnState.EAGER
    d.flushed()
    assert d.texts(m.Preflight) == ["what time is it"]


def test_flush_reply_after_resume_emits_no_preflight() -> None:
    # A preflight sent after the speaker resumed would be stale on arrival and waste a
    # preemptive generation.
    d = _until_eager(Driver().speak("wait")).steps(SPEECH).flushed()
    assert d.state is m.TurnState.SPEAKING
    assert not d.of(m.Preflight)


def test_unchanged_words_are_not_preflighted_twice() -> None:
    # A score dip with no new words leaves the first preemptive generation valid; a
    # second identical preflight would restart it and spend another framework slot.
    d = _until_eager(Driver().speak("maybe")).flushed()
    _until_eager(d.steps(SPEECH)).flushed()
    assert len(d.of(m.SendFlush)) == 2
    assert d.texts(m.Preflight) == ["maybe"]


def test_words_after_the_commit_point_open_the_next_turn() -> None:
    # While the commit waits for its flush reply, a word of the committed turn joins its
    # final, and a word spoken after the commit opens the next turn once it is emitted.
    d = Driver(preflight=False).speak("first").steps(HOLD).seconds(END, 0.3)
    commit_t = d.t
    d.word("turn", start=commit_t - 0.5).word("second", start=commit_t + 0.1).flushed()
    assert d.texts(m.Final) == ["first turn"]
    assert len(d.of(m.StartOfSpeech)) == 2
    assert d.texts(m.Interim)[-1] == "second"
    assert d.state is m.TurnState.SPEAKING


def test_word_arriving_after_its_turn_was_emitted_opens_the_next_turn() -> None:
    # After a false cut-off the user is still talking. A word that reaches us once the
    # turn is emitted must not be lost: it opens a new turn, which lets the user barge in
    # on the reply, even when the word started before the commit point.
    d = Driver(preflight=False).speak("done").steps(HOLD).seconds(END, 0.3).flushed()
    d.clear().word("actually", start=d.t - 0.5)
    assert len(d.of(m.StartOfSpeech)) == 1
    assert d.texts(m.Interim) == ["actually"]
    assert d.state is m.TurnState.SPEAKING


def test_turn_without_words_ends_without_a_final() -> None:
    # A turn opened on the score alone (a cough, noise) ends the user turn without
    # inventing a transcript or a preflight.
    d = _until_eager(Driver(start_threshold=0.05).steps(SPEECH))
    assert d.state is m.TurnState.EAGER
    d.flushed().steps(END)
    assert not d.of(m.Final) and not d.of(m.Preflight)
    assert len(d.of(m.EndOfSpeech)) == 1
    assert d.state is m.TurnState.IDLE


def test_manual_flush_overtaken_by_the_commit_gets_a_single_final() -> None:
    # commit_user_turn during EAGER: the eager reply lets the turn commit before the
    # manual flush is answered. That final already answers the manual flush; its reply
    # must not add an empty FINAL_TRANSCRIPT outside any turn (live check).
    d = _until_eager(Driver().speak("yes", "exactly")).request_flush()
    eager_id, manual_id = (a.flush_id for a in d.of(m.SendFlush))
    d.flushed(eager_id).steps(END)
    assert d.texts(m.Final) == ["yes exactly"]
    d.flushed(manual_id)
    assert d.texts(m.Final) == ["yes exactly"]
    assert d.state is m.TurnState.IDLE


def test_finalize_keeping_the_turn_hands_it_to_the_next_connection() -> None:
    # A reconnect mid-sentence must not end the user's turn: the words so far are
    # emitted as a final (the framework appends it to the turn) and the turn ends on the
    # next connection, with no second START_OF_SPEECH.
    d = Driver().speak("so", "what").finalize_keeping_turn()
    assert d.texts(m.Final) == ["so what"]
    assert not d.of(m.EndOfSpeech)
    assert d.state is m.TurnState.SPEAKING
    # the next connection: its audio clock restarts at zero
    d.on(signals.Ready(10, 1920, 24000, "test-model", "req-2"))
    d.t = 0.0
    d.steps(HOLD).seconds(END, PROFILE.min_silence + signals.STEP_DURATION).flushed()
    assert len(d.of(m.EndOfSpeech)) == 1
    assert len(d.of(m.StartOfSpeech)) == 1
    assert d.state is m.TurnState.IDLE


def test_turn_carried_across_a_reconnect_mid_pause_ends_without_waiting() -> None:
    # Live check: a reconnect while the user was silent restarted the turn in SPEAKING,
    # so it ended seconds late and END_OF_SPEECH named that late point as the speech end.
    # The pause must go on from the new connection's first audio.
    d = Driver().speak("yes").steps(HOLD, 3).finalize_keeping_turn()
    assert d.state is m.TurnState.PAUSED
    d.on(signals.Ready(10, 1920, 24000, "test-model", "req-2"))
    d.t = 0.0
    d.seconds(HOLD, PROFILE.max_silence + signals.STEP_DURATION).flushed()
    (eos,) = d.of(m.EndOfSpeech)
    assert eos.end_time == 0.0
    assert d.state is m.TurnState.IDLE


def test_one_flush_reply_answers_every_older_flush() -> None:
    # With an eager flush and a manual flush in flight, a single reply to the newer one
    # must still answer both: the manual one gets its final.
    d = _until_eager(Driver().speak("ok")).request_flush()
    eager_id, manual_id = (a.flush_id for a in d.of(m.SendFlush))
    assert eager_id < manual_id
    d.flushed(manual_id)
    assert d.texts(m.Final) == ["ok"]
    assert d.state is m.TurnState.IDLE


# ---------------------------------------------------------------------------
# signals and presets


def test_warmup_steps_are_dropped_and_do_not_shift_timestamps() -> None:
    # The server opens every stream with steps at total_duration_s == 0. Counting them
    # stamped every Gradium score 240 ms late on eot-bench.
    def step(total: float) -> dict[str, Any]:
        return {
            "type": "step",
            "step_idx": 0,
            "total_duration_s": total,
            "vad": [{"horizon_s": h, "inactivity_prob": 0.5} for h in signals.HORIZONS],
        }

    parsed = [signals.parse(step(t)) for t in (0.0, 0.0, 0.0, 0.08, 0.16)]
    assert parsed[:3] == [None, None, None]
    assert [s.audio_time for s in parsed[3:] if isinstance(s, signals.Step)] == [0.08, 0.16]


def test_every_preset_is_valid_and_overrides_are_validated() -> None:
    # Users pick presets by name and tweak single knobs; a bad combination must fail at
    # construction rather than as a turn that can never end. Learned presets load the
    # bundled weights; fallback presets must not, or the fallback would never run.
    for kind in ("learned", "fallback"):
        for eagerness in ("responsive", "balanced", "prudent"):
            p = preset(eagerness, scorer=kind)  # type: ignore[arg-type]
            assert p.weights == ("default" if kind == "learned" else None)
    assert preset("prudent", overrides={"max_silence": 6.0}).max_silence == 6.0
    with pytest.raises(ValueError):
        preset("balanced", overrides={"eager_threshold": 0.95, "commit_threshold": 0.9})
    with pytest.raises(ValueError):
        preset("balanced", overrides={"delay_in_frames": -1})
    with pytest.raises(ValueError):
        preset("balanced", overrides={"no_such_knob": 1})  # type: ignore[typeddict-unknown-key]


# ---------------------------------------------------------------------------
# stream helpers

SAMPLE_RATE = 24000
NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5)
ONE_RETRY = APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=5)


@dataclass
class _Msg:
    type: aiohttp.WSMsgType
    data: Any = None
    extra: Any = None


class AwaitFlush:
    """Script marker: wait for the client's next flush and answer it with `flushed`."""


class Server:
    """Frame builders for the scripted Gradium server; `t` is the audio clock."""

    def __init__(self) -> None:
        self.t = 0.0

    @staticmethod
    def ready(model: str = "test-model") -> dict[str, Any]:
        return {
            "type": "ready",
            "request_id": "req-1",
            "model_ext": model,
            "sample_rate": SAMPLE_RATE,
            "frame_size": 1920,
            "delay_in_frames": 10,
        }

    def steps(self, score: float, n: int = 1) -> list[dict[str, Any]]:
        out = []
        for _ in range(n):
            self.t = round(self.t + signals.STEP_DURATION, 6)
            out.append(
                {
                    "type": "step",
                    "step_idx": round(self.t / signals.STEP_DURATION),
                    "total_duration_s": self.t,
                    "vad": [{"horizon_s": h, "inactivity_prob": score} for h in signals.HORIZONS],
                }
            )
        return out

    @staticmethod
    def word(text: str, start: float, stop: float) -> list[dict[str, Any]]:
        return [
            {"type": "text", "text": text, "start_s": start},
            {"type": "end_text", "stop_s": stop},
        ]


class FakeWebSocket:
    """Replays a script of server frames, reacting to the client's flush and EOS.

    Script items: a dict is sent as a TEXT frame, a str as a raw TEXT frame, `AwaitFlush`
    waits for a client flush and answers it, and the int `close_code` closes the socket.
    Once the script is exhausted, every client flush is answered at once and the
    client's end_of_stream gets the server's end_of_stream and a normal close.
    """

    def __init__(self, script: Sequence[Any], *, close_code: int = 1000) -> None:
        self._script: deque[Any] = deque(script)
        self._out: deque[_Msg] = deque()
        self._client: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.close_code = close_code
        self.sent: list[dict[str, Any]] = []

    def sent_of(self, type_: str) -> list[dict[str, Any]]:
        return [msg for msg in self.sent if msg["type"] == type_]

    async def send_str(self, data: str) -> None:
        msg = json.loads(data)
        self.sent.append(msg)
        if msg["type"] in ("flush", "end_of_stream"):
            self._client.put_nowait(msg)

    async def receive(self) -> _Msg:
        while not self._out:
            if self._script:
                item = self._script.popleft()
                if isinstance(item, AwaitFlush):
                    flush = await self._client.get()
                    assert flush["type"] == "flush", flush
                    self._text({"type": "flushed", "flush_id": flush["flush_id"]})
                elif isinstance(item, int):
                    self._out.append(_Msg(aiohttp.WSMsgType.CLOSE, item))
                else:
                    self._text(item)
                continue
            client = await self._client.get()
            if client["type"] == "flush":
                self._text({"type": "flushed", "flush_id": client["flush_id"]})
            else:
                self._text({"type": "end_of_stream"})
                self._out.append(_Msg(aiohttp.WSMsgType.CLOSE, 1000))
        return self._out.popleft()

    def _text(self, frame: dict[str, Any] | str) -> None:
        data = frame if isinstance(frame, str) else json.dumps(frame)
        self._out.append(_Msg(aiohttp.WSMsgType.TEXT, data))

    async def ping(self) -> None:
        pass

    async def close(self) -> bool:
        return True

    def exception(self) -> BaseException | None:
        return None


class FakeSession:
    """Hands out one scripted socket per connection attempt."""

    def __init__(self, *sockets: FakeWebSocket) -> None:
        self._sockets = deque(sockets)
        self.connects: list[dict[str, Any]] = []
        self.urls: list[str] = []

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
        self.connects.append(kwargs)
        self.urls.append(url)
        return self._sockets.popleft()


def make_stream(
    *sockets: FakeWebSocket,
    conn_options: APIConnectOptions = NO_RETRY,
    pinned: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any, FakeSession]:
    """A provider and stream on `sockets`; `pinned` sets every knob to PROFILE."""
    from livekit.plugins.gradium import STTExperimental

    session = FakeSession(*sockets)
    if pinned:
        kwargs["overrides"] = dataclasses.asdict(PROFILE)
    provider = STTExperimental(
        api_key="test-key",
        http_session=session,  # type: ignore[arg-type]
        **kwargs,
    )
    return provider, provider.stream(conn_options=conn_options), session


def audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame.create(sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=1920)


async def collect(
    stream: Any, until: Callable[[SpeechEvent], bool] | None = None
) -> list[SpeechEvent]:
    """Events until `until` matches one (included) or the stream ends."""
    events: list[SpeechEvent] = []
    async for ev in stream:
        events.append(ev)
        if until is not None and until(ev):
            break
    return events


def of_type(events: list[SpeechEvent], type_: SpeechEventType) -> list[SpeechEvent]:
    return [ev for ev in events if ev.type == type_]


def texts(events: list[SpeechEvent], type_: SpeechEventType) -> list[str]:
    return [ev.alternatives[0].text for ev in of_type(events, type_)]


def is_type(type_: SpeechEventType) -> Callable[[SpeechEvent], bool]:
    return lambda ev: ev.type == type_


# ---------------------------------------------------------------------------
# stream


async def test_stream_flushes_once_and_preflight_matches_final() -> None:
    # On a likely end the stream sends exactly one flush, previews the turn with a
    # preflight that the final then matches (so the generation is adopted), and stamps
    # END_OF_SPEECH with the last word's end in the caller's clock.
    srv = Server()
    script = [
        srv.ready(),
        *srv.steps(SPEECH, 2),
        *srv.word("hello", 0.1, 0.4),
        *srv.word("there", 0.5, 0.83),
        *srv.steps(HOLD),
        *srv.steps(LIKELY, 2),
        AwaitFlush(),
        *srv.steps(END),
    ]
    ws = FakeWebSocket(script)
    _, stream, session = make_stream(ws)
    stream.start_time_offset = 7.0
    stream.push_frame(audio_frame())
    events = await collect(stream, until=is_type(SpeechEventType.END_OF_SPEECH))
    await stream.aclose()

    assert [ev.type for ev in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.PREFLIGHT_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert texts(events, SpeechEventType.PREFLIGHT_TRANSCRIPT) == ["hello there"]
    assert texts(events, SpeechEventType.FINAL_TRANSCRIPT) == ["hello there"]
    offset = stream.start_time_offset
    assert offset > 6.0
    assert events[-1].alternatives[0].end_time == pytest.approx(offset + 0.83)
    assert len(ws.sent_of("flush")) == 1
    setup = ws.sent[0]
    assert setup["type"] == "setup"
    assert setup["json_config"] == {"delay_in_frames": 10, "language": "en"}
    assert session.connects[0]["headers"]["x-api-key"] == "test-key"
    assert "test-key" not in json.dumps(ws.sent)


@pytest.mark.parametrize(("close_code", "connects"), [(1011, 2), (1002, 1)])
async def test_unexpected_close_keeps_words_and_retries_only_transient_failures(
    close_code: int, connects: int
) -> None:
    # A dropped socket must not lose the words received mid-pause; a server failure is
    # retried, while a request the server rejects as malformed (1002) is not. The words
    # arrive as a final at once. A retried connection inherits the turn, while a
    # terminal failure ends it before the stream raises.
    srv = Server()
    first = FakeWebSocket(
        [
            srv.ready(),
            *srv.steps(SPEECH, 2),
            *srv.word("still", 0.0, 0.3),
            *srv.steps(HOLD, 3),
            close_code,
        ],
        close_code=close_code,
    )
    second = FakeWebSocket([Server.ready()])
    _, stream, session = make_stream(first, second, conn_options=ONE_RETRY)
    stream.push_frame(audio_frame())

    if connects == 1:
        events: list[SpeechEvent] = []
        with pytest.raises(APIStatusError):
            async for ev in stream:
                events.append(ev)
        assert events[-1].type == SpeechEventType.END_OF_SPEECH
    else:
        events = await collect(stream, until=is_type(SpeechEventType.FINAL_TRANSCRIPT))
        assert not of_type(events, SpeechEventType.END_OF_SPEECH)
        stream.end_input()
        events += await collect(stream)
        assert [ev.type for ev in events][-2:] == [
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
        ]
        assert len(of_type(events, SpeechEventType.START_OF_SPEECH)) == 1
    assert texts(events, SpeechEventType.FINAL_TRANSCRIPT)[0] == "still"
    assert len(session.connects) == connects


async def test_failed_retry_setup_ends_the_inherited_turn() -> None:
    # A turn handed to a retry that never connects has no later connection to end it.
    # The session must still get END_OF_SPEECH before the stream raises, or it would
    # wait on a user turn that can no longer end.
    srv = Server()
    ws = FakeWebSocket([srv.ready(), *srv.word("hello", 0.0, 0.3), 1011], close_code=1011)
    _, stream, session = make_stream(ws, conn_options=ONE_RETRY)
    async with stream:
        events: list[SpeechEvent] = []
        with pytest.raises(APIConnectionError):
            async for ev in stream:
                events.append(ev)
    assert len(session.connects) == 2
    assert texts(events, SpeechEventType.FINAL_TRANSCRIPT) == ["hello"]
    assert events[-1].type == SpeechEventType.END_OF_SPEECH


async def test_malformed_messages_are_skipped() -> None:
    # A message the stream cannot parse is logged and dropped; the session goes on.
    srv = Server()
    script = [
        srv.ready(),
        "not json",
        "[1, 2]",
        {"type": "step", "total_duration_s": 0.08},
        {"type": "text", "start_s": 0.0},
        *srv.steps(SPEECH, 2),
        *srv.word("fine", 0.1, 0.3),
    ]
    _, stream, session = make_stream(FakeWebSocket(script))
    stream.push_frame(audio_frame())
    events = await collect(stream, until=is_type(SpeechEventType.INTERIM_TRANSCRIPT))
    stream.end_input()
    events += await collect(stream)
    assert texts(events, SpeechEventType.FINAL_TRANSCRIPT) == ["fine"]
    assert len(session.connects) == 1


async def test_update_options_reconnects_only_for_setup_options() -> None:
    # Thresholds change live on the running connection. A setup option needs a new
    # connection: the old one is drained first, so the words it held are not lost, and
    # the user's turn is not ended by the reconnect (live check: a reconnect mid-sentence
    # used to emit END_OF_SPEECH, a false cut-off caused by a settings change).
    srv = Server()
    first = FakeWebSocket([srv.ready(), *srv.steps(SPEECH, 2), *srv.word("bonjour", 0.0, 0.3)])
    second = FakeWebSocket([Server.ready()])
    provider, stream, session = make_stream(first, second)
    stream.push_frame(audio_frame())
    events = await collect(stream, until=is_type(SpeechEventType.INTERIM_TRANSCRIPT))

    provider.update_options(overrides={"commit_threshold": 0.95})
    assert stream.machine.profile.commit_threshold == 0.95
    provider.update_options(language="fr")
    events += await collect(stream, until=is_type(SpeechEventType.FINAL_TRANSCRIPT))
    assert not of_type(events, SpeechEventType.END_OF_SPEECH)
    stream.end_input()
    events += await collect(stream)

    assert texts(events, SpeechEventType.FINAL_TRANSCRIPT) == ["bonjour"]
    assert [ev.type for ev in events].count(SpeechEventType.START_OF_SPEECH) == 1
    assert events[-1].type == SpeechEventType.END_OF_SPEECH
    assert len(first.sent_of("end_of_stream")) == 1
    assert len(session.connects) == 2
    assert second.sent[0]["json_config"]["language"] == "fr"
    assert stream.machine.profile.commit_threshold == 0.95


async def test_end_input_ends_the_turn_without_an_empty_final() -> None:
    # end_input() queues a flush before closing the input. It is not a manual flush:
    # answering it emitted an empty FINAL_TRANSCRIPT outside any turn on every stream
    # end (live check). The open turn ends with its words, once.
    srv = Server()
    ws = FakeWebSocket([srv.ready(), *srv.steps(SPEECH, 2), *srv.word("bye", 0.0, 0.3)])
    _, stream, _ = make_stream(ws)
    stream.push_frame(audio_frame())
    events = await collect(stream, until=is_type(SpeechEventType.INTERIM_TRANSCRIPT))
    stream.end_input()
    events += await collect(stream)
    assert [ev.type for ev in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert not ws.sent_of("flush")


async def test_drain_collects_words_without_sending_a_new_flush() -> None:
    # After end_of_stream the server still returns steps and the last words. The words
    # belong in the final transcript, but the steps must not drive the turn: a likely
    # end of turn there would send a flush on a socket that is closing.
    srv = Server()

    class DrainSocket(FakeWebSocket):
        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if json.loads(data)["type"] == "end_of_stream":
                for msg in [*srv.steps(LIKELY, 8), *srv.word("there", 0.24, 0.32)]:
                    self._text(msg)

    ws = DrainSocket([srv.ready(), *srv.word("hello", 0.0, 0.16)])
    _, stream, _ = make_stream(ws)
    async with stream:
        await collect(stream, until=is_type(SpeechEventType.INTERIM_TRANSCRIPT))
        stream.end_input()
        events = await collect(stream)

    assert texts(events, SpeechEventType.FINAL_TRANSCRIPT) == ["hello there"]
    assert events[-1].type == SpeechEventType.END_OF_SPEECH
    assert not ws.sent_of("flush")


@pytest.mark.parametrize("language", ["en", "ja-JP", "any", "not-a-language"])
async def test_language_reaches_the_server_as_given(language: str) -> None:
    # The server owns the list of languages: the hint is sent untouched, so a new server
    # language needs no plugin release and a wrong one gets the server's own error.
    srv = Server()
    ws = FakeWebSocket([srv.ready(), *srv.steps(SPEECH, 2), *srv.word("ok", 0.0, 0.2)])
    _, stream, _ = make_stream(ws, language=language)
    stream.push_frame(audio_frame())
    await collect(stream, until=is_type(SpeechEventType.INTERIM_TRANSCRIPT))
    await stream.aclose()
    assert ws.sent[0]["json_config"]["language"] == language


async def test_api_host_selects_the_websocket_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Users choose the API host only; scheme and path belong to the plugin. A full URL
    # (the old `model_endpoint` form) must fail at construction, not at connect time.
    from livekit.plugins.gradium import STTExperimental

    monkeypatch.setenv("GRADIUM_API", "env.example.test")
    for kwargs, host in (
        ({}, "env.example.test"),
        ({"gradium_api": "eu.example.test"}, "eu.example.test"),
    ):
        ws = FakeWebSocket([Server().ready()])
        _, stream, session = make_stream(ws, **kwargs)
        stream.push_frame(audio_frame())
        stream.end_input()
        await collect(stream)
        await stream.aclose()
        assert session.urls == [f"wss://{host}/api/speech/asr"]
    with pytest.raises(ValueError):
        STTExperimental(api_key="test-key", gradium_api="wss://api.gradium.ai/api/speech/asr")


@pytest.mark.parametrize("model_matches", [True, False])
async def test_thresholds_follow_the_scorer_that_loaded(model_matches: bool) -> None:
    # The learned and fallback scores live on different scales. When the server runs
    # another model than the weights were trained on, the stream falls back to
    # min(p2, p3) and must also switch to the thresholds calibrated for it; otherwise the
    # preset's false cut-off budget no longer holds.
    from livekit.plugins.gradium.experimental.scorer import LearnedScorer, load_weights

    model = load_weights("default").model if model_matches else "another-model"
    srv = Server()
    ws = FakeWebSocket([srv.ready(model), *srv.steps(SPEECH, 2), *srv.word("ok", 0.0, 0.2)])
    _, stream, _ = make_stream(ws, pinned=False, eagerness="prudent")
    stream.push_frame(audio_frame())
    await collect(stream, until=is_type(SpeechEventType.INTERIM_TRANSCRIPT))
    kind = "learned" if model_matches else "fallback"
    assert isinstance(stream.machine.scorer, LearnedScorer) is model_matches
    assert stream.machine.profile == preset("prudent", scorer=kind)
    # a turn knob update keeps the thresholds of the scorer in use
    stream.update_options(eagerness="responsive")
    assert stream.machine.profile == preset("responsive", scorer=kind)
    await stream.aclose()
