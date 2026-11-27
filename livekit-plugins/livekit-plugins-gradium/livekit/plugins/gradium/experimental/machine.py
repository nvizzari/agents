"""The turn state machine: Gradium signals in, turn-taking actions out.

Everything the plugin decides about turns happens here. The machine has no asyncio, no
sockets and no wall clock: its only clock is the audio time carried by `Step` messages
(`Step.audio_time`, seconds of audio the server has processed on this connection). The
stream, the offline simulator and the eot-bench adapter all drive this same class, so the
benchmark measures the code that ships.

Inputs:
    `on_signal(sig)`: one parsed server message (see `signals`).
    `request_flush()`: the framework asked for a manual flush (`manual_flush=True`).
    `finalize(keep_turn)`: the connection is over (end of input, socket lost, reconnect).

Outputs: a sequence of actions (`StartOfSpeech`, `Interim`, `Preflight`, `Final`,
`EndOfSpeech`, `SendFlush`). The caller executes them in order.

States:
    IDLE: No turn open.
    SPEAKING: The user is talking. Entered from IDLE with `StartOfSpeech`.
    PAUSED: A silence inside the turn is being scored. The silence clock starts on entry.
    EAGER: The turn has probably ended. A `flush` is sent on entry; its `flushed` reply
        emits `Preflight` with every pending word of the turn.
    COMMITTED: The turn has ended; waiting for the flush that returns the words still
        held by the model delay. Its reply emits `Final` then `EndOfSpeech` and returns
        to IDLE.

Transitions (each is also commented where it fires):
    IDLE -> SPEAKING       a word, or a score at or
                           below `start_threshold`.
    SPEAKING -> PAUSED     score >= `pause_threshold`.
    PAUSED -> SPEAKING     score < `resume_threshold`, or a word that starts after the
                           pause began.
    PAUSED -> EAGER        eager entry allowed, silence >= `min_silence` and
                           score >= `eager_threshold`.
    PAUSED -> COMMITTED    no eager entry allowed (`preflight=False` or
                           `max_eager_per_turn` reached), silence >= `min_silence` and
                           score >= `commit_threshold` for `confirm_steps` steps; or
                           silence >= `max_silence`. A flush is sent.
    EAGER -> SPEAKING      score < `resume_threshold`, or a word that starts after the
                           pause began. The interim that follows the new words makes the
                           framework cancel the preemptive generation.
    EAGER -> COMMITTED     score >= `commit_threshold` for `confirm_steps` steps,
                           `confirm_timeout` in EAGER, or silence >= `max_silence`. The
                           eager flush is reused: no second flush.
    COMMITTED -> IDLE      the flush reply arrives, or `FLUSH_REPLY_TIMEOUT` of audio
                           passes without it.
    any -> IDLE            `finalize()`, server `end_of_stream`, server `error`, a new
                           `ready` after steps on the old connection: buffered words are
                           emitted as `Final`, then `EndOfSpeech`.
    open -> SPEAKING       `finalize(keep_turn=True)` (a reconnect, or a lost socket the
                           stream retries): buffered words are emitted as `Final` without
                           `EndOfSpeech`. The framework appends that final to the user's
                           turn, which continues on the next connection; ending the turn
                           there would cut the user off mid-sentence. A turn that was
                           PAUSED or EAGER continues as PAUSED, its silence counted from
                           the new connection's first audio. A COMMITTED turn is
                           completed instead: it had already ended.
    any open -> IDLE       the reply to a manual flush: `Final` (possibly empty), then
                           `EndOfSpeech`.

Choices for the edge cases:
    Late words. Words arrive about 1.2 s after they start, so the words spoken just
    before a pause arrive while PAUSED or EAGER. A word is a resume only when it starts
    at or after the pause began (`Word.start_time >= pause start`); earlier words are
    appended to the turn without leaving the pause. In COMMITTED, words that start
    before the commit point belong to the committed turn; later ones are held and open
    the next turn once the committed one is emitted. A word that arrives after the turn
    was emitted opens the next turn, even when it started before the commit point: the
    server sends each word once, so it is speech the committed turn does not contain.
    This happens when the speaker resumes between an eager flush and the commit (the
    commit reuses that flush, which predates the word). Opening a turn lets the user
    barge in on a reply to a turn that was cut short; dropping the word would lose it.
    Flush replies after a resume. A `flushed` for an eager flush only emits a preflight
    while still in EAGER for that same flush; after a resume it is ignored (the words it
    returned were already emitted as interims).
    Flushes in flight. Every flush gets a fresh id. The server answers in order, so a
    reply answers every older flush still pending. A manual flush and an eager flush can
    be in flight together; each gets its own answer. A manual flush is answered by the
    first `Final` emitted after it was sent: when the turn ends on its own before the
    reply (the commit reused an eager flush), the reply adds no second, empty `Final`.
    No words. A turn opened by `start_threshold` may end with no word: it emits
    `EndOfSpeech` without `Final` (the framework ignores empty finals anyway), except in
    answer to a manual flush, which always gets a `Final`.
    Duplicate preflight. A second EAGER entry whose words did not change emits no new
    preflight: the generation started for the first one is still valid, and another
    preflight would cancel and restart it, spending one of the framework's preemptive
    generation slots.
    Speech end time. The end time of the last word (`WordEnd.stop_time`) when known,
    else its start time; with no word, the audio time at which the pause began, else the
    last step's audio time.
"""

from __future__ import annotations

import enum
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from .profiles import TurnProfile
from .scorer import Scorer
from .signals import (
    STEP_DURATION,
    EndOfStream,
    Flushed,
    Ready,
    ServerError,
    Signal,
    Step,
    Word,
    WordEnd,
)

# Audio, in seconds, COMMITTED waits for its flush reply before emitting what it has.
# Measured replies take about 0.32 s; steps stall during a flush and then catch up, so
# this only fires when the server never answers.
FLUSH_REPLY_TIMEOUT = 1.5
# Flushes awaiting a reply. Only a server that never answers fills this; the oldest is
# then forgotten.
_MAX_PENDING_FLUSHES = 16
# Words of the next turn heard while COMMITTED waits for its flush reply.
_MAX_HELD_WORDS = 64
# Works around floating-point rounding. Audio times are multiples of the 80 ms step
# stored as floats, and so are most time limits, but the difference of two such times is
# not exact: 0.72 - 0.56 gives 0.15999999999999992, which is below a limit of 0.16.
# Without a margin, a limit would be reached one step (80 ms) late on some steps and on
# time on others. A duration within this margin of a limit counts as reaching it; the
# margin is far below one step, so it never makes a limit fire early.
_TIME_EPSILON = 1e-6


class TurnState(enum.Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    PAUSED = "paused"
    EAGER = "eager"
    COMMITTED = "committed"


@dataclass(frozen=True, slots=True)
class StartOfSpeech:
    """A turn opened. `start_time` is in connection audio time."""

    start_time: float


@dataclass(frozen=True, slots=True)
class Interim:
    """Every pending word of the turn, after a new word arrived."""

    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class Preflight:
    """Every pending word of the turn, which has probably ended."""

    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class Final:
    """Every pending word of the turn, now final. Empty only in answer to a manual flush."""

    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class EndOfSpeech:
    """The turn ended. `end_time` is the speech end time, in connection audio time."""

    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class SendFlush:
    """Send `{"type": "flush", "flush_id": flush_id}` to the server."""

    flush_id: int


Action: TypeAlias = StartOfSpeech | Interim | Preflight | Final | EndOfSpeech | SendFlush

_NO_ACTIONS: tuple[Action, ...] = ()


class _Purpose(enum.Enum):
    EAGER = "eager"
    COMMIT = "commit"
    MANUAL = "manual"


class _Word:
    __slots__ = ("text", "start", "stop")

    def __init__(self, text: str, start: float) -> None:
        self.text = text
        self.start = start
        self.stop: float | None = None

    @property
    def end(self) -> float:
        return self.stop if self.stop is not None else self.start


class TurnMachine:
    """Turn-taking decisions for one Gradium stream. See the module docstring."""

    def __init__(self, profile: TurnProfile, scorer: Scorer) -> None:
        self._profile = profile
        self._scorer = scorer
        self._state = TurnState.IDLE
        self._out: list[Action] = []

        # connection scope
        self._audio_time = 0.0
        self._steps_seen = 0
        self._last_score: float | None = None
        self._next_flush_id = 1
        # (flush id, purpose, audio time when sent), oldest first
        self._pending_flushes: deque[tuple[int, _Purpose, float]] = deque(
            maxlen=_MAX_PENDING_FLUSHES
        )
        # manual flushes already answered by a `Final` emitted after they were sent
        self._answered_manual: set[int] = set()
        self._last_word: _Word | None = None

        # turn scope
        self._words: list[_Word] = []
        self._held_words: list[_Word] = []
        self._turn_start = 0.0
        self._eager_count = 0
        self._preflight_text: str | None = None

        # pause scope
        self._pause_start = 0.0
        self._eager_start = 0.0
        self._confirm_count = 0
        self._pause_flush: int | None = None
        self._pause_flush_answered = False
        self._commit_time = 0.0

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def last_score(self) -> float | None:
        """Score of the last scored step; None during warm-up."""
        return self._last_score

    @property
    def audio_time(self) -> float:
        """Audio time of the last step on this connection."""
        return self._audio_time

    @property
    def scorer(self) -> Scorer:
        return self._scorer

    @property
    def profile(self) -> TurnProfile:
        return self._profile

    @profile.setter
    def profile(self, profile: TurnProfile) -> None:
        """Takes effect on the next step; the current state is kept."""
        self._profile = profile

    def set_scorer(self, scorer: Scorer) -> None:
        """Replace the scorer, for new weights or a new server model."""
        scorer.reset()
        self._scorer = scorer

    # inputs

    def on_signal(self, sig: Signal) -> Sequence[Action]:
        if isinstance(sig, Step):
            self._on_step(sig)
        elif isinstance(sig, Word):
            self._on_word(sig)
        elif isinstance(sig, WordEnd):
            if self._last_word is not None:
                self._last_word.stop = sig.stop_time
        elif isinstance(sig, Flushed):
            self._on_flushed(sig.flush_id)
        elif isinstance(sig, (EndOfStream, ServerError)):
            # the server has sent every word it will send on this connection
            self._finalize()
        elif isinstance(sig, Ready):
            if self._audio_time > 0 or self._last_word is not None:
                # a new connection, and the old one was never finalized: its audio clock
                # restarts at zero, so nothing open on it can continue
                self._finalize()
            else:
                # already finalized (possibly keeping the turn open): only the scorer's
                # window restarts with the clock
                self._scorer.reset()
        return self._take()

    def request_flush(self) -> Sequence[Action]:
        """The framework asked to finalise what was heard (`manual_flush=True`).

        Answered when the flush reply arrives: a `Final` with the pending words, empty
        when nothing was heard, then `EndOfSpeech` if a turn was open.
        """
        self._send_flush(_Purpose.MANUAL)
        return self._take()

    def finalize(self, *, keep_turn: bool = False) -> Sequence[Action]:
        """The connection is over: emit every buffered word.

        Called on end of input, a lost socket and a reconnect. Words the model delay
        still holds on the server are lost with the socket; every word received is
        emitted. A pending manual flush is answered here.

        Args:
            keep_turn: Another connection takes over (a reconnect or a retry). An open
                turn is not ended: its words are emitted as `Final` and the machine
                stays in SPEAKING, so the turn ends on the next connection. Without it,
                the machine returns to IDLE with `EndOfSpeech`.
        """
        self._finalize(keep_turn=keep_turn)
        return self._take()

    # step handling

    def _on_step(self, step: Step) -> None:
        self._audio_time = t = step.audio_time
        self._steps_seen += 1
        p = self._profile
        if self._steps_seen <= p.warmup_steps:
            # warm-up: the head has no context yet, its score is not used
            return
        self._last_score = score = self._scorer.on_step(step)
        state = self._state

        if state is TurnState.IDLE:
            if p.start_threshold is not None and score <= p.start_threshold:
                # IDLE -> SPEAKING: onset score, before any word
                self._open_turn(t - STEP_DURATION)
            return

        if state is TurnState.SPEAKING:
            if score >= p.pause_threshold:
                # SPEAKING -> PAUSED: the silence clock starts now
                self._state = TurnState.PAUSED
                self._pause_start = t
                self._confirm_count = 0
                self._pause_flush = None
                self._pause_flush_answered = False
            return

        if state is TurnState.COMMITTED:
            if t - self._commit_time + _TIME_EPSILON >= FLUSH_REPLY_TIMEOUT:
                # COMMITTED -> IDLE: the flush reply never came; emit what we have
                self._complete_turn()
            return

        # PAUSED or EAGER
        if score < p.resume_threshold:
            # PAUSED/EAGER -> SPEAKING: the speaker resumed
            self._resume()
            return
        silence = t - self._pause_start + _TIME_EPSILON
        if silence >= p.max_silence:
            # PAUSED/EAGER -> COMMITTED: `max_silence` ends the turn whatever the score
            self._commit()
            return

        if state is TurnState.PAUSED:
            if self._eager_allowed():
                if silence >= p.min_silence and score >= p.eager_threshold:
                    # PAUSED -> EAGER
                    self._enter_eager()
            elif self._confirmed(score, silence >= p.min_silence):
                # PAUSED -> COMMITTED: no eager entry allowed, the passive score confirms
                self._commit()
            return

        # EAGER
        if t - self._eager_start + _TIME_EPSILON >= p.confirm_timeout:
            # EAGER -> COMMITTED: `confirm_timeout`
            self._commit()
        elif self._confirmed(score, True):
            # EAGER -> COMMITTED: the post-flush score confirms
            self._commit()

    def _confirmed(self, score: float, eligible: bool) -> bool:
        if eligible and score >= self._profile.commit_threshold:
            self._confirm_count += 1
        else:
            self._confirm_count = 0
        return self._confirm_count >= self._profile.confirm_steps

    def _eager_allowed(self) -> bool:
        return self._profile.preflight and self._eager_count < self._profile.max_eager_per_turn

    # word handling

    def _on_word(self, sig: Word) -> None:
        self._scorer.on_word(sig, self._audio_time)
        text = sig.text.strip()
        if not text:
            return
        word = _Word(text, sig.start_time)
        self._last_word = word
        state = self._state

        if state is TurnState.IDLE:
            # IDLE -> SPEAKING: first word of a turn
            self._words.append(word)
            self._open_turn(word.start)
            self._emit_interim()
            return

        if state is TurnState.COMMITTED:
            if word.start < self._commit_time:
                # a delayed word of the committed turn, returned by its flush
                self._words.append(word)
                self._emit_interim()
            elif len(self._held_words) < _MAX_HELD_WORDS:
                # the next turn started while the flush reply is pending
                self._held_words.append(word)
            return

        self._words.append(word)
        self._emit_interim()
        if state is not TurnState.SPEAKING and word.start >= self._pause_start:
            # PAUSED/EAGER -> SPEAKING: a word spoken after the pause began; earlier
            # words are the delayed tail of the speech before the pause
            self._resume()

    # flush handling

    def _send_flush(self, purpose: _Purpose) -> int:
        flush_id = self._next_flush_id
        self._next_flush_id += 1
        self._pending_flushes.append((flush_id, purpose, self._audio_time))
        self._scorer.on_flush()
        self._out.append(SendFlush(flush_id))
        return flush_id

    def _on_flushed(self, flush_id: int) -> None:
        # replies come in order: this one also answers every older pending flush
        while self._pending_flushes and self._pending_flushes[0][0] <= flush_id:
            pending_id, purpose, sent_at = self._pending_flushes.popleft()
            if purpose is _Purpose.MANUAL:
                if pending_id in self._answered_manual:
                    self._answered_manual.discard(pending_id)
                else:
                    self._answer_manual_flush(sent_at)
            elif pending_id == self._pause_flush:
                self._pause_flush_answered = True
                if self._state is TurnState.EAGER:
                    self._emit_preflight()
                elif self._state is TurnState.COMMITTED:
                    # COMMITTED -> IDLE: every held word of the turn is in
                    self._complete_turn()
            # else: a flush from a pause the speaker already resumed from; its words
            # were emitted as interims when they arrived

    def _answer_manual_flush(self, sent_at: float) -> None:
        if self._state is TurnState.IDLE:
            self._out.append(Final("", sent_at, sent_at))
            return
        # any open state -> IDLE: the caller ends the turn
        self._emit_turn_end(final_always=True)
        self._end_turn()

    # transitions

    def _open_turn(self, start: float) -> None:
        self._state = TurnState.SPEAKING
        self._turn_start = start
        self._out.append(StartOfSpeech(start))

    def _resume(self) -> None:
        self._state = TurnState.SPEAKING
        self._confirm_count = 0
        # a flush from this pause no longer covers the words to come
        self._pause_flush = None
        self._pause_flush_answered = False

    def _enter_eager(self) -> None:
        self._state = TurnState.EAGER
        self._eager_start = self._audio_time
        self._eager_count += 1
        self._confirm_count = 0
        self._pause_flush = self._send_flush(_Purpose.EAGER)
        self._pause_flush_answered = False

    def _commit(self) -> None:
        self._state = TurnState.COMMITTED
        self._commit_time = self._audio_time
        if self._pause_flush is None:
            self._pause_flush = self._send_flush(_Purpose.COMMIT)
            self._pause_flush_answered = False
        elif self._pause_flush_answered:
            # COMMITTED -> IDLE at once: the eager flush already returned the words
            self._complete_turn()

    def _complete_turn(self) -> None:
        self._emit_turn_end(final_always=False)
        self._end_turn()

    def _end_turn(self) -> None:
        self._scorer.on_turn_end()
        self._state = TurnState.IDLE
        self._words = []
        self._eager_count = 0
        self._preflight_text = None
        self._confirm_count = 0
        self._pause_flush = None
        self._pause_flush_answered = False
        if self._held_words:
            # IDLE -> SPEAKING: the next turn began while the last one was committing
            self._words, self._held_words = self._held_words, []
            self._open_turn(self._words[0].start)
            self._emit_interim()

    def _finalize(self, *, keep_turn: bool = False) -> None:
        manual_pending = any(
            p is _Purpose.MANUAL and i not in self._answered_manual
            for i, p, _ in self._pending_flushes
        )
        while self._state is not TurnState.IDLE:
            if keep_turn and self._state is not TurnState.COMMITTED:
                # open -> SPEAKING or PAUSED: hand the turn to the next connection
                text = self._text()
                if text or manual_pending:
                    self._out.append(Final(text, self._start_time(), self._speech_end()))
                manual_pending = False
                paused = self._state in (TurnState.PAUSED, TurnState.EAGER)
                self._words = []
                self._preflight_text = None
                self._confirm_count = 0
                self._pause_flush = None
                self._pause_flush_answered = False
                self._turn_start = 0.0
                # A speaker who was silent stays silent: the pause goes on from the new
                # connection's first audio (its time 0), which is also the speech end if
                # no word follows. Restarting in SPEAKING would wait for the new
                # connection's score to find the pause again, seconds later.
                self._state = TurnState.PAUSED if paused else TurnState.SPEAKING
                self._pause_start = 0.0
                break
            # any -> IDLE; held words open one more turn, closed by the next iteration
            self._emit_turn_end(final_always=manual_pending)
            manual_pending = False
            self._end_turn()
        if manual_pending:
            self._out.append(Final("", self._audio_time, self._audio_time))
        self._pending_flushes.clear()
        self._answered_manual.clear()
        self._audio_time = 0.0
        self._steps_seen = 0
        self._last_score = None
        self._last_word = None
        self._scorer.reset()

    # output

    def _text(self) -> str:
        return " ".join(w.text for w in self._words)

    def _start_time(self) -> float:
        return self._words[0].start if self._words else self._turn_start

    def _speech_end(self) -> float:
        if self._words:
            return self._words[-1].end
        if self._state in (TurnState.PAUSED, TurnState.EAGER, TurnState.COMMITTED):
            return self._pause_start
        return self._audio_time

    def _emit_interim(self) -> None:
        self._out.append(Interim(self._text(), self._start_time(), self._words[-1].end))

    def _emit_preflight(self) -> None:
        text = self._text()
        if not text or text == self._preflight_text:
            return
        self._preflight_text = text
        self._out.append(Preflight(text, self._start_time(), self._speech_end()))

    def _emit_turn_end(self, *, final_always: bool) -> None:
        text = self._text()
        start, end = self._start_time(), self._speech_end()
        if text or final_always:
            self._out.append(Final(text, start, end))
            # this final answers every manual flush sent before it
            self._answered_manual.update(
                i for i, p, _ in self._pending_flushes if p is _Purpose.MANUAL
            )
        self._out.append(EndOfSpeech(text, start, end))

    def _take(self) -> Sequence[Action]:
        if not self._out:
            return _NO_ACTIONS
        out, self._out = self._out, []
        return out
