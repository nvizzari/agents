"""End-of-turn score: one number in [0, 1] per step, higher when the turn has ended.

The scorer only sees what the stream carries (steps, words, the fact that a flush was
sent), so the same object runs live, in the offline simulator, in the benchmark adapter
and in the training tools (`tools/eot/`), which extract their features by driving
`FeatureWindow` over recorded streams. Live and training features therefore cannot drift.
This module has no network I/O.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from ..log import logger
from .signals import HORIZONS, Step, Word


class Scorer(Protocol):
    """Contract between the scorer and the state machine.

    Calls arrive in stream order. `on_step` is the only call that returns a score.
    """

    @property
    def name(self) -> str:
        """Identifies the scorer and its weights in logs and benchmark manifests."""
        ...

    def reset(self) -> None:
        """Forget everything: a new connection restarts the audio clock at zero."""
        ...

    def on_step(self, step: Step) -> float:
        """Consume one step and return the end-of-turn score after it."""
        ...

    def on_word(self, word: Word, audio_time: float) -> None:
        """Record a word. `audio_time` is that of the last step received before it."""
        ...

    def on_flush(self) -> None:
        """A flush was sent: later steps read higher until the speaker resumes."""
        ...

    def on_turn_end(self) -> None:
        """The turn was committed: per-turn context (words, pauses) starts over."""
        ...


class FallbackScorer:
    """`min(p2, p3)`: the 2 s and 3 s horizons must both agree.

    Used when no weights are available. It is the best hand-written fusion measured on
    eot-bench (978 ms mean latency at a 5% false cut-off budget, English).
    """

    name = "fallback-min-p2-p3"

    def reset(self) -> None:
        pass

    def on_step(self, step: Step) -> float:
        return min(step.probs[2], step.probs[3])

    def on_word(self, word: Word, audio_time: float) -> None:
        pass

    def on_flush(self) -> None:
        pass

    def on_turn_end(self) -> None:
        pass


FORMAT_VERSION = 1
"""Version of the weights file layout. Files of any other version are rejected."""

DEFAULT_WEIGHTS = "default"
"""Bundled weight set used when the caller does not choose one."""

WEIGHTS_DIR = Path(__file__).parent / "weights"

_LAGS = (1, 2, 3, 5)
_WINDOW = 5
# Values used when there is no word yet; also the cap on the word timing features, so a
# long-lived stream never produces values outside the range seen in training.
_NO_WORD_AGE = 10.0


def _feature_names() -> tuple[str, ...]:
    names = [f"p{h}" for h in range(len(HORIZONS))]
    names += [f"p{h}_l{lag}" for lag in _LAGS for h in range(len(HORIZONS))]
    names += [f"p{h}_{agg}{_WINDOW}" for agg in ("min", "mean") for h in range(len(HORIZONS))]
    names += ["since_word", "word_start_age", "n_words"]
    names += ["steps_since_flush"]
    return tuple(names)


FEATURES: tuple[str, ...] = _feature_names()
"""Every feature `FeatureWindow` computes, in vector order. A weight set uses a subset."""

_ZERO = (0.0,) * len(HORIZONS)


class FeatureWindow:
    """Causal feature vector over the stream, recomputed at every step.

    Features, for horizon `h` (0.5, 1, 2, 3 s):

    - `p{h}`: the step's inactivity probability; `p{h}_l{k}` the same `k` steps earlier
      (0 before the stream has that many steps); `p{h}_min5` / `p{h}_mean5` over the
      last 5 steps.
    - `since_word`: audio time since the last word arrived; `word_start_age`: since that
      word started; both capped at 10 s, and 10 s when the turn has no word yet.
    - `n_words`: words in the turn.
    - `steps_since_flush`: steps since the last flush, 0 when none is in effect. A flush
      stops being in effect when a word starting after it arrives (the speaker resumed)
      or the turn ends.

    A count of pauses in the turn (gaps between word starts) and the turn's age were
    tried and dropped: trained on other languages, they cost 50 ms on English at the 5%
    budget.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._probs: deque[tuple[float, ...]] = deque(maxlen=max(_LAGS) + 1)
        self._audio_time = 0.0
        self._flush_time: float | None = None
        self._steps_since_flush = 0
        self._n_words = 0
        self._last_arrival = 0.0
        self._last_start = 0.0
        self.vector: list[float] = [0.0] * len(FEATURES)

    @property
    def flushed(self) -> bool:
        """A flush was sent in the current pause and the speaker has not resumed."""
        return self._flush_time is not None

    def on_turn_end(self) -> None:
        self._flush_time = None
        self._steps_since_flush = 0
        self._n_words = 0

    def on_flush(self) -> None:
        self._flush_time = self._audio_time
        self._steps_since_flush = 0

    def on_word(self, word: Word, audio_time: float) -> None:
        if self._flush_time is not None and word.start_time > self._flush_time:
            self._flush_time = None
            self._steps_since_flush = 0
        self._n_words += 1
        self._last_arrival = audio_time
        self._last_start = word.start_time

    def on_step(self, step: Step) -> list[float]:
        """Consume a step and return the feature vector after it (in `FEATURES` order)."""
        self._audio_time = t = step.audio_time
        probs = self._probs
        probs.append(step.probs)
        if self._flush_time is not None:
            self._steps_since_flush += 1

        v = list(step.probs)
        n = len(probs)
        for lag in _LAGS:
            v.extend(probs[n - 1 - lag] if n > lag else _ZERO)
        recent = list(probs)[-_WINDOW:]
        cols = list(zip(*recent, strict=True))
        v.extend(min(c) for c in cols)
        v.extend(sum(c) / len(c) for c in cols)
        if self._n_words:
            v.append(min(t - self._last_arrival, _NO_WORD_AGE))
            v.append(min(t - self._last_start, _NO_WORD_AGE))
            v.append(float(self._n_words))
        else:
            v.extend((_NO_WORD_AGE, _NO_WORD_AGE, 0.0))
        v.append(float(self._steps_since_flush))
        self.vector = v
        return v


@dataclass(frozen=True)
class _Tree:
    feature: tuple[int, ...]
    threshold: tuple[float, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    value: tuple[float, ...]


class _Forest:
    """Gradient-boosted trees: sigmoid(baseline + sum of leaf values).

    A sample goes left when its feature is <= the node threshold, which is how
    scikit-learn's `HistGradientBoostingClassifier` predicts on raw (unbinned) values.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        names = spec["features"]
        if not isinstance(names, list) or not names:
            raise ValueError("a weight set needs a non-empty feature list")
        unknown = [n for n in names if n not in FEATURES]
        if unknown:
            raise ValueError(f"unknown features {unknown}")
        self.features: tuple[str, ...] = tuple(names)
        self.columns: tuple[int, ...] = tuple(FEATURES.index(n) for n in names)
        self.baseline = _finite(spec["baseline"])
        self.trees = tuple(_parse_tree(t, len(names)) for t in spec["trees"])
        if not self.trees:
            raise ValueError("a weight set needs at least one tree")

    def predict(self, vector: list[float]) -> float:
        x = [vector[i] for i in self.columns]
        z = self.baseline
        for tree in self.trees:
            left, feature, threshold = tree.left, tree.feature, tree.threshold
            node = 0
            while left[node] >= 0:
                node = left[node] if x[feature[node]] <= threshold[node] else tree.right[node]
            z += tree.value[node]
        # clamp so exp() cannot overflow on a pathological sum
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))


def _finite(v: Any) -> float:
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"non-finite value {v!r}")
    return f


def _parse_tree(spec: dict[str, Any], n_features: int) -> _Tree:
    tree = _Tree(
        feature=tuple(int(i) for i in spec["feature"]),
        threshold=tuple(float(v) for v in spec["threshold"]),
        left=tuple(int(i) for i in spec["left"]),
        right=tuple(int(i) for i in spec["right"]),
        value=tuple(_finite(v) for v in spec["value"]),
    )
    n = len(tree.feature)
    if n == 0 or any(len(a) != n for a in (tree.threshold, tree.left, tree.right, tree.value)):
        raise ValueError("tree arrays are empty or of unequal length")
    for i in range(n):
        if tree.left[i] < 0:
            if tree.right[i] >= 0:
                raise ValueError(f"node {i} has a right child but no left child")
            continue
        # children after their parent: every walk terminates
        if not (i < tree.left[i] < n and i < tree.right[i] < n):
            raise ValueError(f"node {i} has children out of range")
        if not 0 <= tree.feature[i] < n_features:
            raise ValueError(f"node {i} splits on feature {tree.feature[i]} out of range")
        if math.isnan(tree.threshold[i]):
            raise ValueError(f"node {i} has a NaN threshold")
    return tree


@dataclass(frozen=True)
class Weights:
    """A validated weights file. Immutable, shared between scorers."""

    name: str
    model: str
    """Server model version (`model_ext`) the probabilities were recorded with."""
    delay_in_frames: int
    """Model delay the training streams were recorded at. The word-timing features
    depend on it, so the weights only apply to a server running the same delay."""
    passive: _Forest
    """Steps with no flush in effect."""
    post_flush: _Forest
    """Steps after a flush, until the speaker resumes."""
    meta: dict[str, Any]
    """Everything else in the file: training data description, hyper-parameters."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Weights:
        if not isinstance(data, dict):
            raise ValueError("weights file must hold a JSON object")
        version = data.get("format")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported weights format {version!r} (expected {FORMAT_VERSION})")
        sets = data["sets"]
        delay = data["delay_in_frames"]
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValueError(f"delay_in_frames must be a non-negative integer, got {delay!r}")
        return cls(
            name=str(data["name"]),
            model=str(data["model"]),
            delay_in_frames=delay,
            passive=_Forest(sets["passive"]),
            post_flush=_Forest(sets["post_flush"]),
            meta={k: v for k, v in data.items() if k != "sets"},
        )


class LearnedScorer:
    """Gradient-boosted trees over `FeatureWindow`, one forest per flush state."""

    def __init__(self, weights: Weights) -> None:
        self._weights = weights
        self._window = FeatureWindow()

    @property
    def name(self) -> str:
        return self._weights.name

    @property
    def weights(self) -> Weights:
        return self._weights

    @property
    def flushed(self) -> bool:
        """The `post_flush` weights are in effect: a flush was sent and no word followed."""
        return self._window.flushed

    def reset(self) -> None:
        self._window.reset()

    def on_step(self, step: Step) -> float:
        vector = self._window.on_step(step)
        w = self._weights
        return (w.post_flush if self._window.flushed else w.passive).predict(vector)

    def on_word(self, word: Word, audio_time: float) -> None:
        self._window.on_word(word, audio_time)

    def on_flush(self) -> None:
        self._window.on_flush()

    def on_turn_end(self) -> None:
        self._window.on_turn_end()


_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def weights_path(weights: str | Path) -> Path:
    """Resolve a bundled weight set name (`"default"`) or a path to a weights file."""
    if isinstance(weights, str) and _NAME_RE.fullmatch(weights):
        return WEIGHTS_DIR / f"{weights}.json"
    return Path(weights)


@lru_cache(maxsize=8)
def _read_weights(path: Path, mtime_ns: int) -> Weights:
    # keyed on mtime so a rewritten file is read again; failures are not cached
    with path.open("rb") as f:
        return Weights.from_dict(json.load(f))


def load_weights(weights: str | Path) -> Weights:
    """Read and validate a weights file.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is not valid JSON, has another format version, lacks the
            training delay, or holds a malformed weight set.
    """
    path = weights_path(weights).resolve()
    try:
        return _read_weights(path, path.stat().st_mtime_ns)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"malformed weights file: {e!r}") from e


def load_scorer(
    weights: str | Path | None, *, model: str | None = None, delay_in_frames: int | None = None
) -> Scorer:
    """Return the learned scorer for `weights`, or the fallback when it cannot be used.

    Args:
        weights: Name of a bundled weight set, a path to a weights file, or None to
            force the fallback.
        model: Server model version from the `ready` message. Weights trained for
            another version are rejected, since the probabilities they read may have
            shifted.
        delay_in_frames: Model delay reported by the server. The weights must have
            been trained at this delay for their word-timing features to match.
    """
    if weights is None:
        return FallbackScorer()
    try:
        loaded = load_weights(weights)
    except (OSError, ValueError) as e:
        logger.warning(
            "Gradium scorer weights %r are unusable (%s), using the min(p2, p3) fallback",
            str(weights),
            e,
        )
        return FallbackScorer()
    if model is not None and loaded.model != model:
        logger.warning(
            "Gradium scorer weights %r were trained for model %r but the server runs %r, "
            "using the min(p2, p3) fallback",
            loaded.name,
            loaded.model,
            model,
        )
        return FallbackScorer()
    if delay_in_frames is not None and loaded.delay_in_frames != delay_in_frames:
        logger.warning(
            "Gradium scorer weights %r were trained with delay_in_frames=%d but the server "
            "reports %d, using the min(p2, p3) fallback",
            loaded.name,
            loaded.delay_in_frames,
            delay_in_frames,
        )
        return FallbackScorer()
    return LearnedScorer(loaded)
