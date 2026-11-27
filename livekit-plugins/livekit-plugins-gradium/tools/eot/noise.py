"""Background noise for the noisy-preset replays: pink noise and multi-talker babble.

eot-bench has no noisy audio, so the `noisy` presets are measured on dataset turns with
noise mixed in at a fixed speech-to-noise ratio. Everything is seeded by the turn id, so
a turn gets the same noise on every run.

- `pink`: 1/f noise, generated in the frequency domain.
- `babble`: six talkers summed, each a chain of other dataset turns of the same
  language (never one of the turns being replayed). Six talkers is dense enough that no
  single voice is intelligible, which is the usual definition of babble.

SNR is measured against the speech only: the RMS of the turn outside its annotated
silence spans, against the RMS of the noise segment. The noise also fills the pauses,
which is the point: it is what the inactivity head has to see through.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from typing import Any

import numpy as np

KINDS = ("pink", "babble")
BABBLE_TALKERS = 6
_BABBLE_POOL = 36
_BED_SECONDS = 180.0


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def pink(n: int, seed: int) -> np.ndarray:
    """Unit-RMS pink noise of `n` samples."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spectrum), dtype=np.float64)
    f[0] = 1.0
    out = np.fft.irfft(spectrum / np.sqrt(f), n)
    out -= out.mean()
    return (out / (np.sqrt(np.mean(out**2)) + 1e-12)).astype(np.float32)


def speech_rms(audio: np.ndarray, spans: Iterable[dict[str, Any]], sample_rate: int) -> float:
    """RMS of `audio` outside the annotated silence spans."""
    mask = np.ones(len(audio), dtype=bool)
    for span in spans:
        a = int(float(span["start"]) * sample_rate)
        b = int(float(span["end"]) * sample_rate)
        mask[max(a, 0) : max(b, 0)] = False
    voiced = audio[mask] if mask.any() else audio
    return float(np.sqrt(np.mean(voiced.astype(np.float64) ** 2)))


class BabbleBed:
    """A long multi-talker recording built from a pool of turns, cut per turn."""

    def __init__(self, pool: list[np.ndarray], sample_rate: int, *, seed: int) -> None:
        if not pool:
            raise ValueError("babble needs at least one source turn")
        n = int(_BED_SECONDS * sample_rate)
        bed = np.zeros(n, dtype=np.float64)
        for talker in range(BABBLE_TALKERS):
            rng = random.Random(seed + talker)
            chain: list[np.ndarray] = []
            total = 0
            while total < n:
                clip = pool[rng.randrange(len(pool))]
                clip = clip / (np.sqrt(np.mean(clip.astype(np.float64) ** 2)) + 1e-12)
                chain.append(clip)
                total += len(clip)
            bed += np.concatenate(chain)[:n]
        bed -= bed.mean()
        self._bed = (bed / (np.sqrt(np.mean(bed**2)) + 1e-12)).astype(np.float32)

    def segment(self, n: int, seed: int) -> np.ndarray:
        """Unit-RMS babble of `n` samples at a position chosen by `seed`."""
        bed = self._bed
        if n >= len(bed):
            reps = n // len(bed) + 1
            return np.tile(bed, reps)[:n]
        start = random.Random(seed).randrange(len(bed) - n)
        seg = bed[start : start + n].astype(np.float64)
        return (seg / (np.sqrt(np.mean(seg**2)) + 1e-12)).astype(np.float32)


def mix(
    audio: np.ndarray,
    spans: Iterable[dict[str, Any]],
    *,
    noise: np.ndarray,
    snr_db: float,
    sample_rate: int,
) -> np.ndarray:
    """`audio` plus unit-RMS `noise` scaled to `snr_db` below the speech, clipped to [-1, 1]."""
    level = speech_rms(audio, spans, sample_rate) / (10.0 ** (snr_db / 20.0))
    return np.clip(audio + level * noise[: len(audio)], -1.0, 1.0).astype(np.float32)


def noise_for(kind: str, turn_id: str, n: int, *, babble: BabbleBed | None = None) -> np.ndarray:
    if kind == "pink":
        return pink(n, _seed("pink", turn_id))
    if kind == "babble":
        if babble is None:
            raise ValueError("babble noise needs a BabbleBed")
        return babble.segment(n, _seed("babble", turn_id))
    raise ValueError(f"unknown noise {kind!r}, expected one of {KINDS}")


def pool_size() -> int:
    return _BABBLE_POOL


def bed_seed(lang: str) -> int:
    return _seed("bed", lang)
