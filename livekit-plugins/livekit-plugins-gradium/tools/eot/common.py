"""Paths, dataset access and the replay loop shared by the end-of-turn tools.

Run every tool from the repository root with the overlay environment described in
README.md, so that `livekit.plugins.gradium.experimental` resolves to the code in this
checkout. Downloaded parquet files are untrusted data: they live under `data/hf/` and are
only ever opened by path.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from livekit.plugins.gradium.experimental.signals import Step, Word, parse

TOOLS_DIR = Path(__file__).resolve().parent
DATA_DIR = TOOLS_DIR / "data"
HF_DIR = DATA_DIR / "hf"
RAW_DIR = DATA_DIR / "raw"
WEIGHTS_DIR = TOOLS_DIR.parents[1] / "livekit" / "plugins" / "gradium" / "experimental" / "weights"
FIXTURE_PATH = TOOLS_DIR.parents[3] / "tests" / "gradium_scorer_steps.json"

DATASET = "livekit/eot-bench-data"
REVISION = "ae834af9e9bddf73d1f550c190aaeb99bf31f523"
LANGUAGES = ("ar", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "nl", "pt", "tr", "zh")
# Client-side message recorded in the stream at the point a flush was sent.
FLUSH_SENT = "flush_sent"


def default_bench_path() -> Path:
    return Path(os.environ.get("EOT_BENCH", Path.home() / "git" / "oss" / "eot-bench"))


def import_harness(bench: Path | None = None) -> None:
    """Put an eot-bench checkout on the path so `eot_harness` is the one being measured."""
    path = str(bench or default_bench_path())
    if path not in sys.path:
        sys.path.insert(0, path)


def load_api_key(env_file: str = "~/.bench.env") -> str:
    """Return GRADIUM_API_KEY from the environment or `env_file`. Never printed."""
    key = os.environ.get("GRADIUM_API_KEY")
    if key:
        return key
    for line in Path(env_file).expanduser().read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == "GRADIUM_API_KEY":
            return value.strip().strip("\"'")
    raise RuntimeError(f"GRADIUM_API_KEY is not set and not found in {env_file}")


def parquet_files(lang: str) -> list[Path]:
    files = sorted((HF_DIR / "data" / lang).glob("validation-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet for {lang!r} under {HF_DIR}; run download.py")
    return files


def iter_rows(lang: str, columns: list[str]) -> Iterator[dict[str, Any]]:
    """Dataset rows of one language, reading only `columns` (audio is large)."""
    import pyarrow.parquet as pq

    for path in parquet_files(lang):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def raw_path(tag: str, lang: str) -> Path:
    """Recording of one language; `tag` is the collection mode or the `--tag` given."""
    return RAW_DIR / tag / f"{lang}.jsonl"


def read_records(tag: str, langs: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Recorded streams, one dict per turn (see collect.py for the format)."""
    for lang in langs:
        path = raw_path(tag, lang)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run collect.py (tag {tag!r}, {lang})")
        with path.open() as f:
            for line in f:
                yield json.loads(line)


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """`["commit_threshold=0.7", "preflight=false"]` to turn knob overrides."""
    out: dict[str, Any] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"expected KNOB=VALUE, got {item!r}")
        low = raw.strip().lower()
        value: Any
        if low in ("none", "null"):
            value = None
        elif low in ("true", "false"):
            value = low == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[name.strip()] = value
    return out


class Driver:
    """Feeds one recorded or live message stream into a scorer, in stream order.

    This is the only translation from Gradium messages to scorer calls in the tools, used
    by the collector (to decide when to flush), by training (to extract features) and by
    evaluation (to score), so all three see what the plugin sees.
    """

    def __init__(self, scorer: Any, on_step: Callable[[Step], None] | None = None) -> None:
        self.scorer = scorer
        self._on_step = on_step
        self.audio_time = 0.0
        self.score = 0.0

    def feed(self, msg: dict[str, Any]) -> Step | None:
        """Apply one message; return the step if it was one (its score is `self.score`)."""
        if msg.get("type") == FLUSH_SENT:
            self.scorer.on_flush()
            return None
        sig = parse(msg)
        if isinstance(sig, Step):
            self.audio_time = sig.audio_time
            self.score = self.scorer.on_step(sig)
            if self._on_step is not None:
                self._on_step(sig)
            return sig
        if isinstance(sig, Word):
            self.scorer.on_word(sig, self.audio_time)
        return None
