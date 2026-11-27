"""Download the eot-bench validation parquet files at the pinned revision.

python tools/eot/download.py [--lang en fr ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATASET, HF_DIR, LANGUAGES, REVISION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"data/{lang}/validation-*.parquet" for lang in args.lang],
        local_dir=HF_DIR,
    )
    print(f"downloaded {', '.join(args.lang)} to {HF_DIR}")


if __name__ == "__main__":
    main()
