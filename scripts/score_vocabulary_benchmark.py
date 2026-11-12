"""Score Kev native rows against every scheduled word; no best-of-retries merging."""

import argparse
from pathlib import Path

from lev.io import digest, jsonl, read, write
from lev.vocabulary import load_words, score_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--rows",
        type=Path,
        required=True,
        help="Kev native rows.json from one fixed evaluation",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    words = load_words(args.dataset)
    result = score_words(
        words, jsonl(args.dataset / "records.jsonl"), read(args.rows), args.threshold
    )
    result["dataset_manifest_sha256"] = digest(args.dataset / "manifest.json")
    result["native_rows_sha256"] = digest(args.rows)
    write(args.out, result)
    print(result["status"], result["overall"])
