"""Derive a benchmark-only Kaggle notebook from the training notebook.

Training produced the checkpoint, but its `job.json` predates the config
`threshold` field, so the run records no threshold at all. Scoring again from
the saved checkpoint with the current config gives a run whose own `job.json`
carries the declared threshold, which is what pre-registration has to look like.

The header, imports, pack discovery and the clone/install cells are reused
verbatim from the training notebook so the two stay in step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REUSED_CELLS = 4  # header, imports, pack discovery, clone/install


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="notebooks/kaggle-encoder-en-joint-010.ipynb")
    parser.add_argument("--out", default="notebooks/kaggle-encoder-benchmark-004.ipynb")
    parser.add_argument("--dev-run", default="enc-dev-005")
    parser.add_argument("--threshold", default="0.39")
    args = parser.parse_args()

    try:
        source = json.loads(Path(args.source).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read the source notebook {args.source}: {error}") from error
    cells = source["cells"][:REUSED_CELLS]

    # Point the reused constants at the benchmark run instead of a training run.
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        cell["source"] = [
            line
            for line in cell["source"]
            if not line.startswith(("TRAIN_RUN =", "DEV_RUN ="))
        ]
    # The reused cell may not end with a newline, so guard before appending or
    # the constant glues onto the previous statement and breaks the notebook.
    if cells[2]["source"] and not cells[2]["source"][-1].endswith("\n"):
        cells[2]["source"][-1] += "\n"
    cells[2]["source"].append(f'DEV_RUN = "{args.dev_run}"\n')

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Score development with the declared threshold\n",
                "\n",
                f"This run re-scores the saved checkpoint at the pre-registered "
                f"`threshold = {args.threshold}` from the config, so its own "
                "`job.json` records the threshold that produced its decisions.\n",
                "\n",
                "`lev encode-benchmark` never trains and never overwrites: a "
                "re-run needs a new `DEV_RUN`.\n",
            ],
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "CHECKPOINT = next(\n",
                '    p.parent for p in Path("/kaggle/input").rglob("model.safetensors")\n',
                ")\n",
                'print("checkpoint:", CHECKPOINT)\n',
                "\n",
                "lev(\n",
                '    "encode-benchmark",\n',
                '    "--dataset", str(DATASET),\n',
                '    "--config", CONFIG,\n',
                '    "--checkpoint", str(CHECKPOINT),\n',
                '    "--split", "development",\n',
                '    "--out", str(WORK / DEV_RUN),\n',
                '    "--device", "cuda",\n',
                '    "--execute",\n',
                ")\n",
                "\n",
                'job = json.loads((WORK / DEV_RUN / "job.json").read_text())\n',
                'print("job.json config.threshold:", job["config"].get("threshold"))\n',
                'print("job.json config_sha256:", job["config_sha256"])\n',
            ],
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "lev(\n",
                '    "report",\n',
                '    "--benchmark", str(WORK / DEV_RUN),\n',
                f'    "--threshold", "{args.threshold}",\n',
                f'    "--out", str(WORK / DEV_RUN / "lev-report-t{args.threshold}.json"),\n',
                ")\n",
                "\n",
                f'report = json.loads((WORK / DEV_RUN / "lev-report-t{args.threshold}.json").read_text())\n',
                'print("threshold:", report["threshold"])\n',
                'd = report["decisions"]\n',
                "print(\n",
                '    f"accuracy_all_requested {d[\'accuracy_all_requested\'][\'rate\']:.2%} "\n',
                '    f"({d[\'accuracy_all_requested\'][\'numerator\']}/{d[\'accuracy_all_requested\'][\'denominator\']})"\n',
                ")\n",
                'print("completed", d["completed"], "missing", d["missing"])\n',
                'for kind, block in report["by_kind"].items():\n',
                '    a = block["accuracy_all_requested"]\n',
                '    print(f"  {kind}: {a[\'numerator\']}/{a[\'denominator\']}")\n',
            ],
        }
    )

    Path(args.out).write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": source["metadata"],
                "nbformat": source["nbformat"],
                "nbformat_minor": source["nbformat_minor"],
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {args.out} with {len(cells)} cells")


if __name__ == "__main__":
    main()
