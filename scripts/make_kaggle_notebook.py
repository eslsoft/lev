"""Emit a self-contained Kaggle notebook for the encoder decision model.

    python scripts/make_kaggle_notebook.py --out notebooks/kaggle-encoder-smoke.ipynb

The notebook clones this repository, installs it without re-resolving torch,
runs `lev encode` / `lev encode-benchmark` / `lev report` against a frozen pack
mounted as a Kaggle Dataset, and prints the development report. It runs the real
CLI, so the evidence it leaves behind is the same `job.json` / `status.json` /
`run.log` a local run produces.
"""

import argparse
import json
from pathlib import Path

REPO = "https://github.com/eslsoft/lev.git"
BRANCH = "experiment/plan-a-modernbert"


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def cells(dataset, config, train_run, dev_run):
    return [
        markdown(
            """
# Lev encoder decision model — Kaggle smoke run

Trains a **ModernBERT** backbone with a two-way `false / true` head on a frozen
Lev pack, then scores it with the unchanged `lev report`.

## Before you run

1. **Settings → Internet → On.** The backbone is pulled from the Hub by commit SHA.
2. **Settings → Accelerator → GPU T4 x2** (or P100). One card is enough; the
   second is unused because this path is single-device.
3. **The frozen pack is already attached** as the `saltbo/lev-en-joint-002`
   Dataset, mounted at `/kaggle/input/lev-en-joint-002`. If you swap packs,
   update `DATASET` below and add the new source to the kernel metadata. The
   pack must contain `manifest.json`, `rubric.json`, `records.jsonl` and the
   four split files; the first cell refuses anything else.

## What this does and does not prove

It proves the encoder path trains, reloads and scores on this hardware, and what
the development accuracy and wall-clock actually are. It does **not** replace a
test-set reading, an independent calibration, or a release decision: pick and
freeze the model on development, then read test once.
"""
        ),
        code(
            """
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

print(subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
    capture_output=True, text=True,
).stdout if shutil.which("nvidia-smi") else
      "no nvidia-smi on PATH: this session has no GPU accelerator")
"""
        ),
        code(
            f"""
REPO = "{REPO}"
BRANCH = "{BRANCH}"
CONFIG = "{config}"
TRAIN_RUN = "{train_run}"
DEV_RUN = "{dev_run}"

WORK = Path("/kaggle/working")
DATASET = Path("{dataset}")
SRC = WORK / "lev"

required = {{"manifest.json", "rubric.json", "records.jsonl"}} | {{
    f"{{split}}.jsonl" for split in ("train", "development", "calibration", "test")
}}
missing = sorted(required - {{p.name for p in DATASET.glob("*")}})
if missing:
    root = Path("/kaggle/input")
    seen = sorted(p.name for p in root.glob("*")) if root.is_dir() else []
    raise SystemExit(
        f"{{DATASET}} is not a frozen pack; missing {{missing}}.\\n"
        f"/kaggle/input holds: {{seen}}\\n"
        "A dataset still processing mounts as an empty directory; wait for "
        "`kaggle datasets status` to report ready, then re-run."
    )
print("frozen pack:", DATASET, "|", len(list(DATASET.glob("*"))), "files")
print("output:", WORK)
"""
        ),
        code(
            """
if SRC.exists():
    shutil.rmtree(SRC)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(SRC)], check=True
)

# --no-deps keeps Kaggle's own torch; only pull the extra if the import fails.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(SRC)],
    check=True,
)

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(SRC / "src")
# Bucketed batches keep padding small; this keeps the allocator from fragmenting.
ENV.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def lev(*args):
    print("$ lev", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", "lev", *args], cwd=SRC, env=ENV, check=True)


lev("check-upstream", "--help")  # cheap import check for the CLI itself

try:
    import torch
    import transformers
    if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 48):
        raise ImportError(f"transformers {transformers.__version__} predates ModernBERT")
except ImportError as error:
    print(f"installing the encoder extra: {error}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"{SRC}[encoder]"], check=True
    )
    import torch
    import transformers

print("torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA device. Set Settings -> Accelerator -> GPU T4 x2 on this notebook "
        "and re-run; training 15,515 pairs on CPU would take hours."
    )
if torch.cuda.is_available():
    # T4 / P100 are sm_75 / sm_60: no flash kernel, so long sequences cost real memory.
    print(
        "sdpa flash", torch.backends.cuda.flash_sdp_enabled(),
        "| mem_efficient", torch.backends.cuda.mem_efficient_sdp_enabled(),
        "| math", torch.backends.cuda.math_sdp_enabled(),
    )
    print("capability", torch.cuda.get_device_capability(0))
"""
        ),
        markdown(
            """
## Train

`lev encode` refuses to overwrite an existing output directory, so a re-run
needs a new `TRAIN_RUN`. Over-long inputs are refused rather than truncated —
if this fails on `--max-len`, raise `max_len` in the config instead of cutting
the state.

Batches are built by sorting records by length and packing them up to
`max_tokens` **padded** tokens, so one long record never pads a whole batch of
short ones. `max_len` is the refusal line; `max_tokens` is the memory budget.
If you hit an OOM, lower `max_tokens` in the config — never `max_len`.
T4 (sm_75) and P100 (sm_60) have no flash-attention kernel, so this distinction
matters more there than on an A100.
"""
        ),
        code(
            """
lev(
    "encode",
    "--dataset", str(DATASET),
    "--config", CONFIG,
    "--out", str(WORK / TRAIN_RUN),
    "--device", "cuda",
    "--execute",
)

trained = json.loads((WORK / TRAIN_RUN / "checkpoint" / "lev-encoder.json").read_text())
print("pairs:", trained["data"]["pairs"], "| total seconds:", trained["seconds"])
for epoch in trained["history"]:
    peak = epoch["peak_device_bytes"]
    print(
        f"epoch {epoch['epoch']}: loss {epoch['train_loss']:.4f} "
        f"{epoch['seconds']}s peak {peak / 2**30:.2f} GiB" if peak
        else f"epoch {epoch['epoch']}: loss {epoch['train_loss']:.4f} {epoch['seconds']}s"
    )
"""
        ),
        markdown(
            """
## Score development

The threshold is fixed at 0.5 here and is **not** tuned on this split's result.
If you later want a different threshold, pre-register it on development before
reading calibration or test.
"""
        ),
        code(
            """
lev(
    "encode-benchmark",
    "--dataset", str(DATASET),
    "--config", CONFIG,
    "--checkpoint", str(WORK / TRAIN_RUN / "checkpoint"),
    "--split", "development",
    "--out", str(WORK / DEV_RUN),
    "--device", "cuda",
    "--execute",
)

lev(
    "report",
    "--benchmark", str(WORK / DEV_RUN),
    "--threshold", "0.5",
    "--out", str(WORK / DEV_RUN / "lev-report.json"),
)

report = json.loads((WORK / DEV_RUN / "lev-report.json").read_text())
d = report["decisions"]
print(f"engine: {report['engine']}")
print(f"split: {report['split']}  threshold: {report['threshold']}")
print(
    f"accuracy_all_requested {d['accuracy_all_requested']['rate']:.2%} "
    f"({d['accuracy_all_requested']['numerator']}/{d['accuracy_all_requested']['denominator']})"
)
print(f"completed {d['completed']}  missing {d['missing']}")
print(f"false_rejections {d['false_rejections']}  false_acceptances {d['false_acceptances']}")
print(f"brier_completed_only {d['brier_completed_only']}")
print()
print(f"{'kind':38} {'correct':>9} {'rate':>8}  missing")
for kind, m in report["by_kind"].items():
    rate = m["accuracy_all_requested"]["rate"]
    print(
        f"{kind:38} "
        f"{m['accuracy_all_requested']['numerator']:>4}/{m['accuracy_all_requested']['denominator']:<4} "
        f"{rate:>7.2%}  {m['missing']}"
    )
"""
        ),
        markdown(
            """
## Compare with the Kev 4B baseline

Reference points from this repository, on **different** packs and splits, so
treat them as scale, not as a paired comparison:

| model | params | development |
|---|---:|---:|
| raw Kev-4B | 4B | 364 / 590 (61.69%) |
| Lev v0.2 (4B + LoRA) | 4B | 483 / 590 (81.86%) |

A ModernBERT-base run on `en-joint-002` development is a **new** measurement on
a pack the 4B numbers were not taken on. Do not put them in the same table
without saying so.

## Before you close the notebook

Save Version writes `/kaggle/working` to the notebook's output. Without it, the
checkpoint and the report are discarded when the session ends.
"""
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="notebooks/kaggle-encoder-smoke.ipynb")
    parser.add_argument("--dataset", default="/kaggle/input/lev-en-joint-002")
    parser.add_argument("--config", default="configs/encoder-modernbert-base.toml")
    parser.add_argument("--train-run", default="enc-train-001")
    parser.add_argument("--dev-run", default="enc-dev-001")
    args = parser.parse_args()
    notebook = {
        "cells": cells(args.dataset, args.config, args.train_run, args.dev_run),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
