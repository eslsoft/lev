"""Recompute paired comparisons from frozen expected questions and raw model rows."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from lev.dataset import bundle
from lev.io import digest, read, write
from lev.report import align, summarize


def compare(dataset, runs, split, out):
    records, manifest = bundle(dataset)
    records = [r for r in records if r["split"] == split]
    groups = defaultdict(list)
    for r in records:
        for qid, q in r["questions"].items():
            groups[r["family_id"]].append(((r["id"], qid), q["label"]))
    reports, predictions, identities = {}, {}, {}
    for label, directory in runs.items():
        directory = Path(directory)
        job, status = read(directory / "job.json"), read(directory / "status.json")
        if (
            status["status"] != "succeeded"
            or job["split"] != split
            or job["dataset_sha256"] != digest(dataset / "manifest.json")
        ):
            raise ValueError(f"{label}: failed or mismatched benchmark")
        rows = read(directory / "native/rows.json")
        report = summarize(records, rows, 0.5)
        report["macro_accuracy"] = sum(
            m["accuracy_all_requested"]["rate"] for m in report["by_kind"].values()
        ) / len(report["by_kind"])
        if report["decisions"]["missing"]:
            raise ValueError(f"{label}: missing model predictions")
        reports[label], predictions[label], identities[label] = (
            report,
            align(records, rows),
            job["checkpoint"],
        )
    labels = list(runs)
    paired = {}
    for label in labels[1:]:
        baseline = labels[0]
        group_counts = []
        wins = losses = 0
        for group in groups.values():
            delta = 0
            for k, truth in group:
                a, b = (
                    (predictions[label][k] >= 0.5) == truth,
                    (predictions[baseline][k] >= 0.5) == truth,
                )
                delta += int(a) - int(b)
                wins += a and not b
                losses += b and not a
            group_counts.append((delta, len(group)))
        rng = random.Random(42)
        samples = []
        for _ in range(2000):
            draw = rng.choices(group_counts, k=len(group_counts))
            samples.append(sum(d for d, n in draw) / sum(n for d, n in draw))
        samples.sort()
        paired[label + "-minus-" + baseline] = {
            "wins": wins,
            "losses": losses,
            "family_cluster_bootstrap_95pct": [samples[49], samples[1949]],
            "note": "Resamples declared family groups; few held-out grammar constructions and pretraining contamination limit generalization claims.",
        }
    write(
        out,
        dict(
            split=split,
            dataset_sha256=digest(dataset / "manifest.json"),
            research_only=manifest["research_only"],
            reports=reports,
            identities=identities,
            paired=paired,
        ),
    )
    print(
        json.dumps(
            {
                label: dict(
                    accuracy=r["decisions"]["accuracy_all_requested"],
                    macro_accuracy=r["macro_accuracy"],
                    by_kind={
                        k: m["accuracy_all_requested"] for k, m in r["by_kind"].items()
                    },
                )
                for label, r in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument(
        "--run", required=True, action="append", help="label=benchmark-directory"
    )
    p.add_argument("--split", choices=["development", "test"], default="development")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    compare(a.dataset, dict(value.split("=", 1) for value in a.run), a.split, a.out)
