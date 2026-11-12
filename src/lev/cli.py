import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .dataset import bundle, prepare, validate
from .io import jsonl, read, write
from .report import summarize
from .upstream import checkout, contract
from .workflow import benchmark_info, config, execute, plan


def parser():
    p = argparse.ArgumentParser(
        description="Lev language-knowledge training and evaluation"
    )
    sub = p.add_subparsers(dest="command", required=True)
    for action in ("validate", "prepare"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--data", required=True)
        cmd.add_argument("--rubric", default="rubrics/en-v1.json")
        if action == "prepare":
            cmd.add_argument(
                "--research",
                action="store_true",
                help="export source-labeled research data; not production gold",
            )
            cmd.add_argument("--out", required=True)
            cmd.add_argument(
                "--examples",
                action="store_true",
                help="mark output as example-only, unusable for training",
            )
    upstream = sub.add_parser("check-upstream")
    upstream.add_argument(
        "--kev",
        default=os.environ.get("LEV_KEV_PATH"),
        required=not os.environ.get("LEV_KEV_PATH"),
    )
    upstream.add_argument("--config", default="configs/lev-4b.toml")
    for action in ("train", "benchmark", "calibrate", "serve"):
        cmd = sub.add_parser(action)
        cmd.add_argument(
            "--kev", help="existing Kev checkout; defaults to LEV_KEV_PATH"
        )
        cmd.add_argument("--config", default="configs/lev-4b.toml")
        cmd.add_argument(
            "--out",
            required=True,
            help="new job directory; never overwrite an earlier run",
        )
        cmd.add_argument(
            "--execute",
            action="store_true",
            help="execute the plan; may download model weights/use GPU",
        )
        if action != "calibrate":
            cmd.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
        if action in ("train", "benchmark"):
            cmd.add_argument("--dataset", required=True)
        if action in ("benchmark", "serve"):
            cmd.add_argument("--checkpoint", required=True)
            cmd.add_argument("--calibration", required=action == "serve")
        if action == "benchmark":
            cmd.add_argument(
                "--split",
                choices=("development", "calibration", "test"),
                default="development",
            )
        if action == "calibrate":
            cmd.add_argument("--benchmark", required=True)
        if action == "serve":
            cmd.add_argument("--port", type=int, default=8009)
    report = sub.add_parser("report")
    report.add_argument("--benchmark", required=True)
    report.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="pre-register on development data; not a calibrated accuracy guarantee",
    )
    report.add_argument("--out", required=True)
    return p


def dispatch(args):
    if args.command == "validate":
        return validate(jsonl(args.data), read(args.rubric))
    if args.command == "prepare":
        return prepare(
            args.data,
            args.rubric,
            args.out,
            examples=args.examples,
            research=args.research,
        )
    if args.command == "check-upstream":
        return {
            **checkout(args.kev, config(args.config)["kev_commit"]),
            **contract(args.kev),
        }
    if args.command == "report":
        info = benchmark_info(args.benchmark)
        records, _ = bundle(info["dataset"])
        result = summarize(
            [r for r in records if r["split"] == info["split"]],
            read(Path(args.benchmark) / "native" / "rows.json"),
            args.threshold,
        )
        result.update(
            example_only=info["example_only"],
            research_only=info.get("research_only", False),
            split=info["split"],
            checkpoint=info["checkpoint"],
            dataset_sha256=info["dataset_sha256"],
            calibration_sha256=info["calibration_sha256"],
        )
        write(args.out, result)
        return result
    job = plan(args)
    if args.execute:
        return execute(job)
    return {"dry_run": True, **job}


def main():
    p = parser()
    args = p.parse_args()
    try:
        result = dispatch(args)
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"lev: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
