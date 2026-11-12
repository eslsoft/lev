"""Runs in Kev's environment. Lev's ordinary data commands need no ML dependencies."""

import argparse
import sys
from pathlib import Path

from .dataset import bundle
from .io import digest, read, write
from .report import align
from .workflow import benchmark_info, model_identity


def benchmark(args):
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions
    from kev.data import load_records
    from kev.device import select
    from kev.model import training_context
    from kev.predictors import LocalPredictor

    context = {**training_context(args.max_state), "truncate": False}
    opts = LoadOptions.from_env()
    device = (
        select(args.device)
        if opts.dtype is None
        else select(args.device, dtype=opts.dtype)
    )
    predictor = LocalPredictor(args.run, device, opts, context=context)
    report, _ = evaluate_records(
        load_records(args.data), predictor, args.out, skip_overlong=True
    )
    # Preserve Kev's original report; record the explicit Lev context alongside it.
    write(
        Path(args.out) / "lev-context.json",
        {
            "context": context,
            "data_sha256": digest(args.data),
            "checkpoint": args.run,
            "environment": predictor.environment,
            "inference_temperature": predictor.temperature,
            "coverage": report["coverage"],
        },
    )


def calibrate(benchmark, out):
    info = benchmark_info(benchmark)
    if info["split"] != "calibration" or info["example_only"]:
        raise ValueError("calibration requires a non-example calibration benchmark")
    identity = info["checkpoint"]
    current = model_identity(identity.get("path", identity.get("hub")))
    if current != identity:
        raise ValueError("checkpoint changed after benchmark")
    records, _ = bundle(info["dataset"])
    records = [r for r in records if r["split"] == "calibration"]
    rows_path = Path(benchmark) / "native" / "rows.json"
    rows = read(rows_path)
    predictions = align(records, rows)
    if len(predictions) != sum(len(r["questions"]) for r in records):
        raise ValueError("incomplete calibration predictions")
    if {q["label"] for r in records for q in r["questions"].values()} != {True, False}:
        raise ValueError("calibration must include positive and negative labels")
    if any("logits" not in r or "inference_temperature" not in r for r in rows):
        raise ValueError("local benchmark logits required for temperature fitting")
    from kev.metrics import TEMPERATURE_FIT, fit_temperature, metrics, raw_row

    raw = [raw_row(r) for r in rows]
    temperature = float(fit_temperature(raw, **TEMPERATURE_FIT))
    write(
        out,
        {
            "temperature": temperature,
            "checkpoint": identity,
            "dataset_sha256": info["dataset_sha256"],
            "rows_sha256": digest(rows_path),
            "method": TEMPERATURE_FIT,
            "questions": len(rows),
            "raw_fit_metrics": metrics(raw),
            "calibrated_fit_metrics": metrics(raw, temperature),
            "scope": "in-domain, family-disjoint calibration; these fit metrics are NOT test accuracy; checkpoint untouched",
        },
    )


def serve(args):
    if not (Path(args.run) / "head.pt").is_file():
        raise ValueError("Lev checkpoint is missing")
    from fastapi import HTTPException
    import kev.serve as backend
    from kev.model import training_context

    context = training_context(args.max_state)

    def lev_admit(model, tok, rec, truncate=False):
        if truncate:
            raise ValueError("Lev does not truncate review inputs")
        enc = model.encode(
            tok,
            rec,
            max_state=context["max_state"],
            max_branch=context["max_branch"],
            strict=True,
        )
        if len(enc["ids"]) > context["max_packed"]:
            raise ValueError(
                "packed review exceeds Lev's configured context; split requests"
            )
        return enc

    backend.admit = lev_admit
    backend.SERVE_MAX_STATE = context["max_state"]
    prepare = backend.prepare

    def lev_request(request):
        if request.model != args.name:
            raise HTTPException(
                422, f"this endpoint serves {args.name}; specify that model explicitly"
            )
        return prepare(request)

    backend.prepare = lev_request
    backend.MODEL_NAMES = (args.name,)
    backend.app.title = "Lev"
    backend.app.description = (
        "Language knowledge decisions; Lev checkpoint, Kev inference engine."
    )
    # The same explicit path as fallback makes a missing checkpoint fail instead of serving a smoke model.
    sys.argv = [
        "lev-serve",
        "--run",
        args.run,
        "--fallback",
        args.run,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device",
        args.device,
    ]
    backend.main()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    for field in ("run", "data", "out", "device"):
        bench.add_argument("--" + field, required=True)
    bench.add_argument("--max-state", type=int, required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--benchmark", required=True)
    cal.add_argument("--out", required=True)
    srv = sub.add_parser("serve")
    for field in ("run", "name", "host", "device"):
        srv.add_argument("--" + field, required=True)
    srv.add_argument("--port", type=int, required=True)
    srv.add_argument("--max-state", type=int, required=True)
    args = parser.parse_args()
    if args.command == "benchmark":
        benchmark(args)
    elif args.command == "calibrate":
        calibrate(args.benchmark, args.out)
    else:
        serve(args)


if __name__ == "__main__":
    main()
