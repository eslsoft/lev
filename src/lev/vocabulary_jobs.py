"""Deterministic full-vocabulary evaluation shards; never training datasets."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .io import digest, encoded, fingerprint, jsonl, read, write
from .modal_runner import build_app, modal_output, snapshot_kev
from .vocabulary import load_words, score_words, verify_bundle
from .workflow import check_calibration, config


def shard(dataset, out, count):
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("shard count must be 1..64")
    words = load_words(dataset)
    # Contiguous fixed rank groups keep all questions for a word together.
    by_id = {
        rid: (w["rank"] - 1) * count // len(words)
        for w in words
        for rid in w["record_ids"]
    }
    if len(by_id) != sum(len(w["record_ids"]) for w in words):
        raise ValueError("duplicate registry request")
    out.mkdir(parents=True, exist_ok=False)
    files = [out / f"shard-{i:02d}.jsonl" for i in range(count)]
    streams = [p.open("x") for p in files]
    requests, judgments, seen = Counter(), Counter(), set()
    try:
        with (dataset / "test.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                rid = row["_meta"]["id"]
                if rid in seen or rid not in by_id:
                    raise ValueError("unknown/duplicate native request")
                i = by_id[rid]
                streams[i].write(line)
                seen.add(rid)
                requests[i] += 1
                judgments[i] += len(row["questions"])
    finally:
        for stream in streams:
            stream.close()
    if seen != set(by_id) or any(requests[i] == 0 for i in range(count)):
        raise ValueError("incomplete or empty shard layout")
    manifest = dict(
        format="lev-vocabulary-shards-v1",
        test_only=True,
        parent_manifest_sha256=digest(dataset / "manifest.json"),
        words=len(words),
        requests=sum(requests.values()),
        judgments=sum(judgments.values()),
        shards=[
            dict(
                index=i,
                file=p.name,
                sha256=digest(p),
                requests=requests[i],
                judgments=judgments[i],
                words=sum((w["rank"] - 1) * count // len(words) == i for w in words),
            )
            for i, p in enumerate(files)
        ],
    )
    write(out / "manifest.json", manifest)
    return manifest


def verify_shards(path):
    manifest = read(path / "manifest.json")
    if (
        manifest.get("format") != "lev-vocabulary-shards-v1"
        or manifest.get("test_only") is not True
    ):
        raise ValueError("not vocabulary evaluation shards")
    if [s["index"] for s in manifest["shards"]] != list(range(len(manifest["shards"]))):
        raise ValueError("invalid shard indices")
    for s in manifest["shards"]:
        if Path(s["file"]).name != s["file"] or digest(path / s["file"]) != s["sha256"]:
            raise ValueError("shard modified")
    return manifest


def run_vocabulary_job(request):
    import modal

    volume = modal.Volume.from_name(request["volume"])
    volume.reload()
    root = Path("/lev-store") / request["run_id"]
    root.mkdir(exist_ok=False)
    try:
        write(root / "job.json", request)
        manifest = verify_shards(Path("/lev-input/dataset"))
        if digest("/lev-input/dataset/manifest.json") != request["shards_sha256"]:
            raise ValueError("uploaded shards changed")
        if fingerprint(request["checkpoint"]) != request["checkpoint_files"]:
            raise ValueError("checkpoint changed after selection/calibration")
        spec = manifest["shards"][request["shard"]]
        shutil.copyfile(Path("/lev-input/dataset") / spec["file"], root / "test.jsonl")
        shutil.copyfile(
            "/lev-input/dataset/manifest.json", root / "shards-manifest.json"
        )
        volume.commit()
        command = [
            "uv",
            "run",
            "--project",
            "/opt/kev",
            "--frozen",
            "--no-dev",
            "python",
            "-m",
            "lev.bridge",
            "benchmark",
            "--run",
            request["checkpoint"],
            "--data",
            str(root / "test.jsonl"),
            "--out",
            str(root / "native"),
            "--device",
            "cuda",
            "--max-state",
            str(request["max_state"]),
        ]
        env = os.environ.copy()
        # Match the frozen candidate's explicit calibrated inference precision.
        for key in (
            "KEV_DTYPE",
            "KEV_LORA_SCALE",
            "KEV_MERGE",
            "KEV_ATTN",
            "KEV_BACKEND",
            "KEV_TEMPERATURE",
        ):
            env.pop(key, None)
        env["KEV_TEMPERATURE"] = str(request["temperature"])
        with (root / "run.log").open("x") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError("vocabulary inference failed: " + str(result.returncode))
        coverage = read(root / "native/report.json")["coverage"]
        if (
            coverage["requested_records"] != spec["requests"]
            or coverage["requested_questions"] != spec["judgments"]
        ):
            raise ValueError("inference requested population differs from shard")
        result = dict(status="succeeded", coverage=coverage, shard=request["shard"])
        write(root / "modal-status.json", result)
        return result
    except BaseException as error:
        write(root / "modal-status.json", dict(status="failed", error=str(error)))
        raise
    finally:
        volume.commit()


def launch(args):
    manifest = verify_shards(args.shards)
    selection = read(args.selection)
    if (
        selection.get("candidate_selected") is not True
        or selection["checkpoint"] != args.checkpoint
    ):
        raise ValueError(
            "candidate must be selected on development before vocabulary inference"
        )
    if not re.fullmatch(
        r"/lev-store/[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}/result/checkpoint", args.checkpoint
    ):
        raise ValueError("expected existing Modal checkpoint")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.run_id):
        raise ValueError("invalid run id")
    if not 0 <= args.shard < len(manifest["shards"]):
        raise ValueError("invalid shard")
    if not 1 <= args.timeout <= 86400:
        raise ValueError("timeout must be 1..86400 seconds")
    cfg = config(args.config)
    identity = dict(path=args.checkpoint, files=selection["checkpoint_files"])
    calibration = check_calibration(args.calibration, identity)
    request = dict(
        run_id=args.run_id,
        volume=args.volume,
        shard=args.shard,
        shards_sha256=digest(args.shards / "manifest.json"),
        parent_manifest_sha256=manifest["parent_manifest_sha256"],
        checkpoint=args.checkpoint,
        checkpoint_files=selection["checkpoint_files"],
        selection_sha256=digest(args.selection),
        calibration_sha256=digest(args.calibration),
        temperature=calibration["temperature"],
        max_state=cfg["training"]["max_state"],
        threshold=0.5,
        gpu=args.gpu,
        timeout=args.timeout,
        kev_commit=cfg["kev_commit"],
    )
    if not args.execute:
        print(encoded(dict(dry_run=True, **request)))
        return
    with tempfile.TemporaryDirectory(prefix="lev-vocabulary-modal-") as temp:
        snapshot = Path(temp) / "kev"
        snapshot_kev(args.kev, snapshot, cfg["kev_commit"])
        app, worker = build_app(
            args, snapshot, args.shards, args.config, job_function=run_vocabulary_job
        )
        with modal_output(), app.run(detach=True):
            print(encoded(worker.remote(request)))


def merge(dataset, shards, runs, out):
    verify_bundle(dataset)
    plan = verify_shards(shards)
    if plan["parent_manifest_sha256"] != digest(dataset / "manifest.json"):
        raise ValueError("shards belong to another vocabulary benchmark")
    rows, seen, identities, sources = [], set(), set(), []
    environments = set()
    raw_environments = set()
    for run in runs:
        job, status = read(run / "job.json"), read(run / "modal-status.json")
        i = job["shard"]
        if (
            status["status"] != "succeeded"
            or i in seen
            or not 0 <= i < len(plan["shards"])
        ):
            raise ValueError(
                "failed, duplicate or unknown shard; no retry-and-pick merging"
            )
        if job["shards_sha256"] != digest(shards / "manifest.json"):
            raise ValueError("evaluation plan differs")
        spec = plan["shards"][i]
        if digest(run / "test.jsonl") != spec["sha256"]:
            raise ValueError("evaluated input changed")
        identities.add(
            encoded(
                {
                    k: job[k]
                    for k in (
                        "checkpoint",
                        "checkpoint_files",
                        "temperature",
                        "calibration_sha256",
                        "selection_sha256",
                        "max_state",
                        "threshold",
                        "kev_commit",
                    )
                }
            )
        )
        execution = read(run / "native/lev-context.json")
        if (
            execution["data_sha256"] != spec["sha256"]
            or execution["checkpoint"] != job["checkpoint"]
            or execution["inference_temperature"] != job["temperature"]
            or execution["context"]["max_state"] != job["max_state"]
            or execution["context"]["truncate"] is not False
        ):
            raise ValueError("executed inference differs from frozen plan")
        environment = execution["environment"]
        raw_environments.add(encoded(environment))
        comparable = dict(environment)
        # Modal's A100-80GB allocation can use either physical form factor.
        # Preserve the actual device per shard; this is not bitwise equivalence.
        if comparable.get("gpu") in {
            "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"
        }:
            comparable["gpu"] = "A100-80GB (PCIe or SXM4)"
        environments.add(encoded(comparable))
        native = read(run / "native/rows.json")
        allowed = {r["_meta"]["id"] for r in jsonl(shards / spec["file"])}
        if any(r["id"] not in allowed for r in native):
            raise ValueError("prediction assigned to wrong shard")
        rows.extend(native)
        seen.add(i)
        sources.append(
            dict(run=str(run), shard=i, rows_sha256=digest(run / "native/rows.json"),
                 inference_environment=environment)
        )
    if (
        seen != set(range(len(plan["shards"])))
        or len(identities) != 1
        or len(environments) != 1
    ):
        raise ValueError("missing shards or mixed checkpoint/calibration/selection/environment")
    result = score_words(
        load_words(dataset), jsonl(dataset / "records.jsonl"), rows, 0.5
    )
    result.update(
        dataset_manifest_sha256=plan["parent_manifest_sha256"],
        sources=sources,
        inference_identity=json.loads(next(iter(identities))),
        inference_environment=json.loads(next(iter(environments))),
        observed_inference_environments=[json.loads(e) for e in sorted(raw_environments)],
    )
    write(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--dataset", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--count", type=int, default=16)
    m = sub.add_parser("merge")
    m.add_argument("--dataset", type=Path, required=True)
    m.add_argument("--shards", type=Path, required=True)
    m.add_argument("--run", type=Path, action="append", required=True)
    m.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    for name in ("shards", "selection", "calibration", "config", "kev"):
        r.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "checkpoint"):
        r.add_argument("--" + name, required=True)
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--volume", default="lev-experiments")
    r.add_argument("--gpu", default="A100-80GB")
    r.add_argument("--timeout", type=int, default=21600)
    r.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.operation == "shard":
        print(encoded(shard(args.dataset, args.out, args.count)))
    elif args.operation == "merge":
        print(encoded(merge(args.dataset, args.shards, args.run, args.out)["overall"]))
    else:
        launch(args)


if __name__ == "__main__":
    main()
