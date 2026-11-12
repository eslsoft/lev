"""Run pinned Kev tokenization/admission functions without loading model weights."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

from transformers import AutoTokenizer

from lev.io import digest, read, write
from lev.upstream import checkout
from lev.workflow import config


def check(dataset, kev, config_path, out):
    cfg = config(config_path)
    checkout(kev, cfg["kev_commit"])
    spec = importlib.util.spec_from_file_location(
        "lev_check_kev_api", kev / "kev/api.py"
    )
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    scope = dict(
        Path=Path,
        json=json,
        hashlib=hashlib,
        re=re,
        SystemOneRequest=api.SystemOneRequest,
        to_record=api.to_record,
    )
    data_tree = ast.parse((kev / "kev/data.py").read_text())
    functions = [
        n
        for n in data_tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"load_records", "api_request", "materialize"}
    ]
    if len(functions) != 3:
        raise ValueError("upstream data functions changed")
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "kev/data.py", "exec"),
        scope,
    )
    model_tree = ast.parse((kev / "kev/model.py").read_text())
    constants = {
        "SPECIAL",
        "MAX_STATE",
        "MAX_BRANCH",
        "MAX_PACKED",
        "MAX_TRAIN_STATE",
        "SERVE_MAX_STATE",
        "_SPECIAL_RE",
        "OPT_NONE",
        "OPT_DECIDE",
    }
    nodes = []
    for node in model_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id in constants
            for target in node.targets
            for n in ast.walk(target)
        ):
            nodes.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {
            "ContextOverflow",
            "training_context",
            "user_tokens",
            "encode",
        }:
            nodes.append(node)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "kev/model.py", "exec"), scope
    )
    tok = AutoTokenizer.from_pretrained(
        cfg["base"], revision=cfg["base_revision"], token=False
    )
    limits = scope["training_context"](cfg["training"]["max_state"])
    splits, failures = {}, []
    names = ["train", "development", "calibration", "test"]
    if read(dataset / "manifest.json").get("format") == "lev-vocabulary-shards-v1":
        from lev.vocabulary_jobs import verify_shards

        names = [Path(s["file"]).stem for s in verify_shards(dataset)["shards"]]
    for split in names:
        rows = scope["load_records"](dataset / (split + ".jsonl"))
        max_state = max_packed = 0
        for row in rows:
            try:
                encoded = scope["encode"](
                    tok,
                    scope["materialize"](row),
                    strict=True,
                    max_state=limits["max_state"],
                    max_branch=limits["max_branch"],
                )
                if (
                    encoded["state_truncated"]
                    or len(encoded["ids"]) > limits["max_packed"]
                ):
                    raise ValueError("packed context overflow or truncation")
                max_state = max(max_state, encoded["state_tokens"])
                max_packed = max(max_packed, len(encoded["ids"]))
            except (ValueError, KeyError) as error:
                failures.append(
                    dict(split=split, id=row["_meta"]["id"], error=str(error))
                )
        splits[split] = dict(
            requests=len(rows), max_state_tokens=max_state, max_packed_tokens=max_packed
        )
    result = dict(
        status="failed" if failures else "passed",
        splits=splits,
        failures=failures,
        limits=limits,
        model_execution=False,
        tokenizer_execution=True,
        kev_commit=cfg["kev_commit"],
        base=cfg["base"],
        base_revision=cfg["base_revision"],
        source_sha256={
            name: digest(kev / "kev" / name)
            for name in ("api.py", "data.py", "model.py")
        },
        checker_sha256=digest(__file__),
    )
    write(out, result)
    print(json.dumps(result))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("artifacts/en-context-002"))
    p.add_argument("--kev", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/lev-4b-v02.toml"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    check(args.dataset, args.kev, args.config, args.out)
