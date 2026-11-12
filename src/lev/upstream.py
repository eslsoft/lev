"""A local Kev checkout is a pinned dependency, not vendored or modified by Lev."""

import ast
import subprocess
from pathlib import Path

from .io import digest


def checkout(path, commit):
    path = Path(path).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    actual = git("rev-parse", "HEAD")
    if actual != commit:
        raise ValueError(f"Kev revision mismatch: expected {commit}, found {actual}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Kev has tracked changes; use the pinned unmodified checkout")
    # Untracked Python modules could change import resolution even with a clean diff.
    untracked = git("ls-files", "--others", "--exclude-standard", "kev")
    if untracked:
        raise ValueError("Kev has untracked implementation files")
    return {
        "path": str(path),
        "commit": actual,
        "lock_sha256": digest(path / "uv.lock"),
    }


def parser_flags(path):
    tree = ast.parse(Path(path).read_text())
    return {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value.startswith("--")
    }


def contract(path):
    path = Path(path)
    needed = {
        "train": {
            "--data",
            "--base",
            "--base_revision",
            "--init_from",
            "--out",
            "--device",
            "--dtype",
            "--epochs",
            "--lr",
            "--batch",
            "--accum",
            "--lora",
            "--head_dim",
            "--checkpointing",
            "--max_state",
            "--seed",
        },
        "benchmark": {"--run", "--data", "--out", "--device"},
        "serve": {"--run", "--fallback", "--device", "--host", "--port"},
    }
    for module, flags in needed.items():
        missing = flags - parser_flags(path / "kev" / f"{module}.py")
        if missing:
            raise ValueError(f"Kev {module} missing flags: {sorted(missing)}")
    return {
        "checked": sorted(needed),
        "scope": "source CLI contract only; no weights loaded",
    }
