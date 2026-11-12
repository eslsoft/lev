"""Strict JSON and immutable artifact helpers."""

import hashlib
import json
from pathlib import Path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def parse(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def read(path):
    return parse(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    result = []
    # JSONL records end at LF; U+2028/U+2029/U+0085 are legal string content.
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").split("\n"), 1
    ):
        if line.strip():
            try:
                result.append(parse(line))
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not result:
        raise ValueError(f"empty dataset: {path}")
    return result


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )


def fingerprint(directory):
    files = sorted(p for p in Path(directory).rglob("*") if p.is_file())
    if not (Path(directory) / "head.pt").is_file():
        raise ValueError(f"not a Kev checkpoint: {directory}")
    return {str(p.relative_to(directory)): digest(p) for p in files}
