"""Review labels stay outside model inputs; word families never cross partitions."""

from collections import Counter
from pathlib import Path
import re

from .io import digest, encoded, jsonl, read, write

SPLITS = ("train", "development", "calibration", "test")
KINDS = (
    "sense",
    "form",
    "ipa",
    "syllable",
    "morpheme",
    "membership",
    "relation",
    "distinctness",
)


def fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    missing, extra = (
        set(required) - value.keys(),
        value.keys() - set(required) - set(optional),
    )
    if missing or extra:
        raise ValueError(
            f"missing fields {sorted(missing)}; unknown fields {sorted(extra)}"
        )


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a nonempty string")


def validate(records, rubric):
    fields(
        rubric,
        ("version", "language", "status", "rules"),
        ("format_version", "provenance_policy"),
    )
    generic = rubric.get("format_version", 1) == 2
    if rubric.get("format_version", 1) not in (1, 2):
        raise ValueError("unsupported rubric format")
    if rubric["language"] != "en" or rubric["status"] not in (
        "draft",
        "approved",
        "research",
    ):
        raise ValueError(
            "v0 supports English rubrics with draft/approved/research status"
        )
    text(rubric["version"])
    if set(rubric["rules"]) != ({"default"} if generic else set(KINDS)):
        raise ValueError("rubric rules must match its format version")
    for rule in rubric["rules"].values():
        text(rule)
    ids, families, cases, content_splits = set(), {}, {}, {}
    counts = Counter()
    for r in records:
        fields(
            r,
            (
                "id",
                "family_id",
                "case_id",
                "split",
                "language",
                "scope",
                "origin",
                "rubric_version",
                "state",
                "questions",
            ),
        )
        for key in ("id", "family_id", "case_id"):
            text(r[key])
        if r["id"] in ids:
            raise ValueError(f"duplicate record id: {r['id']}")
        ids.add(r["id"])
        if r["split"] not in SPLITS or r["scope"] not in ("partial", "complete"):
            raise ValueError(f"{r['id']}: invalid split/scope")
        if (
            r["language"] != rubric["language"]
            or r["rubric_version"] != rubric["version"]
        ):
            raise ValueError(f"{r['id']}: language/rubric mismatch")
        if r["origin"] not in ("example", "generated", "curated", "mutation"):
            raise ValueError(f"{r['id']}: invalid origin")
        for table, key, value in (
            (families, r["family_id"], r["split"]),
            (cases, r["case_id"], (r["family_id"], r["split"], r["scope"])),
        ):
            if key in table and table[key] != value:
                raise ValueError(
                    f"{r['id']}: family/case crosses split or scope: {key}"
                )
            table[key] = value
        if not isinstance(r["state"], (str, dict)) or not r["state"]:
            raise ValueError(f"{r['id']}: state must be nonempty text or object")
        # Catch copied facts with renamed IDs as well as explicit family leakage.
        state_key = " ".join(encoded(r["state"]).casefold().split())
        if state_key in content_splits and content_splits[state_key] != r["split"]:
            raise ValueError(f"{r['id']}: identical state crosses partitions")
        content_splits[state_key] = r["split"]
        if not isinstance(r["questions"], dict) or not r["questions"]:
            raise ValueError(f"{r['id']}: questions must be a nonempty object")
        for qid, q in r["questions"].items():
            text(qid)
            fields(q, ("kind", "claim", "label", "review"))
            text(q["claim"])
            if (
                not isinstance(q["kind"], str)
                or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", q["kind"])
                or (not generic and q["kind"] not in KINDS)
            ) or (q["label"] is not None and type(q["label"]) is not bool):
                raise ValueError(
                    f"{r['id']}/{qid}: invalid kind/label (bool or null required)"
                )
            review = q["review"]
            fields(review, ("status", "reviewer", "rationale", "evidence"))
            if review["status"] not in (
                "draft",
                "adjudicated",
                "uncertain",
                "source_labeled",
                "source_derived",
            ):
                raise ValueError(f"{r['id']}/{qid}: invalid review status")
            if not isinstance(review["evidence"], list):
                raise ValueError("evidence must be a list")
            for source in review["evidence"]:
                fields(source, ("reference", "locator"))
                text(source["reference"])
                text(source["locator"])
            if review["status"] in ("adjudicated", "source_labeled", "source_derived"):
                if q["label"] is None or not review["evidence"]:
                    raise ValueError(
                        f"{r['id']}/{qid}: adjudication needs label and evidence"
                    )
                text(review["reviewer"])
                text(review["rationale"])
            if review["status"] == "uncertain" and q["label"] is not None:
                raise ValueError("uncertain labels must be null")
            counts[q["kind"]] += 1
    return {
        "records": len(records),
        "questions": sum(counts.values()),
        "families": len(families),
        "by_kind": dict(counts),
        "splits": dict(Counter(r["split"] for r in records)),
    }


def to_kev(r, rubric):
    return {
        "state": r["state"],
        "questions": {
            qid: {
                "type": "noul",
                "instructions": rubric["rules"][
                    "default" if rubric.get("format_version", 1) == 2 else q["kind"]
                ]
                + "\nClaim to judge:\n"
                + q["claim"],
                "label": q["label"],
                "src": q["kind"],
            }
            for qid, q in r["questions"].items()
        },
        "_meta": {
            "id": r["id"],
            "group_id": r["family_id"],
            "source": "lev-en",
            "variant": "clean",
            "split": r["split"],
            "case_id": r["case_id"],
            "scope": r["scope"],
        },
    }


def exportable(records, rubric, *, examples, research=False):
    if examples and research:
        raise ValueError("example and research modes are mutually exclusive")
    if research and (
        rubric.get("format_version") != 2 or rubric["status"] != "research"
    ):
        raise ValueError("research export requires an explicit v2 research rubric")
    if not examples and not research and rubric["status"] != "approved":
        raise ValueError(
            "rubric is draft; approve its semantics before production export"
        )
    for r in records:
        if not examples and r["origin"] == "example":
            raise ValueError("example records cannot enter production datasets")
        if any(
            q["label"] is None
            or (
                not examples
                and q["review"]["status"]
                not in (
                    {"adjudicated", "source_labeled", "source_derived"}
                    if research
                    else {"adjudicated"}
                )
            )
            for q in r["questions"].values()
        ):
            raise ValueError(
                f"{r['id']}: unresolved labels; export refused, no silent filtering"
            )


def prepare(source, rubric_path, out, *, examples=False, research=False):
    records, rubric = jsonl(source), read(rubric_path)
    summary = validate(records, rubric)
    exportable(records, rubric, examples=examples, research=research)
    missing = set(SPLITS) - set(summary["splits"])
    if missing:
        raise ValueError(f"all four partitions required; missing {sorted(missing)}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "rubric.json", rubric)
    with (out / "records.jsonl").open("x", encoding="utf-8") as stream:
        for r in sorted(records, key=lambda r: r["id"]):
            stream.write(encoded(r) + "\n")
    for split in SPLITS:
        with (out / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for r in sorted(records, key=lambda r: r["id"]):
                if r["split"] == split:
                    stream.write(encoded(to_kev(r, rubric)) + "\n")
    manifest = {
        "format_version": 1,
        "example_only": examples,
        "research_only": research,
        "summary": summary,
        "source_sha256": digest(source),
        "files": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    write(out / "manifest.json", manifest)
    return manifest


def bundle(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    required = {"records.jsonl", "rubric.json", *(f"{s}.jsonl" for s in SPLITS)}
    if manifest.get("format_version") != 1 or set(manifest["files"]) != required:
        raise ValueError("unsupported/incomplete dataset manifest")
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"dataset was modified: {name}")
    records, rubric = (
        jsonl(directory / "records.jsonl"),
        read(directory / "rubric.json"),
    )
    validate(records, rubric)
    if type(manifest["example_only"]) is not bool:
        raise ValueError("manifest example_only must be boolean")
    if type(manifest.get("research_only", False)) is not bool:
        raise ValueError("manifest research_only must be boolean")
    exportable(
        records,
        rubric,
        examples=manifest["example_only"],
        research=manifest.get("research_only", False),
    )
    for split in SPLITS:
        expected = [to_kev(r, rubric) for r in records if r["split"] == split]
        if jsonl(directory / f"{split}.jsonl") != expected:
            raise ValueError(f"{split}: export differs from source records")
    return records, manifest
