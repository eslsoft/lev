"""Build a broader research corpus, reserving prior holdouts and unseen tasks."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import zipfile

from build_research_corpus import Corpus, key, partition
from lev.dataset import prepare
from lev.io import digest, encoded, jsonl, write

GRAMMAR_SPLITS = {
    "island_effects": "development",
    "control_raising": "calibration",
    "ellipsis": "test",
    "s-selection": "test",
    "anaphor_agreement": "excluded",
    "binding": "excluded",
    "quantifiers": "excluded",
}


def phrasing(claim, label, split, identity):
    """Truth-preserving transformations; polarity does not depend on the label."""
    choices = {
        "train": [
            ("Is the answer to this question yes: ", False),
            ("Would the correct answer be no: ", True),
            (
                "Please judge whether the following question has an affirmative answer: ",
                False,
            ),
            ("Is a negative answer warranted for this question: ", True),
        ],
        "development": [
            ("Can this question correctly be answered in the affirmative: ", False),
            ("Would answering yes to this question be incorrect: ", True),
        ],
        "calibration": [
            ("Does the evidence support answering yes here: ", False),
            ("Should this question receive a no answer: ", True),
        ],
        "test": [
            ("Is yes the appropriate response to the following query: ", False),
            ("Is the affirmative response to this query mistaken: ", True),
        ],
    }
    prefix, invert = choices[split][int(key(identity)[:8], 16) % len(choices[split])]
    return prefix + claim, (not label if invert else label)


def build(raw, extra, previous, out, bundle):
    previous_sources = json.loads(
        (previous.parent / "source-manifest.json").read_text()
    )
    for name, expected in previous_sources["files"].items():
        if digest(raw / name) != expected:
            raise ValueError("Original source changed: " + name)
    for event in jsonl(extra / "download-events.jsonl"):
        if (
            event["status"] != "downloaded"
            or digest(extra / event["file"]) != event["sha256"]
        ):
            raise ValueError("Extra source failed or changed: " + event["file"])
    # Recreate the original complete union graph, including families not selected
    # as data, before extending it. Existing frozen records remain untouched.
    base = Corpus(raw)
    for name in ("wordnet", "cmu", "ud", "blimp", "wiktionary"):
        getattr(base, name)()
    prior = jsonl(previous)
    base_ids = {r["id"] for r in prior}
    c = Corpus(raw)
    c.parents, c.aliases = dict(base.parents), dict(base.aliases)
    c.wordnet(limit=1000)
    c.cmu(limit=500)
    c.ud(limit=700, unique_groups=True)
    c.wiktionary()
    # Include the original training examples as replay, only after checking
    # whether the expanded union graph touches an old holdout.
    replay = [deepcopy(r) for r in prior if r["split"] == "train"]
    c.rows = [r for r in c.rows if r["id"] not in base_ids]
    for r in c.rows:
        source = r["questions"]["q0"]["review"]["evidence"][0]["reference"]
        if source == "cmudict.dict":
            phones = r["state"]["pronunciation"].split()
            count = sum(p[-1:] in {"0", "1", "2"} for p in phones)
            template = deepcopy(next(iter(r["questions"].values())))
            for j, (value, truth) in enumerate(((count, True), (count + 1, False))):
                q = deepcopy(template)
                q.update(
                    claim=f"Does the supplied pronunciation contain exactly {value} syllabic nuclei?",
                    label=truth,
                )
                q["review"]["rationale"] = (
                    "Count digit-marked vowels in the supplied CMU pronunciation; this is not a claim about every possible pronunciation."
                )
                r["questions"][f"q{j + 2}"] = q
    # Full BLiMP snapshot; construction families stay together, including minimal pairs.
    with zipfile.ZipFile(extra / "blimp.zip") as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".jsonl") or name.startswith("__MACOSX"):
                continue
            rows = [json.loads(line) for line in z.read(name).splitlines()]
            term = rows[0]["linguistics_term"]
            if GRAMMAR_SPLITS.get(term) == "excluded":
                c.event(
                    "blimp.zip",
                    name,
                    "excluded",
                    "prior holdout construction, never reused for training or new evaluation",
                )
                continue
            for item in sorted(
                rows, key=lambda r: key("v3:" + name + str(r["pairID"]))
            )[:20]:
                for field, truth in (("sentence_good", True), ("sentence_bad", False)):
                    locator = name + ":" + str(item["pairID"]) + ":" + field
                    rid = key("v3:" + locator)[:20]
                    c.rows.append(
                        dict(
                            id=rid,
                            family_id="construction:" + term,
                            case_id=rid,
                            split="train",
                            language="en",
                            scope="partial",
                            origin="curated",
                            rubric_version="en-general-research-v1",
                            state={
                                "sentence": item[field],
                                "variety": "English under the BLiMP grammaticality task; evaluate syntax and the stipulated reading, not event plausibility.",
                            },
                            questions={
                                "q0": dict(
                                    kind="grammaticality",
                                    claim="Is the sentence acceptable under the stated linguistic task?",
                                    label=truth,
                                    review=dict(
                                        status="source_labeled",
                                        reviewer="source:BLiMP",
                                        rationale="Published good/bad label; linguist-authored generation templates, not independent Lev adjudication.",
                                        evidence=[
                                            dict(
                                                reference=str(extra / "blimp.zip"),
                                                locator=locator,
                                            ),
                                            dict(
                                                reference="https://github.com/alexwarstadt/blimp",
                                                locator="CC-BY; snapshot sha256:"
                                                + digest(extra / "blimp.zip"),
                                            ),
                                        ],
                                    ),
                                )
                            },
                        )
                    )
    # Join duplicate model inputs before assigning splits; no silently dropped conflicts.
    states = defaultdict(list)
    for r in c.rows + replay + prior:
        states[" ".join(encoded(r["state"]).casefold().split())].append(r["family_id"])
    for anchors in states.values():
        c.union(anchors)
    pins = defaultdict(set)
    for r in prior:
        pins[c.find(r["family_id"])].add(r["split"])
    result, seen = [], set()
    for r in replay + c.rows:
        root = c.find(r["family_id"])
        previous_splits = pins[root]
        if previous_splits - {"train"}:
            c.event(
                "prior-family",
                r["id"],
                "excluded",
                "expanded family touches prior holdout: "
                + ",".join(sorted(previous_splits)),
            )
            continue
        if root.startswith("construction:"):
            split = GRAMMAR_SPLITS.get(root.removeprefix("construction:"), "train")
        else:
            split = "train" if previous_splits == {"train"} else partition(root)
        if split == "excluded":
            continue
        r["family_id"], r["split"] = root, split
        # Retain replay exactly. All new tasks get label-independent polarity/wording.
        if r["id"] not in base_ids:
            for qid, q in r["questions"].items():
                q["claim"], q["label"] = phrasing(
                    q["claim"], q["label"], split, r["id"] + qid
                )
                q["review"]["rationale"] += (
                    " The wrapper asks for the original answer or its logical negation, with label transformed accordingly."
                )
        signature = encoded(
            {
                "state": r["state"],
                "claims": sorted(q["claim"] for q in r["questions"].values()),
            }
        )
        if signature in seen:
            c.event(
                "duplicate",
                r["id"],
                "excluded",
                "identical model input already included",
            )
            continue
        seen.add(signature)
        result.append(r)
    # Unseen-source transfer: evaluation only, never train or calibration.
    implicatures = json.loads((extra / "implicatures.json").read_text())
    for i, item in enumerate(implicatures["examples"]):
        scores = item["target_scores"]
        if set(scores) != {"yes", "no"} or sorted(scores.values()) != [0.0, 1.0]:
            raise ValueError("Unexpected implicature label contract")
        rid = key("implicatures:" + item["input"])[:20]
        # One decision per dialogue avoids counting a redundant complementary pair.
        claim, label = phrasing(
            "Does Speaker 2 intend an affirmative answer to Speaker 1's question?",
            bool(scores["yes"]),
            "test",
            rid,
        )
        result.append(
            dict(
                id=rid,
                family_id="implicatures:" + rid,
                case_id=rid,
                split="test",
                language="en",
                scope="partial",
                origin="curated",
                rubric_version="en-general-research-v1",
                state={
                    "dialogue": item["input"],
                    "task": "Judge the intended conversational response, not just whether the literal word yes appears.",
                },
                questions={
                    "q0": dict(
                        kind="pragmatics",
                        claim=claim,
                        label=label,
                        review=dict(
                            status="source_labeled",
                            reviewer="source:BIG-bench-implicatures",
                            rationale="Published manually filtered implicature label, polarity transformed if the question asks for a negative answer. Evaluation only; not independently adjudicated by Lev.",
                            evidence=[
                                dict(
                                    reference=str(extra / "implicatures.json"),
                                    locator=f"examples[{i}]; sha256:{digest(extra / 'implicatures.json')}",
                                ),
                                dict(
                                    reference=str(extra / "implicatures-README.md"),
                                    locator="Data source and design considerations; CC BY 4.0 underlying source",
                                ),
                            ],
                        ),
                    )
                },
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "records.jsonl").write_text(
        "".join(encoded(r) + "\n" for r in sorted(result, key=lambda r: r["id"]))
    )
    (out / "selection-audit.jsonl").write_text(
        "".join(encoded(e) + "\n" for e in c.audit)
    )
    write(
        out / "source-manifest.json",
        {
            "previous_sha256": digest(previous),
            "extra": {p.name: digest(p) for p in extra.iterdir() if p.is_file()},
            "original": {
                str(p.relative_to(raw)): digest(p)
                for p in raw.rglob("*")
                if p.is_file()
            },
            "policy": "All prior nontraining families excluded. Expanded connected families touching a holdout excluded. Incidental vocabulary can recur in sentences; not globally vocabulary-disjoint. Implicatures strictly test-only.",
        },
    )
    summary = {
        s: {
            "records": sum(r["split"] == s for r in result),
            "questions": dict(
                Counter(
                    q["kind"]
                    for r in result
                    if r["split"] == s
                    for q in r["questions"].values()
                )
            ),
            "labels": dict(
                Counter(
                    str(q["label"])
                    for r in result
                    if r["split"] == s
                    for q in r["questions"].values()
                )
            ),
        }
        for s in ("train", "development", "calibration", "test")
    }
    write(out / "summary.json", summary)
    prepare(
        out / "records.jsonl",
        "rubrics/en-general-research-v1.json",
        bundle,
        research=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/en-research-20261021-a")
    p.add_argument("--extra", default="data/raw/en-general-003")
    p.add_argument("--previous", default="data/research/en-general-002/records.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--bundle", required=True)
    a = p.parse_args()
    build(Path(a.raw), Path(a.extra), Path(a.previous), Path(a.out), Path(a.bundle))
