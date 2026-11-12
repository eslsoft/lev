"""Correct a source-detected holdout leak without using predictions or changing training."""

from copy import deepcopy
from pathlib import Path
import re

from audit_family_isolation import Families
from build_generalization_corpus import phrasing
from build_research_corpus import Corpus, key
from lev.dataset import prepare
from lev.io import digest, encoded, jsonl, read, write

raw = Path("data/raw/en-research-20261021-a")
prior = jsonl("data/research/en-general-002/records.jsonl")
current = jsonl("data/research/en-general-004/records.jsonl")
graph = Families(raw)
audit = graph.audit(prior, current)
assert len(audit["violations"]) == 1
excluded = {
    r["id"]
    for v in audit["violations"]
    for r in v["rows"]
    if r["origin"] == "current" and r["split"] != "train"
}
assert excluded == {"fb633927ffd48d52eb06"}
assert all(r["split"] == "test" for r in current if r["id"] in excluded)
used = {graph.find(r["family_id"]) for r in prior + current}
roles = {
    "nsubj": "nominal subject",
    "obj": "direct object",
    "amod": "adjectival modifier",
    "advmod": "adverbial modifier",
}
other = {"nsubj": "obj", "obj": "nsubj", "amod": "advmod", "advmod": "amod"}
eligible = []
for sid, tokens in graph.ud.items():
    if not 5 <= len(tokens) <= 30:
        continue
    doc = "document:" + sid.rsplit("-", 1)[0]
    for index, target in tokens.items():
        if (
            target[3] not in {"NOUN", "VERB", "ADJ", "ADV"}
            or target[7] not in roles
            or not re.fullmatch("[a-z]{4,16}", target[2])
        ):
            continue
        anchors = [graph.word(target[2]), doc]
        if {graph.find(a) for a in anchors} & used:
            continue
        eligible.append(
            (
                key("lev-005-independent-replacement:" + sid + ":" + index),
                sid,
                index,
                anchors,
            )
        )
assert eligible
_, sid, index, anchors = min(eligible)
tokens = graph.ud[sid]
target = tokens[index]
c = Corpus(raw)
c.add(
    "ud.conllu",
    sid + ":token=" + index,
    anchors,
    {
        "tokens": [{"index": int(t[0]), "text": t[1]} for t in tokens.values()],
        "target_index": int(index),
        "convention": "Universal Dependencies basic analysis. Judge the target token's dependency role, not a possible role in another sentence.",
    },
    [
        (f"Does the target token function as a {roles[target[7]]}?", True, "syntax"),
        (
            f"Does the target token function as a {roles[other[target[7]]]}?",
            False,
            "syntax",
        ),
    ],
    "The EWT source assigns the explicit basic dependency relation in column 8. The alternative is a different exclusive basic relation. Selected by fixed hash among source-eligible unused lexical/document families, without predictions.",
)
r = c.rows[0]
graph.union(anchors)
r["family_id"], r["split"] = graph.find(anchors[0]), "test"
for qid, q in r["questions"].items():
    q["claim"], q["label"] = phrasing(q["claim"], q["label"], "test", r["id"] + qid)
    q["review"]["rationale"] += (
        " Polarity follows the fixed test-question transformation."
    )
assert r["id"] not in {x["id"] for x in prior + current}
repaired = [deepcopy(x) for x in current if x["id"] not in excluded] + [r]
validation = Families(raw).audit(prior, repaired)
assert validation["status"] == "passed"
out = Path("data/research/en-general-005")
out.mkdir(exist_ok=False)
(out / "records.jsonl").write_text(
    "".join(encoded(x) + "\n" for x in sorted(repaired, key=lambda x: x["id"]))
)
write(
    out / "correction.json",
    dict(
        reason="Shared WordNet irregular forms link good and well via better/best; first-owner alias mapping missed an edge.",
        previous_records_sha256=digest("data/research/en-general-004/records.jsonl"),
        excluded_ids=sorted(excluded),
        retained_excluded_records=[x for x in current if x["id"] in excluded],
        replacement_id=r["id"],
        replacement_locator=sid + ":token=" + index,
        eligible_replacements=len(eligible),
        selection="minimum SHA256(lev-005-independent-replacement:sent_id:token_index), after family/document exclusion; no predictions read by this script",
        frozen_models=True,
        train_development_calibration_unchanged=True,
    ),
)
write(out / "family-audit.json", validation)
write(
    out / "source-manifest.json",
    read("data/research/en-general-004/source-manifest.json"),
)
prepare(
    out / "records.jsonl",
    "rubrics/en-general-research-v1.json",
    "artifacts/en-general-005",
    research=True,
)
for split in ["train", "development", "calibration"]:
    assert digest(Path("artifacts/en-general-004") / (split + ".jsonl")) == digest(
        Path("artifacts/en-general-005") / (split + ".jsonl")
    )
print(
    "Corrected one test request; training/development/calibration exports byte-identical."
)
print(
    "Replacement:",
    sid,
    index,
    target[1],
    target[2],
    target[7],
    "; eligible candidates:",
    len(eligible),
)
print("Independent complete-form family audit:", validation["status"])
