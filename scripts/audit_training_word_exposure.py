"""Separate frozen historical exposure from the candidate's supervised target exposure.

This does not estimate pretraining contamination and does not treat incidental
passage vocabulary as explicit supervised word targets.
"""

import argparse
from collections import Counter
from pathlib import Path

from build_context_corpus import RAW, lexical_graph, read_wic
from lev.dataset import bundle
from lev.io import digest, encoded, write
from lev.vocabulary import load_words


def audit(training, vocabulary, out):
    records, _ = bundle(training)
    words = load_words(vocabulary)
    families, _, _, _ = lexical_graph(read_wic(RAW))
    train_roots = {
        families.find(r["family_id"])
        for r in records
        if r["split"] == "train"
        and not r["family_id"].startswith(("wi:", "construction:"))
    }
    # Current contextual targets explicitly bind the target spelling to its family.
    for row in records:
        if row["split"] == "train" and "target_word" in row["state"]:
            train_roots.add(families.find(families.word(row["state"]["target_word"])))
    output = []
    for word in words:
        root = families.find(families.word(word["word"]))
        output.append(
            dict(
                word=word["word"],
                rank=word["rank"],
                frozen_historical_exposure=word["exposure"],
                candidate_supervised_target_family_seen=root in train_roots,
                current_known_family=root,
                pretraining_exposure="unknown",
                incidental_context_tokens_counted=False,
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    with (out / "words.jsonl").open("x") as stream:
        for row in output:
            stream.write(encoded(row) + "\n")
    result = dict(
        status="passed",
        words=len(output),
        candidate_supervised_target_family_counts=dict(
            Counter(
                "seen"
                if r["candidate_supervised_target_family_seen"]
                else "not_detected"
                for r in output
            )
        ),
        historical_counts=dict(
            Counter(r["frozen_historical_exposure"] for r in output)
        ),
        training_manifest_sha256=digest(training / "manifest.json"),
        vocabulary_manifest_sha256=digest(vocabulary / "manifest.json"),
        words_sha256=digest(out / "words.jsonl"),
        script_sha256=digest(__file__),
        interpretation="Known target-family exposure under this supervised training dataset only; no claim that Qwen/Kev pretraining did not contain a word. Frozen historical labels are not rewritten. No model predictions read.",
    )
    write(out / "manifest.json", result)
    print(encoded(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--training", type=Path, required=True)
    p.add_argument(
        "--vocabulary", type=Path, default=Path("artifacts/en-us-vocab-35000-v3")
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    audit(a.training, a.vocabulary, a.out)
