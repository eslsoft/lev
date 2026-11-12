"""Word-level scoring for a frozen test-only vocabulary suite, not full-entry mastery."""

from collections import defaultdict
from pathlib import Path

from .io import digest, jsonl, read
from .report import align, probability, rate


def verify_bundle(directory):
    root = Path(directory)
    manifest = read(root / "manifest.json")
    if (
        manifest.get("format") != "lev-vocabulary-test-v1"
        or manifest.get("test_only") is not True
    ):
        raise ValueError("expected test-only vocabulary manifest")
    required = {
        "words.jsonl",
        "records.jsonl",
        "test.jsonl",
        "rubric.json",
        "coverage.json",
        "source-manifest.json",
    }
    if not required <= manifest["files"].keys():
        raise ValueError("incomplete vocabulary bundle")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or digest(root / name) != expected:
            raise ValueError(f"vocabulary bundle modified: {name}")
    return manifest


def score_words(words, records, rows, threshold=0.5):
    probability(threshold)
    if len({w["word"] for w in words}) != len(words):
        raise ValueError("duplicate target word")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate request")
    expected_ids = [rid for w in words for rid in w["record_ids"]]
    if len(set(expected_ids)) != len(expected_ids) or set(expected_ids) != {
        r["id"] for r in records
    }:
        raise ValueError("word coverage and expected requests disagree")
    predictions = align(records, rows)
    by_id = {r["id"]: r for r in records}
    details = []
    by_kind = defaultdict(lambda: dict(expected=0, completed=0, correct=0))
    for word in words:
        expected = completed = correct = 0
        false_acceptances = false_rejections = 0
        positive_labels = negative_labels = 0
        failures = []
        for rid in word["record_ids"]:
            record = by_id[rid]
            if record["state"]["headword"] != word["word"]:
                raise ValueError("request assigned to wrong word")
            for qid, q in record["questions"].items():
                value = predictions.get((rid, qid))
                passed = value is not None and (value >= threshold) == q["label"]
                expected += 1
                completed += value is not None
                correct += passed
                positive_labels += q["label"] is True
                negative_labels += q["label"] is False
                false_acceptances += (
                    value is not None and value >= threshold and not q["label"]
                )
                false_rejections += (
                    value is not None and value < threshold and q["label"]
                )
                metric = by_kind[q["kind"]]
                metric["expected"] += 1
                metric["completed"] += value is not None
                metric["correct"] += passed
                if not passed:
                    failures.append(
                        dict(
                            id=rid,
                            question=qid,
                            probability_true=value,
                            reason="missing" if value is None else "incorrect",
                        )
                    )
        if not expected or expected != word["judgment_count"]:
            raise ValueError("word judgment count disagrees with frozen questions")
        details.append(
            dict(
                word=word["word"],
                rank=word["rank"],
                exposure=word["exposure"],
                expected=expected,
                completed=completed,
                correct=correct,
                false_acceptances=false_acceptances,
                false_rejections=false_rejections,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                all_scheduled_items_correct=correct == expected,
                accuracy=correct / expected,
                failures=failures,
                coverage=word["coverage"],
                gaps=word["gaps"],
                complete_entry_reviewed=False,
            )
        )

    def summary(group):
        expected = sum(w["expected"] for w in group)
        completed = sum(w["completed"] for w in group)
        return dict(
            words=len(group),
            words_fully_executed=sum(w["completed"] == w["expected"] for w in group),
            words_all_scheduled_items_correct=rate(
                sum(w["all_scheduled_items_correct"] for w in group), len(group)
            ),
            word_macro_accuracy=sum(w["accuracy"] for w in group) / len(group)
            if group
            else None,
            judgment_accuracy=rate(sum(w["correct"] for w in group), expected),
            expected_judgments=expected,
            completed_judgments=completed,
            missing_judgments=expected - completed,
            false_acceptances=sum(w["false_acceptances"] for w in group),
            false_rejections=sum(w["false_rejections"] for w in group),
            positive_labels=sum(w["positive_labels"] for w in group),
            negative_labels=sum(w["negative_labels"] for w in group),
        )

    return dict(
        status="not_run"
        if not predictions
        else "complete"
        if len(predictions) == sum(w["expected"] for w in details)
        else "partial",
        threshold=threshold,
        overall=summary(details),
        by_exposure={
            key: summary([w for w in details if w["exposure"] == key])
            for key in sorted({w["exposure"] for w in details})
        },
        by_rank_band={
            f"{low + 1}-{min(low + 5000, len(words))}": summary(
                [w for w in details if low < w["rank"] <= low + 5000]
            )
            for low in range(0, len(words), 5000)
        },
        by_kind=dict(by_kind),
        words=details,
        interpretation="Passing all scheduled inventory items is not proof of full word knowledge or complete-entry review. Missing questions never pass.",
    )


def load_words(directory):
    manifest = verify_bundle(directory)
    words = jsonl(Path(directory) / "words.jsonl")
    if len(words) != manifest["unique_headwords"] or len(
        {w["word"] for w in words}
    ) != len(words):
        raise ValueError("word count differs from manifest")
    if sorted(w["rank"] for w in words) != list(range(1, len(words) + 1)):
        raise ValueError("invalid word ranks")
    return words
