"""Score expected decisions, including omissions, and separately score complete cases."""

import math
from collections import defaultdict


def probability(value):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("probability must be a finite number in [0, 1]")
    return value


def align(records, rows):
    expected = {(r["id"], qid): q for r in records for qid, q in r["questions"].items()}
    predictions = {}
    for row in rows:
        key = (row["id"], row["question"])
        if key not in expected or key in predictions:
            raise ValueError(f"unknown or duplicate prediction: {key}")
        if row["keys"] != ["false", "true"] or row["type"] != "noul":
            raise ValueError(f"{key}: expected noul keys [false, true]")
        if type(row["label"]) is not int or row["label"] != int(expected[key]["label"]):
            raise ValueError(f"{key}: prediction label differs from frozen data")
        if not isinstance(row["p"], list) or len(row["p"]) != 2:
            raise ValueError(f"{key}: invalid distribution")
        p = [probability(v) for v in row["p"]]
        if abs(sum(p) - 1) > 1e-5:
            raise ValueError(f"{key}: probabilities do not sum to one")
        predictions[key] = p[1]
    return predictions


def rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def summarize(records, rows, threshold):
    probability(threshold)
    predictions = align(records, rows)
    items, cases = [], defaultdict(list)
    for r in records:
        for qid, q in r["questions"].items():
            p = predictions.get((r["id"], qid))
            decision = None if p is None else p >= threshold
            item = {
                "record": r["id"],
                "question": qid,
                "kind": q["kind"],
                "label": q["label"],
                "probability": p,
                "accepted": decision,
            }
            items.append(item)
            if r["scope"] == "complete":
                cases[r["case_id"]].append(item)

    def metrics(group):
        scored = [x for x in group if x["accepted"] is not None]
        positive = [x for x in group if x["label"]]
        negative = [x for x in group if not x["label"]]
        wrong_reject = sum(x["accepted"] is False for x in positive)
        wrong_accept = sum(x["accepted"] is True for x in negative)
        brier = (
            sum((x["probability"] - int(x["label"])) ** 2 for x in scored) / len(scored)
            if scored
            else None
        )
        return {
            "requested": len(group),
            "completed": len(scored),
            "missing": len(group) - len(scored),
            "accuracy_all_requested": rate(
                sum(x["accepted"] == x["label"] for x in scored), len(group)
            ),
            "correct_fact_acceptance": rate(
                sum(x["accepted"] is True for x in positive), len(positive)
            ),
            "incorrect_fact_detection": rate(
                sum(x["accepted"] is False for x in negative), len(negative)
            ),
            "false_rejections": wrong_reject,
            "false_acceptances": wrong_accept,
            "brier_completed_only": brier,
        }

    case_rows = []
    for cid, group in cases.items():
        complete = all(x["accepted"] is not None for x in group)
        correct = all(x["label"] for x in group)
        case_rows.append(
            {
                "case_id": cid,
                "expected_correct": correct,
                "execution_complete": complete,
                "accepted": all(x["accepted"] is True for x in group)
                if complete
                else None,
                "actual_error_detected": any(
                    not x["label"] and x["accepted"] is False for x in group
                ),
            }
        )
    positives = [c for c in case_rows if c["expected_correct"]]
    negatives = [c for c in case_rows if not c["expected_correct"]]
    return {
        "threshold": threshold,
        "decisions": metrics(items),
        "by_kind": {
            kind: metrics([x for x in items if x["kind"] == kind])
            for kind in sorted({x["kind"] for x in items})
        },
        "whole_cases": {
            "count": len(case_rows),
            "incomplete": sum(not c["execution_complete"] for c in case_rows),
            "correct_case_acceptance": rate(
                sum(c["accepted"] is True for c in positives), len(positives)
            ),
            "incorrect_case_rejection": rate(
                sum(c["accepted"] is False for c in negatives), len(negatives)
            ),
            "incorrect_case_error_detection": rate(
                sum(
                    c["execution_complete"] and c["actual_error_detected"]
                    for c in negatives
                ),
                len(negatives),
            ),
            "cases": case_rows,
        },
        "issues": [
            x for x in items if x["accepted"] is None or x["accepted"] != x["label"]
        ],
        "note": "Missing decisions are failures to complete, never successful rejections. Partial cases do not enter whole-case rates. Rates are point estimates, not release guarantees.",
    }
