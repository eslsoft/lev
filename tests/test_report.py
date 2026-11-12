import copy
import math
import unittest

from lev.report import summarize


def record(rid, labels, scope="complete", case=None):
    return {
        "id": rid,
        "case_id": case or rid,
        "scope": scope,
        "questions": {
            f"q{i}": {"kind": "sense", "label": label} for i, label in enumerate(labels)
        },
    }


def prediction(r, probabilities):
    return [
        {
            "id": r["id"],
            "question": qid,
            "keys": ["false", "true"],
            "type": "noul",
            "label": int(q["label"]),
            "p": [1 - p, p],
        }
        for (qid, q), p in zip(r["questions"].items(), probabilities)
    ]


class ReportTests(unittest.TestCase):
    def test_whole_cases_distinct_from_decisions_and_partial_records(self):
        good = record("good", [True, True])
        bad = record("bad", [True, False])
        partial = record("partial", [True], "partial")
        result = summarize(
            [good, bad, partial],
            prediction(good, [0.99, 0.7])
            + prediction(bad, [0.9, 0.1])
            + prediction(partial, [0.99]),
            0.8,
        )
        self.assertEqual(result["decisions"]["accuracy_all_requested"]["rate"], 0.8)
        self.assertEqual(result["whole_cases"]["count"], 2)
        self.assertEqual(result["whole_cases"]["correct_case_acceptance"]["rate"], 0)
        self.assertEqual(
            result["whole_cases"]["incorrect_case_error_detection"]["rate"], 1
        )

    def test_rejecting_a_correct_item_is_not_detecting_actual_error(self):
        bad = record("bad", [True, False])
        result = summarize([bad], prediction(bad, [0.1, 0.9]), 0.8)["whole_cases"]
        self.assertEqual(result["incorrect_case_rejection"]["rate"], 1)
        self.assertEqual(result["incorrect_case_error_detection"]["rate"], 0)

    def test_missing_results_are_not_successful_rejections(self):
        bad = record("bad", [False, False])
        result = summarize([bad], prediction(bad, [0.1]), 0.8)
        self.assertEqual(result["decisions"]["missing"], 1)
        self.assertEqual(result["decisions"]["accuracy_all_requested"]["rate"], 0.5)
        self.assertIsNone(result["whole_cases"]["cases"][0]["accepted"])
        self.assertEqual(result["whole_cases"]["incorrect_case_rejection"]["rate"], 0)
        self.assertEqual(
            result["whole_cases"]["incorrect_case_error_detection"]["rate"], 0
        )

    def test_case_spanning_requests_must_pass_every_request(self):
        a, b = record("a", [True], case="word"), record("b", [True], case="word")
        result = summarize([a, b], prediction(a, [0.99]) + prediction(b, [0.7]), 0.8)
        self.assertEqual(result["whole_cases"]["count"], 1)
        self.assertEqual(result["whole_cases"]["correct_case_acceptance"]["rate"], 0)

    def test_invalid_predictions_fail_explicitly(self):
        r = record("r", [True])
        valid = prediction(r, [0.9])
        for field, value in [
            ("id", "unknown"),
            ("label", 0),
            ("p", [0.5, 0.8]),
            ("p", [0, float("nan")]),
            ("keys", ["true", "false"]),
        ]:
            rows = copy.deepcopy(valid)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                summarize([r], rows, 0.8)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([r], valid * 2, 0.8)
        with self.assertRaises(ValueError):
            summarize([r], valid, math.nan)

    def test_no_complete_cases_does_not_claim_a_whole_word_rate(self):
        r = record("r", [True], "partial")
        result = summarize([r], prediction(r, [0.99]), 0.8)
        self.assertIsNone(result["whole_cases"]["correct_case_acceptance"]["rate"])
