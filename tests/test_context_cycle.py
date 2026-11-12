from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from advance_context_cycle import gate  # noqa: E402


def report(accuracy, passage, missing=0):
    return dict(
        decisions=dict(accuracy_all_requested=dict(rate=accuracy), missing=missing),
        by_kind=dict(
            passage_correction_acceptability=dict(
                accuracy_all_requested=dict(rate=passage)
            )
        ),
    )


class ContextSelectionTests(unittest.TestCase):
    def test_total_improvement_cannot_hide_passage_or_retention_regression(self):
        incumbent = report(0.7, 0.7)
        self.assertFalse(
            gate(report(0.8, 0.69), incumbent, report(0.9, 0.9), report(0.9, 0.9))
        )
        self.assertFalse(
            gate(report(0.8, 0.8), incumbent, report(0.88, 0.88), report(0.9, 0.9))
        )
        self.assertFalse(
            gate(deepcopy(incumbent), incumbent, report(0.9, 0.9), report(0.9, 0.9))
        )
        self.assertTrue(
            gate(report(0.8, 0.8), incumbent, report(0.895, 0.895), report(0.9, 0.9))
        )

    def test_missing_predictions_never_pass_selection(self):
        reports = [
            report(0.8, 0.8),
            report(0.7, 0.7),
            report(0.9, 0.9),
            report(0.9, 0.9),
        ]
        for i in range(4):
            trial = deepcopy(reports)
            trial[i]["decisions"]["missing"] = 1
            self.assertFalse(gate(*trial))


if __name__ == "__main__":
    unittest.main()
