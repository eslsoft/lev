import importlib.util
from pathlib import Path
import sys
import unittest
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "generalization", ROOT / "scripts/build_generalization_corpus.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from advance_generalization_cycle import select_candidate  # noqa: E402


class GeneralizationTests(unittest.TestCase):
    def test_selection_cannot_trade_grammar_or_missing_outputs_for_macro_gain(self):
        base = {
            "decisions": {"missing": 0},
            "macro_accuracy": 0.75,
            "by_kind": {"grammaticality": {"accuracy_all_requested": {"rate": 0.62}}},
        }
        candidate = deepcopy(base)
        candidate["macro_accuracy"] = 0.80
        reports = {"lev01": base, "lev02": candidate}
        self.assertTrue(select_candidate(reports))
        candidate["by_kind"]["grammaticality"]["accuracy_all_requested"]["rate"] = 0.61
        self.assertFalse(select_candidate(reports))
        candidate["by_kind"]["grammaticality"]["accuracy_all_requested"]["rate"] = 0.62
        candidate["decisions"]["missing"] = 1
        self.assertFalse(select_candidate(reports))
        candidate["decisions"]["missing"] = 0
        candidate["macro_accuracy"] = 0.75
        self.assertFalse(select_candidate(reports))

    def test_question_polarity_is_independent_of_original_truth(self):
        for split in ("train", "development", "calibration", "test"):
            forms = set()
            for i in range(100):
                claim, true_label = module.phrasing(
                    "Is A a subtype of B?", True, split, str(i)
                )
                other, false_label = module.phrasing(
                    "Is A a subtype of B?", False, split, str(i)
                )
                self.assertEqual(claim, other)
                self.assertIs(true_label, not false_label)
                forms.add(true_label)
            self.assertEqual(forms, {True, False})

    def test_prior_grammar_holdouts_never_enter_new_training(self):
        for term in ("anaphor_agreement", "binding", "quantifiers"):
            self.assertEqual(module.GRAMMAR_SPLITS[term], "excluded")
        self.assertEqual(module.GRAMMAR_SPLITS["island_effects"], "development")
        self.assertEqual(module.GRAMMAR_SPLITS["ellipsis"], "test")
