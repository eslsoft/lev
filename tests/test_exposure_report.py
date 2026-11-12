"""Validate exposure/report identity and preserve missing-result denominators."""

from pathlib import Path
import tempfile
import unittest

from lev.exposure_report import summarize_exposure
from lev.io import digest, encoded, fingerprint, read, write


class ExposureReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.exposure = root / "exposure"
        self.checkpoint = root / "checkpoint"
        self.report = root / "report.json"
        self.exposure.mkdir()
        self.checkpoint.mkdir()
        # Identity fixture only, never loaded as model weights.
        (self.checkpoint / "head.pt").write_bytes(b"identity fixture")
        write(self.checkpoint / "lev-model.json", {"dataset_sha256": "training-a"})
        rows = [
            dict(
                word="bank",
                rank=1,
                frozen_historical_exposure="prior_train",
                candidate_supervised_target_family_seen=True,
            ),
            dict(
                word="cat",
                rank=2,
                frozen_historical_exposure="no_detected_prior_target_family",
                candidate_supervised_target_family_seen=False,
            ),
        ]
        (self.exposure / "words.jsonl").write_text(
            "".join(encoded(r) + "\n" for r in rows)
        )
        write(
            self.exposure / "manifest.json",
            dict(
                status="passed",
                words=2,
                words_sha256=digest(self.exposure / "words.jsonl"),
                vocabulary_manifest_sha256="vocabulary-a",
                training_manifest_sha256="training-a",
            ),
        )
        write(
            self.report,
            dict(
                status="partial",
                dataset_manifest_sha256="vocabulary-a",
                inference_identity={"checkpoint_files": fingerprint(self.checkpoint)},
                words=[
                    dict(
                        word=r["word"],
                        rank=r["rank"],
                        exposure=r["frozen_historical_exposure"],
                        expected=2,
                        completed=1,
                        correct=1,
                        all_scheduled_items_correct=False,
                        false_acceptances=0,
                        false_rejections=0,
                    )
                    for r in rows
                ],
            ),
        )

    def test_missing_remains_in_both_group_denominators(self):
        result = summarize_exposure(self.report, self.exposure, self.checkpoint)
        self.assertEqual(result["status"], "partial")
        for group in result["by_candidate_supervised_target_exposure"].values():
            self.assertEqual(group["judgment_accuracy"]["rate"], 0.5)
            self.assertEqual(group["missing_judgments"], 1)
            self.assertEqual(group["words_all_scheduled_items_correct"]["numerator"], 0)

    def test_wrong_training_identity_rejected(self):
        path = self.exposure / "manifest.json"
        manifest = read(path)
        manifest["training_manifest_sha256"] = "training-b"
        path.write_text(encoded(manifest))
        with self.assertRaisesRegex(ValueError, "another training dataset"):
            summarize_exposure(self.report, self.exposure, self.checkpoint)

    def test_missing_words_and_changed_checkpoint_rejected(self):
        report = read(self.report)
        report["words"].pop()
        self.report.write_text(encoded(report))
        with self.assertRaisesRegex(ValueError, "populations differ"):
            summarize_exposure(self.report, self.exposure, self.checkpoint)
        (self.checkpoint / "head.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            summarize_exposure(self.report, self.exposure, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
