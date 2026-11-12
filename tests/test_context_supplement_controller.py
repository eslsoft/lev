"""Failed evaluation work is preserved and cannot be silently resubmitted."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import advance_context_supplement as controller  # noqa: E402
from lev.io import read, write  # noqa: E402


class ContextSupplementControllerTests(unittest.TestCase):
    def test_failed_command_preserved_and_no_automatic_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(controller, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "without automatic retry"):
                    controller.execute(
                        "failed", [sys.executable, "-c", "raise SystemExit(7)"]
                    )
                self.assertEqual(read(root / "failed.finished.json")["returncode"], 7)
                sentinel = root / "must-not-exist"
                with self.assertRaises(FileExistsError):
                    controller.execute(
                        "failed",
                        [sys.executable, "-c", f"open({str(sentinel)!r}, 'w').close()"],
                    )
                self.assertFalse(sentinel.exists())

    def test_only_complete_predecessor_can_release_supplement(self):
        for status in (
            "complete",
            "partial",
            "incomplete-shards",
            "awaiting-next-development-candidate",
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write(root / "result.json", dict(status=status))
                with patch.object(controller, "BASE", root):
                    self.assertEqual(controller.wait_for_base(), status == "complete")

    def test_dead_predecessor_is_not_treated_as_a_live_wait(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(controller, "BASE", Path(temp)):
                with self.assertRaisesRegex(RuntimeError, "exited without a result"):
                    controller.wait_for_base()
