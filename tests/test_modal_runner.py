import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

from lev.dataset import prepare
from lev.modal_runner import preflight, snapshot_kev, snapshot_archive
import tarfile
from lev.upstream import checkout

ROOT = Path(__file__).resolve().parents[1]


class ModalRunnerTests(unittest.TestCase):
    def test_example_bundle_is_rejected_before_modal_import_or_access(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / "dataset"
            prepare(
                ROOT / "data/examples/format-only.jsonl",
                ROOT / "rubrics/en-v1.json",
                dataset,
                examples=True,
            )
            args = argparse.Namespace(
                operation="train",
                run_id="probe",
                timeout=60,
                config=ROOT / "configs/lev-4b.toml",
                dataset=dataset,
                kev=None,
                checkpoint=None,
            )
            with self.assertRaisesRegex(ValueError, "example-only"):
                preflight(args)
            args.run_id = "../escape"
            with self.assertRaisesRegex(ValueError, "run-id"):
                preflight(args)

    def test_snapshot_preserves_commit_without_local_secrets_or_untracked_files(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(source), *args], text=True
                ).strip()

            git("init", "-q")
            (source / "uv.lock").write_text("test-lock")
            git("add", "uv.lock")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            )
            commit = git("rev-parse", "HEAD")
            git("config", "http.extraHeader", "SECRET_FIXTURE")
            (source / ".env").write_text("SECRET_FIXTURE")
            target = Path(temp) / "export"
            snapshot_kev(source, target, commit)
            self.assertEqual(checkout(target, commit)["commit"], commit)
            self.assertEqual((target / "uv.lock").read_text(), "test-lock")
            self.assertFalse((target / ".env").exists())
            self.assertNotIn("SECRET_FIXTURE", (target / ".git/config").read_text())
            archive = snapshot_archive(target, Path(temp) / "snapshot.tar.gz")
            extracted = Path(temp) / "extracted"
            extracted.mkdir()
            with tarfile.open(archive) as stream:
                self.assertTrue(
                    all(m.uid == 0 and m.gid == 0 for m in stream.getmembers())
                )
                stream.extractall(extracted, filter="data")
            self.assertEqual(checkout(extracted, commit)["commit"], commit)
            self.assertFalse((extracted / ".env").exists())
            (source / "uv.lock").write_text("dirty")
            with self.assertRaisesRegex(ValueError, "tracked changes"):
                snapshot_kev(source, Path(temp) / "dirty-export", commit)
