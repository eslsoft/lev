"""Wait on the known downloader, then audit completed evidence without making labels."""

import argparse
from pathlib import Path
import subprocess
import time

from audit_corroborated_families import audit_families
from audit_morphology_dictionary import audit
from lev.io import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    directory = Path("data/raw/en-morphology-001/dictionary-check-002")
    audit_out = Path("runs/en-morphology-001-dictionary-audit-002")
    family_out = Path("runs/en-morphology-001-corroborated-family-audit-002")
    result_path = Path("runs/en-morphology-001-finish-002.json")
    if any(p.exists() for p in (audit_out, family_out, result_path)):
        raise ValueError("existing result requires explicit inspection, not overwrite")
    try:
        while not (directory / "result.json").exists():
            process = subprocess.run(
                ["ps", "-p", str(args.pid), "-o", "command="],
                capture_output=True,
                text=True,
            )
            if (
                process.returncode
                or "fetch_morphology_dictionary.py" not in process.stdout
                or str(directory) not in process.stdout
            ):
                raise RuntimeError(
                    "expected downloader no longer live and no completion record exists"
                )
            time.sleep(30)
        print("Dictionary batch complete; auditing source features.", flush=True)
        summary = audit(directory, audit_out)
        print(
            "Dictionary audit complete; checking expanded families and exposure.",
            flush=True,
        )
        families = audit_families(
            [Path("runs/en-morphology-001-dictionary-audit-001"), audit_out],
            Path("artifacts/en-joint-001"),
            Path("artifacts/en-us-vocab-35000-v3"),
            Path("artifacts/en-us-vocab-context-001"),
            family_out,
        )
        write(
            result_path,
            dict(
                status="audited_requires_review",
                dictionary_summary=summary,
                training_family_isolation_passed=families[
                    "training_family_isolation_passed"
                ],
                context_requires_revision=families["context_requires_revision"],
                family_manifest_sha256=digest(family_out / "manifest.json"),
                gold_created=False,
                final_goal_complete=False,
            ),
        )
        print(
            "Audits saved. No gold labels, training changes, or model scores produced.",
            flush=True,
        )
    except BaseException as error:
        if not result_path.exists():
            write(
                result_path,
                dict(status="failed", error=repr(error), final_goal_complete=False),
            )
        raise


if __name__ == "__main__":
    main()
