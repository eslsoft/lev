"""Fetch fixed public source snapshots; preserve failures, licenses and SHA-256."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

SOURCES = {
    "wordnet.xml.gz": "https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025.xml.gz",
    "wordnet-LICENSE": "https://raw.githubusercontent.com/globalwordnet/english-wordnet/45fdb60396dc804526b85f21a23c59983c2c396a/LICENSE.md",
    "ud.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/en_ewt-ud-train.conllu",
    "ud-LICENSE": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/LICENSE.txt",
    "cmudict.dict": "https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict",
    "cmudict-LICENSE": "https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/LICENSE",
    "blimp-README": "https://raw.githubusercontent.com/alexwarstadt/blimp/3e56b06fcabca9b30822fc66435fca6b1aa40bb1/README.md",
}
BLIMP = [
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_irregular_1",
    "regular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_1",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "npi_present_1",
    "sentential_negation_npi_licensor_present",
    "principle_A_domain_1",
    "existential_there_quantifiers_1",
]
for name in BLIMP:
    SOURCES[f"blimp/{name}.jsonl"] = (
        f"https://raw.githubusercontent.com/alexwarstadt/blimp/3e56b06fcabca9b30822fc66435fca6b1aa40bb1/data/{name}.jsonl"
    )


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "LevResearch/0.1"}),
            timeout=120,
        ) as response,
        target.open("xb") as out,
    ):
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    log = root / "download-events.jsonl"
    for name, url in SOURCES.items():
        entry = {
            "file": name,
            "url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            entry["sha256"] = fetch(url, root / name)
            entry["status"] = "downloaded"
        except Exception as error:
            entry.update(status="failed", error=str(error))
        with log.open("a") as out:
            out.write(json.dumps(entry) + "\n")
        print(name, entry["status"], flush=True)
    if any(
        json.loads(line)["status"] != "downloaded"
        for line in log.read_text().splitlines()
    ):
        raise SystemExit(
            "Source download failures preserved; resolve before corpus construction"
        )


if __name__ == "__main__":
    main()
