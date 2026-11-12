"""Fetch a ranked candidate batch for dictionary corroboration, never automatic gold."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lev.io import digest, encoded, write
from lev.vocabulary import load_words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.limit <= 35000:
        raise ValueError("limit must be 1..35000")
    if args.offset < 0:
        raise ValueError("offset must be nonnegative")
    if shutil.disk_usage(".").free < 4 * 1024**3:
        raise RuntimeError("less than 4 GiB free; source fetch not started")
    source = Path("data/raw/en-morphology-001/eng")
    candidates = set()
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        lemma, form, _ = line.split("\t")
        lemma, form = lemma.casefold(), form.casefold()
        if lemma != form and all(s.isascii() and s.isalpha() for s in (lemma, form)):
            candidates.add(lemma)
    population = load_words("artifacts/en-us-vocab-35000-v3")
    eligible = [w["word"] for w in population if w["word"] in candidates]
    if args.offset >= len(eligible):
        raise ValueError("offset is outside candidate population")
    words = eligible[args.offset : args.offset + args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    write(
        args.out / "selection.json",
        dict(
            words=words,
            requested=len(words),
            total_candidates=len(eligible),
            candidate_offset=args.offset,
            remaining_candidates=len(eligible) - args.offset - len(words),
            source_sha256=digest(source),
            population_manifest_sha256=digest(
                "artifacts/en-us-vocab-35000-v3/manifest.json"
            ),
            policy="Ranked dictionary corroboration batch; UniMorph is a quarantined candidate inventory, not gold. No model predictions read. Offset identifies the fixed candidate slice, not successful prior downloads.",
        ),
    )

    def fetch(word):
        url = f"https://kaikki.org/dictionary/English/meaning/{word[0]}/{word[:2]}/{word}.jsonl"
        event = dict(
            word=word, url=url, retrieved_at=datetime.now(timezone.utc).isoformat()
        )
        try:
            if shutil.disk_usage(args.out).free < 4 * 1024**3:
                raise RuntimeError("less than 4 GiB free; this source was not fetched")
            with urlopen(
                Request(url, headers={"User-Agent": "LevResearch/0.1"}), timeout=45
            ) as response:
                data = response.read()
            entries = [json.loads(line) for line in data.decode().splitlines()]
            if not entries or any(
                e.get("lang_code") != "en" or e.get("word") != word for e in entries
            ):
                raise ValueError("source language/headword mismatch")
            path = args.out / f"{word}.jsonl"
            path.write_bytes(data)
            event.update(status="downloaded", bytes=len(data), sha256=digest(path))
        except HTTPError as error:
            event.update(
                status="not_found" if error.code == 404 else "failed", error=str(error)
            )
        except Exception as error:
            event.update(status="failed", error=str(error))
        return event

    counts = Counter()
    with (
        ThreadPoolExecutor(max_workers=2) as pool,
        (args.out / "downloads.jsonl").open("x") as log,
    ):
        for index, event in enumerate(pool.map(fetch, words), 1):
            log.write(encoded(event) + "\n")
            log.flush()
            counts[event["status"]] += 1
            if index % 50 == 0:
                print(
                    encoded(
                        dict(processed=index, requested=len(words), counts=dict(counts))
                    ),
                    flush=True,
                )
    write(
        args.out / "result.json",
        dict(
            status="complete" if counts["failed"] == 0 else "incomplete",
            requested=len(words),
            counts=dict(counts),
            candidate_offset=args.offset,
            remaining_candidates=len(eligible) - args.offset - len(words),
            gold_created=False,
            model_execution=False,
        ),
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
