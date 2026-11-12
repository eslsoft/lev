"""Snapshot Wiktionary evidence for core function words and high-prevalence gaps."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request
from lev.io import digest, encoded, jsonl, write

CORE = """a an the this that these those each every either neither any some no all both enough few fewer less little many more most much several such another other what whatever which whichever whose my your his her its our their mine yours hers ours theirs i you he she it we they me him us them myself yourself himself herself itself ourselves yourselves themselves who whom whoever whomever one ones nobody somebody anybody everybody nothing something anything everything none anyone someone everyone anywhere somewhere nowhere here there where when why how whether if unless until while because since although though whereas as than and or but nor so yet for with without within at by from in into of off on onto out over under up down upon about above across after against along among around before behind below beneath beside besides between beyond despite during except inside near outside past per through throughout till toward towards underneath unlike versus via about aboard amid amongst lest once whenever wherever however therefore thus hence also even only just not never ever always often sometimes can could may might must shall should will would do does did have has had be am is are was were been being to ought dare need yes yeah nope oh hello please thanks""".split()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/en-us-vocab-35000-v1/selection-audit.jsonl"),
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    candidates = sorted(
        set(CORE)
        | {
            r["word"]
            for r in jsonl(a.audit)
            if r["population_rank"] <= 5000
            and r["selection"] == "no_lowercase_single_word_OEWN_definition"
        }
    )
    write(
        a.out / "selection.json",
        {
            "core": CORE,
            "candidates": candidates,
            "policy": "Fixed core function words plus all excluded original US-prevalence top-5000 entries. No model scores.",
        },
    )

    def fetch(word):
        url = f"https://kaikki.org/dictionary/English/meaning/{word[0]}/{word[:2]}/{word}.jsonl"
        event = dict(
            word=word, url=url, retrieved_at=datetime.now(timezone.utc).isoformat()
        )
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "LevResearch/0.1"}),
                timeout=45,
            ) as response:
                data = response.read()
            records = [json.loads(s) for s in data.decode().splitlines()]
            if not records or any(
                r.get("word") != word or r.get("lang_code") != "en" for r in records
            ):
                raise ValueError("response does not match requested English headword")
            file = a.out / (word + ".jsonl")
            file.write_bytes(data)
            event.update(status="downloaded", sha256=digest(file))
        except urllib.error.HTTPError as error:
            event.update(
                status="not_found" if error.code == 404 else "failed", error=str(error)
            )
        except Exception as error:
            event.update(status="failed", error=str(error))
        return event

    failures = []
    with (
        ThreadPoolExecutor(max_workers=4) as pool,
        (a.out / "downloads.jsonl").open("x") as log,
    ):
        for i, event in enumerate(pool.map(fetch, candidates), 1):
            log.write(encoded(event) + "\n")
            log.flush()
            if event["status"] == "failed":
                failures.append(event)
            if i % 50 == 0:
                print(i, "/", len(candidates), "failures", len(failures), flush=True)
    print("Finished", len(candidates), "failures", len(failures), flush=True)
    if failures:
        raise SystemExit("Failures preserved; resolve before using the source set")


if __name__ == "__main__":
    main()
