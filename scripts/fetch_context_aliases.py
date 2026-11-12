"""Download explicit dictionary family edges for the remaining WiC candidates."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import urllib.error
import urllib.request

from build_context_corpus import RAW, lexical_graph, read_wic
from lev.io import digest, encoded, write


def main():
    out = RAW / "lexical-aliases"
    if out.exists():
        raise ValueError("new download directory required")
    rows = read_wic(RAW)
    families, reserved, _, _ = lexical_graph(rows, include_aliases=False)
    words = sorted(
        {
            r["target"]
            for r in rows
            if families.find(families.word(r["target"])) not in reserved
        }
    )
    out.mkdir()
    write(
        out / "selection.json",
        dict(
            words=words,
            policy="All WiC targets surviving the pre-alias lexical reservation; fetch before final selection. No model predictions.",
        ),
    )

    def fetch(word):
        url = f"https://kaikki.org/dictionary/English/meaning/{word[0]}/{word[:2]}/{word}.jsonl"
        event = dict(
            word=word, url=url, retrieved_at=datetime.now(timezone.utc).isoformat()
        )
        try:
            data = urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "LevResearch/0.1"}),
                timeout=45,
            ).read()
            rows = [json.loads(line) for line in data.decode().splitlines()]
            if not rows or any(
                r.get("lang_code") != "en" or r.get("word") != word for r in rows
            ):
                raise ValueError("source language/headword mismatch")
            target = out / f"{word}.jsonl"
            target.write_bytes(data)
            event.update(status="downloaded", sha256=digest(target), bytes=len(data))
        except urllib.error.HTTPError as error:
            event.update(
                status="not_found" if error.code == 404 else "failed", error=str(error)
            )
        except Exception as error:
            event.update(status="failed", error=str(error))
        return event

    with (
        ThreadPoolExecutor(max_workers=4) as pool,
        (out / "downloads.jsonl").open("x") as stream,
    ):
        events = list(pool.map(fetch, words))
        for event in events:
            stream.write(encoded(event) + "\n")
    print(
        encoded(
            dict(
                requested=len(words),
                downloaded=sum(e["status"] == "downloaded" for e in events),
                failures=[e for e in events if e["status"] != "downloaded"],
            )
        )
    )
    if any(e["status"] == "failed" for e in events):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
