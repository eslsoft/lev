"""Locate original example contexts for supported forms, without assigning labels."""

from collections import Counter
from pathlib import Path

from audit_corroborated_families import verified_relations
from lev.io import digest, encoded, jsonl, write


def main():
    out = Path("runs/en-morphology-001-context-candidates-001")
    out.mkdir()
    counts = Counter()
    words = set()
    sources = {}
    seen = set()
    with (out / "candidates.jsonl").open("x") as stream:
        for batch in ("001", "002"):
            audit = Path(f"runs/en-morphology-001-dictionary-audit-{batch}")
            sources[str(audit / "manifest.json")] = digest(audit / "manifest.json")
            cache = {}
            for row in verified_relations(audit):
                key = (row["target"], row["form"], row["features"])
                if key in seen:
                    continue
                seen.add(key)
                word = row["target"]
                path = Path(
                    f"data/raw/en-morphology-001/dictionary-check-{batch}/{word}.jsonl"
                )
                if word not in cache:
                    cache[word] = jsonl(path)
                examples = []
                entry_indices = sorted(
                    {s["entry_index"] for s in row["dictionary_support"]}
                )
                for entry_index in entry_indices:
                    entry = cache[word][entry_index]
                    for sense_index, sense in enumerate(entry.get("senses", [])):
                        for example_index, example in enumerate(
                            sense.get("examples", [])
                        ):
                            # Quotations can carry separate copyright and archaic usage.
                            if example.get("type") != "example":
                                continue
                            text = example.get("text", "")
                            spans = example.get("bold_text_offsets", [])
                            matches = [
                                span
                                for span in spans
                                if len(span) == 2
                                and 0 <= span[0] < span[1] <= len(text)
                                and text[span[0] : span[1]] == row["form"]
                            ]
                            if matches:
                                examples.append(
                                    dict(
                                        entry_index=entry_index,
                                        sense_index=sense_index,
                                        example_index=example_index,
                                        text=text,
                                        spans=matches,
                                        sense_tags=sense.get("tags", []),
                                        sense_raw_tags=sense.get("raw_tags", []),
                                        glosses=sense.get("glosses", []),
                                    )
                                )
                status = (
                    "context_candidate_requires_review"
                    if examples
                    else "no_exact_example"
                )
                counts[status] += 1
                if examples:
                    words.add(word)
                stream.write(
                    encoded(
                        dict(
                            **row,
                            context_status=status,
                            dictionary_path=str(path),
                            dictionary_sha256=digest(path),
                            examples=examples,
                            label=None,
                            review_status="draft",
                            note="Exact form occurrence does not establish its grammatical feature in this sentence.",
                        )
                    )
                    + "\n"
                )
    write(
        out / "summary.json",
        dict(
            relation_counts=dict(counts),
            words_with_context_candidates=len(words),
            gold_created=False,
            model_execution=False,
            interpretation="Source-context candidates only; preserve all unsupported-context rows. No negative labels inferred from absence.",
        ),
    )
    write(
        out / "manifest.json",
        dict(
            source_audits=sources,
            script_sha256=digest(__file__),
            outputs={p.name: digest(p) for p in out.iterdir() if p.is_file()},
        ),
    )
    print(encoded(dict(counts)))


if __name__ == "__main__":
    main()
