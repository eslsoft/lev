"""Build source-derived inflection contributions with explicit exclusion records."""

from collections import Counter
from pathlib import Path

from audit_corroborated_families import verified_relations
from audit_morphology_dictionary import classify
from build_context_corpus import RAW, key, lexical_graph, read_wic, source_files
from lev.dataset import bundle
from lev.io import digest, encoded, jsonl, read, write
from lev.vocabulary import verify_bundle

DESCRIPTIONS = {
    "N;PL": "a plural form of the noun",
    "V;PRS;3;SG": "a third-person singular present-tense form of the verb",
    "V;PST": "a simple-past form of the verb",
    "V;V.PTCP;PST": "a past-participle form of the verb",
    "V;V.PTCP;PRS": "a present-participle form of the verb",
    "ADJ;CMPR": "a comparative form of the adjective",
    "ADJ;SPRL": "a superlative form of the adjective",
}


def main():
    out = Path("data/annotations/en-morphology-relations-001")
    if out.exists():
        raise ValueError("new output directory required")
    training = Path("artifacts/en-joint-002")
    context = Path("artifacts/en-us-vocab-context-002")
    records, _ = bundle(training)
    verify_bundle(context)
    families, _, _, prior = lexical_graph(read_wic(RAW))
    audits = [Path(f"runs/en-morphology-001-dictionary-audit-{n:03d}") for n in (1, 2)]
    supported = []
    dictionaries = {}
    for audit in audits:
        rows = verified_relations(audit)
        for path in read(audit / "manifest.json")["source_files"]:
            p = Path(path)
            if p.suffix == ".jsonl" and p.name != "downloads.jsonl":
                dictionaries[(str(audit), p.stem)] = p
        for row in rows:
            families.union([families.word(row["target"]), families.word(row["form"])])
            supported.append((audit, row))
    blocked = {families.find(r["family_id"]) for r in records + prior + jsonl(context / "records.jsonl") if r["split"] != "train"}
    seen, additions, events, evidence_files = set(), [], [], {}
    raw_lines = Path("data/raw/en-morphology-001/eng").read_text().splitlines()
    for audit, row in supported:
        word, form, features = row["target"], row["form"], row["features"]
        identity = (word, form, features)
        family = families.find(families.word(word))
        reason = ("qualified_variant_requires_scoped_review" if row["status"] != "corroborated_neutral"
                  else "prior_holdout_family" if family in blocked
                  else "duplicate_relation" if identity in seen else None)
        events.append(dict(audit=str(audit), source_line=row["source_line"], target=word,
                           form=form, features=features, reason=reason, included=reason is None))
        if reason:
            continue
        path = dictionaries[(str(audit), word)]
        entries = jsonl(path)
        status, support = classify(row["source_lemma"], form, features, entries)
        if status != row["status"] or support != row["dictionary_support"]:
            raise ValueError("dictionary support cannot be reproduced")
        if raw_lines[row["source_line"] - 1].split("\t") != [row["source_lemma"], form, features]:
            raise ValueError("UniMorph locator mismatch")
        neutral = next(s for s in support if not s["qualifications"])
        evidence_files[str(path)] = digest(path)
        rid = key("morphology-relations-001:" + encoded(identity))[:24]
        review = dict(status="source_derived", reviewer="structured UniMorph and Wiktionary fields; Lev deterministic transformation",
                      rationale="Both sources explicitly attest this inflection relation. Negating the attested proposition yields the false label; no missing form is treated as invalid. Sources may share upstream evidence; not independent adjudication.",
                      evidence=[dict(reference="data/raw/en-morphology-001/eng", locator=f"line {row['source_line']}"),
                                dict(reference=str(path), locator=f"entry {neutral['entry_index']}, forms[{neutral['form_index']}]")])
        positive = f"The form {form!r} is {DESCRIPTIONS[features]} {word!r}."
        negative = f"The form {form!r} is not {DESCRIPTIONS[features]} {word!r}."
        additions.append(dict(id=rid, case_id=rid, family_id=family, split="train", language="en", scope="partial", origin="curated", rubric_version="en-general-research-v1",
                              state=dict(headword=word, candidate_form=form, convention="English inflection. Judge whether the stated relation is attested; do not require it to be the only valid form or the form's only grammatical use."),
                              questions={"q0":dict(kind="inflection_relation",claim=positive,label=True,review=review),
                                         "q1":dict(kind="inflection_relation",claim=negative,label=False,review=review)}))
        seen.add(identity)
    out.mkdir()
    for name, rows in (("records.jsonl", additions), ("selection-audit.jsonl", events)):
        with (out / name).open("x") as stream:
            for row in rows:
                stream.write(encoded(row) + "\n")
    write(out / "manifest.json", dict(words=len({r['state']['headword'] for r in additions}), requests=len(additions), judgments=2*len(additions),
          selection=dict(Counter(e['reason'] or 'included' for e in events)), records_sha256=digest(out/'records.jsonl'),
          selection_sha256=digest(out/'selection-audit.jsonl'), training_manifest_sha256=digest(training/'manifest.json'),
          context_manifest_sha256=digest(context/'manifest.json'), corroboration_manifests={str(p):digest(p/'manifest.json') for p in audits},
          source_files={**source_files(), **evidence_files, 'data/raw/en-morphology-001/eng':digest('data/raw/en-morphology-001/eng')},
          script_sha256=digest(__file__), research_only=True, training_exported=False, model_scores_read=False,
          limitations="Partial lexical inflection relations, not sentence-role labels or complete word coverage. Paired logical negations are not independent hard negatives. Historical vocabulary inventory exposure must be updated before any training."))
    print(encoded(dict(requests=len(additions), words=len({r['state']['headword'] for r in additions}), selection=dict(Counter(e['reason'] or 'included' for e in events)))))


if __name__ == "__main__":
    main()
