"""Derive contextual morphology judgments from explicit UD feature annotations."""

from collections import Counter
from pathlib import Path

from audit_corroborated_families import verified_relations
from build_context_corpus import RAW, lexical_graph, read_wic, key, source_files
from lev.dataset import bundle, validate, exportable, to_kev
from lev.io import digest, encoded, jsonl, read, write

PAIRS = {
    "Number": {"Sing": ("singular number", "plural number"), "Plur": ("plural number", "singular number")},
    "Tense": {"Past": ("past tense", "present tense"), "Pres": ("present tense", "past tense")},
    "VerbForm": {"Fin": ("a finite verb form", "an infinitive verb form"), "Inf": ("an infinitive verb form", "a finite verb form"),
                 "Part": ("a participial verb form", "a finite verb form"), "Ger": ("a gerund verb form", "a finite verb form")},
}


def main():
    out = Path("data/annotations/en-ud-morphology-001")
    if out.exists():
        raise ValueError("new output required")
    parent = Path("artifacts/en-joint-002")
    records, _ = bundle(parent)
    families, _, _, prior = lexical_graph(read_wic(RAW))
    for n in (1, 2):
        for row in verified_relations(Path(f"runs/en-morphology-001-dictionary-audit-{n:03d}")):
            families.union([families.word(row["target"]), families.word(row["form"])])
    heldout = [r for r in records + prior + jsonl("artifacts/en-us-vocab-context-002/records.jsonl") if r["split"] != "train"]
    blocked = {families.find(r["family_id"]) for r in heldout}
    vocabulary = {r["word"] for r in jsonl("artifacts/en-us-vocab-35000-v3/words.jsonl")}
    source = Path("data/raw/en-research-20261021-a/ud.conllu")
    candidates, events = [], []
    for sid, tokens in sorted(families.ud.items()):
        doc = "document:" + sid.rsplit("-", 1)[0]
        for index, token in tokens.items():
            lemma, form = token[2].casefold(), token[1]
            feats = dict(item.split("=", 1) for item in token[5].split("|") if "=" in item)
            options = [(f, v) for f, v in feats.items() if f in PAIRS and v in PAIRS[f]]
            reason = ("outside_target_vocabulary" if lemma not in vocabulary
                      else "no_supported_explicit_feature" if not options
                      else "prior_holdout_family_or_document" if any(families.find(x) in blocked for x in (families.word(lemma), families.word(form), doc))
                      else None)
            events.append(dict(locator=sid+":token="+index, lemma=lemma, reason=reason, included=reason is None))
            if reason is None:
                feature, value = min(options, key=lambda p: key(sid+index+p[0]))
                candidates.append((sid, index, tokens, lemma, form, doc, feature, value))
    # Close all admitted target/document edges together before assigning families.
    for _, _, _, lemma, form, doc, _, _ in candidates:
        families.union([families.word(lemma), families.word(form), doc])
    blocked = {families.find(r["family_id"]) for r in heldout}
    additions = []
    for sid, index, tokens, lemma, form, doc, feature, value in candidates:
        family = families.find(doc)
        if family in blocked:
            raise ValueError("new target/document closure reached holdout")
        correct, wrong = PAIRS[feature][value]
        claims = [(f"The target token has {correct} under the stated annotation convention.", True),
                  (f"The target token has {wrong} under the stated annotation convention.", False),
                  (f"The target token does not have {correct} under the stated annotation convention.", False),
                  (f"The target token does not have {wrong} under the stated annotation convention.", True)]
        locator = sid+":token="+index
        claims.sort(key=lambda pair: key(locator+pair[0]))
        review = dict(status="source_derived", reviewer="Universal Dependencies explicit FEATS annotation; deterministic Lev conversion",
                      rationale=f"The source token has {feature}={value}. The contrasted value is mutually exclusive for this explicitly single-valued source feature. Negated claims invert those labels. This is source annotation agreement, not independent Lev adjudication.",
                      evidence=[dict(reference=str(source), locator=locator)])
        rid = key("ud-morphology-001:"+locator)[:24]
        additions.append(dict(id=rid, case_id=rid, family_id=family, split="train", language="en", scope="partial", origin="curated", rubric_version="en-general-research-v1",
                              state=dict(tokens=[dict(index=int(t[0]),text=t[1]) for t in tokens.values()], target_index=int(index), convention="Universal Dependencies English EWT morphological analysis of this token in this sentence; evaluate this occurrence, not every possible use of the word."),
                              questions={f"q{i}":dict(kind="contextual_morphology",claim=claim,label=label,review=review) for i,(claim,label) in enumerate(claims)}))
    rubric = read("rubrics/en-general-research-v1.json")
    validate(additions, rubric)
    exportable(additions, rubric, examples=False, research=True)
    for row in additions:
        if len(to_kev(row, rubric)["questions"]) != 4:
            raise ValueError("export projection lost a judgment")
    out.mkdir()
    for name, rows in (("records.jsonl", additions), ("selection-audit.jsonl", events)):
        with (out/name).open("x") as stream:
            for row in rows:
                stream.write(encoded(row)+"\n")
    summary = dict(requests=len(additions), judgments=4*len(additions), target_words=len({c[3] for c in candidates}),
                   selection=dict(Counter(e['reason'] or 'included' for e in events)), records_sha256=digest(out/'records.jsonl'),
                   source_sha256=digest(source), family_source_files=source_files(), script_sha256=digest(__file__),
                   parent_manifest_sha256=digest(parent/'manifest.json'), training_exported=False,
                   note="Explicit source-feature labels, partial contextual coverage. This separate task does not repair the inflection_relation polarity shortcut in joint003, which remains unapproved for training.")
    write(out/'manifest.json',summary)
    print(encoded({k:v for k,v in summary.items() if k!='family_source_files'}))


if __name__ == "__main__":
    main()
