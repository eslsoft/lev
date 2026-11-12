"""Independently close every published form edge, including shared irregular forms."""

import argparse
from collections import defaultdict
import gzip
from pathlib import Path
import xml.etree.ElementTree as ET

from lev.io import jsonl, write


class Families:
    def __init__(self, raw):
        self.parents = {}
        lex = ET.parse(gzip.open(raw / "wordnet.xml.gz")).getroot().find("Lexicon")
        entries = {
            e.attrib["id"]: e.find("Lemma").attrib["writtenForm"]
            for e in lex.findall("LexicalEntry")
        }
        senses = {
            s.attrib["id"]: entries[e.attrib["id"]]
            for e in lex.findall("LexicalEntry")
            for s in e.findall("Sense")
        }
        self.members = {
            s.attrib["id"]: [
                entries[m] for m in s.attrib.get("members", "").split() if m in entries
            ]
            for s in lex.findall("Synset")
        }
        for e in lex.findall("LexicalEntry"):
            lemma = entries[e.attrib["id"]]
            self.union(
                [
                    self.word(lemma),
                    *[self.word(f.attrib["writtenForm"]) for f in e.findall("Form")],
                ]
            )
            for s in e.findall("Sense"):
                for rel in s.findall("SenseRelation"):
                    if (
                        rel.attrib["relType"] == "derivation"
                        and rel.attrib["target"] in senses
                    ):
                        self.union(
                            [self.word(lemma), self.word(senses[rel.attrib["target"]])]
                        )
        for file in (raw / "wiktionary").glob("*.jsonl"):
            for entry in jsonl(file):
                self.union(
                    [
                        self.word(file.stem),
                        *[
                            self.word(f["form"])
                            for f in entry.get("forms", [])
                            if f.get("form", "").isascii()
                            and f.get("form", "").isalpha()
                        ],
                    ]
                )
        self.ud = {}
        for block in (raw / "ud.conllu").read_text().split("\n\n"):
            lines = block.splitlines()
            sid = next(
                (
                    line.removeprefix("# sent_id = ")
                    for line in lines
                    if line.startswith("# sent_id = ")
                ),
                None,
            )
            if sid:
                self.ud[sid] = {
                    line.split("\t")[0]: line.split("\t")
                    for line in lines
                    if line
                    and not line.startswith("#")
                    and line.split("\t")[0].isdigit()
                }

    @staticmethod
    def word(value):
        return "lex:" + value.casefold()

    def find(self, item):
        self.parents.setdefault(item, item)
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def union(self, items):
        roots = sorted({self.find(item) for item in items})
        for root in roots[1:]:
            self.parents[root] = roots[0]

    def attach(self, records):
        for row in records:
            evidence = next(iter(row["questions"].values()))["review"]["evidence"][0]
            source, locator = evidence["reference"], evidence["locator"]
            anchors = [row["family_id"]]
            if source == "wordnet.xml.gz":
                for sid in locator.split("->"):
                    anchors.extend(self.word(m) for m in self.members[sid])
            elif source == "cmudict.dict":
                anchors.append(self.word(row["state"]["word"]))
            elif source == "ud.conllu":
                sid, index = locator.split(":token=")
                anchors.extend(
                    [
                        self.word(self.ud[sid][index][2]),
                        "document:" + sid.rsplit("-", 1)[0],
                    ]
                )
            elif source.startswith("wiktionary/"):
                anchors.append(self.word(row["state"]["headword"]))
            self.union(anchors)

    def audit(self, prior, current):
        self.attach(prior + current)
        roots = defaultdict(list)
        for origin, rows in [("prior", prior), ("current", current)]:
            for row in rows:
                roots[self.find(row["family_id"])].append(
                    dict(
                        origin=origin,
                        id=row["id"],
                        family=row["family_id"],
                        split=row["split"],
                    )
                )
        violations = []
        for root, rows in roots.items():
            current_splits = {r["split"] for r in rows if r["origin"] == "current"}
            prior_splits = {r["split"] for r in rows if r["origin"] == "prior"}
            reasons = []
            if len(current_splits) > 1:
                reasons.append("current splits share a known lexical family")
            if "train" in prior_splits and current_splits - {"train"}:
                reasons.append("prior train reaches current holdout")
            if prior_splits - {"train"} and current_splits:
                reasons.append("prior holdout reaches current data")
            if reasons:
                violations.append(dict(root=root, reasons=reasons, rows=rows))
        return dict(
            status="failed" if violations else "passed",
            violations=violations,
            components=len(roots),
            policy="Independent union of every explicit WordNet form/derivation, retained Wiktionary form, relation members, UD target/document and declared family. Incidental sentence vocabulary excluded.",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/en-research-20261021-a", type=Path)
    p.add_argument(
        "--prior", default="data/research/en-general-002/records.jsonl", type=Path
    )
    p.add_argument("--current", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    result = Families(a.raw).audit(jsonl(a.prior), jsonl(a.current))
    write(a.out, result)
    print(result["status"], len(result["violations"]), "family violations")
    if result["violations"]:
        raise SystemExit(1)
