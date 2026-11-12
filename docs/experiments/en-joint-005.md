# Contextual morphology development experiment

This research bundle extends `en-joint-002` with explicitly annotated contextual
morphology. It is not a new final acceptance set and has not been used to train a
model yet. The existing v05 job continues to use `en-joint-002`.

## Frozen data

`artifacts/en-joint-005/manifest.json` SHA-256:
`b41902b1c5974f10fe674557a9de74d45977157c2d89768019a6983cb2180d56`.

- Training: 31,173 requests and 84,587 judgments. This includes 17,268 new
  EWT occurrence requests, 69,072 judgments, and 2,955 explicitly targeted words.
- Development: 1,045 requests. The 898 added GUM occurrences contain 3,592
  judgments covering 445 target words across 30 documents.
- Calibration and test: unchanged from the parent. Previously inspected sets
  remain historical evidence, not an independent final test.

The GUM contribution uses only the published development split. GUM source
commit is `b58e74bc22d17220c9198864c253a50a897bf27f` (UD r2.18). Original
README, license, source files, hashes and download URLs are retained under
`data/raw/en-gum-morphology-001`. The repository license is CC BY-NC-SA 4.0;
underlying text and annotator attribution requirements remain applicable.
This experiment makes no production-use licensing claim.

## Labels and leakage checks

Labels are derived from explicit single-valued Number, Tense or VerbForm fields.
Each request includes a true affirmative, false affirmative, and their logical
negations. A deterministic claim hash determines question order. Sentences and
target positions enter model input; source features and adjudication rationales
do not. These are source-derived research labels, not independent expert gold.

`runs/en-ud-morphology-001-source-audit.json` reconstructs all EWT target tokens
and labels from the source. The question-ID majority baseline is 50.30% and the
negation-only baseline is 50%; this does not exclude every possible shortcut.
861 added requests reuse parent training sentences, including 153 at the same
target position with different questions. No such match crosses into parent
heldout data; see `runs/en-ud-morphology-001-parent-overlap-audit.json`.

GUM exact sentence overlap checks identified 101 matches with EWT and 60 groups
shared across GUM source splits. All implicated copies, 512 sentences, are
recorded in `runs/en-gum-morphology-001-sentence-exclusions.json`. Candidate
selection then excludes targets connected to training, calibration or test
families. Raw sources and rejected candidates are preserved.

`runs/en-joint-005-expanded-family-audit.json` reports zero current split
conflicts and zero historical non-development conflicts for the GUM additions.
Shared documents connect each new contribution into one large component:
word counts must not be presented as independent family counts.

Actual pinned Kev tokenizer admission passes all partitions with no truncation
or overflow (`runs/en-joint-005-tokenization.json`). Maximum packed length is
2,577 tokens in training and 1,596 in development. This is tokenizer execution,
not proof of model performance.

## Baseline execution

Registration is frozen at
`runs/lev-joint005-morphology-development-cycle/registration.json` before scoring.
Modal run `lev-joint005-v04-development-01` uses the completed v04 checkpoint
`lev-joint001-v04-train-02`, config `configs/lev-4b-v04.toml`, and A100-80GB.
There are no automatic retries or threshold searches. The run evaluates the
entire development partition; reports must separately identify GUM and original
parent questions. Missing predictions remain in the expected denominator.

No score or improvement is claimed until remote execution, downloaded artifacts,
checkpoint identity, and prediction alignment are verified. A subsequent
training experiment and its comparison gates have not yet been registered.

A state-free diagnostic baseline fits the majority label for each exact claim
text on the EWT training contribution and applies it unchanged to GUM development.
It scores 2,804/3,592 (78.06%) without reading sentence context. Evidence is in
`runs/en-gum-morphology-development-001-claim-only-baseline.json`. This diagnostic
was computed before inspecting model predictions and does not alter the frozen
sample or model protocol. Real model scores must be presented alongside this
baseline; exceeding the 50% negation-only baseline is insufficient evidence of
contextual understanding.

### Completed v04 baseline

The registered remote run succeeded. Download and report verification completed
under `runs/lev-joint005-morphology-development-cycle`; the full report is
`v04-baseline-report.json`. All 1,045 requests / 3,739 judgments completed, with
zero rejected or truncated records. The execution used float32 on an A100-SXM4
80GB, temperature 1.0, and the registered checkpoint content hashes.

| Subset | Correct / requested | Accuracy |
| --- | ---: | ---: |
| GUM contextual morphology | 2,959 / 3,592 | 82.38% |
| Number | 2,507 / 2,940 | 85.27% |
| Tense | 140 / 196 | 71.43% |
| VerbForm | 312 / 456 | 68.42% |
| Original parent development | 130 / 147 | 88.44% |

GUM has 383 false acceptances out of 1,796 negative judgments and 250 false
rejections out of 1,796 positive judgments. Word macro accuracy is 82.40%; 207
of 445 target words have every scheduled question correct. This is partial
coverage, not proof of full mastery of those words. GUM accuracy exceeds the
claim-only baseline by 155 judgments, or 4.32 percentage points; no independence
or statistical-significance claim is made for these correlated examples.
The original parent score reproduces the earlier v04 result.

## Preserved rejected alternatives

`en-joint-003` contains lexical inflection relations with constant question-ID
and negation polarity labels. Its shortcut audit blocks training; adding a
different task does not repair this defect. Its source relations remain useful
candidate evidence but are not included in `en-joint-005`.

Single-agent occurrence interpretations from morphology review batches 001 and
003 were incorrectly marked source-derived. Batch 004 preserves corrected draft
copies; none of these interpretations entered training. See its manifest for
the preserved predecessor hashes.
