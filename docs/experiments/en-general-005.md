# English generalization 005: source-based correction with frozen models

This correction was frozen on 2026-10-21 after the first 004 evaluation, before any model evaluation on 005. It is **not a wholly new independent replication**: 930 of 932 test decisions are unchanged. No model, training recipe, threshold, or model-selection rule is changed. All 004 artifacts and results remain preserved.

## Finding and cause

An independent family audit closes **every** explicit WordNet form edge, rather than assigning a form only to its first encountered lemma. It found one component linking `good` and `well` through their shared irregular forms `better` and `best`. One test request, `fb633927ffd48d52eb06`, therefore shared a known form-family component with training data. This affects two test judgments. Development and calibration are unaffected.

The original `Corpus.wordnet` used `aliases.setdefault(form, lemma)` without also unioning every form owner. A later owner could remain disconnected. The builder now joins all published form-owner edges after alias collection, and a regression test explicitly verifies that good/well/better/best form one group. Existing frozen corpora are never rewritten; the pre-correction builder is preserved in `runs/generalization-cycle-source/scripts` for reproduction of 004.

## Correction without using model outcomes

`scripts/repair_generalization_holdout.py` reads source records and family relations, **not predictions or model scores**. It retains the excluded request in `data/research/en-general-005/correction.json`, and replaces it with one UD request chosen by the minimum SHA-256 of `lev-005-independent-replacement:sent_id:token_index` among 117 eligible candidates. Eligibility uses the original length, POS and dependency-role rules, and excludes every lexical family or document already used in the prior/current corpus. The replacement is `answers-20111108084119AAhsBk8_ans-0003:token=11`, target `repeatedly`, source relation `advmod`.

The final test still contains 764 partial requests / 932 Boolean decisions. All train, development and calibration **export bytes are identical** to 004. Therefore the trained weights and the valid development-based selection remain applicable; retraining would not change the training data and is not performed. The replacement is a structural data-quality correction, not an attempt to obtain a favorable score. Report both the initial result and the corrected result; use 005 for the corrected family-disjoint claim. Do not count these two evaluations as independent statistical replications.

## Frozen execution

- Selected model: `lev-general004-v02-train-01/result/checkpoint`, selected on 004 development at 2026-10-21T19:58:12.747968+00:00, before either final evaluation.
- Comparators: pinned original Kev and `lev-general-20261021-01/result/checkpoint` (Lev v0.1).
- Exact checkpoint fingerprints are preserved in the development comparison and selection evidence. No weights change after selection.
- Refit the same fixed 121-point 0.25–4 temperature grid on the byte-identical 532-decision calibration partition so the new calibration file correctly binds the 005 manifest; never edit/rebind the old calibration artifact.
- Threshold remains 0.5. Re-evaluate all three fixed models on the same corrected test. All outcomes are retained; no best-of-retries selection.
- New run: `lev-general005-v02-final-01`, with the existing `finalize` operation. No new training or publication/deployment.

Independent family audit for 005: `data/research/en-general-005/family-audit.json`, passed with no violations. This proves the audited published form/derivation/document boundaries, not universal vocabulary disjointness or absence of public-source pretraining contamination.

## Completed results

Training completed: 700 optimizer steps, 5,596/5,596 requests, no rejected or truncated requests. Candidate development: 483/590 (81.86%), macro 89.99%; selection was frozen before test. Original 004 test: Kev 621/932, v0.1 619/932, v0.2 790/932. A source-based family correction produced 005 with unchanged train/development/calibration bytes: Kev 621/932, v0.1 619/932, v0.2 **791/932 (84.87%)**. The 930 shared decisions have identical probabilities across evaluations for every model; this is not independent replication. Calibration T=2.193649959389252.

Supplemental old 002 development direct-question retention: v0.1 193/204 versus v0.2 191/204, a two-question grammar regression. This diagnostic was not used for retuning or selection. All outcomes are retained. See [v0.2 model card](../model-card-v02.md), [correction protocol](en-general-005.md), and `runs/en-general005-delivery-verification.json`. This cycle is closed; further model selection requires a fresh blind test.
