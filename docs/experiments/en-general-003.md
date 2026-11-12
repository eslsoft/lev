# English generalization iteration 003 — preregistered protocol

Created 2026-10-21 before any model scoring on this version. Objective: improve a non-generative, open-question Boolean language reviewer; no fixed category routing or generated answers.

## Data and isolation

`build_generalization_corpus.py` extends the previously verified source snapshots. It recreates the original lexical/derivational union graph, extends selected relation edges and UD document links, excludes every resulting component touching any prior nontraining family, and pins retained original training families to train. Original frozen data is not changed. Original train examples are replay data only. Incidental sentence vocabulary may recur; this is not universal vocabulary disjointness.

New language question wording includes affirmative and negative wrappers, chosen from IDs independently of labels. Wrapper vocabularies differ across splits. CMU tasks also ask syllabic-nucleus count. New BLiMP paradigms come from the complete archive at the previously pinned commit. All minimal pairs/constructions stay together: island effects development, control/raising calibration, ellipsis and s-selection test; all prior held-out constructions are excluded. Remaining constructions train. Category names are only report metadata.

BIG-bench implicatures are **test only**, never training or calibration. The source authors explicitly identify this as benchmark data; this experiment respects that intended use. Original yes/no scores become one Boolean judgment per dialogue, with deterministic polarity transformation. This tests a new task/source, not just template paraphrases. Raw source bytes, download events and hashes are preserved. These published labels are not independent Lev adjudication. Underlying implicature source is identified as CC BY 4.0 by its README; BLiMP is CC-BY. Preserve attribution. No publication authorized.

## Execution and selection

1. Freeze corpus; run local export, source derivation and isolation checks.
2. Score original pinned Kev and incumbent Lev v0.1 on **development only**.
3. Inspect development errors. Any substantive data revision gets a fresh version and audit trail, with final test still unread.
4. Train a fresh LoRA candidate from pinned original Kev using broadened data, with 2 epochs, learning rate 1e-5, seed 42, other engine parameters unchanged. This is a controlled new training iteration, not optimizer-state resume. Use a unique Modal run and preserve failures.
5. Compare development macro accuracy across categories, grammaticality accuracy and total accuracy to both original and incumbent. Select candidate only if macro accuracy improves over incumbent and grammar accuracy does not decline; missing outputs disqualify a model. Selection is recorded **before test**. If not improved, use development errors for one further bounded candidate/data revision; no test-driven iteration.
6. Fit selected model temperature on calibration; evaluate selected model, original Kev and incumbent on the frozen final test. Threshold 0.5. Keep all outcomes, including regressions. Once test is scored this iteration is closed; further model selection needs a fresh blind set.
7. Deliver retained checkpoint, calibration, raw artifacts, per-domain and family-level uncertainty analysis, limitations and training resource evidence. Success means an honestly measured best available model, not claiming arbitrary-question mastery from a finite corpus. Production approval remains separate.

Modal uses developer workspace/profile `saltbo`, as previously discovered Realmroot has no matching Modal integration. Scope: training, evaluation and artifact readback; no deployment/publication. Two new candidate training runs at most in this cycle, each with a 7200-second function timeout. These are resource bounds, not a promised monetary price cap.

## Pre-scoring data audit

Version 003 was frozen but never scored or trained: expanded document/family unioning left only two syntax decisions in development. Version **004** samples at most one new target per existing family/document group before extending unions; it retains 42 development syntax decisions. All 003 artifacts remain preserved. Version 004: train 2,798 requests / 4,424 decisions; development 424 / 590; calibration 323 / 532; test 764 / 932. No lexical-analysis family survives as a new held-out group from the small original Wiktionary selection; lexical_analysis is replay-only, so this iteration does not independently establish lexical-analysis transfer. This limitation is reported rather than fabricating coverage.

The two development-only Modal apps were submitted before training: `ap-on9rUJ4fnECNKfjlpKWxHx` (original) and `ap-c5POywIELgACOnSoLPRAYM` (incumbent).

## Development before training

Original Kev: 364/590 (61.69%), macro 66.83%; Lev v0.1: 427/590 (72.37%), macro 75.87%. Grammar: 171/320 vs 198/320. Pronunciation: 83/124 vs 100/124. Semantic relation: 81/104 vs 103/104. Syntax: 29/42 vs 26/42. Full source-aligned comparison: `runs/en-general004-pretraining-comparison.json`. Both runs succeeded with no missing output.

Development inspection identifies difficult held-out island constructions, syllable-count questions and polarity handling. No model scores became labels, no hard cases were removed. Candidate v0.2 starts from the pinned original Kev checkpoint with config `configs/lev-4b-v02.toml`, 2 epochs and learning rate 1e-5; run `lev-general004-v02-train-01`. Model selection still uses development only.

Further **development-only** diagnostics, before candidate scoring: incumbent affirmative wrappers 214/291 (73.5%), negative wrappers 213/299 (71.2%). Within syntax, affirmative 15/16 (93.8%) versus negative 11/26 (42.3%); syllable count affirmative 22/30 versus negative 20/32. Counts are small and correlated; these are diagnostic observations, not significance claims. Raw breakdown is `runs/en-general004-development-diagnostics-before.json`. The already-fixed v0.2 training includes independent polarity variation; no samples or labels were changed after these diagnostics.

## Completed results

Training completed: 700 optimizer steps, 5,596/5,596 requests, no rejected or truncated requests. Candidate development: 483/590 (81.86%), macro 89.99%; selection was frozen before test. Original 004 test: Kev 621/932, v0.1 619/932, v0.2 790/932. A source-based family correction produced 005 with unchanged train/development/calibration bytes: Kev 621/932, v0.1 619/932, v0.2 **791/932 (84.87%)**. The 930 shared decisions have identical probabilities across evaluations for every model; this is not independent replication. Calibration T=2.193649959389252.

Supplemental old 002 development direct-question retention: v0.1 193/204 versus v0.2 191/204, a two-question grammar regression. This diagnostic was not used for retuning or selection. All outcomes are retained. See [v0.2 model card](../model-card-v02.md), [correction protocol](en-general-005.md), and `runs/en-general005-delivery-verification.json`. This cycle is closed; further model selection requires a fresh blind test.
