# English general-decision research pilot 002

Status: preregistered before model scoring, 2026-10-21. This is a non-generative Kev/JEV-style Boolean decision experiment, not a released model or evidence of universal language-review competence.

## Fixed protocol

- Base and initial adapter: exactly the revisions in `configs/lev-4b.toml`.
- Dataset: `artifacts/en-general-002`, source records SHA-256 `8c5c9f8c3605e2b9febec0117027c49fff015b3f19b9db1ee88a1c8f3a577a71`.
- 1,006 requests / 1,724 decisions; 862 positive and 862 negative labels. Every contribution is partial.
- Split requests: train 581, development 126, calibration 206, test 93.
- Decision counts: train 970, development 204, calibration 388, test 162.
- Probe: 40 training requests, original split assignments retained, 1 epoch. This validates runtime only; its checkpoint is not a selected candidate.
- Candidate: original 2-epoch LoRA configuration, rank 16, learning rate 2e-5, batch 1, accumulation 8, checkpointing on, CUDA bf16 autocast with fp32 backbone. No parameter sweep.
- Fixed decision threshold: 0.5. Baseline uses the pinned Kev checkpoint's existing temperature. Candidate is fit once on calibration using the existing Kev temperature procedure, without changing the checkpoint.
- Run order: original Kev development, candidate training, candidate development, candidate calibration logits, temperature fitting, candidate test, original Kev test. No test-driven changes or candidate selection.
- Output: original native reports, missing-decision-aware Lev reports, raw logs, training metrics and checkpoint. Interrupted/failed attempts remain visible under their own run IDs.

## Sources and label meaning

- Open English WordNet 2025 XML release, with CC-BY attribution material. 180 directed hypernym pairs, definitions supplied to the model; positive and reversed-direction questions. This tests sense-level taxonomy, not dictionary memorization alone.
- UD English EWT commit `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b`, CC BY-SA 4.0. 180 sentence targets; explicit basic dependency relation versus an exclusive alternative. Interpret against the declared UD convention, not all possible linguistic theories.
- CMUdict commit `74790861f652b15e4ac49015a90074ad62a27690`, source license retained. 140 supplied ARPABET pronunciations; primary stress nucleus identification. This does not establish correctness of arbitrary novel IPA.
- BLiMP commit `3e56b06fcabca9b30822fc66435fca6b1aa40bb1`, CC-BY per README. 144 original minimal pairs / 288 single-sentence decisions. Labels come from linguist-written generators and aggregate human validation, not individual Lev adjudication.
- Kaikki extraction of English Wiktionary: 70 explicitly listed headwords, 218 lexical entries; first-sense word-class decisions. Downloaded JSONL bytes and per-file hashes are preserved. Wiktionary attribution and share-alike obligations must be retained for any later redistribution.
- WiC deliberately not included: its published license is noncommercial.

Raw snapshots: `data/raw/en-research-20261021-a/`. URLs and download hashes: `download-events.jsonl` and `wiktionary-downloads.jsonl`. Complete hashes, selection audit and generated records: `data/research/en-general-002/`. Nothing has been published externally as a dataset.

Labels are `source_labeled` or `source_derived`, never relabeled as independent human adjudication. This uses an explicit `research_only` export mode and research rubric. Production export still requires approved semantics and adjudicated evidence-backed labels. No model scores were used as labels.

## Leakage and limitations

WordNet derivations, explicit word forms, selected relation endpoints, UD document IDs and matching input states are grouped before splitting. BLiMP is grouped by construction: anaphor agreement is development, binding is calibration, existential quantifiers are test; remaining selected constructions are training. These assignments precede any model scores. Shared incidental vocabulary in different sentences is not forbidden, so this is not a universal vocabulary-disjoint benchmark.

Sampling and exclusions are deterministic and fully logged (268,440 excluded source candidates, 1,006 selected requests). No rejected candidate is silently dropped during export. Dictionary headwords are a manually specified common-word pilot list, not a representative random sample of English. Large connected lexical families make split proportions uneven. Public sources may already have appeared in base-model pretraining, so held-out Lev records do not prove absence of pretraining contamination.

Version 001 is retained, not used for training: a local audit found label-correlated question phrasing and no grammaticality decisions in development/test. Version 002 uses the same question wording for positive/negative alternatives and predeclared held-out constructions. No model outputs motivated that revision.

The model accepts any natural-language Boolean question; these five reporting categories are only current data coverage. Fixed templates and a small corpus cannot establish generalization to arbitrary instructions, register, pragmatics, etymology, dialects or complete contributions. There is no independent expert review of this pilot, and benchmark point estimates are not release guarantees.

## Runtime acceptance before full experiment

`lev-probe-20261021-02` failed before model loading because Modal directory mounts excluded `.git`; `03` failed during image build because archive ownership triggered Git's ownership check. Both original logs are preserved under `runs/modal-failures/`. The archive now carries portable ownership and is extracted with `--no-same-owner`; the image verifies HEAD before allocating a training function. No scoring or training occurred in those failures.

`lev-probe-20261021-04` succeeded on an A100-SXM4-80GB: 40/40 requests, 5 optimizer steps, 10,196 forward tokens, 35.999 seconds training wall time, 18,376,223,232 bytes peak device allocation, 32,149,401,600 bytes peak host RSS, zero rejected/truncated records. The real adapter and pointer head were saved. All five steps had finite clipped gradients. Kev actually used Python 3.13.8 from its version file; the full-run image now pins that exact interpreter before dependency installation.

Full run uses 4 CPU, 32 GiB requested host RAM with 64 GiB limit, and the same A100-80GB/dtype/model configuration. The probe was not selected by benchmark score.

## Completed result

Modal app `ap-OUF79UVBkkePd0WmzOybQ1`, run `lev-general-20261021-01`, completed all seven stages. Train: 146 optimizer steps; 1,162/1,162 requested records processed; zero truncation/rejection; 981.50 seconds training wall time. Development: original Kev 167/204 versus Lev 193/204. Final test: original Kev 132/162 versus Lev 146/162, all predictions present. Test false rejections: 19 to 9; false acceptances: 11 to 7. Brier: 0.15114 to 0.05570.

The held-out grammaticality subset regressed from 17/24 to 16/24; other results must not hide this. Temperature is 4.0, at the upper edge of the predeclared fit grid. No test-driven tuning followed. These are source-label research metrics, not independent human adjudication or proof of universal linguistic competence.

All 68 files (160,076,115 bytes) were downloaded to `runs/modal-lev-general-20261021-01`. Checkpoint hashes match `test/job.json`, all stage return codes are zero, and reports were independently recomputed from the downloaded frozen data and native rows. Verification and byte-exact executed Lev source snapshots are in `runs/lev-general-20261021-01-verification`. A post-run local fix allows calibration to follow an unchanged checkpoint to another path; changed checkpoint content remains rejected.

The Modal client timed out fetching final app log output after receiving the successful result; server-side statuses, artifacts and byte hashes were verified separately. Modal CLI directory download also failed without producing files; its log is retained and the SDK streaming download succeeded with per-file byte counts. Neither issue changed or reran model scoring. Own ephemeral apps are stopped with zero running tasks.

Identity: current Realmroot discovery had no Modal service; the authorized work used the existing developer Modal profile/workspace `saltbo` to list resources, submit jobs, read own container logs and download artifacts. No credentials were copied into the image, and the pre-existing `kev` deployed app was not modified.
