# v04 full vocabulary evaluation

All 16 immutable scoring runs completed: 35,000 words and 368,056 judgments, with no missing judgments. Research-only source-derived inventory benchmark; not independent expert certification or complete word mastery.

Checkpoint: `runs/modal-lev-joint001-v04-train-02/result/checkpoint`. Frozen temperature: 1.9543199368684914; threshold: 0.5.

Judgment accuracy: 74.7826%; word macro accuracy: 75.3325%. All scheduled items correct: 2,845/35,000 (8.1286%).
False acceptance: 77,326/213,827 negative labels (36.1629%); false rejection: 15,488/154,229 positive labels (10.0422%).

| Inventory task | Correct | Expected | Accuracy |
|---|---:|---:|---:|
| pos_inventory | 94883 | 138868 | 68.3260% |
| pronunciation_inventory | 29763 | 57667 | 51.6118% |
| sense_inventory | 142427 | 160385 | 88.8032% |
| wiktionary_pos_inventory | 3498 | 4776 | 73.2412% |
| wiktionary_sense_inventory | 4671 | 6360 | 73.4434% |

Candidate-specific supervised target-family exposure (corroborated family audit 002):

| Exposure | Words | Judgment accuracy | All scheduled items correct |
|---|---:|---:|---:|
| not_detected | 32476 | 74.9023% | 2757 |
| seen | 2524 | 73.5937% | 88 |

Not detected does not mean absent from pretraining. The frozen historical exposure groups remain unchanged; the candidate-specific overlay is separate. Group differences are descriptive, not a causal training comparison.

The first merge failed because Modal allocated both A100 80GB PCIe and A100 SXM4 80GB. Software versions, checkpoint contents, calibration, selection, precision and inference settings match. The corrected merge permits only these two A100 form factors, retains every actual environment per shard, and does not assert bitwise equivalence. No scoring was rerun. Original failure: `runs/lev-joint001-v04-vocabulary/merge.log`; recovery: `merge-recovery-01.json` in the same directory.

Artifacts: `runs/lev-joint001-v04-vocabulary/word-report.json` (all per-word outcomes), `exposure-report-002.json` (same directory), and all 16 `runs/modal-lev-vocab35000-v04-shard-XX-01` directories.

This result does not satisfy the continuing model objective. Context supplement evaluation is running separately. Preserve this frozen test; do not tune thresholds, select checkpoints or generate training labels from its outcomes. Further training decisions require development evidence and training-family data; future claims after test-informed changes require a fresh independent evaluation.

Modal execution and status access used the authorized saltbo developer workspace because no matching Realmroot integration was available. No deployment or publication occurred.
