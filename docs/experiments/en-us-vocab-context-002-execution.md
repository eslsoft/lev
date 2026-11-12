# Context supplement 002 execution registration

The frozen dataset protocol and admission remain unchanged. Run
`uv run --extra modal python scripts/advance_context_supplement.py --execute`
to wait for the v04 base vocabulary evaluation to finish completely, then score
the two admitted context-002 shards sequentially. Without `--execute`, the command
only verifies admission evidence and shard hashes.

The controller requires the completed development-selected, calibrated v04 cycle
and uses its exact checkpoint, selection and calibration with threshold 0.5.
It does not use base vocabulary scores to tune or select a model. Waiting until
the base evaluation finishes prevents overlapping its two GPU slots. An incomplete
or failed predecessor prevents submission. Failed context submissions retain their
logs and immutable markers; there is no automatic retry or best-run selection.

Run IDs are `lev-context002-v04-shard-00-01` and
`lev-context002-v04-shard-01-01`. Local controller state and the merged word report
live in `runs/lev-joint001-v04-context-supplement`. Each remote result is downloaded
to its separate `runs/modal-<run-id>` directory. The existing merge implementation
checks consistent model/calibration identities and retains missing judgments.

Admission has been exercised against the actual frozen two-shard data; three
controller tests cover unsuccessful prerequisites, a dead predecessor, and
failure preservation without repeat submission. These are orchestration checks,
not evidence of model execution. Starting the waiter does not mean scoring began.
This supplement covers 1,526 words and 3,619 judgments; it does not establish
35,000-word context coverage or complete word mastery.
