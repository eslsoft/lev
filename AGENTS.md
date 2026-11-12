# Lev project instructions

Lev owns language-review data and experiments; Kev owns the ML engine. Keep them separate.

- Python 3.12/3.13, uv, standard-library CLI and unittest. Commands run from this repository.
- `uv sync --locked`; `uv run python -m unittest discover -s tests -v`.
- Set `LEV_KEV_PATH` to an existing checkout; never clone or modify Kev automatically. `configs/lev-4b.toml` pins the compatible commit and model revisions.
- Read `docs/data.md` and `docs/workflow.md` before changing dataset or experiment semantics.
- Never turn model scores into gold labels automatically, silently omit rejected samples, mix word families across splits, or report partial contributions as complete words.
- Example data is not training data. Production export requires an approved rubric and adjudicated evidence-backed labels.
- Preserve original artifacts and failures. New experiments use new output directories; no retry-and-pick scoring.
- Local scaffold changes do not authorize paid training, remote deployment or model publication. Explicit requests to train/deploy own their scope.
- Test numerical/semantic boundaries and actual export contracts. A mock or source-level contract test is not evidence of model execution.
