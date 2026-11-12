#!/bin/sh
# Unattended pipeline: build a capped coverage pack, publish it as a Kaggle
# dataset, train the ModernBERT encoder on it, and pull the run back locally.
# Every stage uses a fresh output directory; nothing is reused or overwritten.
set -eu
cd "$(dirname "$0")/.."

V="${1:?usage: run_coverage_pipeline.sh <version> [relation_records]}"
REL="${2:-2000}"
TRAIN_RUN="${3:-enc-train-004}"
DEV_RUN="${4:-enc-dev-004}"
CFG="configs/encoder-modernbert-base-$V.toml"

echo "== build $V (relation_records=$REL) =="
PYTHONPATH=src:scripts .venv/bin/python scripts/build_coverage_corpus.py \
  --version "$V" --relation-records "$REL" > "/tmp/$V-summary.json" 2>&1 || {
    echo "BUILD_FAILED"; tail -20 "/tmp/$V-summary.json"; exit 1; }

echo "== config $CFG =="
sed "s/^model_name = .*/model_name = \"lev-enc-mbert-base-$V-v0.1\"/" \
  configs/encoder-modernbert-base-joint006.toml > "$CFG"

echo "== stage dataset =="
STAGE="/tmp/kg-$V"
rm -rf "$STAGE"; mkdir -p "$STAGE"
for f in records train development calibration test manifest rubric; do
  cp "artifacts/$V/$f.jsonl" "$STAGE/" 2>/dev/null || cp "artifacts/$V/$f.json" "$STAGE/"
done
cat > "$STAGE/dataset-metadata.json" <<EOF
{"title": "Lev $V coverage pack", "id": "saltbo/lev-$V", "licenses": [{"name": "CC0-1.0"}]}
EOF
kaggle datasets create -p "$STAGE" --dir-mode zip

echo "== wait for dataset =="
for i in $(seq 1 60); do
  s=$(kaggle datasets status "saltbo/lev-$V" 2>&1 | tail -1)
  case "$s" in *ready*) echo "$s"; break;; esac
  sleep 30
done

echo "== notebook =="
PYTHONPATH=src:scripts .venv/bin/python scripts/make_kaggle_notebook.py \
  --out "notebooks/kaggle-encoder-$V.ipynb" --dataset "/kaggle/input/lev-$V" \
  --config "$CFG" --train-run "$TRAIN_RUN" --dev-run "$DEV_RUN"

echo "== push kernel =="
KD="/tmp/kern-$V"; rm -rf "$KD"; mkdir -p "$KD"
cp "notebooks/kaggle-encoder-$V.ipynb" "$KD/"
cat > "$KD/kernel-metadata.json" <<EOF
{"id": "saltbo/lev-encoder-${V}-train", "title": "lev encoder $V train",
 "code_file": "kaggle-encoder-$V.ipynb", "language": "python",
 "kernel_type": "notebook", "is_private": true, "enable_gpu": true,
 "enable_internet": true, "dataset_sources": ["saltbo/lev-$V"],
 "competition_sources": [], "kernel_sources": []}
EOF
kaggle kernels push -p "$KD"

echo "== poll =="
for i in $(seq 1 96); do
  s=$(kaggle kernels status "saltbo/lev-encoder-$V-train" 2>&1 | tail -1)
  case "$s" in
    *RUNNING*|*QUEUED*) sleep 300;;
    *) echo "FINAL: $s"
       rm -rf "runs/kaggle-$TRAIN_RUN"
       kaggle kernels output "saltbo/lev-encoder-$V-train" -p "runs/kaggle-$TRAIN_RUN"
       PYTHONPATH=src .venv/bin/python - "$V" "$TRAIN_RUN" "$DEV_RUN" <<'PY'
import json, sys, statistics as st
v, tr, dv = sys.argv[1], sys.argv[2], sys.argv[3]
rows = json.load(open(f"runs/kaggle-{tr}/{dv}/native/rows.json"))
ok = sum(1 for r in rows if (r["p"][1] > 0.5) == (r["label"] == 1))
p = [r["p"][1] for r in rows]
print(json.dumps(dict(
    version=v,
    accuracy=f"{ok}/{len(rows)} = {round(100*ok/len(rows),2)}%",
    p_median=round(st.median(p), 4),
    near_half=sum(1 for x in p if abs(x - 0.5) < 0.05),
    dataset_sha256=json.load(open(f"runs/kaggle-{tr}/{dv}/lev-report.json"))["dataset_sha256"],
), indent=1))
PY
       exit 0;;
  esac
done
echo "TIMEOUT after 8h"
