#!/usr/bin/env bash
# Rebuild frozen reference timing, Gate 2 reference records/thresholds, and Gate 3
# 6-seed trajectory delta on a 4x MTT S5000 host, then run bench.py check --final.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASSETS="${ASSETS:-$REPO_ROOT/assets}"
export PYTHONUSERBASE="$ASSETS/pyuser"
export PYTHONPATH="$REPO_ROOT:$ASSETS/pyuser/lib/python3.10/site-packages:${PYTHONPATH:-}"
export SANA_LOCAL_MODEL_ROOT="$ASSETS/models/sana_wm"
export SANA_GEMMA2_PATH="$ASSETS/models/sana_wm/gemma-2-2b-it"
export MKL_NUM_THREADS=1
export TORCH_COMPILE_DISABLE=1
export GDN_DISABLE_COMPILE=1
export DISABLE_XFORMERS=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True

cd "$REPO_ROOT"
mkdir -p runs bench/records bench/traj

echo "=== [1/7] Freezing dataset loader digest ==="
python3 bench/bench.py loader --tree targets/reference > task/loader_digest.json

echo "=== [2/7] Freezing reference timing (2 cold-cache runs on 4x MTT S5000) ==="
python3 bench/bench.py time --tree targets/reference --freeze --out runs/repro_ref_timing

echo "=== [3/7] Recording Gate 2 deterministic reference records (2 runs) & calibrating ==="
python3 bench/bench.py record --tree targets/reference --out bench/records/ref
python3 bench/bench.py record --tree targets/reference --out runs/repro_ref_rec_b
python3 -m numerics calibrate --ref bench/records/ref --variants runs/repro_ref_rec_b --out bench/thresholds.json

echo "=== [4/7] Recording Gate 3 6-seed reference trajectories (seeds 3407..3412) ==="
python3 bench/bench.py record --tree targets/reference --traj-seeds 3407,3408,3409,3410,3411,3412 --out bench/traj/ref

echo "=== [5/7] Freezing Gate 3 per-endpoint delta ==="
python3 -m numerics traj-freeze --ref bench/traj/ref --spec bench/traj/spec.json --k 2.0 --out bench/traj/frozen.json

echo "=== [6/7] Freezing input fingerprints into task/formulation.json ==="
python3 bench/bench.py freeze-inputs \
  "$ASSETS/recipe_stage1.yaml" \
  "$ASSETS/fixture" \
  "$ASSETS/models/sana_wm/SANA-WM_bidirectional" \
  "$ASSETS/models/sana_wm/gemma-2-2b-it"

echo "=== [7/7] Running paired A-B-A certification (bench.py check --final) ==="
touch task/final-evaluation
python3 bench/bench.py check --final --tree Sana --config configs/opt.json --out runs/repro_final_check
cat runs/repro_final_check/check.json
