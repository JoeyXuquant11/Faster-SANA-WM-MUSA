#!/usr/bin/env bash
# Rebuild pyuser environment, public SANA-WM Stage-1 DiT + Gemma-2-2B-IT weights,
# deterministic 64-sample Sekai-Game fixture, and targets/reference baseline tree
# so bench/bench.py check --final can be reproduced on any 4x MTT S5000 host.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASSETS="${ASSETS:-$REPO_ROOT/assets}"
MODEL_ROOT="$ASSETS/models/sana_wm"
USE_MODELSCOPE="${USE_MODELSCOPE:-0}"

mkdir -p "$MODEL_ROOT/SANA-WM_bidirectional" "$MODEL_ROOT/gemma-2-2b-it" "$ASSETS/fixture"

echo "=== [1/4] Installing SANA-WM Python dependencies into $ASSETS/pyuser ==="
if [[ ! -d "$ASSETS/pyuser/lib/python3.10/site-packages/diffusers" ]]; then
  bash "$REPO_ROOT/scripts/install_env.sh"
else
  echo "  $ASSETS/pyuser already present, skipping pip install."
fi
export PYTHONUSERBASE="$ASSETS/pyuser"
export PYTHONPATH="$ASSETS/pyuser/lib/python3.10/site-packages:${PYTHONPATH:-}"

echo "=== [2/4] Downloading public SANA-WM DiT & Gemma-2-2B-IT weights ==="
if [[ ! -f "$MODEL_ROOT/SANA-WM_bidirectional/dit/sana_wm_1600m_720p.safetensors" ]]; then
  if [[ "$USE_MODELSCOPE" == "1" ]]; then
    python3 -c "from modelscope import snapshot_download; snapshot_download('Efficient-Large-Model/SANA-WM_bidirectional', local_dir='$MODEL_ROOT/SANA-WM_bidirectional', allow_patterns=['dit/sana_wm_1600m_720p.safetensors', 'config.yaml', 'configuration.json', 'LICENSE', 'README.md'])"
  else
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Efficient-Large-Model/SANA-WM_bidirectional', local_dir='$MODEL_ROOT/SANA-WM_bidirectional', allow_patterns=['dit/sana_wm_1600m_720p.safetensors', 'config.yaml', 'configuration.json', 'LICENSE', 'README.md'])"
  fi
fi

if [[ ! -f "$MODEL_ROOT/gemma-2-2b-it/model-00001-of-00002.safetensors" ]]; then
  if [[ "$USE_MODELSCOPE" == "1" ]]; then
    python3 -c "from modelscope import snapshot_download; snapshot_download('Efficient-Large-Model/gemma-2-2b-it', local_dir='$MODEL_ROOT/gemma-2-2b-it', ignore_patterns=['gemma-2-2b-it.safetensors'])"
  else
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Efficient-Large-Model/gemma-2-2b-it', local_dir='$MODEL_ROOT/gemma-2-2b-it', ignore_patterns=['gemma-2-2b-it.safetensors'])"
  fi
fi

echo "=== [3/4] Building deterministic 64-sample Sekai-Game fixture & recipe_stage1.yaml ==="
cp "$REPO_ROOT/configs/recipe_stage1.yaml" "$ASSETS/recipe_stage1.yaml"
if [[ ! -f "$ASSETS/fixture/raw/sekai_game_train_00000000.zip" ]]; then
  python3 "$REPO_ROOT/scripts/build_fixture.py" --out "$ASSETS/fixture"
fi

echo "=== [4/4] Materializing targets/reference from the baseline branch ==="
if [[ ! -d "$REPO_ROOT/targets/reference" ]]; then
  mkdir -p "$REPO_ROOT/targets"
  if git -C "$REPO_ROOT" rev-parse --verify origin/baseline >/dev/null 2>&1; then
    git -C "$REPO_ROOT" archive origin/baseline Sana | tar -x -C "$REPO_ROOT/targets"
    mv "$REPO_ROOT/targets/Sana" "$REPO_ROOT/targets/reference"
  elif git -C "$REPO_ROOT" rev-parse --verify baseline >/dev/null 2>&1; then
    git -C "$REPO_ROOT" archive baseline Sana | tar -x -C "$REPO_ROOT/targets"
    mv "$REPO_ROOT/targets/Sana" "$REPO_ROOT/targets/reference"
  else
    cp -a "$REPO_ROOT/Sana" "$REPO_ROOT/targets/reference"
    git -C "$REPO_ROOT/targets/reference" apply -R "$REPO_ROOT/patches/sana_wm_r1_to_r11.patch"
  fi
fi

echo "Done. Next run: ASSETS=$ASSETS bash scripts/freeze_judge_and_verify.sh"
