#!/usr/bin/env bash
# Install Python dependencies into $ASSETS/pyuser without overwriting system
# MUSA packages (torch, torch_musa, mate, tilelang).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASSETS="${ASSETS:-$REPO_ROOT/assets}"
export PYTHONUSERBASE="$ASSETS/pyuser"
export PATH="$PYTHONUSERBASE/bin:$PATH"
mkdir -p "$PYTHONUSERBASE"

pip freeze | grep -iE "^(torch|torch_musa|torchvision|torchaudio|numpy|pillow|mate|flash_attn|tilelang)==" > /tmp/musa_constraints.txt || true

pip install --user --no-cache-dir "setuptools<80" wheel packaging importlib-metadata
pip install --user --no-cache-dir -c /tmp/musa_constraints.txt \
  "huggingface-hub==0.36.0" \
  "transformers==4.57.3" \
  "diffusers>=0.37.0" \
  "accelerate>=1.3" \
  "peft==0.18.0" \
  "safetensors" \
  "einops" \
  "omegaconf" \
  "pyrallis" \
  "pyyaml" \
  "timm==0.6.13" \
  "termcolor" \
  "sentencepiece" \
  "protobuf" \
  "ftfy" \
  "regex" \
  "scipy" \
  "matplotlib" \
  "opencv-python-headless" \
  "imageio[ffmpeg]" \
  "av" \
  "moviepy" \
  "tensorboard" \
  "tensorboardX" \
  "pytz" \
  "qwen-vl-utils" \
  "wandb" \
  "beautifulsoup4" \
  "yapf" \
  "addict" \
  "ml_collections" \
  "absl-py" \
  "fire" \
  "plyfile" \
  "came-pytorch" \
  "webdataset" \
  "lmdb"
MMCV_WITH_OPS=0 pip install --user --no-cache-dir --no-build-isolation -c /tmp/musa_constraints.txt mmcv==1.7.2

# Install pure-PyTorch MUSA shim for fla.modules.ShortConvolution (exact state_dict parity, Triton-free)
SITE_DIR="$PYTHONUSERBASE/lib/python3.10/site-packages"
mkdir -p "$SITE_DIR/fla/modules"
cat > "$SITE_DIR/fla/__init__.py" <<'PY'
# Pure-PyTorch MUSA shim for fla (only ShortConvolution is used by SANA-WM)
PY
cat > "$SITE_DIR/fla/modules/__init__.py" <<'PY'
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConvolution(nn.Conv1d):
    """Pure-PyTorch drop-in replacement for fla.modules.ShortConvolution."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = "silu",
        use_fast_conv1d: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        x_t = x.transpose(1, 2).contiguous()
        k = self.kernel_size[0]
        if cache is not None:
            pad_len = k - 1
            if cache.shape[-1] > pad_len:
                cache = cache[..., -pad_len:]
            x_cat = torch.cat([cache.to(x_t.dtype), x_t], dim=-1)
            y_t = F.conv1d(x_cat, self.weight, self.bias, padding=0, groups=self.groups)
            next_cache = x_cat[..., -pad_len:] if output_final_state else None
        else:
            y_t = F.conv1d(x_t, self.weight, self.bias, padding=k - 1, groups=self.groups)[..., :T]
            if output_final_state:
                if T >= k - 1:
                    next_cache = x_t[..., -(k - 1):] if k > 1 else x_t[..., :0]
                else:
                    next_cache = F.pad(x_t, (k - 1 - T, 0))
            else:
                next_cache = None

        if self.activation in ("silu", "swish"):
            y_t = F.silu(y_t)
        elif self.activation is not None:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return y_t.transpose(1, 2).contiguous(), next_cache
PY

python3 -c "import torch, torch_musa, diffusers, transformers, accelerate, mmcv, einops, pyrallis, fla.modules; print('ENV_READY', torch.__version__, diffusers.__version__, transformers.__version__, mmcv.__version__)"
