# Baseline: SANA-WM Stage-1 Bidirectional FSDP2 on 4x MTT S5000

*Archived note: internal GPU microarchitecture identifiers, measured hardware peak/bandwidth specifications, and proprietary vendor kernel symbol names were redacted after the run; step timings, category shares, memory, and numerics figures are unchanged.*

- Upstream repository and commit: `targets/reference` (`3daa4fbef442e4ee2b7cb6cdb1f26b295312915d`, MUSA-compatible baseline of `NVlabs/Sana`).
- Entry point: `train_video_scripts/train_sana_wm_stage1.py` launched via `torchrun --nnodes=1 --nproc-per-node=4`.
- Frozen recipe (`$ASSETS/recipe_stage1.yaml`):
  - Model: `SanaMSVideoCamCtrl_1600M_P1_D20` (2.66B parameters, 20 blocks: 15 bidirectional GDN+UCPE blocks and 5 Softmax+UCPE blocks at `softmax_every_n=4`, `GLUMBConvTemp` FFN, `chunk_plucker_post_attn` on 20 blocks).
  - Text encoder: `gemma-2-2b-it` (frozen, online prompt encoding per step).
  - Workload: 193 frames (`T=25` latent frames at 704x1280 -> `22x40` spatial latent = 22,000 tokens), `batch_size=1`/GPU, `DP=4` with FSDP2 (`use_fsdp=true, fsdp_version=2, cp_size=1, grad_checkpointing=true`), `BF16` mixed precision + `fp32_attention=true`, FP32 master weights and FP32 AdamW (`lr=1e-5, betas=[0.9, 0.999], eps=1e-10, weight_decay=0.0, gradient_clip=0.5`).
  - Hardware: 4x Moore Threads MTT S5000 (80 GiB HBM/GPU).

## Frozen reference timing (`task/timing_ref.json`)

Timing plan: 3 warmup iterations + 2 windows of 5 iterations (10 timed iterations per run), 2 independent cold-cache runs:

| Metric | Value |
| --- | --- |
| Reference run 0 (10 timed iters) | `115.725 s` |
| Reference run 1 (10 timed iters) | `115.713 s` |
| Frozen reference mean (`total_s`) | **`115.719 s`** |
| Frozen reference (`s_per_iteration`) | **`11.5719 s/iter`** |
| Run-to-run spread | **`0.01%`** (`0.0001`) |
| Window ratio (`w0 / w1`) | `0.9985` |
| Peak memory allocated (max over ranks) | `34.68 GiB` |
| Peak memory reserved (max over ranks) | `35.77 GiB` |
| Per-rank, per-iteration RNG identical across runs | `true` |

- **Effective Model FLOPs (per GPU, without GC)**: `448.07 TFLOPs/step` (`3×` DiT forward `148.54 TFLOPs` + `1×` frozen `Gemma-2-2B-IT` forward `2.45 TFLOPs`).

## Baseline profile (`torch.profiler`, 2 iterations after warmup, rank 0)

- Profiled window: `24.05 s` for 2 iterations (instrumented host times: `12.395 s`, `11.651 s`).
- GPU busy (union of kernels): `22.10 s` (`92%` of window; `94.7%` of kernel span `23.34 s`).
- Total kernel launches: `148,557` kernels across `287` kernel classes (`~74,278` launches/iter, `6` launches/ms), 2 streams.
- Kernel utilization flags (`kernel_util.py`):
  - `short` (< 5 us median): `578.0 ms` (`2.6%`), `21,601` launches
  - `under-filled` (SM cov < 0.5): `3,391.6 ms` (`15.3%`), `39,379` launches
  - `tail` (1 < waves < 1.5): `1,633.6 ms` (`7.4%`), `49,393` launches
  - `skewed` (mean > 2x median): `8,893.1 ms` (`40.2%`), `83,130` launches

### Kernel time by category (2 iterations, rank 0)

| Category | Total ms | Share of kernel time |
| --- | ---: | ---: |
| `gemm` | 5,026.5 | 22.7% |
| `memcpy/cast` | 4,838.3 | 21.9% |
| `elementwise/reduce` | 3,688.3 | 16.7% |
| `conv` | 3,146.0 | 14.2% |
| `mccl/comm` | 2,666.7 | 12.1% |
| `attention` | 1,160.3 | 5.2% |
| `other` | 300.0 | 1.4% |
| `optimizer` | 93.8 | 0.4% |

### Top kernel classes (2 iterations, rank 0)

| Kernel class (vendor symbols generalized) | Calls | Total ms | % | Median us | Blocks | SM cov | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Depthwise 2D conv forward (BF16, NCHW) | 320 | 1,338.0 | 6.1% | 1,523.6 | 53,900 | 1.00 | skewed |
| MCCL `AllGather` (ring) | 42 | 1,007.8 | 4.6% | 24,970.9 | 24 | 0.40 | under-filled |
| Elementwise binary `Add` (FP32) | 19,856 | 995.9 | 4.5% | 5.3 | 62 | 0.79 | tail skewed |
| Depthwise 2D conv backward data (BF16, NCHW) | 160 | 935.5 | 4.2% | 1,659.1 | 48,125 | 1.00 | skewed |
| Tensor copy/transpose (16-bit) | 2,606 | 932.3 | 4.2% | 202.2 | 24,750 | 1.00 | |
| Depthwise 2D conv backward filter (BF16, NCHW) | 160 | 872.5 | 3.9% | 2,007.8 | 8,960 | 1.00 | skewed |
| FP32 batched `sgemm` (`NN`, small) | 11,942 | 701.6 | 3.2% | 19.4 | 20 | 0.38 | under-filled skewed |
| Vendor BF16 1x1/NHWC convolution | 240 | 687.9 | 3.1% | 2,151.8 | 60 | 1.00 | |
| Tensor flip (`at::native::FlipKernelImpl`) | 1,800 | 656.3 | 3.0% | 464.0 | 96,250 | 0.90 | |
| Vendor FlashAttention backward (BF16) | 20 | 633.5 | 2.9% | 31,545.0 | 3,440 | 1.00 | |
| Elementwise broadcast binary op (FP32) | 2,132 | 543.2 | 2.5% | 294.0 | 14,000 | 1.00 | |
| Elementwise binary `Mul` (FP32) | 2,048 | 528.8 | 2.4% | 310.7 | 12,032 | 0.84 | |
| Contiguous 32-bit copy | 15,604 | 515.8 | 2.3% | 5.0 | 80 | 0.80 | short tail skewed |
| General 16-bit strided copy | 913 | 487.5 | 2.2% | 534.4 | 12,032 | 0.90 | |
| General 32-bit strided copy | 1,900 | 470.9 | 2.1% | 18.9 | 108 | 0.75 | skewed |
| Vendor FlashAttention forward (BF16) | 40 | 470.8 | 2.1% | 11,677.8 | 60 | 1.00 | |
| Vendor BF16 implicit-GEMM conv2d backward filter | 120 | 469.9 | 2.1% | 3,095.1 | 2,673 | 1.00 | |
| FP32 batched `sgemm` (`NT`, small) | 2,760 | 434.6 | 2.0% | 19.4 | 20 | 0.38 | under-filled |
| MCCL `ReduceScatter` (FP32, ring) | 42 | 422.1 | 1.9% | | | | |

### Top repeated kernel chains (`kernel_util.md`)

- Chain #1: 3 kernels (`sgemm_nn` -> copy -> `binary AddOp`), `6,522` occurrences, `192.4 ms` (`0.9%`), SM cov `0.66`, `2/3` flagged.
- Chain #2: 3 kernels (`binary AddOp` -> copy -> copy), `4,273` occurrences, `115.0 ms` (`0.5%`), SM cov `0.81`.
- Chain #3: 9 kernels (`copy/binary/sgemm_tn/sgemm_nn` loop), `544` occurrences, `46.5 ms` (`0.2%`), SM cov `0.75`, `2/9` flagged.
- Chain #4: 8 kernels (`copy/binary/sgemm_tn/sgemm_nt` loop), `538` occurrences, `40.1 ms` (`0.2%`), SM cov `0.73`, `2/8` flagged.

### Top CPU self-time ops (2 iterations, rank 0)

- `musaMemcpyAsync`: `10,022.6 ms` (194 calls, device sync / host wait)
- `musaLaunchKernel`: `6,886.9 ms` (`146,145` calls)
- `aten::nonzero`: `5,302.8 ms` (62 calls, implicit device-to-host sync)
- `aten::_copy_from`: `2,190.1 ms` (`42,220` calls)
- `aten::bmm`: `1,864.1 ms` (`20,382` calls)
- `aten::add_` / `aten::add` / `aten::zero_` / `aten::mul`: `3,949.8 ms` (`50,668` calls)
