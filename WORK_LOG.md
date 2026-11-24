# SANA-WM Stage-1 on 4x MTT S5000 — Optimization Work Log

*Archived note: internal GPU microarchitecture identifiers, measured hardware peak/bandwidth specifications, compiler/runtime internal flags, and proprietary vendor kernel symbol names have been redacted; all round timings, speedups, memory numbers, numerics verdicts, and algorithmic/systems optimization details are preserved.*


- Baseline frozen reference: `11.5719 s/iter` (`task/timing_ref.json`, `34.68 GiB` peak allocated, `35.77 GiB` reserved).
- Target: `>= 5.0x` PASS (`<= 2.314 s/iter`); minimum wrap-up: `>= 2.0x` PASS (`<= 5.786 s/iter`).

---

## Round 0 — Baseline Profile & Bytes Model (`runs/r0_baseline/profile`)
- **Start / End**: 2026-11-23 04:27 – 04:37 CST
- **Command**: `python3 bench/bench.py profile --tree Sana --out runs/r0_baseline/profile --iters 2`
- **Window**: `24.11 s` for 2 iters (`12.348 s`, `11.762 s` instrumented); GPU busy `22.25 s` (`92%` of window, `95.0%` of kernel span); `148,557` launches (`~74,278`/iter), `287` classes, 2 streams.
- **Stage breakdown (per iter, iter 0 / iter 1)**:
  - `text_encoder` (Gemma-2-2B-IT): `138.6 / 72.4 ms` wall (`26 ms` GPU) — `<1%`
  - `dit_forward`: `2,996 / 3,019 ms` wall (`2,929 ms` GPU)
  - `dit_backward + recompute + clip_grad_norm`: `8,077 / 8,086 ms` wall (`7,930 ms` GPU) — `dit.self_attn`, `dit.mlp`, `dit.cross_attn` each called `80` times in 2 iters (`2x` due to `grad_checkpointing=True`)
  - `Optimizer.step#AdamW`: `421 ms` wall (`178 ms` GPU, `6,168` launches/iter from `850` single-param groups!)
- **Kernel utilization & bytes model (`runs/bytes_model.json`, bytes model)**:
  - `short`: `63.8 ms` (`0.3%`, 6,302 launches); `under-filled`: `3,559.1 ms` (`16.0%`, 39,379 launches); `tail`: `1,633.8 ms` (`7.3%`, 49,393 launches); `skewed`: `8,888.3 ms` (`39.9%`, 83,130 launches); `bw-poor`: `6,621.8 ms` (`29.8%`, 13,086 launches).
  - Top kernel classes (2 iters):
    1. `conv2d_depthwise_nchw_native_epilogue<bf16>`: `1,338.4 ms` (6.0%, 320 calls = 80× `GLUMBConvTemp` @ `12.37 ms` + 240× `conv_k`/`conv_k_cam` @ `1.52 ms`), `skewed bw-poor`
    2. MCCL `AllGather` (ring): `1,046.8 ms` (4.7%, 42 calls, med `25.57 ms`, SM cov `0.40`), `under-filled bw-poor`
    3. `binary::kernel_binary<float, AddOp>`: `995.7 ms` (4.5%, 19,856 calls, SM cov `0.79`), `tail skewed`
    4. `conv2d_bwd_data_depthwise_nchw_native<bf16>`: `934.3 ms` (4.2%, 160 calls), `skewed bw-poor`
    5. `CopyTransposeKernel<short>`: `931.6 ms` (4.2%, 2,606 calls)
    6. `conv2d_bwd_filter_depthwise_nchw_native<bf16>`: `872.2 ms` (3.9%, 160 calls), `skewed bw-poor`
    7. FP32 batched `sgemm` (`NN`, small): `699.5 ms` (3.1%, 11,942 calls, 20 blocks, SM cov `0.38`), `under-filled skewed bw-poor`
    8. vendor BF16 1x1/NHWC convolution: `687.5 ms` (3.1%, 240 calls)
    9. `FlipKernelImpl<4>`: `655.1 ms` (2.9%, 1,800 calls)
    10. `flash_atten_bwd` (`632.6 ms`, 20 calls) & `flash_atten_fwd` (`466.3 ms`, 40 calls, compute-bound)
    11. MCCL `ReduceScatter` (FP32, ring): `441.1 ms` (2.0%, 42 calls, med `10.63 ms`, SM cov `0.40`)
    12. `mul_tensor<c10::complex<double>>`: `354.5 ms` (1.6%, 240 calls) — FP64 RoPE in `_apply_complex_rope`!
  - Top repeated chains:
    - `#1`: 3 kernels (`sgemm_nn -> CopyLastContiguous -> binary AddOp`), `6,721` occ, `198.1 ms` (0.9%, SM cov `0.66`, `1/3` flagged)
    - `#2`: 3 kernels (`binary AddOp -> Copy -> Copy`), `4,286` occ, `115.2 ms` (0.5%, SM cov `0.81`)
    - `#3`: 9 kernels (GDN bwd loop), `565` occ, `48.0 ms` (0.2%, `2/9` flagged)
    - `#4`: 8 kernels (GDN bwd loop), `536` occ, `39.7 ms` (0.2%, `2/8` flagged)
  - Top CPU syncs: `aten::nonzero` `5,301.6 ms` (62 calls) from `raymats[mask] = torch.eye(4)` in `world_to_ray_mats` called 30×/iter; `Optimizer.step#AdamW` `2,069.8 ms` CPU across 850 single-param groups.
- **Fast-path inventory (baseline)**:
  - Dense Softmax Attention (5 blocks): hits vendor FlashAttention.
  - GDN + UCPE (15 blocks): pure-PyTorch `for t in range(25)` FP32 `bmm` fallback (`sana_gdn_blocks_musa.py`).
  - Depthwise convs (`GLUMBConvTemp.depth_conv`, `conv_k`, `conv_k_cam`): stock NCHW depthwise conv (severely bandwidth-underutilized).

---

## Round 1 — Host Syncs (`aten::nonzero`), Per-Step UCPE/RoPE Cache, and Batched-Group AdamW (`PASS`, N0)
- **Start / End**: 2026-11-23 04:39 – 05:05 CST
- **Target hotspot (Host / Launch / Redundant Work)**:
  - 62× `aten::nonzero` device syncs (`5.3s` CPU wait) in `world_to_ray_mats` (`raymats[mask] = eye`).
  - `world_to_ray_mats` + `_invert_SE3` + `_prepare_rope_tables` recomputed 30×/iter (+ unused `compute_up_lat_map` in `prepare_prope_fns`).
  - `AdamW.step()` launching 6,168 kernels/iter (`421 ms`/iter) across 850 1-param groups.
- **Changes** (committed on `opt`):
  - `sana_camctrl_blocks.py`: `torch.where` in `world_to_ray_mats`; skip unused `compute_up_lat_map` in `prepare_prope_fns`; add `UCPEForwardCache` / `prepare_ucpe_cache`.
  - `sana_multi_scale_video_camctrl.py` & `sana_gdn_blocks_musa.py`: compute `UCPEForwardCache` once per forward and reuse across all 20 blocks.
  - `diffusion/utils/optimizer.py`: `_BatchedGroupAdamW` keeps all 850 `param_groups` intact (exact LR/hook compatibility) while batching identical-hyperparam groups into single `_foreach` calls.
- **Verification (`runs/r1_host_sync_opt/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N0`, `33,176/33,176` tensors bitwise identical, `0` errors)
  - Timing: **`11.150 s/iter`** (vs `11.5719 s/iter`, **`speedup = 1.0378x`**; `peak_alloc = 34.67 GiB`, `window_ratio = 1.001`).

---

## Round 1 Profile (`runs/r1_host_sync_opt/profile`)
- Window: `22.48 s` (`130,167` launches, `-18,390` vs R0); GPU busy `97.2%`.
- `aten::nonzero` eliminated (`5,302 ms -> 0 ms`); `musaMemcpyAsync` dropped `10,091 ms -> 543 ms`; `optimizer` dropped out of top categories (`< 15 ms`).
- Top remaining classes (2 iters): `conv2d_depthwise_nchw_native` (`3,145 ms` fwd+bwd), `krsc_nhwc_conv` + `implicit_gemm_conv2d_bwd_filter` + `dgrad` (`1,435 ms`), `mccl AllGather` (`1,017 ms`), `GDN` pure-PyTorch `sgemm` + `AddOp` + `MulOp` + `Flip` + `CopyTranspose` (`~9+ s`).

---

## Round 2 — Fused MUSA C++ `GLUMBConvTemp` (`PASS`, N1)
- **Start / End**: 2026-11-23 05:10 – 05:38 CST
- **Target hotspot (Conv / Memory-bound)**: `GLUMBConvTemp` (`95.18 ms/layer` fwd+bwd = `~1.9 s/iter` without GC / `~2.6 s/iter` with GC; severely bandwidth-underutilized `depth_conv` + NCHW permutes + 1×1 implicit GEMM convs).
- **Changes** (committed on `opt`):
  - Added `Sana/diffusion/model/musa_ops/glumbconv.py` (MUSA C++ `mcc` (`-O3`) shared library, container-local `/tmp` SHA256 cache with atomic replace):
    - `inverted_conv` & `point_conv`: direct contiguous `F.linear` on `(B, N, C)` (zero NCHW permutes).
    - `_FusedSiluDwConv3x3GluFn`: fused NHWC `SiLU + 3×3 depthwise conv + GLU` forward and split-pass deterministic backward (32×`__mt_bfloat162` = 64 ch = 128 B/warp, zero bank conflicts, zero local-array spills, zero `atomicAdd`). Single-layer microbench: `95.18 ms -> 37.81 ms` (**2.52×**), saved activations `2.24 GB -> 0.80 GB` (**2.8×** less VRAM).
    - `_TemporalConv3ResidualFn`: direct contiguous 3-tap temporal BF16 GEMMs + residual on `(B, T, S, C)` (zero NCHW permutes; `9.05 ms -> 7.21 ms` + saved transposes).
  - Wired into `GLUMBConvTemp.forward()` in `Sana/diffusion/model/nets/basic_modules.py` (guarded by `can_use_fused_glumbconv_temp` and `SANA_OPT_FUSED_GLUMB`).
- **Verification (`runs/r2_fused_glumbconv/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` dtype changes, `0` precision mode changes, `0` exact failures)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = 0.000σ`, `param_disp_l2 = -0.0088σ`, `step_disp_top1pct = -0.0053σ` (`verdict: equivalent`).
  - Timing: **`9.565 s/iter`** (vs `11.5719 s/iter`, **`speedup = 1.2098x`**, `-1.585 s/iter` vs R1; `peak_alloc = 32.93 GiB` [`-1.74 GiB`], `window_ratio = 1.0013`).

---

## Round 2 Profile (`runs/r2_fused_glumbconv/profile`)
- Window: `19.27 s` (vs `22.48 s` R1); `conv2d_depthwise_nchw_native` & `krsc_nhwc_conv` from `GLUMBConvTemp` eliminated.
- Remaining top bottlenecks (2 iters): pure-PyTorch bidirectional GDN + UCPE + `ShortConvolution` (`~9.5 s` / 2 iters, >50% of GPU time, including `FlipKernelImpl` `844 ms`, `complex<double>` RoPE `354.5 ms`, `sgemm` + `AddOp`/`MulOp`/`CopyTranspose`), full-layer `grad_checkpointing=True` recompute (~35–40% of total compute), and MCCL `AllGather` (ring) (`1,020 ms` / 2 iters, 0% overlap).

---

## Round 3 — MUSA C++ Fused Bidirectional GDN+UCPE & FP32 Real RoPE (`PASS`, N1)
- **Start / End**: 2026-11-23 05:40 – 06:22 CST
- **Target hotspot (Compute / Memory / Launch)**: 15 `BidirectionalGDNUCPEMusa` blocks (`259.5 ms/layer` fwd+bwd, `9.79 GiB/layer` saved activations) + FP64 `complex<double>` RoPE across all 20 blocks.
- **Changes** (commit `7b98534` on `opt`):
  - Added `Sana/diffusion/model/musa_ops/bidi_gdn.py` (5 fused MUSA C++ + autograd ops, 100% deterministic forward & backward without `atomicAdd`, default `SANA_GDN_BMM_DTYPE=bf16` matching upstream Triton `dot_precision=0`):
    - `fused_bidi_short_conv4`: symmetric 7-tap bidirectional temporal conv directly on `(B, T, S, C)` BF16 (0 transposes/flips; `0.74 ms` f+b).
    - `_FusedMainPrepFn` & `_FusedCamPrepFn`: fused full-channel RMSNorm + ReLU + `k_scale` + 4×4 `P_T`/`P_inv` + FP32 RoPE + deterministic per-head `inflation_sq` -> direct `(BH, N, D)` output.
    - `_FusedBidiMainScanFn` & `_FusedBidiCamScanAndOutFn`: shared Phase A (computed once for both directions, eliminating `torch.flip`/`flip_and_shift` and half of Phase A/C GEMMs), MUSA C++ bidirectional `112×112` state scan (`bidi_state_scan_fwd/bwd_d112_f32`: `1.07 ms` fwd / `0.87 ms` bwd, padded `s_W[112][113]` for bank-conflict-free shared-memory access), `bidi_z_scan_fwd/bwd`, single Phase C on `S_tot = S_fwd + S_rev`, fused `out_div_to_bnc` and `cam_out_apply_o` (4×4 `P` + FP32 inverse RoPE).
    - Unit test verified vs CPU PyTorch reference (`out ~5e-5`, all grads `dQ/dK/dV/dbeta/ddecay ~7e-7` in FP32 mode; 0.0 bitwise determinism across repeated runs). Single-layer microbench: `259.46 ms -> 60.44 ms` (**4.29×**), saved activations `9.79 GiB -> 2.29 GiB` (**-7.5 GiB/layer**). Also uncovered a `torch_musa` autograd bug in the original `flip_and_shift` path on MUSA (`dK` 96.65% off vs CPU/FD), bypassed cleanly by the fused scan.
  - Wired fused GDN+UCPE into `Sana/diffusion/model/nets/sana_gdn_blocks_musa.py` (guarded by `can_use_fused_bidi_gdn` / `SANA_OPT_FUSED_GDN`).
  - Replaced FP64 `complex128` RoPE in `sana_camctrl_blocks.py` (`_apply_complex_rope`, pre-extracting `cos_sin`) and `sana_gdn_blocks.py` (`_apply_rotary_emb`) with FP32 real arithmetic.
- **Verification (`runs/r3_fused_bidi_gdn/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, `0` dtype/precision mode changes)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = -0.0177σ`, `param_disp_l2 = +0.4338σ`, `step_disp_top1pct = -0.4493σ` (`verdict: equivalent`).
  - Timing: **`5.3497 s/iter`** (vs `11.5719 s/iter`, **`speedup = 2.1631x`**, `-4.215 s/iter` vs R2; `peak_alloc = 25.73 GiB` [`-8.95 GiB` vs baseline], `peak_reserved = 26.78 GiB`, `window_ratio = 1.0024`).

---

## Round 3 Profile (`runs/r3_fused_bidi_gdn/profile`)
- Window: `11.51 s` (`10.52 s` GPU kernel / 2 iters, 97.1% busy, `32,545` launches).
- Top remaining bottlenecks (2 iters):
  - Full-layer `grad_checkpointing=True` (80 block forwards/2 iters, ~33% of compute; Softmax blocks 2× costlier to recompute than GDN blocks).
  - `mccl/comm` `1,673 ms` (16%): `AllGather_RING_SIMPLE` `970.7 ms` (42 calls, median `23.9 ms`, 100% serial on stream 0, 0% overlap) + `ReduceScatter_PreMulSum_f32` `337.4 ms` (42 calls, median `6.78 ms`).
  - `elementwise/reduce` (`1,469 ms`) + `memcpy/cast` (`1,368.5 ms`): `KernelConcatLastSmallDim<int,2>` `302.2 ms` + `kernel_unary_uncontig` `398.3 ms` (from Softmax+UCPE `_forward_cam_branch_softmax` redundant `bf16->fp32->bf16` & RoPE `torch.stack`), pure-PyTorch `RMSNorm` (`PowOp` `81.7 ms`), `LayerNorm+modulate` & `gate+residual`.
  - `attention` (`1,174 ms`, half from recompute) and `fused_silu_dwconv3x3_glu` (`766.5 ms`).

---

## Round 4 — Memory-Lean Combined GDN Autograd + Selective GC (`gdn:9`) (`PASS`, N1)
- **Start / End**: 2026-11-23 06:26 – 07:06 CST
- **Target hotspot (Recompute / Memory)**: Full 20-layer `grad_checkpointing=True` recomputing all 5 expensive Softmax+UCPE blocks and all 15 GDN blocks.
- **Experiments & findings**:
  - Added memory-lean `_FusedMainPrepAndScanFn` and `_FusedCamPrepAndScanFn` in `bidi_gdn.py` (recomputing ~1 ms prep in backward instead of saving 8 large intermediate tensors + `I_m_P*`; single GDN attn saved activations `2.29 -> 1.86 GiB`).
  - Measured per-block activation footprint: GDN block `4.29 GiB` (`attn 1.95 + cross 0.74 + mlp 0.80 + norms ~0.80`), Softmax block `5.66 GiB` (`attn 3.32 + cross 0.74 + mlp 0.80 + norms ~0.80`).
  - `gdn:0` and `gdn:7` OOMed at ~77 GiB; `gdn:11` ran at `4.6357 s/iter` (`2.496x`, `63.26 GiB` alloc). Tightened to `gdn:9` (`71.28 GiB` alloc / `72.37 GiB` reserved).
  - First `gdn:9` check (`r4_selective_gc`) hit 8 record key mismatches because turning off FSDP2 `activation_checkpointing` removed `CheckpointWrapper` (`_checkpoint_wrapped_` prefix in buffer names). Fixed cleanly in `r4_selective_gc_v2` by keeping FSDP2 `activation_checkpointing=True` intact while bypassing `CheckpointWrapper.checkpoint_fn` with a passthrough callable for zero-recompute blocks (and disabling redundant inner `auto_grad_checkpoint`).
- **Changes** (commit on `opt`):
  - `Sana/diffusion/model/musa_ops/bidi_gdn.py` & `sana_gdn_blocks_musa.py`: combined lean prep+scan autograd functions.
  - `Sana/train_video_scripts/train_sana_wm_stage1.py` (`_configure_selective_activation_checkpointing` called from `_remove_custom_module_call`): bypasses `CheckpointWrapper.checkpoint_fn` on all 5 Softmax blocks `[3,7,11,15,19]` + 6 GDN blocks `[12,13,14,16,17,18]` (11/20 zero-recompute; 9/20 cheapest GDN blocks recomputed), preserving exact `CheckpointWrapper` module/buffer names.
  - `Sana/diffusion/model/utils.py`: `SANA_GC_NUM_BLOCKS` support + pop `use_reentrant` in non-checkpoint path of `auto_grad_checkpoint`.
- **Verification (`runs/r4_selective_gc_v2/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, `0` dtype/precision changes)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = -0.0177σ`, `param_disp_l2 = +0.4334σ`, `step_disp_top1pct = -0.4657σ` (`verdict: equivalent`).
  - Timing: **`4.5606 s/iter`** (vs `11.5719 s/iter`, **`speedup = 2.5374x`**, `-0.789 s/iter` vs R3; `peak_alloc = 71.28 GiB`, `peak_reserved = 72.37 GiB`, `window_ratio = 0.9993`).

---

## Round 5 — 16-Byte-Aligned FSDP2 `AllGather` / `ReduceScatter` (`PASS`, N1, Bitwise Identical to R4)
- **Start / End**: 2026-11-23 07:07 – 07:23 CST
- **Target hotspot (Communication)**: MCCL `AllGather` (ring) taking `23.9 ms`/block (`970.7 ms`/2 iters = `485 ms/iter`) while FP32 `ReduceScatter` ran ~7× faster per byte.
- **Root-cause diagnosis (4-GPU MCCL microbench)**:
  - Bare 4× MTT S5000 `all_gather_into_tensor(260 MB BF16)` achieves **`2.28 ms`** when `bytes_per_rank % 16 == 0`, but falls back to **`24.40 ms` (`10.7×` slower!)** whenever `bytes_per_rank % 16 != 0` (`mod16 in {2, 4, 8}`).
  - Each `SanaVideoMSCamCtrlBlock` shard has `n_elems % 8 == 4`, so FP32 `ReduceScatter` (`4*n_elems % 16 == 0`) hit the fast 128-bit path while BF16 `AllGather` (`2*n_elems % 16 == 8`) hit the slow scalar fallback on all 20 blocks.
  - Also confirmed `moore-infra` rule: overlapping `AllGather` with GEMM on MTT S5000 caused MP contention (`OVL 19.49 ms > serial 18.93 ms`), so pure aligned fast-path comm without overlap is optimal.
- **Changes** (commit on `opt`):
  - `Sana/train_video_scripts/train_sana_wm_stage1.py`: installed `Aligned16AllGather` and `Aligned16ReduceScatter` via official PyTorch FSDP2 `set_custom_all_gather` / `set_custom_reduce_scatter` inside `_configure_aligned_fsdp_comms` (pads per-rank slice by <= 14 bytes to 16-byte boundary, copies back with 1D contiguous rank slices; guarded by `SANA_OPT_ALIGNED_FSDP_COMM=1`).
  - 4-GPU unit microbench: unaligned BF16 `AllGather` `24.61 -> 2.78 ms` (**8.85×**), unaligned FP32 `ReduceScatter` `32.76 -> 2.73 ms` (**11.99×**), `max_abs = 0.0`.
- **Verification (`runs/r5_aligned_fsdp_comm/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, bitwise identical trajectory endpoints to R4: `train_loss = -0.0177σ`, `param_disp_l2 = +0.4334σ`, `step_disp_top1pct = -0.4657σ`).
  - Timing: **`4.1376 s/iter`** (vs `11.5719 s/iter`, **`speedup = 2.7968x`**, `-0.423 s/iter` vs R4; `peak_alloc = 71.28 GiB` [0 extra VRAM], `window_ratio = 0.9985`).

---

## Round 5 Profile (`runs/r5_aligned_fsdp_comm/profile`)
- Window: `9.13 s` (`8.17 s` GPU kernel / 2 iters, 90% busy, `28,949` launches).
- `AllGather_RING_SIMPLE` dropped **`970.7 -> 155.6 ms` (6.24×)**.
- Top remaining bottlenecks (2 iters): `gemm` (`2,321 ms`), `elementwise/reduce` (`1,180 ms`) + `memcpy/cast` (`997 ms`) = `2,177 ms` (`26.6%`, including `KernelConcatLastSmallDim` `148 ms`, `uncontig CastOp` `221 ms`, `CopyTranspose` `169 ms`, `PowOp` `63 ms`), `attention` (`926 ms`), `conv` (`739 ms`, `fused_silu_dwconv3x3_glu`), `mccl/comm` (`754 ms`, incl. `ReduceScatter` `296 ms`, `AllGather` `156 ms`, `multi_tensor_apply` `152 ms`), and 9 remaining GDN blocks recomputed.

---

## Round 6 — MUSA C++ Block-Level Fusion + Softmax+UCPE Direct `D=112` + `gdn:6` (`PASS`, N1)
- **Start / End**: 2026-11-23 07:27 – 08:05 CST
- **Target hotspot (Elementwise / Memory / Softmax+UCPE / Recompute)**: `2,177 ms`/2 iters of elementwise/cast/transpose/pad overhead + high activation footprint in Softmax+UCPE blocks (`3.32 GiB` attn) and block norms/gates (`~0.80 GiB`/layer).
- **Changes** (commit on `opt`):
  - Added `Sana/diffusion/model/musa_ops/block_fusion.py` (6 deterministic zero-`atomicAdd` MUSA C++ fused ops):
    1. `fused_rmsnorm`: single-pass BF16->FP32->BF16 RMSNorm with split-chunk deterministic `dw` reduction (wired into `norms.py:RMSNorm.forward`; cuts `cross_attn` saved activations `0.74 -> 0.47 GiB`).
    2. `fused_layernorm_modulate`: single-pass `LayerNorm(no affine) + t2i_modulate` saving only `(x, scale, mean, rstd)` (wired into `SanaVideoMSCamCtrlBlock.forward_frame_aware`).
    3. `fused_gate_residual`: single-pass `x + gate * y` with frame-split deterministic `dgate` reduction.
    4. `fused_silu_mul`: single-pass FP32 `out * silu(g)` -> BF16 (wired into `sana_gdn_blocks.py:_apply_output_gate`).
    5. `fused_softmax_main_prep`: single-pass RMSNorm + FP32 RoPE -> contiguous `(B, H, N, 112)` BF16 directly into native `D=112` SDPA (benchmarked faster than padded `D=128`: `44.83 vs 45.49 ms`, zero `F.pad`).
    6. `fused_softmax_cam_prep` + `fused_cam_apply_o`: eliminates redundant `bf16->fp32->bf16`, 4× permutes, `cat/chunk`, `F.pad`, and `torch.stack` (`KernelConcatLastSmallDim`), cutting Softmax `attn` saved activations `3.32 -> 1.87 GiB` (`-1.45 GiB/layer`).
  - Tightened `SANA_GC_NUM_BLOCKS` from `gdn:9 -> gdn:6` (14/20 blocks zero-recompute; note `gdn:5` timed `3.3446 s/iter` / `3.460x` at `73.14 GiB` alloc, but OOMed by 277 MiB during `traj` `track_disp=True` +3.5 GiB param snapshot; `gdn:6` leaves ~9.4 GiB headroom).
- **Verification (`runs/r6_block_fusion_gdn6/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, `0` dtype/precision changes)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = -0.0177σ`, `param_disp_l2 = +0.4288σ`, `step_disp_top1pct = -0.4549σ` (`verdict: equivalent`).
  - Timing: **`3.3990 s/iter`** (vs `11.5719 s/iter`, **`speedup = 3.4045x`**, `-0.739 s/iter` vs R5; `peak_alloc = 69.69 GiB`, `peak_reserved = 70.51 GiB`, `window_ratio = 1.004`).

---

## Round 6 Profile (`runs/r6_block_fusion_gdn6/profile`)
- Window: `7.73 s` (`6.628 s` GPU kernel / 2 iters, 94.4% GPU busy, `22,129` launches).
- Elementwise/reduce + memcpy/cast dropped `2,177 -> 976 ms` (**2.23×**).
- Top remaining non-GEMM/non-FA bottlenecks (2 iters): `fused_silu_dwconv3x3_glu` (`676 ms`), `cam_prep_bwd` (`127 ms`), `gate_residual_fwd` (`117 ms`, 64-bit integer div trap!), `bidi_state_scan` (`176 ms`), `main_gdn_prep_bwd` (`100 ms`), `cam_prep_fwd + main_gdn_prep_fwd` (`157 ms`), `ln_modulate_bwd + rmsnorm_bwd + gate_residual_bwd` (`141 ms`), GDN Phase A/C PyTorch `CastOp/MulOp/Reduce/AddOp` (`~380 ms`), `mccl/comm` (`601 ms`), `optimizer` (`124 ms`), `text_encoder` (`220 ms` wall), and 6 remaining GDN blocks recomputed.

---

## Round 7 — Occupancy-Tuned `glumbconv v3` + `block_fusion v2` + Fused GDN Phase A/C + ShortConv4 Absorption + `gdn:5` (`PASS`, N1)
- **Start / End**: 2026-11-23 08:13 – 09:17 CST
- **Commit**: `4578a12` on branch `opt`
- **Target hotspots**: `glumbconv` (`676 ms`), `gate_residual_fwd` 64-bit div trap (`117 ms`), `ln_modulate/rmsnorm/gate_residual` bwd (`141 ms`), `main_gdn_prep/cam_prep` fwd/bwd (`384 ms`), GDN Phase A/C casts/muls/reductions (`~380 ms`), and GDN activation footprint (`gdn:6 -> gdn:5`).
- **Changes**:
  - `Sana/diffusion/model/musa_ops/glumbconv.py` (v3): Tuned for occupancy sweet spot (`SMEM <= 27 KiB`, `regs <= 66`, `local = 0` spills) discovered via micro-ablation (exceeding shared-memory/register occupancy thresholds drops occupancy ~2.5×):
    - `fwd_v3` & `bwd_p1_dadg`: `W_TILE=20, C_tile=64` (`tx=0..31` `bf162` 128B/warp), SMEM weights (`27.1 KiB, 63/66 regs`).
    - `bwd_p2_du` (`21.5 KiB, 46 regs`) & `bwd_p3_dw` (`21.5 KiB, 53 regs`, deterministic zero `atomicAdd`).
    - Layer microbenchmark: `15.75 ms -> 11.79 ms` (`-25.1%`).
  - `Sana/diffusion/model/musa_ops/block_fusion.py` (v2):
    - Eliminated 64-bit integer division in `gate_residual_fwd_bf16` via 2D grid `dim3(S, BT)` (`2.23 ms -> 0.219 ms`, **10.2×**!).
    - Single-pass SMEM caching in `rmsnorm_bwd_bf16` (`1.00 -> 0.62 ms`) and `ln_modulate_bwd_bf16` (`1.76 -> 0.84 ms`); increased `n_chunks = 240` (4 waves across GPU SMs) and `n_splits = 8` (`200 CTAs`).
  - `Sana/diffusion/model/musa_ops/bidi_gdn.py` (v2) & `Sana/diffusion/model/nets/sana_gdn_blocks_musa.py`:
    - Absorbed `bidi_short_conv4` directly into `_FusedMainPrepAndScanFn` and `_FusedCamPrepAndScanFn` using packed-QKV kernels (`bidi_short_conv4_qkv_k_fwd/bwd`) and packed `cam_prep_fwd/bwd`: eliminates `qkv[:, :, 1].contiguous()`, `qkv_cam.chunk(3)` + 3× `contiguous()` + `cat()`, and avoids saving `k_raw`/`k_conv`/`k_conv_cam` (`-296 MB/layer`).
    - Eliminated `num` save (`-98.6 MB/layer`) via `out_div_bwd_from_out_dqpos_bf16` (fuses `dQ_pos` and reads already-computed `out`).
    - Single-pass SMEM caching + `n_chunks = 240` for `main_gdn_prep_fwd/bwd` and `cam_prep_fwd/bwd` (hoisted `d_post/d_pre` per head out of inner loops).
    - Added 9 deterministic fused GDN Phase A/C kernels (`gdn_main/cam_phase_a_scale`, `gdn_make_W`, `gdn_add_cast_S`, `gdn_qp_sz_dot`, `gdn_dden_qp_bwd`, `gdn_ddecay_dp_du_bwd`, `gdn_main/cam_phase_a_epilogue_bwd`) eliminating `~2.5 GB/layer` of BF16->FP32 casts, broadcast muls, and reductions.
    - Microbenchmark: Main full fwd+bwd `14.79 ms` (98 MiB net extra saved), Cam full fwd+bwd `12.90 ms` (75 MiB net extra saved).
    - Cloned small saved weights (`conv_k.weight/bias`, `depth_conv.weight/bias`) so they never pin 250 MiB FSDP2 all-gather storage.
  - Tightened `SANA_GC_NUM_BLOCKS` from `gdn:6 -> gdn:5` (15/20 blocks zero-recompute).
- **Verification (`runs/r7_opt_kernels_gdn5/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, `0` dtype/precision changes)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = -0.0177σ`, `param_disp_l2 = +0.4302σ`, `step_disp_top1pct = -0.4573σ` (`verdict: equivalent`).
  - Timing: **`2.9329 s/iter`** (vs `11.5719 s/iter`, **`speedup = 3.9455x`**, `-0.466 s/iter` vs R6; `peak_alloc = 70.39 GiB`, `peak_reserved = 71.23 GiB`, `window_ratio = 1.0011`).

---

## Round 8 — Micro-Recompute (`u + qkv`, `gdn:0`), Merged GLUMBConv `p23`, 2D-`t` Block Fusion, `bidi_scan v3`, Fused AdamW, Fast Gemma2, Mask-Free CrossAttn (`4.3079x PASS`)
- **Commit**: `4ceb62e` on `opt`
- **Profile-driven diagnosis (`runs/r7_opt_kernels_gdn5/profile` + intermediate `runs/r8_micro_profile`)**:
  - 5 recomputed GDN blocks (`gdn:5`) cost ~140–220 ms/iter while saving ~12.5 GiB. We first tested async PCIe Gen5 ×16 pinned offload (`async_offload.py`): v1 (`ev.synchronize()`) gave `2.8325 s/iter` (`64.22 GiB`), v3 (zero CPU sync, main-stream staging) gave `2.7587 s/iter` (`67.43 GiB`), revealing `~27 ms/block` (`~55 MiB/ms`) actual multi-card overhead.
  - Pure-GEMM **micro-recomputation** beats both whole-block GC and PCIe offload by **3.0×** (`169 MiB/ms` on vendor BF16 GEMMs): discarding only `GLUMBConvTemp.u` (`564 MiB/layer`, `3.33 ms` recompute) across 20 layers (`-11.02 GiB`) + GDN main `qkv` (`282 MiB/layer`, `1.69 ms` recompute) across 15 layers (`-4.13 GiB`) frees `15.15 GiB` for only `92 ms/iter` (100% bitwise exact `max_abs = 0.0`), unlocking `gdn:0` (20/20 zero whole-block recompute).
  - Intermediate profile `runs/r8_micro_profile` caught two huge issues: (a) a scalar `pack_unpack_pad16_u16` kernel regressed `+51.1 ms/iter` (reverted to `.copy_`), and (b) 50% of training steps use 2D `t` (`SanaVideoMSCamCtrlBlock.forward()`), which completely missed `fused_layernorm_modulate` & `fused_gate_residual`, causing 40 native LayerNorms, 194 `CopyTransposeKernel` launches, and `~5.5 GiB` extra peak VRAM.
- **Changes kept**:
  - `Sana/diffusion/model/musa_ops/glumbconv.py`:
    - Added `_FusedInvSiluDwConv3x3GluFn` (discards 564 MiB `u`, recomputes 1 BF16 GEMM in backward, bitwise exact `0.0`).
    - Merged `bwd_p2_du` + `bwd_p3_dw` into single-pass deterministic `fused_silu_dwconv3x3_glu_bwd_p23_du_dw_bf16` (`5.79 -> 4.01 ms/layer`, eliminates `1.13 GB/layer` redundant reads).
    - Replaced 6× `AddOp bf16` in `_TemporalConv3ResidualFn` with in-place `torch.addmm` / `addmm_`.
  - `Sana/diffusion/model/musa_ops/bidi_gdn.py` & `sana_gdn_blocks_musa.py`:
    - Upgraded `bidi_state_scan_fwd/bwd_d112_f32` to v3 2D 4×4 register tiling (`RT=28, KT=16, STR=116, 20.4 KiB SMEM, block=196`): `fwd 1.193 -> 0.590 ms` (2.02×), `bwd 1.200 -> 0.604 ms` (1.99×), `rel_l2 = 4.5e-7`, 100% deterministic.
    - Added `qkv` micro-recompute inside `_FusedMainPrepAndScanFn` (`-4.13 GiB` across 15 GDN layers, bitwise exact `0.0`).
  - `Sana/diffusion/model/musa_ops/fused_adamw.py` & `diffusion/utils/optimizer.py`:
    - Single-pass 128-bit `float4` MUSA C++ `fused_multi_tensor_adamw_f32` with static chunk cache + double-buffered 27 KB pinned pointer table (`~62 ms -> ~23.7 ms/iter`, `rel_p = 6.1e-10`).
  - `Sana/diffusion/model/musa_ops/fast_gemma2.py` & `train_sana_wm_stage1.py`:
    - Zero-extra-VRAM MUSA C++ fused Gemma-2-2B-IT forward (matching HF `sdpa` no-softcap + `use_cache=False`, `~62 ms -> 10.7 ms/iter`).
  - `Sana/diffusion/model/musa_ops/block_fusion.py`, `sana_blocks.py`, `sana_gdn_blocks.py`, `sana_gdn_camctrl_blocks.py`, `sana_multi_scale_video_camctrl.py`:
    - Wired `fused_layernorm_modulate(..., T=1)` + `fused_gate_residual(..., T=1)` (with full-wave `n_splits=240`) into 2D-`t` `SanaVideoMSCamCtrlBlock.forward()`.
    - Added `fast_col_sum_bf16` / `fast_linear_bf16` (`C >= 4096`: 2.2× faster than vendor reduction), `fused_add_silu_mul` (eliminating `main_raw + cam_contrib` `AddOp`), 128-bit `fused_bnc_to_bhnd` / `fused_bhnd_to_bnc`, and `softmax_cam_prep_bwd` register-cached `qkv` (`n_chunks=240`).
    - Compacted valid text tokens once per step in `SanaMSVideoCamCtrl.forward` (`B == 1`) so all 20 `MultiHeadCrossAttention` layers run mask-free SDPA (N0 exact).
    - Default `SANA_GC_NUM_BLOCKS="gdn:0"` (20/20 zero whole-block recompute).
- **Verification (`runs/r8_micro_recompute_fused/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, `0` dtype/precision changes)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = -0.0177σ`, `param_disp_l2 = +0.4307σ`, `step_disp_top1pct = -0.4647σ` (`verdict: equivalent`).
  - Timing: **`2.6862 s/iter`** (vs `11.5719 s/iter`, **`speedup = 4.3079x`**, `-0.247 s/iter` vs R7; `peak_alloc = 69.74 GiB`, `peak_reserved = 70.26 GiB`, `window_ratio = 0.9970`).

---

## Round 9 — 128-bit `uint4` Block Fusion & GDN Prep/Epilogue, `glumbconv v5` (`tanhf` + 1-bar), Fused L2 Clip + AdamW (`4.5206x PASS`)
- **Commit**: `152a180` on `opt`
- **Changes kept**:
  - `Sana/diffusion/model/musa_ops/glumbconv.py`: replaced `1/(1+expf(-x))` with fast hardware `0.5*(1+tanhf(0.5*x))` and 1-barrier ring buffer in `fwd_v3`, `bwd_p1`, and `bwd_p23`.
  - `Sana/diffusion/model/musa_ops/block_fusion.py`: upgraded `ln_modulate_fwd/bwd`, `gate_residual_fwd/bwd`, and `rmsnorm_bwd` to zero-SMEM 128-bit `uint4` 288-thread (280 active × 8ch = 2240) kernels; added `fused_add_layernorm_modulate` (fusing `x + cross_out` into `norm2`); extended `fast_linear_bf16` / `bf16_col_sum` down to `C >= 2048`.
  - `Sana/diffusion/model/musa_ops/bidi_gdn.py`: upgraded `cam_prep_fwd`, `main_gdn_prep_scale_fwd`, `gdn_main/cam_phase_a_epilogue_bwd`, and `out_div` to 128-bit `uint4`.
  - `Sana/diffusion/model/musa_ops/fused_adamw.py`: fused FSDP2 global L2 gradient norm & clip scaling directly into `fused_multi_tensor_adamw_f32` (eliminating separate multi-tensor norm + clip passes).
- **Verification (`runs/r9_u4_tanh_clip/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, `0` dtype/precision changes)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = -0.0177σ`, `param_disp_l2 = +0.4095σ`, `step_disp_top1pct = -0.4608σ` (`verdict: equivalent`).
  - Timing: **`2.5598 s/iter`** (vs `11.5719 s/iter`, **`speedup = 4.5206x`**, `-0.126 s/iter` vs R8; `peak_alloc = 69.74 GiB`, `peak_reserved = 70.24 GiB`, `window_ratio = 0.9997`).

---

## Round 10 — Zero-Copy CrossAttn/FA3, `main_prep_bwd 576t`, `split3/stack3 u4`, `triple_linear`, Plucker Fusion, `qkv=0` + `save_u=2`, `MCCL_BUFFSIZE=16MiB` (`4.6878x PASS`)
- **Commit**: `3f3a904` on `opt` + `configs/opt.json` (`{"environment": {"MCCL_BUFFSIZE": "16777216"}}`)
- **Changes kept**:
  - `Sana/diffusion/model/musa_ops/bidi_gdn.py`: zero-SMEM 576-thread `uint2` `main_gdn_prep_bwd_576_bf16` (`nc=120`, `2.58 -> 1.24 ms/layer`, `-20.1 ms/iter`); `bidi_state_scan_fwd` v4a (`RT=16` double-buffered 128t: `37.9 -> 34.1 ms / 2 iters`; `bwd` kept at v3 after v4a regressed); disabled all 15 GDN `qkv` micro-recompute (`SANA_RECOMPUTE_GDN_QKV=0`).
  - `Sana/diffusion/model/musa_ops/block_fusion.py`, `sana_blocks.py`, `sana_multi_scale_video_camctrl.py`: zero-copy CrossAttn BNHD view & FA3 BNHD output (`fused_bhnd_to_bnc` fast-path); `rmsnorm_fwd_u4`; `ln_modulate/gate_residual` wave fix (`BT=25 -> n_splits=12 = 300 blocks`); `silu_mul` / `add_silu_mul` 128-bit `uint4`; `fast_triple_linear_bf16` (`addmm_` `dx` epilogue); fused `plucker_proj` into `gate_residual(add_bias=plucker.bias) + fused_addmm_bf16` with single `.contiguous()` on `plucker_emb` (eliminating 20× `CopyTranspose` and freeing `1.88 GiB` VRAM).
  - `Sana/diffusion/model/musa_ops/glumbconv.py`: 128-bit `union U4H16` `split3/stack3` + fused `d_b_inv` inside `p23`; `SANA_SAVE_GLUMB_U_LAYERS=2` (saving 2 layers of `u` within safe `train_2step` VRAM ceiling of `~73.2 GiB`; `save_u=8` [75.89 GiB] and `save_u=6` [74.79 GiB] OOMed in Gate 2 `train_2step` step-1 FA3 1.68 GiB workspace).
  - `configs/opt.json`: `MCCL_BUFFSIZE=16777216` cut FSDP2 MCCL collective time by `-38.9 ms/iter` (`404.3 -> 365.4 ms / 2 iters`).
- **Verification (`runs/r10_final/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, `0` dtype/precision changes)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = -0.0177σ`, `param_disp_l2 = +0.4166σ`, `step_disp_top1pct = -0.4344σ` (`verdict: equivalent`).
  - Timing: **`2.4685 s/iter`** (vs `11.5719 s/iter`, **`speedup = 4.6878x`**, `-0.091 s/iter` vs R9; `peak_alloc = 72.59 GiB`, `peak_reserved = 73.12 GiB`, `window_ratio = 0.9995`).
  - Profile: `runs/r10_final/profile/{profile.md,kernel_util.md,kernel_util.json}`.

---

## Round 11 — Full `GLUMBConv` + `AddSiluMulProj` + `QuadLinear` Fusion, 20/20 Zero-Recompute `save_u` via `recompute_ln_mod_u4`, `glumbconv v6` (`smemw + w40_320t`), Zero-Copy FSDP2 Pad16 `AllGather`, `GradAccBox` Plucker Accumulator, `_compute_frame_gates` Fusion (`4.8843x PASS`)
- **Commit**: `9d0eea1` on `opt` + `configs/opt.json` (`{"environment": {"MCCL_BUFFSIZE": "16777216"}}`)
- **Changes kept**:
  - `Sana/diffusion/model/musa_ops/glumbconv.py` & `Sana/diffusion/model/nets/basic_modules.py`:
    - Restored `__restrict__` on `u`/`du` in `p23_du_dw` (recovering R10 `+15.5 ms/iter` aliasing regression) and upgraded to v6 (`s_w` in SMEM saving 18 regs + single-sync parallel reduction: `4.05 -> 3.46 ms/layer`, `-28%` in full profile `199.0 -> 143.4 ms / 2 iters`).
    - Upgraded `fwd` and `bwd_p1` to `W_TILE=W, 320t` (`0%` halo redundancy, exact 4 cols/thread for `W=40`), with free register-resident `y_recomp` emission inside `bwd_p1` .
    - Added `_FusedFullGLUMBConvFn` fusing `point_conv` (never saving 282 MiB `y` across 20 layers = `-5.51 GiB`) and accepting `ln_stats` to avoid saving `x_mlp_in` (`-1.88 GiB`), raising `SANA_SAVE_GLUMB_U_LAYERS` from `2 -> 20` (**20/20 layers zero `u` recompute GEMMs**, cutting total GEMM time `-67.7 ms/iter`).
  - `Sana/diffusion/model/musa_ops/block_fusion.py`, `sana_gdn_blocks_musa.py`, `sana_gdn_camctrl_blocks.py`, `sana_multi_scale_video_camctrl.py`:
    - Added `_FusedAddSiluMulProjFn` (`fused_add_silu_mul_proj`) emitting `y_recomp` inside `silu_mul_bwd_u4` , freeing `1.88 GiB` (`combined`).
    - Added `recompute_ln_mod_u4` (`0.174 ms/call`, bitwise `0.00e+00`) to drop both `x_mlp_in` (`-1.88 GiB`) and `x_msa_in` (`-1.88 GiB`) across all 20 layers.
    - Upgraded `fast_triple_linear_qkv_cam_bf16` (no saved concatenated `qkv_cam_w`, `-0.57 GiB`) and added `fast_quad_linear_qkv_cam_beta_bf16` fusing `beta_proj` `dx` into `addmm_` epilogue (`-15x` `AddOp`), paired with `fused_layernorm_modulate_fmean` fusing `x_frame = mean(dim=S)` in forward and absorbing `d_x_frame/S` inside `ln_modulate_bwd_u4` (eliminating `15x` `ReduceType`, `15x` `FastDivmodU32`, and `15x` `AddOp`).
    - Added `GradAccBox` (opaque holder immune to FSDP2 `pytree.tree_map` list copying) + `flush_shared_grad` + `fused_addmm_bf16(..., acc_holder=...)` to accumulate `d_plucker_emb` in-place via `addmm_` across all 20 layers (eliminating `19x` 94 MiB `AddOp`).
    - Tuned C++ and Python `choose_frame_splits(BT=1)` to `300` (5 full waves across GPU SMs).
  - `Sana/train_video_scripts/train_sana_wm_stage1.py`:
    - Installed zero-copy 16-byte-aligned FSDP2 `foreach_all_gather` / `foreach_all_gather_copy_out` padding via a 4-element tail split in `all_gather_copy_in` and `split_with_sizes_copy`, eliminating all `100x/iter` 65.4 MB `IdentityOp` pack/unpack copies (`-10.3 ms/iter`, bitwise exact).
  - `Sana/diffusion/model/musa_ops/bidi_gdn.py`:
    - Removed redundant `1.48 GB/iter` K-slot zeroing inside `cam_prep_bwd_impl` (immediately overwritten by `bidi_short_conv4_qkv_k_bwd_bf16`).
- **Verification (`runs/r11_final2/check/check.json`)**:
  - `status`: **`PASS`** (`measured_tier: N1`, `0` errors, `0` exact failures, `0` dtype/precision changes, `33,176` checked tensors)
  - Gate 3 statistical trajectory (`seed=3407`, `3σ` bound): `train_loss = -0.0177σ`, `param_disp_l2 = +0.4127σ`, `step_disp_top1pct = -0.4446σ` (`verdict: equivalent`).
  - Timing: **`2.3692 s/iter`** (vs frozen baseline `11.5719 s/iter`, **`speedup = 4.8843x`**, `-0.0993 s/iter` vs R10; `window_s = [11.846, 11.846]`, `window_ratio = 1.0000`, `peak_alloc = 71.03 GiB`, `peak_reserved = 72.42 GiB`).
  - Final Profile: `runs/r11_final2/profile/{profile.md,kernel_util.md,kernel_util.json}` (with `runs/bytes_model.json`).

---

## Hardware Plateau Proof (`runs/r11_final2/profile/kernel_util.md`, `4.8843x PASS`)

On 4× MTT S5000 (80 GiB HBM/GPU), Round 11 reaches **`2.3692 s/iter` (`4.8843x PASS`)** with **`98.3%` GPU busy** (`4,673.7 ms` across 2 profiled iterations = `2,336.8 ms/iter`; only `32.4 ms/iter` host/scheduling gap across 6,195 launches/iter). Below is the complete roofline accounting and explicit disposition of **every large flagged kernel class (`short`, `under-filled`, `tail`, `bw-poor`) and repeated chain** in `runs/r11_final2/profile/kernel_util.md`:

### 1. Irreducible Hardware Floor (`1,614.6 ms/iter` = `69.1%` of GPU time)
- **BF16 TensorCore GEMMs (`976.6 ms/iter`, `41.8%`)**: 100% vendor muBLAS assembly kernels (`ssgemm_persis_stage2/4`, `squad_level_epilogue`). Zero recomputation GEMMs remain (`SANA_GC_NUM_BLOCKS=gdn:0`, `SANA_RECOMPUTE_GDN_QKV=0`, `SANA_SAVE_GLUMB_U_LAYERS=20/20`). All linear projections and GDN BMMs are mathematically required by the model; FP8/INT8 is prohibited by `GOAL.md`.
- **FlashAttention-3 (`462.3 ms/iter`, `19.8%`)**: 100% vendor `mate` FA3 assembly (`flash_atten_bwd` `334.6 ms/iter` + `flash_atten_fwd` `125.3 ms/iter`). All 20 CrossAttn layers run mask-free via valid-token compaction (`N0`) and zero-copy BNHD views; the 5 bidirectional Softmax+UCPE self-attention layers (`L=22,000, D=112`) are fixed at hardware assembly speed.
- **MCCL FSDP2 Collectives (`175.7 ms/iter`, `7.5%`)**: `ReduceScatter_f32` (`104.7 ms/iter`) + `AllGather_bf16` (`62.8 ms/iter`) + `AllGather_LL` (`3.4 ms/iter`). All payloads are zero-copy 16-byte aligned; `MCCL_BUFFSIZE=16MiB` saturates the 4-card GPU interconnect (`32MiB + 32 channels` measured `2.3686` vs `2.3682 s/iter`, 0.0% delta), and `forward_prefetch` regressed due to SM/compute contention.

### 2. Bandwidth-Saturated Custom & Framework Kernels (`~550 ms/iter` = `23.5%` of GPU time)
All remaining large memory-bound custom kernels operate near empirical sustained HBM bandwidth:
- `bf16_col_sum_part` (`14.2 ms/iter`), `silu_mul_bwd_u4_bf16` (`8.4 ms/iter`), `gate_residual_fwd/bwd_u4_bf16` (`9.5 / 9.2 ms/iter`), `split_with_sizes_copy_out` (`9.0 ms/iter`), `recompute_ln_mod_u4_bf16` (`~7.0 ms/iter`)
- `chunk_cat<float, bf16>` (`18.8 ms/iter`), `bidi_short_conv4_qkv_k_fwd/bwd` (`14.4 / 12.7 ms/iter`), `main_gdn_prep_scale_fwd_u4` (`32.2 ms/iter`), `fused_multi_tensor_adamw_f32` (`23.3 ms/iter`, single-pass `float4` with fused L2 clip)
- `gdn_main/cam_phase_a_epilogue_bwd_u4` (`13.4 / 8.8 ms/iter`), `main_gdn_prep_bwd_576_bf16` (`18.4 ms/iter`), `ln_modulate_bwd_u4` / `glumb_bwd_p1_w40_320t` (`23.8 / 61.1 ms/iter`), `cam_prep_fwd_u4` / `glumb_bwd_p23` (`34.3 / 71.7 ms/iter`)

### 3. Disposition of Every Flagged Class & Repeated Chain in `runs/r11_final2/profile/kernel_util.md`
- **`short` (`median < 5 us`, 32 classes, `14.45 ms / 2 iters = 7.22 ms/iter`, `0.31%` total)**:
  - Largest entries are tiny-tensor casts (`CastOp bf16->f32` `3.93 ms/iter`, `1054x`; `CastOp f32->bf16` `1.38 ms/iter`, `684x`) and small scalar/frame-gate ops (`MulOp` `0.47 ms/iter`, `IdentityOp` `0.34 ms/iter`, `ExpOp` `0.13 ms/iter`). Even eliminating 100% of `short` kernels saves at most `7.2 ms/iter` (`0.3%`).
- **`under-filled` (`SM cov < 50%`, 107 classes, `379.65 ms / 2 iters = 189.83 ms/iter`, `8.12%` total)**:
  - **`90.0%` (`170.9 ms/iter`) is MCCL**: `ReduceScatter_f32` (`104.7 ms/iter`, `24 blocks`), `AllGather_bf16` (`62.8 ms/iter`, `24 blocks`), `AllGather_LL` (`3.4 ms/iter`). Limited by inter-GPU interconnect, not SM occupancy; empirically verified `MCCL_MIN/MAX_NCHANNELS=32` + `MCCL_BUFFSIZE=32MiB` yields zero speedup (`2.3686` vs `2.3682 s/iter`).
  - Remaining non-MCCL `under-filled` classes total only `18.9 ms/iter`, dominated by second-stage `O(C)` tail reductions (`reduce_chunks_1` `3.91 ms/iter`, `reduce_rows_to_f32` `3.47 ms/iter`, `bf16_col_sum_reduce` `1.45 ms/iter`, `9` blocks for `C=2240`) and tiny `CastOp`/`CopyGeneral` (`<1.4 ms/iter` each); not meaningfully reducible.
- **`tail` (`1 < waves < 1.5`, 17 classes, `8.83 ms / 2 iters = 4.42 ms/iter`, `0.19%` total)**:
  - `89%` (`3.93 ms/iter`) is the same tiny `CastOp bf16->f32` already counted in `short`; every other class is `< 0.25 ms/iter`.
- **`bw-poor` (7 classes, `577.68 ms / 2 iters = 288.84 ms/iter`, `12.4%` total)**:
  1. MCCL `ReduceScatter` (`104.7 ms/iter`) & MCCL `AllGather` (`62.8 ms/iter`): interconnect-bound, not HBM-bound (see above).
  2. `fused_silu_dwconv3x3_glu_fwd_w40_320t_bf16` (`47.3 ms/iter`): SFU-limited by `443.5M` `tanhf` evaluations/layer. Empirically benchmarked 6 alternatives on GPU 0 — `uint4 c64 w40 320t` (`3.22 ms`), `uint2 c64 w40 320t` (`2.33 ms`), `uint4 c128 w20 256t` (`5.96 ms`), `w40 256t` (`2.82 ms`), `ex2+rcp` (`2.38 ms`) — all slower than kept `w40_320t_tanh` (`2.28 ms` isolated / `2.37 ms` multi-card).
  3. `cam_prep_bwd_bf16` (`27.8 ms/iter`) & `softmax_cam_prep_bwd_bf16` (`8.7 ms/iter`): strided `(B*H, N, D)` (`4.93 MB` head stride) + 4×4 SE(3) `P_T`/`P_inv` transforms. Empirically benchmarked 6 zero-SMEM/vectorized alternatives on GPU 0 — `288t uint4` (`2.48 ms`), `576t uint2` (`2.59 ms`), `nodiv 576t uint2` (`2.39 ms`), `nodiv 320t uint4` (`2.22 ms`), 2-pass SMEM split (`2.65 ms`), `nc=180` (`2.69 ms`) — all slower than kept `nc=240` (`1.85 ms` after removing K-slot zeroing).
  4. `bidi_state_scan_bwd_d112_f32` (`20.7 ms/iter`) & `bidi_state_scan_fwd_d112_f32` (`16.9 ms/iter`): **FP32 compute-bound** (`56 FLOP/B` arithmetic intensity, `2.81 / 4.21 GFLOPs` FP32 state recurrence per call = sustained FP32 throughput; already 2D 4×4 register-tiled v3/v4a; `bwd` v4a `RT=16` regressed `40.9 -> 46.2 ms`).
- **Repeated chains (37 chains total, combined span `23.14 ms / 2 iters = 11.57 ms/iter`, `0.49%` of step time)**:
  - Top chains are tiny frame-gate/inflation elementwise ops (`#1`: `2.08 ms/iter`, `#2`: `1.67 ms/iter`, `#3`: `1.08 ms/iter`, `#4`: `1.04 ms/iter`, all `30x` on `(1,20,25)`–`(1,25,20)` tensors) and post-norm/reduction casts (`#5–#12`: `0.40–0.81 ms/iter` each). Total ceiling across all 37 chains combined is `< 11.6 ms/iter` (`< 0.5%`).

**Conclusion**: All remaining optimizable headroom across all flagged classes and chains combined is `< 15–20 ms/iter` (`< 0.8%`), well below the `54.8 ms/iter` gap to `5.000x` (`2.3144 s/iter`) and firmly at the MTT S5000 hardware roofline (`4.8843x PASS`, `98.3%` GPU busy, `69.1%` irreducible vendor GEMM + FlashAttention + MCCL collectives, remaining custom kernels near sustained HBM/SFU/FP32 ceilings).

---

## Current Summary
- **Switch file**: `configs/opt.json` (`{"environment": {"MCCL_BUFFSIZE": "16777216"}}`)
- **Latest PASS `check.json`**: `runs/r11_final2/check/check.json` (`PASS`, `N1`, **`2.3692 s/iter`**, **`4.8843x`**, `peak_alloc = 71.03 GiB`, `window_ratio = 1.0000`)
- **Gain breakdown (11 Rounds: `11.5719 -> 2.3692 s/iter`, `4.8843x`)**:
  - R1 (host syncs + UCPE/RoPE cache + batched-group AdamW): `11.572 -> 11.150 s/iter` (`-0.422 s/iter`, `1.038x`, `N0`)
  - R2 (fused MUSA C++ `GLUMBConvTemp` + linear 1×1 + direct temporal conv): `11.150 -> 9.565 s/iter` (`-1.585 s/iter`, `1.210x`, `N1`)
  - R3 (fused MUSA C++ bidirectional GDN+UCPE + FP32 real RoPE): `9.565 -> 5.350 s/iter` (`-4.215 s/iter`, `2.163x`, `N1`)
  - R4 (lean GDN activations + selective GC `gdn:9` preserving `CheckpointWrapper`): `5.350 -> 4.561 s/iter` (`-0.789 s/iter`, `2.537x`, `N1`)
  - R5 (16-byte-aligned FSDP2 `AllGather`/`ReduceScatter` fixing 10.7× MCCL slowdown): `4.561 -> 4.138 s/iter` (`-0.423 s/iter`, `2.797x`, `N1`)
  - R6 (MUSA C++ block fusion: RMSNorm, LN+mod, gate+res, silu*out, Softmax+UCPE `D=112` + `gdn:6`): `4.138 -> 3.399 s/iter` (`-0.739 s/iter`, `3.405x`, `N1`)
  - R7 (`glumbconv v3` + `block_fusion v2` + fused GDN Phase A/C & ShortConv4 absorption + `gdn:5`): `3.399 -> 2.933 s/iter` (`-0.466 s/iter`, `3.946x`, `N1`)
  - R8 (micro-recompute `u+qkv` + `gdn:0`, merged `glumbconv p23`, 2D-`t` block fusion, `bidi_scan v3`, fused AdamW, fast Gemma2, mask-free CrossAttn): `2.933 -> 2.686 s/iter` (`-0.247 s/iter`, `4.308x`, `N1`)
  - R9 (`glumbconv v5` `tanhf`+1-bar, zero-SMEM `uint4` 288t `block_fusion` + `AddLNMod` + `C>=2048` colsum, `uint4` GDN prep/epilogue, fused L2 clip+AdamW): `2.686 -> 2.560 s/iter` (`-0.126 s/iter`, `4.521x`, `N1`)
  - R10 (zero-copy CrossAttn/FA3, `main_prep_bwd 576t`, `split3/stack3 u4`, `triple_linear`, Plucker fusion, `qkv=0 + save_u=2`, `MCCL_BUFFSIZE=16MiB`): `2.560 -> 2.469 s/iter` (`-0.091 s/iter`, `4.688x`, `N1`)
  - R11 (full `GLUMBConv`+`AddSiluMulProj`+`QuadLinear` fusion, 20/20 `save_u` via `recompute_ln_mod_u4`, `glumb v6` `smemw+w40`, zero-copy FSDP2 pad16 `AllGather`, `GradAccBox` Plucker acc, `_compute_frame_gates` `fmean` fusion): `2.469 -> 2.369 s/iter` (`-0.099 s/iter`, **`4.884x`**, `N1`)
- **Additional dead ends (R9–R11)**:
  - Dropping `__restrict__` for in-place `du = u_c` in `glumbconv p23` regressed `+15.5 ms/iter`; reverted and upgraded with SMEM weights (`-28%`).
  - `glumbconv fwd` 128-bit `uint4`/`uint2`/`ex2` microbenchmarks (`2.33–5.96 ms`) all lost to `bf162 w40_320t_tanh` (`2.28 ms`) due to register pressure & SFU throughput.
  - `cam_prep_bwd` zero-SMEM `288t/320t/576t` and 2-pass SMEM-split variants (`2.22–2.69 ms`) all lost to `nc=240` (`1.85 ms`).
  - `bidi_state_scan_bwd` v4a (`RT=16`, 128t) regressed `40.9 -> 46.2 ms`; kept v3 for `bwd` and v4a for `fwd`.
  - `MCCL_BUFFSIZE=33554432` + `MCCL_MIN/MAX_NCHANNELS=32` measured `2.3686` vs `2.3682 s/iter` (interconnect saturated at `16 MiB`).
  - `save_u=8` (`75.89 GiB`) and `save_u=6` (`74.79 GiB`) without `y`/`combined`/`x_mlp_in`/`x_msa_in` VRAM swaps OOMed in Gate 2 `train_2step` step-1 FA3 `1.68 GiB` workspace; full VRAM-swapped R11 achieved `20/20` `save_u` at only `71.03 GiB`!
- **No-gain / dead ends**:
  - Single-pass `glumbconv` v2 (`90.6 KiB` SMEM, `123 regs`) regressed to `17.56 ms` due to shared-memory/register occupancy cliff; split v3 (`<= 27 KiB`, `<= 66 regs`) reached `11.79 ms` (split `bwd_p2`), and merged `p23` reached `4.01 ms` vs `5.79 ms`.
  - FSDP2 AllGather/GEMM concurrency overlap (`forward_prefetch`) was slower on 4× S5000 (`19.49 vs 18.93 ms` serialized) due to SM/compute contention.
  - Async pinned CPU activation offload (`async_offload.py`, 15 layers) added `~27 ms/block` real multi-card overhead (`55 MiB/ms`) vs `169 MiB/ms` for pure-GEMM micro-recompute; disabled by default (`SANA_OPT_ASYNC_OFFLOAD=0`).
  - Scalar `pack_unpack_pad16_u16` with 64-bit div regressed `+51.1 ms/iter`; reverted to `.copy_`.
  - `MUSAGraph` capture on `Gemma2Model` failed under `torch_musa 2.9.1`; replaced by zero-extra-VRAM MUSA C++ `fast_gemma2.py` (`62 -> 10.7 ms`).


