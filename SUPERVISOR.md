# SANA-WM Stage-1 Bidirectional 720p (4× MTT S5000): supervision record

*Archived note: internal GPU microarchitecture identifiers, hardware peak/bandwidth parameters, cluster hostnames, and proprietary vendor kernel symbol names have been redacted; timing, speedup, memory, and numerics records are unchanged.*

One autonomous optimization session (**Gemini 4 Neon (XHigh)**, `10-hour` active wall-clock budget, `4 × Moore Threads MTT S5000`) optimized Stage-1 bidirectional 720p FSDP2 training of **SANA-WM** (`SanaMSVideoCamCtrl_1600M_P1_D20`, `2.66B` params, 15 bidirectional GDN+UCPE blocks + 5 Softmax+UCPE blocks, `GLUMBConvTemp` FFN, 193 frames / `T=25` latent at `704×1280`, `N=22,000` tokens, `batch_size=1`/GPU, DP=4 FSDP2 BF16 + `fp32_attention` + FP32 master/AdamW + frozen online `Gemma-2-2B-IT`); one supervisor formulated the frozen judge, enforced strict sandbox isolation, and handled infrastructure cleanup/migrations without supplying optimization hints. Target `5.0×` (`<= 2.314 s/iter`), minimum wrap-up `2.0×` (`<= 5.786 s/iter`) with full `kernel_util.md` plateau analysis. Workspace layout: candidate `Sana/` (branch `opt`), reference `targets/reference` (commit `3daa4fb`).

## Results

Final outer-harness certification (`bench/bench.py check --final`, candidate `9d0eea1`, switch file `configs/opt.json`, `11` accepted rounds within the 10-hour active budget; full record in [`task/final_check.json`](task/final_check.json)):

| Item | Value |
|---|---|
| status | **PASS** (`final=true`, `accepted=true`, plateau proved in [`WORK_LOG.md`](WORK_LOG.md)) |
| performance.speedup | **`4.8543×`** (`11.5023 → 2.3695 s/iteration`; `ref_a 11.4624 s`, `cand 2.3695 s`, `ref_b 11.5422 s`; reference spread `0.69%`, candidate window spread `0.02%`; **`4.8843×`** vs frozen reference `11.5719 s/iter`) |
| Effective MFU (`@500T` BF16) | **`7.79% → 37.82%`** (`38.95 → 189.10 TFLOPS/GPU`, `448.07 TFLOPs/GPU/step` effective FLOPs: `3×` DiT fwd `148.54T` + `1×` frozen Gemma-2-2B-IT fwd `2.45T`) |
| Peak HBM | **`71.03 GiB`** allocated / **`72.36 GiB`** reserved per GPU (`34.68 / 35.77 GiB` reference) |
| Gate 0 | AST match of `if __name__ == "__main__":` and `main(cfg)` outside the per-step loop + bitwise-identical `SanaWMZipLatentDataset` items and batches (`task/loader_digest.json`) |
| Gate 1 | `cand` `2.3695 s/iter` (`window_s = [11.848, 11.846]`, `window_ratio = 1.0002`), `fingerprint_diffs = {cand: [], ref_b: []}`, `problems = []` |
| Gate 2 | Tier **`N1`**; `7,278` of `33,176` tensors bitwise identical, **`0` exact-item failures**, `0` errors, `0` dtype/precision changes (`bitwise_trusted = true`) |
| Gate 3 | **`equivalent`**, paired deterministic 6-step trajectory on seed `3407`: `train_loss = -0.000651` (`-0.0177σ` vs `±3σ` bound `0.1101`), `param_disp_l2 = +0.013925` (`+0.4127σ` vs `±3σ` bound `0.1012`), `step_disp_top1pct = -2.324e-8` (`-0.4446σ` vs `±3σ` bound `1.568e-7`) |
| Integrity | `evaluator_sha256` (`20ce07df...`) and `judge.sha256` (`69/69` files) verified intact; `31/31` frozen inputs match at start and end; source digest `e0c5d52b...` |
| Cost | `1.129 GPU-hours` (`5` GPU runs) for `check --final` |

**Accepted at `4.8543×` certified (`4.8843×` vs frozen ref, `97.7%` of `5.0×` target) with full hardware-roofline plateau proof and all custom MUSA C++ kernels written from scratch.** Key progression on branch `opt` (detailed in [`TRACE.md`](TRACE.md) and [`WORK_LOG.md`](WORK_LOG.md)): host sync removal & UCPE/RoPE cache (`1.04×`, `N0`), MUSA C++ fused `GLUMBConvTemp` (`1.21×`), fused bidirectional GDN+UCPE & FP32 real RoPE (`2.16×`), lean GDN activations + selective GC `gdn:9` (`2.54×`), 16-byte-aligned FSDP2 `AllGather`/`ReduceScatter` fixing a `10.7×` MCCL slowdown (`2.80×`), fused block ops + `gdn:6` (`3.40×`), occupancy-tuned `glumbconv v3` + 9 GDN Phase A/C kernels + `gdn:5` (`3.95×`), micro-recompute `gdn:0` + `bidi_scan v3` + 1-pass AdamW + fast Gemma2 + mask-free CrossAttn (`4.31×`), `uint4` vectorized block/GDN kernels + fused L2 grad clip (`4.52×`), zero-copy CrossAttn/FA3 + `split3/stack3` + `MCCL_BUFFSIZE=16MiB` (`4.69×`), and VRAM-swapped `20/20` `save_u` zero-recompute + full `GLUMBConv`/`AddSiluMulProj`/`QuadLinear` fusion (`4.88×` / `4.85×` certified).

## Formulation

- **Workload**: Upstream `NVlabs/Sana` Stage-1 bidirectional training (`SanaMSVideoCamCtrl_1600M_P1_D20`, 20 blocks: 15 bidirectional GDN+UCPE + 5 Softmax+UCPE at `softmax_every_n=4`, `GLUMBConvTemp` FFN, `chunk_plucker_post_attn`, real `sana_wm_1600m_720p.safetensors` weights) + frozen online `gemma-2-2b-it`, trained on a frozen 64-sample Sekai-Game latent fixture (`193` frames / `T=25` latent at `704×1280`, `N=22,000` tokens, `batch_size=1`/GPU, DP=4 FSDP2, `BF16` + `fp32_attention=true` + FP32 master weights & AdamW). Details in [`GOAL.md`](GOAL.md) and [`task/baseline.md`](task/baseline.md).
- **Hardware & Baseline**: `4 × MTT S5000` (`80 GiB` HBM/GPU). Frozen reference timing across two independent cold-cache runs (`warmup=3`, `window=5`, `repeats=2`): **`11.5719 s/iter`** (`115.725 s` / `115.713 s`, spread `0.01%`), peak `34.68 GiB` alloc, per-rank per-iteration RNG identical across runs.
- **Deterministic mode & MUSA adaptation**: MUSA's stock SDPA backward is nondeterministic and vendor varlen attention hit a heap corruption under activation checkpointing when forced deterministic. In deterministic judge mode (`torch.are_deterministic_algorithms_enabled()`), [`bench/sana_adapter.py`](bench/sana_adapter.py) routes stock `F.scaled_dot_product_attention` and vendor varlen attention to chunked `SDPBackend.MATH` at the same precision for both reference and candidate, yielding `100%` bitwise repeatability (`33,176 / 33,176` tensors across 2-step deterministic runs); custom candidate kernels are not bypassed and must provide a deterministic path at the same precision.
- **Judge validation (all 5 acceptance checks passed prior to launch)**:
  - `numerics selftest`: `{"ok": true, "failed": []}`
  - `(a)` Unmodified candidate: `check --quick` → `PASS (bitwise=True, N0)`; full `check` → `UNAVAILABLE (no_benefit, 1.0012×)`
  - `(b)` Setup AST edit outside per-step loop: `ERROR (setup_guard)`
  - `(c)` Semantic break (skipped backward): `INCORRECT (structure, 8 exact failures)`
  - `(d)` Mathematically equivalent reordering: `PASS (equivalent, N1)` (`train_loss 0.000σ`, `param_disp_l2 -0.025σ`, `step_disp_top1pct -0.0075σ`)
  - `(e)` Control arm (`lr * 1.02`): `INCORRECT (different)` (`step_disp_top1pct +3.519σ > 3.0σ` on seed `3407`)

## Strict Isolation & Sandbox Setup

- The optimizing agent ran inside a dedicated 4-GPU container with `targets/reference/`, `bench/`, `numerics/`, `task/`, `GOAL.md`, `GOAL_LINE.txt`, `$ASSETS/`, model weights, and user site-packages mounted read-only (`:ro`), and `task/final-evaluation` absent until the outer harness executed `check --final`.
- No pre-existing custom MUSA C++ GDN or `GLUMBConvTemp` kernels were accessible from inside the container; all custom kernels under `Sana/diffusion/model/musa_ops/` were authored from scratch during the session.
- Every sandbox tool call (`1,332` MCP calls) was logged by an outer audit wrapper outside the container.

## Interventions (infrastructure only; zero optimization direction was ever given)

1. **Hung single-GPU unit-test cleanup (Round 9, elapsed ~7.2 h; Round 10, elapsed ~7.9 h)**: During isolated single-GPU kernel debugging, two custom MUSA test binaries deadlocked on GPU 0 without a per-command timeout (`~14 min` and `~4.5 min`), blocking the synchronous MCP wrapper. The supervisor terminated only those hung child test PIDs inside the container; the agent immediately resumed in the same conversation and added `timeout 60` guards to subsequent unit tests.
2. **Cluster GPU contention migrations with stalled-time compensation (elapsed ~8.6 h and ~9.7 h)**: Mid-session, other cluster workloads launched jobs overlapping our 4 GPUs on the initial host, and later a 32-GPU multi-node job occupied the second host. Per strict cluster ownership rules, external jobs were never touched; the supervisor cleanly relocated the identical isolated container to an idle 4-GPU MTT S5000 host sharing the same workspace storage, compensated the lost stall/migration time (`~44 min` total) so the agent received a full `10.0 h` active budget, and resumed the same conversation with an infrastructure-only notice.
3. **Automatic outer `check --final` at budget expiry**: After the agent committed Round 11 (`9d0eea1`), updated `FINAL.json`, and wrote its final plateau analysis in `WORK_LOG.md`, the outer harness automatically verified `judge.sha256` (`69/69` intact) and ran `bench/bench.py check --final` (`paired_ref_a_ref_b` + Gate 2 + Gate 3).

## What the Judge Actually Caught

- **Gate 2 caught a structural hook break in `R11a`**: `R11a` timed **`4.8924×`** (`2.3653 s/iter`) in Gate 1, but refactoring the Plucker projection accumulation path broke the expected module/hook tensor structure (`verdict=structure`, `8` exact failures). Gate 2 rejected it immediately; replacing the accumulator with a hook-preserving `GradAccBox` fixed all 8 checks in `90 s` and passed `R11` cleanly (`9d0eea1`, `4.8843×` vs frozen ref / `4.8543×` certified `check --final`).
- **Gate 2 memory ceiling enforcement (`R4a`, `R6a`, `R10a/b`)**: Aggressive activation-saving configurations (`gdn:5` in R6, `save_u=6/8` in R10 at `74.8`–`75.9 GiB`) timed cleanly in Gate 1 but OOMed under Gate 2's deterministic 2-step check, pushing the agent to develop true VRAM-swapping fusions (`y_recomp` in `glumb_bwd_p1` and `silu_mul_bwd_u4` + `recompute_ln_mod_u4`) that achieved full `20/20` `save_u` at only `71.03 GiB`.
- **Gate 1 rejected regressive/no-gain changes**: Async pinned-CPU activation offload, FSDP2 `forward_prefetch`, MCCL `32MiB/32-channel` sweeps, high-SMEM `glumbconv v2`, and 128-bit `glumbconv fwd` / zero-SMEM `cam_prep_bwd` variants were empirically screened out.

## Known Limits of the Judge

- Gate 3 runs 1 paired deterministic seed (`seed 3407`, 6 steps) against a `±3σ` bound calibrated from 6 reference seeds (`3407..3412`), following the current RBA judge rules; very small long-horizon convergence drift below `3σ` over hundreds of steps is not tested within 6 steps.
- During archival, machine paths in `bench/bench.json`, `task/formulation.json`, and `task/final_check.json` were rewritten to `$ASSETS`/`$WORK`, GPU UUIDs were anonymized to `GPU-0`..`GPU-3`, and internal hardware peak/bandwidth defaults (`peak_bw_gbs`) and vendor symbol regexes were redacted in `bench/bench.json` and `bench/bench.py`; see [`../README.md`](../README.md#archived-records) regarding `evaluator_sha256` after archival sanitization.

## Lessons

- **Inspect collective alignment on non-CUDA fabrics early**: A 16-byte alignment mismatch on BF16 FSDP2 shards caused a `10.7×` MCCL `AllGather` slowdown (`485 ms/iter`); fixing alignment gave an immediate bitwise-identical `+10%` end-to-end speedup (`2.54× → 2.80×`).
- **Cheap recomputation beats both full-block GC and host offload when HBM is tight**: Recomputing lightweight activations (`y_recomp`, `ln_mod_u4`) inside existing backward kernels freed `~10 GiB` HBM for almost zero time cost, unlocking `20/20` zero whole-block GC and full `save_u` at `71.03 GiB`.
- **Bypassing broken stock autograd paths pays double**: Writing the fused bidirectional GDN+UCPE scan in Round 3 not only sped up GDN by `4.29×` and freed `7.5 GiB`/layer, it also cleanly eliminated an upstream MUSA `flip_and_shift` autograd gradient bug.

## Handoff

- **Certified results**: [`task/final_check.json`](task/final_check.json) (verbatim `check --final` record, paths sanitized to `$WORK` and GPU UUIDs to `GPU-0`..`GPU-3`), [`TRACE.md`](TRACE.md), [`trace.json`](trace.json), [`WORK_LOG.md`](WORK_LOG.md), and [`configs/opt.json`](configs/opt.json).
- **Re-running the judge**: Set `ASSETS` to a directory containing `recipe_stage1.yaml`, `fixture/`, `models/sana_wm/`, and `pyuser/`, re-freeze `bench/records/ref`, `bench/thresholds.json`, and `bench/traj/ref` on the target 4× MTT S5000 host, and invoke `python3 bench/bench.py check --tree <candidate_tree> --config configs/opt.json --out runs/<run_name>`.
