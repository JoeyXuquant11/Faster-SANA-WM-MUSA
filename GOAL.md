# Goal

*Archived note: internal hardware microarchitecture identifiers, measured hardware peak/bandwidth specifications, and cluster-specific paths were redacted after the run; task rules and timing/numerics gates are unchanged.*

Bring the **time per training iteration** of SANA-WM Stage-1 bidirectional 720p FSDP2 training (`SanaMSVideoCamCtrl_1600M_P1_D20`, `train_video_scripts/train_sana_wm_stage1.py`) on 4x MTT S5000 **down to 1/5 of the reference (a 5.0x speedup, <= 2.314 s/iter)**, with a minimum of **2.0x (<= 5.786 s/iter)**, while passing all correctness gates below.

- **Time budget**: **600 minutes (10 hours)**. The outer harness stops the session at the deadline without warning. Keep the candidate tree (`Sana/` on branch `opt`), your switch file, `WORK_LOG.md` summary, and `FINAL.json` at the best verified version at all times.
- **Training code**: `Sana/`, upstream `NVlabs/Sana` MUSA baseline at commit `3daa4fbef442e4ee2b7cb6cdb1f26b295312915d`, branch `opt`; this is where you make changes.
  - Entry point: `train_video_scripts/train_sana_wm_stage1.py` launched with `torchrun --nnodes=1 --nproc-per-node=4 --master-addr 127.0.0.1 --master-port <port>`.
- **Workload (frozen recipe in `$ASSETS/recipe_stage1.yaml` and `bench/recipe.py`)**:
  - Upstream bidirectional Stage-1 config (`configs/sana_wm/stage1/sana_wm_stage1_sekai_bidirectional_cp2_fsdp2.yaml`) adapted to 4x MTT S5000 with DP=4 FSDP2 (`use_fsdp=true, fsdp_version=2, cp_size=1`), 193 frames (`T=25` latent frames at 704x1280 -> `22x40` spatial latent = 22,000 tokens), `batch_size=1`/GPU (`gradient_accumulation_steps=1`), real published Stage-1 DiT weights (`sana_wm_1600m_720p.safetensors`, 2.66B params) and real frozen `gemma-2-2b-it` text encoder running online per step, `grad_checkpointing=true`, `BF16` mixed precision + `fp32_attention=true`, FP32 master weights + FP32 AdamW (`lr=1e-5, betas=[0.9, 0.999], eps=1e-10, weight_decay=0.0, gradient_clip=0.5`, constant schedule).
  - Frozen offline 64-sample Sekai-Game fixture (`$ASSETS/fixture`, `SanaWMZipLatentDataset`).
- **Reference**: `targets/reference`, the same commit, unchanged.
  - Frozen reference timing (`task/timing_ref.json`): **`11.5719 s/iter`** (`115.719 s` over 2 windows of 5 iterations after 3 warmup iterations), run-to-run spread **`0.01%`**, peak allocated **`34.68 GiB`** (reserved `35.77 GiB`), per-rank per-iteration RNG fingerprints identical across runs.
  - Baseline profile and kernel utilization breakdown: `task/baseline.md`.

## What counts as reaching the goal (the single criterion)

```bash
python3 bench/bench.py check --tree Sana --config <switch file> --out <output dir>
```

(Omit `--config` when no environment switch file is used.) It writes `<output dir>/check.json`.

- **Done**: `status == "PASS"` and `performance.speedup >= 5.0`.
- **Wrap-up fallback**: If profile and kernel-utilization evidence show that `5.0x` cannot be reached within the rules, `status == "PASS"` with `performance.speedup >= 2.0` is acceptable for wrapping up **only if** every large `short`, `under-filled`, or `bw-poor` kernel class and every large repeated kernel chain in your latest `kernel_util.md` has had an attempt (a written, checked, and timed kernel or fusion) or a concrete logged reason in `WORK_LOG.md` why it cannot be changed.

`check` applies the gates below in order and stops at the first failure:

0. **Structure (CPU, ~15 s)**:
   - The `if __name__ == "__main__":` block and `main(cfg)` outside the per-step `for batch in train_dataloader:` loop body in `train_video_scripts/train_sana_wm_stage1.py` must match the reference AST. Different -> `ERROR`.
     - Reason: `bench/sana_adapter.py` executes `main(cfg)`'s setup and per-step loop body verbatim. Put changes in helper functions (`_build_fsdp2_plugin`, `_encode_prompts`, `_extract_batch`, `build_optimizer`, `diffusion/model/*`, custom kernels) or inside the `for batch in train_dataloader:` body.
   - `SanaWMZipLatentDataset` must yield bitwise-identical items and first batches to `task/loader_digest.json` on the frozen fixture -> else `INCORRECT`.
1. **Benefit (~3 min in `check`; ~9 min in outer `check --final`)**:
   - Fresh `torchrun --nproc-per-node=4` run of `train_video_scripts/train_sana_wm_stage1.py` with a fresh cold compile cache (`warmup=3`, `window=5`, `repeats=2`, 10 timed iterations).
   - Your `check` runs 1 candidate timing run against the frozen reference in `task/timing_ref.json` (`speedup = frozen_total_s / candidate_total_s`; window spread/drift are reported as notes).
   - The outer harness's `check --final` (gated by `task/final-evaluation`, not callable inside the sandbox) runs `ref_a -> cand -> ref_b` with strict fingerprint matching and interference checks.
   - Per-iteration, per-rank RNG fingerprints must match the frozen reference (`task/timing_ref.json`), otherwise `ERROR`.
   - `speedup <= max(1.02, 1 + 2 * reference_spread)` -> `UNAVAILABLE (no_benefit)`, rejected without evaluating precision.
2. **Bitwise (~3.5 min)**:
   - Deterministic 2-step case (`bench/cases.json`, seed `3407`, covering AdamW's second-step moments) recorded with `numerics record --deterministic --sketch-dim 4096`.
   - Every exact check must hold, otherwise `INCORRECT`: phase, RNG fingerprints before and after each step, forward/optimizer call counts, invariants (`global_step`, `epoch`, batch digest, learning rate, batch/grad-clip/FSDP/CP/precision/frame settings), tensor names and shapes, finiteness, and cross-rank sync pattern (`dit: local` under FSDP2). Missing tensors count as an exact failure (`INCORRECT`).
   - All recorded tensors (`loss/`, `out/`, `grad/`, `delta/`, `optim/`, `buffer/`) bitwise identical with unchanged dtypes and math modes -> **`PASS (N0)`** directly.
   - Reference bitwise repeatability (`16,588/16,588` tensors identical across two independent deterministic runs) was verified during formulation.
3. **Statistical (~3 min, only when Gate 2 is not bitwise identical)**:
   - Paired 6-step deterministic trajectory on the first frozen reference seed (`3407`), compared against the 6-seed frozen reference distribution (`bench/traj/frozen.json`, seeds `3407..3412`) with `numerics traj-compare --sigma-k 3`.
   - Endpoints: `train_loss` (mean over steps `[3, 6)`), `param_disp_l2` (global L2 displacement `||theta_t - theta_0||_2` over `[4, 6)`), and `step_disp_top1pct` (mean single-step `|theta_t - theta_{t-1}|` over top 1% moved elements over `[4, 6)`).
   - Every endpoint `|candidate - reference| <= 3 * sigma_seed` -> **`PASS (N1)`** (or `N2` if dtypes/math modes changed). Any endpoint `> 3 * sigma_seed` -> **`INCORRECT`**.
   - **Tier**: `N0` = bitwise identical with same dtypes/math modes; `N2` = dtype or math mode changed; `N1` otherwise. Above `N2` -> `INCORRECT`.
   - Per-tensor tolerances (`bench/thresholds.json`) are diagnostic output only and do not enter the verdict.

Other `check` modes (use `time` and `--quick` to screen day-to-day; run full `check` once confident):
- `bench.py time --tree Sana [--config <file>] --out <dir>`: ~3 min, reports `s_per_iteration` and `speedup_vs_frozen_reference`.
- `bench.py check --quick`: Gates 0 and 2 only (~4 min).
- `bench.py check --n1`: Gates 0 and 3 only (~4.5 min), to re-check statistical equivalence after a non-bitwise Gate 2.
- `bench.py check --skip-timing`: Gates 0, 2, 3 (~7.5 min).

## What every round must do

1. **Profile first, then change.** Before each round's edit, profile the current best version and select one concrete hotspot from the profile:
   ```bash
   python3 bench/bench.py profile --tree Sana [--config <file>] --out runs/<round>/profile --iters 2
   ```
   - Records 2 iterations after 3 warmup iterations under `torch.profiler` (`CPU + PrivateUse1`) and writes `profile.json`, `profile.md`, `timing_profile.trace.json`, and the kernel utilization report `kernel_util.md` / `kernel_util.json`.
   - Maintain a **bytes model** (`bytes` and optionally `flops` per call) for the top kernel classes (`kernel_util.py --bytes-model <file>`) so you can tell which kernels are bandwidth-bound vs launch-bound vs under-filled vs compute-bound.
   - Log the top kernel classes (with SM coverage, flags, and achieved BW % when modeled) and top repeated chains from `kernel_util.md` in `WORK_LOG.md`.
   - You may run small targeted localization scripts under the GPU lock (`flock /tmp/sana_wm_gpu.lock <cmd>`), only for diagnostics and unit tests, never for reporting end-to-end speed.
2. **Time only with `bench/bench.py`.** Never write your own end-to-end timing script or report speedup from anything other than `bench/bench.py time` / `check`.
3. **Log every round in `WORK_LOG.md`** (including failed and no-gain rounds): start/end timestamps, profile & kernel-utilization findings, exact change made, `time` result, and `check` result (if run).

## Pace: at most 30 minutes per round (mandatory)

- One round = (1) profile -> (2) change one focused hotspot -> (3) re-time with `bench.py time` (plus `check --quick` or `check --n1` when needed) -> (4) log in `WORK_LOG.md` and commit kept changes on branch `opt`.
- Typical command times: `profile` ~2.5 min, `time` ~3 min, `check --quick` ~4 min, `check --n1` ~4.5 min, full `check` ~7-11 min (full `check` does not count toward the 30-minute round cap).
- Keep each round <= 30 minutes wall clock from taking the GPU lock. Do not pile unrelated hotspots into one round, and do not spend long stretches reading without measuring.

## Constraints

- **Training semantics must not change**: same dataset & order, batch size (`1`/GPU, `DP=4`), resolution (`704x1280`, 193 frames / `T=25` latent), model architecture & weights, loss formulation, optimizer (`AdamW`, `lr=1e-5`, `betas=[0.9,0.999]`, `eps=1e-10`, `weight_decay=0.0`, `gradient_clip=0.5`), 1 optimizer step per iteration, and identical per-iteration random-number stream (verified in Gates 1 and 2).
- **Precision floor**:
  - Baseline precision is `BF16` mixed precision + `fp32_attention: true` + FP32 master weights and FP32 AdamW optimizer state.
  - Lowering training to FP8 or INT8 is forbidden. Removing FP32 master weights or FP32 AdamW optimizer state is forbidden.
  - Local precision changes are allowed only within tier `<= N2` and must pass Gate 3.
- **Deterministic mode and attention routing**:
  - In Gates 2 and 3, `bench/sana_adapter.py` enables strict deterministic mode (`torch.use_deterministic_algorithms(True, warn_only=False)`) and routes MUSA `F.scaled_dot_product_attention` and `mate.flash_attn_varlen_func` to chunked `SDPBackend.MATH` at the same precision for both reference and candidate (because MUSA's default SDPA backward is non-deterministic).
  - Any timed attention or custom kernel you add must compute at the same precision as the judged deterministic path. If a custom kernel you add is non-deterministic, it must dispatch to a deterministic implementation of the same precision when `torch.are_deterministic_algorithms_enabled()` is true.
  - Apart from `torch.are_deterministic_algorithms_enabled()`, candidate code must NEVER inspect `SANA_BENCH_*` or behave differently depending on whether it is being timed or judged.
- **Keep bench hook points**:
  - `main(cfg)` outside the `for batch in train_dataloader:` body and `if __name__ == "__main__":` must remain AST-identical to `targets/reference`.
  - `global_step` must increment by 1 per iteration inside the loop body, `logs[config.loss_report_name]` and `loss` must be populated per iteration, and `optimizer` must remain a `torch.optim.Optimizer` firing one optimizer post-step hook per iteration.
- **Switch file format**: JSON `{"environment": {"VAR": "value"}}`, stored under `configs/` (or workspace root). Variables in `bench/bench.json` `reserved_env` are refused.
- **Read-only paths & offline sandbox**: `bench/`, `numerics/`, `task/`, `targets/`, `GOAL.md`, `GOAL_LINE.txt`, `$ASSETS`, `$ASSETS/models/sana_wm`, and `$ASSETS/pyuser` are read-only. No internet downloads or `pip install`.

## Environment

- **Hardware**: 4x Moore Threads MTT S5000 (80 GiB HBM/GPU), visible as devices `0..3`. GPU lock: `/tmp/sana_wm_gpu.lock` (`bench.py` acquires it automatically; wrap custom GPU scripts in `flock /tmp/sana_wm_gpu.lock <cmd>`).
- **Software stack**: Python 3.10.12 (`/usr/bin/python3`), `torch 2.9.1.post1+musa5.2.0`, `torch_musa 2.9.1.post1+musa5.2.0`, `mate 0.2.5`, `tilelang_musa 0.1.8+musa.3`, `mcc` (MUSA C++ compiler), `diffusers 0.39.0`, `transformers 4.57.3`, `accelerate 1.15.0`.
- **Important MUSA platform rules**:
  - Do NOT use Triton on MTT S5000; for custom kernels, use TileLang-MUSA or MUSA C++ (`mcc`).
  - Keep `MKL_NUM_THREADS=1`, `TORCH_COMPILE_DISABLE=1`, `GDN_DISABLE_COMPILE=1`, `PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True`.
  - Keep JIT/compile caches on container-local `/tmp` (never on shared storage).
  - Use `MUSA_LAUNCH_BLOCKING=1` when debugging or unit-testing new kernels.
- **Frozen reference artifacts** (read-only): `task/timing_ref.json`, `task/baseline.md`, `task/formulation.json`, `task/loader_digest.json`, `bench/records/ref/`, `bench/traj/`.

## References (skills)

Read the relevant skill cards under `/workspace/home/.codex/skills/` (also mounted at `/root/.codex/skills/`):
- `moore-infra` (`SKILL.md`, `references/kernels.md`, `references/infra.md`, `references/tools-and-debug.md`)
- `kernel-utilization-analysis`
- `train-opt-tiered-profiling`, `train-opt-measurement`, `train-opt-action-ladder`, `train-opt-numerics-validation`, `train-opt-fusion-patterns`, `train-opt-fsdp-multi-forward`
- `sglang-diffusion-benchmark-profile`, `sglang-diffusion-performance`

These are priors summarized from other optimization tasks; re-verify them on this task before treating them as conclusions.

## Wrap-up & continuous state

- Commit every kept change separately on branch `opt` of `Sana/`, with a commit message stating what changed, the numerics tier, and the `bench.py time` / `check` result.
- Keep `FINAL.json` in the workspace root updated at all times as:
  ```json
  {"tree": "Sana", "config": "<switch file path or null>", "check": "<path to latest PASS check.json>"}
  ```
- Keep a current summary at the end of `WORK_LOG.md` after every round: current switch file, latest `check.json` path and verdict/speedup, gain breakdown by change, and directions tried that did not work and why.
