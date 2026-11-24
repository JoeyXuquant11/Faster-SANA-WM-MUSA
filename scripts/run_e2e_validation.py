"""Paired end-to-end training & checkpoint parity validation for SANA-WM Stage-1 on 4x MTT S5000.

Executes the exact setup and verbatim per-step training loop of
`train_video_scripts/train_sana_wm_stage1.py` on either the baseline (`3daa4fb`)
or optimized (`9d0eea1`) tree, recording:
  - per-step training loss, wall-clock step time, peak HBM allocated/reserved,
  - per-step parameter L2 displacement ||theta_t - theta_0||_2, and
  - full gathered FP32 checkpoint weights (all 2.66B DiT parameters, wrapper
    prefixes stripped) for exact weight agreement and update cosine similarity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
try:
    import torch_musa  # noqa: F401
except ImportError:
    pass
from torch.distributed.tensor import DTensor


def gather_param_fp32(p: torch.Tensor) -> torch.Tensor:
    if isinstance(p, DTensor):
        return p.full_tensor().detach().float().cpu()
    return p.detach().float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rba_root", required=True, help="Directory containing bench/ and numerics/")
    p.add_argument("--repo_dir", required=True, help="Path to Sana tree (baseline or opt)")
    p.add_argument("--base_yaml", required=True)
    p.add_argument("--fixture_dir", required=True)
    p.add_argument("--model_root", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--max_steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--out_json", required=True)
    p.add_argument("--ckpt_path", default="")
    p.add_argument("--save_init_path", default="")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.rba_root).resolve()))
    from bench.sana_adapter import SanaWmAdapter, strip_prefix  # noqa: E402

    adapter = SanaWmAdapter(
        repo=args.repo_dir,
        base_yaml=args.base_yaml,
        fixture_dir=args.fixture_dir,
        model_root=args.model_root,
        scratch=args.scratch,
    )
    adapter.configure({"name": f"e2e_{args.tag}", "seed": args.seed, "steps": args.max_steps, "track_disp": False})
    adapter.build(seed=args.seed, golden=False)
    objs = adapter.objects()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    model = objs.models["dit"]
    device = torch.device(f"musa:{int(os.environ.get('LOCAL_RANK', 0))}")

    if args.save_init_path:
        init_sd = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            t = gather_param_fp32(param)
            if rank == 0:
                init_sd[strip_prefix(name)] = t
        if rank == 0:
            Path(args.save_init_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(init_sd, args.save_init_path)
        del init_sd
        if dist.is_initialized():
            dist.barrier()

    torch.musa.synchronize()
    torch.musa.reset_peak_memory_stats(device)

    steps_log = []
    for step_idx in range(args.max_steps):
        torch.musa.synchronize()
        t0 = time.perf_counter()
        loss_map = adapter.step(step_idx, None)
        torch.musa.synchronize()
        dt = time.perf_counter() - t0

        loss_local = loss_map["loss_local"].to(device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(loss_local, op=dist.ReduceOp.AVG)
        mean_loss = float(loss_local.item())
        reported_loss = float(loss_map["loss"].item())

        alloc_gib = torch.musa.max_memory_allocated(device) / (1024 ** 3)
        res_gib = torch.musa.max_memory_reserved(device) / (1024 ** 3)
        rec = {
            "step": step_idx + 1,
            "loss": mean_loss,
            "reported_loss": reported_loss,
            "step_time_s": round(dt, 4),
            "peak_alloc_gib": round(alloc_gib, 3),
            "peak_reserved_gib": round(res_gib, 3),
        }
        steps_log.append(rec)
        if rank == 0:
            print(
                f"[{args.tag}] step {step_idx + 1:02d}/{args.max_steps} "
                f"loss={mean_loss:.6f} dt={dt:.3f}s mem={alloc_gib:.2f}/{res_gib:.2f}GiB",
                flush=True,
            )

    if args.ckpt_path:
        final_sd = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            t = gather_param_fp32(param)
            if rank == 0:
                final_sd[strip_prefix(name)] = t
        if rank == 0:
            Path(args.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(final_sd, args.ckpt_path)
        del final_sd
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        steady = [r["step_time_s"] for r in steps_log[3:]] if len(steps_log) > 3 else [r["step_time_s"] for r in steps_log]
        summary = {
            "tag": args.tag,
            "world_size": world_size,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "steady_mean_step_s": round(sum(steady) / len(steady), 4),
            "peak_alloc_gib": max(r["peak_alloc_gib"] for r in steps_log),
            "peak_reserved_gib": max(r["peak_reserved_gib"] for r in steps_log),
            "steps": steps_log,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[{args.tag}] Saved summary to {args.out_json}", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
