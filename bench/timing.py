"""Iteration timing and torch.profiler wrapper for SANA-WM Stage-1 (train_video_scripts/train_sana_wm_stage1.py).

Run under `torchrun` with the source tree as working directory:

  torchrun --nnodes=1 --nproc-per-node=4 <bench>/timing.py --out <prefix> [options] -- <train_sana_wm_stage1.py args>

One iteration is one optimizer.step() (gradient_accumulation_steps=1). Window edges are bracketed by
torch.musa.synchronize(); nothing else synchronizes during timing windows. At every iteration end the hook
fingerprints CPU, MUSA and Python random RNG state (host-side state, no device sync).
"""
import argparse
import gc
import hashlib
import json
import os
import random
import runpy
import signal
import sys
import time
from pathlib import Path

import torch
try:
    import torch_musa  # noqa: F401
except ImportError:
    pass
from torch.optim.optimizer import register_optimizer_step_post_hook

T_START = time.perf_counter()


def _dev_sync():
    if hasattr(torch, "musa") and torch.musa.is_available() and torch.musa.is_initialized():
        torch.musa.synchronize()
    elif torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _max_mem_alloc():
    if hasattr(torch, "musa") and torch.musa.is_available() and torch.musa.is_initialized():
        return int(torch.musa.max_memory_allocated())
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _max_mem_reserved():
    if hasattr(torch, "musa") and torch.musa.is_available() and torch.musa.is_initialized():
        return int(torch.musa.max_memory_reserved())
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return int(torch.cuda.max_memory_reserved())
    return 0


def compile_count():
    utils = sys.modules.get("torch._dynamo.utils")
    if utils is None:
        return 0
    return utils.counters["stats"]["unique_graphs"] + utils.counters["aot_autograd"]["total"]


def rng_fingerprint():
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if hasattr(torch, "musa") and torch.musa.is_available() and torch.musa.is_initialized():
        h.update(torch.musa.get_rng_state().cpu().numpy().tobytes())
    elif torch.cuda.is_available() and torch.cuda.is_initialized():
        h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()[:16]


class StopTiming(BaseException):
    """BaseException so training code cannot swallow it."""


def kill_descendants():
    children = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    stack, found = [os.getpid()], []
    while stack:
        for c in children.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    for pid in found:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(found)


class TorchProfile:
    """torch.profiler over the profiled span with record_function annotations around major SANA-WM modules."""

    def __init__(self, out):
        self.out, self.prof, self.handles = out, None, []

    def _wrap(self, mod, label):
        ctx = {}

        def pre(_m, _a):
            r = torch.autograd.profiler.record_function(f"## {label}")
            r.__enter__()
            ctx.setdefault("stack", []).append(r)

        def post(_m, _a, _o):
            if ctx.get("stack"):
                ctx["stack"].pop().__exit__(None, None, None)

        self.handles.append(mod.register_forward_pre_hook(pre))
        self.handles.append(mod.register_forward_hook(post))

    def _attach_annotations(self):
        seen_ids = set()
        dit_mod = None
        txt_mod = None
        for obj in gc.get_objects():
            if not isinstance(obj, torch.nn.Module):
                continue
            cls = type(obj).__name__
            if dit_mod is None and "SanaMSVideoCamCtrl" in cls:
                dit_mod = obj
            elif txt_mod is None and cls in ("Gemma2Model", "Gemma2ForCausalLM"):
                txt_mod = obj
        if txt_mod is not None and id(txt_mod) not in seen_ids:
            seen_ids.add(id(txt_mod))
            self._wrap(txt_mod, "text_encoder")
        if dit_mod is not None and id(dit_mod) not in seen_ids:
            seen_ids.add(id(dit_mod))
            self._wrap(dit_mod, "dit_forward")
            for sub_name in ("x_embedder", "t_embedder", "y_embedder", "final_layer"):
                sub = getattr(dit_mod, sub_name, None)
                if isinstance(sub, torch.nn.Module):
                    self._wrap(sub, f"dit.{sub_name}")
            blocks = getattr(dit_mod, "blocks", None)
            if isinstance(blocks, torch.nn.ModuleList) and len(blocks) > 0:
                for blk in blocks:
                    for part, lbl in (
                        ("attn", "dit.self_attn"),
                        ("cross_attn", "dit.cross_attn"),
                        ("mlp", "dit.mlp"),
                    ):
                        m = getattr(blk, part, None)
                        if isinstance(m, torch.nn.Module):
                            self._wrap(m, lbl)

    def start(self):
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            self._attach_annotations()
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CPU]
        if hasattr(ProfilerActivity, "PrivateUse1") and hasattr(torch, "musa") and torch.musa.is_available():
            acts.append(ProfilerActivity.PrivateUse1)
        elif hasattr(ProfilerActivity, "CUDA") and torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        self.prof = profile(
            activities=acts,
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        )
        self.prof.__enter__()

    def stop(self, span_s, iterations):
        self.prof.__exit__(None, None, None)
        for h in self.handles:
            h.remove()
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return
        kernels, annotations, intervals, ops = {}, {}, [], {}
        for e in self.prof.events():
            dev = str(getattr(e, "device_type", ""))
            if dev.endswith("PrivateUse1") or dev.endswith("CUDA"):
                k = kernels.setdefault(e.name, [0.0, 0])
                k[0] += e.time_range.elapsed_us()
                k[1] += 1
                intervals.append((e.time_range.start, e.time_range.end))
            elif e.name.startswith("## "):
                a = annotations.setdefault(e.name[3:], [0.0, 0, 0.0])
                dev_t = getattr(e, "device_time_total", None)
                if dev_t is None:
                    dev_t = getattr(e, "privateuse1_time_total", 0.0) or getattr(e, "cuda_time_total", 0.0)
                a[0] += float(dev_t or 0.0)
                a[1] += 1
                a[2] += float(getattr(e, "cpu_time_total", 0.0) or 0.0)
            elif e.name.startswith(("aten::", "c10d::", "mccl", "nccl", "musa")):
                o = ops.setdefault(e.name, [0.0, 0])
                o[0] += float(getattr(e, "self_cpu_time_total", 0.0) or 0.0)
                o[1] += 1
        intervals.sort()
        busy, cur = 0.0, None
        for s, t in intervals:
            if cur is None or s > cur[1]:
                if cur:
                    busy += cur[1] - cur[0]
                cur = [s, t]
            else:
                cur[1] = max(cur[1], t)
        if cur:
            busy += cur[1] - cur[0]
        total_kernel = sum(v[0] for v in kernels.values())
        summary = {
            "span_s": span_s,
            "iterations": iterations,
            "gpu_busy_s": busy / 1e6,
            "gpu_busy_fraction": (busy / 1e6) / span_s if span_s else None,
            "kernel_time_s": total_kernel / 1e6,
            "kernels": [
                {
                    "name": n[:300],
                    "total_ms": round(v[0] / 1000, 3),
                    "calls": v[1],
                    "share_of_kernel_time": round(v[0] / total_kernel, 4) if total_kernel else None,
                }
                for n, v in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:80]
            ],
            "annotations": {
                n: {"gpu_ms": round(v[0] / 1000, 2), "calls": v[1], "cpu_ms": round(v[2] / 1000, 2)}
                for n, v in sorted(annotations.items())
            },
            "cpu_ops_self_time": [
                {"name": n, "self_cpu_ms": round(v[0] / 1000, 2), "calls": v[1]}
                for n, v in sorted(ops.items(), key=lambda kv: -kv[1][0])[:40]
            ],
        }
        Path(f"{self.out}.torchprof.json").write_text(json.dumps(summary, indent=1) + "\n")
        self.prof.export_chrome_trace(f"{self.out}.trace.json")


class IterTimer:
    def __init__(self, warmup, window, repeats, profile=None, profile_iters=2, out=None):
        self.warmup, self.window, self.repeats, self.profile = warmup, window, repeats, profile
        self.marks = {warmup + k * window - 1: k for k in range(repeats + 1)}
        self.last = max(self.marks)
        if profile:
            self.marks = {warmup - 1: 0}
            self.last = warmup - 1 + profile_iters
        self.profile_iters = profile_iters
        self.it = -1
        self.edges = {}
        self.first_iter_s = None
        self.ends = []
        self.rng = []
        self.torchprof = TorchProfile(out) if profile == "torch" else None
        self.profile_t0 = None

    def __call__(self, _opt, _args, _kwargs):
        self.it += 1
        self.ends.append(time.perf_counter())
        self.rng.append(rng_fingerprint())
        if self.it == 0:
            _dev_sync()
            self.first_iter_s = time.perf_counter() - T_START
        if self.profile:
            if self.it == self.warmup - 1:
                _dev_sync()
                self.profile_t0 = time.perf_counter()
                self.edges[0] = (self.profile_t0, time.time(), compile_count())
                if self.torchprof is not None:
                    self.torchprof.start()
            elif self.it == self.last:
                _dev_sync()
                t1 = time.perf_counter()
                self.edges[1] = (t1, time.time(), compile_count())
                if self.torchprof is not None:
                    self.torchprof.stop(t1 - self.profile_t0, self.profile_iters)
                raise StopTiming()
            return
        k = self.marks.get(self.it)
        if k is not None:
            _dev_sync()
            self.edges[k] = (time.perf_counter(), time.time(), compile_count())
            if self.it == self.last:
                raise StopTiming()

    def report(self):
        windows = []
        n = 1 if self.profile else self.repeats
        size = self.profile_iters if self.profile else self.window
        for k in range(n):
            if k not in self.edges or k + 1 not in self.edges:
                continue
            a, b = self.edges[k], self.edges[k + 1]
            windows.append(
                {
                    "k": k,
                    "iterations": size,
                    "seconds": b[0] - a[0],
                    "wall_start": a[1],
                    "wall_end": b[1],
                    "compiles_in_window": b[2] - a[2],
                }
            )
        return windows


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: timing.py [options] -- <train_sana_wm_stage1.py arguments>")
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--entry", default="train_video_scripts/train_sana_wm_stage1.py")
    parser.add_argument("--profile", choices=("torch",))
    parser.add_argument("--profile-iters", type=int, default=2)
    args = parser.parse_args(sys.argv[1:split])
    train_argv = sys.argv[split + 1 :]

    timer = IterTimer(args.warmup, args.window, args.repeats, args.profile, args.profile_iters, args.out)
    handle = register_optimizer_step_post_hook(timer)
    entry = Path(args.entry).resolve()
    tree = Path.cwd().resolve()
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    sys.argv = [str(entry), *train_argv]
    rank = int(os.environ.get("RANK", "0"))
    status, error = "incomplete", None
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except StopTiming:
        status = "ok"
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        handle.remove()

    musa_tf32 = None
    try:
        musa_tf32 = torch.backends.musa.matmul.allow_tf32
    except AttributeError:
        pass
    result = {
        "status": status,
        "error": error,
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "warmup": args.warmup,
        "window": args.window,
        "repeats": args.repeats,
        "profile": args.profile,
        "iterations_seen": timer.it + 1,
        "startup_to_first_iteration_s": timer.first_iter_s,
        "peak_memory_allocated_bytes": _max_mem_alloc(),
        "peak_memory_reserved_bytes": _max_mem_reserved(),
        "windows": timer.report() if status == "ok" else [],
        "iteration_end_s": [round(t - T_START, 4) for t in timer.ends],
        "rng_fingerprints": timer.rng,
        "train_argv": train_argv,
        "math_flags_at_end": {
            "musa.matmul.allow_tf32": musa_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    Path(f"{args.out}.r{rank}.json").write_text(json.dumps(result, indent=1) + "\n")
    sys.stdout.flush()
    sys.stderr.flush()
    kill_descendants()
    os._exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
