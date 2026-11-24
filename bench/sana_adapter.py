"""numerics adapter for SANA-WM Stage-1 bidirectional FSDP2 training (train_video_scripts/train_sana_wm_stage1.py).

build() parses the frozen recipe YAML with pyrallis and executes the setup body of main(cfg) (everything before
the `for epoch in ...:` loop) from the tree under test, so the candidate's own setup helpers (_build_fsdp2_plugin,
build_model, build_optimizer, etc.) run verbatim. step() executes the verbatim AST body of `for batch in
train_dataloader:` from the candidate's main(cfg). Gate 0 requires main(cfg) outside that per-step loop body
and the `__main__` block to equal the reference AST.

Under FSDP2, each rank owns a local shard of every DiT parameter (`DTensor`), so `sync={"dit": "local"}` and
`canonical()` maps `DTensor`s to their local shard via `.to_local()`.
"""
import ast
import copy
import gc
import hashlib
import importlib
import os
import sys
from pathlib import Path

import pyrallis
import torch
try:
    import torch_musa  # noqa: F401
except ImportError:
    pass
import torch.distributed as dist
from torch.utils.data import DistributedSampler

from numerics.adapter import Adapter, Objects

from bench import recipe

WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "module.",
    "_orig_mod.",
    "_checkpoint_wrapped_module.",
)


def strip_prefix(name: str) -> str:
    for p in WRAPPER_PREFIXES:
        name = name.replace(p, "")
    return name


def _digest_obj(obj) -> str:
    h = hashlib.sha256()

    def _walk(x):
        if isinstance(x, torch.Tensor):
            h.update(b"T")
            h.update(str(x.dtype).encode() + str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b"D")
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                _walk(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"L")
            for item in x:
                _walk(item)
        else:
            h.update(b"V" + repr(x).encode())

    _walk(obj)
    return h.hexdigest()[:16]


class EpochFeed:
    """Epoch-aware wrapper around train_dataloader that reproduces main()'s outer epoch loop and digests batches."""

    def __init__(self, dataloader, max_epochs: int):
        self.dl = dataloader
        self.max_epochs = max_epochs
        self.epoch = 0
        self._it = None

    def next_batch(self):
        while True:
            if self._it is None:
                self.epoch += 1
                if self.epoch > self.max_epochs:
                    raise StopIteration("exhausted num_epochs")
                if isinstance(self.dl.sampler, DistributedSampler):
                    self.dl.sampler.set_epoch(self.epoch)
                self._it = iter(self.dl)
            try:
                batch = next(self._it)
                return self.epoch, batch, _digest_obj(batch)
            except StopIteration:
                self._it = None


def _extract_main_parts(entry: Path):
    """Split main(cfg) into (setup_code, step_fn_code, arg_name) from the verbatim AST of entry."""
    tree = ast.parse(entry.read_text())
    main_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break
    if main_fn is None:
        raise RuntimeError(f"could not find def main() in {entry}")

    epoch_idx = None
    epoch_for = None
    for idx, stmt in enumerate(main_fn.body):
        if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and stmt.target.id == "epoch":
            epoch_idx = idx
            epoch_for = stmt
            break
    if epoch_for is None:
        raise RuntimeError(f"could not find `for epoch in ...:` loop in main() of {entry}")

    batch_for = None
    for stmt in epoch_for.body:
        if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and stmt.target.id == "batch":
            batch_for = stmt
            break
    if batch_for is None:
        raise RuntimeError(f"could not find `for batch in train_dataloader:` loop in main() of {entry}")

    setup_mod = ast.Module(body=copy.deepcopy(main_fn.body[:epoch_idx]), type_ignores=[])
    ast.fix_missing_locations(setup_mod)
    setup_code = compile(setup_mod, filename=str(entry), mode="exec")

    # Collect names assigned in setup + loop body so mutations persist in self._ns
    global_names = sorted(
        {
            n.id
            for stmt in main_fn.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        - {"batch"}
    )
    fn_body = [ast.Global(names=global_names)] + copy.deepcopy(batch_for.body)
    step_fn_ast = ast.FunctionDef(
        name="_rba_one_step",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="batch")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=fn_body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    step_mod = ast.Module(body=[step_fn_ast], type_ignores=[])
    ast.fix_missing_locations(step_mod)
    step_code = compile(step_mod, filename=str(entry), mode="exec")
    arg_name = main_fn.args.args[0].arg if main_fn.args.args else "cfg"
    return setup_code, step_code, arg_name



_DET_ATTN_INSTALLED = False


def _install_deterministic_attention():
    """Route attention kernels without a deterministic backward on MUSA to chunked SDPBackend.MATH
    (same precision, O(chunk*Lk) memory, 100% bitwise deterministic) when strict deterministic mode is active."""
    global _DET_ATTN_INSTALLED
    if _DET_ATTN_INSTALLED:
        return
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.checkpoint import checkpoint
    try:
        import mate
    except ImportError:
        mate = None

    orig_sdpa = F.scaled_dot_product_attention
    orig_mate_varlen = getattr(mate, "flash_attn_varlen_func", None) if mate is not None else None

    def _math_sdpa_chunk(q_c, k, v, mask_c, dropout_p, is_causal, scale, enable_gqa):
        with sdpa_kernel(SDPBackend.MATH):
            return orig_sdpa(
                q_c, k, v, attn_mask=mask_c, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )

    def _det_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if not torch.are_deterministic_algorithms_enabled() or query.device.type != "musa":
            return orig_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )
        Lq, Lk = query.shape[-2], key.shape[-2]
        chunk = 2048
        if attn_mask is None and not is_causal and Lq > chunk and Lq * Lk > 16_000_000:
            outs = []
            use_ckpt = torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad)
            for s in range(0, Lq, chunk):
                qc = query[..., s : s + chunk, :]
                if use_ckpt:
                    out_c = checkpoint(
                        _math_sdpa_chunk, qc, key, value, None, dropout_p, False, scale, enable_gqa,
                        use_reentrant=False,
                    )
                else:
                    out_c = _math_sdpa_chunk(qc, key, value, None, dropout_p, False, scale, enable_gqa)
                outs.append(out_c)
            return torch.cat(outs, dim=-2)
        return _math_sdpa_chunk(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    F.scaled_dot_product_attention = _det_sdpa

    if orig_mate_varlen is not None:
        def _det_mate_varlen(
            q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, *args, **kwargs
        ):
            if not torch.are_deterministic_algorithms_enabled() or q.device.type != "musa":
                return orig_mate_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs)
            scale = kwargs.get("softmax_scale", None)
            causal = bool(kwargs.get("causal", False))
            if cu_seqlens_q is None or cu_seqlens_k is None:
                # [B, L, H, D]
                out = _det_sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal, scale=scale)
                return out.transpose(1, 2).contiguous()
            cu_q = cu_seqlens_q.tolist()
            cu_k = cu_seqlens_k.tolist()
            outs = []
            for i in range(len(cu_q) - 1):
                qs, qe = cu_q[i], cu_q[i + 1]
                ks, ke = cu_k[i], cu_k[i + 1]
                qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
                ki = k[ks:ke].unsqueeze(0).transpose(1, 2)
                vi = v[ks:ke].unsqueeze(0).transpose(1, 2)
                oi = _det_sdpa(qi, ki, vi, is_causal=causal, scale=scale)
                outs.append(oi.transpose(1, 2).squeeze(0))
            return torch.cat(outs, dim=0).contiguous()

        mate.flash_attn_varlen_func = _det_mate_varlen
        if hasattr(mate, "mha_interface"):
            mate.mha_interface.flash_attn_varlen_func = _det_mate_varlen

    _DET_ATTN_INSTALLED = True


class SanaWmAdapter(Adapter):
    name = "sana_wm_stage1_fsdp2"

    def __init__(self, repo, base_yaml, fixture_dir, model_root, scratch):
        self.repo = Path(repo).resolve()
        self.base_yaml = Path(base_yaml).resolve()
        self.fixture_dir = Path(fixture_dir).resolve()
        self.model_root = Path(model_root).resolve()
        self.scratch = Path(scratch)
        self.case = {}
        self._ns = None
        self._step_fn = None
        self._feed = None
        self._last_batch_digest = None
        self._theta0 = None

    def configure(self, case):
        self.case = dict(case)

    def _release(self):
        if self._ns is not None:
            self._ns = None
            self._step_fn = None
            self._feed = None
            self._theta0 = None
            gc.collect()
            if hasattr(torch, "musa") and torch.musa.is_available():
                torch.musa.empty_cache()

    def build(self, seed, golden=False):
        if golden:
            raise SystemExit("SANA-WM 2.66B FSDP2 has no fp64 golden record")
        if torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(True, warn_only=False)
            _install_deterministic_attention()
        self._release()
        os.chdir(self.repo)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

        # Ensure imports come strictly from self.repo
        for mod_name in list(sys.modules):
            if mod_name == "diffusion" or mod_name.startswith("diffusion.") or mod_name.startswith("train_video_scripts"):
                sys.modules.pop(mod_name, None)

        T = importlib.import_module("train_video_scripts.train_sana_wm_stage1")
        if not Path(T.__file__).resolve().is_relative_to(self.repo):
            raise RuntimeError(f"train_sana_wm_stage1 imported from {T.__file__}, not {self.repo}")

        case_scratch = self.scratch / f"{self.case.get('name', 'case')}_s{seed}"
        cfg_argv = recipe.argv(
            case_scratch,
            self.base_yaml,
            self.fixture_dir,
            self.model_root,
            seed=seed,
            num_workers=0,
        )
        cfg = pyrallis.parse(config_class=T.SanaVideoCamCtrlConfig, args=cfg_argv)

        entry = self.repo / recipe.ENTRY
        setup_code, step_code, arg_name = _extract_main_parts(entry)
        self._ns = dict(vars(T))
        self._ns[arg_name] = cfg
        exec(setup_code, self._ns)
        exec(step_code, self._ns)
        self._step_fn = self._ns["_rba_one_step"]
        self._feed = EpochFeed(self._ns["train_dataloader"], int(self._ns["config"].train.num_epochs))
        self._ns["epoch"] = 1

        if self.case.get("track_disp"):
            self._theta0 = []
            for p in self._ns["model"].parameters():
                if not p.requires_grad:
                    continue
                loc = p.to_local() if hasattr(p, "to_local") else p
                self._theta0.append((p, loc.detach().float().cpu().clone()))

    @staticmethod
    def _unwrap_opt(opt):
        inner = getattr(opt, "optimizer", None)
        return inner if inner is not None else opt

    def objects(self):
        return Objects(
            models={"dit": self._ns["model"]},
            optimizers={"dit": self._unwrap_opt(self._ns["optimizer"])},
            sync={"dit": "local"},
            capture_outputs=("dit",),
        )

    def phases(self):
        return ["train"]

    def set_phase(self, phase, step):
        if phase != "train":
            raise RuntimeError(f"unexpected phase {phase!r}")

    def batch(self, step, rank):
        return None

    def _param_disp_metrics(self, prev_cpu):
        sq_sum = 0.0
        abs_chunks = []
        step_chunks = []
        for (p, p0_cpu), prev in zip(self._theta0, prev_cpu):
            loc = p.to_local() if hasattr(p, "to_local") else p
            cur = loc.detach().float().cpu()
            diff = cur - p0_cpu
            sq_sum += float(diff.pow(2).sum())
            abs_chunks.append(diff.abs().reshape(-1))
            step_chunks.append((cur - prev).abs().reshape(-1))
        sq_t = torch.tensor(sq_sum, dtype=torch.float64, device=self._ns["accelerator"].device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sq_t, op=dist.ReduceOp.SUM)
        disp_l2 = torch.sqrt(sq_t.cpu())

        all_abs = torch.cat(abs_chunks)
        k = max(1, all_abs.numel() // 100)
        disp_top1 = torch.topk(all_abs, k, sorted=False).values.double().mean()

        all_step = torch.cat(step_chunks)
        step_top1 = torch.topk(all_step, k, sorted=False).values.double().mean()
        return disp_l2, disp_top1, step_top1

    def step(self, step, batch):
        ns = self._ns
        epoch, real_batch, digest = self._feed.next_batch()
        ns["epoch"] = epoch
        want_disp = self._theta0 is not None and step >= int(self.case.get("disp_start_step", 0))
        prev_cpu = None
        if want_disp:
            prev_cpu = [
                (p.to_local() if hasattr(p, "to_local") else p).detach().float().cpu().clone()
                for p, _ in self._theta0
            ]
        gs_before = int(ns["global_step"])
        self._step_fn(real_batch)
        if int(ns["global_step"]) != gs_before + 1:
            raise RuntimeError(f"global_step moved from {gs_before} to {ns['global_step']}, expected {gs_before + 1}")
        self._last_batch_digest = digest

        loss_key = ns["config"].loss_report_name
        losses = {
            "loss": torch.tensor(float(ns["logs"][loss_key]), dtype=torch.float64),
            "loss_local": ns["loss"].detach().reshape(()).to(torch.float64),
        }
        if want_disp:
            disp_l2, disp_top1, step_top1 = self._param_disp_metrics(prev_cpu)
            losses["param_disp_l2"] = disp_l2
            losses["param_disp_top1pct"] = disp_top1
            losses["step_disp_top1pct"] = step_top1
        return losses

    def invariants(self):
        ns = self._ns
        cfg = ns["config"]
        return {
            "global_step": int(ns["global_step"]),
            "epoch": int(ns["epoch"]),
            "data/batch": self._last_batch_digest or "",
            "train_batch_size": int(cfg.train.train_batch_size),
            "gradient_accumulation_steps": int(cfg.train.gradient_accumulation_steps),
            "gradient_clip": float(cfg.train.gradient_clip),
            "use_fsdp": bool(cfg.train.use_fsdp),
            "cp_size": int(ns["cp_size"]),
            "mixed_precision": str(cfg.model.mixed_precision),
            "fp32_attention": bool(getattr(cfg.model, "fp32_attention", False)),
            "num_frames": int(cfg.data.num_frames),
        }

    def canonical(self, kind, name, tensor):
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        return {strip_prefix(name): tensor}

    def save_state(self, path):
        raise NotImplementedError("cases start from the frozen published checkpoint")

    def load_state(self, path):
        raise NotImplementedError("cases start from the frozen published checkpoint")
