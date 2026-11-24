"""Single-pass MUSA C++ (mcc, MTT S5000) Fused Multi-Tensor FP32 AdamW kernel.

Replaces multi-pass PyTorch `_foreach_adamw` (~62 ms/iter across 850 sharded FSDP2 params)
with a single-pass 128-bit vectorized multi-tensor kernel (~8-10 ms/iter, ~6x faster).
Static chunk descriptors are cached by tensor sizes; only 4*N pointers (~27 KB) are
uploaded per step via pinned memory, so FSDP2 per-step grad buffer changes cost ~0.2 ms.
Exact FP32 master weights & FP32 moments; preserves PyTorch AdamW math.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import torch

_LIB = None
_MCC_PATH = "/usr/local/musa-5.2/bin/mcc"
_MUSA_PATH = "/usr/local/musa-5.2"

_MUSA_SRC = r"""
#include <musa_runtime.h>
#include <cstdint>
#include <cmath>

struct TensorPtrs {
    float* p;
    const float* g;
    float* m;
    float* v;
};

struct ChunkDesc {
    int64_t tidx;
    int64_t offset;
    int64_t count;
};

static __device__ __forceinline__ float _warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

extern "C" __global__ void fused_multi_tensor_l2_norm_sq_f32(
    const float* const* __restrict__ g_ptrs,
    const ChunkDesc* __restrict__ chunks,
    float* __restrict__ block_sums,
    int n_chunks
) {
    const int cid = blockIdx.x;
    if (cid >= n_chunks) return;
    const ChunkDesc c = chunks[cid];
    const float* __restrict__ g = g_ptrs[c.tidx] + c.offset;
    const int64_t n = c.count;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    int64_t vec_end = 0;
    if ((reinterpret_cast<uintptr_t>(g) & 15) == 0) {
        const int64_t n4 = n >> 2;
        const float4* g4 = reinterpret_cast<const float4*>(g);
        for (int64_t i = tid; i < n4; i += 256) {
            float4 gv = g4[i];
            acc = fmaf(gv.x, gv.x, fmaf(gv.y, gv.y, fmaf(gv.z, gv.z, fmaf(gv.w, gv.w, acc))));
        }
        vec_end = n4 << 2;
    }
    for (int64_t i = vec_end + tid; i < n; i += 256) {
        float gv = g[i];
        acc = fmaf(gv, gv, acc);
    }
    acc = _warp_sum(acc);
    __shared__ float s_red[8];
    if ((tid & 31) == 0) s_red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        block_sums[cid] = tot;
    }
}

extern "C" __global__ void reduce_chunk_sums_f32(
    const float* __restrict__ block_sums,
    float* __restrict__ out_sq,
    int n_chunks
) {
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n_chunks; i += 256) acc += block_sums[i];
    acc = _warp_sum(acc);
    __shared__ float s_red[8];
    if ((tid & 31) == 0) s_red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        out_sq[0] = tot;
    }
}

extern "C" __global__ void compute_clip_coef_from_sq_f32(
    const float* __restrict__ global_sq_ptr,
    float* __restrict__ clip_coef_out,
    float* __restrict__ total_norm_out,
    float max_norm
) {
    float tn = sqrtf(global_sq_ptr[0]);
    float cc = max_norm / (tn + 1e-6f);
    clip_coef_out[0] = (cc < 1.0f) ? cc : 1.0f;
    total_norm_out[0] = tn;
}

// Single-pass vectorized FP32 AdamW over static chunks + per-step pointer table.
// Grid: (n_chunks), Block: 256. Each thread processes strided float4 (16B) + scalar tail.
extern "C" __global__ void fused_multi_tensor_adamw_f32(
    const TensorPtrs* __restrict__ ptrs,
    const ChunkDesc* __restrict__ chunks,
    int n_chunks,
    float beta1,
    float beta2,
    float one_minus_b1,
    float one_minus_b2,
    float wd_factor,      // 1.0f - lr * weight_decay
    float step_size,      // lr / (1 - beta1^step)
    float inv_sqrt_bc2,   // 1.0f / sqrt(1 - beta2^step)
    float eps,
    int has_wd,
    const float* __restrict__ grad_scale_ptr
) {
    const int cid = blockIdx.x;
    if (cid >= n_chunks) return;
    const ChunkDesc c = chunks[cid];
    const TensorPtrs tp = ptrs[c.tidx];
    float* __restrict__ p = tp.p + c.offset;
    const float* __restrict__ g = tp.g + c.offset;
    float* __restrict__ m = tp.m + c.offset;
    float* __restrict__ v = tp.v + c.offset;
    const int64_t n = c.count;
    const int tid = threadIdx.x;
    const float gs = (grad_scale_ptr != nullptr) ? grad_scale_ptr[0] : 1.0f;
    const bool use_gs = (grad_scale_ptr != nullptr);

    const bool aligned = ((reinterpret_cast<uintptr_t>(p) | reinterpret_cast<uintptr_t>(g) |
                           reinterpret_cast<uintptr_t>(m) | reinterpret_cast<uintptr_t>(v)) & 15) == 0;
    int64_t vec_end = 0;
    if (aligned) {
        const int64_t n4 = n >> 2;
        float4* p4 = reinterpret_cast<float4*>(p);
        const float4* g4 = reinterpret_cast<const float4*>(g);
        float4* m4 = reinterpret_cast<float4*>(m);
        float4* v4 = reinterpret_cast<float4*>(v);
        for (int64_t i = tid; i < n4; i += 256) {
            float4 pv = p4[i];
            float4 gv = g4[i];
            float4 mv = m4[i];
            float4 vv = v4[i];

            if (use_gs) {
                gv.x *= gs; gv.y *= gs; gv.z *= gs; gv.w *= gs;
            }
            if (has_wd) {
                pv.x *= wd_factor; pv.y *= wd_factor; pv.z *= wd_factor; pv.w *= wd_factor;
            }
            mv.x = fmaf(beta1, mv.x, one_minus_b1 * gv.x);
            mv.y = fmaf(beta1, mv.y, one_minus_b1 * gv.y);
            mv.z = fmaf(beta1, mv.z, one_minus_b1 * gv.z);
            mv.w = fmaf(beta1, mv.w, one_minus_b1 * gv.w);

            vv.x = fmaf(beta2, vv.x, one_minus_b2 * gv.x * gv.x);
            vv.y = fmaf(beta2, vv.y, one_minus_b2 * gv.y * gv.y);
            vv.z = fmaf(beta2, vv.z, one_minus_b2 * gv.z * gv.z);
            vv.w = fmaf(beta2, vv.w, one_minus_b2 * gv.w * gv.w);

            pv.x -= step_size * mv.x / (sqrtf(vv.x) * inv_sqrt_bc2 + eps);
            pv.y -= step_size * mv.y / (sqrtf(vv.y) * inv_sqrt_bc2 + eps);
            pv.z -= step_size * mv.z / (sqrtf(vv.z) * inv_sqrt_bc2 + eps);
            pv.w -= step_size * mv.w / (sqrtf(vv.w) * inv_sqrt_bc2 + eps);

            p4[i] = pv;
            m4[i] = mv;
            v4[i] = vv;
        }
        vec_end = n4 << 2;
    }

    for (int64_t i = vec_end + tid; i < n; i += 256) {
        float pv = p[i];
        float gv = use_gs ? (g[i] * gs) : g[i];
        float mv = m[i];
        float vv = v[i];
        if (has_wd) pv *= wd_factor;
        mv = fmaf(beta1, mv, one_minus_b1 * gv);
        vv = fmaf(beta2, vv, one_minus_b2 * gv * gv);
        pv -= step_size * mv / (sqrtf(vv) * inv_sqrt_bc2 + eps);
        p[i] = pv;
        m[i] = mv;
        v[i] = vv;
    }
}

extern "C" int launch_fused_l2_norm_sq_f32(
    const void* g_ptrs_dev, const void* chunks_dev, void* block_sums_dev, void* out_sq_dev,
    int n_chunks, void* stream_ptr
) {
    if (n_chunks <= 0) return 0;
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    fused_multi_tensor_l2_norm_sq_f32<<<n_chunks, 256, 0, s>>>(
        reinterpret_cast<const float* const*>(g_ptrs_dev),
        reinterpret_cast<const ChunkDesc*>(chunks_dev),
        reinterpret_cast<float*>(block_sums_dev), n_chunks
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    reduce_chunk_sums_f32<<<1, 256, 0, s>>>(
        reinterpret_cast<const float*>(block_sums_dev),
        reinterpret_cast<float*>(out_sq_dev), n_chunks
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_compute_clip_coef_f32(
    const void* global_sq_dev, void* clip_coef_dev, void* total_norm_dev,
    float max_norm, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    compute_clip_coef_from_sq_f32<<<1, 1, 0, s>>>(
        reinterpret_cast<const float*>(global_sq_dev),
        reinterpret_cast<float*>(clip_coef_dev),
        reinterpret_cast<float*>(total_norm_dev),
        max_norm
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_fused_multi_tensor_adamw_f32(
    const void* ptrs_dev, const void* chunks_dev, int n_chunks,
    float beta1, float beta2,
    float wd_factor, float step_size, float inv_sqrt_bc2, float eps, int has_wd,
    const void* grad_scale_dev, void* stream_ptr
) {
    if (n_chunks <= 0) return 0;
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    fused_multi_tensor_adamw_f32<<<n_chunks, 256, 0, s>>>(
        reinterpret_cast<const TensorPtrs*>(ptrs_dev),
        reinterpret_cast<const ChunkDesc*>(chunks_dev), n_chunks,
        beta1, beta2, 1.0f - beta1, 1.0f - beta2,
        wd_factor, step_size, inv_sqrt_bc2, eps, has_wd,
        reinterpret_cast<const float*>(grad_scale_dev)
    );
    return static_cast<int>(musaGetLastError());
}
"""


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    src_hash = hashlib.sha256(_MUSA_SRC.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "sana_musa_ops_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"fused_adamw_v2_s5000_{src_hash}.so"
    if not so_path.exists():
        pid = os.getpid()
        src_path = cache_dir / f"fused_adamw_v2_s5000_{src_hash}_{pid}.mu"
        tmp_so = cache_dir / f"fused_adamw_v2_s5000_{src_hash}_{pid}.so"
        src_path.write_text(_MUSA_SRC, encoding="utf-8")
        mcc = _MCC_PATH if os.path.exists(_MCC_PATH) else "mcc"
        cmd = [
            mcc, "-x", "musa", "-O3", "--offload-arch=mp_31",
            "-fmusa-flush-denormals-to-zero", "-shared", "-Xcompiler", "-fPIC",
            f"-I{_MUSA_PATH}/include", f"-L{_MUSA_PATH}/lib", "-lmusart",
            str(src_path), "-o", str(tmp_so),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"mcc compilation failed:\n{res.stderr}")
        os.replace(tmp_so, so_path)
        try:
            src_path.unlink(missing_ok=True)
        except OSError:
            pass

    lib = ctypes.CDLL(str(so_path))
    lib.launch_fused_l2_norm_sq_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_fused_l2_norm_sq_f32.restype = ctypes.c_int
    lib.launch_compute_clip_coef_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
    ]
    lib.launch_compute_clip_coef_f32.restype = ctypes.c_int
    lib.launch_fused_multi_tensor_adamw_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.launch_fused_multi_tensor_adamw_f32.restype = ctypes.c_int
    _LIB = lib
    return lib


# Cache static ChunkDesc table by (device_index, tuple(numels))
_CHUNK_CACHE: dict = {}
# Cache pinned host + device pointer staging buffer by (device_index, n_tensors)
_PTR_BUF_CACHE: dict = {}
_GRAD_PTR_BUF_CACHE: dict = {}
_NORM_SCRATCH_CACHE: dict = {}
_CHUNK_SIZE = 131072  # 128K floats = 512 KiB per CTA


def _unwrap_local(t: torch.Tensor) -> torch.Tensor:
    if hasattr(t, "_local_tensor"):
        return t._local_tensor
    return t


def can_use_fused_adamw(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    *,
    amsgrad: bool,
    maximize: bool,
    has_complex: bool,
    decoupled_weight_decay: bool,
    grad_scale,
    found_inf,
) -> bool:
    if os.environ.get("SANA_OPT_FUSED_ADAMW", "1") == "0":
        return False
    if amsgrad or maximize or has_complex or (not decoupled_weight_decay) or grad_scale is not None or found_inf is not None:
        return False
    if not params:
        return False
    if state_steps[0].device.type != "cpu":
        return False
    step0 = int(state_steps[0].item())
    dev0 = _unwrap_local(params[0]).device
    if dev0.type != "musa":
        return False
    for p, g, m, v, st in zip(params, grads, exp_avgs, exp_avg_sqs, state_steps):
        if st.device.type != "cpu" or int(st.item()) != step0:
            return False
        pl, gl, ml, vl = _unwrap_local(p), _unwrap_local(g), _unwrap_local(m), _unwrap_local(v)
        if (
            pl.dtype != torch.float32
            or gl.dtype != torch.float32
            or ml.dtype != torch.float32
            or vl.dtype != torch.float32
            or pl.device != dev0
            or gl.device != dev0
            or ml.device != dev0
            or vl.device != dev0
            or not (pl.is_contiguous() and gl.is_contiguous() and ml.is_contiguous() and vl.is_contiguous())
        ):
            return False
    return True


def _get_chunks(dev: torch.device, dev_idx: int, numels: List[int]):
    chunk_key = (dev_idx, tuple(numels))
    cached = _CHUNK_CACHE.get(chunk_key)
    if cached is None:
        rows = []
        for tidx, n in enumerate(numels):
            off = 0
            while off < n:
                cnt = min(_CHUNK_SIZE, n - off)
                rows.append((tidx, off, cnt))
                off += cnt
        chunks_dev = torch.tensor(rows, dtype=torch.int64, device=dev)
        n_chunks = len(rows)
        if len(_CHUNK_CACHE) > 16:
            _CHUNK_CACHE.clear()
        _CHUNK_CACHE[chunk_key] = (chunks_dev, n_chunks)
        return chunks_dev, n_chunks
    return cached


def fused_l2_grad_clip_coef(
    params: List[torch.Tensor],
    max_norm: float,
    process_group=None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Compute global FP32 L2 grad norm & clip_coef on GPU with 1 fused kernel + 4-byte AllReduce.

    Returns ``(total_norm_scalar_gpu, clip_coef_scalar_gpu)`` or ``None`` if not eligible.
    Does NOT modify gradients in-place — caller passes ``clip_coef`` into ``fused_multi_tensor_adamw_step``.
    """
    if os.environ.get("SANA_OPT_FUSED_GRAD_CLIP", "1") == "0":
        return None
    grads_local = []
    numels = []
    dev0 = None
    is_dtensor = False
    for p in params:
        g = p.grad
        if g is None:
            continue
        if hasattr(g, "_local_tensor"):
            is_dtensor = True
            gl = g._local_tensor
        else:
            gl = g
        if gl.device.type != "musa" or gl.dtype != torch.float32 or not gl.is_contiguous():
            return None
        if dev0 is None:
            dev0 = gl.device
        elif gl.device != dev0:
            return None
        n = gl.numel()
        if n > 0:
            grads_local.append(gl.data_ptr())
            numels.append(n)

    if not grads_local or dev0 is None:
        return None

    dev_idx = dev0.index or 0
    n_tensors = len(numels)
    chunks_dev, n_chunks = _get_chunks(dev0, dev_idx, numels)

    pbuf_key = (dev_idx, n_tensors)
    pbuf = _GRAD_PTR_BUF_CACHE.get(pbuf_key)
    if pbuf is None:
        h0 = torch.empty((n_tensors,), dtype=torch.int64, pin_memory=True)
        h1 = torch.empty((n_tensors,), dtype=torch.int64, pin_memory=True)
        d0 = torch.empty((n_tensors,), dtype=torch.int64, device=dev0)
        d1 = torch.empty((n_tensors,), dtype=torch.int64, device=dev0)
        pbuf = [
            (h0, d0, h0.numpy().view(np.uint64), None),
            (h1, d1, h1.numpy().view(np.uint64), None),
            0,
        ]
        if len(_GRAD_PTR_BUF_CACHE) > 16:
            _GRAD_PTR_BUF_CACHE.clear()
        _GRAD_PTR_BUF_CACHE[pbuf_key] = pbuf

    slot = pbuf[2]
    host_t, dev_t, host_np, prev_ev = pbuf[slot]
    if prev_ev is not None:
        prev_ev.synchronize()
    host_np[:] = grads_local
    dev_t.copy_(host_t, non_blocking=True)
    st = torch.musa.current_stream(dev0)
    ev_copy = torch.musa.Event()
    ev_copy.record(st)
    pbuf[slot] = (host_t, dev_t, host_np, ev_copy)
    pbuf[2] = 1 - slot

    sc_key = (dev_idx, n_chunks)
    sc = _NORM_SCRATCH_CACHE.get(sc_key)
    if sc is None:
        sc = (
            torch.empty((n_chunks,), dtype=torch.float32, device=dev0),
            torch.empty((1,), dtype=torch.float32, device=dev0),
            torch.empty((1,), dtype=torch.float32, device=dev0),
            torch.empty((1,), dtype=torch.float32, device=dev0),
        )
        if len(_NORM_SCRATCH_CACHE) > 8:
            _NORM_SCRATCH_CACHE.clear()
        _NORM_SCRATCH_CACHE[sc_key] = sc
    block_sums, sq_buf, clip_buf, norm_buf = sc

    lib = _load_lib()
    rc = lib.launch_fused_l2_norm_sq_f32(
        dev_t.data_ptr(), chunks_dev.data_ptr(), block_sums.data_ptr(), sq_buf.data_ptr(),
        n_chunks, st.musa_stream,
    )
    if rc != 0:
        raise RuntimeError(f"fused_l2_norm_sq_f32 failed rc={rc}")

    if is_dtensor and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(sq_buf, op=torch.distributed.ReduceOp.SUM, group=process_group)

    rc = lib.launch_compute_clip_coef_f32(
        sq_buf.data_ptr(), clip_buf.data_ptr(), norm_buf.data_ptr(),
        float(max_norm), st.musa_stream,
    )
    if rc != 0:
        raise RuntimeError(f"compute_clip_coef_f32 failed rc={rc}")
    return norm_buf.squeeze(0), clip_buf


def fused_multi_tensor_adamw_step(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    grad_scale_tensor: torch.Tensor | None = None,
) -> None:
    if not params:
        return
    step_val = int(state_steps[0].item()) + 1
    for st in state_steps:
        st.add_(1)

    bc1 = 1.0 - (beta1 ** step_val)
    bc2 = 1.0 - (beta2 ** step_val)
    lr_f = float(lr.item()) if isinstance(lr, torch.Tensor) else float(lr)
    step_size = lr_f / bc1
    inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)
    has_wd = 1 if weight_decay != 0.0 else 0
    wd_factor = 1.0 - lr_f * float(weight_decay)

    dev = _unwrap_local(params[0]).device
    dev_idx = dev.index or 0

    numels = []
    ptrs_list = []
    for p, g, m, v in zip(params, grads, exp_avgs, exp_avg_sqs):
        pl, gl, ml, vl = _unwrap_local(p), _unwrap_local(g), _unwrap_local(m), _unwrap_local(v)
        n = pl.numel()
        if n > 0:
            numels.append(n)
            ptrs_list.append((pl.data_ptr(), gl.data_ptr(), ml.data_ptr(), vl.data_ptr()))

    n_tensors = len(numels)
    if n_tensors == 0:
        return

    chunks_dev, n_chunks = _get_chunks(dev, dev_idx, numels)

    # Reuse double-buffered pinned CPU + GPU pointer staging buffer (4 * n_tensors uint64 = ~27 KB)
    # Double-buffering eliminates any host-device sync (`non_blocking=True` safely!)
    pbuf_key = (dev_idx, n_tensors)
    pbuf = _PTR_BUF_CACHE.get(pbuf_key)
    if pbuf is None:
        h0 = torch.empty((n_tensors, 4), dtype=torch.int64, pin_memory=True)
        h1 = torch.empty((n_tensors, 4), dtype=torch.int64, pin_memory=True)
        d0 = torch.empty((n_tensors, 4), dtype=torch.int64, device=dev)
        d1 = torch.empty((n_tensors, 4), dtype=torch.int64, device=dev)
        pbuf = [
            (h0, d0, h0.numpy().view(np.uint64), None),
            (h1, d1, h1.numpy().view(np.uint64), None),
            0,
        ]
        if len(_PTR_BUF_CACHE) > 16:
            _PTR_BUF_CACHE.clear()
        _PTR_BUF_CACHE[pbuf_key] = pbuf

    slot = pbuf[2]
    host_t, dev_t, host_np, prev_ev = pbuf[slot]
    if prev_ev is not None:
        prev_ev.synchronize()
    host_np[:] = ptrs_list
    dev_t.copy_(host_t, non_blocking=True)
    ev_copy = torch.musa.Event()
    ev_copy.record(torch.musa.current_stream(dev))
    pbuf[slot] = (host_t, dev_t, host_np, ev_copy)
    pbuf[2] = 1 - slot

    gs_ptr = grad_scale_tensor.data_ptr() if grad_scale_tensor is not None else 0
    rc = _load_lib().launch_fused_multi_tensor_adamw_f32(
        dev_t.data_ptr(), chunks_dev.data_ptr(), n_chunks,
        float(beta1), float(beta2),
        wd_factor, step_size, inv_sqrt_bc2, float(eps), has_wd,
        gs_ptr, torch.musa.current_stream(dev).musa_stream,
    )
    if rc != 0:
        raise RuntimeError(f"fused_multi_tensor_adamw_f32 failed rc={rc}")
