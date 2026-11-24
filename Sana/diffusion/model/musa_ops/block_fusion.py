"""MUSA C++ fused block-level ops for SANA-WM Stage-1 on MTT S5000.

Contains 6 deterministic (zero atomicAdd) fused autograd ops:
  1. fused_rmsnorm(x, weight, eps)                  -- drop-in RMSNorm (BF16, FP32 internal)
  2. fused_layernorm_modulate(x, shift, scale, eps) -- LayerNorm(no affine) + t2i_modulate
  3. fused_gate_residual(x, y, gate)                -- x + gate * y with frame-wise gate
  4. fused_silu_mul(out, g)                         -- out * silu(g) in FP32 -> BF16
  5. fused_softmax_main_prep(...)                   -- Softmax main RMSNorm + RoPE -> (B, H, N, D) BF16
  6. fused_softmax_cam_prep(...) & apply_o          -- Softmax cam RMSNorm + 4x4 UCPE + RoPE -> (B, H, N, D) BF16
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import threading

import torch
import torch.nn.functional as F

_MUSA_SRC = r"""
#include <musa_runtime.h>
#include <musa_bf16.h>
#include <math.h>
#include <stdint.h>

static __device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

static __device__ __forceinline__ float block_sum_256(float v, float* s_red, int tid) {
    v = warp_sum(v);
    if ((tid & 31) == 0) s_red[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_red[0] = tot;
    }
    __syncthreads();
    return s_red[0];
}

// =========================================================================
// 1. Fused RMSNorm (BF16 in/out, FP32 weight & stats, zero atomicAdd)
// =========================================================================
static __device__ __forceinline__ float sigmoid_tanh_bf(float x) {
    return 0.5f * (1.0f + tanhf(0.5f * x));
}

// Fast zero-SMEM uint4 288t kernels for C % 8 == 0 && C <= 2304 (e.g. C=2240 -> C8=280, 9 warps=288t)
extern "C" __global__ void rmsnorm_bwd_u4_bf16(
    const __mt_bfloat16* __restrict__ x,
    const __mt_bfloat16* __restrict__ dy,
    const float* __restrict__ w,
    const float* __restrict__ inv_rms_in,
    __mt_bfloat16* __restrict__ dx,
    float* __restrict__ dw_part,
    int M, int C, int n_chunks
) {
    const int ch = blockIdx.x, tid = threadIdx.x, C8 = C >> 3;
    const int n_warps = blockDim.x >> 5;
    const bool active = (tid < C8);
    const int c0 = tid << 3;
    float wv[8], dw[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { dw[i] = 0.0f; wv[i] = active ? w[c0 + i] : 0.0f; }
    const int m0 = (static_cast<int64_t>(ch) * M) / n_chunks;
    const int m1 = (static_cast<int64_t>(ch + 1) * M) / n_chunks;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    const uint4* dy4 = reinterpret_cast<const uint4*>(dy);
    uint4* dx4 = reinterpret_cast<uint4*>(dx);
    __shared__ float s1[16];
    const float invC = 1.0f / static_cast<float>(C);

    for (int m = m0; m < m1; ++m) {
        const float ir = inv_rms_in[m];
        float xf[8], df[8];
        float dot = 0.0f;
        if (active) {
            uint4 xv = x4[static_cast<int64_t>(m) * C8 + tid];
            uint4 dv = dy4[static_cast<int64_t>(m) * C8 + tid];
            const __mt_bfloat162* xh = reinterpret_cast<const __mt_bfloat162*>(&xv);
            const __mt_bfloat162* dh = reinterpret_cast<const __mt_bfloat162*>(&dv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float a0 = __low2float(xh[k]), a1 = __high2float(xh[k]);
                float b0 = __low2float(dh[k]), b1 = __high2float(dh[k]);
                xf[2*k] = a0; xf[2*k+1] = a1; df[2*k] = b0; df[2*k+1] = b1;
                dw[2*k]   = fmaf(b0 * a0, ir, dw[2*k]);
                dw[2*k+1] = fmaf(b1 * a1, ir, dw[2*k+1]);
                dot = fmaf(b0 * wv[2*k], a0, fmaf(b1 * wv[2*k+1], a1, dot));
            }
        }
        dot = warp_sum(dot);
        if ((tid & 31) == 0) s1[tid >> 5] = dot;
        __syncthreads();
        if (tid == 0) {
            float t = 0.0f;
            for (int i = 0; i < n_warps; ++i) t += s1[i];
            s1[0] = t * invC * ir * ir;
        }
        __syncthreads();
        if (active) {
            const float coeff = s1[0];
            uint4 out;
            __mt_bfloat162* oh = reinterpret_cast<__mt_bfloat162*>(&out);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                oh[k] = __floats2bfloat162_rn(
                    ir * (df[2*k]   * wv[2*k]   - xf[2*k]   * coeff),
                    ir * (df[2*k+1] * wv[2*k+1] - xf[2*k+1] * coeff)
                );
            }
            dx4[static_cast<int64_t>(m) * C8 + tid] = out;
        }
    }
    if (active) {
        int64_t base = static_cast<int64_t>(ch) * C + c0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dw_part[base + i] = dw[i];
    }
}

extern "C" __global__ void recompute_ln_mod_u4_bf16(
    const __mt_bfloat16* __restrict__ x,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    __mt_bfloat16* __restrict__ y,
    int BT, int S, int C, int n_splits
) {
    const int bt = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x, C8 = C >> 3;
    if (tid >= C8) return;
    const int c0 = tid << 3;
    const float* shp = shift + static_cast<int64_t>(bt) * C + c0;
    const float* scp = scale + static_cast<int64_t>(bt) * C + c0;
    float sh[8], sc1[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { sh[i] = shp[i]; sc1[i] = 1.0f + scp[i]; }
    const int s0 = (static_cast<int64_t>(sp) * S) / n_splits;
    const int s1 = (static_cast<int64_t>(sp + 1) * S) / n_splits;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    uint4* y4 = reinterpret_cast<uint4*>(y);
    for (int s = s0; s < s1; ++s) {
        int m = bt * S + s;
        float mu = mean_in[m], rs = rstd_in[m];
        int64_t idx = static_cast<int64_t>(m) * C8 + tid;
        uint4 xv = x4[idx];
        const __mt_bfloat162* xh = reinterpret_cast<const __mt_bfloat162*>(&xv);
        uint4 out;
        __mt_bfloat162* oh = reinterpret_cast<__mt_bfloat162*>(&out);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            oh[k] = __floats2bfloat162_rn(
                fmaf((__low2float(xh[k])  - mu) * rs, sc1[2*k],   sh[2*k]),
                fmaf((__high2float(xh[k]) - mu) * rs, sc1[2*k+1], sh[2*k+1])
            );
        }
        y4[idx] = out;
    }
}

extern "C" __global__ void frame_mean_part_u4_bf16(
    const __mt_bfloat16* __restrict__ x,
    float* __restrict__ part_out,
    int BT, int S, int C, int n_splits
) {
    const int bt = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x, C8 = C >> 3;
    if (tid >= C8) return;
    const int c0 = tid << 3;
    const int s0 = (static_cast<int64_t>(sp) * S) / n_splits;
    const int s1 = (static_cast<int64_t>(sp + 1) * S) / n_splits;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s = s0; s < s1; ++s) {
        int64_t idx = static_cast<int64_t>(bt * S + s) * C8 + tid;
        uint4 xv = x4[idx];
        const __mt_bfloat162* xh = reinterpret_cast<const __mt_bfloat162*>(&xv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc[2*k]   += __low2float(xh[k]);
            acc[2*k+1] += __high2float(xh[k]);
        }
    }
    int64_t base = (static_cast<int64_t>(bt) * n_splits + sp) * C + c0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) part_out[base + i] = acc[i];
}

extern "C" __global__ void reduce_frame_mean_bf16(
    const float* __restrict__ part,
    __mt_bfloat16* __restrict__ out,
    int BT, int n_splits, int C, float invS
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (c >= C) return;
    float s = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        s += part[(static_cast<int64_t>(bt) * n_splits + sp) * C + c];
    }
    out[static_cast<int64_t>(bt) * C + c] = __float2bfloat16_rn(s * invS);
}

extern "C" __global__ void ln_modulate_fwd_u4_bf16(
    const __mt_bfloat16* __restrict__ x,
    const __mt_bfloat16* __restrict__ b_add,  // nullptr for plain ln_modulate, non-null for fused (a+b)
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    __mt_bfloat16* __restrict__ s_out,        // nullptr if b_add == nullptr
    __mt_bfloat16* __restrict__ y,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    float* __restrict__ y_sum_part,           // optional (BT, n_splits, C) sum of y for frame mean
    int BT, int S, int C, int n_splits, float eps
) {
    const int bt = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x, C8 = C >> 3;
    const int n_warps = blockDim.x >> 5;
    const bool active = (tid < C8);
    const int c0 = tid << 3;
    float sh[8], sc1[8], ysum[8];
    if (active) {
        const float* shp = shift + static_cast<int64_t>(bt) * C + c0;
        const float* scp = scale + static_cast<int64_t>(bt) * C + c0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { sh[i] = shp[i]; sc1[i] = 1.0f + scp[i]; ysum[i] = 0.0f; }
    }
    const int s0 = (static_cast<int64_t>(sp) * S) / n_splits;
    const int s1 = (static_cast<int64_t>(sp + 1) * S) / n_splits;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    const uint4* b4 = reinterpret_cast<const uint4*>(b_add);
    uint4* s4 = reinterpret_cast<uint4*>(s_out);
    uint4* y4 = reinterpret_cast<uint4*>(y);
    __shared__ float sr1[16], sr2[16];
    const float invC = 1.0f / static_cast<float>(C);
    const bool do_ysum = (y_sum_part != nullptr);

    for (int s = s0; s < s1; ++s) {
        int m = bt * S + s;
        float v[8], sum = 0.0f, sum2 = 0.0f;
        if (active) {
            int64_t idx = static_cast<int64_t>(m) * C8 + tid;
            uint4 xv = x4[idx];
            if (b4 != nullptr) {
                uint4 bv = b4[idx];
                const __mt_bfloat162* xh = reinterpret_cast<const __mt_bfloat162*>(&xv);
                const __mt_bfloat162* bh = reinterpret_cast<const __mt_bfloat162*>(&bv);
                uint4 sv;
                __mt_bfloat162* sh2 = reinterpret_cast<__mt_bfloat162*>(&sv);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    __mt_bfloat162 s2 = __floats2bfloat162_rn(
                        __low2float(xh[k])  + __low2float(bh[k]),
                        __high2float(xh[k]) + __high2float(bh[k])
                    );
                    sh2[k] = s2;
                    float a = __low2float(s2), b = __high2float(s2);
                    v[2*k] = a; v[2*k+1] = b;
                    sum += a + b; sum2 = fmaf(a, a, fmaf(b, b, sum2));
                }
                s4[idx] = sv;
            } else {
                const __mt_bfloat162* xh = reinterpret_cast<const __mt_bfloat162*>(&xv);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float a = __low2float(xh[k]), b = __high2float(xh[k]);
                    v[2*k] = a; v[2*k+1] = b;
                    sum += a + b; sum2 = fmaf(a, a, fmaf(b, b, sum2));
                }
            }
        }
        sum = warp_sum(sum); sum2 = warp_sum(sum2);
        if ((tid & 31) == 0) { sr1[tid >> 5] = sum; sr2[tid >> 5] = sum2; }
        __syncthreads();
        if (tid == 0) {
            float ts = 0.0f, ts2 = 0.0f;
            for (int w = 0; w < n_warps; ++w) { ts += sr1[w]; ts2 += sr2[w]; }
            float mu = ts * invC;
            float rs = rsqrtf(fmaxf(ts2 * invC - mu * mu, 0.0f) + eps);
            sr1[0] = mu; sr2[0] = rs;
            mean_out[m] = mu; rstd_out[m] = rs;
        }
        __syncthreads();
        if (active) {
            const float mu = sr1[0], rs = sr2[0];
            uint4 out;
            __mt_bfloat162* oh = reinterpret_cast<__mt_bfloat162*>(&out);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                __mt_bfloat162 y2 = __floats2bfloat162_rn(
                    fmaf((v[2*k]   - mu) * rs, sc1[2*k],   sh[2*k]),
                    fmaf((v[2*k+1] - mu) * rs, sc1[2*k+1], sh[2*k+1])
                );
                oh[k] = y2;
                if (do_ysum) {
                    ysum[2*k]   += __low2float(y2);
                    ysum[2*k+1] += __high2float(y2);
                }
            }
            y4[static_cast<int64_t>(m) * C8 + tid] = out;
        }
    }
    if (do_ysum && active) {
        int64_t base = (static_cast<int64_t>(bt) * n_splits + sp) * C + c0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) y_sum_part[base + i] = ysum[i];
    }
}

extern "C" __global__ void ln_modulate_bwd_u4_bf16(
    const __mt_bfloat16* __restrict__ x,
    const __mt_bfloat16* __restrict__ dy,
    const __mt_bfloat16* __restrict__ ds_res,  // optional residual grad added into dx!
    const float* __restrict__ dy_frame_add,    // optional (B*T_frame, C) fp32 grad/S_frame added into dy!
    const float* __restrict__ scale,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    __mt_bfloat16* __restrict__ dx,
    float* __restrict__ dsh_part,
    float* __restrict__ dsc_part,
    int BT, int S, int C, int n_splits, int S_frame
) {
    const int bt = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x, C8 = C >> 3;
    const int n_warps = blockDim.x >> 5;
    const bool active = (tid < C8);
    const int c0 = tid << 3;
    const float* sc = scale + static_cast<int64_t>(bt) * C + c0;
    float sp1[8], dsh[8], dsc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { dsh[i] = 0.0f; dsc[i] = 0.0f; sp1[i] = active ? (1.0f + sc[i]) : 0.0f; }
    const bool has_dfa = (dy_frame_add != nullptr);
    const bool const_frame = (has_dfa && S_frame == S);
    float dfa_c[8];
    if (const_frame && active) {
        const float* fp = dy_frame_add + static_cast<int64_t>(bt) * C + c0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dfa_c[i] = fp[i];
    }

    const int s0 = (static_cast<int64_t>(sp) * S) / n_splits;
    const int s1 = (static_cast<int64_t>(sp + 1) * S) / n_splits;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    const uint4* dy4 = reinterpret_cast<const uint4*>(dy);
    const uint4* dr4 = reinterpret_cast<const uint4*>(ds_res);
    uint4* dx4 = reinterpret_cast<uint4*>(dx);
    __shared__ float s_r1[16], s_r2[16];
    const float invC = 1.0f / static_cast<float>(C);

    for (int s = s0; s < s1; ++s) {
        int m = bt * S + s;
        const float mu = mean_in[m], rs = rstd_in[m];
        float xh[8], g[8];
        float sum_g = 0.0f, sum_gxh = 0.0f;
        if (active) {
            int64_t idx = static_cast<int64_t>(m) * C8 + tid;
            uint4 xv = x4[idx], dv = dy4[idx];
            const __mt_bfloat162* xh2 = reinterpret_cast<const __mt_bfloat162*>(&xv);
            const __mt_bfloat162* dh2 = reinterpret_cast<const __mt_bfloat162*>(&dv);
            const float* dfa_p = (has_dfa && !const_frame)
                ? (dy_frame_add + static_cast<int64_t>(m / S_frame) * C + c0)
                : nullptr;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float a0 = (__low2float(xh2[k]) - mu) * rs, a1 = (__high2float(xh2[k]) - mu) * rs;
                float b0 = __low2float(dh2[k]), b1 = __high2float(dh2[k]);
                if (const_frame) { b0 += dfa_c[2*k]; b1 += dfa_c[2*k+1]; }
                else if (dfa_p != nullptr) { b0 += dfa_p[2*k]; b1 += dfa_p[2*k+1]; }
                xh[2*k] = a0; xh[2*k+1] = a1;
                dsh[2*k] += b0; dsh[2*k+1] += b1;
                dsc[2*k] = fmaf(b0, a0, dsc[2*k]); dsc[2*k+1] = fmaf(b1, a1, dsc[2*k+1]);
                float g0 = b0 * sp1[2*k], g1 = b1 * sp1[2*k+1];
                g[2*k] = g0; g[2*k+1] = g1;
                sum_g += g0 + g1;
                sum_gxh = fmaf(g0, a0, fmaf(g1, a1, sum_gxh));
            }
        }
        sum_g = warp_sum(sum_g); sum_gxh = warp_sum(sum_gxh);
        if ((tid & 31) == 0) { s_r1[tid >> 5] = sum_g; s_r2[tid >> 5] = sum_gxh; }
        __syncthreads();
        if (tid == 0) {
            float sg = 0.0f, sgx = 0.0f;
            for (int w = 0; w < n_warps; ++w) { sg += s_r1[w]; sgx += s_r2[w]; }
            s_r1[0] = sg * invC; s_r2[0] = sgx * invC;
        }
        __syncthreads();
        if (active) {
            const float mg = s_r1[0], mgx = s_r2[0];
            int64_t idx = static_cast<int64_t>(m) * C8 + tid;
            uint4 out;
            __mt_bfloat162* oh2 = reinterpret_cast<__mt_bfloat162*>(&out);
            if (dr4 != nullptr) {
                uint4 rv = dr4[idx];
                const __mt_bfloat162* rh2 = reinterpret_cast<const __mt_bfloat162*>(&rv);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float d0 = rs * (g[2*k]   - mg - xh[2*k]   * mgx) + __low2float(rh2[k]);
                    float d1 = rs * (g[2*k+1] - mg - xh[2*k+1] * mgx) + __high2float(rh2[k]);
                    oh2[k] = __floats2bfloat162_rn(d0, d1);
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    oh2[k] = __floats2bfloat162_rn(
                        rs * (g[2*k]   - mg - xh[2*k]   * mgx),
                        rs * (g[2*k+1] - mg - xh[2*k+1] * mgx)
                    );
                }
            }
            dx4[idx] = out;
        }
    }
    if (active) {
        int64_t base = (static_cast<int64_t>(bt) * n_splits + sp) * C + c0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { dsh_part[base + i] = dsh[i]; dsc_part[base + i] = dsc[i]; }
    }
}

extern "C" __global__ void rmsnorm_fwd_u4_bf16(
    const __mt_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    __mt_bfloat16* __restrict__ y,
    float* __restrict__ inv_rms_out,
    int M, int C, int n_chunks, float eps
) {
    const int ch = blockIdx.x, tid = threadIdx.x, C8 = C >> 3;
    const int n_warps = blockDim.x >> 5;
    const bool active = (tid < C8);
    const int c0 = tid << 3;
    float wv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) wv[i] = active ? w[c0 + i] : 0.0f;
    const int m0 = (static_cast<int64_t>(ch) * M) / n_chunks;
    const int m1 = (static_cast<int64_t>(ch + 1) * M) / n_chunks;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    uint4* y4 = reinterpret_cast<uint4*>(y);
    __shared__ float s1[16];
    const float invC = 1.0f / static_cast<float>(C);

    for (int m = m0; m < m1; ++m) {
        float xf[8];
        float sq = 0.0f;
        if (active) {
            uint4 xv = x4[static_cast<int64_t>(m) * C8 + tid];
            const __mt_bfloat162* xh = reinterpret_cast<const __mt_bfloat162*>(&xv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float a0 = __low2float(xh[k]), a1 = __high2float(xh[k]);
                xf[2*k] = a0; xf[2*k+1] = a1;
                sq = fmaf(a0, a0, fmaf(a1, a1, sq));
            }
        }
        sq = warp_sum(sq);
        if ((tid & 31) == 0) s1[tid >> 5] = sq;
        __syncthreads();
        if (tid == 0) {
            float t = 0.0f;
            for (int i = 0; i < n_warps; ++i) t += s1[i];
            float ir = rsqrtf(t * invC + eps);
            s1[0] = ir;
            inv_rms_out[m] = ir;
        }
        __syncthreads();
        if (active) {
            const float ir = s1[0];
            uint4 out;
            __mt_bfloat162* oh = reinterpret_cast<__mt_bfloat162*>(&out);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                oh[k] = __floats2bfloat162_rn(
                    xf[2*k]   * ir * wv[2*k],
                    xf[2*k+1] * ir * wv[2*k+1]
                );
            }
            y4[static_cast<int64_t>(m) * C8 + tid] = out;
        }
    }
}

extern "C" __global__ void gate_residual_fwd_u4_bf16(
    const __mt_bfloat16* __restrict__ x,
    const __mt_bfloat16* __restrict__ y,
    const float* __restrict__ gate,
    const __mt_bfloat16* __restrict__ add_bias,
    __mt_bfloat16* __restrict__ out,
    int BT, int S, int C, int n_splits
) {
    const int bt = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x, C8 = C >> 3;
    if (tid >= C8) return;
    const int c0 = tid << 3;
    float g[8], bv[8];
    const float* gp = gate + static_cast<int64_t>(bt) * C + c0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) g[i] = gp[i];
    const bool has_b = (add_bias != nullptr);
    if (has_b) {
        uint4 b4 = reinterpret_cast<const uint4*>(add_bias)[tid];
        const __mt_bfloat162* bh = reinterpret_cast<const __mt_bfloat162*>(&b4);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            bv[2*k]   = __low2float(bh[k]);
            bv[2*k+1] = __high2float(bh[k]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) bv[i] = 0.0f;
    }
    const int s0 = (static_cast<int64_t>(sp) * S) / n_splits;
    const int s1 = (static_cast<int64_t>(sp + 1) * S) / n_splits;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    const uint4* y4 = reinterpret_cast<const uint4*>(y);
    uint4* o4 = reinterpret_cast<uint4*>(out);
    for (int s = s0; s < s1; ++s) {
        int64_t idx = static_cast<int64_t>(bt * S + s) * C8 + tid;
        uint4 xv = x4[idx], yv = y4[idx];
        const __mt_bfloat162* xh = reinterpret_cast<const __mt_bfloat162*>(&xv);
        const __mt_bfloat162* yh = reinterpret_cast<const __mt_bfloat162*>(&yv);
        uint4 ov;
        __mt_bfloat162* oh = reinterpret_cast<__mt_bfloat162*>(&ov);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float x0 = __low2float(xh[k])  + bv[2*k];
            float x1 = __high2float(xh[k]) + bv[2*k+1];
            oh[k] = __floats2bfloat162_rn(
                fmaf(g[2*k],   __low2float(yh[k]),  x0),
                fmaf(g[2*k+1], __high2float(yh[k]), x1)
            );
        }
        o4[idx] = ov;
    }
}

extern "C" __global__ void gate_residual_bwd_u4_bf16(
    const __mt_bfloat16* __restrict__ dout,
    const __mt_bfloat16* __restrict__ y,
    const float* __restrict__ gate,
    __mt_bfloat16* __restrict__ dy,
    float* __restrict__ dgate_part,
    float* __restrict__ dbias_part,
    int BT, int S, int C, int n_splits
) {
    const int bt = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x, C8 = C >> 3;
    if (tid >= C8) return;
    const int c0 = tid << 3;
    const float* gp = gate + static_cast<int64_t>(bt) * C + c0;
    const bool has_b = (dbias_part != nullptr);
    float g[8], dg[8], db[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { g[i] = gp[i]; dg[i] = 0.0f; db[i] = 0.0f; }
    const int s0 = (static_cast<int64_t>(sp) * S) / n_splits;
    const int s1 = (static_cast<int64_t>(sp + 1) * S) / n_splits;
    const uint4* do4 = reinterpret_cast<const uint4*>(dout);
    const uint4* y4  = reinterpret_cast<const uint4*>(y);
    uint4* dy4 = reinterpret_cast<uint4*>(dy);
    for (int s = s0; s < s1; ++s) {
        int64_t idx = static_cast<int64_t>(bt * S + s) * C8 + tid;
        uint4 dv = do4[idx], yv = y4[idx];
        const __mt_bfloat162* dh = reinterpret_cast<const __mt_bfloat162*>(&dv);
        const __mt_bfloat162* yh = reinterpret_cast<const __mt_bfloat162*>(&yv);
        uint4 ov;
        __mt_bfloat162* oh = reinterpret_cast<__mt_bfloat162*>(&ov);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float d0 = __low2float(dh[k]), d1 = __high2float(dh[k]);
            dg[2*k]   = fmaf(d0, __low2float(yh[k]),  dg[2*k]);
            dg[2*k+1] = fmaf(d1, __high2float(yh[k]), dg[2*k+1]);
            if (has_b) {
                db[2*k]   += d0;
                db[2*k+1] += d1;
            }
            oh[k] = __floats2bfloat162_rn(d0 * g[2*k], d1 * g[2*k+1]);
        }
        dy4[idx] = ov;
    }
    int64_t base = (static_cast<int64_t>(bt) * n_splits + sp) * C + c0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) dgate_part[base + i] = dg[i];
    if (has_b) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) dbias_part[base + i] = db[i];
    }
}

extern "C" __global__ void rmsnorm_fwd_bf16(
    const __mt_bfloat162* __restrict__ x2,
    const float* __restrict__ w,
    __mt_bfloat162* __restrict__ y2,
    float* __restrict__ inv_rms_out,
    int M, int C, float eps
) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int C2 = C >> 1;
    const __mt_bfloat162* xr = x2 + static_cast<int64_t>(m) * C2;
    __mt_bfloat162* yr = y2 + static_cast<int64_t>(m) * C2;

    __shared__ float s_red[8];
    float sq = 0.0f;
    for (int p = tid; p < C2; p += 256) {
        __mt_bfloat162 xv = xr[p];
        float x0 = __low2float(xv), x1 = __high2float(xv);
        sq = fmaf(x0, x0, fmaf(x1, x1, sq));
    }
    float tot = block_sum_256(sq, s_red, tid);
    float ir = rsqrtf(tot / static_cast<float>(C) + eps);
    if (tid == 0) inv_rms_out[m] = ir;

    for (int p = tid; p < C2; p += 256) {
        int c0 = p << 1;
        __mt_bfloat162 xv = xr[p];
        float y0 = __low2float(xv)  * ir * w[c0];
        float y1 = __high2float(xv) * ir * w[c0 + 1];
        yr[p] = __floats2bfloat162_rn(y0, y1);
    }
}

extern "C" __global__ void rmsnorm_bwd_bf16(
    const __mt_bfloat162* __restrict__ x2,
    const __mt_bfloat162* __restrict__ dy2,
    const float* __restrict__ w,
    const float* __restrict__ inv_rms_in,
    __mt_bfloat162* __restrict__ dx2,
    float* __restrict__ dw_part,  // (n_chunks, C)
    int M, int C, int n_chunks
) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int C2 = C >> 1;
    const int m_start = (static_cast<int64_t>(chunk) * M) / n_chunks;
    const int m_end   = (static_cast<int64_t>(chunk + 1) * M) / n_chunks;

    extern __shared__ float smem[];  // C + 8 floats + 2*C2 bf162 (2*C floats total)
    float* s_dw = smem;
    float* s_red = smem + C;
    __mt_bfloat162* s_x2  = reinterpret_cast<__mt_bfloat162*>(s_red + 8);
    __mt_bfloat162* s_dy2 = s_x2 + C2;

    for (int c = tid; c < C; c += 256) s_dw[c] = 0.0f;
    __syncthreads();

    for (int m = m_start; m < m_end; ++m) {
        const __mt_bfloat162* xr = x2 + static_cast<int64_t>(m) * C2;
        const __mt_bfloat162* dyr = dy2 + static_cast<int64_t>(m) * C2;
        __mt_bfloat162* dxr = dx2 + static_cast<int64_t>(m) * C2;
        const float ir = inv_rms_in[m];

        float dot = 0.0f;
        for (int p = tid; p < C2; p += 256) {
            int c0 = p << 1;
            __mt_bfloat162 xv = xr[p], dv = dyr[p];
            s_x2[p] = xv;
            s_dy2[p] = dv;
            float x0 = __low2float(xv), x1 = __high2float(xv);
            float g0 = __low2float(dv), g1 = __high2float(dv);
            s_dw[c0]     += g0 * x0 * ir;
            s_dw[c0 + 1] += g1 * x1 * ir;
            dot = fmaf(g0 * w[c0], x0, fmaf(g1 * w[c0 + 1], x1, dot));
        }
        float tot_dot = block_sum_256(dot, s_red, tid);
        float coeff = tot_dot * ir * ir / static_cast<float>(C);

        for (int p = tid; p < C2; p += 256) {
            int c0 = p << 1;
            __mt_bfloat162 xv = s_x2[p], dv = s_dy2[p];
            float x0 = __low2float(xv), x1 = __high2float(xv);
            float g0 = __low2float(dv), g1 = __high2float(dv);
            float dx0 = ir * (g0 * w[c0]     - x0 * coeff);
            float dx1 = ir * (g1 * w[c0 + 1] - x1 * coeff);
            dxr[p] = __floats2bfloat162_rn(dx0, dx1);
        }
    }
    __syncthreads();

    int64_t base = static_cast<int64_t>(chunk) * C;
    for (int c = tid; c < C; c += 256) dw_part[base + c] = s_dw[c];
}

extern "C" __global__ void reduce_chunks_1(
    const float* __restrict__ part, float* __restrict__ out, int n_chunks, int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s = 0.0f;
    for (int k = 0; k < n_chunks; ++k) s += part[static_cast<int64_t>(k) * C + c];
    out[c] = s;
}

extern "C" __global__ void reduce_chunks_2(
    const float* __restrict__ p1, const float* __restrict__ p2,
    float* __restrict__ o1, float* __restrict__ o2, int n_chunks, int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s1 = 0.0f, s2 = 0.0f;
    for (int k = 0; k < n_chunks; ++k) {
        int64_t idx = static_cast<int64_t>(k) * C + c;
        s1 += p1[idx];
        s2 += p2[idx];
    }
    o1[c] = s1;
    o2[c] = s2;
}

// =========================================================================
// 2. Fused LayerNorm(no affine) + t2i_modulate(shift, scale)
//    x: (BT, S, C) BF16; shift, scale: (BT, C) FP32 -> y: (BT, S, C) BF16
// =========================================================================
extern "C" __global__ void ln_modulate_fwd_bf16(
    const __mt_bfloat162* __restrict__ x2,
    const float* __restrict__ shift,  // (BT, C)
    const float* __restrict__ scale,  // (BT, C)
    __mt_bfloat162* __restrict__ y2,
    float* __restrict__ mean_out,     // (BT * S,)
    float* __restrict__ rstd_out,     // (BT * S,)
    int BT, int S, int C, float eps
) {
    const int m = blockIdx.x;  // 0 .. BT*S - 1
    const int bt = m / S;
    const int tid = threadIdx.x;
    const int C2 = C >> 1;

    const __mt_bfloat162* xr = x2 + static_cast<int64_t>(m) * C2;
    __mt_bfloat162* yr = y2 + static_cast<int64_t>(m) * C2;
    const float* sh = shift + static_cast<int64_t>(bt) * C;
    const float* sc = scale + static_cast<int64_t>(bt) * C;

    __shared__ float s_red1[8], s_red2[8];
    float sum_x = 0.0f, sum_x2 = 0.0f;
    for (int p = tid; p < C2; p += 256) {
        __mt_bfloat162 xv = xr[p];
        float x0 = __low2float(xv), x1 = __high2float(xv);
        sum_x  += x0 + x1;
        sum_x2 = fmaf(x0, x0, fmaf(x1, x1, sum_x2));
    }
    sum_x = warp_sum(sum_x);
    sum_x2 = warp_sum(sum_x2);
    if ((tid & 31) == 0) { s_red1[tid >> 5] = sum_x; s_red2[tid >> 5] = sum_x2; }
    __syncthreads();
    if (tid == 0) {
        float sx = 0.0f, sx2 = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { sx += s_red1[w]; sx2 += s_red2[w]; }
        float invC = 1.0f / static_cast<float>(C);
        float mu = sx * invC;
        float var = fmaxf(0.0f, sx2 * invC - mu * mu);
        float rs = rsqrtf(var + eps);
        s_red1[0] = mu;
        s_red2[0] = rs;
        mean_out[m] = mu;
        rstd_out[m] = rs;
    }
    __syncthreads();
    const float mu = s_red1[0], rs = s_red2[0];

    for (int p = tid; p < C2; p += 256) {
        int c0 = p << 1;
        __mt_bfloat162 xv = xr[p];
        float xh0 = (__low2float(xv)  - mu) * rs;
        float xh1 = (__high2float(xv) - mu) * rs;
        float y0 = fmaf(xh0, 1.0f + sc[c0],     sh[c0]);
        float y1 = fmaf(xh1, 1.0f + sc[c0 + 1], sh[c0 + 1]);
        yr[p] = __floats2bfloat162_rn(y0, y1);
    }
}

extern "C" __global__ void ln_modulate_bwd_bf16(
    const __mt_bfloat162* __restrict__ x2,
    const __mt_bfloat162* __restrict__ dy2,
    const float* __restrict__ scale,     // (BT, C)
    const float* __restrict__ mean_in,   // (BT * S,)
    const float* __restrict__ rstd_in,   // (BT * S,)
    __mt_bfloat162* __restrict__ dx2,
    float* __restrict__ dshift_part,     // (BT, n_splits, C)
    float* __restrict__ dscale_part,     // (BT, n_splits, C)
    int BT, int S, int C, int n_splits
) {
    const int bt = blockIdx.x;
    const int sp = blockIdx.y;
    const int tid = threadIdx.x;
    const int C2 = C >> 1;

    const int s_start = (sp * S) / n_splits;
    const int s_end   = ((sp + 1) * S) / n_splits;
    const float* sc = scale + static_cast<int64_t>(bt) * C;

    extern __shared__ float smem[];  // 2 * C + 16 floats + 2 * C2 bf162
    float* s_dsh = smem;
    float* s_dsc = smem + C;
    float* s_r1  = smem + 2 * C;
    float* s_r2  = s_r1 + 8;
    __mt_bfloat162* s_x2  = reinterpret_cast<__mt_bfloat162*>(s_r2 + 8);
    __mt_bfloat162* s_dy2 = s_x2 + C2;

    for (int c = tid; c < C; c += 256) { s_dsh[c] = 0.0f; s_dsc[c] = 0.0f; }
    __syncthreads();

    const float invC = 1.0f / static_cast<float>(C);

    for (int s = s_start; s < s_end; ++s) {
        int m = bt * S + s;
        const __mt_bfloat162* xr = x2 + static_cast<int64_t>(m) * C2;
        const __mt_bfloat162* dyr = dy2 + static_cast<int64_t>(m) * C2;
        __mt_bfloat162* dxr = dx2 + static_cast<int64_t>(m) * C2;
        const float mu = mean_in[m], rs = rstd_in[m];

        float sum_g = 0.0f, sum_gxh = 0.0f;
        for (int p = tid; p < C2; p += 256) {
            int c0 = p << 1;
            __mt_bfloat162 xv = xr[p], dv = dyr[p];
            s_x2[p] = xv;
            s_dy2[p] = dv;
            float xh0 = (__low2float(xv)  - mu) * rs;
            float xh1 = (__high2float(xv) - mu) * rs;
            float dy0 = __low2float(dv), dy1 = __high2float(dv);
            s_dsh[c0]     += dy0;
            s_dsh[c0 + 1] += dy1;
            s_dsc[c0]     += dy0 * xh0;
            s_dsc[c0 + 1] += dy1 * xh1;
            float g0 = dy0 * (1.0f + sc[c0]);
            float g1 = dy1 * (1.0f + sc[c0 + 1]);
            sum_g   += g0 + g1;
            sum_gxh = fmaf(g0, xh0, fmaf(g1, xh1, sum_gxh));
        }
        sum_g = warp_sum(sum_g);
        sum_gxh = warp_sum(sum_gxh);
        if ((tid & 31) == 0) { s_r1[tid >> 5] = sum_g; s_r2[tid >> 5] = sum_gxh; }
        __syncthreads();
        if (tid == 0) {
            float sg = 0.0f, sgx = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) { sg += s_r1[w]; sgx += s_r2[w]; }
            s_r1[0] = sg * invC;
            s_r2[0] = sgx * invC;
        }
        __syncthreads();
        const float mg = s_r1[0], mgx = s_r2[0];

        for (int p = tid; p < C2; p += 256) {
            int c0 = p << 1;
            __mt_bfloat162 xv = s_x2[p], dv = s_dy2[p];
            float xh0 = (__low2float(xv)  - mu) * rs;
            float xh1 = (__high2float(xv) - mu) * rs;
            float g0 = __low2float(dv)  * (1.0f + sc[c0]);
            float g1 = __high2float(dv) * (1.0f + sc[c0 + 1]);
            float dx0 = rs * (g0 - mg - xh0 * mgx);
            float dx1 = rs * (g1 - mg - xh1 * mgx);
            dxr[p] = __floats2bfloat162_rn(dx0, dx1);
        }
    }
    __syncthreads();

    int64_t base = (static_cast<int64_t>(bt) * n_splits + sp) * C;
    for (int c = tid; c < C; c += 256) {
        dshift_part[base + c] = s_dsh[c];
        dscale_part[base + c] = s_dsc[c];
    }
}

extern "C" __global__ void reduce_frame_splits_2(
    const float* __restrict__ p1, const float* __restrict__ p2,
    float* __restrict__ o1, float* __restrict__ o2,
    int BT, int n_splits, int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (c >= C) return;
    float s1 = 0.0f, s2 = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        int64_t idx = (static_cast<int64_t>(bt) * n_splits + sp) * C + c;
        s1 += p1[idx];
        s2 += p2[idx];
    }
    int64_t out_idx = static_cast<int64_t>(bt) * C + c;
    o1[out_idx] = s1;
    o2[out_idx] = s2;
}

// =========================================================================
// 3. Fused Gate + Residual: out = x + gate * y  (ZERO 64-bit div! 128-bit/fast row grid!)
//    Grid: (S, BT), Block: 256 -> zero division, 10x faster than 64-bit div!
// =========================================================================
extern "C" __global__ void gate_residual_fwd_bf16(
    const __mt_bfloat162* __restrict__ x2,
    const __mt_bfloat162* __restrict__ y2,
    const float* __restrict__ gate,
    __mt_bfloat162* __restrict__ out2,
    int BT, int S, int C
) {
    const int s  = blockIdx.x;
    const int bt = blockIdx.y;
    const int tid = threadIdx.x;
    const int C2 = C >> 1;
    const int64_t row_off = static_cast<int64_t>(bt * S + s) * C2;
    const __mt_bfloat162* xr = x2 + row_off;
    const __mt_bfloat162* yr = y2 + row_off;
    __mt_bfloat162* or_      = out2 + row_off;
    const float* g = gate + static_cast<int64_t>(bt) * C;

    for (int p = tid; p < C2; p += 256) {
        int c0 = p << 1;
        __mt_bfloat162 xv = xr[p], yv = yr[p];
        float o0 = fmaf(g[c0],     __low2float(yv),  __low2float(xv));
        float o1 = fmaf(g[c0 + 1], __high2float(yv), __high2float(xv));
        or_[p] = __floats2bfloat162_rn(o0, o1);
    }
}

extern "C" __global__ void gate_residual_bwd_bf16(
    const __mt_bfloat162* __restrict__ dout2,
    const __mt_bfloat162* __restrict__ y2,
    const float* __restrict__ gate,
    __mt_bfloat162* __restrict__ dy2,
    float* __restrict__ dgate_part,  // (BT, n_splits, C)
    int BT, int S, int C, int n_splits
) {
    const int bt = blockIdx.x;
    const int sp = blockIdx.y;
    const int tid = threadIdx.x;
    const int C2 = C >> 1;
    const int s_start = (sp * S) / n_splits;
    const int s_end   = ((sp + 1) * S) / n_splits;
    const float* g = gate + static_cast<int64_t>(bt) * C;

    extern __shared__ float s_dg[];  // C floats
    for (int c = tid; c < C; c += 256) s_dg[c] = 0.0f;
    __syncthreads();

    for (int s = s_start; s < s_end; ++s) {
        int m = bt * S + s;
        const __mt_bfloat162* dr = dout2 + static_cast<int64_t>(m) * C2;
        const __mt_bfloat162* yr = y2    + static_cast<int64_t>(m) * C2;
        __mt_bfloat162* dyr = dy2        + static_cast<int64_t>(m) * C2;
        for (int p = tid; p < C2; p += 256) {
            int c0 = p << 1;
            __mt_bfloat162 dv = dr[p], yv = yr[p];
            float d0 = __low2float(dv), d1 = __high2float(dv);
            float y0 = __low2float(yv), y1 = __high2float(yv);
            s_dg[c0]     += d0 * y0;
            s_dg[c0 + 1] += d1 * y1;
            dyr[p] = __floats2bfloat162_rn(d0 * g[c0], d1 * g[c0 + 1]);
        }
    }
    __syncthreads();

    int64_t base = (static_cast<int64_t>(bt) * n_splits + sp) * C;
    for (int c = tid; c < C; c += 256) dgate_part[base + c] = s_dg[c];
}

extern "C" __global__ void reduce_frame_splits_1(
    const float* __restrict__ p1, float* __restrict__ o1, int BT, int n_splits, int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (c >= C) return;
    float s1 = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        s1 += p1[(static_cast<int64_t>(bt) * n_splits + sp) * C + c];
    }
    o1[static_cast<int64_t>(bt) * C + c] = s1;
}

// =========================================================================
// 4. Fused Output Gate: y = out * silu(g) (FP32 internal, BF16 in/out)
// =========================================================================
extern "C" __global__ void silu_mul_fwd_u4_bf16(
    const uint4* __restrict__ out4,
    const uint4* __restrict__ g4,
    uint4* __restrict__ y4,
    int64_t n8
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint4 ov = out4[i], gv = g4[i], yv;
    const __mt_bfloat162* oh = reinterpret_cast<const __mt_bfloat162*>(&ov);
    const __mt_bfloat162* gh = reinterpret_cast<const __mt_bfloat162*>(&gv);
    __mt_bfloat162* yh = reinterpret_cast<__mt_bfloat162*>(&yv);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float o0 = __low2float(oh[k]), o1 = __high2float(oh[k]);
        float g0 = __low2float(gh[k]), g1 = __high2float(gh[k]);
        yh[k] = __floats2bfloat162_rn(o0 * g0 * sigmoid_tanh_bf(g0), o1 * g1 * sigmoid_tanh_bf(g1));
    }
    y4[i] = yv;
}

extern "C" __global__ void silu_mul_bwd_u4_bf16(
    const uint4* __restrict__ dy4,
    const uint4* __restrict__ out4,
    const uint4* __restrict__ g4,
    uint4* __restrict__ dout4,
    uint4* __restrict__ dg4,
    uint4* __restrict__ y4_out,
    int64_t n8
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint4 dv = dy4[i], ov = out4[i], gv = g4[i], dov, dgv, yv;
    const __mt_bfloat162* dh = reinterpret_cast<const __mt_bfloat162*>(&dv);
    const __mt_bfloat162* oh = reinterpret_cast<const __mt_bfloat162*>(&ov);
    const __mt_bfloat162* gh = reinterpret_cast<const __mt_bfloat162*>(&gv);
    __mt_bfloat162* doh = reinterpret_cast<__mt_bfloat162*>(&dov);
    __mt_bfloat162* dgh = reinterpret_cast<__mt_bfloat162*>(&dgv);
    __mt_bfloat162* yh  = reinterpret_cast<__mt_bfloat162*>(&yv);
    const bool emit_y = (y4_out != nullptr);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float d0 = __low2float(dh[k]), d1 = __high2float(dh[k]);
        float o0 = __low2float(oh[k]), o1 = __high2float(oh[k]);
        float g0 = __low2float(gh[k]), g1 = __high2float(gh[k]);
        float sig0 = sigmoid_tanh_bf(g0), sig1 = sigmoid_tanh_bf(g1);
        float silu0 = g0 * sig0, silu1 = g1 * sig1;
        float dsilu0 = sig0 * (1.0f + g0 * (1.0f - sig0));
        float dsilu1 = sig1 * (1.0f + g1 * (1.0f - sig1));
        doh[k] = __floats2bfloat162_rn(d0 * silu0, d1 * silu1);
        dgh[k] = __floats2bfloat162_rn(d0 * o0 * dsilu0, d1 * o1 * dsilu1);
        if (emit_y) yh[k] = __floats2bfloat162_rn(o0 * silu0, o1 * silu1);
    }
    dout4[i] = dov;
    dg4[i]   = dgv;
    if (emit_y) y4_out[i] = yv;
}

extern "C" __global__ void silu_mul_fwd_bf16(
    const __mt_bfloat162* __restrict__ out2,
    const __mt_bfloat162* __restrict__ g2,
    __mt_bfloat162* __restrict__ y2,
    int64_t n2
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    __mt_bfloat162 ov = out2[i], gv = g2[i];
    float o0 = __low2float(ov), o1 = __high2float(ov);
    float g0 = __low2float(gv), g1 = __high2float(gv);
    float s0 = g0 * sigmoid_tanh_bf(g0);
    float s1 = g1 * sigmoid_tanh_bf(g1);
    y2[i] = __floats2bfloat162_rn(o0 * s0, o1 * s1);
}

extern "C" __global__ void silu_mul_bwd_bf16(
    const __mt_bfloat162* __restrict__ dy2,
    const __mt_bfloat162* __restrict__ out2,
    const __mt_bfloat162* __restrict__ g2,
    __mt_bfloat162* __restrict__ dout2,
    __mt_bfloat162* __restrict__ dg2,
    __mt_bfloat162* __restrict__ y2_out,
    int64_t n2
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    __mt_bfloat162 dv = dy2[i], ov = out2[i], gv = g2[i];
    float d0 = __low2float(dv), d1 = __high2float(dv);
    float o0 = __low2float(ov), o1 = __high2float(ov);
    float g0 = __low2float(gv), g1 = __high2float(gv);
    float sig0 = sigmoid_tanh_bf(g0);
    float sig1 = sigmoid_tanh_bf(g1);
    float silu0 = g0 * sig0, silu1 = g1 * sig1;
    float dsilu0 = sig0 * (1.0f + g0 * (1.0f - sig0));
    float dsilu1 = sig1 * (1.0f + g1 * (1.0f - sig1));
    dout2[i] = __floats2bfloat162_rn(d0 * silu0, d1 * silu1);
    dg2[i]   = __floats2bfloat162_rn(d0 * o0 * dsilu0, d1 * o1 * dsilu1);
    if (y2_out != nullptr) y2_out[i] = __floats2bfloat162_rn(o0 * silu0, o1 * silu1);
}

// =========================================================================
// 5. Fused Softmax Main Prep: qkv(B,N,3,H,D) BF16 -> RMSNorm(q,k) + RoPE -> q,k,v (B,H,N,D) BF16
// =========================================================================
extern "C" __global__ void softmax_main_prep_fwd_bf16(
    const __mt_bfloat162* __restrict__ qkv2,  // (B, N, 3*C/2)
    const float* __restrict__ q_nw,           // (C,) or null
    const float* __restrict__ k_nw,           // (C,) or null
    const float* __restrict__ rope_cos,       // (N, D)
    const float* __restrict__ rope_sin,       // (N, D)
    __mt_bfloat162* __restrict__ q_out2,      // (B*H, N, D/2)
    __mt_bfloat162* __restrict__ k_out2,      // (B*H, N, D/2)
    __mt_bfloat162* __restrict__ v_out2,      // (B*H, N, D/2)
    float* __restrict__ q_ir_out,             // (B*N,)
    float* __restrict__ k_ir_out,             // (B*N,)
    int B, int N, int H, int D, float norm_eps, int has_norm
) {
    const int bn = blockIdx.x;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int tid = threadIdx.x;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D2 = D >> 1;

    const __mt_bfloat162* q_ptr = qkv2 + static_cast<int64_t>(bn) * 3 * C2;
    const __mt_bfloat162* k_ptr = q_ptr + C2;
    const __mt_bfloat162* v_ptr = q_ptr + 2 * C2;

    __shared__ float s_qred[8], s_kred[8];
    float q_ir = 1.0f, k_ir = 1.0f;
    if (has_norm) {
        float qsq = 0.0f, ksq = 0.0f;
        for (int p = tid; p < C2; p += 256) {
            __mt_bfloat162 qv = q_ptr[p], kv = k_ptr[p];
            float q0 = __low2float(qv), q1 = __high2float(qv);
            float k0 = __low2float(kv), k1 = __high2float(kv);
            qsq = fmaf(q0, q0, fmaf(q1, q1, qsq));
            ksq = fmaf(k0, k0, fmaf(k1, k1, ksq));
        }
        qsq = warp_sum(qsq); ksq = warp_sum(ksq);
        if ((tid & 31) == 0) { s_qred[tid >> 5] = qsq; s_kred[tid >> 5] = ksq; }
        __syncthreads();
        if (tid == 0) {
            float tq = 0.0f, tk = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) { tq += s_qred[w]; tk += s_kred[w]; }
            s_qred[0] = rsqrtf(tq / static_cast<float>(C) + norm_eps);
            s_kred[0] = rsqrtf(tk / static_cast<float>(C) + norm_eps);
            q_ir_out[bn] = s_qred[0];
            k_ir_out[bn] = s_kred[0];
        }
        __syncthreads();
        q_ir = s_qred[0]; k_ir = s_kred[0];
    }

    const float* cos_row = rope_cos + static_cast<int64_t>(n_idx) * D;
    const float* sin_row = rope_sin + static_cast<int64_t>(n_idx) * D;

    for (int p = tid; p < C2; p += 256) {
        int h = p / D2;
        int d2 = p - h * D2;
        int d0 = d2 << 1;
        int c0 = p << 1;

        __mt_bfloat162 qv = q_ptr[p], kv = k_ptr[p], vv = v_ptr[p];
        float q0 = __low2float(qv), q1 = __high2float(qv);
        float k0 = __low2float(kv), k1 = __high2float(kv);
        if (has_norm) {
            q0 = q0 * q_ir * q_nw[c0];     q1 = q1 * q_ir * q_nw[c0 + 1];
            k0 = k0 * k_ir * k_nw[c0];     k1 = k1 * k_ir * k_nw[c0 + 1];
        }
        float ce = cos_row[d0], co = cos_row[d0 + 1];
        float se = sin_row[d0], so = sin_row[d0 + 1];
        float qr0 = fmaf(q0, ce, q1 * se), qr1 = fmaf(q1, co, q0 * so);
        float kr0 = fmaf(k0, ce, k1 * se), kr1 = fmaf(k1, co, k0 * so);

        int64_t out_p = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D2 + d2;
        q_out2[out_p] = __floats2bfloat162_rn(qr0, qr1);
        k_out2[out_p] = __floats2bfloat162_rn(kr0, kr1);
        v_out2[out_p] = vv;
    }
}

extern "C" __global__ void softmax_main_prep_bwd_bf16(
    const __mt_bfloat162* __restrict__ qkv2,
    const float* __restrict__ q_nw,
    const float* __restrict__ k_nw,
    const float* __restrict__ q_ir_in,
    const float* __restrict__ k_ir_in,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const __mt_bfloat162* __restrict__ dq_in2,  // (B*H, N, D/2)
    const __mt_bfloat162* __restrict__ dk_in2,
    const __mt_bfloat162* __restrict__ dv_in2,
    __mt_bfloat162* __restrict__ dqkv2,
    float* __restrict__ dq_nw_part,
    float* __restrict__ dk_nw_part,
    int B, int N, int H, int D, int has_norm, int n_chunks
) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int BN = B * N;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D2 = D >> 1;
    const int bn_start = (static_cast<int64_t>(chunk) * BN) / n_chunks;
    const int bn_end   = (static_cast<int64_t>(chunk + 1) * BN) / n_chunks;

    extern __shared__ float smem[];  // 4*C + 16 floats: s_dqnw, s_dknw, s_gq, s_gk, s_red[16]
    float* s_dqnw = smem;
    float* s_dknw = s_dqnw + C;
    float* s_gq   = s_dknw + C;
    float* s_gk   = s_gq + C;
    float* s_red  = s_gk + C;

    if (has_norm) {
        for (int i = tid; i < C; i += 256) { s_dqnw[i] = 0.0f; s_dknw[i] = 0.0f; }
    }
    __syncthreads();

    for (int bn = bn_start; bn < bn_end; ++bn) {
        const int b_idx = bn / N;
        const int n_idx = bn - b_idx * N;
        const __mt_bfloat162* q_ptr = qkv2 + static_cast<int64_t>(bn) * 3 * C2;
        const __mt_bfloat162* k_ptr = q_ptr + C2;
        __mt_bfloat162* dq_ptr = dqkv2 + static_cast<int64_t>(bn) * 3 * C2;
        __mt_bfloat162* dk_ptr = dq_ptr + C2;
        __mt_bfloat162* dv_ptr = dq_ptr + 2 * C2;

        const float* cos_row = rope_cos + static_cast<int64_t>(n_idx) * D;
        const float* sin_row = rope_sin + static_cast<int64_t>(n_idx) * D;

        float q_dot = 0.0f, k_dot = 0.0f;
        const float q_ir = has_norm ? q_ir_in[bn] : 1.0f;
        const float k_ir = has_norm ? k_ir_in[bn] : 1.0f;

        for (int p = tid; p < C2; p += 256) {
            int h = p / D2;
            int d2 = p - h * D2;
            int d0 = d2 << 1;
            int c0 = p << 1;
            int64_t in_p = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D2 + d2;

            __mt_bfloat162 dqv = dq_in2[in_p], dkv = dk_in2[in_p];
            dv_ptr[p] = dv_in2[in_p];

            float ce = cos_row[d0], co = cos_row[d0 + 1];
            float se = sin_row[d0], so = sin_row[d0 + 1];
            float gq0 = fmaf(__low2float(dqv), ce, __high2float(dqv) * so);
            float gq1 = fmaf(__high2float(dqv), co, __low2float(dqv) * se);
            float gk0 = fmaf(__low2float(dkv), ce, __high2float(dkv) * so);
            float gk1 = fmaf(__high2float(dkv), co, __low2float(dkv) * se);

            if (has_norm) {
                s_gq[c0] = gq0; s_gq[c0 + 1] = gq1;
                s_gk[c0] = gk0; s_gk[c0 + 1] = gk1;
                __mt_bfloat162 qv = q_ptr[p], kv = k_ptr[p];
                float q0 = __low2float(qv), q1 = __high2float(qv);
                float k0 = __low2float(kv), k1 = __high2float(kv);
                s_dqnw[c0]     += gq0 * q0 * q_ir;
                s_dqnw[c0 + 1] += gq1 * q1 * q_ir;
                s_dknw[c0]     += gk0 * k0 * k_ir;
                s_dknw[c0 + 1] += gk1 * k1 * k_ir;
                q_dot = fmaf(gq0 * q_nw[c0], q0, fmaf(gq1 * q_nw[c0 + 1], q1, q_dot));
                k_dot = fmaf(gk0 * k_nw[c0], k0, fmaf(gk1 * k_nw[c0 + 1], k1, k_dot));
            } else {
                dq_ptr[p] = __floats2bfloat162_rn(gq0, gq1);
                dk_ptr[p] = __floats2bfloat162_rn(gk0, gk1);
            }
        }

        if (has_norm) {
            q_dot = warp_sum(q_dot); k_dot = warp_sum(k_dot);
            if ((tid & 31) == 0) { s_red[tid >> 5] = q_dot; s_red[8 + (tid >> 5)] = k_dot; }
            __syncthreads();
            if (tid == 0) {
                float sq = 0.0f, sk = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; ++w) { sq += s_red[w]; sk += s_red[8 + w]; }
                s_red[0] = sq * q_ir * q_ir / static_cast<float>(C);
                s_red[8] = sk * k_ir * k_ir / static_cast<float>(C);
            }
            __syncthreads();
            float q_coeff = s_red[0], k_coeff = s_red[8];

            for (int p = tid; p < C2; p += 256) {
                int c0 = p << 1;
                __mt_bfloat162 qv = q_ptr[p], kv = k_ptr[p];
                float q0 = __low2float(qv), q1 = __high2float(qv);
                float k0 = __low2float(kv), k1 = __high2float(kv);
                dq_ptr[p] = __floats2bfloat162_rn(
                    q_ir * (s_gq[c0]     * q_nw[c0]     - q0 * q_coeff),
                    q_ir * (s_gq[c0 + 1] * q_nw[c0 + 1] - q1 * q_coeff)
                );
                dk_ptr[p] = __floats2bfloat162_rn(
                    k_ir * (s_gk[c0]     * k_nw[c0]     - k0 * k_coeff),
                    k_ir * (s_gk[c0 + 1] * k_nw[c0 + 1] - k1 * k_coeff)
                );
            }
            __syncthreads();
        }
    }

    if (has_norm) {
        int64_t part_base = static_cast<int64_t>(chunk) * C;
        for (int i = tid; i < C; i += 256) {
            dq_nw_part[part_base + i] = s_dqnw[i];
            dk_nw_part[part_base + i] = s_dknw[i];
        }
    }
}

// =========================================================================
// 6. Fused Softmax Cam Prep: qkv_cam(B,N,3,C) BF16 -> RMSNorm(q,k) + 4x4 UCPE + RoPE -> q,k,v (B,H,N,D) BF16
//    Takes qkv_cam directly (no extra chunk/copy!). No ReLU, no k_scale, no inflation_sq.
// =========================================================================
extern "C" __global__ void softmax_cam_prep_fwd_bf16(
    const __mt_bfloat162* __restrict__ qkv2,  // (B, N, 3*C/2)
    const float* __restrict__ q_nw,           // (C,)
    const float* __restrict__ k_nw,           // (C,)
    const float* __restrict__ P_T,            // (B, N, 16)
    const float* __restrict__ P_inv,          // (B, N, 16)
    const float* __restrict__ rope_cos,       // (N, D/2)
    const float* __restrict__ rope_sin,       // (N, D/2)
    __mt_bfloat162* __restrict__ q_out2,      // (B*H, N, D/2)
    __mt_bfloat162* __restrict__ k_out2,
    __mt_bfloat162* __restrict__ v_out2,
    float* __restrict__ q_ir_out,
    float* __restrict__ k_ir_out,
    int B, int N, int H, int D, float norm_eps
) {
    const int bn = blockIdx.x;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int tid = threadIdx.x;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D2 = D >> 1;
    const int D_half = D >> 1;
    const int n_grp4 = D_half >> 2;
    const int D_half2 = D_half >> 1;

    __shared__ float s_PT[16], s_Pinv[16];
    __shared__ float s_cos[64], s_sin[64];
    __shared__ float s_qred[8], s_kred[8];

    if (tid < 16) {
        s_PT[tid] = P_T[static_cast<int64_t>(bn) * 16 + tid];
        s_Pinv[tid] = P_inv[static_cast<int64_t>(bn) * 16 + tid];
    }
    if (tid < D_half) {
        s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D_half + tid];
        s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D_half + tid];
    }

    const __mt_bfloat162* qr = qkv2 + static_cast<int64_t>(bn) * 3 * C2;
    const __mt_bfloat162* kr = qr + C2;
    const __mt_bfloat162* vr = qr + 2 * C2;

    float qsq = 0.0f, ksq = 0.0f;
    for (int p = tid; p < C2; p += 256) {
        __mt_bfloat162 qv = qr[p], kv = kr[p];
        float q0 = __low2float(qv), q1 = __high2float(qv);
        float k0 = __low2float(kv), k1 = __high2float(kv);
        qsq = fmaf(q0, q0, fmaf(q1, q1, qsq));
        ksq = fmaf(k0, k0, fmaf(k1, k1, ksq));
    }
    qsq = warp_sum(qsq); ksq = warp_sum(ksq);
    if ((tid & 31) == 0) { s_qred[tid >> 5] = qsq; s_kred[tid >> 5] = ksq; }
    __syncthreads();
    if (tid == 0) {
        float tq = 0.0f, tk = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { tq += s_qred[w]; tk += s_kred[w]; }
        s_qred[0] = rsqrtf(tq / static_cast<float>(C) + norm_eps);
        s_kred[0] = rsqrtf(tk / static_cast<float>(C) + norm_eps);
        q_ir_out[bn] = s_qred[0];
        k_ir_out[bn] = s_kred[0];
    }
    __syncthreads();
    const float q_ir = s_qred[0], k_ir = s_kred[0];

    // First half [0..D_half): 4x4 P_T / P_inv
    const int total_g4 = H * n_grp4;
    for (int gi = tid; gi < total_g4; gi += 256) {
        int h = gi / n_grp4;
        int g = gi - h * n_grp4;
        int d0 = g * 4;
        int c0 = h * D + d0;
        int p0 = c0 >> 1;

        __mt_bfloat162 qa = qr[p0], qb = qr[p0 + 1];
        __mt_bfloat162 ka = kr[p0], kb = kr[p0 + 1];
        __mt_bfloat162 va = vr[p0], vb = vr[p0 + 1];

        float qx0 = __low2float(qa)  * q_ir * q_nw[c0 + 0];
        float qx1 = __high2float(qa) * q_ir * q_nw[c0 + 1];
        float qx2 = __low2float(qb)  * q_ir * q_nw[c0 + 2];
        float qx3 = __high2float(qb) * q_ir * q_nw[c0 + 3];

        float kx0 = __low2float(ka)  * k_ir * k_nw[c0 + 0];
        float kx1 = __high2float(ka) * k_ir * k_nw[c0 + 1];
        float kx2 = __low2float(kb)  * k_ir * k_nw[c0 + 2];
        float kx3 = __high2float(kb) * k_ir * k_nw[c0 + 3];

        float vx0 = __low2float(va), vx1 = __high2float(va), vx2 = __low2float(vb), vx3 = __high2float(vb);

        float qy0 = s_PT[0]*qx0  + s_PT[1]*qx1  + s_PT[2]*qx2  + s_PT[3]*qx3;
        float qy1 = s_PT[4]*qx0  + s_PT[5]*qx1  + s_PT[6]*qx2  + s_PT[7]*qx3;
        float qy2 = s_PT[8]*qx0  + s_PT[9]*qx1  + s_PT[10]*qx2 + s_PT[11]*qx3;
        float qy3 = s_PT[12]*qx0 + s_PT[13]*qx1 + s_PT[14]*qx2 + s_PT[15]*qx3;

        float ky0 = s_Pinv[0]*kx0  + s_Pinv[1]*kx1  + s_Pinv[2]*kx2  + s_Pinv[3]*kx3;
        float ky1 = s_Pinv[4]*kx0  + s_Pinv[5]*kx1  + s_Pinv[6]*kx2  + s_Pinv[7]*kx3;
        float ky2 = s_Pinv[8]*kx0  + s_Pinv[9]*kx1  + s_Pinv[10]*kx2 + s_Pinv[11]*kx3;
        float ky3 = s_Pinv[12]*kx0 + s_Pinv[13]*kx1 + s_Pinv[14]*kx2 + s_Pinv[15]*kx3;

        float vy0 = s_Pinv[0]*vx0  + s_Pinv[1]*vx1  + s_Pinv[2]*vx2  + s_Pinv[3]*vx3;
        float vy1 = s_Pinv[4]*vx0  + s_Pinv[5]*vx1  + s_Pinv[6]*vx2  + s_Pinv[7]*vx3;
        float vy2 = s_Pinv[8]*vx0  + s_Pinv[9]*vx1  + s_Pinv[10]*vx2 + s_Pinv[11]*vx3;
        float vy3 = s_Pinv[12]*vx0 + s_Pinv[13]*vx1 + s_Pinv[14]*vx2 + s_Pinv[15]*vx3;

        int64_t op = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D2 + (d0 >> 1);
        q_out2[op] = __floats2bfloat162_rn(qy0, qy1); q_out2[op + 1] = __floats2bfloat162_rn(qy2, qy3);
        k_out2[op] = __floats2bfloat162_rn(ky0, ky1); k_out2[op + 1] = __floats2bfloat162_rn(ky2, ky3);
        v_out2[op] = __floats2bfloat162_rn(vy0, vy1); v_out2[op + 1] = __floats2bfloat162_rn(vy2, vy3);
    }

    // Second half [D_half..D): FP32 RoPE
    const int total_r2 = H * D_half2;
    for (int ri = tid; ri < total_r2; ri += 256) {
        int h = ri / D_half2;
        int d2 = ri - h * D_half2;
        int d_rel = d2 << 1;
        int d0 = D_half + d_rel;
        int c0 = h * D + d0;
        int p0 = c0 >> 1;

        __mt_bfloat162 qv = qr[p0], kv = kr[p0], vv = vr[p0];
        float q0 = __low2float(qv)  * q_ir * q_nw[c0];
        float q1 = __high2float(qv) * q_ir * q_nw[c0 + 1];
        float k0 = __low2float(kv)  * k_ir * k_nw[c0];
        float k1 = __high2float(kv) * k_ir * k_nw[c0 + 1];
        float v0 = __low2float(vv), v1 = __high2float(vv);

        float ce = s_cos[d_rel], co = s_cos[d_rel + 1];
        float se = s_sin[d_rel], so = s_sin[d_rel + 1];

        int64_t op = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D2 + (d0 >> 1);
        q_out2[op] = __floats2bfloat162_rn(fmaf(q0, ce, q1 * se), fmaf(q1, co, q0 * so));
        k_out2[op] = __floats2bfloat162_rn(fmaf(k0, ce, k1 * se), fmaf(k1, co, k0 * so));
        v_out2[op] = __floats2bfloat162_rn(fmaf(v0, ce, v1 * se), fmaf(v1, co, v0 * so));
    }
}

extern "C" __global__ void softmax_cam_prep_bwd_bf16(
    const __mt_bfloat162* __restrict__ qkv2,
    const float* __restrict__ q_nw,
    const float* __restrict__ k_nw,
    const float* __restrict__ P_T,
    const float* __restrict__ P_inv,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const float* __restrict__ q_ir_in,
    const float* __restrict__ k_ir_in,
    const __mt_bfloat162* __restrict__ dq_in2,
    const __mt_bfloat162* __restrict__ dk_in2,
    const __mt_bfloat162* __restrict__ dv_in2,
    __mt_bfloat162* __restrict__ dqkv2,
    float* __restrict__ dq_nw_part,
    float* __restrict__ dk_nw_part,
    int B, int N, int H, int D, int n_chunks
) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int BN = B * N;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D2 = D >> 1;
    const int D_half = D >> 1;
    const int n_grp4 = D_half >> 2;
    const int D_half2 = D_half >> 1;
    const int bn_start = (static_cast<int64_t>(chunk) * BN) / n_chunks;
    const int bn_end   = (static_cast<int64_t>(chunk + 1) * BN) / n_chunks;

    extern __shared__ float smem[];
    float* s_dqnw = smem;            // C
    float* s_dknw = s_dqnw + C;      // C
    float* s_gq   = s_dknw + C;      // C
    float* s_gk   = s_gq + C;        // C
    float* s_PT   = s_gk + C;        // 16
    float* s_Pinv = s_PT + 16;       // 16
    float* s_cos  = s_Pinv + 16;     // 64
    float* s_sin  = s_cos + 64;      // 64
    float* s_red  = s_sin + 64;      // 16

    for (int i = tid; i < C; i += 256) { s_dqnw[i] = 0.0f; s_dknw[i] = 0.0f; }
    __syncthreads();

    for (int bn = bn_start; bn < bn_end; ++bn) {
        const int b_idx = bn / N;
        const int n_idx = bn - b_idx * N;

        if (tid < 16) {
            s_PT[tid]   = P_T[static_cast<int64_t>(bn) * 16 + tid];
            s_Pinv[tid] = P_inv[static_cast<int64_t>(bn) * 16 + tid];
        }
        if (tid < D_half) {
            s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D_half + tid];
            s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D_half + tid];
        }
        __syncthreads();

        const __mt_bfloat162* qr = qkv2 + static_cast<int64_t>(bn) * 3 * C2;
        const __mt_bfloat162* kr = qr + C2;
        __mt_bfloat162* dqr_w = dqkv2 + static_cast<int64_t>(bn) * 3 * C2;
        __mt_bfloat162* dkr_w = dqr_w + C2;
        __mt_bfloat162* dvr_w = dqr_w + 2 * C2;
        const float q_ir = q_ir_in[bn], k_ir = k_ir_in[bn];

        const int total_g4 = H * n_grp4;
        for (int gi = tid; gi < total_g4; gi += 256) {
            int h = gi / n_grp4;
            int g = gi - h * n_grp4;
            int d0 = g * 4;
            int c0 = h * D + d0;
            int p0 = c0 >> 1;
            int64_t ip = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D2 + (d0 >> 1);

            __mt_bfloat162 dqa = dq_in2[ip], dqb = dq_in2[ip + 1];
            __mt_bfloat162 dka = dk_in2[ip], dkb = dk_in2[ip + 1];
            __mt_bfloat162 dva = dv_in2[ip], dvb = dv_in2[ip + 1];
            float dq0 = __low2float(dqa), dq1 = __high2float(dqa), dq2 = __low2float(dqb), dq3 = __high2float(dqb);
            float dk0 = __low2float(dka), dk1 = __high2float(dka), dk2 = __low2float(dkb), dk3 = __high2float(dkb);
            float dv0 = __low2float(dva), dv1 = __high2float(dva), dv2 = __low2float(dvb), dv3 = __high2float(dvb);

            s_gq[c0 + 0] = s_PT[0]*dq0 + s_PT[4]*dq1 + s_PT[8]*dq2  + s_PT[12]*dq3;
            s_gq[c0 + 1] = s_PT[1]*dq0 + s_PT[5]*dq1 + s_PT[9]*dq2  + s_PT[13]*dq3;
            s_gq[c0 + 2] = s_PT[2]*dq0 + s_PT[6]*dq1 + s_PT[10]*dq2 + s_PT[14]*dq3;
            s_gq[c0 + 3] = s_PT[3]*dq0 + s_PT[7]*dq1 + s_PT[11]*dq2 + s_PT[15]*dq3;

            s_gk[c0 + 0] = s_Pinv[0]*dk0 + s_Pinv[4]*dk1 + s_Pinv[8]*dk2  + s_Pinv[12]*dk3;
            s_gk[c0 + 1] = s_Pinv[1]*dk0 + s_Pinv[5]*dk1 + s_Pinv[9]*dk2  + s_Pinv[13]*dk3;
            s_gk[c0 + 2] = s_Pinv[2]*dk0 + s_Pinv[6]*dk1 + s_Pinv[10]*dk2 + s_Pinv[14]*dk3;
            s_gk[c0 + 3] = s_Pinv[3]*dk0 + s_Pinv[7]*dk1 + s_Pinv[11]*dk2 + s_Pinv[15]*dk3;

            float gvx0 = s_Pinv[0]*dv0 + s_Pinv[4]*dv1 + s_Pinv[8]*dv2  + s_Pinv[12]*dv3;
            float gvx1 = s_Pinv[1]*dv0 + s_Pinv[5]*dv1 + s_Pinv[9]*dv2  + s_Pinv[13]*dv3;
            float gvx2 = s_Pinv[2]*dv0 + s_Pinv[6]*dv1 + s_Pinv[10]*dv2 + s_Pinv[14]*dv3;
            float gvx3 = s_Pinv[3]*dv0 + s_Pinv[7]*dv1 + s_Pinv[11]*dv2 + s_Pinv[15]*dv3;
            dvr_w[p0]     = __floats2bfloat162_rn(gvx0, gvx1);
            dvr_w[p0 + 1] = __floats2bfloat162_rn(gvx2, gvx3);
        }

        const int total_r2 = H * D_half2;
        for (int ri = tid; ri < total_r2; ri += 256) {
            int h = ri / D_half2;
            int d2 = ri - h * D_half2;
            int d_rel = d2 << 1;
            int d0 = D_half + d_rel;
            int c0 = h * D + d0;
            int p0 = c0 >> 1;
            int64_t ip = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D2 + (d0 >> 1);

            __mt_bfloat162 dq = dq_in2[ip], dk = dk_in2[ip], dv = dv_in2[ip];
            float ce = s_cos[d_rel], co = s_cos[d_rel + 1];
            float se = s_sin[d_rel], so = s_sin[d_rel + 1];

            s_gq[c0]     = fmaf(__low2float(dq),  ce, __high2float(dq) * so);
            s_gq[c0 + 1] = fmaf(__high2float(dq), co, __low2float(dq)  * se);
            s_gk[c0]     = fmaf(__low2float(dk),  ce, __high2float(dk) * so);
            s_gk[c0 + 1] = fmaf(__high2float(dk), co, __low2float(dk)  * se);
            dvr_w[p0] = __floats2bfloat162_rn(
                fmaf(__low2float(dv), ce, __high2float(dv) * so),
                fmaf(__high2float(dv), co, __low2float(dv) * se)
            );
        }
        __syncthreads();

        float q_dot = 0.0f, k_dot = 0.0f;
        if (C2 <= 1280) {
            float rq0[5], rq1[5], rk0[5], rk1[5];
            int idx_r = 0;
            for (int p = tid; p < C2; p += 256, ++idx_r) {
                int c0 = p << 1;
                __mt_bfloat162 qv = qr[p], kv = kr[p];
                float q0 = __low2float(qv), q1 = __high2float(qv);
                float k0 = __low2float(kv), k1 = __high2float(kv);
                rq0[idx_r] = q0; rq1[idx_r] = q1;
                rk0[idx_r] = k0; rk1[idx_r] = k1;
                float gq0 = s_gq[c0], gq1 = s_gq[c0 + 1];
                float gk0 = s_gk[c0], gk1 = s_gk[c0 + 1];
                s_dqnw[c0]     += gq0 * q0 * q_ir;
                s_dqnw[c0 + 1] += gq1 * q1 * q_ir;
                s_dknw[c0]     += gk0 * k0 * k_ir;
                s_dknw[c0 + 1] += gk1 * k1 * k_ir;
                q_dot = fmaf(gq0 * q_nw[c0], q0, fmaf(gq1 * q_nw[c0 + 1], q1, q_dot));
                k_dot = fmaf(gk0 * k_nw[c0], k0, fmaf(gk1 * k_nw[c0 + 1], k1, k_dot));
            }
            q_dot = warp_sum(q_dot); k_dot = warp_sum(k_dot);
            if ((tid & 31) == 0) { s_red[tid >> 5] = q_dot; s_red[8 + (tid >> 5)] = k_dot; }
            __syncthreads();
            if (tid == 0) {
                float sq = 0.0f, sk = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; ++w) { sq += s_red[w]; sk += s_red[8 + w]; }
                s_red[0] = sq * q_ir * q_ir / static_cast<float>(C);
                s_red[8] = sk * k_ir * k_ir / static_cast<float>(C);
            }
            __syncthreads();
            float q_coeff = s_red[0], k_coeff = s_red[8];

            idx_r = 0;
            for (int p = tid; p < C2; p += 256, ++idx_r) {
                int c0 = p << 1;
                dqr_w[p] = __floats2bfloat162_rn(
                    q_ir * (s_gq[c0]     * q_nw[c0]     - rq0[idx_r] * q_coeff),
                    q_ir * (s_gq[c0 + 1] * q_nw[c0 + 1] - rq1[idx_r] * q_coeff)
                );
                dkr_w[p] = __floats2bfloat162_rn(
                    k_ir * (s_gk[c0]     * k_nw[c0]     - rk0[idx_r] * k_coeff),
                    k_ir * (s_gk[c0 + 1] * k_nw[c0 + 1] - rk1[idx_r] * k_coeff)
                );
            }
        } else {
            for (int p = tid; p < C2; p += 256) {
                int c0 = p << 1;
                __mt_bfloat162 qv = qr[p], kv = kr[p];
                float q0 = __low2float(qv), q1 = __high2float(qv);
                float k0 = __low2float(kv), k1 = __high2float(kv);
                float gq0 = s_gq[c0], gq1 = s_gq[c0 + 1];
                float gk0 = s_gk[c0], gk1 = s_gk[c0 + 1];
                s_dqnw[c0]     += gq0 * q0 * q_ir;
                s_dqnw[c0 + 1] += gq1 * q1 * q_ir;
                s_dknw[c0]     += gk0 * k0 * k_ir;
                s_dknw[c0 + 1] += gk1 * k1 * k_ir;
                q_dot = fmaf(gq0 * q_nw[c0], q0, fmaf(gq1 * q_nw[c0 + 1], q1, q_dot));
                k_dot = fmaf(gk0 * k_nw[c0], k0, fmaf(gk1 * k_nw[c0 + 1], k1, k_dot));
            }
            q_dot = warp_sum(q_dot); k_dot = warp_sum(k_dot);
            if ((tid & 31) == 0) { s_red[tid >> 5] = q_dot; s_red[8 + (tid >> 5)] = k_dot; }
            __syncthreads();
            if (tid == 0) {
                float sq = 0.0f, sk = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; ++w) { sq += s_red[w]; sk += s_red[8 + w]; }
                s_red[0] = sq * q_ir * q_ir / static_cast<float>(C);
                s_red[8] = sk * k_ir * k_ir / static_cast<float>(C);
            }
            __syncthreads();
            float q_coeff = s_red[0], k_coeff = s_red[8];

            for (int p = tid; p < C2; p += 256) {
                int c0 = p << 1;
                __mt_bfloat162 qv = qr[p], kv = kr[p];
                float q0 = __low2float(qv), q1 = __high2float(qv);
                float k0 = __low2float(kv), k1 = __high2float(kv);
                dqr_w[p] = __floats2bfloat162_rn(
                    q_ir * (s_gq[c0]     * q_nw[c0]     - q0 * q_coeff),
                    q_ir * (s_gq[c0 + 1] * q_nw[c0 + 1] - q1 * q_coeff)
                );
                dkr_w[p] = __floats2bfloat162_rn(
                    k_ir * (s_gk[c0]     * k_nw[c0]     - k0 * k_coeff),
                    k_ir * (s_gk[c0 + 1] * k_nw[c0 + 1] - k1 * k_coeff)
                );
            }
        }
        __syncthreads();
    }

    int64_t part_base = static_cast<int64_t>(chunk) * C;
    for (int i = tid; i < C; i += 256) {
        dq_nw_part[part_base + i] = s_dqnw[i];
        dk_nw_part[part_base + i] = s_dknw[i];
    }
}

// =========================================================================
// 7. Fast 128-bit Vectorized BF16 Column Sum (replaces slow ReduceABCKernel in F.linear bwd)
// =========================================================================
extern "C" __global__ void bf16_col_sum_part(
    const uint4* __restrict__ dy8,    // (M, C/8)
    float* __restrict__ partial,      // (n_splits, C)
    int M, int C8, int n_splits
) {
    const int tx = threadIdx.x & 31;
    const int ty = threadIdx.x >> 5;
    const int c_vec = blockIdx.x * 32 + tx;
    const int sp = blockIdx.y;
    const int m_start = (static_cast<int64_t>(sp) * M) / n_splits;
    const int m_end   = (static_cast<int64_t>(sp + 1) * M) / n_splits;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (c_vec < C8) {
        for (int m = m_start + ty; m < m_end; m += 8) {
            uint4 v = dy8[static_cast<int64_t>(m) * C8 + c_vec];
            const __mt_bfloat162* b2 = reinterpret_cast<const __mt_bfloat162*>(&v);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                acc[2 * k]     += __low2float(b2[k]);
                acc[2 * k + 1] += __high2float(b2[k]);
            }
        }
    }
    __shared__ float s[8][32][8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) s[ty][tx][k] = acc[k];
    __syncthreads();
    if (ty == 0 && c_vec < C8) {
        const int C = C8 << 3;
        float* out = partial + static_cast<int64_t>(sp) * C + (c_vec << 3);
        float r[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) sum += s[w][tx][k];
            r[k] = sum;
        }
        *reinterpret_cast<float4*>(out)     = make_float4(r[0], r[1], r[2], r[3]);
        *reinterpret_cast<float4*>(out + 4) = make_float4(r[4], r[5], r[6], r[7]);
    }
}

extern "C" __global__ void bf16_col_sum_reduce(
    const float* __restrict__ partial, // (n_splits, C)
    __mt_bfloat16* __restrict__ out,   // (C,)
    int n_splits, int C
) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    float s = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) s += partial[static_cast<int64_t>(sp) * C + c];
    out[c] = __float2bfloat16_rn(s);
}

// =========================================================================
// 8. Fused (a + b) * silu(g) Forward (writes both sum s = a+b and y = s * silu(g))
// =========================================================================
extern "C" __global__ void add_silu_mul_fwd_u4_bf16(
    const uint4* __restrict__ a4,
    const uint4* __restrict__ b4,
    const uint4* __restrict__ g4,
    uint4* __restrict__ s4,
    uint4* __restrict__ y4,
    int64_t n8
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint4 av = a4[i], bv = b4[i], gv = g4[i], sv, yv;
    const __mt_bfloat162* ah = reinterpret_cast<const __mt_bfloat162*>(&av);
    const __mt_bfloat162* bh = reinterpret_cast<const __mt_bfloat162*>(&bv);
    const __mt_bfloat162* gh = reinterpret_cast<const __mt_bfloat162*>(&gv);
    __mt_bfloat162* sh = reinterpret_cast<__mt_bfloat162*>(&sv);
    __mt_bfloat162* yh = reinterpret_cast<__mt_bfloat162*>(&yv);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        __mt_bfloat162 s2 = __floats2bfloat162_rn(__low2float(ah[k]) + __low2float(bh[k]),
                                                  __high2float(ah[k]) + __high2float(bh[k]));
        sh[k] = s2;
        float g0 = __low2float(gh[k]), g1 = __high2float(gh[k]);
        yh[k] = __floats2bfloat162_rn(__low2float(s2) * g0 * sigmoid_tanh_bf(g0),
                                      __high2float(s2) * g1 * sigmoid_tanh_bf(g1));
    }
    s4[i] = sv;
    y4[i] = yv;
}

extern "C" __global__ void add_silu_mul_fwd_bf16(
    const __mt_bfloat162* __restrict__ a2,
    const __mt_bfloat162* __restrict__ b2,
    const __mt_bfloat162* __restrict__ g2,
    __mt_bfloat162* __restrict__ s2,
    __mt_bfloat162* __restrict__ y2,
    int64_t n2
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    __mt_bfloat162 av = a2[i], bv = b2[i], gv = g2[i];
    float o0 = __low2float(av)  + __low2float(bv);
    float o1 = __high2float(av) + __high2float(bv);
    __mt_bfloat162 sv = __floats2bfloat162_rn(o0, o1);
    s2[i] = sv;
    float s0 = __low2float(sv), s1 = __high2float(sv);
    float g0 = __low2float(gv), g1 = __high2float(gv);
    float sg0 = g0 * sigmoid_tanh_bf(g0);
    float sg1 = g1 * sigmoid_tanh_bf(g1);
    y2[i] = __floats2bfloat162_rn(s0 * sg0, s1 * sg1);
}

// =========================================================================
// 9. Fused RMSNorm + Transpose (B, N, H*D) <-> (B, H, N, D) for CrossAttn Q
// =========================================================================
extern "C" __global__ void rmsnorm_bnhd_to_bhnd_fwd_bf16(
    const __mt_bfloat162* __restrict__ x2,     // (B*N, C/2)
    const float* __restrict__ w,               // (C,)
    __mt_bfloat162* __restrict__ y_bhnd2,      // (B*H, N, D/2)
    float* __restrict__ inv_rms_out,           // (B*N,)
    int B, int N, int H, int D, float eps
) {
    const int bn = blockIdx.x;
    const int b = (B == 1) ? 0 : (bn / N);
    const int n = (B == 1) ? bn : (bn - b * N);
    const int tid = threadIdx.x;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D2 = D >> 1;
    const __mt_bfloat162* xr = x2 + static_cast<int64_t>(bn) * C2;

    __shared__ float s_red[8];
    float sq = 0.0f;
    for (int p = tid; p < C2; p += 256) {
        __mt_bfloat162 xv = xr[p];
        float x0 = __low2float(xv), x1 = __high2float(xv);
        sq = fmaf(x0, x0, fmaf(x1, x1, sq));
    }
    float tot = block_sum_256(sq, s_red, tid);
    float ir = rsqrtf(tot / static_cast<float>(C) + eps);
    if (tid == 0) inv_rms_out[bn] = ir;

    for (int p = tid; p < C2; p += 256) {
        int h = p / D2;
        int d2 = p - h * D2;
        int c0 = p << 1;
        __mt_bfloat162 xv = xr[p];
        float y0 = __low2float(xv)  * ir * w[c0];
        float y1 = __high2float(xv) * ir * w[c0 + 1];
        int64_t op = (static_cast<int64_t>(b * H + h) * N + n) * D2 + d2;
        y_bhnd2[op] = __floats2bfloat162_rn(y0, y1);
    }
}

extern "C" __global__ void rmsnorm_bhnd_to_bnhd_bwd_bf16(
    const __mt_bfloat162* __restrict__ x2,      // (B*N, C/2)
    const __mt_bfloat162* __restrict__ dy_bhnd2,// (B*H, N, D/2)
    const float* __restrict__ w,                // (C,)
    const float* __restrict__ inv_rms_in,       // (B*N,)
    __mt_bfloat162* __restrict__ dx2,           // (B*N, C/2)
    float* __restrict__ dw_part,                // (n_chunks, C)
    int B, int N, int H, int D, int n_chunks
) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int BN = B * N;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D2 = D >> 1;
    const int bn_start = (static_cast<int64_t>(chunk) * BN) / n_chunks;
    const int bn_end   = (static_cast<int64_t>(chunk + 1) * BN) / n_chunks;

    extern __shared__ float smem[];
    float* s_dw = smem;                                       // C floats
    float* s_red = s_dw + C;                                  // 8 floats
    __mt_bfloat162* s_x2  = reinterpret_cast<__mt_bfloat162*>(s_red + 8); // C2 bf162
    __mt_bfloat162* s_dy2 = s_x2 + C2;                                    // C2 bf162

    for (int i = tid; i < C; i += 256) s_dw[i] = 0.0f;
    __syncthreads();

    const float invC = 1.0f / static_cast<float>(C);
    for (int bn = bn_start; bn < bn_end; ++bn) {
        const int b = (B == 1) ? 0 : (bn / N);
        const int n = (B == 1) ? bn : (bn - b * N);
        const __mt_bfloat162* xr = x2 + static_cast<int64_t>(bn) * C2;
        __mt_bfloat162* dxr = dx2 + static_cast<int64_t>(bn) * C2;
        const float ir = inv_rms_in[bn];

        float dot = 0.0f;
        for (int p = tid; p < C2; p += 256) {
            int h = p / D2;
            int d2 = p - h * D2;
            int c0 = p << 1;
            int64_t ip = (static_cast<int64_t>(b * H + h) * N + n) * D2 + d2;
            __mt_bfloat162 xv = xr[p];
            __mt_bfloat162 dv = dy_bhnd2[ip];
            s_x2[p] = xv;
            s_dy2[p] = dv;
            float x0 = __low2float(xv), x1 = __high2float(xv);
            float d0 = __low2float(dv), d1 = __high2float(dv);
            s_dw[c0]     += d0 * x0 * ir;
            s_dw[c0 + 1] += d1 * x1 * ir;
            dot = fmaf(d0 * w[c0], x0, fmaf(d1 * w[c0 + 1], x1, dot));
        }
        float tot = block_sum_256(dot, s_red, tid);
        float coeff = tot * ir * ir * invC;

        for (int p = tid; p < C2; p += 256) {
            int c0 = p << 1;
            __mt_bfloat162 xv = s_x2[p], dv = s_dy2[p];
            float dx0 = ir * (__low2float(dv)  * w[c0]     - __low2float(xv)  * coeff);
            float dx1 = ir * (__high2float(dv) * w[c0 + 1] - __high2float(xv) * coeff);
            dxr[p] = __floats2bfloat162_rn(dx0, dx1);
        }
    }
    __syncthreads();
    int64_t base = static_cast<int64_t>(chunk) * C;
    for (int i = tid; i < C; i += 256) dw_part[base + i] = s_dw[i];
}

// =========================================================================
// 10. Fast 128-bit Vectorized Transpose (B, H, N, D) <-> (B, N, H, D) (D % 8 == 0)
// =========================================================================
extern "C" __global__ void transpose_bhnd_bnhd_bf16(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    int B, int H, int N, int D8, int dir  // dir=0: BHND->BNHD, dir=1: BNHD->BHND
) {
    constexpr int TN = 16;
    const int n_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh - b * H;
    const int tid = threadIdx.x;
    const int total = TN * D8;
    for (int idx = tid; idx < total; idx += 256) {
        int dn = idx / D8;
        int d8 = idx - dn * D8;
        int n = n_tile * TN + dn;
        if (n < N) {
            int64_t p_bhnd = (static_cast<int64_t>(bh) * N + n) * D8 + d8;
            int64_t p_bnhd = ((static_cast<int64_t>(b) * N + n) * H + h) * D8 + d8;
            if (dir == 0) dst[p_bnhd] = src[p_bhnd];
            else          dst[p_bhnd] = src[p_bnhd];
        }
    }
}

// =========================================================================
// 11. Fast Pack/Unpack 16-byte-padded FSDP2 collectives (world ranks in 1 launch)
// =========================================================================
extern "C" __global__ void pack_unpack_pad16_u16(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int world, int64_t n_u16, int64_t pad_u16, int dir // dir=0: pack (w,n)->(w,pad), dir=1: unpack (w,pad)->(w,n)
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(world) * (dir == 0 ? pad_u16 : n_u16);
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    if (dir == 0) {
        for (int64_t i = idx; i < total; i += stride) {
            int64_t r = i / pad_u16;
            int64_t off = i - r * pad_u16;
            dst[i] = (off < n_u16) ? src[r * n_u16 + off] : static_cast<uint16_t>(0);
        }
    } else {
        for (int64_t i = idx; i < total; i += stride) {
            int64_t r = i / n_u16;
            int64_t off = i - r * n_u16;
            dst[i] = src[r * pad_u16 + off];
        }
    }
}

// =========================================================================
// Host Launch Wrappers
// =========================================================================
static inline bool is_u4_fast(int C, const void* p0, const void* p1 = nullptr, const void* p2 = nullptr, const void* p3 = nullptr) {
    if ((C & 7) != 0 || C <= 0 || C > 2304) return false;
    uintptr_t m = reinterpret_cast<uintptr_t>(p0) | reinterpret_cast<uintptr_t>(p1)
                | reinterpret_cast<uintptr_t>(p2) | reinterpret_cast<uintptr_t>(p3);
    return (m & 15u) == 0;
}

static inline int choose_frame_splits(int BT, int S) {
    if (BT == 25 && S >= 12) return 12;
    if (BT == 1) return (S < 300) ? S : 300;
    int ns = (240 + BT - 1) / BT;
    if (ns < 12) ns = 12;
    if (ns > S) ns = S;
    return ns;
}

extern "C" int launch_rmsnorm_fwd(
    const void* x, const void* w, void* y, void* inv_rms,
    int M, int C, float eps, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    if (M >= 8 && is_u4_fast(C, x, y)) {
        int threads = (((C >> 3) + 31) >> 5) << 5;
        int n_chunks = (M < 240) ? M : 240;
        rmsnorm_fwd_u4_bf16<<<n_chunks, threads, 0, st>>>(
            reinterpret_cast<const __mt_bfloat16*>(x),
            reinterpret_cast<const float*>(w),
            reinterpret_cast<__mt_bfloat16*>(y),
            reinterpret_cast<float*>(inv_rms),
            M, C, n_chunks, eps
        );
    } else {
        rmsnorm_fwd_bf16<<<M, 256, 0, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(x),
            reinterpret_cast<const float*>(w),
            reinterpret_cast<__mt_bfloat162*>(y),
            reinterpret_cast<float*>(inv_rms),
            M, C, eps
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_rmsnorm_bwd(
    const void* x, const void* dy, const void* w, const void* inv_rms,
    void* dx, void* dw_part, void* dw_out,
    int M, int C, int n_chunks, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    if (is_u4_fast(C, x, dy, dx)) {
        int threads = (((C >> 3) + 31) >> 5) << 5;
        rmsnorm_bwd_u4_bf16<<<n_chunks, threads, 0, st>>>(
            reinterpret_cast<const __mt_bfloat16*>(x),
            reinterpret_cast<const __mt_bfloat16*>(dy),
            reinterpret_cast<const float*>(w),
            reinterpret_cast<const float*>(inv_rms),
            reinterpret_cast<__mt_bfloat16*>(dx),
            reinterpret_cast<float*>(dw_part),
            M, C, n_chunks
        );
    } else {
        size_t smem = static_cast<size_t>(2 * C + 8) * sizeof(float);
        rmsnorm_bwd_bf16<<<n_chunks, 256, smem, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(x),
            reinterpret_cast<const __mt_bfloat162*>(dy),
            reinterpret_cast<const float*>(w),
            reinterpret_cast<const float*>(inv_rms),
            reinterpret_cast<__mt_bfloat162*>(dx),
            reinterpret_cast<float*>(dw_part),
            M, C, n_chunks
        );
    }
    reduce_chunks_1<<<(C + 255) / 256, 256, 0, st>>>(
        reinterpret_cast<const float*>(dw_part), reinterpret_cast<float*>(dw_out), n_chunks, C
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_recompute_ln_mod_u4(
    const void* x, const void* mean, const void* rstd, const void* shift, const void* scale,
    void* y, int BT, int S, int C, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int threads = (((C >> 3) + 31) >> 5) << 5;
    int n_splits = choose_frame_splits(BT, S);
    recompute_ln_mod_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
        reinterpret_cast<const __mt_bfloat16*>(x),
        reinterpret_cast<const float*>(mean),
        reinterpret_cast<const float*>(rstd),
        reinterpret_cast<const float*>(shift),
        reinterpret_cast<const float*>(scale),
        reinterpret_cast<__mt_bfloat16*>(y),
        BT, S, C, n_splits
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_ln_modulate_fwd_fmean(
    const void* x, const void* shift, const void* scale,
    void* y, void* mean, void* rstd,
    void* y_sum_part, void* x_frame_out,
    int BT, int S, int C, int T_frame, int S_frame, float eps, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int threads = (((C >> 3) + 31) >> 5) << 5;
    int n_splits = choose_frame_splits(BT, S);
    if (BT == T_frame && S == S_frame && x_frame_out != nullptr) {
        ln_modulate_fwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
            reinterpret_cast<const __mt_bfloat16*>(x), nullptr,
            reinterpret_cast<const float*>(shift),
            reinterpret_cast<const float*>(scale),
            nullptr,
            reinterpret_cast<__mt_bfloat16*>(y),
            reinterpret_cast<float*>(mean),
            reinterpret_cast<float*>(rstd),
            reinterpret_cast<float*>(y_sum_part),
            BT, S, C, n_splits, eps
        );
        reduce_frame_mean_bf16<<<dim3((C + 255) / 256, BT), 256, 0, st>>>(
            reinterpret_cast<const float*>(y_sum_part),
            reinterpret_cast<__mt_bfloat16*>(x_frame_out),
            BT, n_splits, C, 1.0f / static_cast<float>(S_frame)
        );
    } else {
        ln_modulate_fwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
            reinterpret_cast<const __mt_bfloat16*>(x), nullptr,
            reinterpret_cast<const float*>(shift),
            reinterpret_cast<const float*>(scale),
            nullptr,
            reinterpret_cast<__mt_bfloat16*>(y),
            reinterpret_cast<float*>(mean),
            reinterpret_cast<float*>(rstd),
            nullptr,
            BT, S, C, n_splits, eps
        );
        if (x_frame_out != nullptr) {
            int f_splits = choose_frame_splits(T_frame, S_frame);
            frame_mean_part_u4_bf16<<<dim3(T_frame, f_splits), threads, 0, st>>>(
                reinterpret_cast<const __mt_bfloat16*>(y),
                reinterpret_cast<float*>(y_sum_part),
                T_frame, S_frame, C, f_splits
            );
            reduce_frame_mean_bf16<<<dim3((C + 255) / 256, T_frame), 256, 0, st>>>(
                reinterpret_cast<const float*>(y_sum_part),
                reinterpret_cast<__mt_bfloat16*>(x_frame_out),
                T_frame, f_splits, C, 1.0f / static_cast<float>(S_frame)
            );
        }
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_ln_modulate_fwd(
    const void* x, const void* shift, const void* scale,
    void* y, void* mean, void* rstd,
    int BT, int S, int C, float eps, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    if (S >= 8 && is_u4_fast(C, x, y)) {
        int threads = (((C >> 3) + 31) >> 5) << 5;
        int n_splits = choose_frame_splits(BT, S);
        ln_modulate_fwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
            reinterpret_cast<const __mt_bfloat16*>(x), nullptr,
            reinterpret_cast<const float*>(shift),
            reinterpret_cast<const float*>(scale),
            nullptr,
            reinterpret_cast<__mt_bfloat16*>(y),
            reinterpret_cast<float*>(mean),
            reinterpret_cast<float*>(rstd),
            nullptr,
            BT, S, C, n_splits, eps
        );
    } else {
        ln_modulate_fwd_bf16<<<BT * S, 256, 0, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(x),
            reinterpret_cast<const float*>(shift),
            reinterpret_cast<const float*>(scale),
            reinterpret_cast<__mt_bfloat162*>(y),
            reinterpret_cast<float*>(mean),
            reinterpret_cast<float*>(rstd),
            BT, S, C, eps
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_add_ln_modulate_fwd(
    const void* a, const void* b, const void* shift, const void* scale,
    void* s_out, void* y, void* mean, void* rstd,
    int BT, int S, int C, float eps, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int threads = (((C >> 3) + 31) >> 5) << 5;
    int n_splits = choose_frame_splits(BT, S);
    ln_modulate_fwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
        reinterpret_cast<const __mt_bfloat16*>(a),
        reinterpret_cast<const __mt_bfloat16*>(b),
        reinterpret_cast<const float*>(shift),
        reinterpret_cast<const float*>(scale),
        reinterpret_cast<__mt_bfloat16*>(s_out),
        reinterpret_cast<__mt_bfloat16*>(y),
        reinterpret_cast<float*>(mean),
        reinterpret_cast<float*>(rstd),
        nullptr,
        BT, S, C, n_splits, eps
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_ln_modulate_bwd_fmean(
    const void* x, const void* dy, const void* dy_frame_add,
    const void* scale, const void* mean, const void* rstd,
    void* dx, void* dsh_part, void* dsc_part, void* dsh_out, void* dsc_out,
    int BT, int S, int C, int n_splits, int S_frame, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int threads = (((C >> 3) + 31) >> 5) << 5;
    ln_modulate_bwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
        reinterpret_cast<const __mt_bfloat16*>(x),
        reinterpret_cast<const __mt_bfloat16*>(dy),
        nullptr,
        reinterpret_cast<const float*>(dy_frame_add),
        reinterpret_cast<const float*>(scale),
        reinterpret_cast<const float*>(mean),
        reinterpret_cast<const float*>(rstd),
        reinterpret_cast<__mt_bfloat16*>(dx),
        reinterpret_cast<float*>(dsh_part),
        reinterpret_cast<float*>(dsc_part),
        BT, S, C, n_splits, S_frame
    );
    reduce_frame_splits_2<<<dim3((C + 255) / 256, BT), 256, 0, st>>>(
        reinterpret_cast<const float*>(dsh_part),
        reinterpret_cast<const float*>(dsc_part),
        reinterpret_cast<float*>(dsh_out),
        reinterpret_cast<float*>(dsc_out),
        BT, n_splits, C
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_ln_modulate_bwd(
    const void* x, const void* dy, const void* scale, const void* mean, const void* rstd,
    void* dx, void* dsh_part, void* dsc_part, void* dsh_out, void* dsc_out,
    int BT, int S, int C, int n_splits, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    if (is_u4_fast(C, x, dy, dx)) {
        int threads = (((C >> 3) + 31) >> 5) << 5;
        ln_modulate_bwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
            reinterpret_cast<const __mt_bfloat16*>(x),
            reinterpret_cast<const __mt_bfloat16*>(dy),
            nullptr,
            nullptr,
            reinterpret_cast<const float*>(scale),
            reinterpret_cast<const float*>(mean),
            reinterpret_cast<const float*>(rstd),
            reinterpret_cast<__mt_bfloat16*>(dx),
            reinterpret_cast<float*>(dsh_part),
            reinterpret_cast<float*>(dsc_part),
            BT, S, C, n_splits, S
        );
    } else {
        size_t smem = static_cast<size_t>(3 * C + 16) * sizeof(float);
        ln_modulate_bwd_bf16<<<dim3(BT, n_splits), 256, smem, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(x),
            reinterpret_cast<const __mt_bfloat162*>(dy),
            reinterpret_cast<const float*>(scale),
            reinterpret_cast<const float*>(mean),
            reinterpret_cast<const float*>(rstd),
            reinterpret_cast<__mt_bfloat162*>(dx),
            reinterpret_cast<float*>(dsh_part),
            reinterpret_cast<float*>(dsc_part),
            BT, S, C, n_splits
        );
    }
    reduce_frame_splits_2<<<dim3((C + 255) / 256, BT), 256, 0, st>>>(
        reinterpret_cast<const float*>(dsh_part),
        reinterpret_cast<const float*>(dsc_part),
        reinterpret_cast<float*>(dsh_out),
        reinterpret_cast<float*>(dsc_out),
        BT, n_splits, C
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_add_ln_modulate_bwd(
    const void* x, const void* dy, const void* ds_res, const void* scale, const void* mean, const void* rstd,
    void* dx, void* dsh_part, void* dsc_part, void* dsh_out, void* dsc_out,
    int BT, int S, int C, int n_splits, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int threads = (((C >> 3) + 31) >> 5) << 5;
    ln_modulate_bwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
        reinterpret_cast<const __mt_bfloat16*>(x),
        reinterpret_cast<const __mt_bfloat16*>(dy),
        reinterpret_cast<const __mt_bfloat16*>(ds_res),
        nullptr,
        reinterpret_cast<const float*>(scale),
        reinterpret_cast<const float*>(mean),
        reinterpret_cast<const float*>(rstd),
        reinterpret_cast<__mt_bfloat16*>(dx),
        reinterpret_cast<float*>(dsh_part),
        reinterpret_cast<float*>(dsc_part),
        BT, S, C, n_splits, S
    );
    reduce_frame_splits_2<<<dim3((C + 255) / 256, BT), 256, 0, st>>>(
        reinterpret_cast<const float*>(dsh_part),
        reinterpret_cast<const float*>(dsc_part),
        reinterpret_cast<float*>(dsh_out),
        reinterpret_cast<float*>(dsc_out),
        BT, n_splits, C
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gate_residual_fwd(
    const void* x, const void* y, const void* gate, const void* add_bias, void* out,
    int BT, int S, int C, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    if (S >= 8 && is_u4_fast(C, x, y, out, add_bias)) {
        int threads = (((C >> 3) + 31) >> 5) << 5;
        int n_splits = choose_frame_splits(BT, S);
        gate_residual_fwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
            reinterpret_cast<const __mt_bfloat16*>(x),
            reinterpret_cast<const __mt_bfloat16*>(y),
            reinterpret_cast<const float*>(gate),
            reinterpret_cast<const __mt_bfloat16*>(add_bias),
            reinterpret_cast<__mt_bfloat16*>(out),
            BT, S, C, n_splits
        );
    } else {
        gate_residual_fwd_bf16<<<dim3(S, BT), 256, 0, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(x),
            reinterpret_cast<const __mt_bfloat162*>(y),
            reinterpret_cast<const float*>(gate),
            reinterpret_cast<__mt_bfloat162*>(out),
            BT, S, C
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gate_residual_bwd(
    const void* dout, const void* y, const void* gate,
    void* dy, void* dg_part, void* dg_out, void* db_part, void* db_out,
    int BT, int S, int C, int n_splits, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    if (is_u4_fast(C, dout, y, dy)) {
        int threads = (((C >> 3) + 31) >> 5) << 5;
        gate_residual_bwd_u4_bf16<<<dim3(BT, n_splits), threads, 0, st>>>(
            reinterpret_cast<const __mt_bfloat16*>(dout),
            reinterpret_cast<const __mt_bfloat16*>(y),
            reinterpret_cast<const float*>(gate),
            reinterpret_cast<__mt_bfloat16*>(dy),
            reinterpret_cast<float*>(dg_part),
            reinterpret_cast<float*>(db_part),
            BT, S, C, n_splits
        );
    } else {
        size_t smem = static_cast<size_t>(C) * sizeof(float);
        gate_residual_bwd_bf16<<<dim3(BT, n_splits), 256, smem, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(dout),
            reinterpret_cast<const __mt_bfloat162*>(y),
            reinterpret_cast<const float*>(gate),
            reinterpret_cast<__mt_bfloat162*>(dy),
            reinterpret_cast<float*>(dg_part),
            BT, S, C, n_splits
        );
    }
    reduce_frame_splits_1<<<dim3((C + 255) / 256, BT), 256, 0, st>>>(
        reinterpret_cast<const float*>(dg_part),
        reinterpret_cast<float*>(dg_out),
        BT, n_splits, C
    );
    if (db_part != nullptr && db_out != nullptr) {
        reduce_chunks_1<<<(C + 255) / 256, 256, 0, st>>>(
            reinterpret_cast<const float*>(db_part),
            reinterpret_cast<float*>(db_out),
            BT * n_splits, C
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_silu_mul_fwd(
    const void* out, const void* g, void* y, int64_t n2, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    uintptr_t m = reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(g) | reinterpret_cast<uintptr_t>(y);
    if ((n2 & 3) == 0 && (m & 15u) == 0) {
        int64_t n8 = n2 >> 2;
        int grid = static_cast<int>((n8 + 255) / 256);
        silu_mul_fwd_u4_bf16<<<grid, 256, 0, st>>>(
            reinterpret_cast<const uint4*>(out),
            reinterpret_cast<const uint4*>(g),
            reinterpret_cast<uint4*>(y),
            n8
        );
    } else {
        int grid = static_cast<int>((n2 + 255) / 256);
        silu_mul_fwd_bf16<<<grid, 256, 0, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(out),
            reinterpret_cast<const __mt_bfloat162*>(g),
            reinterpret_cast<__mt_bfloat162*>(y),
            n2
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_silu_mul_bwd_y(
    const void* dy, const void* out, const void* g, void* dout, void* dg, void* y_out,
    int64_t n2, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    uintptr_t m = reinterpret_cast<uintptr_t>(dy) | reinterpret_cast<uintptr_t>(out)
                | reinterpret_cast<uintptr_t>(g) | reinterpret_cast<uintptr_t>(dout)
                | reinterpret_cast<uintptr_t>(dg) | reinterpret_cast<uintptr_t>(y_out);
    if ((n2 & 3) == 0 && (m & 15u) == 0) {
        int64_t n8 = n2 >> 2;
        int grid = static_cast<int>((n8 + 255) / 256);
        silu_mul_bwd_u4_bf16<<<grid, 256, 0, st>>>(
            reinterpret_cast<const uint4*>(dy),
            reinterpret_cast<const uint4*>(out),
            reinterpret_cast<const uint4*>(g),
            reinterpret_cast<uint4*>(dout),
            reinterpret_cast<uint4*>(dg),
            reinterpret_cast<uint4*>(y_out),
            n8
        );
    } else {
        int grid = static_cast<int>((n2 + 255) / 256);
        silu_mul_bwd_bf16<<<grid, 256, 0, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(dy),
            reinterpret_cast<const __mt_bfloat162*>(out),
            reinterpret_cast<const __mt_bfloat162*>(g),
            reinterpret_cast<__mt_bfloat162*>(dout),
            reinterpret_cast<__mt_bfloat162*>(dg),
            reinterpret_cast<__mt_bfloat162*>(y_out),
            n2
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_silu_mul_bwd(
    const void* dy, const void* out, const void* g, void* dout, void* dg,
    int64_t n2, void* stream_ptr
) {
    return launch_silu_mul_bwd_y(dy, out, g, dout, dg, nullptr, n2, stream_ptr);
}

extern "C" int launch_softmax_main_prep_fwd(
    const void* qkv, const void* q_nw, const void* k_nw,
    const void* rope_cos, const void* rope_sin,
    void* q_out, void* k_out, void* v_out, void* q_ir, void* k_ir,
    int B, int N, int H, int D, float norm_eps, int has_norm, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    softmax_main_prep_fwd_bf16<<<B * N, 256, 0, st>>>(
        reinterpret_cast<const __mt_bfloat162*>(qkv),
        reinterpret_cast<const float*>(q_nw),
        reinterpret_cast<const float*>(k_nw),
        reinterpret_cast<const float*>(rope_cos),
        reinterpret_cast<const float*>(rope_sin),
        reinterpret_cast<__mt_bfloat162*>(q_out),
        reinterpret_cast<__mt_bfloat162*>(k_out),
        reinterpret_cast<__mt_bfloat162*>(v_out),
        reinterpret_cast<float*>(q_ir),
        reinterpret_cast<float*>(k_ir),
        B, N, H, D, norm_eps, has_norm
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_softmax_main_prep_bwd(
    const void* qkv, const void* q_nw, const void* k_nw,
    const void* q_ir, const void* k_ir, const void* rope_cos, const void* rope_sin,
    const void* dq_in, const void* dk_in, const void* dv_in,
    void* dqkv, void* dq_part, void* dk_part, void* dq_out, void* dk_out,
    int B, int N, int H, int D, int has_norm, int n_chunks, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int C = H * D;
    size_t smem = has_norm ? (static_cast<size_t>(4 * C + 16) * sizeof(float)) : 64;
    softmax_main_prep_bwd_bf16<<<n_chunks, 256, smem, st>>>(
        reinterpret_cast<const __mt_bfloat162*>(qkv),
        reinterpret_cast<const float*>(q_nw),
        reinterpret_cast<const float*>(k_nw),
        reinterpret_cast<const float*>(q_ir),
        reinterpret_cast<const float*>(k_ir),
        reinterpret_cast<const float*>(rope_cos),
        reinterpret_cast<const float*>(rope_sin),
        reinterpret_cast<const __mt_bfloat162*>(dq_in),
        reinterpret_cast<const __mt_bfloat162*>(dk_in),
        reinterpret_cast<const __mt_bfloat162*>(dv_in),
        reinterpret_cast<__mt_bfloat162*>(dqkv),
        reinterpret_cast<float*>(dq_part),
        reinterpret_cast<float*>(dk_part),
        B, N, H, D, has_norm, n_chunks
    );
    if (has_norm) {
        reduce_chunks_2<<<(C + 255) / 256, 256, 0, st>>>(
            reinterpret_cast<const float*>(dq_part), reinterpret_cast<const float*>(dk_part),
            reinterpret_cast<float*>(dq_out), reinterpret_cast<float*>(dk_out), n_chunks, C
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_bf16_col_sum(
    const void* dy, void* partial, void* out,
    int M, int C, int n_splits, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int C8 = C >> 3;
    dim3 grid((C8 + 31) / 32, n_splits);
    bf16_col_sum_part<<<grid, 256, 0, st>>>(
        reinterpret_cast<const uint4*>(dy), reinterpret_cast<float*>(partial), M, C8, n_splits
    );
    bf16_col_sum_reduce<<<(C + 255) / 256, 256, 0, st>>>(
        reinterpret_cast<const float*>(partial), reinterpret_cast<__mt_bfloat16*>(out), n_splits, C
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_add_silu_mul_fwd(
    const void* a, const void* b, const void* g, void* s, void* y,
    int64_t n2, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    uintptr_t m = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)
                | reinterpret_cast<uintptr_t>(g) | reinterpret_cast<uintptr_t>(s) | reinterpret_cast<uintptr_t>(y);
    if ((n2 & 3) == 0 && (m & 15u) == 0) {
        int64_t n8 = n2 >> 2;
        int blocks = static_cast<int>((n8 + 255) / 256);
        add_silu_mul_fwd_u4_bf16<<<blocks, 256, 0, st>>>(
            reinterpret_cast<const uint4*>(a),
            reinterpret_cast<const uint4*>(b),
            reinterpret_cast<const uint4*>(g),
            reinterpret_cast<uint4*>(s),
            reinterpret_cast<uint4*>(y),
            n8
        );
    } else {
        int blocks = static_cast<int>((n2 + 255) / 256);
        add_silu_mul_fwd_bf16<<<blocks, 256, 0, st>>>(
            reinterpret_cast<const __mt_bfloat162*>(a),
            reinterpret_cast<const __mt_bfloat162*>(b),
            reinterpret_cast<const __mt_bfloat162*>(g),
            reinterpret_cast<__mt_bfloat162*>(s),
            reinterpret_cast<__mt_bfloat162*>(y),
            n2
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_rmsnorm_bnhd_to_bhnd_fwd(
    const void* x, const void* w, void* y_bhnd, void* inv_rms,
    int B, int N, int H, int D, float eps, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    rmsnorm_bnhd_to_bhnd_fwd_bf16<<<B * N, 256, 0, st>>>(
        reinterpret_cast<const __mt_bfloat162*>(x),
        reinterpret_cast<const float*>(w),
        reinterpret_cast<__mt_bfloat162*>(y_bhnd),
        reinterpret_cast<float*>(inv_rms),
        B, N, H, D, eps
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_rmsnorm_bhnd_to_bnhd_bwd(
    const void* x, const void* dy_bhnd, const void* w, const void* inv_rms,
    void* dx, void* dw_part, void* dw_out,
    int B, int N, int H, int D, int n_chunks, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int C = H * D;
    size_t smem = static_cast<size_t>(2 * C + 8) * sizeof(float);
    rmsnorm_bhnd_to_bnhd_bwd_bf16<<<n_chunks, 256, smem, st>>>(
        reinterpret_cast<const __mt_bfloat162*>(x),
        reinterpret_cast<const __mt_bfloat162*>(dy_bhnd),
        reinterpret_cast<const float*>(w),
        reinterpret_cast<const float*>(inv_rms),
        reinterpret_cast<__mt_bfloat162*>(dx),
        reinterpret_cast<float*>(dw_part),
        B, N, H, D, n_chunks
    );
    reduce_chunks_1<<<(C + 255) / 256, 256, 0, st>>>(
        reinterpret_cast<const float*>(dw_part), reinterpret_cast<float*>(dw_out), n_chunks, C
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_transpose_bhnd_bnhd(
    const void* src, void* dst, int B, int H, int N, int D, int dir, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int D8 = D >> 3;
    dim3 grid((N + 15) / 16, B * H);
    transpose_bhnd_bnhd_bf16<<<grid, 256, 0, st>>>(
        reinterpret_cast<const uint4*>(src), reinterpret_cast<uint4*>(dst), B, H, N, D8, dir
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_pack_unpack_pad16_u16(
    const void* src, void* dst, int world, int64_t n_u16, int64_t pad_u16, int dir, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int64_t total = static_cast<int64_t>(world) * (dir == 0 ? pad_u16 : n_u16);
    int blocks = static_cast<int>(min(static_cast<int64_t>(4096), (total + 255) / 256));
    if (blocks <= 0) return 0;
    pack_unpack_pad16_u16<<<blocks, 256, 0, st>>>(
        reinterpret_cast<const uint16_t*>(src), reinterpret_cast<uint16_t*>(dst),
        world, n_u16, pad_u16, dir
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_softmax_cam_prep_fwd(
    const void* qkv, const void* q_nw, const void* k_nw,
    const void* P_T, const void* P_inv, const void* rope_cos, const void* rope_sin,
    void* q_out, void* k_out, void* v_out, void* q_ir, void* k_ir,
    int B, int N, int H, int D, float norm_eps, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    softmax_cam_prep_fwd_bf16<<<B * N, 256, 0, st>>>(
        reinterpret_cast<const __mt_bfloat162*>(qkv),
        reinterpret_cast<const float*>(q_nw),
        reinterpret_cast<const float*>(k_nw),
        reinterpret_cast<const float*>(P_T),
        reinterpret_cast<const float*>(P_inv),
        reinterpret_cast<const float*>(rope_cos),
        reinterpret_cast<const float*>(rope_sin),
        reinterpret_cast<__mt_bfloat162*>(q_out),
        reinterpret_cast<__mt_bfloat162*>(k_out),
        reinterpret_cast<__mt_bfloat162*>(v_out),
        reinterpret_cast<float*>(q_ir),
        reinterpret_cast<float*>(k_ir),
        B, N, H, D, norm_eps
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_softmax_cam_prep_bwd(
    const void* qkv, const void* q_nw, const void* k_nw,
    const void* P_T, const void* P_inv, const void* rope_cos, const void* rope_sin,
    const void* q_ir, const void* k_ir,
    const void* dq_in, const void* dk_in, const void* dv_in,
    void* dqkv, void* dq_part, void* dk_part, void* dq_out, void* dk_out,
    int B, int N, int H, int D, int n_chunks, void* stream_ptr
) {
    musaStream_t st = reinterpret_cast<musaStream_t>(stream_ptr);
    int C = H * D;
    size_t smem = static_cast<size_t>(4 * C + 176) * sizeof(float);
    softmax_cam_prep_bwd_bf16<<<n_chunks, 256, smem, st>>>(
        reinterpret_cast<const __mt_bfloat162*>(qkv),
        reinterpret_cast<const float*>(q_nw),
        reinterpret_cast<const float*>(k_nw),
        reinterpret_cast<const float*>(P_T),
        reinterpret_cast<const float*>(P_inv),
        reinterpret_cast<const float*>(rope_cos),
        reinterpret_cast<const float*>(rope_sin),
        reinterpret_cast<const float*>(q_ir),
        reinterpret_cast<const float*>(k_ir),
        reinterpret_cast<const __mt_bfloat162*>(dq_in),
        reinterpret_cast<const __mt_bfloat162*>(dk_in),
        reinterpret_cast<const __mt_bfloat162*>(dv_in),
        reinterpret_cast<__mt_bfloat162*>(dqkv),
        reinterpret_cast<float*>(dq_part),
        reinterpret_cast<float*>(dk_part),
        B, N, H, D, n_chunks
    );
    reduce_chunks_2<<<(C + 255) / 256, 256, 0, st>>>(
        reinterpret_cast<const float*>(dq_part), reinterpret_cast<const float*>(dk_part),
        reinterpret_cast<float*>(dq_out), reinterpret_cast<float*>(dk_out), n_chunks, C
    );
    return static_cast<int>(musaGetLastError());
}
"""

_LIB = None
_LIB_LOCK = threading.Lock()


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        h = hashlib.sha256(_MUSA_SRC.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join(tempfile.gettempdir(), "sana_musa_ops_cache")
        os.makedirs(cache_dir, exist_ok=True)
        so_path = os.path.join(cache_dir, f"block_fusion_s5000_{h}.so")
        if not os.path.exists(so_path):
            pid = os.getpid()
            mu_tmp = os.path.join(cache_dir, f"block_fusion_{h}_{pid}.mu")
            so_tmp = os.path.join(cache_dir, f"block_fusion_{h}_{pid}.so")
            with open(mu_tmp, "w", encoding="utf-8") as f:
                f.write(_MUSA_SRC)
            mcc = "/usr/local/musa/bin/mcc"
            if not os.path.exists(mcc):
                mcc = "/usr/local/musa-5.2/bin/mcc"
            cmd = [
                mcc, "-shared", "-fPIC", "-O3",
                "--offload-arch=mp_31",
                "-fmusa-flush-denormals-to-zero",
                "-I/usr/local/musa/include",
                mu_tmp, "-o", so_tmp,
                "-L/usr/local/musa/lib", "-lmusart",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"mcc failed compiling block_fusion:\n{res.stderr}\n{res.stdout}")
            os.replace(so_tmp, so_path)
            try:
                os.remove(mu_tmp)
            except OSError:
                pass

        lib = ctypes.CDLL(so_path)
        p = ctypes.c_void_p
        i = ctypes.c_int
        i64 = ctypes.c_int64
        f = ctypes.c_float

        lib.launch_rmsnorm_fwd.argtypes = [p, p, p, p, i, i, f, p]
        lib.launch_rmsnorm_fwd.restype = i
        lib.launch_rmsnorm_bwd.argtypes = [p, p, p, p, p, p, p, i, i, i, p]
        lib.launch_rmsnorm_bwd.restype = i

        lib.launch_ln_modulate_fwd.argtypes = [p, p, p, p, p, p, i, i, i, f, p]
        lib.launch_ln_modulate_fwd.restype = i
        lib.launch_ln_modulate_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, i, i, i, i, p]
        lib.launch_ln_modulate_bwd.restype = i

        lib.launch_gate_residual_fwd.argtypes = [p, p, p, p, p, i, i, i, p]
        lib.launch_gate_residual_fwd.restype = i
        lib.launch_gate_residual_bwd.argtypes = [p, p, p, p, p, p, p, p, i, i, i, i, p]
        lib.launch_gate_residual_bwd.restype = i

        lib.launch_silu_mul_fwd.argtypes = [p, p, p, i64, p]
        lib.launch_silu_mul_fwd.restype = i
        lib.launch_silu_mul_bwd.argtypes = [p, p, p, p, p, i64, p]
        lib.launch_silu_mul_bwd.restype = i
        lib.launch_silu_mul_bwd_y.argtypes = [p, p, p, p, p, p, i64, p]
        lib.launch_silu_mul_bwd_y.restype = i

        lib.launch_softmax_main_prep_fwd.argtypes = [p, p, p, p, p, p, p, p, p, p, i, i, i, i, f, i, p]
        lib.launch_softmax_main_prep_fwd.restype = i
        lib.launch_softmax_main_prep_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, i, i, p]
        lib.launch_softmax_main_prep_bwd.restype = i

        lib.launch_softmax_cam_prep_fwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, f, p]
        lib.launch_softmax_cam_prep_fwd.restype = i
        lib.launch_softmax_cam_prep_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, i, p]
        lib.launch_softmax_cam_prep_bwd.restype = i

        lib.launch_bf16_col_sum.argtypes = [p, p, p, i, i, i, p]
        lib.launch_bf16_col_sum.restype = i
        lib.launch_add_silu_mul_fwd.argtypes = [p, p, p, p, p, i64, p]
        lib.launch_add_silu_mul_fwd.restype = i
        lib.launch_rmsnorm_bnhd_to_bhnd_fwd.argtypes = [p, p, p, p, i, i, i, i, f, p]
        lib.launch_rmsnorm_bnhd_to_bhnd_fwd.restype = i
        lib.launch_rmsnorm_bhnd_to_bnhd_bwd.argtypes = [p, p, p, p, p, p, p, i, i, i, i, i, p]
        lib.launch_rmsnorm_bhnd_to_bnhd_bwd.restype = i
        lib.launch_transpose_bhnd_bnhd.argtypes = [p, p, i, i, i, i, i, p]
        lib.launch_transpose_bhnd_bnhd.restype = i
        lib.launch_pack_unpack_pad16_u16.argtypes = [p, p, i, i64, i64, i, p]
        lib.launch_pack_unpack_pad16_u16.restype = i
        lib.launch_add_ln_modulate_fwd.argtypes = [p, p, p, p, p, p, p, p, i, i, i, f, p]
        lib.launch_add_ln_modulate_fwd.restype = i
        lib.launch_add_ln_modulate_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, p]
        lib.launch_add_ln_modulate_bwd.restype = i
        lib.launch_recompute_ln_mod_u4.argtypes = [p, p, p, p, p, p, i, i, i, p]
        lib.launch_recompute_ln_mod_u4.restype = i
        lib.launch_ln_modulate_fwd_fmean.argtypes = [p, p, p, p, p, p, p, p, i, i, i, i, i, f, p]
        lib.launch_ln_modulate_fwd_fmean.restype = i
        lib.launch_ln_modulate_bwd_fmean.argtypes = [p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, i, p]
        lib.launch_ln_modulate_bwd_fmean.restype = i

        _LIB = lib
        return _LIB


def _stream(device) -> int:
    return int(torch.musa.current_stream(device).musa_stream)


# ---------------------------------------------------------------------------
# 1. Fused RMSNorm
# ---------------------------------------------------------------------------
class _FusedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        x_c = x.contiguous()
        w_f = weight.float().contiguous()
        orig_shape = x_c.shape
        C = orig_shape[-1]
        M = x_c.numel() // C
        y = torch.empty_like(x_c)
        inv_rms = torch.empty((M,), device=x_c.device, dtype=torch.float32)
        rc = _load_lib().launch_rmsnorm_fwd(
            x_c.data_ptr(), w_f.data_ptr(), y.data_ptr(), inv_rms.data_ptr(),
            M, C, float(eps), _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"rmsnorm_fwd failed rc={rc}")
        ctx.save_for_backward(x_c, w_f, inv_rms)
        ctx.meta = (M, C, weight.dtype)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, w_f, inv_rms = ctx.saved_tensors
        M, C, w_dt = ctx.meta
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        n_chunks = min(240, max(1, M))
        dw_part = torch.empty((n_chunks, C), device=x_c.device, dtype=torch.float32)
        dw_out = torch.empty((C,), device=x_c.device, dtype=torch.float32)
        rc = _load_lib().launch_rmsnorm_bwd(
            x_c.data_ptr(), dy_c.data_ptr(), w_f.data_ptr(), inv_rms.data_ptr(),
            dx.data_ptr(), dw_part.data_ptr(), dw_out.data_ptr(),
            M, C, n_chunks, _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"rmsnorm_bwd failed rc={rc}")
        return dx, dw_out.to(w_dt), None


def can_use_fused_rmsnorm(x: torch.Tensor, norm_dim: int) -> bool:
    if os.environ.get("SANA_OPT_BLOCK_FUSION", "1") == "0":
        return False
    C = x.shape[-1] if x.ndim >= 1 else 0
    return (
        x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and (norm_dim == -1 or norm_dim == x.ndim - 1)
        and C >= 2
        and (C % 2 == 0)
        and C <= 10240
    )


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _FusedRMSNormFn.apply(x, weight, eps)


def _choose_frame_splits(BT: int, S: int) -> int:
    if S < 16:
        return 1
    if S < 32:
        return 4
    if BT == 25 and S >= 12:
        return 12
    if BT == 1:
        return min(S, 300)
    return min(S, max(8, (240 + BT - 1) // BT))


# ---------------------------------------------------------------------------
# 2. Fused LayerNorm(no affine) + t2i_modulate
# ---------------------------------------------------------------------------
def recompute_ln_mod_u4(
    x_c: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor,
    sh_f: torch.Tensor, sc_f: torch.Tensor, BT: int, S: int
) -> torch.Tensor:
    y = torch.empty_like(x_c)
    C = x_c.shape[-1]
    rc = _load_lib().launch_recompute_ln_mod_u4(
        x_c.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
        sh_f.data_ptr(), sc_f.data_ptr(), y.data_ptr(),
        BT, S, C, _stream(x_c.device),
    )
    if rc != 0:
        raise RuntimeError(f"recompute_ln_mod_u4 failed rc={rc}")
    return y


class _FusedLNModulateFMeanFn(torch.autograd.Function):
    """Fused LayerNorm+modulate that ALSO computes x_frame = y.view(B,T_frame,S_frame,C).mean(dim=2)
    and absorbs d_x_frame directly inside ln_modulate_bwd_u4 (zero FastDivmod/AddOp!)."""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
        T: int, T_frame: int, eps: float, stats_out: list | None = None
    ):
        x_c = x.contiguous()
        B, N, C = x_c.shape
        S = N // T
        BT = B * T
        BT_f = B * T_frame
        S_f = N // T_frame
        sh_f = shift.float().reshape(BT, C).contiguous()
        sc_f = scale.float().reshape(BT, C).contiguous()
        y = torch.empty_like(x_c)
        mean = torch.empty((BT * S,), device=x_c.device, dtype=torch.float32)
        rstd = torch.empty((BT * S,), device=x_c.device, dtype=torch.float32)
        f_splits = _choose_frame_splits(BT_f, S_f)
        y_sum_part = torch.empty((BT_f, f_splits, C), device=x_c.device, dtype=torch.float32)
        x_frame = torch.empty((B, T_frame, C), device=x_c.device, dtype=torch.bfloat16)
        rc = _load_lib().launch_ln_modulate_fwd_fmean(
            x_c.data_ptr(), sh_f.data_ptr(), sc_f.data_ptr(),
            y.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            y_sum_part.data_ptr(), x_frame.data_ptr(),
            BT, S, C, BT_f, S_f, float(eps), _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"ln_modulate_fwd_fmean failed rc={rc}")
        del y_sum_part
        if stats_out is not None:
            stats_out.append((x_c, mean, rstd, sh_f, sc_f, BT, S))
        ctx.save_for_backward(x_c, sc_f, mean, rstd)
        ctx.meta = (B, T, S, C, BT_f, S_f, shift.shape, shift.dtype, scale.dtype)
        return y, x_frame

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dx_frame: torch.Tensor | None):
        x_c, sc_f, mean, rstd = ctx.saved_tensors
        B, T, S, C, BT_f, S_f, sh_shape, sh_dt, sc_dt = ctx.meta
        BT = B * T
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        n_splits = _choose_frame_splits(BT, S)
        dsh_part = torch.empty((BT, n_splits, C), device=x_c.device, dtype=torch.float32)
        dsc_part = torch.empty((BT, n_splits, C), device=x_c.device, dtype=torch.float32)
        dsh_out = torch.empty((BT, C), device=x_c.device, dtype=torch.float32)
        dsc_out = torch.empty((BT, C), device=x_c.device, dtype=torch.float32)
        if dx_frame is not None:
            dy_f = (dx_frame.float().reshape(BT_f, C) * (1.0 / float(S_f))).contiguous()
        else:
            dy_f = None
        rc = _load_lib().launch_ln_modulate_bwd_fmean(
            x_c.data_ptr(), dy_c.data_ptr(),
            dy_f.data_ptr() if dy_f is not None else 0,
            sc_f.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            dx.data_ptr(), dsh_part.data_ptr(), dsc_part.data_ptr(),
            dsh_out.data_ptr(), dsc_out.data_ptr(),
            BT, S, C, n_splits, S_f, _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"ln_modulate_bwd_fmean failed rc={rc}")
        return dx, dsh_out.view(sh_shape).to(sh_dt), dsc_out.view(sh_shape).to(sc_dt), None, None, None, None


def fused_layernorm_modulate_fmean(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
    T: int, T_frame: int, eps: float = 1e-6, stats_out: list | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return _FusedLNModulateFMeanFn.apply(x, shift, scale, T, T_frame, eps, stats_out)


class _FusedLNModulateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, T: int, eps: float, stats_out: list | None = None):
        # x: (B, N, C) BF16; shift, scale: (B, T, 1, C) BF16/FP32 -> y: (B, N, C) BF16
        x_c = x.contiguous()
        B, N, C = x_c.shape
        S = N // T
        BT = B * T
        sh_f = shift.float().reshape(BT, C).contiguous()
        sc_f = scale.float().reshape(BT, C).contiguous()
        y = torch.empty_like(x_c)
        mean = torch.empty((BT * S,), device=x_c.device, dtype=torch.float32)
        rstd = torch.empty((BT * S,), device=x_c.device, dtype=torch.float32)
        rc = _load_lib().launch_ln_modulate_fwd(
            x_c.data_ptr(), sh_f.data_ptr(), sc_f.data_ptr(),
            y.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            BT, S, C, float(eps), _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"ln_modulate_fwd failed rc={rc}")
        if stats_out is not None:
            stats_out.append((x_c, mean, rstd, sh_f, sc_f, BT, S))
        ctx.save_for_backward(x_c, sc_f, mean, rstd)
        ctx.meta = (B, T, S, C, shift.shape, shift.dtype, scale.dtype)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, sc_f, mean, rstd = ctx.saved_tensors
        B, T, S, C, sh_shape, sh_dt, sc_dt = ctx.meta
        BT = B * T
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        n_splits = _choose_frame_splits(BT, S)
        dsh_part = torch.empty((BT, n_splits, C), device=x_c.device, dtype=torch.float32)
        dsc_part = torch.empty((BT, n_splits, C), device=x_c.device, dtype=torch.float32)
        dsh_out = torch.empty((BT, C), device=x_c.device, dtype=torch.float32)
        dsc_out = torch.empty((BT, C), device=x_c.device, dtype=torch.float32)
        rc = _load_lib().launch_ln_modulate_bwd(
            x_c.data_ptr(), dy_c.data_ptr(), sc_f.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            dx.data_ptr(), dsh_part.data_ptr(), dsc_part.data_ptr(),
            dsh_out.data_ptr(), dsc_out.data_ptr(),
            BT, S, C, n_splits, _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"ln_modulate_bwd failed rc={rc}")
        return dx, dsh_out.view(sh_shape).to(sh_dt), dsc_out.view(sh_shape).to(sc_dt), None, None, None


def fused_layernorm_modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, T: int, eps: float = 1e-6, stats_out: list | None = None
) -> torch.Tensor:
    return _FusedLNModulateFn.apply(x, shift, scale, T, eps, stats_out)


class _FusedAddLNModulateFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, a: torch.Tensor, b: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
        T: int, eps: float, stats_out: list | None = None
    ):
        a_c = a.contiguous()
        b_c = b.contiguous()
        B, N, C = a_c.shape
        S = N // T
        BT = B * T
        sh_f = shift.float().reshape(BT, C).contiguous()
        sc_f = scale.float().reshape(BT, C).contiguous()
        s_out = torch.empty_like(a_c)
        y = torch.empty_like(a_c)
        mean = torch.empty((BT * S,), device=a_c.device, dtype=torch.float32)
        rstd = torch.empty((BT * S,), device=a_c.device, dtype=torch.float32)
        rc = _load_lib().launch_add_ln_modulate_fwd(
            a_c.data_ptr(), b_c.data_ptr(), sh_f.data_ptr(), sc_f.data_ptr(),
            s_out.data_ptr(), y.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            BT, S, C, float(eps), _stream(a_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"add_ln_modulate_fwd failed rc={rc}")
        if stats_out is not None:
            stats_out.append((s_out, mean, rstd, sh_f, sc_f, BT, S))
        ctx.save_for_backward(s_out, sc_f, mean, rstd)
        ctx.meta = (B, T, S, C, shift.shape, shift.dtype, scale.dtype)
        return s_out, y

    @staticmethod
    def backward(ctx, ds_res: torch.Tensor, dy: torch.Tensor):
        s_out, sc_f, mean, rstd = ctx.saved_tensors
        B, T, S, C, sh_shape, sh_dt, sc_dt = ctx.meta
        BT = B * T
        dy_c = dy.contiguous()
        ds_c = ds_res.contiguous() if ds_res is not None else None
        dx = torch.empty_like(s_out)
        n_splits = _choose_frame_splits(BT, S)
        dsh_part = torch.empty((BT, n_splits, C), device=s_out.device, dtype=torch.float32)
        dsc_part = torch.empty((BT, n_splits, C), device=s_out.device, dtype=torch.float32)
        dsh_out = torch.empty((BT, C), device=s_out.device, dtype=torch.float32)
        dsc_out = torch.empty((BT, C), device=s_out.device, dtype=torch.float32)
        rc = _load_lib().launch_add_ln_modulate_bwd(
            s_out.data_ptr(), dy_c.data_ptr(), ds_c.data_ptr() if ds_c is not None else 0,
            sc_f.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            dx.data_ptr(), dsh_part.data_ptr(), dsc_part.data_ptr(),
            dsh_out.data_ptr(), dsc_out.data_ptr(),
            BT, S, C, n_splits, _stream(s_out.device),
        )
        if rc != 0:
            raise RuntimeError(f"add_ln_modulate_bwd failed rc={rc}")
        return dx, dx, dsh_out.view(sh_shape).to(sh_dt), dsc_out.view(sh_shape).to(sc_dt), None, None, None


def fused_add_layernorm_modulate(
    a: torch.Tensor, b: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
    T: int, eps: float = 1e-6, stats_out: list | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    C = a.shape[-1]
    S = a.shape[1] // T
    if (
        S >= 8
        and (C & 7) == 0
        and C <= 2304
        and (a.data_ptr() & 15) == 0
        and (b.data_ptr() & 15) == 0
    ):
        return _FusedAddLNModulateFn.apply(a, b, shift, scale, T, eps, stats_out)
    s = a + b
    return s, fused_layernorm_modulate(s, shift, scale, T, eps, stats_out=stats_out)


# ---------------------------------------------------------------------------
# 3. Fused Gate + Residual: out = x + gate * y (+ optional add_bias)
# ---------------------------------------------------------------------------
class _FusedGateResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, T: int, add_bias: torch.Tensor | None):
        x_c = x.contiguous()
        y_c = y.contiguous()
        B, N, C = x_c.shape
        S = N // T
        BT = B * T
        g_f = gate.float().reshape(BT, C).contiguous()
        has_bias = add_bias is not None
        b_c = add_bias.contiguous() if has_bias else None
        out = torch.empty_like(x_c)
        rc = _load_lib().launch_gate_residual_fwd(
            x_c.data_ptr(), y_c.data_ptr(), g_f.data_ptr(),
            b_c.data_ptr() if has_bias else 0,
            out.data_ptr(),
            BT, S, C, _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"gate_residual_fwd failed rc={rc}")
        ctx.save_for_backward(y_c, g_f)
        ctx.meta = (BT, S, C, gate.shape, gate.dtype, has_bias, add_bias.dtype if has_bias else None)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        y_c, g_f = ctx.saved_tensors
        BT, S, C, g_shape, g_dt, has_bias, b_dt = ctx.meta
        dout_c = dout.contiguous()
        dy = torch.empty_like(y_c)
        n_splits = _choose_frame_splits(BT, S)
        dg_part = torch.empty((BT, n_splits, C), device=y_c.device, dtype=torch.float32)
        dg_out = torch.empty((BT, C), device=y_c.device, dtype=torch.float32)
        if has_bias and ctx.needs_input_grad[4]:
            db_part = torch.empty((BT, n_splits, C), device=y_c.device, dtype=torch.float32)
            db_out = torch.empty((C,), device=y_c.device, dtype=torch.float32)
        else:
            db_part = None
            db_out = None
        rc = _load_lib().launch_gate_residual_bwd(
            dout_c.data_ptr(), y_c.data_ptr(), g_f.data_ptr(),
            dy.data_ptr(), dg_part.data_ptr(), dg_out.data_ptr(),
            db_part.data_ptr() if db_part is not None else 0,
            db_out.data_ptr() if db_out is not None else 0,
            BT, S, C, n_splits, _stream(y_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"gate_residual_bwd failed rc={rc}")
        dbias = db_out.to(b_dt) if db_out is not None else None
        return dout_c, dy, dg_out.view(g_shape).to(g_dt), None, dbias


def fused_gate_residual(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    T: int,
    add_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if add_bias is not None:
        C = x.shape[-1]
        S = x.shape[1] // T
        if (
            S >= 8
            and (C & 7) == 0
            and C <= 2304
            and add_bias.device == x.device
            and add_bias.dtype == torch.bfloat16
            and add_bias.ndim == 1
            and add_bias.shape[0] == C
            and add_bias.is_contiguous()
            and ((x.data_ptr() | y.data_ptr() | add_bias.data_ptr()) & 15) == 0
        ):
            return _FusedGateResidualFn.apply(x, y, gate, T, add_bias)
        return _FusedGateResidualFn.apply(x, y, gate, T, None) + add_bias
    return _FusedGateResidualFn.apply(x, y, gate, T, None)


class GradAccBox:
    """Opaque single-slot holder not copied by FSDP2/pytree list/dict traversal."""

    __slots__ = ("grad",)

    def __init__(self):
        self.grad = None


class _FusedAddMMBF16Fn(torch.autograd.Function):
    """Fused residual + linear (without bias): out = x_res + inp @ weight.T via GEMM epilogue.

    When ``acc_holder`` (a ``GradAccBox``) is passed and ``inp`` is detached,
    d_inp is accumulated directly into ``acc_holder.grad`` via ``addmm_`` epilogue across
    layers, eliminating 19 full-tensor AddOp launches on shared ``plucker_emb``.
    """

    @staticmethod
    def forward(ctx, x_res: torch.Tensor, inp: torch.Tensor, weight: torch.Tensor, acc_holder=None):
        x_2d = x_res.contiguous().view(-1, weight.shape[0])
        inp_2d = inp.contiguous().view(-1, weight.shape[1])
        out_2d = torch.addmm(x_2d, inp_2d, weight.t())
        ctx.save_for_backward(inp_2d, weight)
        ctx.inp_shape = inp.shape
        ctx.acc_holder = acc_holder
        return out_2d.view_as(x_res)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        inp_2d, weight = ctx.saved_tensors
        dout_c = dout.contiguous()
        dout_2d = dout_c.view(-1, weight.shape[0])
        dx_res = dout_c if ctx.needs_input_grad[0] else None
        acc_holder = ctx.acc_holder
        if acc_holder is not None:
            if acc_holder.grad is None:
                acc_holder.grad = torch.mm(dout_2d, weight)
            else:
                acc_holder.grad.addmm_(dout_2d, weight)
            dinp = None
        else:
            dinp = torch.mm(dout_2d, weight).view(ctx.inp_shape) if ctx.needs_input_grad[1] else None
        dw = torch.mm(dout_2d.t(), inp_2d) if ctx.needs_input_grad[2] else None
        return dx_res, dinp, dw, None


class _FlushSharedGradFn(torch.autograd.Function):
    """Receives accumulated d_shared from acc_holder after all blocks finish backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, shared_inp: torch.Tensor, acc_holder: GradAccBox):
        ctx.acc_holder = acc_holder
        ctx.shared_shape = shared_inp.shape
        ctx.shared_dtype = shared_inp.dtype
        return x.view_as(x)

    @staticmethod
    def backward(ctx, dx: torch.Tensor):
        g = ctx.acc_holder.grad
        ctx.acc_holder.grad = None
        if g is not None and ctx.needs_input_grad[1]:
            g = g.view(ctx.shared_shape)
            if g.dtype != ctx.shared_dtype:
                g = g.to(ctx.shared_dtype)
        else:
            g = None
        return dx, g, None


def flush_shared_grad(x: torch.Tensor, shared_inp: torch.Tensor, acc_holder: GradAccBox) -> torch.Tensor:
    return _FlushSharedGradFn.apply(x, shared_inp, acc_holder)


def fused_addmm_bf16(
    x_res: torch.Tensor, inp: torch.Tensor, weight: torch.Tensor, acc_holder=None
) -> torch.Tensor:
    if (
        os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and x_res.device.type == "musa"
        and x_res.dtype == torch.bfloat16
        and inp.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
    ):
        return _FusedAddMMBF16Fn.apply(x_res, inp, weight, acc_holder)
    return x_res + torch.nn.functional.linear(inp, weight, None)


# ---------------------------------------------------------------------------
# 4. Fused SiLU * Out (Output Gate)
# ---------------------------------------------------------------------------
class _FusedSiluMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out: torch.Tensor, g: torch.Tensor):
        out_c = out.contiguous()
        g_c = g.contiguous()
        y = torch.empty_like(out_c)
        n2 = out_c.numel() // 2
        rc = _load_lib().launch_silu_mul_fwd(
            out_c.data_ptr(), g_c.data_ptr(), y.data_ptr(), n2, _stream(out_c.device)
        )
        if rc != 0:
            raise RuntimeError(f"silu_mul_fwd failed rc={rc}")
        ctx.save_for_backward(out_c, g_c)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        out_c, g_c = ctx.saved_tensors
        dy_c = dy.contiguous()
        dout = torch.empty_like(out_c)
        dg = torch.empty_like(g_c)
        n2 = out_c.numel() // 2
        rc = _load_lib().launch_silu_mul_bwd(
            dy_c.data_ptr(), out_c.data_ptr(), g_c.data_ptr(), dout.data_ptr(), dg.data_ptr(),
            n2, _stream(out_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"silu_mul_bwd failed rc={rc}")
        return dout, dg


def fused_silu_mul(out: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return _FusedSiluMulFn.apply(out, g)


# ---------------------------------------------------------------------------
# 5. Fused Softmax Main Prep: qkv(B,N,3,H,D) -> q,k,v (B,H,N,D) BF16
# ---------------------------------------------------------------------------
class _FusedSoftmaxMainPrepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv: torch.Tensor, q_nw, k_nw, rope_cos: torch.Tensor, rope_sin: torch.Tensor, norm_eps: float):
        qkv_c = qkv.contiguous()
        B, N, _, H, D = qkv_c.shape
        has_norm = q_nw is not None
        q_nw_f = q_nw.float().contiguous() if has_norm else None
        k_nw_f = k_nw.float().contiguous() if has_norm else None
        rc_c = rope_cos.float().contiguous()
        rs_c = rope_sin.float().contiguous()
        dev = qkv_c.device

        q_out = torch.empty((B, H, N, D), device=dev, dtype=torch.bfloat16)
        k_out = torch.empty((B, H, N, D), device=dev, dtype=torch.bfloat16)
        v_out = torch.empty((B, H, N, D), device=dev, dtype=torch.bfloat16)
        q_ir = torch.empty((B * N,), device=dev, dtype=torch.float32) if has_norm else torch.empty(0, device=dev)
        k_ir = torch.empty((B * N,), device=dev, dtype=torch.float32) if has_norm else torch.empty(0, device=dev)

        rc = _load_lib().launch_softmax_main_prep_fwd(
            qkv_c.data_ptr(), q_nw_f.data_ptr() if has_norm else 0, k_nw_f.data_ptr() if has_norm else 0,
            rc_c.data_ptr(), rs_c.data_ptr(),
            q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(),
            q_ir.data_ptr() if has_norm else 0, k_ir.data_ptr() if has_norm else 0,
            B, N, H, D, float(norm_eps), 1 if has_norm else 0, _stream(dev),
        )
        if rc != 0:
            raise RuntimeError(f"softmax_main_prep_fwd failed rc={rc}")

        empty = torch.empty(0, device=dev, dtype=torch.float32)
        ctx.save_for_backward(qkv_c, q_nw_f if has_norm else empty, k_nw_f if has_norm else empty, q_ir, k_ir, rc_c, rs_c)
        ctx.meta = (B, N, H, D, has_norm, q_nw.dtype if has_norm else None, k_nw.dtype if has_norm else None)
        return q_out, k_out, v_out

    @staticmethod
    def backward(ctx, dq, dk, dv):
        qkv_c, q_nw_f, k_nw_f, q_ir, k_ir, rc_c, rs_c = ctx.saved_tensors
        B, N, H, D, has_norm, q_dt, k_dt = ctx.meta
        C = H * D
        dev = qkv_c.device
        dq_c, dk_c, dv_c = dq.contiguous(), dk.contiguous(), dv.contiguous()
        dqkv = torch.empty_like(qkv_c)
        n_chunks = 240
        dq_p = torch.empty((n_chunks, C), device=dev, dtype=torch.float32) if has_norm else None
        dk_p = torch.empty((n_chunks, C), device=dev, dtype=torch.float32) if has_norm else None
        dq_o = torch.empty((C,), device=dev, dtype=torch.float32) if has_norm else None
        dk_o = torch.empty((C,), device=dev, dtype=torch.float32) if has_norm else None

        rc = _load_lib().launch_softmax_main_prep_bwd(
            qkv_c.data_ptr(), q_nw_f.data_ptr() if has_norm else 0, k_nw_f.data_ptr() if has_norm else 0,
            q_ir.data_ptr() if has_norm else 0, k_ir.data_ptr() if has_norm else 0,
            rc_c.data_ptr(), rs_c.data_ptr(),
            dq_c.data_ptr(), dk_c.data_ptr(), dv_c.data_ptr(),
            dqkv.data_ptr(),
            dq_p.data_ptr() if has_norm else 0, dk_p.data_ptr() if has_norm else 0,
            dq_o.data_ptr() if has_norm else 0, dk_o.data_ptr() if has_norm else 0,
            B, N, H, D, 1 if has_norm else 0, n_chunks, _stream(dev),
        )
        if rc != 0:
            raise RuntimeError(f"softmax_main_prep_bwd failed rc={rc}")
        return dqkv, (dq_o.to(q_dt) if has_norm else None), (dk_o.to(k_dt) if has_norm else None), None, None, None


def fused_softmax_main_prep(qkv, q_nw, k_nw, rope_cos, rope_sin, norm_eps: float):
    return _FusedSoftmaxMainPrepFn.apply(qkv, q_nw, k_nw, rope_cos, rope_sin, norm_eps)


# ---------------------------------------------------------------------------
# 6. Fused Softmax Cam Prep & Apply O (reusing cam_out_apply_o from bidi_gdn)
# ---------------------------------------------------------------------------
class _FusedSoftmaxCamPrepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv_cam: torch.Tensor, q_nw, k_nw, P_T, P_inv, rope_cos, rope_sin, H: int, D: int, norm_eps: float):
        # qkv_cam: (B, N, 3*C) BF16
        qkv_c = qkv_cam.contiguous()
        B, N, _ = qkv_c.shape
        q_nw_f = q_nw.float().contiguous()
        k_nw_f = k_nw.float().contiguous()
        pt_f = P_T.float().contiguous()
        pinv_f = P_inv.float().contiguous()
        rc_f = rope_cos.float().contiguous()
        rs_f = rope_sin.float().contiguous()
        dev = qkv_c.device

        q_out = torch.empty((B, H, N, D), device=dev, dtype=torch.bfloat16)
        k_out = torch.empty((B, H, N, D), device=dev, dtype=torch.bfloat16)
        v_out = torch.empty((B, H, N, D), device=dev, dtype=torch.bfloat16)
        q_ir = torch.empty((B * N,), device=dev, dtype=torch.float32)
        k_ir = torch.empty((B * N,), device=dev, dtype=torch.float32)

        rc = _load_lib().launch_softmax_cam_prep_fwd(
            qkv_c.data_ptr(), q_nw_f.data_ptr(), k_nw_f.data_ptr(),
            pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
            q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(),
            q_ir.data_ptr(), k_ir.data_ptr(),
            B, N, H, D, float(norm_eps), _stream(dev),
        )
        if rc != 0:
            raise RuntimeError(f"softmax_cam_prep_fwd failed rc={rc}")

        ctx.save_for_backward(qkv_c, q_nw_f, k_nw_f, pt_f, pinv_f, rc_f, rs_f, q_ir, k_ir)
        ctx.meta = (B, N, H, D, q_nw.dtype, k_nw.dtype)
        return q_out, k_out, v_out

    @staticmethod
    def backward(ctx, dq, dk, dv):
        qkv_c, q_nw_f, k_nw_f, pt_f, pinv_f, rc_f, rs_f, q_ir, k_ir = ctx.saved_tensors
        B, N, H, D, q_dt, k_dt = ctx.meta
        C = H * D
        dev = qkv_c.device
        dq_c, dk_c, dv_c = dq.contiguous(), dk.contiguous(), dv.contiguous()
        dqkv = torch.empty_like(qkv_c)
        n_chunks = 240
        dq_p = torch.empty((n_chunks, C), device=dev, dtype=torch.float32)
        dk_p = torch.empty((n_chunks, C), device=dev, dtype=torch.float32)
        dq_o = torch.empty((C,), device=dev, dtype=torch.float32)
        dk_o = torch.empty((C,), device=dev, dtype=torch.float32)

        rc = _load_lib().launch_softmax_cam_prep_bwd(
            qkv_c.data_ptr(), q_nw_f.data_ptr(), k_nw_f.data_ptr(),
            pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
            q_ir.data_ptr(), k_ir.data_ptr(),
            dq_c.data_ptr(), dk_c.data_ptr(), dv_c.data_ptr(),
            dqkv.data_ptr(), dq_p.data_ptr(), dk_p.data_ptr(), dq_o.data_ptr(), dk_o.data_ptr(),
            B, N, H, D, n_chunks, _stream(dev),
        )
        if rc != 0:
            raise RuntimeError(f"softmax_cam_prep_bwd failed rc={rc}")
        return dqkv, dq_o.to(q_dt), dk_o.to(k_dt), None, None, None, None, None, None, None


class _FusedCamApplyOFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_out: torch.Tensor, P: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        # raw_out: (B, H, N, D) BF16 -> out: (B, N, H*D) BF16
        from diffusion.model.musa_ops.bidi_gdn import _load_lib as _load_gdn_lib

        raw_c = raw_out.contiguous()
        B, H, N, D = raw_c.shape
        P_f = P.float().contiguous()
        rc_f = rope_cos.float().contiguous()
        rs_f = rope_sin.float().contiguous()
        out = torch.empty((B, N, H * D), device=raw_c.device, dtype=torch.bfloat16)
        rc = _load_gdn_lib().launch_cam_out_apply_o_fwd(
            raw_c.data_ptr(), P_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(), out.data_ptr(),
            B, N, H, D, 1, _stream(raw_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"cam_out_apply_o_fwd failed rc={rc}")
        ctx.save_for_backward(P_f, rc_f, rs_f)
        ctx.meta = (B, H, N, D)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        from diffusion.model.musa_ops.bidi_gdn import _load_lib as _load_gdn_lib

        P_f, rc_f, rs_f = ctx.saved_tensors
        B, H, N, D = ctx.meta
        dout_c = dout.contiguous()
        d_raw = torch.empty((B, H, N, D), device=dout_c.device, dtype=torch.bfloat16)
        rc = _load_gdn_lib().launch_cam_out_apply_o_bwd(
            dout_c.data_ptr(), P_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(), d_raw.data_ptr(),
            B, N, H, D, 1, _stream(dout_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"cam_out_apply_o_bwd failed rc={rc}")
        return d_raw, None, None, None


def fused_softmax_cam_prep(qkv_cam, q_nw, k_nw, P_T, P_inv, rope_cos, rope_sin, H: int, D: int, norm_eps: float):
    return _FusedSoftmaxCamPrepFn.apply(qkv_cam, q_nw, k_nw, P_T, P_inv, rope_cos, rope_sin, H, D, norm_eps)


def fused_cam_apply_o(raw_out: torch.Tensor, P: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
    return _FusedCamApplyOFn.apply(raw_out, P, rope_cos, rope_sin)


def can_use_block_fusion(x: torch.Tensor) -> bool:
    if os.environ.get("SANA_OPT_BLOCK_FUSION", "1") == "0":
        return False
    return x.device.type == "musa" and x.dtype == torch.bfloat16 and x.shape[-1] % 2 == 0 and x.shape[-1] <= 4096


# ---------------------------------------------------------------------------
# 7. Fast BF16 Column Sum & Fast BF16 Linear (eliminates slow ReduceABCKernel)
# ---------------------------------------------------------------------------
def fast_col_sum_bf16(dy_2d: torch.Tensor) -> torch.Tensor:
    M, C = dy_2d.shape
    if (
        os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and dy_2d.device.type == "musa"
        and dy_2d.dtype == torch.bfloat16
        and dy_2d.is_contiguous()
        and (C % 8) == 0
        and M >= 2048
    ):
        C8 = C >> 3
        gx = (C8 + 31) // 32
        n_splits = min(64, max(16, (288 + gx - 1) // gx))
        n_splits = min(n_splits, M)
        dev = dy_2d.device
        partial = torch.empty((n_splits, C), device=dev, dtype=torch.float32)
        out = torch.empty((C,), device=dev, dtype=torch.bfloat16)
        rc = _load_lib().launch_bf16_col_sum(
            dy_2d.data_ptr(), partial.data_ptr(), out.data_ptr(),
            M, C, n_splits, _stream(dev),
        )
        if rc != 0:
            raise RuntimeError(f"bf16_col_sum failed rc={rc}")
        return out
    return dy_2d.sum(dim=0)


class _FastLinearBF16Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight = ctx.saved_tensors
        dy_c = dy.contiguous()
        x_c = x.contiguous()
        orig_shape = x.shape
        K = orig_shape[-1]
        N_out = weight.shape[0]
        M = x_c.numel() // K
        dy_2d = dy_c.view(M, N_out)
        x_2d = x_c.view(M, K)

        dx = torch.mm(dy_2d, weight).view(orig_shape) if ctx.needs_input_grad[0] else None
        dw = torch.mm(dy_2d.t(), x_2d) if ctx.needs_input_grad[1] else None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = fast_col_sum_bf16(dy_2d)
            if db.dtype != ctx.bias_dtype:
                db = db.to(ctx.bias_dtype)
        else:
            db = None
        return dx, dw, db


def fast_linear_bf16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    if (
        bias is not None
        and os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and bias.dtype == torch.bfloat16
        and weight.shape[0] >= 2048
        and (weight.shape[0] % 8) == 0
        and x.numel() // x.shape[-1] >= 2048
        and torch.is_grad_enabled()
    ):
        return _FastLinearBF16Fn.apply(x, weight, bias)
    return F.linear(x, weight, bias)


class _FastTripleLinearBF16Fn(torch.autograd.Function):
    """3-way shared-input BF16 linear fusing dx accumulation into GEMM epilogues (addmm_).

    Eliminates 2x large AddOp bf16 kernels (94 MiB each) per layer when x feeds
    qkv_main + qkv_cam + output_gate simultaneously. Outputs remain contiguous (zero copy).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor, b1: torch.Tensor | None,
        w2: torch.Tensor, b2: torch.Tensor | None,
        w3: torch.Tensor, b3: torch.Tensor | None,
    ):
        x_c = x.contiguous()
        y1 = F.linear(x_c, w1, b1)
        y2 = F.linear(x_c, w2, b2)
        y3 = F.linear(x_c, w3, b3)
        ctx.save_for_backward(x_c, w1, w2, w3)
        ctx.meta = (
            x.shape,
            b1 is not None, b1.dtype if b1 is not None else None,
            b2 is not None, b2.dtype if b2 is not None else None,
            b3 is not None, b3.dtype if b3 is not None else None,
        )
        return y1, y2, y3

    @staticmethod
    def backward(ctx, dy1: torch.Tensor, dy2: torch.Tensor, dy3: torch.Tensor):
        x_c, w1, w2, w3 = ctx.saved_tensors
        orig_shape, hb1, dt1, hb2, dt2, hb3, dt3 = ctx.meta
        K = orig_shape[-1]
        M = x_c.numel() // K
        x_2d = x_c.view(M, K)

        dy1_2d = dy1.contiguous().view(M, w1.shape[0]) if dy1 is not None else None
        dy2_2d = dy2.contiguous().view(M, w2.shape[0]) if dy2 is not None else None
        dy3_2d = dy3.contiguous().view(M, w3.shape[0]) if dy3 is not None else None

        dx = None
        if ctx.needs_input_grad[0]:
            if dy1_2d is not None:
                dx = torch.mm(dy1_2d, w1)
            if dy2_2d is not None:
                if dx is None:
                    dx = torch.mm(dy2_2d, w2)
                else:
                    dx.addmm_(dy2_2d, w2)
            if dy3_2d is not None:
                if dx is None:
                    dx = torch.mm(dy3_2d, w3)
                else:
                    dx.addmm_(dy3_2d, w3)
            if dx is not None:
                dx = dx.view(orig_shape)

        dw1 = torch.mm(dy1_2d.t(), x_2d) if (dy1_2d is not None and ctx.needs_input_grad[1]) else None
        db1 = fast_col_sum_bf16(dy1_2d).to(dt1) if (dy1_2d is not None and hb1 and ctx.needs_input_grad[2]) else None
        dw2 = torch.mm(dy2_2d.t(), x_2d) if (dy2_2d is not None and ctx.needs_input_grad[3]) else None
        db2 = fast_col_sum_bf16(dy2_2d).to(dt2) if (dy2_2d is not None and hb2 and ctx.needs_input_grad[4]) else None
        dw3 = torch.mm(dy3_2d.t(), x_2d) if (dy3_2d is not None and ctx.needs_input_grad[5]) else None
        db3 = fast_col_sum_bf16(dy3_2d).to(dt3) if (dy3_2d is not None and hb3 and ctx.needs_input_grad[6]) else None
        return dx, dw1, db1, dw2, db2, dw3, db3


class _FastTripleLinearQKVCamBF16Fn(torch.autograd.Function):
    """3-way (or 4-way with optional beta_proj) shared-input BF16 linear.
    Avoids saving cat([q_w, k_w, v_w]) (-574 MiB) and optionally avoids saving x
    when ln_stats=(x_pre, mean, rstd, sh_f, sc_f, BT, S) is provided (-1.88 GiB!)."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor, b1: torch.Tensor | None,
        qw2: torch.Tensor, kw2: torch.Tensor, vw2: torch.Tensor,
        qb2: torch.Tensor | None, kb2: torch.Tensor | None, vb2: torch.Tensor | None,
        w3: torch.Tensor, b3: torch.Tensor | None,
        w4: torch.Tensor | None = None, b4: torch.Tensor | None = None,
        ln_stats: tuple | None = None,
    ):
        x_c = x.contiguous()
        y1 = F.linear(x_c, w1, b1)
        w2 = torch.cat([qw2, kw2, vw2], dim=0)
        has_b2 = (qb2 is not None) and (kb2 is not None) and (vb2 is not None)
        b2 = torch.cat([qb2, kb2, vb2], dim=0) if has_b2 else None
        y2 = F.linear(x_c, w2, b2)
        del w2, b2
        y3 = F.linear(x_c, w3, b3)
        has_w4 = w4 is not None
        y4 = F.linear(x_c, w4, b4) if has_w4 else torch.empty(0, device=x_c.device, dtype=x_c.dtype)
        empty = torch.empty(0, device=x_c.device, dtype=x_c.dtype)
        if ln_stats is not None:
            x_pre, mean, rstd, sh_f, sc_f, ln_BT, ln_S = ln_stats
            ctx.save_for_backward(
                empty, w1, qw2, kw2, vw2, w3, w4 if has_w4 else empty,
                x_pre, mean, rstd, sh_f, sc_f,
            )
            has_ln = True
        else:
            ln_BT, ln_S = 0, 0
            ctx.save_for_backward(
                x_c, w1, qw2, kw2, vw2, w3, w4 if has_w4 else empty,
                empty, empty, empty, empty, empty,
            )
            has_ln = False
        ctx.meta = (
            x.shape,
            qw2.shape[0], kw2.shape[0], vw2.shape[0],
            b1 is not None, b1.dtype if b1 is not None else None,
            has_b2, qb2.dtype if has_b2 else None,
            b3 is not None, b3.dtype if b3 is not None else None,
            has_w4, b4 is not None, b4.dtype if b4 is not None else None,
            has_ln, ln_BT, ln_S,
        )
        return y1, y2, y3, y4

    @staticmethod
    def backward(ctx, dy1: torch.Tensor, dy2: torch.Tensor, dy3: torch.Tensor, dy4: torch.Tensor | None):
        (
            x_saved, w1, qw2, kw2, vw2, w3, w4_saved,
            x_pre, mean, rstd, sh_f, sc_f,
        ) = ctx.saved_tensors
        (
            orig_shape, cq, ck, cv, hb1, dt1, hb2, dt2, hb3, dt3,
            has_w4, hb4, dt4, has_ln, ln_BT, ln_S,
        ) = ctx.meta
        K = orig_shape[-1]
        M = 1
        for d in orig_shape[:-1]:
            M *= d

        dy1_2d = dy1.contiguous().view(M, w1.shape[0]) if dy1 is not None else None
        dy2_2d = dy2.contiguous().view(M, cq + ck + cv) if dy2 is not None else None
        dy3_2d = dy3.contiguous().view(M, w3.shape[0]) if dy3 is not None else None
        dy4_2d = dy4.contiguous().view(M, w4_saved.shape[0]) if (has_w4 and dy4 is not None) else None

        w2 = torch.cat([qw2, kw2, vw2], dim=0) if dy2_2d is not None else None
        dx = None
        if ctx.needs_input_grad[0]:
            if dy1_2d is not None:
                dx = torch.mm(dy1_2d, w1)
            if dy2_2d is not None:
                if dx is None:
                    dx = torch.mm(dy2_2d, w2)
                else:
                    dx.addmm_(dy2_2d, w2)
            if dy3_2d is not None:
                if dx is None:
                    dx = torch.mm(dy3_2d, w3)
                else:
                    dx.addmm_(dy3_2d, w3)
            if dy4_2d is not None:
                if dx is None:
                    dx = torch.mm(dy4_2d, w4_saved)
                else:
                    dx.addmm_(dy4_2d, w4_saved)
            if dx is not None:
                dx = dx.view(orig_shape)
        del w2

        need_any_w = (
            (dy1_2d is not None and ctx.needs_input_grad[1])
            or (dy2_2d is not None and (ctx.needs_input_grad[3] or ctx.needs_input_grad[4] or ctx.needs_input_grad[5]))
            or (dy3_2d is not None and ctx.needs_input_grad[9])
            or (dy4_2d is not None and ctx.needs_input_grad[11])
        )
        if need_any_w:
            if has_ln:
                x_rec = recompute_ln_mod_u4(x_pre, mean, rstd, sh_f, sc_f, ln_BT, ln_S)
                x_2d = x_rec.view(M, K)
            else:
                x_rec = None
                x_2d = x_saved.view(M, K)
        else:
            x_rec = None
            x_2d = None

        dw1 = torch.mm(dy1_2d.t(), x_2d) if (dy1_2d is not None and ctx.needs_input_grad[1]) else None
        db1 = fast_col_sum_bf16(dy1_2d).to(dt1) if (dy1_2d is not None and hb1 and ctx.needs_input_grad[2]) else None

        dqw2 = dkw2 = dvw2 = dqb2 = dkb2 = dvb2 = None
        if dy2_2d is not None:
            if ctx.needs_input_grad[3] or ctx.needs_input_grad[4] or ctx.needs_input_grad[5]:
                dw2 = torch.mm(dy2_2d.t(), x_2d)
                dqw2, dkw2, dvw2 = dw2.split([cq, ck, cv], dim=0)
                if not ctx.needs_input_grad[3]:
                    dqw2 = None
                if not ctx.needs_input_grad[4]:
                    dkw2 = None
                if not ctx.needs_input_grad[5]:
                    dvw2 = None
            if hb2 and (ctx.needs_input_grad[6] or ctx.needs_input_grad[7] or ctx.needs_input_grad[8]):
                db2 = fast_col_sum_bf16(dy2_2d).to(dt2)
                dqb2, dkb2, dvb2 = db2.split([cq, ck, cv], dim=0)
                if not ctx.needs_input_grad[6]:
                    dqb2 = None
                if not ctx.needs_input_grad[7]:
                    dkb2 = None
                if not ctx.needs_input_grad[8]:
                    dvb2 = None

        dw3 = torch.mm(dy3_2d.t(), x_2d) if (dy3_2d is not None and ctx.needs_input_grad[9]) else None
        db3 = fast_col_sum_bf16(dy3_2d).to(dt3) if (dy3_2d is not None and hb3 and ctx.needs_input_grad[10]) else None

        dw4 = torch.mm(dy4_2d.t(), x_2d) if (dy4_2d is not None and ctx.needs_input_grad[11]) else None
        db4 = dy4_2d.float().sum(dim=0).to(dt4) if (dy4_2d is not None and hb4 and ctx.needs_input_grad[12]) else None
        del x_rec, x_2d
        return dx, dw1, db1, dqw2, dkw2, dvw2, dqb2, dkb2, dvb2, dw3, db3, dw4, db4, None


def fast_triple_linear_qkv_cam_bf16(
    x: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor | None,
    qw2: torch.Tensor, kw2: torch.Tensor, vw2: torch.Tensor,
    qb2: torch.Tensor | None, kb2: torch.Tensor | None, vb2: torch.Tensor | None,
    w3: torch.Tensor, b3: torch.Tensor | None,
    ln_stats: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and w1.dtype == torch.bfloat16
        and qw2.dtype == torch.bfloat16
        and w3.dtype == torch.bfloat16
        and (w1.shape[0] % 8) == 0
        and ((qw2.shape[0] + kw2.shape[0] + vw2.shape[0]) % 8) == 0
        and (w3.shape[0] % 8) == 0
        and x.numel() // x.shape[-1] >= 2048
        and torch.is_grad_enabled()
    ):
        y1, y2, y3, _ = _FastTripleLinearQKVCamBF16Fn.apply(
            x, w1, b1, qw2, kw2, vw2, qb2, kb2, vb2, w3, b3, None, None, ln_stats
        )
        return y1, y2, y3
    w2 = torch.cat([qw2, kw2, vw2], dim=0)
    b2 = torch.cat([qb2, kb2, vb2], dim=0) if (qb2 is not None and kb2 is not None and vb2 is not None) else None
    return fast_triple_linear_bf16(x, w1, b1, w2, b2, w3, b3)


def fast_quad_linear_qkv_cam_beta_bf16(
    x: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor | None,
    qw2: torch.Tensor, kw2: torch.Tensor, vw2: torch.Tensor,
    qb2: torch.Tensor | None, kb2: torch.Tensor | None, vb2: torch.Tensor | None,
    w3: torch.Tensor, b3: torch.Tensor | None,
    w4: torch.Tensor, b4: torch.Tensor | None,
    ln_stats: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and w1.dtype == torch.bfloat16
        and qw2.dtype == torch.bfloat16
        and w3.dtype == torch.bfloat16
        and w4.dtype == torch.bfloat16
        and (w1.shape[0] % 8) == 0
        and ((qw2.shape[0] + kw2.shape[0] + vw2.shape[0]) % 8) == 0
        and (w3.shape[0] % 8) == 0
        and x.numel() // x.shape[-1] >= 2048
        and torch.is_grad_enabled()
    ):
        return _FastTripleLinearQKVCamBF16Fn.apply(
            x, w1, b1, qw2, kw2, vw2, qb2, kb2, vb2, w3, b3, w4, b4, ln_stats
        )
    y1, y2, y3 = fast_triple_linear_qkv_cam_bf16(x, w1, b1, qw2, kw2, vw2, qb2, kb2, vb2, w3, b3)
    y4 = F.linear(x, w4, b4)
    return y1, y2, y3, y4


def fast_triple_linear_bf16(
    x: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor | None,
    w2: torch.Tensor, b2: torch.Tensor | None,
    w3: torch.Tensor, b3: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and w3.dtype == torch.bfloat16
        and (w1.shape[0] % 8) == 0
        and (w2.shape[0] % 8) == 0
        and (w3.shape[0] % 8) == 0
        and x.numel() // x.shape[-1] >= 2048
        and torch.is_grad_enabled()
    ):
        return _FastTripleLinearBF16Fn.apply(x, w1, b1, w2, b2, w3, b3)
    return (
        fast_linear_bf16(x, w1, b1),
        fast_linear_bf16(x, w2, b2),
        fast_linear_bf16(x, w3, b3),
    )


# ---------------------------------------------------------------------------
# 8. Fused (a + b) * silu(g)
# ---------------------------------------------------------------------------
class _FusedAddSiluMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, g: torch.Tensor):
        a_c = a.contiguous()
        b_c = b.contiguous()
        g_c = g.contiguous()
        s = torch.empty_like(a_c)
        y = torch.empty_like(a_c)
        n2 = a_c.numel() // 2
        rc = _load_lib().launch_add_silu_mul_fwd(
            a_c.data_ptr(), b_c.data_ptr(), g_c.data_ptr(), s.data_ptr(), y.data_ptr(),
            n2, _stream(a_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"add_silu_mul_fwd failed rc={rc}")
        ctx.save_for_backward(s, g_c)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        s_c, g_c = ctx.saved_tensors
        dy_c = dy.contiguous()
        ds = torch.empty_like(s_c)
        dg = torch.empty_like(g_c)
        n2 = s_c.numel() // 2
        rc = _load_lib().launch_silu_mul_bwd(
            dy_c.data_ptr(), s_c.data_ptr(), g_c.data_ptr(), ds.data_ptr(), dg.data_ptr(),
            n2, _stream(s_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"silu_mul_bwd failed rc={rc}")
        return ds, ds, dg


def fused_add_silu_mul(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return _FusedAddSiluMulFn.apply(a, b, g)


class _FusedAddSiluMulProjFn(torch.autograd.Function):
    """Fused (a + b) * silu(g) -> Linear(w_proj, b_proj) without saving 94 MiB combined y (-1.88 GiB across 20 layers!)."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, g: torch.Tensor, w_proj: torch.Tensor, b_proj: torch.Tensor | None):
        a_c = a.contiguous()
        b_c = b.contiguous()
        g_c = g.contiguous()
        s = torch.empty_like(a_c)
        y = torch.empty_like(a_c)
        n2 = a_c.numel() // 2
        rc = _load_lib().launch_add_silu_mul_fwd(
            a_c.data_ptr(), b_c.data_ptr(), g_c.data_ptr(), s.data_ptr(), y.data_ptr(),
            n2, _stream(a_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"add_silu_mul_fwd failed rc={rc}")
        out = F.linear(y, w_proj, b_proj)
        del y
        ctx.save_for_backward(s, g_c, w_proj)
        ctx.meta = (a_c.shape, b_proj is not None, b_proj.dtype if b_proj is not None else None)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        s_c, g_c, w_proj = ctx.saved_tensors
        orig_shape, has_b, b_dt = ctx.meta
        C_in = orig_shape[-1]
        M = s_c.numel() // C_in
        dout_c = dout.contiguous()
        dout_2d = dout_c.view(M, w_proj.shape[0])

        dy = torch.mm(dout_2d, w_proj).view(orig_shape)
        if has_b and ctx.needs_input_grad[4]:
            db_proj = fast_col_sum_bf16(dout_2d)
            if db_proj.dtype != b_dt:
                db_proj = db_proj.to(b_dt)
        else:
            db_proj = None

        ds = torch.empty_like(s_c)
        dg = torch.empty_like(g_c)
        need_w = ctx.needs_input_grad[3]
        y_recomp = torch.empty_like(s_c) if need_w else None
        n2 = s_c.numel() // 2
        rc = _load_lib().launch_silu_mul_bwd_y(
            dy.data_ptr(), s_c.data_ptr(), g_c.data_ptr(), ds.data_ptr(), dg.data_ptr(),
            y_recomp.data_ptr() if need_w else 0,
            n2, _stream(s_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"silu_mul_bwd_y failed rc={rc}")
        del dy
        dw_proj = torch.mm(dout_2d.t(), y_recomp.view(M, C_in)) if need_w else None
        del y_recomp
        return ds, ds, dg, dw_proj, db_proj


def fused_add_silu_mul_proj(
    a: torch.Tensor, b: torch.Tensor, g: torch.Tensor, w_proj: torch.Tensor, b_proj: torch.Tensor | None
) -> torch.Tensor:
    if (
        os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and a.device.type == "musa"
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and w_proj.dtype == torch.bfloat16
        and (w_proj.shape[0] % 8) == 0
        and a.numel() // a.shape[-1] >= 2048
        and torch.is_grad_enabled()
    ):
        return _FusedAddSiluMulProjFn.apply(a, b, g, w_proj, b_proj)
    return fast_linear_bf16(fused_add_silu_mul(a, b, g).to(w_proj.dtype), w_proj, b_proj)


# ---------------------------------------------------------------------------
# 9. Fused RMSNorm + Transpose (B, N, H*D) <-> (B, H, N, D)
# ---------------------------------------------------------------------------
class _FusedRMSNormToBHNDFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, H: int, D: int, eps: float):
        x_c = x.contiguous()
        B, N, C = x_c.shape
        w_f = weight.float().contiguous()
        dev = x_c.device
        y_bhnd = torch.empty((B, H, N, D), device=dev, dtype=torch.bfloat16)
        inv_rms = torch.empty((B * N,), device=dev, dtype=torch.float32)
        rc = _load_lib().launch_rmsnorm_bnhd_to_bhnd_fwd(
            x_c.data_ptr(), w_f.data_ptr(), y_bhnd.data_ptr(), inv_rms.data_ptr(),
            B, N, H, D, float(eps), _stream(dev),
        )
        if rc != 0:
            raise RuntimeError(f"rmsnorm_bnhd_to_bhnd_fwd failed rc={rc}")
        ctx.save_for_backward(x_c, w_f, inv_rms)
        ctx.meta = (B, N, H, D, weight.dtype)
        return y_bhnd

    @staticmethod
    def backward(ctx, dy_bhnd: torch.Tensor):
        x_c, w_f, inv_rms = ctx.saved_tensors
        B, N, H, D, w_dtype = ctx.meta
        C = H * D
        dev = x_c.device
        st = _stream(dev)
        lib = _load_lib()
        dy_c = dy_bhnd.contiguous()
        dy_bnc = torch.empty((B, N, C), device=dev, dtype=torch.bfloat16)
        rc = lib.launch_transpose_bhnd_bnhd(dy_c.data_ptr(), dy_bnc.data_ptr(), B, H, N, D, 0, st)
        if rc != 0:
            raise RuntimeError(f"transpose_bhnd_bnhd (rmsnorm bwd) failed rc={rc}")
        dx = torch.empty_like(x_c)
        M = B * N
        n_chunks = min(240, max(1, M))
        dw_part = torch.empty((n_chunks, C), device=dev, dtype=torch.float32)
        dw_out = torch.empty((C,), device=dev, dtype=torch.float32)
        rc = lib.launch_rmsnorm_bwd(
            x_c.data_ptr(), dy_bnc.data_ptr(), w_f.data_ptr(), inv_rms.data_ptr(),
            dx.data_ptr(), dw_part.data_ptr(), dw_out.data_ptr(),
            M, C, n_chunks, st,
        )
        if rc != 0:
            raise RuntimeError(f"rmsnorm_bwd (from bhnd) failed rc={rc}")
        return dx, dw_out.to(w_dtype), None, None, None


def fused_rmsnorm_to_bhnd(x: torch.Tensor, weight: torch.Tensor, H: int, D: int, eps: float) -> torch.Tensor:
    return _FusedRMSNormToBHNDFn.apply(x, weight, H, D, eps)


# ---------------------------------------------------------------------------
# 10. Fast 128-bit Transpose (B, H, N, D) -> (B, N, H*D)
# ---------------------------------------------------------------------------
class _FusedBHNDToBNCFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bhnd: torch.Tensor):
        x_c = x_bhnd.contiguous()
        B, H, N, D = x_c.shape
        out = torch.empty((B, N, H * D), device=x_c.device, dtype=torch.bfloat16)
        rc = _load_lib().launch_transpose_bhnd_bnhd(
            x_c.data_ptr(), out.data_ptr(), B, H, N, D, 0, _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"transpose_bhnd_bnhd fwd failed rc={rc}")
        ctx.meta = (B, H, N, D)
        return out

    @staticmethod
    def backward(ctx, dy_bnc: torch.Tensor):
        B, H, N, D = ctx.meta
        dy_c = dy_bnc.contiguous()
        dx = torch.empty((B, H, N, D), device=dy_c.device, dtype=torch.bfloat16)
        rc = _load_lib().launch_transpose_bhnd_bnhd(
            dy_c.data_ptr(), dx.data_ptr(), B, H, N, D, 1, _stream(dy_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"transpose_bhnd_bnhd bwd failed rc={rc}")
        return dx


def fused_bhnd_to_bnc(x_bhnd: torch.Tensor) -> torch.Tensor:
    if x_bhnd.ndim == 4 and x_bhnd.transpose(1, 2).is_contiguous():
        B, H, N, D = x_bhnd.shape
        return x_bhnd.transpose(1, 2).reshape(B, N, H * D)
    if (
        os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and x_bhnd.device.type == "musa"
        and x_bhnd.dtype == torch.bfloat16
        and x_bhnd.ndim == 4
        and (x_bhnd.shape[-1] % 8) == 0
    ):
        return _FusedBHNDToBNCFn.apply(x_bhnd)
    B, H, N, D = x_bhnd.shape
    return x_bhnd.transpose(1, 2).reshape(B, N, H * D)


class _FusedBNCToBHNDFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bnc: torch.Tensor, H: int, D: int):
        x_c = x_bnc.contiguous()
        B, N, _ = x_c.shape
        out = torch.empty((B, H, N, D), device=x_c.device, dtype=torch.bfloat16)
        rc = _load_lib().launch_transpose_bhnd_bnhd(
            x_c.data_ptr(), out.data_ptr(), B, H, N, D, 1, _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"transpose_bnhd_bhnd fwd failed rc={rc}")
        ctx.meta = (B, H, N, D)
        return out

    @staticmethod
    def backward(ctx, dy_bhnd: torch.Tensor):
        B, H, N, D = ctx.meta
        dy_c = dy_bhnd.contiguous()
        dx = torch.empty((B, N, H * D), device=dy_c.device, dtype=torch.bfloat16)
        rc = _load_lib().launch_transpose_bhnd_bnhd(
            dy_c.data_ptr(), dx.data_ptr(), B, H, N, D, 0, _stream(dy_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"transpose_bnhd_bhnd bwd failed rc={rc}")
        return dx, None, None


def fused_bnc_to_bhnd(x_bnc: torch.Tensor, H: int, D: int) -> torch.Tensor:
    if (
        os.environ.get("SANA_OPT_BLOCK_FUSION", "1") != "0"
        and x_bnc.device.type == "musa"
        and x_bnc.dtype == torch.bfloat16
        and x_bnc.ndim == 3
        and (D % 8) == 0
    ):
        return _FusedBNCToBHNDFn.apply(x_bnc, H, D)
    B, N, _ = x_bnc.shape
    return x_bnc.view(B, N, H, D).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# 11. Fast 16-byte Pack / Unpack for FSDP2 Aligned Collectives
# ---------------------------------------------------------------------------
def fused_pack_pad16(src_2d: torch.Tensor, dst_2d: torch.Tensor) -> None:
    world, n = src_2d.shape
    pad_n = dst_2d.shape[1]
    elem_size = src_2d.element_size()
    assert elem_size in (2, 4), f"unsupported elem_size={elem_size}"
    scale = elem_size // 2
    rc = _load_lib().launch_pack_unpack_pad16_u16(
        src_2d.data_ptr(), dst_2d.data_ptr(),
        world, n * scale, pad_n * scale, 0, _stream(src_2d.device),
    )
    if rc != 0:
        raise RuntimeError(f"pack_pad16 failed rc={rc}")


def fused_unpack_pad16(src_2d: torch.Tensor, dst_2d: torch.Tensor) -> None:
    world, n = dst_2d.shape
    pad_n = src_2d.shape[1]
    elem_size = dst_2d.element_size()
    assert elem_size in (2, 4), f"unsupported elem_size={elem_size}"
    scale = elem_size // 2
    rc = _load_lib().launch_pack_unpack_pad16_u16(
        src_2d.data_ptr(), dst_2d.data_ptr(),
        world, n * scale, pad_n * scale, 1, _stream(dst_2d.device),
    )
    if rc != 0:
        raise RuntimeError(f"unpack_pad16 failed rc={rc}")
