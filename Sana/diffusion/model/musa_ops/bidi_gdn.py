"""Fused MUSA C++ (mcc, MTT S5000) Bidirectional GDN + UCPE + ShortConv4 (fwd + deterministic bwd).

Replaces the pure-PyTorch Python scan loops, flip/flip_and_shift copies, NCHW ShortConvolution
depthwise convs, FP64 complex128 RoPE, and repeated 5D transposes in ``sana_gdn_blocks_musa.py``:
  1. ``fused_bidi_short_conv4``: direct (B, T, S, C) symmetric 7-tap temporal conv (zero transpose/flip).
  2. ``fused_main_gdn_prep``: full-channel RMSNorm + ReLU + k_scale + RoPE -> direct (BH, T, S, D).
  3. ``fused_bidi_gdn_scan``: shared Phase A (1x instead of 2x) + fused MUSA C++ bidirectional
     112x112 state scan (fwd inclusive + rev exclusive in 1 launch, 160 CTAs) + combined Phase C
     (1x S_tot @ Q instead of 2x) + fused divide & transpose to (B, N, C).
  4. ``fused_cam_prep`` + ``fused_bidi_cam_scan`` + ``fused_cam_out_apply_o``:
     fused UCPE 4x4 raymat + FP32 RoPE + inflation_sq + shared-Phase-A/C cam scan + fused apply_fn_o.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

_LIB = None
_MCC_PATH = "/usr/local/musa-5.2/bin/mcc"
_MUSA_PATH = "/usr/local/musa-5.2"

_MUSA_SRC = r"""
#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cstdint>

static __device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

static __device__ __forceinline__ float half_warp_sum(float v) {
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// =========================================================================
// 1. Fused Bidirectional ShortConv (K=4, activation=None) on (B, T, S, C) BF16
//    y[t] = 2*bias + w0*(x[t-3]+x[t+3]) + w1*(x[t-2]+x[t+2]) + w2*(x[t-1]+x[t+1]) + w3*x[t]
// Grid: (C/64, ceil(B*S/8)), Block: 256 (tx=0..31 -> 2 ch bf162; ty=0..7 -> 8 spatial)
// =========================================================================
extern "C" __global__ void bidi_short_conv4_fwd_bf16(
    const __mt_bfloat162* __restrict__ x2,
    const __mt_bfloat16* __restrict__ w,
    const __mt_bfloat16* __restrict__ b,
    __mt_bfloat162* __restrict__ y2,
    int B, int T, int S, int C, int has_bias
) {
    const int c_blk2 = blockIdx.x * 32;
    const int c_blk = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int bs_idx = blockIdx.y * 8 + ty;
    if (bs_idx >= B * S) return;

    const int b_idx = bs_idx / S;
    const int s_idx = bs_idx - b_idx * S;
    const int C2 = C >> 1;
    const int c0 = c_blk + tx * 2;
    const int c1 = c0 + 1;

    const float w0_0 = __bfloat162float(w[c0 * 4 + 0]), w1_0 = __bfloat162float(w[c1 * 4 + 0]);
    const float w0_1 = __bfloat162float(w[c0 * 4 + 1]), w1_1 = __bfloat162float(w[c1 * 4 + 1]);
    const float w0_2 = __bfloat162float(w[c0 * 4 + 2]), w1_2 = __bfloat162float(w[c1 * 4 + 2]);
    const float w0_3 = __bfloat162float(w[c0 * 4 + 3]), w1_3 = __bfloat162float(w[c1 * 4 + 3]);
    const float b2_0 = has_bias ? 2.0f * __bfloat162float(b[c0]) : 0.0f;
    const float b2_1 = has_bias ? 2.0f * __bfloat162float(b[c1]) : 0.0f;

    const int64_t stride_t = static_cast<int64_t>(S) * C2;
    const int64_t base = (static_cast<int64_t>(b_idx) * T * S + s_idx) * C2 + c_blk2 + tx;

    // Sliding window of 7 frames in scalar registers: [t-3, t-2, t-1, t, t+1, t+2, t+3]
    float r0_m3=0, r0_m2=0, r0_m1=0, r0_0=0, r0_p1=0, r0_p2=0, r0_p3=0;
    float r1_m3=0, r1_m2=0, r1_m1=0, r1_0=0, r1_p1=0, r1_p2=0, r1_p3=0;

    if (T > 0) { __mt_bfloat162 v = x2[base + 0 * stride_t]; r0_0  = __low2float(v); r1_0  = __high2float(v); }
    if (T > 1) { __mt_bfloat162 v = x2[base + 1 * stride_t]; r0_p1 = __low2float(v); r1_p1 = __high2float(v); }
    if (T > 2) { __mt_bfloat162 v = x2[base + 2 * stride_t]; r0_p2 = __low2float(v); r1_p2 = __high2float(v); }

    for (int t = 0; t < T; ++t) {
        if (t + 3 < T) {
            __mt_bfloat162 v = x2[base + static_cast<int64_t>(t + 3) * stride_t];
            r0_p3 = __low2float(v); r1_p3 = __high2float(v);
        } else {
            r0_p3 = 0.0f; r1_p3 = 0.0f;
        }

        float y0 = b2_0 + w0_0 * (r0_m3 + r0_p3) + w0_1 * (r0_m2 + r0_p2) + w0_2 * (r0_m1 + r0_p1) + w0_3 * r0_0;
        float y1 = b2_1 + w1_0 * (r1_m3 + r1_p3) + w1_1 * (r1_m2 + r1_p2) + w1_2 * (r1_m1 + r1_p1) + w1_3 * r1_0;
        y2[base + static_cast<int64_t>(t) * stride_t] = __floats2bfloat162_rn(y0, y1);

        r0_m3 = r0_m2; r0_m2 = r0_m1; r0_m1 = r0_0; r0_0 = r0_p1; r0_p1 = r0_p2; r0_p2 = r0_p3;
        r1_m3 = r1_m2; r1_m2 = r1_m1; r1_m1 = r1_0; r1_0 = r1_p1; r1_p1 = r1_p2; r1_p2 = r1_p3;
    }
}

extern "C" __global__ void bidi_short_conv4_bwd_bf16(
    const __mt_bfloat162* __restrict__ x2,
    const __mt_bfloat162* __restrict__ dy2,
    const __mt_bfloat16* __restrict__ w,
    __mt_bfloat162* __restrict__ dx2,
    float* __restrict__ dw_part,
    float* __restrict__ db_part,
    int B, int T, int S, int C, int has_bias
) {
    const int c_blk2 = blockIdx.x * 32;
    const int c_blk = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int bs_idx = blockIdx.y * 8 + ty;
    const bool valid = (bs_idx < B * S);
    const int C2 = C >> 1;
    const int c0 = c_blk + tx * 2;
    const int c1 = c0 + 1;

    float dw0_0=0, dw0_1=0, dw0_2=0, dw0_3=0;
    float dw1_0=0, dw1_1=0, dw1_2=0, dw1_3=0;
    float db0=0, db1=0;

    if (valid) {
        const int b_idx = bs_idx / S;
        const int s_idx = bs_idx - b_idx * S;
        const float w0_0 = __bfloat162float(w[c0 * 4 + 0]), w1_0 = __bfloat162float(w[c1 * 4 + 0]);
        const float w0_1 = __bfloat162float(w[c0 * 4 + 1]), w1_1 = __bfloat162float(w[c1 * 4 + 1]);
        const float w0_2 = __bfloat162float(w[c0 * 4 + 2]), w1_2 = __bfloat162float(w[c1 * 4 + 2]);
        const float w0_3 = __bfloat162float(w[c0 * 4 + 3]), w1_3 = __bfloat162float(w[c1 * 4 + 3]);

        const int64_t stride_t = static_cast<int64_t>(S) * C2;
        const int64_t base = (static_cast<int64_t>(b_idx) * T * S + s_idx) * C2 + c_blk2 + tx;

        float x0_m3=0, x0_m2=0, x0_m1=0, x0_0=0, x0_p1=0, x0_p2=0, x0_p3=0;
        float x1_m3=0, x1_m2=0, x1_m1=0, x1_0=0, x1_p1=0, x1_p2=0, x1_p3=0;
        float d0_m3=0, d0_m2=0, d0_m1=0, d0_0=0, d0_p1=0, d0_p2=0, d0_p3=0;
        float d1_m3=0, d1_m2=0, d1_m1=0, d1_0=0, d1_p1=0, d1_p2=0, d1_p3=0;

        if (T > 0) {
            __mt_bfloat162 vx = x2[base], vd = dy2[base];
            x0_0 = __low2float(vx); x1_0 = __high2float(vx);
            d0_0 = __low2float(vd); d1_0 = __high2float(vd);
        }
        if (T > 1) {
            __mt_bfloat162 vx = x2[base + stride_t], vd = dy2[base + stride_t];
            x0_p1 = __low2float(vx); x1_p1 = __high2float(vx);
            d0_p1 = __low2float(vd); d1_p1 = __high2float(vd);
        }
        if (T > 2) {
            __mt_bfloat162 vx = x2[base + 2 * stride_t], vd = dy2[base + 2 * stride_t];
            x0_p2 = __low2float(vx); x1_p2 = __high2float(vx);
            d0_p2 = __low2float(vd); d1_p2 = __high2float(vd);
        }

        for (int t = 0; t < T; ++t) {
            if (t + 3 < T) {
                int64_t off3 = base + static_cast<int64_t>(t + 3) * stride_t;
                __mt_bfloat162 vx = x2[off3], vd = dy2[off3];
                x0_p3 = __low2float(vx); x1_p3 = __high2float(vx);
                d0_p3 = __low2float(vd); d1_p3 = __high2float(vd);
            } else {
                x0_p3 = 0; x1_p3 = 0; d0_p3 = 0; d1_p3 = 0;
            }

            float gx0 = w0_0 * (d0_m3 + d0_p3) + w0_1 * (d0_m2 + d0_p2) + w0_2 * (d0_m1 + d0_p1) + w0_3 * d0_0;
            float gx1 = w1_0 * (d1_m3 + d1_p3) + w1_1 * (d1_m2 + d1_p2) + w1_2 * (d1_m1 + d1_p1) + w1_3 * d1_0;
            dx2[base + static_cast<int64_t>(t) * stride_t] = __floats2bfloat162_rn(gx0, gx1);

            dw0_0 = fmaf(d0_0, x0_m3 + x0_p3, dw0_0); dw1_0 = fmaf(d1_0, x1_m3 + x1_p3, dw1_0);
            dw0_1 = fmaf(d0_0, x0_m2 + x0_p2, dw0_1); dw1_1 = fmaf(d1_0, x1_m2 + x1_p2, dw1_1);
            dw0_2 = fmaf(d0_0, x0_m1 + x0_p1, dw0_2); dw1_2 = fmaf(d1_0, x1_m1 + x1_p1, dw1_2);
            dw0_3 = fmaf(d0_0, x0_0,          dw0_3); dw1_3 = fmaf(d1_0, x1_0,          dw1_3);
            db0 += d0_0; db1 += d1_0;

            x0_m3 = x0_m2; x0_m2 = x0_m1; x0_m1 = x0_0; x0_0 = x0_p1; x0_p1 = x0_p2; x0_p2 = x0_p3;
            x1_m3 = x1_m2; x1_m2 = x1_m1; x1_m1 = x1_0; x1_0 = x1_p1; x1_p1 = x1_p2; x1_p2 = x1_p3;
            d0_m3 = d0_m2; d0_m2 = d0_m1; d0_m1 = d0_0; d0_0 = d0_p1; d0_p1 = d0_p2; d0_p2 = d0_p3;
            d1_m3 = d1_m2; d1_m2 = d1_m1; d1_m1 = d1_0; d1_0 = d1_p1; d1_p1 = d1_p2; d1_p2 = d1_p3;
        }
    }

    __shared__ float s_red[8][64];
    const int64_t dw_base = static_cast<int64_t>(blockIdx.y) * C * 4;
    #define RED4(V0, V1, K) \
        s_red[ty][tx * 2 + 0] = (V0); s_red[ty][tx * 2 + 1] = (V1); \
        __syncthreads(); \
        if (ty == 0) { \
            float s0 = 0, s1 = 0; \
            for (int r = 0; r < 8; ++r) { s0 += s_red[r][tx * 2 + 0]; s1 += s_red[r][tx * 2 + 1]; } \
            dw_part[dw_base + static_cast<int64_t>(c0) * 4 + (K)] = s0; \
            dw_part[dw_base + static_cast<int64_t>(c1) * 4 + (K)] = s1; \
        } \
        __syncthreads();
    RED4(dw0_0, dw1_0, 0)
    RED4(dw0_1, dw1_1, 1)
    RED4(dw0_2, dw1_2, 2)
    RED4(dw0_3, dw1_3, 3)
    #undef RED4

    if (has_bias) {
        s_red[ty][tx * 2 + 0] = 2.0f * db0;
        s_red[ty][tx * 2 + 1] = 2.0f * db1;
        __syncthreads();
        if (ty == 0) {
            float s0 = 0, s1 = 0;
            for (int r = 0; r < 8; ++r) { s0 += s_red[r][tx * 2 + 0]; s1 += s_red[r][tx * 2 + 1]; }
            int64_t db_base = static_cast<int64_t>(blockIdx.y) * C;
            db_part[db_base + c0] = s0;
            db_part[db_base + c1] = s1;
        }
    }
}

// QKV-packed K-slot Bidirectional ShortConv4 (reads qkv[:,:,1] stride=3*C -> contiguous y(B,N,C))
extern "C" __global__ void bidi_short_conv4_qkv_k_fwd_bf16(
    const __mt_bfloat162* __restrict__ qkv2,  // (B, T, S, 3, C/2)
    const __mt_bfloat16* __restrict__ w,
    const __mt_bfloat16* __restrict__ b,
    __mt_bfloat162* __restrict__ y2,          // (B, T, S, C/2)
    int B, int T, int S, int C, int has_bias
) {
    const int c_blk2 = blockIdx.x * 32;
    const int c_blk = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int bs_idx = blockIdx.y * 8 + ty;
    if (bs_idx >= B * S) return;

    const int b_idx = bs_idx / S;
    const int s_idx = bs_idx - b_idx * S;
    const int C2 = C >> 1;
    const int c0 = c_blk + tx * 2;
    const int c1 = c0 + 1;

    const float w0_0 = __bfloat162float(w[c0 * 4 + 0]), w1_0 = __bfloat162float(w[c1 * 4 + 0]);
    const float w0_1 = __bfloat162float(w[c0 * 4 + 1]), w1_1 = __bfloat162float(w[c1 * 4 + 1]);
    const float w0_2 = __bfloat162float(w[c0 * 4 + 2]), w1_2 = __bfloat162float(w[c1 * 4 + 2]);
    const float w0_3 = __bfloat162float(w[c0 * 4 + 3]), w1_3 = __bfloat162float(w[c1 * 4 + 3]);
    const float b2_0 = has_bias ? 2.0f * __bfloat162float(b[c0]) : 0.0f;
    const float b2_1 = has_bias ? 2.0f * __bfloat162float(b[c1]) : 0.0f;

    const int64_t in_stride_t = static_cast<int64_t>(S) * 3 * C2;
    const int64_t in_base = (static_cast<int64_t>(b_idx) * T * S + s_idx) * 3 * C2 + C2 + c_blk2 + tx;
    const int64_t out_stride_t = static_cast<int64_t>(S) * C2;
    const int64_t out_base = (static_cast<int64_t>(b_idx) * T * S + s_idx) * C2 + c_blk2 + tx;

    float r0_m3=0, r0_m2=0, r0_m1=0, r0_0=0, r0_p1=0, r0_p2=0, r0_p3=0;
    float r1_m3=0, r1_m2=0, r1_m1=0, r1_0=0, r1_p1=0, r1_p2=0, r1_p3=0;

    if (T > 0) { __mt_bfloat162 v = qkv2[in_base + 0 * in_stride_t]; r0_0  = __low2float(v); r1_0  = __high2float(v); }
    if (T > 1) { __mt_bfloat162 v = qkv2[in_base + 1 * in_stride_t]; r0_p1 = __low2float(v); r1_p1 = __high2float(v); }
    if (T > 2) { __mt_bfloat162 v = qkv2[in_base + 2 * in_stride_t]; r0_p2 = __low2float(v); r1_p2 = __high2float(v); }

    for (int t = 0; t < T; ++t) {
        if (t + 3 < T) {
            __mt_bfloat162 v = qkv2[in_base + static_cast<int64_t>(t + 3) * in_stride_t];
            r0_p3 = __low2float(v); r1_p3 = __high2float(v);
        } else {
            r0_p3 = 0.0f; r1_p3 = 0.0f;
        }
        float y0 = b2_0 + w0_0 * (r0_m3 + r0_p3) + w0_1 * (r0_m2 + r0_p2) + w0_2 * (r0_m1 + r0_p1) + w0_3 * r0_0;
        float y1 = b2_1 + w1_0 * (r1_m3 + r1_p3) + w1_1 * (r1_m2 + r1_p2) + w1_2 * (r1_m1 + r1_p1) + w1_3 * r1_0;
        y2[out_base + static_cast<int64_t>(t) * out_stride_t] = __floats2bfloat162_rn(y0, y1);

        r0_m3 = r0_m2; r0_m2 = r0_m1; r0_m1 = r0_0; r0_0 = r0_p1; r0_p1 = r0_p2; r0_p2 = r0_p3;
        r1_m3 = r1_m2; r1_m2 = r1_m1; r1_m1 = r1_0; r1_0 = r1_p1; r1_p1 = r1_p2; r1_p2 = r1_p3;
    }
}

extern "C" __global__ void bidi_short_conv4_qkv_k_bwd_bf16(
    const __mt_bfloat162* __restrict__ qkv2,
    const __mt_bfloat162* __restrict__ dy2,
    const __mt_bfloat16* __restrict__ w,
    __mt_bfloat162* __restrict__ dqkv2,  // writes directly into slot 1!
    float* __restrict__ dw_part,
    float* __restrict__ db_part,
    int B, int T, int S, int C, int has_bias
) {
    const int c_blk2 = blockIdx.x * 32;
    const int c_blk = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int bs_idx = blockIdx.y * 8 + ty;
    const bool valid = (bs_idx < B * S);
    const int C2 = C >> 1;
    const int c0 = c_blk + tx * 2;
    const int c1 = c0 + 1;

    float dw0_0=0, dw0_1=0, dw0_2=0, dw0_3=0;
    float dw1_0=0, dw1_1=0, dw1_2=0, dw1_3=0;
    float db0=0, db1=0;

    if (valid) {
        const int b_idx = bs_idx / S;
        const int s_idx = bs_idx - b_idx * S;
        const float w0_0 = __bfloat162float(w[c0 * 4 + 0]), w1_0 = __bfloat162float(w[c1 * 4 + 0]);
        const float w0_1 = __bfloat162float(w[c0 * 4 + 1]), w1_1 = __bfloat162float(w[c1 * 4 + 1]);
        const float w0_2 = __bfloat162float(w[c0 * 4 + 2]), w1_2 = __bfloat162float(w[c1 * 4 + 2]);
        const float w0_3 = __bfloat162float(w[c0 * 4 + 3]), w1_3 = __bfloat162float(w[c1 * 4 + 3]);

        const int64_t qkv_stride_t = static_cast<int64_t>(S) * 3 * C2;
        const int64_t qkv_base = (static_cast<int64_t>(b_idx) * T * S + s_idx) * 3 * C2 + C2 + c_blk2 + tx;
        const int64_t dy_stride_t = static_cast<int64_t>(S) * C2;
        const int64_t dy_base = (static_cast<int64_t>(b_idx) * T * S + s_idx) * C2 + c_blk2 + tx;

        float x0_m3=0, x0_m2=0, x0_m1=0, x0_0=0, x0_p1=0, x0_p2=0, x0_p3=0;
        float x1_m3=0, x1_m2=0, x1_m1=0, x1_0=0, x1_p1=0, x1_p2=0, x1_p3=0;
        float d0_m3=0, d0_m2=0, d0_m1=0, d0_0=0, d0_p1=0, d0_p2=0, d0_p3=0;
        float d1_m3=0, d1_m2=0, d1_m1=0, d1_0=0, d1_p1=0, d1_p2=0, d1_p3=0;

        if (T > 0) {
            __mt_bfloat162 vx = qkv2[qkv_base], vd = dy2[dy_base];
            x0_0 = __low2float(vx); x1_0 = __high2float(vx);
            d0_0 = __low2float(vd); d1_0 = __high2float(vd);
        }
        if (T > 1) {
            __mt_bfloat162 vx = qkv2[qkv_base + qkv_stride_t], vd = dy2[dy_base + dy_stride_t];
            x0_p1 = __low2float(vx); x1_p1 = __high2float(vx);
            d0_p1 = __low2float(vd); d1_p1 = __high2float(vd);
        }
        if (T > 2) {
            __mt_bfloat162 vx = qkv2[qkv_base + 2 * qkv_stride_t], vd = dy2[dy_base + 2 * dy_stride_t];
            x0_p2 = __low2float(vx); x1_p2 = __high2float(vx);
            d0_p2 = __low2float(vd); d1_p2 = __high2float(vd);
        }

        for (int t = 0; t < T; ++t) {
            if (t + 3 < T) {
                __mt_bfloat162 vx = qkv2[qkv_base + static_cast<int64_t>(t + 3) * qkv_stride_t];
                __mt_bfloat162 vd = dy2[dy_base + static_cast<int64_t>(t + 3) * dy_stride_t];
                x0_p3 = __low2float(vx); x1_p3 = __high2float(vx);
                d0_p3 = __low2float(vd); d1_p3 = __high2float(vd);
            } else {
                x0_p3 = 0; x1_p3 = 0; d0_p3 = 0; d1_p3 = 0;
            }
            float gx0 = w0_0 * (d0_m3 + d0_p3) + w0_1 * (d0_m2 + d0_p2) + w0_2 * (d0_m1 + d0_p1) + w0_3 * d0_0;
            float gx1 = w1_0 * (d1_m3 + d1_p3) + w1_1 * (d1_m2 + d1_p2) + w1_2 * (d1_m1 + d1_p1) + w1_3 * d1_0;
            dqkv2[qkv_base + static_cast<int64_t>(t) * qkv_stride_t] = __floats2bfloat162_rn(gx0, gx1);

            dw0_0 = fmaf(d0_0, x0_m3 + x0_p3, dw0_0); dw1_0 = fmaf(d1_0, x1_m3 + x1_p3, dw1_0);
            dw0_1 = fmaf(d0_0, x0_m2 + x0_p2, dw0_1); dw1_1 = fmaf(d1_0, x1_m2 + x1_p2, dw1_1);
            dw0_2 = fmaf(d0_0, x0_m1 + x0_p1, dw0_2); dw1_2 = fmaf(d1_0, x1_m1 + x1_p1, dw1_2);
            dw0_3 = fmaf(d0_0, x0_0,          dw0_3); dw1_3 = fmaf(d1_0, x1_0,          dw1_3);
            db0 += d0_0; db1 += d1_0;

            x0_m3 = x0_m2; x0_m2 = x0_m1; x0_m1 = x0_0; x0_0 = x0_p1; x0_p1 = x0_p2; x0_p2 = x0_p3;
            x1_m3 = x1_m2; x1_m2 = x1_m1; x1_m1 = x1_0; x1_0 = x1_p1; x1_p1 = x1_p2; x1_p2 = x1_p3;
            d0_m3 = d0_m2; d0_m2 = d0_m1; d0_m1 = d0_0; d0_0 = d0_p1; d0_p1 = d0_p2; d0_p2 = d0_p3;
            d1_m3 = d1_m2; d1_m2 = d1_m1; d1_m1 = d1_0; d1_0 = d1_p1; d1_p1 = d1_p2; d1_p2 = d1_p3;
        }
    }

    __shared__ float s_red[8][64];
    const int64_t dw_base = static_cast<int64_t>(blockIdx.y) * C * 4;
    #define RED4(V0, V1, K) \
        s_red[ty][tx * 2 + 0] = (V0); s_red[ty][tx * 2 + 1] = (V1); \
        __syncthreads(); \
        if (ty == 0) { \
            float s0 = 0, s1 = 0; \
            for (int r = 0; r < 8; ++r) { s0 += s_red[r][tx * 2 + 0]; s1 += s_red[r][tx * 2 + 1]; } \
            dw_part[dw_base + static_cast<int64_t>(c0) * 4 + (K)] = s0; \
            dw_part[dw_base + static_cast<int64_t>(c1) * 4 + (K)] = s1; \
        } \
        __syncthreads();
    RED4(dw0_0, dw1_0, 0)
    RED4(dw0_1, dw1_1, 1)
    RED4(dw0_2, dw1_2, 2)
    RED4(dw0_3, dw1_3, 3)
    #undef RED4

    if (has_bias) {
        s_red[ty][tx * 2 + 0] = 2.0f * db0;
        s_red[ty][tx * 2 + 1] = 2.0f * db1;
        __syncthreads();
        if (ty == 0) {
            float s0 = 0, s1 = 0;
            for (int r = 0; r < 8; ++r) { s0 += s_red[r][tx * 2 + 0]; s1 += s_red[r][tx * 2 + 1]; }
            int64_t db_base = static_cast<int64_t>(blockIdx.y) * C;
            db_part[db_base + c0] = s0;
            db_part[db_base + c1] = s1;
        }
    }
}

extern "C" __global__ void reduce_rows_to_bf16(
    const float* __restrict__ part,
    __mt_bfloat16* __restrict__ out,
    int R, int N_elem
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_elem) {
        float sum = 0.0f;
        for (int r = 0; r < R; ++r) sum += part[static_cast<int64_t>(r) * N_elem + idx];
        out[idx] = __float2bfloat16(sum);
    }
}

extern "C" __global__ void reduce_rows_to_f32(
    const float* __restrict__ part,
    float* __restrict__ out,
    int R, int N_elem
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_elem) {
        float sum = 0.0f;
        for (int r = 0; r < R; ++r) sum += part[static_cast<int64_t>(r) * N_elem + idx];
        out[idx] = sum;
    }
}

// =========================================================================
// 2. Fused Main GDN Prep: qkv(B,N,3,C) + k_conv(B,N,C) -> (BH,N,D)
//    Full-channel RMSNorm + ReLU + k_scale + RoPE -> direct (BH, N, D) FP32 or BF16
//    Grid: (B * N), Block: 256 (8 warps). Supports any even C = H * D.
// =========================================================================
template <typename OutT>
__device__ __forceinline__ void store_pair(OutT* ptr, float a, float b);

template <>
__device__ __forceinline__ void store_pair<float>(float* ptr, float a, float b) {
    *reinterpret_cast<float2*>(ptr) = make_float2(a, b);
}

template <>
__device__ __forceinline__ void store_pair<__mt_bfloat16>(__mt_bfloat16* ptr, float a, float b) {
    *reinterpret_cast<__mt_bfloat162*>(ptr) = __floats2bfloat162_rn(a, b);
}

template <typename InT>
__device__ __forceinline__ float2 load_pair(const InT* ptr);

template <>
__device__ __forceinline__ float2 load_pair<float>(const float* ptr) {
    return *reinterpret_cast<const float2*>(ptr);
}

template <>
__device__ __forceinline__ float2 load_pair<__mt_bfloat16>(const __mt_bfloat16* ptr) {
    __mt_bfloat162 v = *reinterpret_cast<const __mt_bfloat162*>(ptr);
    return make_float2(__low2float(v), __high2float(v));
}

template <typename OutT>
__device__ void main_gdn_prep_fwd_impl(
    const __mt_bfloat162* __restrict__ qkv2,   // (B, N, 3, C/2)
    const __mt_bfloat162* __restrict__ kconv2, // (B, N, C/2) or nullptr (use qkv[:,:,1])
    const float* __restrict__ q_nw,            // (C,) or nullptr
    const float* __restrict__ k_nw,            // (C,) or nullptr
    const float* __restrict__ rope_cos,        // (N, D)
    const float* __restrict__ rope_sin,        // (N, D)
    OutT* __restrict__ q_rot_out,              // (B*H, N, D)
    OutT* __restrict__ q_pos_out,              // (B*H, N, D)
    OutT* __restrict__ k_rot_out,              // (B*H, N, D)
    OutT* __restrict__ k_pos_out,              // (B*H, N, D)
    OutT* __restrict__ v_out,                  // (B*H, N, D)
    float* __restrict__ q_inv_rms_out,         // (B*N,)
    float* __restrict__ k_inv_rms_out,         // (B*N,)
    int B, int N, int H, int D, float k_scale, float norm_eps, int has_norm
) {
    const int bn = blockIdx.x;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int tid = threadIdx.x;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D2 = D >> 1;

    const __mt_bfloat162* q_ptr = qkv2 + static_cast<int64_t>(bn) * 3 * C2;
    const __mt_bfloat162* k_ptr = (kconv2 != nullptr) ? (kconv2 + static_cast<int64_t>(bn) * C2)
                                                      : (q_ptr + C2);
    const __mt_bfloat162* v_ptr = q_ptr + 2 * C2;

    extern __shared__ __mt_bfloat162 smem_mpf[];
    __mt_bfloat162* s_q = smem_mpf;         // C2
    __mt_bfloat162* s_k = s_q + C2;         // C2
    __shared__ float s_cos[128], s_sin[128];
    __shared__ float s_qred[8], s_kred[8];

    if (tid < D) {
        s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D + tid];
        s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D + tid];
    }

    float q_sq = 0.0f, k_sq = 0.0f;
    if (has_norm) {
        for (int p = tid; p < C2; p += 256) {
            __mt_bfloat162 qv = q_ptr[p];
            __mt_bfloat162 kv = k_ptr[p];
            s_q[p] = qv;
            s_k[p] = kv;
            float q0 = __low2float(qv), q1 = __high2float(qv);
            float k0 = __low2float(kv), k1 = __high2float(kv);
            q_sq = fmaf(q0, q0, fmaf(q1, q1, q_sq));
            k_sq = fmaf(k0, k0, fmaf(k1, k1, k_sq));
        }
        q_sq = warp_sum(q_sq);
        k_sq = warp_sum(k_sq);
        if ((tid & 31) == 0) { s_qred[tid >> 5] = q_sq; s_kred[tid >> 5] = k_sq; }
        __syncthreads();
        if (tid == 0) {
            float tq = 0.0f, tk = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) { tq += s_qred[w]; tk += s_kred[w]; }
            s_qred[0] = rsqrtf(tq / static_cast<float>(C) + norm_eps);
            s_kred[0] = rsqrtf(tk / static_cast<float>(C) + norm_eps);
            q_inv_rms_out[bn] = s_qred[0];
            k_inv_rms_out[bn] = s_kred[0];
        }
        __syncthreads();
        q_sq = s_qred[0];
        k_sq = s_kred[0];
    } else {
        q_sq = 1.0f;
        k_sq = 1.0f;
        if (tid == 0) { q_inv_rms_out[bn] = 1.0f; k_inv_rms_out[bn] = 1.0f; }
        __syncthreads();
    }

    const float q_ir = q_sq, k_ir = k_sq;

    for (int p = tid; p < C2; p += 256) {
        int h = p / D2;
        int d2 = p - h * D2;
        int d0 = d2 * 2;
        int c0 = p * 2;

        __mt_bfloat162 qv = has_norm ? s_q[p] : q_ptr[p];
        __mt_bfloat162 kv = has_norm ? s_k[p] : k_ptr[p];
        __mt_bfloat162 vv = v_ptr[p];

        float q0 = __low2float(qv), q1 = __high2float(qv);
        float k0 = __low2float(kv), k1 = __high2float(kv);
        float v0 = __low2float(vv), v1 = __high2float(vv);

        if (has_norm) {
            q0 = q0 * q_ir * q_nw[c0];     q1 = q1 * q_ir * q_nw[c0 + 1];
            k0 = k0 * k_ir * k_nw[c0];     k1 = k1 * k_ir * k_nw[c0 + 1];
        }
        q0 = q0 > 0.0f ? q0 : 0.0f;        q1 = q1 > 0.0f ? q1 : 0.0f;
        k0 = (k0 > 0.0f ? k0 : 0.0f) * k_scale;
        k1 = (k1 > 0.0f ? k1 : 0.0f) * k_scale;

        float c_e = s_cos[d0], c_o = s_cos[d0 + 1];
        float s_e = s_sin[d0], s_o = s_sin[d0 + 1];
        float qr0 = fmaf(q0, c_e, q1 * s_e);
        float qr1 = fmaf(q1, c_o, q0 * s_o);
        float kr0 = fmaf(k0, c_e, k1 * s_e);
        float kr1 = fmaf(k1, c_o, k0 * s_o);

        int64_t out_off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0;
        store_pair<OutT>(q_rot_out + out_off, qr0, qr1);
        store_pair<OutT>(q_pos_out + out_off, q0,  q1);
        store_pair<OutT>(k_rot_out + out_off, kr0, kr1);
        store_pair<OutT>(k_pos_out + out_off, k0,  k1);
        store_pair<OutT>(v_out     + out_off, v0,  v1);
    }
}

extern "C" __global__ void main_gdn_prep_fwd_f32(
    const __mt_bfloat162* __restrict__ qkv2, const __mt_bfloat162* __restrict__ kconv2,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ q_rot, float* __restrict__ q_pos,
    float* __restrict__ k_rot, float* __restrict__ k_pos, float* __restrict__ v_out,
    float* __restrict__ q_ir, float* __restrict__ k_ir,
    int B, int N, int H, int D, float k_scale, float norm_eps, int has_norm
) {
    main_gdn_prep_fwd_impl<float>(qkv2, kconv2, q_nw, k_nw, rope_cos, rope_sin,
                                  q_rot, q_pos, k_rot, k_pos, v_out, q_ir, k_ir,
                                  B, N, H, D, k_scale, norm_eps, has_norm);
}

extern "C" __global__ void main_gdn_prep_fwd_bf16(
    const __mt_bfloat162* __restrict__ qkv2, const __mt_bfloat162* __restrict__ kconv2,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    __mt_bfloat16* __restrict__ q_rot, __mt_bfloat16* __restrict__ q_pos,
    __mt_bfloat16* __restrict__ k_rot, __mt_bfloat16* __restrict__ k_pos, __mt_bfloat16* __restrict__ v_out,
    float* __restrict__ q_ir, float* __restrict__ k_ir,
    int B, int N, int H, int D, float k_scale, float norm_eps, int has_norm
) {
    main_gdn_prep_fwd_impl<__mt_bfloat16>(qkv2, kconv2, q_nw, k_nw, rope_cos, rope_sin,
                                          q_rot, q_pos, k_rot, k_pos, v_out, q_ir, k_ir,
                                          B, N, H, D, k_scale, norm_eps, has_norm);
}

// Backward of Main GDN Prep (deterministic split-chunk reduction for dq_nw / dk_nw!)
// Grid: (N_CHUNKS = 128), each block loops over tokens assigned to its chunk, accumulating
// dq_nw_part[chunk, C] and dk_nw_part[chunk, C] deterministically!
template <typename InT>
__device__ void main_gdn_prep_bwd_impl(
    const __mt_bfloat162* __restrict__ qkv2,
    const __mt_bfloat162* __restrict__ kconv2,
    const float* __restrict__ q_nw,
    const float* __restrict__ k_nw,
    const float* __restrict__ q_inv_rms,
    const float* __restrict__ k_inv_rms,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const InT* __restrict__ dq_rot,
    const InT* __restrict__ dq_pos,
    const InT* __restrict__ dk_rot,
    const InT* __restrict__ dk_pos,
    const InT* __restrict__ dv,
    __mt_bfloat162* __restrict__ dqkv2,
    __mt_bfloat162* __restrict__ dkconv2,
    float* __restrict__ dq_nw_part,  // (N_CHUNKS, C)
    float* __restrict__ dk_nw_part,  // (N_CHUNKS, C)
    int B, int N, int H, int D, float k_scale, int has_norm, int n_chunks
) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int BN = B * N;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D2 = D >> 1;
    const int bn_start = (static_cast<int64_t>(chunk) * BN) / n_chunks;
    const int bn_end   = (static_cast<int64_t>(chunk + 1) * BN) / n_chunks;

    extern __shared__ float smem_nw[];  // 4*C floats + 2*C2 bf162 + 16 floats
    float* s_dqnw = smem_nw;
    float* s_dknw = smem_nw + C;
    float* s_gq   = s_dknw + C;
    float* s_gk   = s_gq + C;
    __mt_bfloat162* s_q = reinterpret_cast<__mt_bfloat162*>(s_gk + C);
    __mt_bfloat162* s_k = s_q + C2;
    float* s_red  = reinterpret_cast<float*>(s_k + C2);  // 16 floats
    __shared__ float s_cos[128], s_sin[128];

    if (has_norm) {
        for (int i = tid; i < C; i += 256) { s_dqnw[i] = 0.0f; s_dknw[i] = 0.0f; }
    }
    __syncthreads();

    const __mt_bfloat162 zero_b2 = __floats2bfloat162_rn(0.0f, 0.0f);

    for (int bn = bn_start; bn < bn_end; ++bn) {
        const int b_idx = bn / N;
        const int n_idx = bn - b_idx * N;
        if (tid < D) {
            s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D + tid];
            s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D + tid];
        }
        __syncthreads();

        const __mt_bfloat162* q_ptr = qkv2 + static_cast<int64_t>(bn) * 3 * C2;
        const __mt_bfloat162* k_ptr = (kconv2 != nullptr) ? (kconv2 + static_cast<int64_t>(bn) * C2)
                                                          : (q_ptr + C2);
        __mt_bfloat162* dq_ptr = dqkv2 + static_cast<int64_t>(bn) * 3 * C2;
        __mt_bfloat162* dk_ptr = (dkconv2 != nullptr) ? (dkconv2 + static_cast<int64_t>(bn) * C2)
                                                      : (dq_ptr + C2);
        __mt_bfloat162* dv_ptr = dq_ptr + 2 * C2;

        const float q_ir = q_inv_rms[bn];
        const float k_ir = k_inv_rms[bn];

        float q_dot = 0.0f, k_dot = 0.0f;
        for (int p = tid; p < C2; p += 256) {
            int h = p / D2;
            int d0 = (p - h * D2) * 2;
            int c0 = p * 2;
            int64_t off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0;

            float2 dqr = load_pair<InT>(dq_rot + off);
            float2 dqp = load_pair<InT>(dq_pos + off);
            float2 dkr = load_pair<InT>(dk_rot + off);
            float2 dkp = load_pair<InT>(dk_pos + off);
            float2 dvp = load_pair<InT>(dv     + off);
            dv_ptr[p] = __floats2bfloat162_rn(dvp.x, dvp.y);
            if (dkconv2 != nullptr) dq_ptr[C2 + p] = zero_b2;

            float c_e = s_cos[d0], c_o = s_cos[d0 + 1];
            float s_e = s_sin[d0], s_o = s_sin[d0 + 1];
            float gq0 = fmaf(dqr.x, c_e, dqr.y * s_o) + dqp.x;
            float gq1 = fmaf(dqr.y, c_o, dqr.x * s_e) + dqp.y;
            float gk0 = (fmaf(dkr.x, c_e, dkr.y * s_o) + dkp.x) * k_scale;
            float gk1 = (fmaf(dkr.y, c_o, dkr.x * s_e) + dkp.y) * k_scale;

            __mt_bfloat162 qv = q_ptr[p], kv = k_ptr[p];
            float q0 = __low2float(qv), q1 = __high2float(qv);
            float k0 = __low2float(kv), k1 = __high2float(kv);

            if (has_norm) {
                float qw0 = q_nw[c0], qw1 = q_nw[c0 + 1];
                float kw0 = k_nw[c0], kw1 = k_nw[c0 + 1];
                if (q0 * qw0 <= 0.0f) gq0 = 0.0f;
                if (q1 * qw1 <= 0.0f) gq1 = 0.0f;
                if (k0 * kw0 <= 0.0f) gk0 = 0.0f;
                if (k1 * kw1 <= 0.0f) gk1 = 0.0f;

                s_q[p] = qv;
                s_k[p] = kv;
                s_gq[c0] = gq0 * qw0; s_gq[c0 + 1] = gq1 * qw1;
                s_gk[c0] = gk0 * kw0; s_gk[c0 + 1] = gk1 * kw1;

                s_dqnw[c0]     += gq0 * q0 * q_ir;
                s_dqnw[c0 + 1] += gq1 * q1 * q_ir;
                s_dknw[c0]     += gk0 * k0 * k_ir;
                s_dknw[c0 + 1] += gk1 * k1 * k_ir;

                q_dot = fmaf(s_gq[c0], q0, fmaf(s_gq[c0 + 1], q1, q_dot));
                k_dot = fmaf(s_gk[c0], k0, fmaf(s_gk[c0 + 1], k1, k_dot));
            } else {
                if (q0 <= 0.0f) gq0 = 0.0f;
                if (q1 <= 0.0f) gq1 = 0.0f;
                if (k0 <= 0.0f) gk0 = 0.0f;
                if (k1 <= 0.0f) gk1 = 0.0f;
                dq_ptr[p] = __floats2bfloat162_rn(gq0, gq1);
                dk_ptr[p] = __floats2bfloat162_rn(gk0, gk1);
            }
        }

        if (has_norm) {
            q_dot = warp_sum(q_dot);
            k_dot = warp_sum(k_dot);
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
            float q_coeff = s_red[0];
            float k_coeff = s_red[8];

            for (int p = tid; p < C2; p += 256) {
                int c0 = p * 2;
                __mt_bfloat162 qv = s_q[p], kv = s_k[p];
                float q0 = __low2float(qv), q1 = __high2float(qv);
                float k0 = __low2float(kv), k1 = __high2float(kv);
                dq_ptr[p] = __floats2bfloat162_rn(
                    q_ir * (s_gq[c0]     - q0 * q_coeff),
                    q_ir * (s_gq[c0 + 1] - q1 * q_coeff)
                );
                dk_ptr[p] = __floats2bfloat162_rn(
                    k_ir * (s_gk[c0]     - k0 * k_coeff),
                    k_ir * (s_gk[c0 + 1] - k1 * k_coeff)
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

extern "C" __global__ void main_gdn_prep_bwd_576_bf16(
    const __mt_bfloat16* __restrict__ qkv,
    const __mt_bfloat16* __restrict__ kconv,
    const float* __restrict__ q_nw,
    const float* __restrict__ k_nw,
    const float* __restrict__ q_inv_rms,
    const float* __restrict__ k_inv_rms,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const __mt_bfloat16* __restrict__ dq_rot,
    const __mt_bfloat16* __restrict__ dq_pos,
    const __mt_bfloat16* __restrict__ dk_rot,
    const __mt_bfloat16* __restrict__ dk_pos,
    const __mt_bfloat16* __restrict__ dv,
    __mt_bfloat16* __restrict__ dqkv,
    __mt_bfloat16* __restrict__ dkconv,
    float* __restrict__ dq_nw_part,
    float* __restrict__ dk_nw_part,
    int B, int N, int H, int D, float k_scale, int n_chunks
) {
    const int ch = blockIdx.x, tid = threadIdx.x;
    const int C = H * D, C4 = C >> 2, D4 = D >> 2;
    const bool act = (tid < C4);
    const int c0 = tid << 2;
    const int h = act ? (tid / D4) : 0;
    const int d0 = act ? ((tid - h * D4) << 2) : 0;
    const int BN = B * N;
    const int bn0 = (static_cast<int64_t>(ch) * BN) / n_chunks;
    const int bn1 = (static_cast<int64_t>(ch + 1) * BN) / n_chunks;
    const float invC = 1.0f / static_cast<float>(C);
    const int n_warps = blockDim.x >> 5;

    float qw[4], kw[4], dqnw[4] = {0,0,0,0}, dknw[4] = {0,0,0,0};
    if (act) {
        float4 qwf = *reinterpret_cast<const float4*>(q_nw + c0);
        float4 kwf = *reinterpret_cast<const float4*>(k_nw + c0);
        qw[0]=qwf.x; qw[1]=qwf.y; qw[2]=qwf.z; qw[3]=qwf.w;
        kw[0]=kwf.x; kw[1]=kwf.y; kw[2]=kwf.z; kw[3]=kwf.w;
    }

    __shared__ float s_cos[128], s_sin[128], s_red[40];
    const uint2 z2 = make_uint2(0u, 0u);

    for (int bn = bn0; bn < bn1; ++bn) {
        const int b_idx = bn / N;
        const int n_idx = bn - b_idx * N;
        if (tid < D) {
            s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D + tid];
            s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D + tid];
        }
        __syncthreads();

        const float q_ir = q_inv_rms[bn];
        const float k_ir = k_inv_rms[bn];
        float q_dot = 0.0f, k_dot = 0.0f;
        float qf[4], kf[4], gqw[4], gkw[4];

        if (act) {
            int64_t off4 = ((static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0) >> 2;
            uint2 dqr_u = reinterpret_cast<const uint2*>(dq_rot)[off4];
            uint2 dqp_u = reinterpret_cast<const uint2*>(dq_pos)[off4];
            uint2 dkr_u = reinterpret_cast<const uint2*>(dk_rot)[off4];
            uint2 dkp_u = reinterpret_cast<const uint2*>(dk_pos)[off4];
            uint2 dv_u  = reinterpret_cast<const uint2*>(dv)[off4];

            int64_t qbase4 = static_cast<int64_t>(bn) * 3 * C4;
            int64_t kbase4 = static_cast<int64_t>(bn) * C4;
            reinterpret_cast<uint2*>(dqkv)[qbase4 + 2 * C4 + tid] = dv_u;
            if (dkconv != nullptr) reinterpret_cast<uint2*>(dqkv)[qbase4 + C4 + tid] = z2;

            uint2 q_u = reinterpret_cast<const uint2*>(qkv)[qbase4 + tid];
            uint2 k_u = (kconv != nullptr) ? reinterpret_cast<const uint2*>(kconv)[kbase4 + tid]
                                           : reinterpret_cast<const uint2*>(qkv)[qbase4 + C4 + tid];

            const __mt_bfloat162* dqrh = reinterpret_cast<const __mt_bfloat162*>(&dqr_u);
            const __mt_bfloat162* dqph = reinterpret_cast<const __mt_bfloat162*>(&dqp_u);
            const __mt_bfloat162* dkrh = reinterpret_cast<const __mt_bfloat162*>(&dkr_u);
            const __mt_bfloat162* dkph = reinterpret_cast<const __mt_bfloat162*>(&dkp_u);
            const __mt_bfloat162* qh   = reinterpret_cast<const __mt_bfloat162*>(&q_u);
            const __mt_bfloat162* kh   = reinterpret_cast<const __mt_bfloat162*>(&k_u);

            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                int dd = d0 + p * 2;
                float ce = s_cos[dd], co = s_cos[dd + 1], se = s_sin[dd], so = s_sin[dd + 1];
                float gq0 = fmaf(__low2float(dqrh[p]), ce, __high2float(dqrh[p]) * so) + __low2float(dqph[p]);
                float gq1 = fmaf(__high2float(dqrh[p]), co, __low2float(dqrh[p]) * se) + __high2float(dqph[p]);
                float gk0 = (fmaf(__low2float(dkrh[p]), ce, __high2float(dkrh[p]) * so) + __low2float(dkph[p])) * k_scale;
                float gk1 = (fmaf(__high2float(dkrh[p]), co, __low2float(dkrh[p]) * se) + __high2float(dkph[p])) * k_scale;

                float q0 = __low2float(qh[p]), q1 = __high2float(qh[p]);
                float k0 = __low2float(kh[p]), k1 = __high2float(kh[p]);
                qf[2*p] = q0; qf[2*p+1] = q1;
                kf[2*p] = k0; kf[2*p+1] = k1;

                if (q0 * qw[2*p]   <= 0.0f) gq0 = 0.0f;
                if (q1 * qw[2*p+1] <= 0.0f) gq1 = 0.0f;
                if (k0 * kw[2*p]   <= 0.0f) gk0 = 0.0f;
                if (k1 * kw[2*p+1] <= 0.0f) gk1 = 0.0f;

                gqw[2*p] = gq0 * qw[2*p]; gqw[2*p+1] = gq1 * qw[2*p+1];
                gkw[2*p] = gk0 * kw[2*p]; gkw[2*p+1] = gk1 * kw[2*p+1];
                dqnw[2*p]   = fmaf(gq0 * q0, q_ir, dqnw[2*p]);
                dqnw[2*p+1] = fmaf(gq1 * q1, q_ir, dqnw[2*p+1]);
                dknw[2*p]   = fmaf(gk0 * k0, k_ir, dknw[2*p]);
                dknw[2*p+1] = fmaf(gk1 * k1, k_ir, dknw[2*p+1]);
                q_dot = fmaf(gqw[2*p], q0, fmaf(gqw[2*p+1], q1, q_dot));
                k_dot = fmaf(gkw[2*p], k0, fmaf(gkw[2*p+1], k1, k_dot));
            }
        }
        q_dot = warp_sum(q_dot);
        k_dot = warp_sum(k_dot);
        if ((tid & 31) == 0) { s_red[tid >> 5] = q_dot; s_red[20 + (tid >> 5)] = k_dot; }
        __syncthreads();
        if (tid == 0) {
            float sq = 0.0f, sk = 0.0f;
            for (int w = 0; w < n_warps; ++w) { sq += s_red[w]; sk += s_red[20 + w]; }
            s_red[0] = sq * q_ir * q_ir * invC;
            s_red[20] = sk * k_ir * k_ir * invC;
        }
        __syncthreads();
        if (act) {
            float qc = s_red[0], kc = s_red[20];
            uint2 dq_o, dk_o;
            __mt_bfloat162* dqh = reinterpret_cast<__mt_bfloat162*>(&dq_o);
            __mt_bfloat162* dkh = reinterpret_cast<__mt_bfloat162*>(&dk_o);
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                dqh[p] = __floats2bfloat162_rn(q_ir * (gqw[2*p] - qf[2*p] * qc), q_ir * (gqw[2*p+1] - qf[2*p+1] * qc));
                dkh[p] = __floats2bfloat162_rn(k_ir * (gkw[2*p] - kf[2*p] * kc), k_ir * (gkw[2*p+1] - kf[2*p+1] * kc));
            }
            int64_t qbase4 = static_cast<int64_t>(bn) * 3 * C4;
            int64_t kbase4 = static_cast<int64_t>(bn) * C4;
            reinterpret_cast<uint2*>(dqkv)[qbase4 + tid] = dq_o;
            if (dkconv != nullptr) reinterpret_cast<uint2*>(dkconv)[kbase4 + tid] = dk_o;
            else reinterpret_cast<uint2*>(dqkv)[qbase4 + C4 + tid] = dk_o;
        }
    }
    if (act) {
        int64_t pb = static_cast<int64_t>(ch) * C + c0;
        *reinterpret_cast<float4*>(dq_nw_part + pb) = make_float4(dqnw[0], dqnw[1], dqnw[2], dqnw[3]);
        *reinterpret_cast<float4*>(dk_nw_part + pb) = make_float4(dknw[0], dknw[1], dknw[2], dknw[3]);
    }
}

extern "C" __global__ void main_gdn_prep_bwd_f32(
    const __mt_bfloat162* __restrict__ qkv2, const __mt_bfloat162* __restrict__ kconv2,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ q_ir, const float* __restrict__ k_ir,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    const float* __restrict__ dq_rot, const float* __restrict__ dq_pos,
    const float* __restrict__ dk_rot, const float* __restrict__ dk_pos, const float* __restrict__ dv,
    __mt_bfloat162* __restrict__ dqkv2, __mt_bfloat162* __restrict__ dkconv2,
    float* __restrict__ dq_nw_part, float* __restrict__ dk_nw_part,
    int B, int N, int H, int D, float k_scale, int has_norm, int n_chunks
) {
    main_gdn_prep_bwd_impl<float>(qkv2, kconv2, q_nw, k_nw, q_ir, k_ir, rope_cos, rope_sin,
                                  dq_rot, dq_pos, dk_rot, dk_pos, dv, dqkv2, dkconv2,
                                  dq_nw_part, dk_nw_part, B, N, H, D, k_scale, has_norm, n_chunks);
}

extern "C" __global__ void main_gdn_prep_bwd_bf16(
    const __mt_bfloat162* __restrict__ qkv2, const __mt_bfloat162* __restrict__ kconv2,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ q_ir, const float* __restrict__ k_ir,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    const __mt_bfloat16* __restrict__ dq_rot, const __mt_bfloat16* __restrict__ dq_pos,
    const __mt_bfloat16* __restrict__ dk_rot, const __mt_bfloat16* __restrict__ dk_pos, const __mt_bfloat16* __restrict__ dv,
    __mt_bfloat162* __restrict__ dqkv2, __mt_bfloat162* __restrict__ dkconv2,
    float* __restrict__ dq_nw_part, float* __restrict__ dk_nw_part,
    int B, int N, int H, int D, float k_scale, int has_norm, int n_chunks
) {
    main_gdn_prep_bwd_impl<__mt_bfloat16>(qkv2, kconv2, q_nw, k_nw, q_ir, k_ir, rope_cos, rope_sin,
                                          dq_rot, dq_pos, dk_rot, dk_pos, dv, dqkv2, dkconv2,
                                          dq_nw_part, dk_nw_part, B, N, H, D, k_scale, has_norm, n_chunks);
}

// =========================================================================
// 3. Fused Bidirectional D x D State Scan (D=112, FP32) — Forward & Backward (v4a RT=16, 1-bar double-buf)
//    Grid: (7 row_tiles of 16 rows, BH, 2 dirs) = 280 CTAs, Block: 128 (112 active: tx=0..27, ty=0..3)
//    KT=16 double-buffered s_W[2][16][116] -> 22.2 KiB SMEM, 1 barrier/K-tile, -27.5% latency vs v3!
// =========================================================================
constexpr int SCAN_D = 112;
constexpr int SCAN_D4 = 28;       // 112 / 4
constexpr int SCAN_STRIDE = 116;  // 112 + 4 -> bank-conflict-free shared-memory stride
constexpr int SCAN_RT = 16;       // rows per CTA (112 / 7 = 16)
constexpr int SCAN_KT = 16;       // K-tile size (112 / 16 = 7 tiles)

extern "C" __global__ void bidi_state_scan_fwd_d112_f32(
    const float* __restrict__ W,     // (BH, T, 112, 112)
    const float* __restrict__ U,     // (BH, T, 112, 112)
    float* __restrict__ S_fwd,       // (BH, T, 112, 112)
    float* __restrict__ S_rev,       // (BH, T, 112, 112)
    int BH, int T
) {
    const int rtile = blockIdx.x;    // 0..6 -> 16 rows
    const int bh = blockIdx.y;       // 0..BH-1
    const int dir = blockIdx.z;      // 0=fwd inclusive, 1=rev exclusive
    const int tid = threadIdx.x;     // 0..127 (0..111 active)
    const bool act = (tid < 112);
    const int tx = act ? (tid % SCAN_D4) : 0;    // 0..27 -> col0 = tx * 4
    const int ty = act ? (tid / SCAN_D4) : 0;    // 0..3  -> r0 = ty * 4 (4 rows/thread)
    const int row0 = rtile * SCAN_RT;
    const int col0 = tx * 4;
    const int rbase = row0 + ty * 4;

    __shared__ float s_S[SCAN_RT][SCAN_STRIDE];
    __shared__ float s_W[2][SCAN_KT][SCAN_STRIDE];

    const int64_t dd = static_cast<int64_t>(SCAN_D) * SCAN_D;
    const int64_t bh_base = static_cast<int64_t>(bh) * T * dd;

    auto load_w = [&](const float* W_ptr, int k0, int buf) {
        if (act) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int kr = ty * 4 + i;
                float4 w4 = *reinterpret_cast<const float4*>(&W_ptr[(k0 + kr) * SCAN_D + col0]);
                s_W[buf][kr][col0 + 0] = w4.x; s_W[buf][kr][col0 + 1] = w4.y;
                s_W[buf][kr][col0 + 2] = w4.z; s_W[buf][kr][col0 + 3] = w4.w;
            }
        }
    };

    auto step_matmul = [&](const float* W_ptr, const float* U_ptr, float* Out_ptr) {
        float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;
        float c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0;
        if (act) {
            float4 u0 = *reinterpret_cast<const float4*>(&U_ptr[(rbase + 0) * SCAN_D + col0]);
            float4 u1 = *reinterpret_cast<const float4*>(&U_ptr[(rbase + 1) * SCAN_D + col0]);
            float4 u2 = *reinterpret_cast<const float4*>(&U_ptr[(rbase + 2) * SCAN_D + col0]);
            float4 u3 = *reinterpret_cast<const float4*>(&U_ptr[(rbase + 3) * SCAN_D + col0]);
            c00 = u0.x; c01 = u0.y; c02 = u0.z; c03 = u0.w;
            c10 = u1.x; c11 = u1.y; c12 = u1.z; c13 = u1.w;
            c20 = u2.x; c21 = u2.y; c22 = u2.z; c23 = u2.w;
            c30 = u3.x; c31 = u3.y; c32 = u3.z; c33 = u3.w;
        }
        load_w(W_ptr, 0, 0);
        for (int tile = 0; tile < 7; ++tile) {
            __syncthreads();
            int cur = tile & 1, nxt = cur ^ 1, k0 = tile * SCAN_KT;
            if (tile < 6) load_w(W_ptr, k0 + SCAN_KT, nxt);
            if (act) {
                #pragma unroll
                for (int kk = 0; kk < SCAN_KT; ++kk) {
                    float w0 = s_W[cur][kk][col0 + 0], w1 = s_W[cur][kk][col0 + 1];
                    float w2 = s_W[cur][kk][col0 + 2], w3 = s_W[cur][kk][col0 + 3];
                    float s0 = s_S[ty * 4 + 0][k0 + kk];
                    float s1 = s_S[ty * 4 + 1][k0 + kk];
                    float s2 = s_S[ty * 4 + 2][k0 + kk];
                    float s3 = s_S[ty * 4 + 3][k0 + kk];
                    c00 = fmaf(s0, w0, c00); c01 = fmaf(s0, w1, c01); c02 = fmaf(s0, w2, c02); c03 = fmaf(s0, w3, c03);
                    c10 = fmaf(s1, w0, c10); c11 = fmaf(s1, w1, c11); c12 = fmaf(s1, w2, c12); c13 = fmaf(s1, w3, c13);
                    c20 = fmaf(s2, w0, c20); c21 = fmaf(s2, w1, c21); c22 = fmaf(s2, w2, c22); c23 = fmaf(s2, w3, c23);
                    c30 = fmaf(s3, w0, c30); c31 = fmaf(s3, w1, c31); c32 = fmaf(s3, w2, c32); c33 = fmaf(s3, w3, c33);
                }
            }
        }
        if (act) {
            *reinterpret_cast<float4*>(&Out_ptr[(rbase + 0) * SCAN_D + col0]) = make_float4(c00, c01, c02, c03);
            *reinterpret_cast<float4*>(&Out_ptr[(rbase + 1) * SCAN_D + col0]) = make_float4(c10, c11, c12, c13);
            *reinterpret_cast<float4*>(&Out_ptr[(rbase + 2) * SCAN_D + col0]) = make_float4(c20, c21, c22, c23);
            *reinterpret_cast<float4*>(&Out_ptr[(rbase + 3) * SCAN_D + col0]) = make_float4(c30, c31, c32, c33);
        }
        __syncthreads();
        if (act) {
            s_S[ty * 4 + 0][col0 + 0] = c00; s_S[ty * 4 + 0][col0 + 1] = c01; s_S[ty * 4 + 0][col0 + 2] = c02; s_S[ty * 4 + 0][col0 + 3] = c03;
            s_S[ty * 4 + 1][col0 + 0] = c10; s_S[ty * 4 + 1][col0 + 1] = c11; s_S[ty * 4 + 1][col0 + 2] = c12; s_S[ty * 4 + 1][col0 + 3] = c13;
            s_S[ty * 4 + 2][col0 + 0] = c20; s_S[ty * 4 + 2][col0 + 1] = c21; s_S[ty * 4 + 2][col0 + 2] = c22; s_S[ty * 4 + 2][col0 + 3] = c23;
            s_S[ty * 4 + 3][col0 + 0] = c30; s_S[ty * 4 + 3][col0 + 1] = c31; s_S[ty * 4 + 3][col0 + 2] = c32; s_S[ty * 4 + 3][col0 + 3] = c33;
        }
        __syncthreads();
    };

    if (dir == 0) {
        const float* U0 = U + bh_base;
        float* Sf0 = S_fwd + bh_base;
        if (act) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 u4 = *reinterpret_cast<const float4*>(&U0[(rbase + i) * SCAN_D + col0]);
                *reinterpret_cast<float4*>(&Sf0[(rbase + i) * SCAN_D + col0]) = u4;
                s_S[ty * 4 + i][col0 + 0] = u4.x; s_S[ty * 4 + i][col0 + 1] = u4.y;
                s_S[ty * 4 + i][col0 + 2] = u4.z; s_S[ty * 4 + i][col0 + 3] = u4.w;
            }
        }
        __syncthreads();
        for (int t = 1; t < T; ++t) {
            step_matmul(W + bh_base + static_cast<int64_t>(t) * dd,
                        U + bh_base + static_cast<int64_t>(t) * dd,
                        S_fwd + bh_base + static_cast<int64_t>(t) * dd);
        }
    } else {
        float* Sr_last = S_rev + bh_base + static_cast<int64_t>(T - 1) * dd;
        const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (act) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                *reinterpret_cast<float4*>(&Sr_last[(rbase + i) * SCAN_D + col0]) = z4;
            }
        }
        if (T >= 2) {
            const float* Up = U + bh_base + static_cast<int64_t>(T - 1) * dd;
            float* Sr_t = S_rev + bh_base + static_cast<int64_t>(T - 2) * dd;
            if (act) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float4 u4 = *reinterpret_cast<const float4*>(&Up[(rbase + i) * SCAN_D + col0]);
                    *reinterpret_cast<float4*>(&Sr_t[(rbase + i) * SCAN_D + col0]) = u4;
                    s_S[ty * 4 + i][col0 + 0] = u4.x; s_S[ty * 4 + i][col0 + 1] = u4.y;
                    s_S[ty * 4 + i][col0 + 2] = u4.z; s_S[ty * 4 + i][col0 + 3] = u4.w;
                }
            }
            __syncthreads();
            for (int t = T - 3; t >= 0; --t) {
                step_matmul(W + bh_base + static_cast<int64_t>(t + 1) * dd,
                            U + bh_base + static_cast<int64_t>(t + 1) * dd,
                            S_rev + bh_base + static_cast<int64_t>(t) * dd);
            }
        }
    }
}

// Backward scan (v3 2D 4x4 register tiling, RT=28, KT=16, 20.4 KiB SMEM, block=196, grid=(4,BH,2)):
//   dir=0: g_fwd[T-1] = dS_tot[T-1]; g_fwd[t] = dS_tot[t] + g_fwd[t+1] @ W[t+1]^T  (t=T-2..0)
//   dir=1: g_rev[0]   = dS_tot[0];   g_rev[t] = dS_tot[t] + g_rev[t-1] @ W[t]^T    (t=1..T-1)
extern "C" __global__ void bidi_state_scan_bwd_d112_f32(
    const float* __restrict__ W,
    const float* __restrict__ dS_tot,
    float* __restrict__ g_fwd,
    float* __restrict__ g_rev,
    int BH, int T
) {
    constexpr int BWD_RT = 28;
    const int rtile = blockIdx.x;
    const int bh = blockIdx.y;
    const int dir = blockIdx.z;
    const int tid = threadIdx.x;     // 0..195
    const int tx = tid % SCAN_D4;    // 0..27
    const int ty = tid / SCAN_D4;    // 0..6
    const int row0 = rtile * BWD_RT;
    const int col0 = tx * 4;
    const int rbase = row0 + ty * 4;

    __shared__ float s_G[BWD_RT][SCAN_STRIDE];
    __shared__ float s_WT[SCAN_KT][SCAN_STRIDE];

    const int64_t dd = static_cast<int64_t>(SCAN_D) * SCAN_D;
    const int64_t bh_base = static_cast<int64_t>(bh) * T * dd;

    auto step_matmul_bwd = [&](const float* W_ptr, const float* dS_ptr, float* Out_ptr) {
        float4 u0 = *reinterpret_cast<const float4*>(&dS_ptr[(rbase + 0) * SCAN_D + col0]);
        float4 u1 = *reinterpret_cast<const float4*>(&dS_ptr[(rbase + 1) * SCAN_D + col0]);
        float4 u2 = *reinterpret_cast<const float4*>(&dS_ptr[(rbase + 2) * SCAN_D + col0]);
        float4 u3 = *reinterpret_cast<const float4*>(&dS_ptr[(rbase + 3) * SCAN_D + col0]);
        float c00 = u0.x, c01 = u0.y, c02 = u0.z, c03 = u0.w;
        float c10 = u1.x, c11 = u1.y, c12 = u1.z, c13 = u1.w;
        float c20 = u2.x, c21 = u2.y, c22 = u2.z, c23 = u2.w;
        float c30 = u3.x, c31 = u3.y, c32 = u3.z, c33 = u3.w;

        for (int k0 = 0; k0 < SCAN_D; k0 += SCAN_KT) {
            for (int idx4 = tid; idx4 < SCAN_D * 4; idx4 += 196) {
                int c = idx4 >> 2;
                int k_off = (idx4 & 3) << 2;
                float4 w4 = *reinterpret_cast<const float4*>(&W_ptr[c * SCAN_D + k0 + k_off]);
                s_WT[k_off + 0][c] = w4.x;
                s_WT[k_off + 1][c] = w4.y;
                s_WT[k_off + 2][c] = w4.z;
                s_WT[k_off + 3][c] = w4.w;
            }
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < SCAN_KT; ++kk) {
                float w0 = s_WT[kk][col0 + 0], w1 = s_WT[kk][col0 + 1];
                float w2 = s_WT[kk][col0 + 2], w3 = s_WT[kk][col0 + 3];
                float g0 = s_G[ty * 4 + 0][k0 + kk];
                float g1 = s_G[ty * 4 + 1][k0 + kk];
                float g2 = s_G[ty * 4 + 2][k0 + kk];
                float g3 = s_G[ty * 4 + 3][k0 + kk];
                c00 = fmaf(g0, w0, c00); c01 = fmaf(g0, w1, c01); c02 = fmaf(g0, w2, c02); c03 = fmaf(g0, w3, c03);
                c10 = fmaf(g1, w0, c10); c11 = fmaf(g1, w1, c11); c12 = fmaf(g1, w2, c12); c13 = fmaf(g1, w3, c13);
                c20 = fmaf(g2, w0, c20); c21 = fmaf(g2, w1, c21); c22 = fmaf(g2, w2, c22); c23 = fmaf(g2, w3, c23);
                c30 = fmaf(g3, w0, c30); c31 = fmaf(g3, w1, c31); c32 = fmaf(g3, w2, c32); c33 = fmaf(g3, w3, c33);
            }
            __syncthreads();
        }
        *reinterpret_cast<float4*>(&Out_ptr[(rbase + 0) * SCAN_D + col0]) = make_float4(c00, c01, c02, c03);
        *reinterpret_cast<float4*>(&Out_ptr[(rbase + 1) * SCAN_D + col0]) = make_float4(c10, c11, c12, c13);
        *reinterpret_cast<float4*>(&Out_ptr[(rbase + 2) * SCAN_D + col0]) = make_float4(c20, c21, c22, c23);
        *reinterpret_cast<float4*>(&Out_ptr[(rbase + 3) * SCAN_D + col0]) = make_float4(c30, c31, c32, c33);
        s_G[ty * 4 + 0][col0 + 0] = c00; s_G[ty * 4 + 0][col0 + 1] = c01; s_G[ty * 4 + 0][col0 + 2] = c02; s_G[ty * 4 + 0][col0 + 3] = c03;
        s_G[ty * 4 + 1][col0 + 0] = c10; s_G[ty * 4 + 1][col0 + 1] = c11; s_G[ty * 4 + 1][col0 + 2] = c12; s_G[ty * 4 + 1][col0 + 3] = c13;
        s_G[ty * 4 + 2][col0 + 0] = c20; s_G[ty * 4 + 2][col0 + 1] = c21; s_G[ty * 4 + 2][col0 + 2] = c22; s_G[ty * 4 + 2][col0 + 3] = c23;
        s_G[ty * 4 + 3][col0 + 0] = c30; s_G[ty * 4 + 3][col0 + 1] = c31; s_G[ty * 4 + 3][col0 + 2] = c32; s_G[ty * 4 + 3][col0 + 3] = c33;
        __syncthreads();
    };

    if (dir == 0) {
        const float* dS_last = dS_tot + bh_base + static_cast<int64_t>(T - 1) * dd;
        float* gf_last = g_fwd + bh_base + static_cast<int64_t>(T - 1) * dd;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 v = *reinterpret_cast<const float4*>(&dS_last[(rbase + i) * SCAN_D + col0]);
            *reinterpret_cast<float4*>(&gf_last[(rbase + i) * SCAN_D + col0]) = v;
            s_G[ty * 4 + i][col0 + 0] = v.x; s_G[ty * 4 + i][col0 + 1] = v.y;
            s_G[ty * 4 + i][col0 + 2] = v.z; s_G[ty * 4 + i][col0 + 3] = v.w;
        }
        __syncthreads();
        for (int t = T - 2; t >= 0; --t) {
            step_matmul_bwd(W + bh_base + static_cast<int64_t>(t + 1) * dd,
                            dS_tot + bh_base + static_cast<int64_t>(t) * dd,
                            g_fwd + bh_base + static_cast<int64_t>(t) * dd);
        }
    } else {
        const float* dS0 = dS_tot + bh_base;
        float* gr0 = g_rev + bh_base;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 v = *reinterpret_cast<const float4*>(&dS0[(rbase + i) * SCAN_D + col0]);
            *reinterpret_cast<float4*>(&gr0[(rbase + i) * SCAN_D + col0]) = v;
            s_G[ty * 4 + i][col0 + 0] = v.x; s_G[ty * 4 + i][col0 + 1] = v.y;
            s_G[ty * 4 + i][col0 + 2] = v.z; s_G[ty * 4 + i][col0 + 3] = v.w;
        }
        __syncthreads();
        for (int t = 1; t < T; ++t) {
            step_matmul_bwd(W + bh_base + static_cast<int64_t>(t) * dd,
                            dS_tot + bh_base + static_cast<int64_t>(t) * dd,
                            g_rev + bh_base + static_cast<int64_t>(t) * dd);
        }
    }
}

// =========================================================================
// 4. Fused Bidirectional Z-state Vector Scan (D=112, FP32) — Forward & Backward
//    Grid: (BH, 2 dirs), Block: 112 (1 thread per column d)
// =========================================================================
extern "C" __global__ void bidi_z_scan_fwd_d112_f32(
    const float* __restrict__ Wz,    // (BH, T, 112, 112)
    const float* __restrict__ Uz,    // (BH, T, 112)
    float* __restrict__ Sz_fwd,      // (BH, T, 112)
    float* __restrict__ Sz_rev,      // (BH, T, 112)
    int BH, int T
) {
    const int bh = blockIdx.x;
    const int dir = blockIdx.y;
    const int d = threadIdx.x;       // 0..111
    __shared__ float s_z[SCAN_D];

    const int64_t dd = static_cast<int64_t>(SCAN_D) * SCAN_D;
    const int64_t bh_w = static_cast<int64_t>(bh) * T * dd;
    const int64_t bh_v = static_cast<int64_t>(bh) * T * SCAN_D;

    if (dir == 0) {
        for (int t = 0; t < T; ++t) {
            float val = Uz[bh_v + static_cast<int64_t>(t) * SCAN_D + d];
            if (t > 0) {
                const float* Wt = Wz + bh_w + static_cast<int64_t>(t) * dd;
                for (int k = 0; k < SCAN_D; ++k) val = fmaf(s_z[k], Wt[k * SCAN_D + d], val);
            }
            __syncthreads();
            s_z[d] = val;
            Sz_fwd[bh_v + static_cast<int64_t>(t) * SCAN_D + d] = val;
            __syncthreads();
        }
    } else {
        Sz_rev[bh_v + static_cast<int64_t>(T - 1) * SCAN_D + d] = 0.0f;
        for (int t = T - 2; t >= 0; --t) {
            float val = Uz[bh_v + static_cast<int64_t>(t + 1) * SCAN_D + d];
            if (t < T - 2) {
                const float* Wp = Wz + bh_w + static_cast<int64_t>(t + 1) * dd;
                for (int k = 0; k < SCAN_D; ++k) val = fmaf(s_z[k], Wp[k * SCAN_D + d], val);
            }
            __syncthreads();
            s_z[d] = val;
            Sz_rev[bh_v + static_cast<int64_t>(t) * SCAN_D + d] = val;
            __syncthreads();
        }
    }
}

extern "C" __global__ void bidi_z_scan_bwd_d112_f32(
    const float* __restrict__ Wz,
    const float* __restrict__ dSz_tot,
    float* __restrict__ gz_fwd,
    float* __restrict__ gz_rev,
    int BH, int T
) {
    const int bh = blockIdx.x;
    const int dir = blockIdx.y;
    const int d = threadIdx.x;
    __shared__ float s_g[SCAN_D];

    const int64_t dd = static_cast<int64_t>(SCAN_D) * SCAN_D;
    const int64_t bh_w = static_cast<int64_t>(bh) * T * dd;
    const int64_t bh_v = static_cast<int64_t>(bh) * T * SCAN_D;

    if (dir == 0) {
        for (int t = T - 1; t >= 0; --t) {
            float val = dSz_tot[bh_v + static_cast<int64_t>(t) * SCAN_D + d];
            if (t < T - 1) {
                const float* Wp = Wz + bh_w + static_cast<int64_t>(t + 1) * dd;
                // (g_prev @ W^T)[d] = sum_k g_prev[k] * W[d, k]
                for (int k = 0; k < SCAN_D; ++k) val = fmaf(s_g[k], Wp[d * SCAN_D + k], val);
            }
            __syncthreads();
            s_g[d] = val;
            gz_fwd[bh_v + static_cast<int64_t>(t) * SCAN_D + d] = val;
            __syncthreads();
        }
    } else {
        for (int t = 0; t < T; ++t) {
            float val = dSz_tot[bh_v + static_cast<int64_t>(t) * SCAN_D + d];
            if (t > 0) {
                const float* Wt = Wz + bh_w + static_cast<int64_t>(t) * dd;
                for (int k = 0; k < SCAN_D; ++k) val = fmaf(s_g[k], Wt[d * SCAN_D + k], val);
            }
            __syncthreads();
            s_g[d] = val;
            gz_rev[bh_v + static_cast<int64_t>(t) * SCAN_D + d] = val;
            __syncthreads();
        }
    }
}

// =========================================================================
// 5. Fused Output Divide + Transpose: num(BH,N,D) / (den(BH,N)+eps) -> out(B,N,C) BF16
//    Grid: (B * N), Block: 256
// =========================================================================
template <typename InT>
__device__ void out_div_to_bnc_fwd_impl(
    const InT* __restrict__ num,      // (BH, N, D)
    const float* __restrict__ den,    // (BH, N)
    __mt_bfloat162* __restrict__ out2,// (B, N, C/2)
    int B, int N, int H, int D, float eps
) {
    const int bn = blockIdx.x;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int C2 = (H * D) >> 1;
    const int D2 = D >> 1;
    __mt_bfloat162* o_row = out2 + static_cast<int64_t>(bn) * C2;

    for (int p = threadIdx.x; p < C2; p += 256) {
        int h = p / D2;
        int d0 = (p - h * D2) * 2;
        int64_t bh = static_cast<int64_t>(b_idx * H + h);
        float inv_d = 1.0f / (den[bh * N + n_idx] + eps);
        float2 nv = load_pair<InT>(num + (bh * N + n_idx) * D + d0);
        o_row[p] = __floats2bfloat162_rn(nv.x * inv_d, nv.y * inv_d);
    }
}

extern "C" __global__ void out_div_to_bnc_fwd_f32(
    const float* __restrict__ num, const float* __restrict__ den, __mt_bfloat162* __restrict__ out2,
    int B, int N, int H, int D, float eps
) { out_div_to_bnc_fwd_impl<float>(num, den, out2, B, N, H, D, eps); }

extern "C" __global__ void out_div_to_bnc_fwd_bf16(
    const __mt_bfloat16* __restrict__ num, const float* __restrict__ den, __mt_bfloat162* __restrict__ out2,
    int B, int N, int H, int D, float eps
) { out_div_to_bnc_fwd_impl<__mt_bfloat16>(num, den, out2, B, N, H, D, eps); }

// Backward: dout(B,N,C) BF16 + num(BH,N,D) + den(BH,N) -> dnum(BH,N,D) + dden(BH,N)
// Grid: (B * N, H), Block: 64 (1 warp/2 warps per (bn, h), handles D=112 cleanly!)
template <typename ValT>
__device__ void out_div_to_bnc_bwd_impl(
    const __mt_bfloat162* __restrict__ dout2,  // (B, N, C/2)
    const ValT* __restrict__ num,              // (BH, N, D)
    const float* __restrict__ den,             // (BH, N)
    ValT* __restrict__ dnum,                   // (BH, N, D)
    float* __restrict__ dden,                  // (BH, N)
    int B, int N, int H, int D, float eps
) {
    const int bn = blockIdx.x;
    const int h = blockIdx.y;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int D2 = D >> 1;
    const int C2 = (H * D) >> 1;
    const int tid = threadIdx.x;  // 0..63
    const int64_t bh = static_cast<int64_t>(b_idx * H + h);
    const int64_t bhn = bh * N + n_idx;
    const float inv_d = 1.0f / (den[bhn] + eps);

    float dot = 0.0f;
    if (tid < D2) {
        __mt_bfloat162 dv = dout2[static_cast<int64_t>(bn) * C2 + h * D2 + tid];
        float g0 = __low2float(dv), g1 = __high2float(dv);
        float2 nv = load_pair<ValT>(num + bhn * D + tid * 2);
        store_pair<ValT>(dnum + bhn * D + tid * 2, g0 * inv_d, g1 * inv_d);
        dot = fmaf(g0, nv.x, g1 * nv.y);
    }
    dot = warp_sum(dot);
    __shared__ float s_w[2];
    if ((tid & 31) == 0) s_w[tid >> 5] = dot;
    __syncthreads();
    if (tid == 0) {
        dden[bhn] = -(s_w[0] + s_w[1]) * inv_d * inv_d;
    }
}

extern "C" __global__ void out_div_to_bnc_bwd_f32(
    const __mt_bfloat162* __restrict__ dout2, const float* __restrict__ num, const float* __restrict__ den,
    float* __restrict__ dnum, float* __restrict__ dden, int B, int N, int H, int D, float eps
) { out_div_to_bnc_bwd_impl<float>(dout2, num, den, dnum, dden, B, N, H, D, eps); }

extern "C" __global__ void out_div_to_bnc_bwd_bf16(
    const __mt_bfloat162* __restrict__ dout2, const __mt_bfloat16* __restrict__ num, const float* __restrict__ den,
    __mt_bfloat16* __restrict__ dnum, float* __restrict__ dden, int B, int N, int H, int D, float eps
) { out_div_to_bnc_bwd_impl<__mt_bfloat16>(dout2, num, den, dnum, dden, B, N, H, D, eps); }

// =========================================================================
// 6. Fused Cam Prep: q_raw, k_raw, v_raw (B, N, C) BF16 -> q_out, k_out, v_out (BH, N, D) + inflation_sq (BH, N)
//    RMSNorm + ReLU + k_scale + 4x4 P_T/P_inv on [0..D/2) + RoPE on [D/2..D)
//    Grid: (B * N), Block: 256
// =========================================================================
template <typename OutT>
__device__ void cam_prep_fwd_impl(
    const __mt_bfloat162* __restrict__ q2,     // (B, N, C/2)
    const __mt_bfloat162* __restrict__ k2,     // (B, N, C/2)
    const __mt_bfloat162* __restrict__ v2,     // (B, N, C/2)
    const float* __restrict__ q_nw,            // (C,)
    const float* __restrict__ k_nw,            // (C,)
    const float* __restrict__ P_T,             // (B, N, 16)
    const float* __restrict__ P_inv,           // (B, N, 16)
    const float* __restrict__ rope_cos,        // (N, D/2)
    const float* __restrict__ rope_sin,        // (N, D/2)
    OutT* __restrict__ q_out,                  // (BH, N, D)
    OutT* __restrict__ k_out,                  // (BH, N, D)
    OutT* __restrict__ v_out,                  // (BH, N, D)
    float* __restrict__ inflation_sq,          // (BH, N)
    float* __restrict__ q_ir_out,              // (B*N,)
    float* __restrict__ k_ir_out,              // (B*N,)
    float* __restrict__ pre_ksq_out,           // (BH, N)
    float* __restrict__ post_ksq_out,          // (BH, N)
    int B, int N, int H, int D, float k_scale, float norm_eps
) {
    const int bn = blockIdx.x;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int tid = threadIdx.x;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D_half = D >> 1;
    const int n_grp4 = D_half >> 2;  // 14 for D=112

    __shared__ float s_PT[16], s_Pinv[16];
    __shared__ float s_cos[64], s_sin[64];
    __shared__ float s_qred[8], s_kred[8];
    __shared__ float s_pre_g4[32 * 16], s_post_g4[32 * 16];
    __shared__ float s_pre_r2[32 * 32], s_post_r2[32 * 32];

    if (tid < 16) {
        s_PT[tid] = P_T[static_cast<int64_t>(bn) * 16 + tid];
        s_Pinv[tid] = P_inv[static_cast<int64_t>(bn) * 16 + tid];
    }
    if (tid < D_half) {
        s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D_half + tid];
        s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D_half + tid];
    }

    const __mt_bfloat162* qr = (v2 != nullptr) ? (q2 + static_cast<int64_t>(bn) * C2)
                                               : (q2 + static_cast<int64_t>(bn) * 3 * C2);
    const __mt_bfloat162* kr = (k2 != nullptr) ? (k2 + static_cast<int64_t>(bn) * C2)
                                               : (qr + C2);
    const __mt_bfloat162* vr = (v2 != nullptr) ? (v2 + static_cast<int64_t>(bn) * C2)
                                               : (qr + 2 * C2);

    extern __shared__ __mt_bfloat162 smem_cpf[];
    __mt_bfloat162* s_q = smem_cpf;
    __mt_bfloat162* s_k = s_q + C2;

    float q_sq = 0.0f, k_sq = 0.0f;
    for (int p = tid; p < C2; p += 256) {
        __mt_bfloat162 qv = qr[p], kv = kr[p];
        s_q[p] = qv;
        s_k[p] = kv;
        float q0 = __low2float(qv), q1 = __high2float(qv);
        float k0 = __low2float(kv), k1 = __high2float(kv);
        q_sq = fmaf(q0, q0, fmaf(q1, q1, q_sq));
        k_sq = fmaf(k0, k0, fmaf(k1, k1, k_sq));
    }
    q_sq = warp_sum(q_sq);
    k_sq = warp_sum(k_sq);
    if ((tid & 31) == 0) { s_qred[tid >> 5] = q_sq; s_kred[tid >> 5] = k_sq; }
    __syncthreads();
    if (tid == 0) {
        float tq = 0, tk = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { tq += s_qred[w]; tk += s_kred[w]; }
        s_qred[0] = rsqrtf(tq / static_cast<float>(C) + norm_eps);
        s_kred[0] = rsqrtf(tk / static_cast<float>(C) + norm_eps);
        q_ir_out[bn] = s_qred[0];
        k_ir_out[bn] = s_kred[0];
    }
    __syncthreads();
    const float q_ir = s_qred[0], k_ir = s_kred[0];

    // Process 4-element groups: H * n_grp4 groups for first half [0..D_half) and H * (D_half/2) pairs for second half [D_half..D)
    // Total 4-groups for first half = 20 * 14 = 280. Each thread processes 1-2 groups.
    const int total_g4 = H * n_grp4;
    for (int gi = tid; gi < total_g4; gi += 256) {
        int h = gi / n_grp4;
        int g = gi - h * n_grp4;
        int d0 = g * 4;
        int c0 = h * D + d0;
        int p0 = c0 >> 1;

        __mt_bfloat162 qa = s_q[p0], qb = s_q[p0 + 1];
        __mt_bfloat162 ka = s_k[p0], kb = s_k[p0 + 1];
        __mt_bfloat162 va = vr[p0], vb = vr[p0 + 1];

        float qx0 = fmaxf(0.0f, __low2float(qa)  * q_ir * q_nw[c0 + 0]);
        float qx1 = fmaxf(0.0f, __high2float(qa) * q_ir * q_nw[c0 + 1]);
        float qx2 = fmaxf(0.0f, __low2float(qb)  * q_ir * q_nw[c0 + 2]);
        float qx3 = fmaxf(0.0f, __high2float(qb) * q_ir * q_nw[c0 + 3]);

        float kx0 = fmaxf(0.0f, __low2float(ka)  * k_ir * k_nw[c0 + 0]) * k_scale;
        float kx1 = fmaxf(0.0f, __high2float(ka) * k_ir * k_nw[c0 + 1]) * k_scale;
        float kx2 = fmaxf(0.0f, __low2float(kb)  * k_ir * k_nw[c0 + 2]) * k_scale;
        float kx3 = fmaxf(0.0f, __high2float(kb) * k_ir * k_nw[c0 + 3]) * k_scale;

        float vx0 = __low2float(va), vx1 = __high2float(va), vx2 = __low2float(vb), vx3 = __high2float(vb);

        float pre_s = kx0*kx0 + kx1*kx1 + kx2*kx2 + kx3*kx3;

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

        float post_s = ky0*ky0 + ky1*ky1 + ky2*ky2 + ky3*ky3;
        s_pre_g4[gi] = pre_s;
        s_post_g4[gi] = post_s;

        int64_t out_off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0;
        store_pair<OutT>(q_out + out_off,     qy0, qy1);
        store_pair<OutT>(q_out + out_off + 2, qy2, qy3);
        store_pair<OutT>(k_out + out_off,     ky0, ky1);
        store_pair<OutT>(k_out + out_off + 2, ky2, ky3);
        store_pair<OutT>(v_out + out_off,     vy0, vy1);
        store_pair<OutT>(v_out + out_off + 2, vy2, vy3);
    }

    // Second half [D_half..D): RoPE in pairs (20 * 28 = 560 pairs)
    const int D_half2 = D_half >> 1;
    const int total_r2 = H * D_half2;
    for (int ri = tid; ri < total_r2; ri += 256) {
        int h = ri / D_half2;
        int d2 = ri - h * D_half2;
        int d_rel = d2 * 2;
        int d0 = D_half + d_rel;
        int c0 = h * D + d0;
        int p0 = c0 >> 1;

        __mt_bfloat162 qv = s_q[p0], kv = s_k[p0], vv = vr[p0];
        float q0 = fmaxf(0.0f, __low2float(qv)  * q_ir * q_nw[c0]);
        float q1 = fmaxf(0.0f, __high2float(qv) * q_ir * q_nw[c0 + 1]);
        float k0 = fmaxf(0.0f, __low2float(kv)  * k_ir * k_nw[c0]) * k_scale;
        float k1 = fmaxf(0.0f, __high2float(kv) * k_ir * k_nw[c0 + 1]) * k_scale;
        float v0 = __low2float(vv), v1 = __high2float(vv);

        float ce = s_cos[d_rel], co = s_cos[d_rel + 1];
        float se = s_sin[d_rel], so = s_sin[d_rel + 1];

        float qr0 = fmaf(q0, ce, q1 * se), qr1 = fmaf(q1, co, q0 * so);
        float kr0 = fmaf(k0, ce, k1 * se), kr1 = fmaf(k1, co, k0 * so);
        float vr0 = fmaf(v0, ce, v1 * se), vr1 = fmaf(v1, co, v0 * so);

        // RoPE is orthogonal, but compute exact pre/post squared norms deterministically
        s_pre_r2[ri] = k0*k0 + k1*k1;
        s_post_r2[ri] = kr0*kr0 + kr1*kr1;

        int64_t out_off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0;
        store_pair<OutT>(q_out + out_off, qr0, qr1);
        store_pair<OutT>(k_out + out_off, kr0, kr1);
        store_pair<OutT>(v_out + out_off, vr0, vr1);
    }
    __syncthreads();

    if (tid < H) {
        int h = tid;
        float pre = 0.0f, post = 0.0f;
        int g_base = h * n_grp4;
        for (int g = 0; g < n_grp4; ++g) {
            pre += s_pre_g4[g_base + g];
            post += s_post_g4[g_base + g];
        }
        int r_base = h * D_half2;
        for (int r = 0; r < D_half2; ++r) {
            pre += s_pre_r2[r_base + r];
            post += s_post_r2[r_base + r];
        }
        int64_t bhn = static_cast<int64_t>(b_idx * H + h) * N + n_idx;
        pre_ksq_out[bhn] = pre;
        post_ksq_out[bhn] = post;
        inflation_sq[bhn] = fmaxf(post, 1e-12f) / fmaxf(pre, 1e-12f);
    }
}

extern "C" __global__ void cam_prep_fwd_f32(
    const __mt_bfloat162* __restrict__ q2, const __mt_bfloat162* __restrict__ k2, const __mt_bfloat162* __restrict__ v2,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ P_T, const float* __restrict__ P_inv,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ v_out,
    float* __restrict__ inflation_sq, float* __restrict__ q_ir, float* __restrict__ k_ir,
    float* __restrict__ pre_ksq, float* __restrict__ post_ksq,
    int B, int N, int H, int D, float k_scale, float norm_eps
) {
    cam_prep_fwd_impl<float>(q2, k2, v2, q_nw, k_nw, P_T, P_inv, rope_cos, rope_sin,
                             q_out, k_out, v_out, inflation_sq, q_ir, k_ir, pre_ksq, post_ksq,
                             B, N, H, D, k_scale, norm_eps);
}

extern "C" __global__ void cam_prep_fwd_bf16(
    const __mt_bfloat162* __restrict__ q2, const __mt_bfloat162* __restrict__ k2, const __mt_bfloat162* __restrict__ v2,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ P_T, const float* __restrict__ P_inv,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    __mt_bfloat16* __restrict__ q_out, __mt_bfloat16* __restrict__ k_out, __mt_bfloat16* __restrict__ v_out,
    float* __restrict__ inflation_sq, float* __restrict__ q_ir, float* __restrict__ k_ir,
    float* __restrict__ pre_ksq, float* __restrict__ post_ksq,
    int B, int N, int H, int D, float k_scale, float norm_eps
) {
    cam_prep_fwd_impl<__mt_bfloat16>(q2, k2, v2, q_nw, k_nw, P_T, P_inv, rope_cos, rope_sin,
                                     q_out, k_out, v_out, inflation_sq, q_ir, k_ir, pre_ksq, post_ksq,
                                     B, N, H, D, k_scale, norm_eps);
}

// Note: Cam Prep forward uses SMEM atomicAdd for pre_ksq/post_ksq which can vary by 1 ULP
// in non-deterministic order; to guarantee 100% BITWISE DETERMINISM in all modes, let's also
// avoid SMEM atomicAdd by using deterministic per-head reduction in cam_prep_bwd!
// Even better: in cam_prep_fwd above, for H=20 we only have 20 heads — let's check determinism
// in unit test, and in cam_prep_bwd we use 100% deterministic split-chunk reduction (zero atomics!).
template <typename InT>
__device__ void cam_prep_bwd_impl(
    const __mt_bfloat162* __restrict__ q2,
    const __mt_bfloat162* __restrict__ k2,
    const float* __restrict__ q_nw,
    const float* __restrict__ k_nw,
    const float* __restrict__ P_T,
    const float* __restrict__ P_inv,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const float* __restrict__ q_ir_in,
    const float* __restrict__ k_ir_in,
    const float* __restrict__ pre_ksq_in,
    const float* __restrict__ post_ksq_in,
    const InT* __restrict__ dq_out,
    const InT* __restrict__ dk_out,
    const InT* __restrict__ dv_out,
    const float* __restrict__ d_inflation_sq,
    __mt_bfloat162* __restrict__ dq_raw2,
    __mt_bfloat162* __restrict__ dk_raw2,
    __mt_bfloat162* __restrict__ dv_raw2,
    float* __restrict__ dq_nw_part,
    float* __restrict__ dk_nw_part,
    int B, int N, int H, int D, float k_scale, int n_chunks
) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    const int BN = B * N;
    const int C = H * D;
    const int C2 = C >> 1;
    const int D_half = D >> 1;
    const int n_grp4 = D_half >> 2;
    const int D_half2 = D_half >> 1;
    const int bn_start = (static_cast<int64_t>(chunk) * BN) / n_chunks;
    const int bn_end   = (static_cast<int64_t>(chunk + 1) * BN) / n_chunks;

    extern __shared__ float smem_b[];
    float* s_dqnw = smem_b;            // C
    float* s_dknw = s_dqnw + C;        // C
    float* s_gq   = s_dknw + C;        // C (temp post-relu grad for RMSNorm pass 2)
    float* s_gk   = s_gq + C;          // C
    __mt_bfloat162* s_q = reinterpret_cast<__mt_bfloat162*>(s_gk + C);  // C2
    __mt_bfloat162* s_k = s_q + C2;                                      // C2
    float* s_PT   = reinterpret_cast<float*>(s_k + C2);                  // 16
    float* s_Pinv = s_PT + 16;         // 16
    float* s_cos  = s_Pinv + 16;       // 64
    float* s_sin  = s_cos + 64;        // 64
    float* s_dpost= s_sin + 64;        // 32
    float* s_dpre = s_dpost + 32;      // 32
    float* s_red  = s_dpre + 32;       // 16

    for (int i = tid; i < C; i += 256) { s_dqnw[i] = 0.0f; s_dknw[i] = 0.0f; }
    __syncthreads();

    const bool qkv_packed = (dv_raw2 == nullptr);
    const __mt_bfloat162 zero_b2 = __floats2bfloat162_rn(0.0f, 0.0f);

    for (int bn = bn_start; bn < bn_end; ++bn) {
        const int b_idx = bn / N;
        const int n_idx = bn - b_idx * N;

        if (tid < 16) {
            s_PT[tid] = P_T[static_cast<int64_t>(bn) * 16 + tid];
            s_Pinv[tid] = P_inv[static_cast<int64_t>(bn) * 16 + tid];
        }
        if (tid < D_half) {
            s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D_half + tid];
            s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D_half + tid];
        }
        if (tid < H) {
            int64_t bhn = static_cast<int64_t>(b_idx * H + tid) * N + n_idx;
            float pre = pre_ksq_in[bhn], post = post_ksq_in[bhn];
            float dinf = d_inflation_sq[bhn];
            float pre_c = fmaxf(pre, 1e-12f);
            float post_c = fmaxf(post, 1e-12f);
            s_dpost[tid] = (post >= 1e-12f) ? (dinf / pre_c) : 0.0f;
            s_dpre[tid]  = (pre  >= 1e-12f) ? (-dinf * post_c / (pre_c * pre_c)) : 0.0f;
        }

        const __mt_bfloat162* qr = qkv_packed ? (q2 + static_cast<int64_t>(bn) * 3 * C2)
                                              : (q2 + static_cast<int64_t>(bn) * C2);
        const __mt_bfloat162* kr = (k2 != nullptr) ? (k2 + static_cast<int64_t>(bn) * C2)
                                                   : (qr + C2);
        __mt_bfloat162* dqr_w = qkv_packed ? (dq_raw2 + static_cast<int64_t>(bn) * 3 * C2)
                                           : (dq_raw2 + static_cast<int64_t>(bn) * C2);
        __mt_bfloat162* dkr_w = (dk_raw2 != nullptr) ? (dk_raw2 + static_cast<int64_t>(bn) * C2)
                                                     : (dqr_w + C2);
        __mt_bfloat162* dvr_w = qkv_packed ? (dqr_w + 2 * C2)
                                           : (dv_raw2 + static_cast<int64_t>(bn) * C2);

        for (int p = tid; p < C2; p += 256) {
            s_q[p] = qr[p];
            s_k[p] = kr[p];
        }
        __syncthreads();

        const float q_ir = q_ir_in[bn];
        const float k_ir = k_ir_in[bn];

        // First half [0..D_half): 4x4 transpose + inflation_sq contribution
        const int total_g4 = H * n_grp4;
        for (int gi = tid; gi < total_g4; gi += 256) {
            int h = gi / n_grp4;
            int g = gi - h * n_grp4;
            int d0 = g * 4;
            int c0 = h * D + d0;
            int p0 = c0 >> 1;
            int64_t bhn = static_cast<int64_t>(b_idx * H + h) * N + n_idx;
            int64_t off = bhn * D + d0;

            float d_post = s_dpost[h];
            float d_pre  = s_dpre[h];

            float2 dq01 = load_pair<InT>(dq_out + off),     dq23 = load_pair<InT>(dq_out + off + 2);
            float2 dk01 = load_pair<InT>(dk_out + off),     dk23 = load_pair<InT>(dk_out + off + 2);
            float2 dv01 = load_pair<InT>(dv_out + off),     dv23 = load_pair<InT>(dv_out + off + 2);

            __mt_bfloat162 ka = s_k[p0], kb = s_k[p0 + 1];
            float kx0 = fmaxf(0.0f, __low2float(ka)  * k_ir * k_nw[c0 + 0]) * k_scale;
            float kx1 = fmaxf(0.0f, __high2float(ka) * k_ir * k_nw[c0 + 1]) * k_scale;
            float kx2 = fmaxf(0.0f, __low2float(kb)  * k_ir * k_nw[c0 + 2]) * k_scale;
            float kx3 = fmaxf(0.0f, __high2float(kb) * k_ir * k_nw[c0 + 3]) * k_scale;

            float ky0 = s_Pinv[0]*kx0  + s_Pinv[1]*kx1  + s_Pinv[2]*kx2  + s_Pinv[3]*kx3;
            float ky1 = s_Pinv[4]*kx0  + s_Pinv[5]*kx1  + s_Pinv[6]*kx2  + s_Pinv[7]*kx3;
            float ky2 = s_Pinv[8]*kx0  + s_Pinv[9]*kx1  + s_Pinv[10]*kx2 + s_Pinv[11]*kx3;
            float ky3 = s_Pinv[12]*kx0 + s_Pinv[13]*kx1 + s_Pinv[14]*kx2 + s_Pinv[15]*kx3;

            float gky0 = dk01.x + 2.0f * d_post * ky0;
            float gky1 = dk01.y + 2.0f * d_post * ky1;
            float gky2 = dk23.x + 2.0f * d_post * ky2;
            float gky3 = dk23.y + 2.0f * d_post * ky3;

            s_gq[c0 + 0] = s_PT[0]*dq01.x + s_PT[4]*dq01.y + s_PT[8]*dq23.x  + s_PT[12]*dq23.y;
            s_gq[c0 + 1] = s_PT[1]*dq01.x + s_PT[5]*dq01.y + s_PT[9]*dq23.x  + s_PT[13]*dq23.y;
            s_gq[c0 + 2] = s_PT[2]*dq01.x + s_PT[6]*dq01.y + s_PT[10]*dq23.x + s_PT[14]*dq23.y;
            s_gq[c0 + 3] = s_PT[3]*dq01.x + s_PT[7]*dq01.y + s_PT[11]*dq23.x + s_PT[15]*dq23.y;

            s_gk[c0 + 0] = (s_Pinv[0]*gky0 + s_Pinv[4]*gky1 + s_Pinv[8]*gky2  + s_Pinv[12]*gky3 + 2.0f * d_pre * kx0) * k_scale;
            s_gk[c0 + 1] = (s_Pinv[1]*gky0 + s_Pinv[5]*gky1 + s_Pinv[9]*gky2  + s_Pinv[13]*gky3 + 2.0f * d_pre * kx1) * k_scale;
            s_gk[c0 + 2] = (s_Pinv[2]*gky0 + s_Pinv[6]*gky1 + s_Pinv[10]*gky2 + s_Pinv[14]*gky3 + 2.0f * d_pre * kx2) * k_scale;
            s_gk[c0 + 3] = (s_Pinv[3]*gky0 + s_Pinv[7]*gky1 + s_Pinv[11]*gky2 + s_Pinv[15]*gky3 + 2.0f * d_pre * kx3) * k_scale;

            float gvx0 = s_Pinv[0]*dv01.x + s_Pinv[4]*dv01.y + s_Pinv[8]*dv23.x  + s_Pinv[12]*dv23.y;
            float gvx1 = s_Pinv[1]*dv01.x + s_Pinv[5]*dv01.y + s_Pinv[9]*dv23.x  + s_Pinv[13]*dv23.y;
            float gvx2 = s_Pinv[2]*dv01.x + s_Pinv[6]*dv01.y + s_Pinv[10]*dv23.x + s_Pinv[14]*dv23.y;
            float gvx3 = s_Pinv[3]*dv01.x + s_Pinv[7]*dv01.y + s_Pinv[11]*dv23.x + s_Pinv[15]*dv23.y;
            dvr_w[p0]     = __floats2bfloat162_rn(gvx0, gvx1);
            dvr_w[p0 + 1] = __floats2bfloat162_rn(gvx2, gvx3);
        }

        // Second half [D_half..D): un-RoPE + inflation_sq
        const int total_r2 = H * D_half2;
        for (int ri = tid; ri < total_r2; ri += 256) {
            int h = ri / D_half2;
            int d2 = ri - h * D_half2;
            int d_rel = d2 * 2;
            int d0 = D_half + d_rel;
            int c0 = h * D + d0;
            int p0 = c0 >> 1;
            int64_t bhn = static_cast<int64_t>(b_idx * H + h) * N + n_idx;
            int64_t off = bhn * D + d0;

            float d_post = s_dpost[h];
            float d_pre  = s_dpre[h];

            float2 dq = load_pair<InT>(dq_out + off);
            float2 dk = load_pair<InT>(dk_out + off);
            float2 dv = load_pair<InT>(dv_out + off);

            float ce = s_cos[d_rel], co = s_cos[d_rel + 1];
            float se = s_sin[d_rel], so = s_sin[d_rel + 1];

            __mt_bfloat162 kv = s_k[p0];
            float kx0 = fmaxf(0.0f, __low2float(kv)  * k_ir * k_nw[c0])     * k_scale;
            float kx1 = fmaxf(0.0f, __high2float(kv) * k_ir * k_nw[c0 + 1]) * k_scale;
            float kr0 = fmaf(kx0, ce, kx1 * se), kr1 = fmaf(kx1, co, kx0 * so);

            float gky0 = dk.x + 2.0f * d_post * kr0;
            float gky1 = dk.y + 2.0f * d_post * kr1;

            s_gq[c0]     = fmaf(dq.x, ce, dq.y * so);
            s_gq[c0 + 1] = fmaf(dq.y, co, dq.x * se);
            s_gk[c0]     = (fmaf(gky0, ce, gky1 * so) + 2.0f * d_pre * kx0) * k_scale;
            s_gk[c0 + 1] = (fmaf(gky1, co, gky0 * se) + 2.0f * d_pre * kx1) * k_scale;

            dvr_w[p0] = __floats2bfloat162_rn(fmaf(dv.x, ce, dv.y * so), fmaf(dv.y, co, dv.x * se));
        }
        __syncthreads();

        // ReLU + full-channel RMSNorm backward (using SMEM s_q / s_k!)
        float q_dot = 0.0f, k_dot = 0.0f;
        for (int p = tid; p < C2; p += 256) {
            int c0 = p * 2;
            __mt_bfloat162 qv = s_q[p], kv = s_k[p];
            float q0 = __low2float(qv), q1 = __high2float(qv);
            float k0 = __low2float(kv), k1 = __high2float(kv);
            float qw0 = q_nw[c0], qw1 = q_nw[c0 + 1];
            float kw0 = k_nw[c0], kw1 = k_nw[c0 + 1];

            float gq0 = (q0 * qw0 > 0.0f) ? s_gq[c0]     : 0.0f;
            float gq1 = (q1 * qw1 > 0.0f) ? s_gq[c0 + 1] : 0.0f;
            float gk0 = (k0 * kw0 > 0.0f) ? s_gk[c0]     : 0.0f;
            float gk1 = (k1 * kw1 > 0.0f) ? s_gk[c0 + 1] : 0.0f;

            s_gq[c0] = gq0 * qw0; s_gq[c0 + 1] = gq1 * qw1;
            s_gk[c0] = gk0 * kw0; s_gk[c0 + 1] = gk1 * kw1;

            s_dqnw[c0]     += gq0 * q0 * q_ir;
            s_dqnw[c0 + 1] += gq1 * q1 * q_ir;
            s_dknw[c0]     += gk0 * k0 * k_ir;
            s_dknw[c0 + 1] += gk1 * k1 * k_ir;

            q_dot = fmaf(s_gq[c0], q0, fmaf(s_gq[c0 + 1], q1, q_dot));
            k_dot = fmaf(s_gk[c0], k0, fmaf(s_gk[c0 + 1], k1, k_dot));
        }
        q_dot = warp_sum(q_dot);
        k_dot = warp_sum(k_dot);
        if ((tid & 31) == 0) { s_red[tid >> 5] = q_dot; s_red[8 + (tid >> 5)] = k_dot; }
        __syncthreads();
        if (tid == 0) {
            float sq = 0, sk = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) { sq += s_red[w]; sk += s_red[8 + w]; }
            s_red[0] = sq * q_ir * q_ir / static_cast<float>(C);
            s_red[8] = sk * k_ir * k_ir / static_cast<float>(C);
        }
        __syncthreads();
        float q_coeff = s_red[0], k_coeff = s_red[8];

        for (int p = tid; p < C2; p += 256) {
            int c0 = p * 2;
            __mt_bfloat162 qv = s_q[p], kv = s_k[p];
            float q0 = __low2float(qv), q1 = __high2float(qv);
            float k0 = __low2float(kv), k1 = __high2float(kv);
            dqr_w[p] = __floats2bfloat162_rn(
                q_ir * (s_gq[c0]     - q0 * q_coeff),
                q_ir * (s_gq[c0 + 1] - q1 * q_coeff)
            );
            dkr_w[p] = __floats2bfloat162_rn(
                k_ir * (s_gk[c0]     - k0 * k_coeff),
                k_ir * (s_gk[c0 + 1] - k1 * k_coeff)
            );
        }
        __syncthreads();
    }

    int64_t part_base = static_cast<int64_t>(chunk) * C;
    for (int i = tid; i < C; i += 256) {
        dq_nw_part[part_base + i] = s_dqnw[i];
        dk_nw_part[part_base + i] = s_dknw[i];
    }
}

extern "C" __global__ void cam_prep_bwd_f32(
    const __mt_bfloat162* __restrict__ q2, const __mt_bfloat162* __restrict__ k2,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ P_T, const float* __restrict__ P_inv,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    const float* __restrict__ q_ir, const float* __restrict__ k_ir,
    const float* __restrict__ pre_ksq, const float* __restrict__ post_ksq,
    const float* __restrict__ dq_out, const float* __restrict__ dk_out, const float* __restrict__ dv_out,
    const float* __restrict__ dinf,
    __mt_bfloat162* __restrict__ dq_raw2, __mt_bfloat162* __restrict__ dk_raw2, __mt_bfloat162* __restrict__ dv_raw2,
    float* __restrict__ dq_nw_part, float* __restrict__ dk_nw_part,
    int B, int N, int H, int D, float k_scale, int n_chunks
) {
    cam_prep_bwd_impl<float>(q2, k2, q_nw, k_nw, P_T, P_inv, rope_cos, rope_sin, q_ir, k_ir,
                             pre_ksq, post_ksq, dq_out, dk_out, dv_out, dinf,
                             dq_raw2, dk_raw2, dv_raw2, dq_nw_part, dk_nw_part,
                             B, N, H, D, k_scale, n_chunks);
}

extern "C" __global__ void cam_prep_bwd_bf16(
    const __mt_bfloat162* __restrict__ q2, const __mt_bfloat162* __restrict__ k2,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ P_T, const float* __restrict__ P_inv,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    const float* __restrict__ q_ir, const float* __restrict__ k_ir,
    const float* __restrict__ pre_ksq, const float* __restrict__ post_ksq,
    const __mt_bfloat16* __restrict__ dq_out, const __mt_bfloat16* __restrict__ dk_out, const __mt_bfloat16* __restrict__ dv_out,
    const float* __restrict__ dinf,
    __mt_bfloat162* __restrict__ dq_raw2, __mt_bfloat162* __restrict__ dk_raw2, __mt_bfloat162* __restrict__ dv_raw2,
    float* __restrict__ dq_nw_part, float* __restrict__ dk_nw_part,
    int B, int N, int H, int D, float k_scale, int n_chunks
) {
    cam_prep_bwd_impl<__mt_bfloat16>(q2, k2, q_nw, k_nw, P_T, P_inv, rope_cos, rope_sin, q_ir, k_ir,
                                     pre_ksq, post_ksq, dq_out, dk_out, dv_out, dinf,
                                     dq_raw2, dk_raw2, dv_raw2, dq_nw_part, dk_nw_part,
                                     B, N, H, D, k_scale, n_chunks);
}

// =========================================================================
// 7. Fused Cam Output Apply O: x(BH, N, D) -> out(B, N, C) BF16
//    Applies 4x4 P on [0..D/2) and inverse FP32 RoPE on [D/2..D) (zero FP64!)
//    Grid: (B * N), Block: 256
// =========================================================================
template <typename ValT>
__device__ void cam_out_apply_o_fwd_impl(
    const ValT* __restrict__ x,          // (BH, N, D)
    const float* __restrict__ P,         // (B, N, 16)
    const float* __restrict__ rope_cos,  // (N, D/2)
    const float* __restrict__ rope_sin,  // (N, D/2)
    __mt_bfloat162* __restrict__ out2,   // (B, N, C/2)
    int B, int N, int H, int D
) {
    const int bn = blockIdx.x;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int tid = threadIdx.x;
    const int C2 = (H * D) >> 1;
    const int D_half = D >> 1;
    const int n_grp4 = D_half >> 2;
    const int D_half2 = D_half >> 1;

    __shared__ float s_P[16], s_cos[64], s_sin[64];
    if (tid < 16) s_P[tid] = P[static_cast<int64_t>(bn) * 16 + tid];
    if (tid < D_half) {
        s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D_half + tid];
        s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D_half + tid];
    }
    __syncthreads();

    __mt_bfloat162* o_row = out2 + static_cast<int64_t>(bn) * C2;

    for (int gi = tid; gi < H * n_grp4; gi += 256) {
        int h = gi / n_grp4;
        int d0 = (gi - h * n_grp4) * 4;
        int64_t off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0;
        float2 x01 = load_pair<ValT>(x + off);
        float2 x23 = load_pair<ValT>(x + off + 2);
        // Match baseline rounding: out_bf16 before apply_fn_o
        float a0 = __bfloat162float(__float2bfloat16(x01.x));
        float a1 = __bfloat162float(__float2bfloat16(x01.y));
        float a2 = __bfloat162float(__float2bfloat16(x23.x));
        float a3 = __bfloat162float(__float2bfloat16(x23.y));
        float y0 = s_P[0]*a0  + s_P[1]*a1  + s_P[2]*a2  + s_P[3]*a3;
        float y1 = s_P[4]*a0  + s_P[5]*a1  + s_P[6]*a2  + s_P[7]*a3;
        float y2 = s_P[8]*a0  + s_P[9]*a1  + s_P[10]*a2 + s_P[11]*a3;
        float y3 = s_P[12]*a0 + s_P[13]*a1 + s_P[14]*a2 + s_P[15]*a3;
        int p0 = (h * D + d0) >> 1;
        o_row[p0]     = __floats2bfloat162_rn(y0, y1);
        o_row[p0 + 1] = __floats2bfloat162_rn(y2, y3);
    }
    for (int ri = tid; ri < H * D_half2; ri += 256) {
        int h = ri / D_half2;
        int d_rel = (ri - h * D_half2) * 2;
        int d0 = D_half + d_rel;
        int64_t off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0;
        float2 x01 = load_pair<ValT>(x + off);
        float a0 = __bfloat162float(__float2bfloat16(x01.x));
        float a1 = __bfloat162float(__float2bfloat16(x01.y));
        // Inverse RoPE: y0 = a0*cos + a1*sin_half = a0*cos - a1*rope_sin[even]
        //               y1 = a1*cos - a0*sin_half = a1*cos - a0*rope_sin[odd]
        float ce = s_cos[d_rel], co = s_cos[d_rel + 1];
        float se = s_sin[d_rel], so = s_sin[d_rel + 1];
        float y0 = fmaf(a0, ce, -a1 * se);
        float y1 = fmaf(a1, co, -a0 * so);
        o_row[(h * D + d0) >> 1] = __floats2bfloat162_rn(y0, y1);
    }
}

template <typename ValT>
__device__ void cam_out_apply_o_bwd_impl(
    const __mt_bfloat162* __restrict__ dout2, // (B, N, C/2)
    const float* __restrict__ P,              // (B, N, 16)
    const float* __restrict__ rope_cos,       // (N, D/2)
    const float* __restrict__ rope_sin,       // (N, D/2)
    ValT* __restrict__ dx,                    // (BH, N, D)
    int B, int N, int H, int D
) {
    const int bn = blockIdx.x;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int tid = threadIdx.x;
    const int C2 = (H * D) >> 1;
    const int D_half = D >> 1;
    const int n_grp4 = D_half >> 2;
    const int D_half2 = D_half >> 1;

    __shared__ float s_P[16], s_cos[64], s_sin[64];
    if (tid < 16) s_P[tid] = P[static_cast<int64_t>(bn) * 16 + tid];
    if (tid < D_half) {
        s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D_half + tid];
        s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D_half + tid];
    }
    __syncthreads();

    const __mt_bfloat162* d_row = dout2 + static_cast<int64_t>(bn) * C2;

    for (int gi = tid; gi < H * n_grp4; gi += 256) {
        int h = gi / n_grp4;
        int d0 = (gi - h * n_grp4) * 4;
        int p0 = (h * D + d0) >> 1;
        __mt_bfloat162 ga = d_row[p0], gb = d_row[p0 + 1];
        float g0 = __low2float(ga), g1 = __high2float(ga), g2 = __low2float(gb), g3 = __high2float(gb);
        // Transpose P^T @ g
        float x0 = s_P[0]*g0 + s_P[4]*g1 + s_P[8]*g2  + s_P[12]*g3;
        float x1 = s_P[1]*g0 + s_P[5]*g1 + s_P[9]*g2  + s_P[13]*g3;
        float x2 = s_P[2]*g0 + s_P[6]*g1 + s_P[10]*g2 + s_P[14]*g3;
        float x3 = s_P[3]*g0 + s_P[7]*g1 + s_P[11]*g2 + s_P[15]*g3;
        int64_t off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0;
        store_pair<ValT>(dx + off,     x0, x1);
        store_pair<ValT>(dx + off + 2, x2, x3);
    }
    for (int ri = tid; ri < H * D_half2; ri += 256) {
        int h = ri / D_half2;
        int d_rel = (ri - h * D_half2) * 2;
        int d0 = D_half + d_rel;
        __mt_bfloat162 ga = d_row[(h * D + d0) >> 1];
        float g0 = __low2float(ga), g1 = __high2float(ga);
        // Adjoint of inverse RoPE is forward RoPE: x0 = g0*cos + g1*rope_sin[even], x1 = g1*cos + g0*rope_sin[odd]
        float ce = s_cos[d_rel], co = s_cos[d_rel + 1];
        float se = s_sin[d_rel], so = s_sin[d_rel + 1];
        float x0 = fmaf(g0, ce, g1 * se);
        float x1 = fmaf(g1, co, g0 * so);
        int64_t off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D + d0;
        store_pair<ValT>(dx + off, x0, x1);
    }
}

extern "C" __global__ void cam_out_apply_o_fwd_f32(
    const float* __restrict__ x, const float* __restrict__ P,
    const float* __restrict__ rc, const float* __restrict__ rs, __mt_bfloat162* __restrict__ out2,
    int B, int N, int H, int D
) { cam_out_apply_o_fwd_impl<float>(x, P, rc, rs, out2, B, N, H, D); }

extern "C" __global__ void cam_out_apply_o_fwd_bf16(
    const __mt_bfloat16* __restrict__ x, const float* __restrict__ P,
    const float* __restrict__ rc, const float* __restrict__ rs, __mt_bfloat162* __restrict__ out2,
    int B, int N, int H, int D
) { cam_out_apply_o_fwd_impl<__mt_bfloat16>(x, P, rc, rs, out2, B, N, H, D); }

extern "C" __global__ void cam_out_apply_o_bwd_f32(
    const __mt_bfloat162* __restrict__ dout2, const float* __restrict__ P,
    const float* __restrict__ rc, const float* __restrict__ rs, float* __restrict__ dx,
    int B, int N, int H, int D
) { cam_out_apply_o_bwd_impl<float>(dout2, P, rc, rs, dx, B, N, H, D); }

extern "C" __global__ void cam_out_apply_o_bwd_bf16(
    const __mt_bfloat162* __restrict__ dout2, const float* __restrict__ P,
    const float* __restrict__ rc, const float* __restrict__ rs, __mt_bfloat16* __restrict__ dx,
    int B, int N, int H, int D
) { cam_out_apply_o_bwd_impl<__mt_bfloat16>(dout2, P, rc, rs, dx, B, N, H, D); }

// =========================================================================
// 8. Memory-Lean & Phase A/C Fused GDN Kernels (BF16 I/O, FP32 accum, 100% deterministic)
// =========================================================================

// 8a. out_div backward from ALREADY-COMPUTED out(B,N,C) BF16 + fused dQ_pos!
//     Eliminates saving num (98.6 MB/layer) AND eliminates standalone dQ_pos kernel!
//     Grid: dim3(S, T, BH), Block: 64 (tid < D/2 = 56)
extern "C" __global__ void out_div_bwd_from_out_dqpos_bf16(
    const __mt_bfloat162* __restrict__ dout2,  // (B, N, C/2)
    const __mt_bfloat162* __restrict__ out2,   // (B, N, C/2)
    const float* __restrict__ den,             // (BH, N)
    const float* __restrict__ Sz_tot,          // (BH*T, D)
    __mt_bfloat162* __restrict__ dnum2,        // (BH, N, D/2)
    float* __restrict__ dden,                  // (BH, N)
    __mt_bfloat162* __restrict__ dq_pos2,      // (BH, N, D/2)
    int H, int T, int S, int D, float eps
) {
    const int s_idx = blockIdx.x;
    const int t_idx = blockIdx.y;
    const int bh = blockIdx.z;
    const int tid = threadIdx.x;  // 0..63
    const int D2 = D >> 1;
    const int C2 = H * D2;
    const int N = T * S;
    const int b_idx = bh / H;
    const int h = bh - b_idx * H;
    const int n_idx = t_idx * S + s_idx;
    const int64_t bn = static_cast<int64_t>(b_idx) * N + n_idx;
    const int64_t bhn = static_cast<int64_t>(bh) * N + n_idx;
    const float inv_d = 1.0f / (den[bhn] + eps);

    float dot = 0.0f;
    if (tid < D2) {
        int64_t off_bnc = bn * C2 + h * D2 + tid;
        __mt_bfloat162 dv = dout2[off_bnc];
        __mt_bfloat162 ov = out2[off_bnc];
        float g0 = __low2float(dv), g1 = __high2float(dv);
        float o0 = __low2float(ov), o1 = __high2float(ov);
        dnum2[bhn * D2 + tid] = __floats2bfloat162_rn(g0 * inv_d, g1 * inv_d);
        dot = fmaf(g0, o0, g1 * o1);
    }
    dot = warp_sum(dot);
    __shared__ float s_w[2];
    if ((tid & 31) == 0) s_w[tid >> 5] = dot;
    __syncthreads();
    const float dden_val = -(s_w[0] + s_w[1]) * inv_d;
    if (tid == 0) {
        dden[bhn] = dden_val;
    }
    if (tid < D2) {
        int64_t m = static_cast<int64_t>(bh) * T + t_idx;
        float sz0 = Sz_tot[m * D + tid * 2 + 0];
        float sz1 = Sz_tot[m * D + tid * 2 + 1];
        dq_pos2[bhn * D2 + tid] = __floats2bfloat162_rn(dden_val * sz0, dden_val * sz1);
    }
}

// 8b. Main Phase A Scale (+ optional U_z = sum_s(kp_b) in FP32)
//     Grid: M = BH*T, Block: 224 (tx=0..55 -> D2=56 bf162, ty=0..3 -> S)
extern "C" __global__ void gdn_main_phase_a_scale_bf16(
    const __mt_bfloat162* __restrict__ kr2,    // (M, S, D/2)
    const __mt_bfloat162* __restrict__ kp2,    // (M, S, D/2)
    const __mt_bfloat162* __restrict__ v2,     // (M, S, D/2)
    const float* __restrict__ beta,            // (M, S)
    __mt_bfloat162* __restrict__ kr_b2,        // (M, S, D/2)
    __mt_bfloat162* __restrict__ kp_b2,        // (M, S, D/2)
    __mt_bfloat162* __restrict__ v_b2,         // (M, S, D/2)
    float* __restrict__ U_z,                   // (M, D) or nullptr
    int S, int D
) {
    const int m = blockIdx.x;
    const int D2 = D >> 1;  // 56
    const int tid = threadIdx.x;
    const int tx = tid % D2;
    const int ty = tid / D2;  // 0..3
    const int64_t base_ms = static_cast<int64_t>(m) * S;
    const int64_t base_m = base_ms * D2;

    float uz0 = 0.0f, uz1 = 0.0f;
    for (int s = ty; s < S; s += 4) {
        float bf = __bfloat162float(__float2bfloat16_rn(beta[base_ms + s]));
        int64_t off = base_m + static_cast<int64_t>(s) * D2 + tx;
        __mt_bfloat162 r = kr2[off];
        __mt_bfloat162 p = kp2[off];
        __mt_bfloat162 v = v2[off];
        __mt_bfloat162 pb = __floats2bfloat162_rn(__low2float(p) * bf, __high2float(p) * bf);
        kr_b2[off] = __floats2bfloat162_rn(__low2float(r) * bf, __high2float(r) * bf);
        kp_b2[off] = pb;
        v_b2[off]  = __floats2bfloat162_rn(__low2float(v) * bf, __high2float(v) * bf);
        if (U_z != nullptr) {
            uz0 += __low2float(pb);
            uz1 += __high2float(pb);
        }
    }
    if (U_z != nullptr) {
        __shared__ float s_uz[4][112];
        s_uz[ty][tx * 2 + 0] = uz0;
        s_uz[ty][tx * 2 + 1] = uz1;
        __syncthreads();
        if (ty == 0) {
            float a0 = s_uz[0][tx*2+0] + s_uz[1][tx*2+0] + s_uz[2][tx*2+0] + s_uz[3][tx*2+0];
            float a1 = s_uz[0][tx*2+1] + s_uz[1][tx*2+1] + s_uz[2][tx*2+1] + s_uz[3][tx*2+1];
            U_z[static_cast<int64_t>(m) * D + tx * 2 + 0] = a0;
            U_z[static_cast<int64_t>(m) * D + tx * 2 + 1] = a1;
        }
    }
}

// 8c. Cam Phase A Scale: k_b = kk * beta, v_b = vv * beta (supports beta 4D (M,S) or 3D (M,))
//     Grid: M = BH*T, Block: 224
extern "C" __global__ void gdn_cam_phase_a_scale_bf16(
    const __mt_bfloat162* __restrict__ k2,
    const __mt_bfloat162* __restrict__ v2,
    const float* __restrict__ beta,
    __mt_bfloat162* __restrict__ k_b2,
    __mt_bfloat162* __restrict__ v_b2,
    int S, int D, int beta_is_4d
) {
    const int m = blockIdx.x;
    const int D2 = D >> 1;
    const int tid = threadIdx.x;
    const int tx = tid % D2;
    const int ty = tid / D2;
    const int64_t base_ms = static_cast<int64_t>(m) * S;
    const int64_t base_m = base_ms * D2;
    const float bf_3d = beta_is_4d ? 0.0f : __bfloat162float(__float2bfloat16_rn(beta[m]));

    for (int s = ty; s < S; s += 4) {
        float bf = beta_is_4d ? __bfloat162float(__float2bfloat16_rn(beta[base_ms + s])) : bf_3d;
        int64_t off = base_m + static_cast<int64_t>(s) * D2 + tx;
        __mt_bfloat162 kv = k2[off], vv = v2[off];
        k_b2[off] = __floats2bfloat162_rn(__low2float(kv) * bf, __high2float(kv) * bf);
        v_b2[off] = __floats2bfloat162_rn(__low2float(vv) * bf, __high2float(vv) * bf);
    }
}

// 8d. Make W = decay * (I - P) for 1 or 2 matrices in 1 launch!
//     Grid: M, Block: 256
extern "C" __global__ void gdn_make_W_fwd_f32(
    const float* __restrict__ P1,
    const float* __restrict__ P2,  // optional (nullptr for cam)
    const float* __restrict__ decay,
    float* __restrict__ W1,
    float* __restrict__ W2,        // optional
    int D
) {
    const int m = blockIdx.x;
    const float dec = decay[m];
    const int dd = D * D;
    const int64_t base = static_cast<int64_t>(m) * dd;
    for (int idx = threadIdx.x; idx < dd; idx += 256) {
        int r = idx / D;
        int c = idx - r * D;
        float eye = (r == c) ? 1.0f : 0.0f;
        W1[base + idx] = dec * (eye - P1[base + idx]);
        if (P2 != nullptr) {
            W2[base + idx] = dec * (eye - P2[base + idx]);
        }
    }
}

// 8e. S_tot_bf16 = bf16(S_fwd + S_rev) + optional Sz_tot = Sz_fwd + Sz_rev
//     Grid: M, Block: 256
extern "C" __global__ void gdn_add_cast_S_bf16(
    const float* __restrict__ S_fwd,
    const float* __restrict__ S_rev,
    const float* __restrict__ Sz_fwd,  // optional
    const float* __restrict__ Sz_rev,  // optional
    __mt_bfloat162* __restrict__ S_tot_bf2,
    float* __restrict__ Sz_tot,        // optional
    int D
) {
    const int m = blockIdx.x;
    const int dd2 = (D * D) >> 1;
    const int64_t base2 = static_cast<int64_t>(m) * dd2;
    const float2* sf2 = reinterpret_cast<const float2*>(S_fwd) + base2;
    const float2* sr2 = reinterpret_cast<const float2*>(S_rev) + base2;
    __mt_bfloat162* dst2 = S_tot_bf2 + base2;
    for (int idx = threadIdx.x; idx < dd2; idx += 256) {
        float2 a = sf2[idx], b = sr2[idx];
        dst2[idx] = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
    }
    if (Sz_tot != nullptr && threadIdx.x < D) {
        int64_t zoff = static_cast<int64_t>(m) * D + threadIdx.x;
        Sz_tot[zoff] = Sz_fwd[zoff] + Sz_rev[zoff];
    }
}

// 8f. den(M, S) = sum_d(bf16_to_f32(qp[m,s,d]) * Sz_tot[m,d])  [Eliminates 197MB qp.float() cast + bmm!]
//     Grid: dim3(M, (S+7)/8), Block: 256 (8 warps, 1 warp per s)
extern "C" __global__ void gdn_qp_sz_dot_fwd_bf16(
    const __mt_bfloat162* __restrict__ qp2,  // (M, S, D/2)
    const float* __restrict__ Sz_tot,        // (M, D)
    float* __restrict__ den,                 // (M, S)
    int S, int D
) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int s = blockIdx.y * 8 + warp;
    const int D2 = D >> 1;

    __shared__ float s_sz[112];
    if (tid < D) s_sz[tid] = Sz_tot[static_cast<int64_t>(m) * D + tid];
    __syncthreads();

    if (s >= S) return;
    const __mt_bfloat162* row = qp2 + (static_cast<int64_t>(m) * S + s) * D2;
    float acc = 0.0f;
    for (int p = lane; p < D2; p += 32) {
        __mt_bfloat162 qv = row[p];
        acc = fmaf(__low2float(qv), s_sz[p * 2], fmaf(__high2float(qv), s_sz[p * 2 + 1], acc));
    }
    acc = warp_sum(acc);
    if (lane == 0) den[static_cast<int64_t>(m) * S + s] = acc;
}

// 8g. dSz_tot(M, D) = sum_s(dden[m,s] * bf16_to_f32(qp[m,s,d]))  [Eliminates 197MB qp.float() cast + bmm!]
//     Grid: M, Block: 224 (tx=0..55 -> D2, ty=0..3 -> S)
extern "C" __global__ void gdn_dden_qp_bwd_bf16(
    const float* __restrict__ dden,          // (M, S)
    const __mt_bfloat162* __restrict__ qp2,  // (M, S, D/2)
    float* __restrict__ dSz_tot,             // (M, D)
    int S, int D
) {
    const int m = blockIdx.x;
    const int D2 = D >> 1;
    const int tid = threadIdx.x;
    const int tx = tid % D2;
    const int ty = tid / D2;
    const int64_t base_ms = static_cast<int64_t>(m) * S;
    const int64_t base_m = base_ms * D2;

    float a0 = 0.0f, a1 = 0.0f;
    for (int s = ty; s < S; s += 4) {
        float dd = dden[base_ms + s];
        __mt_bfloat162 qv = qp2[base_m + static_cast<int64_t>(s) * D2 + tx];
        a0 = fmaf(dd, __low2float(qv), a0);
        a1 = fmaf(dd, __high2float(qv), a1);
    }
    __shared__ float s_red[4][112];
    s_red[ty][tx * 2 + 0] = a0;
    s_red[ty][tx * 2 + 1] = a1;
    __syncthreads();
    if (ty == 0) {
        dSz_tot[static_cast<int64_t>(m) * D + tx * 2 + 0] = s_red[0][tx*2+0] + s_red[1][tx*2+0] + s_red[2][tx*2+0] + s_red[3][tx*2+0];
        dSz_tot[static_cast<int64_t>(m) * D + tx * 2 + 1] = s_red[0][tx*2+1] + s_red[1][tx*2+1] + s_red[2][tx*2+1] + s_red[3][tx*2+1];
    }
}

// 8h. Combined dU + d_decay + dP_bf16 + dU_bf16 (+ optional dP_z_bf16 + dU_z_f32) in 1 pass!
//     Grid: M = BH*T, Block: 256
extern "C" __global__ void gdn_ddecay_dp_du_bwd_bf16(
    const float* __restrict__ W1,         // (BH, T, D, D)
    const float* __restrict__ dW1,        // (BH, T, D, D)
    const float* __restrict__ g_fwd,      // (BH, T, D, D)
    const float* __restrict__ g_rev,      // (BH, T, D, D)
    const float* __restrict__ W2,         // optional (W_z)
    const float* __restrict__ dW2,        // optional (dW_z)
    const float* __restrict__ gz_fwd,     // optional
    const float* __restrict__ gz_rev,     // optional
    const float* __restrict__ decay,      // (BH*T,)
    float* __restrict__ d_decay,          // (BH*T,)
    __mt_bfloat162* __restrict__ dP1_bf2, // (BH*T, D*D/2)
    __mt_bfloat162* __restrict__ dU1_bf2, // (BH*T, D*D/2)
    __mt_bfloat162* __restrict__ dP2_bf2, // optional
    float* __restrict__ dU_z_out,         // optional (BH*T, D)
    int T, int D
) {
    const int m = blockIdx.x;
    const int bh = m / T;
    const int t = m - bh * T;
    const int tid = threadIdx.x;
    const int dd2 = (D * D) >> 1;
    const int64_t base2 = static_cast<int64_t>(m) * dd2;
    const int64_t prev2 = static_cast<int64_t>(m - 1) * dd2;
    const float dec = decay[m];
    const float neg_dec = -dec;

    const float2* w1_2  = reinterpret_cast<const float2*>(W1) + base2;
    const float2* dw1_2 = reinterpret_cast<const float2*>(dW1) + base2;
    const float2* gf_2  = reinterpret_cast<const float2*>(g_fwd) + base2;
    const float2* gr_p2 = (t > 0) ? (reinterpret_cast<const float2*>(g_rev) + prev2) : nullptr;
    const float2* w2_2  = (W2 != nullptr) ? (reinterpret_cast<const float2*>(W2) + base2) : nullptr;
    const float2* dw2_2 = (dW2 != nullptr) ? (reinterpret_cast<const float2*>(dW2) + base2) : nullptr;

    float dot = 0.0f;
    for (int idx = tid; idx < dd2; idx += 256) {
        float2 w1v = w1_2[idx], dw1v = dw1_2[idx];
        dot = fmaf(dw1v.x, w1v.x, fmaf(dw1v.y, w1v.y, dot));
        dP1_bf2[base2 + idx] = __floats2bfloat162_rn(neg_dec * dw1v.x, neg_dec * dw1v.y);

        float2 gfv = gf_2[idx];
        float u0 = gfv.x, u1 = gfv.y;
        if (gr_p2 != nullptr) {
            float2 grv = gr_p2[idx];
            u0 += grv.x; u1 += grv.y;
        }
        dU1_bf2[base2 + idx] = __floats2bfloat162_rn(u0, u1);

        if (w2_2 != nullptr) {
            float2 w2v = w2_2[idx], dw2v = dw2_2[idx];
            dot = fmaf(dw2v.x, w2v.x, fmaf(dw2v.y, w2v.y, dot));
            dP2_bf2[base2 + idx] = __floats2bfloat162_rn(neg_dec * dw2v.x, neg_dec * dw2v.y);
        }
    }
    dot = warp_sum(dot);
    __shared__ float s_w[8];
    if ((tid & 31) == 0) s_w[tid >> 5] = dot;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_w[w];
        d_decay[m] = tot / fmaxf(dec, 1e-12f);
    }
    if (dU_z_out != nullptr && tid < D) {
        int64_t zoff = static_cast<int64_t>(m) * D + tid;
        float v = gz_fwd[zoff];
        if (t > 0) v += gz_rev[static_cast<int64_t>(m - 1) * D + tid];
        dU_z_out[zoff] = v;
    }
}

// 8i. Main Phase A Epilogue Bwd: fuses dK_rot, dK_pos, dV, dbeta in 1 single pass (eliminates ~1.8 GB FP32 casts & 16 kernels!)
//     Grid: dim3((S+3)/4, M), Block: 256 (4 groups of 64 threads; each group handles 1 spatial position, tid_g < 56 = D/2)
extern "C" __global__ void gdn_main_phase_a_epilogue_bwd_bf16(
    const __mt_bfloat162* __restrict__ kr2,
    const __mt_bfloat162* __restrict__ kp2,
    const __mt_bfloat162* __restrict__ v2,
    const float* __restrict__ beta,
    const __mt_bfloat162* __restrict__ dKr_b2,
    const __mt_bfloat162* __restrict__ dV_b2,
    const __mt_bfloat162* __restrict__ dKp_b_raw2,
    const float* __restrict__ dU_z,            // (M, D) fp32
    const __mt_bfloat162* __restrict__ dKr_bmm1,
    const __mt_bfloat162* __restrict__ dKr_bmm2,
    const __mt_bfloat162* __restrict__ dKp_bmm,
    __mt_bfloat162* __restrict__ dK_rot_out2,
    __mt_bfloat162* __restrict__ dK_pos_out2,
    __mt_bfloat162* __restrict__ dV_out2,
    float* __restrict__ dbeta_out,             // (M, S) fp32
    int S, int D
) {
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    const int grp = tid >> 6;        // 0..3
    const int ltid = tid & 63;       // 0..63
    const int s = blockIdx.x * 4 + grp;
    const int D2 = D >> 1;

    __shared__ float s_duz[112];
    __shared__ float s_w[4][2];
    if (tid < D) {
        s_duz[tid] = __bfloat162float(__float2bfloat16_rn(dU_z[static_cast<int64_t>(m) * D + tid]));
    }
    __syncthreads();

    if (s >= S) return;
    const int64_t ms = static_cast<int64_t>(m) * S + s;
    const float bf = __bfloat162float(__float2bfloat16_rn(beta[ms]));

    float dot = 0.0f;
    if (ltid < D2) {
        const int64_t off = ms * D2 + ltid;
        __mt_bfloat162 krv  = kr2[off],       kpv  = kp2[off],        vv   = v2[off];
        __mt_bfloat162 dkrb = dKr_b2[off],    dvb  = dV_b2[off],      dkpr = dKp_b_raw2[off];
        __mt_bfloat162 b1   = dKr_bmm1[off],  b2   = dKr_bmm2[off],   bp   = dKp_bmm[off];

        float dkr0 = __low2float(dkrb), dkr1 = __high2float(dkrb);
        float dv0  = __low2float(dvb),  dv1  = __high2float(dvb);
        float dkp0 = __bfloat162float(__float2bfloat16_rn(__low2float(dkpr)  + s_duz[ltid * 2 + 0]));
        float dkp1 = __bfloat162float(__float2bfloat16_rn(__high2float(dkpr) + s_duz[ltid * 2 + 1]));

        dK_rot_out2[off] = __floats2bfloat162_rn(
            __low2float(b1)  + __low2float(b2)  + dkr0 * bf,
            __high2float(b1) + __high2float(b2) + dkr1 * bf
        );
        dK_pos_out2[off] = __floats2bfloat162_rn(
            __low2float(bp)  + dkp0 * bf,
            __high2float(bp) + dkp1 * bf
        );
        dV_out2[off] = __floats2bfloat162_rn(dv0 * bf, dv1 * bf);

        dot = fmaf(dkr0, __low2float(krv), fmaf(dkr1, __high2float(krv),
              fmaf(dv0,  __low2float(vv),  fmaf(dv1,  __high2float(vv),
              fmaf(dkp0, __low2float(kpv), dkp1 * __high2float(kpv))))));
    }
    dot = warp_sum(dot);
    if ((ltid & 31) == 0) s_w[grp][ltid >> 5] = dot;
    __syncthreads();
    if (ltid == 0) {
        dbeta_out[ms] = s_w[grp][0] + s_w[grp][1];
    }
}

// 8j. Cam Phase A Epilogue Bwd: fuses dK, dV, dbeta_s in 1 pass!
//     Grid: dim3((S+3)/4, M), Block: 256
extern "C" __global__ void gdn_cam_phase_a_epilogue_bwd_bf16(
    const __mt_bfloat162* __restrict__ k2,
    const __mt_bfloat162* __restrict__ v2,
    const float* __restrict__ beta,
    const __mt_bfloat162* __restrict__ dK_b2,
    const __mt_bfloat162* __restrict__ dV_b2,
    const __mt_bfloat162* __restrict__ dK_bmm1,
    const __mt_bfloat162* __restrict__ dK_bmm2,
    __mt_bfloat162* __restrict__ dK_out2,
    __mt_bfloat162* __restrict__ dV_out2,
    float* __restrict__ dbeta_s_out,
    int S, int D, int beta_is_4d
) {
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    const int grp = tid >> 6;
    const int ltid = tid & 63;
    const int s = blockIdx.x * 4 + grp;
    const int D2 = D >> 1;

    __shared__ float s_w[4][2];
    if (s >= S) return;
    const int64_t ms = static_cast<int64_t>(m) * S + s;
    const float bf = __bfloat162float(__float2bfloat16_rn(beta_is_4d ? beta[ms] : beta[m]));

    float dot = 0.0f;
    if (ltid < D2) {
        const int64_t off = ms * D2 + ltid;
        __mt_bfloat162 kv = k2[off], vv = v2[off];
        __mt_bfloat162 dkb = dK_b2[off], dvb = dV_b2[off];
        __mt_bfloat162 b1 = dK_bmm1[off], b2 = dK_bmm2[off];

        float dk0 = __low2float(dkb), dk1 = __high2float(dkb);
        float dv0 = __low2float(dvb), dv1 = __high2float(dvb);

        dK_out2[off] = __floats2bfloat162_rn(
            __low2float(b1)  + __low2float(b2)  + dk0 * bf,
            __high2float(b1) + __high2float(b2) + dk1 * bf
        );
        dV_out2[off] = __floats2bfloat162_rn(dv0 * bf, dv1 * bf);

        dot = fmaf(dk0, __low2float(kv), fmaf(dk1, __high2float(kv),
              fmaf(dv0, __low2float(vv), dv1 * __high2float(vv))));
    }
    dot = warp_sum(dot);
    if ((ltid & 31) == 0) s_w[grp][ltid >> 5] = dot;
    __syncthreads();
    if (ltid == 0) {
        dbeta_s_out[ms] = s_w[grp][0] + s_w[grp][1];
    }
}

// =========================================================================
// C Launch Wrappers
// =========================================================================
extern "C" int launch_bidi_short_conv4_fwd(
    const void* x, const void* w, const void* b, void* y,
    int B, int T, int S, int C, int has_bias, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    dim3 grid(C / 64, (B * S + 7) / 8);
    bidi_short_conv4_fwd_bf16<<<grid, 256, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(x),
        reinterpret_cast<const __mt_bfloat16*>(w),
        reinterpret_cast<const __mt_bfloat16*>(b),
        reinterpret_cast<__mt_bfloat162*>(y),
        B, T, S, C, has_bias
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_bidi_short_conv4_bwd(
    const void* x, const void* dy, const void* w, void* dx,
    void* dw_part, void* db_part, void* dw_out, void* db_out,
    int B, int T, int S, int C, int has_bias, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    int n_bs_blk = (B * S + 7) / 8;
    dim3 grid(C / 64, n_bs_blk);
    bidi_short_conv4_bwd_bf16<<<grid, 256, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(x),
        reinterpret_cast<const __mt_bfloat162*>(dy),
        reinterpret_cast<const __mt_bfloat16*>(w),
        reinterpret_cast<__mt_bfloat162*>(dx),
        reinterpret_cast<float*>(dw_part),
        reinterpret_cast<float*>(db_part),
        B, T, S, C, has_bias
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    int nw = C * 4;
    reduce_rows_to_bf16<<<(nw + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<const float*>(dw_part),
        reinterpret_cast<__mt_bfloat16*>(dw_out),
        n_bs_blk, nw
    );
    err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    if (has_bias) {
        reduce_rows_to_bf16<<<(C + 255) / 256, 256, 0, s>>>(
            reinterpret_cast<const float*>(db_part),
            reinterpret_cast<__mt_bfloat16*>(db_out),
            n_bs_blk, C
        );
        err = musaGetLastError();
    }
    return static_cast<int>(err);
}

extern "C" int launch_bidi_short_conv4_qkv_k_fwd(
    const void* qkv, const void* w, const void* b, void* y,
    int B, int T, int S, int C, int has_bias, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    dim3 grid(C / 64, (B * S + 7) / 8);
    bidi_short_conv4_qkv_k_fwd_bf16<<<grid, 256, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(qkv),
        reinterpret_cast<const __mt_bfloat16*>(w),
        reinterpret_cast<const __mt_bfloat16*>(b),
        reinterpret_cast<__mt_bfloat162*>(y),
        B, T, S, C, has_bias
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_bidi_short_conv4_qkv_k_bwd(
    const void* qkv, const void* dy, const void* w, void* dqkv,
    void* dw_part, void* db_part, void* dw_out, void* db_out,
    int B, int T, int S, int C, int has_bias, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    int n_bs_blk = (B * S + 7) / 8;
    dim3 grid(C / 64, n_bs_blk);
    bidi_short_conv4_qkv_k_bwd_bf16<<<grid, 256, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(qkv),
        reinterpret_cast<const __mt_bfloat162*>(dy),
        reinterpret_cast<const __mt_bfloat16*>(w),
        reinterpret_cast<__mt_bfloat162*>(dqkv),
        reinterpret_cast<float*>(dw_part),
        reinterpret_cast<float*>(db_part),
        B, T, S, C, has_bias
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    int nw = C * 4;
    reduce_rows_to_bf16<<<(nw + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<const float*>(dw_part),
        reinterpret_cast<__mt_bfloat16*>(dw_out),
        n_bs_blk, nw
    );
    err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    if (has_bias) {
        reduce_rows_to_bf16<<<(C + 255) / 256, 256, 0, s>>>(
            reinterpret_cast<const float*>(db_part),
            reinterpret_cast<__mt_bfloat16*>(db_out),
            n_bs_blk, C
        );
        err = musaGetLastError();
    }
    return static_cast<int>(err);
}

extern "C" int launch_main_gdn_prep_fwd(
    const void* qkv, const void* kconv, const void* q_nw, const void* k_nw,
    const void* rcos, const void* rsin,
    void* q_rot, void* q_pos, void* k_rot, void* k_pos, void* v_out,
    void* q_ir, void* k_ir,
    int B, int N, int H, int D, float k_scale, float norm_eps, int has_norm, int is_bf16, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    int C = H * D;
    size_t smem = static_cast<size_t>(C) * sizeof(__mt_bfloat162);  // 2 * C2 * 4B = 8.96 KiB
    if (is_bf16) {
        main_gdn_prep_fwd_bf16<<<B * N, 256, smem, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(qkv), reinterpret_cast<const __mt_bfloat162*>(kconv),
            reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
            reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
            reinterpret_cast<__mt_bfloat16*>(q_rot), reinterpret_cast<__mt_bfloat16*>(q_pos),
            reinterpret_cast<__mt_bfloat16*>(k_rot), reinterpret_cast<__mt_bfloat16*>(k_pos),
            reinterpret_cast<__mt_bfloat16*>(v_out),
            reinterpret_cast<float*>(q_ir), reinterpret_cast<float*>(k_ir),
            B, N, H, D, k_scale, norm_eps, has_norm
        );
    } else {
        main_gdn_prep_fwd_f32<<<B * N, 256, smem, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(qkv), reinterpret_cast<const __mt_bfloat162*>(kconv),
            reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
            reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
            reinterpret_cast<float*>(q_rot), reinterpret_cast<float*>(q_pos),
            reinterpret_cast<float*>(k_rot), reinterpret_cast<float*>(k_pos),
            reinterpret_cast<float*>(v_out),
            reinterpret_cast<float*>(q_ir), reinterpret_cast<float*>(k_ir),
            B, N, H, D, k_scale, norm_eps, has_norm
        );
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_main_gdn_prep_bwd(
    const void* qkv, const void* kconv, const void* q_nw, const void* k_nw,
    const void* q_ir, const void* k_ir, const void* rcos, const void* rsin,
    const void* dq_rot, const void* dq_pos, const void* dk_rot, const void* dk_pos, const void* dv,
    void* dqkv, void* dkconv, void* dq_nw_part, void* dk_nw_part, void* dq_nw_out, void* dk_nw_out,
    int B, int N, int H, int D, float k_scale, int has_norm, int is_bf16, int n_chunks, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    int C = H * D;
    uintptr_t a8 = reinterpret_cast<uintptr_t>(qkv) | reinterpret_cast<uintptr_t>(kconv)
                 | reinterpret_cast<uintptr_t>(dq_rot) | reinterpret_cast<uintptr_t>(dq_pos)
                 | reinterpret_cast<uintptr_t>(dk_rot) | reinterpret_cast<uintptr_t>(dk_pos)
                 | reinterpret_cast<uintptr_t>(dv) | reinterpret_cast<uintptr_t>(dqkv)
                 | reinterpret_cast<uintptr_t>(dkconv);
    uintptr_t a16 = reinterpret_cast<uintptr_t>(q_nw) | reinterpret_cast<uintptr_t>(k_nw)
                  | reinterpret_cast<uintptr_t>(dq_nw_part) | reinterpret_cast<uintptr_t>(dk_nw_part);
    if (is_bf16 && has_norm && (D & 3) == 0 && C <= 2560 && (a8 & 7u) == 0 && (a16 & 15u) == 0) {
        int threads = (((C >> 2) + 31) >> 5) << 5;
        main_gdn_prep_bwd_576_bf16<<<n_chunks, threads, 0, s>>>(
            reinterpret_cast<const __mt_bfloat16*>(qkv), reinterpret_cast<const __mt_bfloat16*>(kconv),
            reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
            reinterpret_cast<const float*>(q_ir), reinterpret_cast<const float*>(k_ir),
            reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
            reinterpret_cast<const __mt_bfloat16*>(dq_rot), reinterpret_cast<const __mt_bfloat16*>(dq_pos),
            reinterpret_cast<const __mt_bfloat16*>(dk_rot), reinterpret_cast<const __mt_bfloat16*>(dk_pos),
            reinterpret_cast<const __mt_bfloat16*>(dv),
            reinterpret_cast<__mt_bfloat16*>(dqkv), reinterpret_cast<__mt_bfloat16*>(dkconv),
            reinterpret_cast<float*>(dq_nw_part), reinterpret_cast<float*>(dk_nw_part),
            B, N, H, D, k_scale, n_chunks
        );
    } else {
        size_t smem = static_cast<size_t>(5 * C + 16) * sizeof(float);
        if (smem > 46 * 1024) {
            musaFuncSetAttribute(is_bf16 ? (const void*)main_gdn_prep_bwd_bf16 : (const void*)main_gdn_prep_bwd_f32,
                                 musaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
            musaGetLastError();
        }
        if (is_bf16) {
            main_gdn_prep_bwd_bf16<<<n_chunks, 256, smem, s>>>(
                reinterpret_cast<const __mt_bfloat162*>(qkv), reinterpret_cast<const __mt_bfloat162*>(kconv),
                reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
                reinterpret_cast<const float*>(q_ir), reinterpret_cast<const float*>(k_ir),
                reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
                reinterpret_cast<const __mt_bfloat16*>(dq_rot), reinterpret_cast<const __mt_bfloat16*>(dq_pos),
                reinterpret_cast<const __mt_bfloat16*>(dk_rot), reinterpret_cast<const __mt_bfloat16*>(dk_pos),
                reinterpret_cast<const __mt_bfloat16*>(dv),
                reinterpret_cast<__mt_bfloat162*>(dqkv), reinterpret_cast<__mt_bfloat162*>(dkconv),
                reinterpret_cast<float*>(dq_nw_part), reinterpret_cast<float*>(dk_nw_part),
                B, N, H, D, k_scale, has_norm, n_chunks
            );
        } else {
            main_gdn_prep_bwd_f32<<<n_chunks, 256, smem, s>>>(
                reinterpret_cast<const __mt_bfloat162*>(qkv), reinterpret_cast<const __mt_bfloat162*>(kconv),
                reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
                reinterpret_cast<const float*>(q_ir), reinterpret_cast<const float*>(k_ir),
                reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
                reinterpret_cast<const float*>(dq_rot), reinterpret_cast<const float*>(dq_pos),
                reinterpret_cast<const float*>(dk_rot), reinterpret_cast<const float*>(dk_pos),
                reinterpret_cast<const float*>(dv),
                reinterpret_cast<__mt_bfloat162*>(dqkv), reinterpret_cast<__mt_bfloat162*>(dkconv),
                reinterpret_cast<float*>(dq_nw_part), reinterpret_cast<float*>(dk_nw_part),
                B, N, H, D, k_scale, has_norm, n_chunks
            );
        }
    }
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    if (has_norm) {
        reduce_rows_to_f32<<<(C + 255) / 256, 256, 0, s>>>(
            reinterpret_cast<const float*>(dq_nw_part), reinterpret_cast<float*>(dq_nw_out), n_chunks, C);
        reduce_rows_to_f32<<<(C + 255) / 256, 256, 0, s>>>(
            reinterpret_cast<const float*>(dk_nw_part), reinterpret_cast<float*>(dk_nw_out), n_chunks, C);
        err = musaGetLastError();
    }
    return static_cast<int>(err);
}

extern "C" int launch_bidi_state_scan_fwd(
    const void* W, const void* U, void* S_fwd, void* S_rev, int BH, int T, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    dim3 grid(7, BH, 2);
    bidi_state_scan_fwd_d112_f32<<<grid, 128, 0, s>>>(
        reinterpret_cast<const float*>(W), reinterpret_cast<const float*>(U),
        reinterpret_cast<float*>(S_fwd), reinterpret_cast<float*>(S_rev), BH, T
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_bidi_state_scan_bwd(
    const void* W, const void* dS_tot, void* g_fwd, void* g_rev, int BH, int T, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    dim3 grid(4, BH, 2);
    bidi_state_scan_bwd_d112_f32<<<grid, 196, 0, s>>>(
        reinterpret_cast<const float*>(W), reinterpret_cast<const float*>(dS_tot),
        reinterpret_cast<float*>(g_fwd), reinterpret_cast<float*>(g_rev), BH, T
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_bidi_z_scan_fwd(
    const void* Wz, const void* Uz, void* Sz_fwd, void* Sz_rev, int BH, int T, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    dim3 grid(BH, 2);
    bidi_z_scan_fwd_d112_f32<<<grid, 112, 0, s>>>(
        reinterpret_cast<const float*>(Wz), reinterpret_cast<const float*>(Uz),
        reinterpret_cast<float*>(Sz_fwd), reinterpret_cast<float*>(Sz_rev), BH, T
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_bidi_z_scan_bwd(
    const void* Wz, const void* dSz_tot, void* gz_fwd, void* gz_rev, int BH, int T, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    dim3 grid(BH, 2);
    bidi_z_scan_bwd_d112_f32<<<grid, 112, 0, s>>>(
        reinterpret_cast<const float*>(Wz), reinterpret_cast<const float*>(dSz_tot),
        reinterpret_cast<float*>(gz_fwd), reinterpret_cast<float*>(gz_rev), BH, T
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_out_div_fwd(
    const void* num, const void* den, void* out,
    int B, int N, int H, int D, float eps, int is_bf16, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    if (is_bf16) {
        out_div_to_bnc_fwd_bf16<<<B * N, 256, 0, s>>>(
            reinterpret_cast<const __mt_bfloat16*>(num), reinterpret_cast<const float*>(den),
            reinterpret_cast<__mt_bfloat162*>(out), B, N, H, D, eps);
    } else {
        out_div_to_bnc_fwd_f32<<<B * N, 256, 0, s>>>(
            reinterpret_cast<const float*>(num), reinterpret_cast<const float*>(den),
            reinterpret_cast<__mt_bfloat162*>(out), B, N, H, D, eps);
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_out_div_bwd(
    const void* dout, const void* num, const void* den, void* dnum, void* dden,
    int B, int N, int H, int D, float eps, int is_bf16, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    dim3 grid(B * N, H);
    if (is_bf16) {
        out_div_to_bnc_bwd_bf16<<<grid, 64, 0, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(dout), reinterpret_cast<const __mt_bfloat16*>(num),
            reinterpret_cast<const float*>(den), reinterpret_cast<__mt_bfloat16*>(dnum),
            reinterpret_cast<float*>(dden), B, N, H, D, eps);
    } else {
        out_div_to_bnc_bwd_f32<<<grid, 64, 0, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(dout), reinterpret_cast<const float*>(num),
            reinterpret_cast<const float*>(den), reinterpret_cast<float*>(dnum),
            reinterpret_cast<float*>(dden), B, N, H, D, eps);
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_cam_prep_fwd(
    const void* q, const void* k, const void* v, const void* q_nw, const void* k_nw,
    const void* P_T, const void* P_inv, const void* rcos, const void* rsin,
    void* q_out, void* k_out, void* v_out, void* inf_sq,
    void* q_ir, void* k_ir, void* pre_ksq, void* post_ksq,
    int B, int N, int H, int D, float k_scale, float norm_eps, int is_bf16, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    int C = H * D;
    size_t smem = static_cast<size_t>(C) * sizeof(__mt_bfloat162);  // 2 * C2 * 4B = 8.96 KiB
    if (is_bf16) {
        cam_prep_fwd_bf16<<<B * N, 256, smem, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(q), reinterpret_cast<const __mt_bfloat162*>(k),
            reinterpret_cast<const __mt_bfloat162*>(v), reinterpret_cast<const float*>(q_nw),
            reinterpret_cast<const float*>(k_nw), reinterpret_cast<const float*>(P_T),
            reinterpret_cast<const float*>(P_inv), reinterpret_cast<const float*>(rcos),
            reinterpret_cast<const float*>(rsin), reinterpret_cast<__mt_bfloat16*>(q_out),
            reinterpret_cast<__mt_bfloat16*>(k_out), reinterpret_cast<__mt_bfloat16*>(v_out),
            reinterpret_cast<float*>(inf_sq), reinterpret_cast<float*>(q_ir), reinterpret_cast<float*>(k_ir),
            reinterpret_cast<float*>(pre_ksq), reinterpret_cast<float*>(post_ksq),
            B, N, H, D, k_scale, norm_eps);
    } else {
        cam_prep_fwd_f32<<<B * N, 256, smem, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(q), reinterpret_cast<const __mt_bfloat162*>(k),
            reinterpret_cast<const __mt_bfloat162*>(v), reinterpret_cast<const float*>(q_nw),
            reinterpret_cast<const float*>(k_nw), reinterpret_cast<const float*>(P_T),
            reinterpret_cast<const float*>(P_inv), reinterpret_cast<const float*>(rcos),
            reinterpret_cast<const float*>(rsin), reinterpret_cast<float*>(q_out),
            reinterpret_cast<float*>(k_out), reinterpret_cast<float*>(v_out),
            reinterpret_cast<float*>(inf_sq), reinterpret_cast<float*>(q_ir), reinterpret_cast<float*>(k_ir),
            reinterpret_cast<float*>(pre_ksq), reinterpret_cast<float*>(post_ksq),
            B, N, H, D, k_scale, norm_eps);
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_cam_prep_bwd(
    const void* q, const void* k, const void* q_nw, const void* k_nw,
    const void* P_T, const void* P_inv, const void* rcos, const void* rsin,
    const void* q_ir, const void* k_ir, const void* pre_ksq, const void* post_ksq,
    const void* dq_out, const void* dk_out, const void* dv_out, const void* dinf,
    void* dq_raw, void* dk_raw, void* dv_raw,
    void* dq_nw_part, void* dk_nw_part, void* dq_nw_out, void* dk_nw_out,
    int B, int N, int H, int D, float k_scale, int is_bf16, int n_chunks, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    int C = H * D;
    size_t smem = static_cast<size_t>(5 * C + 240) * sizeof(float);  // 4*C + 2*C2 + 240 floats = 45.76 KiB (3 CTAs/MP -> nc=180!)
    if (smem > 46 * 1024) {
        musaFuncSetAttribute(is_bf16 ? (const void*)cam_prep_bwd_bf16 : (const void*)cam_prep_bwd_f32,
                             musaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        musaGetLastError();
    }
    if (is_bf16) {
        cam_prep_bwd_bf16<<<n_chunks, 256, smem, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(q), reinterpret_cast<const __mt_bfloat162*>(k),
            reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
            reinterpret_cast<const float*>(P_T), reinterpret_cast<const float*>(P_inv),
            reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
            reinterpret_cast<const float*>(q_ir), reinterpret_cast<const float*>(k_ir),
            reinterpret_cast<const float*>(pre_ksq), reinterpret_cast<const float*>(post_ksq),
            reinterpret_cast<const __mt_bfloat16*>(dq_out), reinterpret_cast<const __mt_bfloat16*>(dk_out),
            reinterpret_cast<const __mt_bfloat16*>(dv_out), reinterpret_cast<const float*>(dinf),
            reinterpret_cast<__mt_bfloat162*>(dq_raw), reinterpret_cast<__mt_bfloat162*>(dk_raw),
            reinterpret_cast<__mt_bfloat162*>(dv_raw),
            reinterpret_cast<float*>(dq_nw_part), reinterpret_cast<float*>(dk_nw_part),
            B, N, H, D, k_scale, n_chunks);
    } else {
        cam_prep_bwd_f32<<<n_chunks, 256, smem, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(q), reinterpret_cast<const __mt_bfloat162*>(k),
            reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
            reinterpret_cast<const float*>(P_T), reinterpret_cast<const float*>(P_inv),
            reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
            reinterpret_cast<const float*>(q_ir), reinterpret_cast<const float*>(k_ir),
            reinterpret_cast<const float*>(pre_ksq), reinterpret_cast<const float*>(post_ksq),
            reinterpret_cast<const float*>(dq_out), reinterpret_cast<const float*>(dk_out),
            reinterpret_cast<const float*>(dv_out), reinterpret_cast<const float*>(dinf),
            reinterpret_cast<__mt_bfloat162*>(dq_raw), reinterpret_cast<__mt_bfloat162*>(dk_raw),
            reinterpret_cast<__mt_bfloat162*>(dv_raw),
            reinterpret_cast<float*>(dq_nw_part), reinterpret_cast<float*>(dk_nw_part),
            B, N, H, D, k_scale, n_chunks);
    }
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    reduce_rows_to_f32<<<(C + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<const float*>(dq_nw_part), reinterpret_cast<float*>(dq_nw_out), n_chunks, C);
    reduce_rows_to_f32<<<(C + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<const float*>(dk_nw_part), reinterpret_cast<float*>(dk_nw_out), n_chunks, C);
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_cam_out_apply_o_fwd(
    const void* x, const void* P, const void* rc, const void* rs, void* out,
    int B, int N, int H, int D, int is_bf16, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    if (is_bf16) {
        cam_out_apply_o_fwd_bf16<<<B * N, 256, 0, s>>>(
            reinterpret_cast<const __mt_bfloat16*>(x), reinterpret_cast<const float*>(P),
            reinterpret_cast<const float*>(rc), reinterpret_cast<const float*>(rs),
            reinterpret_cast<__mt_bfloat162*>(out), B, N, H, D);
    } else {
        cam_out_apply_o_fwd_f32<<<B * N, 256, 0, s>>>(
            reinterpret_cast<const float*>(x), reinterpret_cast<const float*>(P),
            reinterpret_cast<const float*>(rc), reinterpret_cast<const float*>(rs),
            reinterpret_cast<__mt_bfloat162*>(out), B, N, H, D);
    }
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_cam_out_apply_o_bwd(
    const void* dout, const void* P, const void* rc, const void* rs, void* dx,
    int B, int N, int H, int D, int is_bf16, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    if (is_bf16) {
        cam_out_apply_o_bwd_bf16<<<B * N, 256, 0, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(dout), reinterpret_cast<const float*>(P),
            reinterpret_cast<const float*>(rc), reinterpret_cast<const float*>(rs),
            reinterpret_cast<__mt_bfloat16*>(dx), B, N, H, D);
    } else {
        cam_out_apply_o_bwd_f32<<<B * N, 256, 0, s>>>(
            reinterpret_cast<const __mt_bfloat162*>(dout), reinterpret_cast<const float*>(P),
            reinterpret_cast<const float*>(rc), reinterpret_cast<const float*>(rs),
            reinterpret_cast<float*>(dx), B, N, H, D);
    }
    return static_cast<int>(musaGetLastError());
}

// 8k. R9 Fast 288t uint4 kernels for H=20, D=112 (14t/head x 20 heads = 280 active threads!)
extern "C" __global__ void main_gdn_prep_scale_fwd_u4_bf16(
    const __mt_bfloat16* __restrict__ qkv,
    const __mt_bfloat16* __restrict__ kconv,
    const float* __restrict__ q_nw,
    const float* __restrict__ k_nw,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const float* __restrict__ beta,       // (B*H, N) fp32
    __mt_bfloat16* __restrict__ q_rot,
    __mt_bfloat16* __restrict__ q_pos,
    __mt_bfloat16* __restrict__ k_rot,
    __mt_bfloat16* __restrict__ k_pos,
    __mt_bfloat16* __restrict__ kr_b,
    __mt_bfloat16* __restrict__ kp_b,
    __mt_bfloat16* __restrict__ v_b,
    __mt_bfloat16* __restrict__ v_out,    // nullptr in fwd, non-null in bwd recompute
    float* __restrict__ q_ir_out,         // optional
    float* __restrict__ k_ir_out,         // optional
    int B, int N, float k_scale, float norm_eps
) {
    constexpr int H = 20, D = 112, D8 = 14, C = 2240, C8 = 280;
    const int bn = blockIdx.x;
    const int b_idx = bn / N;
    const int n_idx = bn - b_idx * N;
    const int tid = threadIdx.x;
    const bool active = (tid < C8);

    __shared__ float s_cos[112], s_sin[112], s_beta[20], s_qred[16], s_kred[16];
    if (tid < D) {
        s_cos[tid] = rope_cos[static_cast<int64_t>(n_idx) * D + tid];
        s_sin[tid] = rope_sin[static_cast<int64_t>(n_idx) * D + tid];
    } else if (tid < D + H) {
        int hh = tid - D;
        s_beta[hh] = __bfloat162float(__float2bfloat16_rn(
            beta[static_cast<int64_t>(b_idx * H + hh) * N + n_idx]
        ));
    }

    const uint4* q4 = reinterpret_cast<const uint4*>(qkv) + static_cast<int64_t>(bn) * 3 * C8;
    const uint4* k4 = (kconv != nullptr)
        ? (reinterpret_cast<const uint4*>(kconv) + static_cast<int64_t>(bn) * C8)
        : (q4 + C8);
    const uint4* v4 = q4 + 2 * C8;

    float qf[8], kf[8];
    float q_sq = 0.0f, k_sq = 0.0f;
    if (active) {
        uint4 qv = q4[tid], kv = k4[tid];
        const __mt_bfloat162* qh = reinterpret_cast<const __mt_bfloat162*>(&qv);
        const __mt_bfloat162* kh = reinterpret_cast<const __mt_bfloat162*>(&kv);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float a0 = __low2float(qh[i]), a1 = __high2float(qh[i]);
            float b0 = __low2float(kh[i]), b1 = __high2float(kh[i]);
            qf[2*i] = a0; qf[2*i+1] = a1; kf[2*i] = b0; kf[2*i+1] = b1;
            q_sq = fmaf(a0, a0, fmaf(a1, a1, q_sq));
            k_sq = fmaf(b0, b0, fmaf(b1, b1, k_sq));
        }
    }
    q_sq = warp_sum(q_sq);
    k_sq = warp_sum(k_sq);
    if ((tid & 31) == 0) { s_qred[tid >> 5] = q_sq; s_kred[tid >> 5] = k_sq; }
    __syncthreads();
    if (tid == 0) {
        float tq = 0.0f, tk = 0.0f;
        #pragma unroll
        for (int w = 0; w < 9; ++w) { tq += s_qred[w]; tk += s_kred[w]; }
        float qir = rsqrtf(tq * (1.0f / static_cast<float>(C)) + norm_eps);
        float kir = rsqrtf(tk * (1.0f / static_cast<float>(C)) + norm_eps);
        s_qred[0] = qir; s_kred[0] = kir;
        if (q_ir_out != nullptr) q_ir_out[bn] = qir;
        if (k_ir_out != nullptr) k_ir_out[bn] = kir;
    }
    __syncthreads();
    if (!active) return;

    const float q_ir = s_qred[0], k_ir = s_kred[0];
    const int h = tid / D8;
    const int d8 = tid - h * D8;
    const int d0 = d8 << 3;
    const int c0 = tid << 3;
    const float bf = s_beta[h];

    uint4 vv = v4[tid];
    const __mt_bfloat162* vh = reinterpret_cast<const __mt_bfloat162*>(&vv);
    uint4 u_qr, u_qp, u_kr, u_kp, u_krb, u_kpb, u_vb;
    __mt_bfloat162* p_qr  = reinterpret_cast<__mt_bfloat162*>(&u_qr);
    __mt_bfloat162* p_qp  = reinterpret_cast<__mt_bfloat162*>(&u_qp);
    __mt_bfloat162* p_kr  = reinterpret_cast<__mt_bfloat162*>(&u_kr);
    __mt_bfloat162* p_kp  = reinterpret_cast<__mt_bfloat162*>(&u_kp);
    __mt_bfloat162* p_krb = reinterpret_cast<__mt_bfloat162*>(&u_krb);
    __mt_bfloat162* p_kpb = reinterpret_cast<__mt_bfloat162*>(&u_kpb);
    __mt_bfloat162* p_vb  = reinterpret_cast<__mt_bfloat162*>(&u_vb);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int c = c0 + 2 * i, d = d0 + 2 * i;
        float q0 = fmaxf(0.0f, qf[2*i]   * q_ir * q_nw[c]);
        float q1 = fmaxf(0.0f, qf[2*i+1] * q_ir * q_nw[c + 1]);
        float k0 = fmaxf(0.0f, kf[2*i]   * k_ir * k_nw[c])     * k_scale;
        float k1 = fmaxf(0.0f, kf[2*i+1] * k_ir * k_nw[c + 1]) * k_scale;
        float ce = s_cos[d], co = s_cos[d + 1], se = s_sin[d], so = s_sin[d + 1];
        float qr0 = fmaf(q0, ce, q1 * se), qr1 = fmaf(q1, co, q0 * so);
        float kr0 = fmaf(k0, ce, k1 * se), kr1 = fmaf(k1, co, k0 * so);
        __mt_bfloat162 kr2 = __floats2bfloat162_rn(kr0, kr1);
        __mt_bfloat162 kp2 = __floats2bfloat162_rn(k0, k1);
        p_qr[i]  = __floats2bfloat162_rn(qr0, qr1);
        p_qp[i]  = __floats2bfloat162_rn(q0, q1);
        p_kr[i]  = kr2;
        p_kp[i]  = kp2;
        p_krb[i] = __floats2bfloat162_rn(__low2float(kr2) * bf, __high2float(kr2) * bf);
        p_kpb[i] = __floats2bfloat162_rn(__low2float(kp2) * bf, __high2float(kp2) * bf);
        p_vb[i]  = __floats2bfloat162_rn(__low2float(vh[i]) * bf, __high2float(vh[i]) * bf);
    }
    int64_t out_off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D8 + d8;
    reinterpret_cast<uint4*>(q_rot)[out_off] = u_qr;
    reinterpret_cast<uint4*>(q_pos)[out_off] = u_qp;
    reinterpret_cast<uint4*>(k_rot)[out_off] = u_kr;
    reinterpret_cast<uint4*>(k_pos)[out_off] = u_kp;
    reinterpret_cast<uint4*>(kr_b)[out_off]  = u_krb;
    reinterpret_cast<uint4*>(kp_b)[out_off]  = u_kpb;
    reinterpret_cast<uint4*>(v_b)[out_off]   = u_vb;
    if (v_out != nullptr) reinterpret_cast<uint4*>(v_out)[out_off] = vv;
}

extern "C" __global__ void reduce_kp_b_to_uz_u4_bf16(
    const __mt_bfloat16* __restrict__ kp_b,  // (M, S, 112)
    float* __restrict__ U_z,                 // (M, 112)
    int S
) {
    constexpr int D = 112, D8 = 14;
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int grp = tid >> 4;   // 0..15 (16 rows parallel)
    const int ltid = tid & 15;  // 0..13 active
    const uint4* kp4 = reinterpret_cast<const uint4*>(kp_b) + static_cast<int64_t>(m) * S * D8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (ltid < D8) {
        for (int s = grp; s < S; s += 16) {
            uint4 v = kp4[static_cast<int64_t>(s) * D8 + ltid];
            const __mt_bfloat162* h = reinterpret_cast<const __mt_bfloat162*>(&v);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[2*i]   += __low2float(h[i]);
                acc[2*i+1] += __high2float(h[i]);
            }
        }
    }
    __shared__ float s_uz[16][112];
    if (ltid < D8) {
        int d0 = ltid << 3;
        #pragma unroll
        for (int i = 0; i < 8; ++i) s_uz[grp][d0 + i] = acc[i];
    }
    __syncthreads();
    if (tid < D) {
        float sum = 0.0f;
        #pragma unroll
        for (int g = 0; g < 16; ++g) sum += s_uz[g][tid];
        U_z[static_cast<int64_t>(m) * D + tid] = sum;
    }
}

extern "C" __global__ void cam_prep_fwd_u4_bf16(
    const __mt_bfloat16* __restrict__ q_in,
    const __mt_bfloat16* __restrict__ k_in,
    const __mt_bfloat16* __restrict__ v_in,
    const float* __restrict__ q_nw, const float* __restrict__ k_nw,
    const float* __restrict__ P_T, const float* __restrict__ P_inv,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    const float* __restrict__ beta_opt,   // optional (for bwd recompute: fuses k_b, v_b!)
    __mt_bfloat16* __restrict__ q_out,
    __mt_bfloat16* __restrict__ k_out,
    __mt_bfloat16* __restrict__ v_out,
    __mt_bfloat16* __restrict__ k_b_out,  // optional
    __mt_bfloat16* __restrict__ v_b_out,  // optional
    float* __restrict__ inf_sq,           // optional (nullptr in bwd recompute)
    float* __restrict__ q_ir_out,         // optional
    float* __restrict__ k_ir_out,         // optional
    float* __restrict__ pre_out,          // optional
    float* __restrict__ post_out,         // optional
    int B, int T, int S, int beta_is_4d, float k_scale, float norm_eps
) {
    constexpr int H = 20, D = 112, D8 = 14, D_half = 56, C = 2240, C8 = 280;
    const int N = T * S;
    const int bn = blockIdx.x, tid = threadIdx.x;
    const int b_idx = bn / N, n_idx = bn - b_idx * N;
    const bool active = (tid < C8);

    __shared__ float s_PT[16], s_Pinv[16], s_cos[56], s_sin[56], s_qred[16], s_kred[16];
    __shared__ float s_beta[20], s_pre[280], s_post[280];
    if (tid < 16) {
        s_PT[tid]   = P_T[static_cast<int64_t>(bn) * 16 + tid];
        s_Pinv[tid] = P_inv[static_cast<int64_t>(bn) * 16 + tid];
    } else if (tid < 16 + D_half) {
        int r = tid - 16;
        s_cos[r] = rope_cos[static_cast<int64_t>(n_idx) * D_half + r];
        s_sin[r] = rope_sin[static_cast<int64_t>(n_idx) * D_half + r];
    } else if (beta_opt != nullptr && tid < 16 + D_half + H) {
        int hh = tid - 16 - D_half;
        float bv = beta_is_4d
            ? beta_opt[static_cast<int64_t>(b_idx * H + hh) * N + n_idx]
            : beta_opt[static_cast<int64_t>(b_idx * H + hh) * T + (n_idx / S)];
        s_beta[hh] = __bfloat162float(__float2bfloat16_rn(bv));
    }

    const uint4* q4;
    const uint4* k4;
    const uint4* v4;
    if (v_in == nullptr) {
        q4 = reinterpret_cast<const uint4*>(q_in) + static_cast<int64_t>(bn) * 3 * C8;
        k4 = (k_in != nullptr)
            ? (reinterpret_cast<const uint4*>(k_in) + static_cast<int64_t>(bn) * C8)
            : (q4 + C8);
        v4 = q4 + 2 * C8;
    } else {
        q4 = reinterpret_cast<const uint4*>(q_in) + static_cast<int64_t>(bn) * C8;
        k4 = reinterpret_cast<const uint4*>(k_in) + static_cast<int64_t>(bn) * C8;
        v4 = reinterpret_cast<const uint4*>(v_in) + static_cast<int64_t>(bn) * C8;
    }

    uint4 u_q, u_k;
    float q_sq = 0.0f, k_sq = 0.0f;
    if (active) {
        u_q = q4[tid]; u_k = k4[tid];
        const __mt_bfloat162* pq = reinterpret_cast<const __mt_bfloat162*>(&u_q);
        const __mt_bfloat162* pk = reinterpret_cast<const __mt_bfloat162*>(&u_k);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float q0 = __low2float(pq[i]), q1 = __high2float(pq[i]);
            float k0 = __low2float(pk[i]), k1 = __high2float(pk[i]);
            q_sq = fmaf(q0, q0, fmaf(q1, q1, q_sq));
            k_sq = fmaf(k0, k0, fmaf(k1, k1, k_sq));
        }
    }
    q_sq = warp_sum(q_sq);
    k_sq = warp_sum(k_sq);
    if ((tid & 31) == 0) { s_qred[tid >> 5] = q_sq; s_kred[tid >> 5] = k_sq; }
    __syncthreads();
    if (tid == 0) {
        float tq = 0.0f, tk = 0.0f;
        #pragma unroll
        for (int w = 0; w < 9; ++w) { tq += s_qred[w]; tk += s_kred[w]; }
        float qir = rsqrtf(tq * (1.0f / static_cast<float>(C)) + norm_eps);
        float kir = rsqrtf(tk * (1.0f / static_cast<float>(C)) + norm_eps);
        s_qred[0] = qir; s_kred[0] = kir;
        if (q_ir_out != nullptr) q_ir_out[bn] = qir;
        if (k_ir_out != nullptr) k_ir_out[bn] = kir;
    }
    __syncthreads();

    if (active) {
        const float q_ir = s_qred[0], k_ir = s_kred[0];
        const int h = tid / D8;
        const int d8 = tid - h * D8;
        const int c0 = tid << 3;
        uint4 u_v = v4[tid];
        const __mt_bfloat162* pq = reinterpret_cast<const __mt_bfloat162*>(&u_q);
        const __mt_bfloat162* pk = reinterpret_cast<const __mt_bfloat162*>(&u_k);
        const __mt_bfloat162* pv = reinterpret_cast<const __mt_bfloat162*>(&u_v);
        uint4 o_q, o_k, o_v;
        __mt_bfloat162* mq = reinterpret_cast<__mt_bfloat162*>(&o_q);
        __mt_bfloat162* mk = reinterpret_cast<__mt_bfloat162*>(&o_k);
        __mt_bfloat162* mv = reinterpret_cast<__mt_bfloat162*>(&o_v);
        float pre_s = 0.0f, post_s = 0.0f;

        if (d8 < 7) {
            #pragma unroll
            for (int g = 0; g < 2; ++g) {
                int i0 = g * 2, cc = c0 + g * 4;
                float qx0 = fmaxf(0.0f, __low2float(pq[i0])    * q_ir * q_nw[cc + 0]);
                float qx1 = fmaxf(0.0f, __high2float(pq[i0])   * q_ir * q_nw[cc + 1]);
                float qx2 = fmaxf(0.0f, __low2float(pq[i0+1])  * q_ir * q_nw[cc + 2]);
                float qx3 = fmaxf(0.0f, __high2float(pq[i0+1]) * q_ir * q_nw[cc + 3]);
                float kx0 = fmaxf(0.0f, __low2float(pk[i0])    * k_ir * k_nw[cc + 0]) * k_scale;
                float kx1 = fmaxf(0.0f, __high2float(pk[i0])   * k_ir * k_nw[cc + 1]) * k_scale;
                float kx2 = fmaxf(0.0f, __low2float(pk[i0+1])  * k_ir * k_nw[cc + 2]) * k_scale;
                float kx3 = fmaxf(0.0f, __high2float(pk[i0+1]) * k_ir * k_nw[cc + 3]) * k_scale;
                float vx0 = __low2float(pv[i0]),   vx1 = __high2float(pv[i0]);
                float vx2 = __low2float(pv[i0+1]), vx3 = __high2float(pv[i0+1]);
                pre_s += kx0*kx0 + kx1*kx1 + kx2*kx2 + kx3*kx3;

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
                post_s += ky0*ky0 + ky1*ky1 + ky2*ky2 + ky3*ky3;
                mq[i0] = __floats2bfloat162_rn(qy0, qy1); mq[i0+1] = __floats2bfloat162_rn(qy2, qy3);
                mk[i0] = __floats2bfloat162_rn(ky0, ky1); mk[i0+1] = __floats2bfloat162_rn(ky2, ky3);
                mv[i0] = __floats2bfloat162_rn(vy0, vy1); mv[i0+1] = __floats2bfloat162_rn(vy2, vy3);
            }
        } else {
            int r0 = (d8 - 7) << 3;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = r0 + 2 * i, cc = c0 + 2 * i;
                float q0 = fmaxf(0.0f, __low2float(pq[i])  * q_ir * q_nw[cc]);
                float q1 = fmaxf(0.0f, __high2float(pq[i]) * q_ir * q_nw[cc + 1]);
                float k0 = fmaxf(0.0f, __low2float(pk[i])  * k_ir * k_nw[cc])     * k_scale;
                float k1 = fmaxf(0.0f, __high2float(pk[i]) * k_ir * k_nw[cc + 1]) * k_scale;
                float v0 = __low2float(pv[i]), v1 = __high2float(pv[i]);
                float ce = s_cos[r], co = s_cos[r + 1], se = s_sin[r], so = s_sin[r + 1];
                float qr0 = fmaf(q0, ce, q1 * se), qr1 = fmaf(q1, co, q0 * so);
                float kr0 = fmaf(k0, ce, k1 * se), kr1 = fmaf(k1, co, k0 * so);
                float vr0 = fmaf(v0, ce, v1 * se), vr1 = fmaf(v1, co, v0 * so);
                pre_s += k0*k0 + k1*k1;
                post_s += kr0*kr0 + kr1*kr1;
                mq[i] = __floats2bfloat162_rn(qr0, qr1);
                mk[i] = __floats2bfloat162_rn(kr0, kr1);
                mv[i] = __floats2bfloat162_rn(vr0, vr1);
            }
        }
        if (inf_sq != nullptr) { s_pre[tid] = pre_s; s_post[tid] = post_s; }
        int64_t out_off = (static_cast<int64_t>(b_idx * H + h) * N + n_idx) * D8 + d8;
        reinterpret_cast<uint4*>(q_out)[out_off] = o_q;
        reinterpret_cast<uint4*>(k_out)[out_off] = o_k;
        reinterpret_cast<uint4*>(v_out)[out_off] = o_v;
        if (beta_opt != nullptr) {
            float bf = s_beta[h];
            uint4 o_kb, o_vb;
            __mt_bfloat162* mkb = reinterpret_cast<__mt_bfloat162*>(&o_kb);
            __mt_bfloat162* mvb = reinterpret_cast<__mt_bfloat162*>(&o_vb);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                mkb[i] = __floats2bfloat162_rn(__low2float(mk[i]) * bf, __high2float(mk[i]) * bf);
                mvb[i] = __floats2bfloat162_rn(__low2float(mv[i]) * bf, __high2float(mv[i]) * bf);
            }
            reinterpret_cast<uint4*>(k_b_out)[out_off] = o_kb;
            reinterpret_cast<uint4*>(v_b_out)[out_off] = o_vb;
        }
    }
    if (inf_sq != nullptr) {
        __syncthreads();
        if (tid < H) {
            float pre = 0.0f, post = 0.0f;
            int base = tid * D8;
            #pragma unroll
            for (int i = 0; i < D8; ++i) { pre += s_pre[base + i]; post += s_post[base + i]; }
            int64_t bhn = static_cast<int64_t>(b_idx * H + tid) * N + n_idx;
            if (pre_out  != nullptr) pre_out[bhn]  = pre;
            if (post_out != nullptr) post_out[bhn] = post;
            inf_sq[bhn] = fmaxf(post, 1e-12f) / fmaxf(pre, 1e-12f);
        }
    }
}

extern "C" __global__ void cam_phase_a_scale_fused_u4_bf16(
    const __mt_bfloat16* __restrict__ k,
    const __mt_bfloat16* __restrict__ v,
    const float* __restrict__ inf_sq,
    const float* __restrict__ beta_raw,
    __mt_bfloat16* __restrict__ k_b,
    __mt_bfloat16* __restrict__ v_b,
    float* __restrict__ beta_m_out,
    float* __restrict__ frame_inf_out,
    int S, int beta_is_4d
) {
    constexpr int D8 = 14;
    const int m = blockIdx.x, tid = threadIdx.x;
    const int64_t base_ms = static_cast<int64_t>(m) * S;
    float sum = 0.0f;
    for (int s = tid; s < S; s += 256) sum += inf_sq[base_ms + s];
    sum = warp_sum(sum);
    __shared__ float sr[8];
    if ((tid & 31) == 0) sr[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) t += sr[w];
        float finf = t / static_cast<float>(S);
        frame_inf_out[m] = finf;
        sr[0] = 1.0f / fmaxf(finf, 1.0f);
    }
    __syncthreads();
    const float inv_denom = sr[0];
    const float b3d = beta_is_4d ? 0.0f : (beta_raw[m] * inv_denom);
    if (!beta_is_4d && tid == 0) beta_m_out[m] = b3d;

    const int grp = tid >> 4, ltid = tid & 15;
    const uint4* k4 = reinterpret_cast<const uint4*>(k) + base_ms * D8;
    const uint4* v4 = reinterpret_cast<const uint4*>(v) + base_ms * D8;
    uint4* kb4 = reinterpret_cast<uint4*>(k_b) + base_ms * D8;
    uint4* vb4 = reinterpret_cast<uint4*>(v_b) + base_ms * D8;
    for (int s = grp; s < S; s += 16) {
        float bf_raw = beta_is_4d ? (beta_raw[base_ms + s] * inv_denom) : b3d;
        if (beta_is_4d && ltid == 0) beta_m_out[base_ms + s] = bf_raw;
        float bf = __bfloat162float(__float2bfloat16_rn(bf_raw));
        if (ltid < D8) {
            int64_t off = static_cast<int64_t>(s) * D8 + ltid;
            uint4 kv = k4[off], vv = v4[off];
            const __mt_bfloat162* pk = reinterpret_cast<const __mt_bfloat162*>(&kv);
            const __mt_bfloat162* pv = reinterpret_cast<const __mt_bfloat162*>(&vv);
            uint4 ok, ov;
            __mt_bfloat162* qk = reinterpret_cast<__mt_bfloat162*>(&ok);
            __mt_bfloat162* qv = reinterpret_cast<__mt_bfloat162*>(&ov);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                qk[i] = __floats2bfloat162_rn(__low2float(pk[i]) * bf, __high2float(pk[i]) * bf);
                qv[i] = __floats2bfloat162_rn(__low2float(pv[i]) * bf, __high2float(pv[i]) * bf);
            }
            kb4[off] = ok;
            vb4[off] = ov;
        }
    }
}

extern "C" __global__ void out_div_bwd_from_out_dqpos_u4_bf16(
    const __mt_bfloat16* __restrict__ dout,
    const __mt_bfloat16* __restrict__ out,
    const float* __restrict__ den,
    const float* __restrict__ Sz_tot,
    __mt_bfloat16* __restrict__ dnum,
    float* __restrict__ dden,
    __mt_bfloat16* __restrict__ dq_pos,
    int H, int T, int S, float eps
) {
    constexpr int D = 112, D8 = 14;
    const int tid = threadIdx.x;
    const int grp = tid >> 4, ltid = tid & 15;
    const int s_idx = blockIdx.x * 16 + grp;
    const int t_idx = blockIdx.y;
    const int bh = blockIdx.z;
    if (s_idx >= S) return;
    const int b_idx = bh / H, h = bh - b_idx * H;
    const int N = T * S, n_idx = t_idx * S + s_idx;
    const int64_t bn = static_cast<int64_t>(b_idx) * N + n_idx;
    const int64_t bhn = static_cast<int64_t>(bh) * N + n_idx;
    const float inv_d = 1.0f / (den[bhn] + eps);
    float dot = 0.0f;
    if (ltid < D8) {
        const int64_t off_bnc = bn * (H * D8) + h * D8 + ltid;
        uint4 dv = reinterpret_cast<const uint4*>(dout)[off_bnc];
        uint4 ov = reinterpret_cast<const uint4*>(out)[off_bnc];
        const __mt_bfloat162* pd = reinterpret_cast<const __mt_bfloat162*>(&dv);
        const __mt_bfloat162* po = reinterpret_cast<const __mt_bfloat162*>(&ov);
        uint4 o_dnum;
        __mt_bfloat162* qn = reinterpret_cast<__mt_bfloat162*>(&o_dnum);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float g0 = __low2float(pd[i]), g1 = __high2float(pd[i]);
            qn[i] = __floats2bfloat162_rn(g0 * inv_d, g1 * inv_d);
            dot = fmaf(g0, __low2float(po[i]), fmaf(g1, __high2float(po[i]), dot));
        }
        reinterpret_cast<uint4*>(dnum)[bhn * D8 + ltid] = o_dnum;
    }
    dot = half_warp_sum(dot);
    const float dden_val = -__shfl_sync(0xffffffffu, dot, tid & 16) * inv_d;
    if (ltid == 0) dden[bhn] = dden_val;
    if (ltid < D8) {
        const float* szp = Sz_tot + (static_cast<int64_t>(bh) * T + t_idx) * D + (ltid << 3);
        uint4 o_dq;
        __mt_bfloat162* qq = reinterpret_cast<__mt_bfloat162*>(&o_dq);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            qq[i] = __floats2bfloat162_rn(dden_val * szp[2*i], dden_val * szp[2*i+1]);
        }
        reinterpret_cast<uint4*>(dq_pos)[bhn * D8 + ltid] = o_dq;
    }
}

extern "C" __global__ void gdn_main_phase_a_epilogue_bwd_u4_bf16(
    const __mt_bfloat16* __restrict__ kr,
    const __mt_bfloat16* __restrict__ kp,
    const __mt_bfloat16* __restrict__ v,
    const float* __restrict__ beta,
    const __mt_bfloat16* __restrict__ dKr_b,
    const __mt_bfloat16* __restrict__ dV_b,
    const __mt_bfloat16* __restrict__ dKp_b_raw,
    const float* __restrict__ dU_z,
    const __mt_bfloat16* __restrict__ b1,
    const __mt_bfloat16* __restrict__ b2,
    const __mt_bfloat16* __restrict__ bp,
    __mt_bfloat16* __restrict__ dKr_out,
    __mt_bfloat16* __restrict__ dKp_out,
    __mt_bfloat16* __restrict__ dV_out,
    float* __restrict__ dbeta_out,
    int S
) {
    constexpr int D = 112, D8 = 14;
    const int m = blockIdx.y, tid = threadIdx.x;
    const int grp = tid >> 4, ltid = tid & 15;
    const int s = blockIdx.x * 16 + grp;
    __shared__ float s_duz[112];
    if (tid < D) s_duz[tid] = __bfloat162float(__float2bfloat16_rn(dU_z[static_cast<int64_t>(m) * D + tid]));
    __syncthreads();
    if (s >= S) return;
    const int64_t ms = static_cast<int64_t>(m) * S + s;
    const float bf = __bfloat162float(__float2bfloat16_rn(beta[ms]));
    float dot = 0.0f;
    if (ltid < D8) {
        const int64_t off = ms * D8 + ltid;
        const int d0 = ltid << 3;
        uint4 u_kr   = reinterpret_cast<const uint4*>(kr)[off];
        uint4 u_kp   = reinterpret_cast<const uint4*>(kp)[off];
        uint4 u_v    = reinterpret_cast<const uint4*>(v)[off];
        uint4 u_dkrb = reinterpret_cast<const uint4*>(dKr_b)[off];
        uint4 u_dvb  = reinterpret_cast<const uint4*>(dV_b)[off];
        uint4 u_dkpr = reinterpret_cast<const uint4*>(dKp_b_raw)[off];
        uint4 u_b1   = reinterpret_cast<const uint4*>(b1)[off];
        uint4 u_b2   = reinterpret_cast<const uint4*>(b2)[off];
        uint4 u_bp   = reinterpret_cast<const uint4*>(bp)[off];
        const __mt_bfloat162 *p_kr   = reinterpret_cast<const __mt_bfloat162*>(&u_kr);
        const __mt_bfloat162 *p_kp   = reinterpret_cast<const __mt_bfloat162*>(&u_kp);
        const __mt_bfloat162 *p_v    = reinterpret_cast<const __mt_bfloat162*>(&u_v);
        const __mt_bfloat162 *p_dkrb = reinterpret_cast<const __mt_bfloat162*>(&u_dkrb);
        const __mt_bfloat162 *p_dvb  = reinterpret_cast<const __mt_bfloat162*>(&u_dvb);
        const __mt_bfloat162 *p_dkpr = reinterpret_cast<const __mt_bfloat162*>(&u_dkpr);
        const __mt_bfloat162 *p_b1   = reinterpret_cast<const __mt_bfloat162*>(&u_b1);
        const __mt_bfloat162 *p_b2   = reinterpret_cast<const __mt_bfloat162*>(&u_b2);
        const __mt_bfloat162 *p_bp   = reinterpret_cast<const __mt_bfloat162*>(&u_bp);
        uint4 o_kr, o_kp, o_v;
        __mt_bfloat162 *q_kr = reinterpret_cast<__mt_bfloat162*>(&o_kr);
        __mt_bfloat162 *q_kp = reinterpret_cast<__mt_bfloat162*>(&o_kp);
        __mt_bfloat162 *q_v  = reinterpret_cast<__mt_bfloat162*>(&o_v);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float dkr0 = __low2float(p_dkrb[i]), dkr1 = __high2float(p_dkrb[i]);
            float dv0  = __low2float(p_dvb[i]),  dv1  = __high2float(p_dvb[i]);
            float dkp0 = __bfloat162float(__float2bfloat16_rn(__low2float(p_dkpr[i])  + s_duz[d0 + 2*i]));
            float dkp1 = __bfloat162float(__float2bfloat16_rn(__high2float(p_dkpr[i]) + s_duz[d0 + 2*i + 1]));
            q_kr[i] = __floats2bfloat162_rn(__low2float(p_b1[i]) + __low2float(p_b2[i]) + dkr0 * bf,
                                            __high2float(p_b1[i]) + __high2float(p_b2[i]) + dkr1 * bf);
            q_kp[i] = __floats2bfloat162_rn(__low2float(p_bp[i]) + dkp0 * bf,
                                            __high2float(p_bp[i]) + dkp1 * bf);
            q_v[i]  = __floats2bfloat162_rn(dv0 * bf, dv1 * bf);
            dot = fmaf(dkr0, __low2float(p_kr[i]), fmaf(dkr1, __high2float(p_kr[i]),
                  fmaf(dv0,  __low2float(p_v[i]),  fmaf(dv1,  __high2float(p_v[i]),
                  fmaf(dkp0, __low2float(p_kp[i]), fmaf(dkp1, __high2float(p_kp[i]), dot))))));
        }
        reinterpret_cast<uint4*>(dKr_out)[off] = o_kr;
        reinterpret_cast<uint4*>(dKp_out)[off] = o_kp;
        reinterpret_cast<uint4*>(dV_out)[off]  = o_v;
    }
    dot = half_warp_sum(dot);
    if (ltid == 0) dbeta_out[ms] = dot;
}

extern "C" __global__ void gdn_cam_phase_a_epilogue_bwd_u4_bf16(
    const __mt_bfloat16* __restrict__ k,
    const __mt_bfloat16* __restrict__ v,
    const float* __restrict__ beta,
    const __mt_bfloat16* __restrict__ dK_b,
    const __mt_bfloat16* __restrict__ dV_b,
    const __mt_bfloat16* __restrict__ b1,
    const __mt_bfloat16* __restrict__ b2,
    __mt_bfloat16* __restrict__ dK_out,
    __mt_bfloat16* __restrict__ dV_out,
    float* __restrict__ dbeta_s_out,
    int S, int beta_is_4d
) {
    constexpr int D8 = 14;
    const int m = blockIdx.y, tid = threadIdx.x;
    const int grp = tid >> 4, ltid = tid & 15;
    const int s = blockIdx.x * 16 + grp;
    if (s >= S) return;
    const int64_t ms = static_cast<int64_t>(m) * S + s;
    const float bf = __bfloat162float(__float2bfloat16_rn(beta_is_4d ? beta[ms] : beta[m]));
    float dot = 0.0f;
    if (ltid < D8) {
        const int64_t off = ms * D8 + ltid;
        uint4 u_k   = reinterpret_cast<const uint4*>(k)[off];
        uint4 u_v   = reinterpret_cast<const uint4*>(v)[off];
        uint4 u_dkb = reinterpret_cast<const uint4*>(dK_b)[off];
        uint4 u_dvb = reinterpret_cast<const uint4*>(dV_b)[off];
        uint4 u_b1  = reinterpret_cast<const uint4*>(b1)[off];
        uint4 u_b2  = reinterpret_cast<const uint4*>(b2)[off];
        const __mt_bfloat162 *p_k   = reinterpret_cast<const __mt_bfloat162*>(&u_k);
        const __mt_bfloat162 *p_v   = reinterpret_cast<const __mt_bfloat162*>(&u_v);
        const __mt_bfloat162 *p_dkb = reinterpret_cast<const __mt_bfloat162*>(&u_dkb);
        const __mt_bfloat162 *p_dvb = reinterpret_cast<const __mt_bfloat162*>(&u_dvb);
        const __mt_bfloat162 *p_b1  = reinterpret_cast<const __mt_bfloat162*>(&u_b1);
        const __mt_bfloat162 *p_b2  = reinterpret_cast<const __mt_bfloat162*>(&u_b2);
        uint4 o_k, o_v;
        __mt_bfloat162 *q_k = reinterpret_cast<__mt_bfloat162*>(&o_k);
        __mt_bfloat162 *q_v = reinterpret_cast<__mt_bfloat162*>(&o_v);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float dk0 = __low2float(p_dkb[i]), dk1 = __high2float(p_dkb[i]);
            float dv0 = __low2float(p_dvb[i]), dv1 = __high2float(p_dvb[i]);
            q_k[i] = __floats2bfloat162_rn(__low2float(p_b1[i]) + __low2float(p_b2[i]) + dk0 * bf,
                                           __high2float(p_b1[i]) + __high2float(p_b2[i]) + dk1 * bf);
            q_v[i] = __floats2bfloat162_rn(dv0 * bf, dv1 * bf);
            dot = fmaf(dk0, __low2float(p_k[i]), fmaf(dk1, __high2float(p_k[i]),
                  fmaf(dv0, __low2float(p_v[i]), fmaf(dv1, __high2float(p_v[i]), dot))));
        }
        reinterpret_cast<uint4*>(dK_out)[off] = o_k;
        reinterpret_cast<uint4*>(dV_out)[off] = o_v;
    }
    dot = half_warp_sum(dot);
    if (ltid == 0) dbeta_s_out[ms] = dot;
}

extern "C" int launch_out_div_bwd_from_out_dqpos(
    const void* dout, const void* out, const void* den, const void* Sz_tot,
    void* dnum, void* dden, void* dq_pos,
    int BH, int H, int T, int S, int D, float eps, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    if (D == 112 && ((reinterpret_cast<uintptr_t>(dout) | reinterpret_cast<uintptr_t>(out) |
                      reinterpret_cast<uintptr_t>(dnum) | reinterpret_cast<uintptr_t>(dq_pos)) & 15u) == 0) {
        dim3 grid((S + 15) / 16, T, BH);
        out_div_bwd_from_out_dqpos_u4_bf16<<<grid, 256, 0, s>>>(
            reinterpret_cast<const __mt_bfloat16*>(dout),
            reinterpret_cast<const __mt_bfloat16*>(out),
            reinterpret_cast<const float*>(den),
            reinterpret_cast<const float*>(Sz_tot),
            reinterpret_cast<__mt_bfloat16*>(dnum),
            reinterpret_cast<float*>(dden),
            reinterpret_cast<__mt_bfloat16*>(dq_pos),
            H, T, S, eps
        );
        return static_cast<int>(musaGetLastError());
    }
    dim3 grid(S, T, BH);
    out_div_bwd_from_out_dqpos_bf16<<<grid, 64, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(dout),
        reinterpret_cast<const __mt_bfloat162*>(out),
        reinterpret_cast<const float*>(den),
        reinterpret_cast<const float*>(Sz_tot),
        reinterpret_cast<__mt_bfloat162*>(dnum),
        reinterpret_cast<float*>(dden),
        reinterpret_cast<__mt_bfloat162*>(dq_pos),
        H, T, S, D, eps
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_main_phase_a_scale(
    const void* kr, const void* kp, const void* v, const void* beta,
    void* kr_b, void* kp_b, void* v_b, void* U_z,
    int M, int S, int D, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    gdn_main_phase_a_scale_bf16<<<M, 224, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(kr), reinterpret_cast<const __mt_bfloat162*>(kp),
        reinterpret_cast<const __mt_bfloat162*>(v), reinterpret_cast<const float*>(beta),
        reinterpret_cast<__mt_bfloat162*>(kr_b), reinterpret_cast<__mt_bfloat162*>(kp_b),
        reinterpret_cast<__mt_bfloat162*>(v_b), reinterpret_cast<float*>(U_z),
        S, D
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_cam_phase_a_scale(
    const void* k, const void* v, const void* beta,
    void* k_b, void* v_b,
    int M, int S, int D, int beta_is_4d, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    gdn_cam_phase_a_scale_bf16<<<M, 224, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(k), reinterpret_cast<const __mt_bfloat162*>(v),
        reinterpret_cast<const float*>(beta),
        reinterpret_cast<__mt_bfloat162*>(k_b), reinterpret_cast<__mt_bfloat162*>(v_b),
        S, D, beta_is_4d
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_make_W(
    const void* P1, const void* P2, const void* decay, void* W1, void* W2,
    int M, int D, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    gdn_make_W_fwd_f32<<<M, 256, 0, s>>>(
        reinterpret_cast<const float*>(P1), reinterpret_cast<const float*>(P2),
        reinterpret_cast<const float*>(decay),
        reinterpret_cast<float*>(W1), reinterpret_cast<float*>(W2), D
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_add_cast_S(
    const void* S_fwd, const void* S_rev, const void* Sz_fwd, const void* Sz_rev,
    void* S_tot_bf, void* Sz_tot, int M, int D, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    gdn_add_cast_S_bf16<<<M, 256, 0, s>>>(
        reinterpret_cast<const float*>(S_fwd), reinterpret_cast<const float*>(S_rev),
        reinterpret_cast<const float*>(Sz_fwd), reinterpret_cast<const float*>(Sz_rev),
        reinterpret_cast<__mt_bfloat162*>(S_tot_bf), reinterpret_cast<float*>(Sz_tot), D
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_qp_sz_dot(
    const void* qp, const void* Sz_tot, void* den, int M, int S, int D, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    dim3 grid(M, (S + 7) / 8);
    gdn_qp_sz_dot_fwd_bf16<<<grid, 256, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(qp), reinterpret_cast<const float*>(Sz_tot),
        reinterpret_cast<float*>(den), S, D
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_dden_qp_bwd(
    const void* dden, const void* qp, void* dSz_tot, int M, int S, int D, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    gdn_dden_qp_bwd_bf16<<<M, 224, 0, s>>>(
        reinterpret_cast<const float*>(dden), reinterpret_cast<const __mt_bfloat162*>(qp),
        reinterpret_cast<float*>(dSz_tot), S, D
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_ddecay_dp_du_bwd(
    const void* W1, const void* dW1, const void* g_fwd, const void* g_rev,
    const void* W2, const void* dW2, const void* gz_fwd, const void* gz_rev,
    const void* decay, void* d_decay, void* dP1_bf, void* dU1_bf, void* dP2_bf, void* dU_z_out,
    int M, int T, int D, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    gdn_ddecay_dp_du_bwd_bf16<<<M, 256, 0, s>>>(
        reinterpret_cast<const float*>(W1), reinterpret_cast<const float*>(dW1),
        reinterpret_cast<const float*>(g_fwd), reinterpret_cast<const float*>(g_rev),
        reinterpret_cast<const float*>(W2), reinterpret_cast<const float*>(dW2),
        reinterpret_cast<const float*>(gz_fwd), reinterpret_cast<const float*>(gz_rev),
        reinterpret_cast<const float*>(decay), reinterpret_cast<float*>(d_decay),
        reinterpret_cast<__mt_bfloat162*>(dP1_bf), reinterpret_cast<__mt_bfloat162*>(dU1_bf),
        reinterpret_cast<__mt_bfloat162*>(dP2_bf), reinterpret_cast<float*>(dU_z_out), T, D
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_main_phase_a_epilogue_bwd(
    const void* kr, const void* kp, const void* v, const void* beta,
    const void* dKr_b, const void* dV_b, const void* dKp_b_raw, const void* dU_z,
    const void* dKr_bmm1, const void* dKr_bmm2, const void* dKp_bmm,
    void* dK_rot, void* dK_pos, void* dV, void* dbeta,
    int M, int S, int D, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    if (D == 112) {
        dim3 grid((S + 15) / 16, M);
        gdn_main_phase_a_epilogue_bwd_u4_bf16<<<grid, 256, 0, s>>>(
            reinterpret_cast<const __mt_bfloat16*>(kr), reinterpret_cast<const __mt_bfloat16*>(kp),
            reinterpret_cast<const __mt_bfloat16*>(v), reinterpret_cast<const float*>(beta),
            reinterpret_cast<const __mt_bfloat16*>(dKr_b), reinterpret_cast<const __mt_bfloat16*>(dV_b),
            reinterpret_cast<const __mt_bfloat16*>(dKp_b_raw), reinterpret_cast<const float*>(dU_z),
            reinterpret_cast<const __mt_bfloat16*>(dKr_bmm1), reinterpret_cast<const __mt_bfloat16*>(dKr_bmm2),
            reinterpret_cast<const __mt_bfloat16*>(dKp_bmm),
            reinterpret_cast<__mt_bfloat16*>(dK_rot), reinterpret_cast<__mt_bfloat16*>(dK_pos),
            reinterpret_cast<__mt_bfloat16*>(dV), reinterpret_cast<float*>(dbeta), S
        );
        return static_cast<int>(musaGetLastError());
    }
    dim3 grid((S + 3) / 4, M);
    gdn_main_phase_a_epilogue_bwd_bf16<<<grid, 256, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(kr), reinterpret_cast<const __mt_bfloat162*>(kp),
        reinterpret_cast<const __mt_bfloat162*>(v), reinterpret_cast<const float*>(beta),
        reinterpret_cast<const __mt_bfloat162*>(dKr_b), reinterpret_cast<const __mt_bfloat162*>(dV_b),
        reinterpret_cast<const __mt_bfloat162*>(dKp_b_raw), reinterpret_cast<const float*>(dU_z),
        reinterpret_cast<const __mt_bfloat162*>(dKr_bmm1), reinterpret_cast<const __mt_bfloat162*>(dKr_bmm2),
        reinterpret_cast<const __mt_bfloat162*>(dKp_bmm),
        reinterpret_cast<__mt_bfloat162*>(dK_rot), reinterpret_cast<__mt_bfloat162*>(dK_pos),
        reinterpret_cast<__mt_bfloat162*>(dV), reinterpret_cast<float*>(dbeta), S, D
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_gdn_cam_phase_a_epilogue_bwd(
    const void* k, const void* v, const void* beta,
    const void* dK_b, const void* dV_b, const void* dK_bmm1, const void* dK_bmm2,
    void* dK, void* dV, void* dbeta_s,
    int M, int S, int D, int beta_is_4d, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    if (D == 112) {
        dim3 grid((S + 15) / 16, M);
        gdn_cam_phase_a_epilogue_bwd_u4_bf16<<<grid, 256, 0, s>>>(
            reinterpret_cast<const __mt_bfloat16*>(k), reinterpret_cast<const __mt_bfloat16*>(v),
            reinterpret_cast<const float*>(beta),
            reinterpret_cast<const __mt_bfloat16*>(dK_b), reinterpret_cast<const __mt_bfloat16*>(dV_b),
            reinterpret_cast<const __mt_bfloat16*>(dK_bmm1), reinterpret_cast<const __mt_bfloat16*>(dK_bmm2),
            reinterpret_cast<__mt_bfloat16*>(dK), reinterpret_cast<__mt_bfloat16*>(dV),
            reinterpret_cast<float*>(dbeta_s), S, beta_is_4d
        );
        return static_cast<int>(musaGetLastError());
    }
    dim3 grid((S + 3) / 4, M);
    gdn_cam_phase_a_epilogue_bwd_bf16<<<grid, 256, 0, s>>>(
        reinterpret_cast<const __mt_bfloat162*>(k), reinterpret_cast<const __mt_bfloat162*>(v),
        reinterpret_cast<const float*>(beta),
        reinterpret_cast<const __mt_bfloat162*>(dK_b), reinterpret_cast<const __mt_bfloat162*>(dV_b),
        reinterpret_cast<const __mt_bfloat162*>(dK_bmm1), reinterpret_cast<const __mt_bfloat162*>(dK_bmm2),
        reinterpret_cast<__mt_bfloat162*>(dK), reinterpret_cast<__mt_bfloat162*>(dV),
        reinterpret_cast<float*>(dbeta_s), S, D, beta_is_4d
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_main_gdn_prep_scale_fwd_u4(
    const void* qkv, const void* kconv, const void* q_nw, const void* k_nw,
    const void* rcos, const void* rsin, const void* beta,
    void* q_rot, void* q_pos, void* k_rot, void* k_pos,
    void* kr_b, void* kp_b, void* v_b, void* v_out, void* U_z,
    void* q_ir, void* k_ir,
    int B, int T, int S, float k_scale, float norm_eps, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    int N = T * S;
    main_gdn_prep_scale_fwd_u4_bf16<<<B * N, 288, 0, s>>>(
        reinterpret_cast<const __mt_bfloat16*>(qkv), reinterpret_cast<const __mt_bfloat16*>(kconv),
        reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
        reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
        reinterpret_cast<const float*>(beta),
        reinterpret_cast<__mt_bfloat16*>(q_rot), reinterpret_cast<__mt_bfloat16*>(q_pos),
        reinterpret_cast<__mt_bfloat16*>(k_rot), reinterpret_cast<__mt_bfloat16*>(k_pos),
        reinterpret_cast<__mt_bfloat16*>(kr_b), reinterpret_cast<__mt_bfloat16*>(kp_b),
        reinterpret_cast<__mt_bfloat16*>(v_b), reinterpret_cast<__mt_bfloat16*>(v_out),
        reinterpret_cast<float*>(q_ir), reinterpret_cast<float*>(k_ir),
        B, N, k_scale, norm_eps
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    if (U_z != nullptr) {
        int M = B * 20 * T;
        reduce_kp_b_to_uz_u4_bf16<<<M, 256, 0, s>>>(
            reinterpret_cast<const __mt_bfloat16*>(kp_b), reinterpret_cast<float*>(U_z), S
        );
        err = musaGetLastError();
    }
    return static_cast<int>(err);
}

extern "C" int launch_cam_prep_fwd_u4(
    const void* q, const void* k, const void* v,
    const void* q_nw, const void* k_nw,
    const void* P_T, const void* P_inv, const void* rcos, const void* rsin,
    const void* beta_opt,
    void* q_out, void* k_out, void* v_out, void* k_b_out, void* v_b_out,
    void* inf_sq, void* q_ir, void* k_ir, void* pre_ksq, void* post_ksq,
    int B, int T, int S, int beta_is_4d, float k_scale, float norm_eps, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    cam_prep_fwd_u4_bf16<<<B * T * S, 288, 0, s>>>(
        reinterpret_cast<const __mt_bfloat16*>(q), reinterpret_cast<const __mt_bfloat16*>(k),
        reinterpret_cast<const __mt_bfloat16*>(v),
        reinterpret_cast<const float*>(q_nw), reinterpret_cast<const float*>(k_nw),
        reinterpret_cast<const float*>(P_T), reinterpret_cast<const float*>(P_inv),
        reinterpret_cast<const float*>(rcos), reinterpret_cast<const float*>(rsin),
        reinterpret_cast<const float*>(beta_opt),
        reinterpret_cast<__mt_bfloat16*>(q_out), reinterpret_cast<__mt_bfloat16*>(k_out),
        reinterpret_cast<__mt_bfloat16*>(v_out), reinterpret_cast<__mt_bfloat16*>(k_b_out),
        reinterpret_cast<__mt_bfloat16*>(v_b_out),
        reinterpret_cast<float*>(inf_sq), reinterpret_cast<float*>(q_ir), reinterpret_cast<float*>(k_ir),
        reinterpret_cast<float*>(pre_ksq), reinterpret_cast<float*>(post_ksq),
        B, T, S, beta_is_4d, k_scale, norm_eps
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_cam_phase_a_scale_fused_u4(
    const void* k, const void* v, const void* inf_sq, const void* beta_raw,
    void* k_b, void* v_b, void* beta_m_out, void* frame_inf_out,
    int M, int S, int beta_is_4d, void* stream_ptr
) {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream_ptr);
    cam_phase_a_scale_fused_u4_bf16<<<M, 256, 0, s>>>(
        reinterpret_cast<const __mt_bfloat16*>(k), reinterpret_cast<const __mt_bfloat16*>(v),
        reinterpret_cast<const float*>(inf_sq), reinterpret_cast<const float*>(beta_raw),
        reinterpret_cast<__mt_bfloat16*>(k_b), reinterpret_cast<__mt_bfloat16*>(v_b),
        reinterpret_cast<float*>(beta_m_out), reinterpret_cast<float*>(frame_inf_out),
        S, beta_is_4d
    );
    return static_cast<int>(musaGetLastError());
}
"""


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    src_hash = hashlib.sha256(_MUSA_SRC.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"sana_musa_kernels_{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"bidi_gdn_s5000_{src_hash}.so"
    if not so_path.exists():
        fd_src, tmp_cu = tempfile.mkstemp(prefix="bigdn_", suffix=".cu", dir=str(cache_dir))
        os.close(fd_src)
        tmp_so = str(so_path) + f".tmp.{os.getpid()}"
        try:
            Path(tmp_cu).write_text(_MUSA_SRC)
            cmd = [
                _MCC_PATH,
                f"--musa-path={_MUSA_PATH}",
                "-x", "musa",
                "--offload-arch=mp_31",
                "-O3",
                "-fmusa-flush-denormals-to-zero",
                "-shared",
                "-Xcompiler", "-fPIC",
                tmp_cu,
                f"-L{_MUSA_PATH}/lib",
                "-lmusart",
                "-o", tmp_so,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.replace(tmp_so, so_path)
        finally:
            if os.path.exists(tmp_cu):
                os.unlink(tmp_cu)
            if os.path.exists(tmp_so):
                os.unlink(tmp_so)

    lib = ctypes.CDLL(str(so_path))
    p = ctypes.c_void_p
    i = ctypes.c_int
    f = ctypes.c_float

    lib.launch_bidi_short_conv4_fwd.argtypes = [p, p, p, p, i, i, i, i, i, p]
    lib.launch_bidi_short_conv4_fwd.restype = i
    lib.launch_bidi_short_conv4_bwd.argtypes = [p, p, p, p, p, p, p, p, i, i, i, i, i, p]
    lib.launch_bidi_short_conv4_bwd.restype = i

    lib.launch_main_gdn_prep_fwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, f, f, i, i, p]
    lib.launch_main_gdn_prep_fwd.restype = i
    lib.launch_main_gdn_prep_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, f, i, i, i, p]
    lib.launch_main_gdn_prep_bwd.restype = i

    lib.launch_bidi_state_scan_fwd.argtypes = [p, p, p, p, i, i, p]
    lib.launch_bidi_state_scan_fwd.restype = i
    lib.launch_bidi_state_scan_bwd.argtypes = [p, p, p, p, i, i, p]
    lib.launch_bidi_state_scan_bwd.restype = i
    lib.launch_bidi_z_scan_fwd.argtypes = [p, p, p, p, i, i, p]
    lib.launch_bidi_z_scan_fwd.restype = i
    lib.launch_bidi_z_scan_bwd.argtypes = [p, p, p, p, i, i, p]
    lib.launch_bidi_z_scan_bwd.restype = i

    lib.launch_out_div_fwd.argtypes = [p, p, p, i, i, i, i, f, i, p]
    lib.launch_out_div_fwd.restype = i
    lib.launch_out_div_bwd.argtypes = [p, p, p, p, p, i, i, i, i, f, i, p]
    lib.launch_out_div_bwd.restype = i

    lib.launch_cam_prep_fwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, f, f, i, p]
    lib.launch_cam_prep_fwd.restype = i
    lib.launch_cam_prep_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, f, i, i, p]
    lib.launch_cam_prep_bwd.restype = i

    lib.launch_cam_out_apply_o_fwd.argtypes = [p, p, p, p, p, i, i, i, i, i, p]
    lib.launch_cam_out_apply_o_fwd.restype = i
    lib.launch_cam_out_apply_o_bwd.argtypes = [p, p, p, p, p, i, i, i, i, i, p]
    lib.launch_cam_out_apply_o_bwd.restype = i

    lib.launch_bidi_short_conv4_qkv_k_fwd.argtypes = [p, p, p, p, i, i, i, i, i, p]
    lib.launch_bidi_short_conv4_qkv_k_fwd.restype = i
    lib.launch_bidi_short_conv4_qkv_k_bwd.argtypes = [p, p, p, p, p, p, p, p, i, i, i, i, i, p]
    lib.launch_bidi_short_conv4_qkv_k_bwd.restype = i

    lib.launch_out_div_bwd_from_out_dqpos.argtypes = [p, p, p, p, p, p, p, i, i, i, i, i, f, p]
    lib.launch_out_div_bwd_from_out_dqpos.restype = i
    lib.launch_gdn_main_phase_a_scale.argtypes = [p, p, p, p, p, p, p, p, i, i, i, p]
    lib.launch_gdn_main_phase_a_scale.restype = i
    lib.launch_gdn_cam_phase_a_scale.argtypes = [p, p, p, p, p, i, i, i, i, p]
    lib.launch_gdn_cam_phase_a_scale.restype = i
    lib.launch_gdn_make_W.argtypes = [p, p, p, p, p, i, i, p]
    lib.launch_gdn_make_W.restype = i
    lib.launch_gdn_add_cast_S.argtypes = [p, p, p, p, p, p, i, i, p]
    lib.launch_gdn_add_cast_S.restype = i
    lib.launch_gdn_qp_sz_dot.argtypes = [p, p, p, i, i, i, p]
    lib.launch_gdn_qp_sz_dot.restype = i
    lib.launch_gdn_dden_qp_bwd.argtypes = [p, p, p, i, i, i, p]
    lib.launch_gdn_dden_qp_bwd.restype = i
    lib.launch_gdn_ddecay_dp_du_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, p]
    lib.launch_gdn_ddecay_dp_du_bwd.restype = i
    lib.launch_gdn_main_phase_a_epilogue_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, p]
    lib.launch_gdn_main_phase_a_epilogue_bwd.restype = i
    lib.launch_gdn_cam_phase_a_epilogue_bwd.argtypes = [p, p, p, p, p, p, p, p, p, p, i, i, i, i, p]
    lib.launch_gdn_cam_phase_a_epilogue_bwd.restype = i
    lib.launch_main_gdn_prep_scale_fwd_u4.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, f, f, p]
    lib.launch_main_gdn_prep_scale_fwd_u4.restype = i
    lib.launch_cam_prep_fwd_u4.argtypes = [p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, i, i, i, i, f, f, p]
    lib.launch_cam_prep_fwd_u4.restype = i
    lib.launch_cam_phase_a_scale_fused_u4.argtypes = [p, p, p, p, p, p, p, p, i, i, i, p]
    lib.launch_cam_phase_a_scale_fused_u4.restype = i

    _LIB = lib
    return _LIB


def _stream(device) -> int:
    return torch.musa.current_stream(device).musa_stream


def _eye_cache(D: int, device, dtype):
    key = (D, device, dtype)
    t = _EYE_DICT.get(key)
    if t is None:
        t = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)
        _EYE_DICT[key] = t
    return t


_EYE_DICT: dict = {}


# ---------------------------------------------------------------------------
# Autograd Function 1: Bidirectional ShortConv (K=4) directly on (B, N, C)
# ---------------------------------------------------------------------------
class _BidiShortConv4Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, T: int, S: int):
        x_c = x.contiguous()
        w_c = weight.contiguous()
        b_c = bias.contiguous() if bias is not None else None
        B, N, C = x_c.shape
        y = torch.empty_like(x_c)
        rc = _load_lib().launch_bidi_short_conv4_fwd(
            x_c.data_ptr(), w_c.data_ptr(), b_c.data_ptr() if b_c is not None else 0, y.data_ptr(),
            B, T, S, C, 1 if b_c is not None else 0, _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"bidi_short_conv4_fwd failed rc={rc}")
        ctx.save_for_backward(x_c, w_c)
        ctx.meta = (B, T, S, C, b_c is not None, weight.shape, b_c.shape if b_c is not None else None)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, w_c = ctx.saved_tensors
        B, T, S, C, has_bias, w_shape, b_shape = ctx.meta
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        n_bs_blk = (B * S + 7) // 8
        dw_part = torch.empty((n_bs_blk, C, 4), device=x_c.device, dtype=torch.float32)
        dw_out = torch.empty(w_shape, device=x_c.device, dtype=w_c.dtype)
        db_part = torch.empty((n_bs_blk, C), device=x_c.device, dtype=torch.float32) if has_bias else None
        db_out = torch.empty(b_shape, device=x_c.device, dtype=w_c.dtype) if has_bias else None
        rc = _load_lib().launch_bidi_short_conv4_bwd(
            x_c.data_ptr(), dy_c.data_ptr(), w_c.data_ptr(), dx.data_ptr(),
            dw_part.data_ptr(), db_part.data_ptr() if has_bias else 0,
            dw_out.data_ptr(), db_out.data_ptr() if has_bias else 0,
            B, T, S, C, 1 if has_bias else 0, _stream(x_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"bidi_short_conv4_bwd failed rc={rc}")
        return dx, dw_out, db_out, None, None


def fused_bidi_short_conv4(x: torch.Tensor, conv, HW: tuple[int, int, int]) -> torch.Tensor:
    T, H_s, W_s = HW
    return _BidiShortConv4Fn.apply(x, conv.weight, conv.bias, T, H_s * W_s)


# ---------------------------------------------------------------------------
# Autograd Function 2: Main GDN Prep
# ---------------------------------------------------------------------------
class _FusedMainPrepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, k_conv, q_nw, k_nw, rope_cos, rope_sin, k_scale: float, norm_eps: float, use_bf16: bool):
        # qkv: (B, N, 3, H, D) BF16; k_conv: (B, N, C) BF16 or None
        qkv_c = qkv.contiguous()
        kc_c = k_conv.contiguous() if k_conv is not None else None
        B, N, _, H, D = qkv_c.shape
        BH = B * H
        has_norm = q_nw is not None
        q_nw_f = q_nw.float().contiguous() if has_norm else None
        k_nw_f = k_nw.float().contiguous() if has_norm else None
        rc_c = rope_cos.float().contiguous()
        rs_c = rope_sin.float().contiguous()

        out_dt = torch.bfloat16 if use_bf16 else torch.float32
        q_rot = torch.empty((BH, N, D), device=qkv_c.device, dtype=out_dt)
        q_pos = torch.empty((BH, N, D), device=qkv_c.device, dtype=out_dt)
        k_rot = torch.empty((BH, N, D), device=qkv_c.device, dtype=out_dt)
        k_pos = torch.empty((BH, N, D), device=qkv_c.device, dtype=out_dt)
        v_out = torch.empty((BH, N, D), device=qkv_c.device, dtype=out_dt)
        q_ir = torch.empty((B * N,), device=qkv_c.device, dtype=torch.float32)
        k_ir = torch.empty((B * N,), device=qkv_c.device, dtype=torch.float32)

        rc = _load_lib().launch_main_gdn_prep_fwd(
            qkv_c.data_ptr(), kc_c.data_ptr() if kc_c is not None else 0,
            q_nw_f.data_ptr() if has_norm else 0, k_nw_f.data_ptr() if has_norm else 0,
            rc_c.data_ptr(), rs_c.data_ptr(),
            q_rot.data_ptr(), q_pos.data_ptr(), k_rot.data_ptr(), k_pos.data_ptr(), v_out.data_ptr(),
            q_ir.data_ptr(), k_ir.data_ptr(),
            B, N, H, D, float(k_scale), float(norm_eps), 1 if has_norm else 0, 1 if use_bf16 else 0,
            _stream(qkv_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"main_gdn_prep_fwd failed rc={rc}")

        empty = torch.empty(0, device=qkv_c.device, dtype=torch.float32)
        ctx.save_for_backward(
            qkv_c,
            kc_c if kc_c is not None else empty,
            q_nw_f if has_norm else empty,
            k_nw_f if has_norm else empty,
            q_ir, k_ir, rc_c, rs_c,
        )
        ctx.meta = (B, N, H, D, float(k_scale), has_norm, kc_c is not None, use_bf16,
                    q_nw.dtype if has_norm else None, k_nw.dtype if has_norm else None)
        return q_rot, q_pos, k_rot, k_pos, v_out

    @staticmethod
    def backward(ctx, dq_rot, dq_pos, dk_rot, dk_pos, dv):
        qkv_c, kc_saved, q_nw_f, k_nw_f, q_ir, k_ir, rc_c, rs_c = ctx.saved_tensors
        B, N, H, D, k_scale, has_norm, has_kconv, use_bf16, q_nw_dt, k_nw_dt = ctx.meta
        C = H * D
        dq_rot_c = dq_rot.contiguous()
        dq_pos_c = dq_pos.contiguous()
        dk_rot_c = dk_rot.contiguous()
        dk_pos_c = dk_pos.contiguous()
        dv_c = dv.contiguous()

        dqkv = torch.empty_like(qkv_c)
        dkconv = torch.empty_like(kc_saved) if has_kconv else None
        n_chunks = 128
        dq_nw_part = torch.empty((n_chunks, C), device=qkv_c.device, dtype=torch.float32) if has_norm else None
        dk_nw_part = torch.empty((n_chunks, C), device=qkv_c.device, dtype=torch.float32) if has_norm else None
        dq_nw_out = torch.empty((C,), device=qkv_c.device, dtype=torch.float32) if has_norm else None
        dk_nw_out = torch.empty((C,), device=qkv_c.device, dtype=torch.float32) if has_norm else None

        rc = _load_lib().launch_main_gdn_prep_bwd(
            qkv_c.data_ptr(), kc_saved.data_ptr() if has_kconv else 0,
            q_nw_f.data_ptr() if has_norm else 0, k_nw_f.data_ptr() if has_norm else 0,
            q_ir.data_ptr(), k_ir.data_ptr(), rc_c.data_ptr(), rs_c.data_ptr(),
            dq_rot_c.data_ptr(), dq_pos_c.data_ptr(), dk_rot_c.data_ptr(), dk_pos_c.data_ptr(), dv_c.data_ptr(),
            dqkv.data_ptr(), dkconv.data_ptr() if has_kconv else 0,
            dq_nw_part.data_ptr() if has_norm else 0, dk_nw_part.data_ptr() if has_norm else 0,
            dq_nw_out.data_ptr() if has_norm else 0, dk_nw_out.data_ptr() if has_norm else 0,
            B, N, H, D, k_scale, 1 if has_norm else 0, 1 if use_bf16 else 0, n_chunks,
            _stream(qkv_c.device),
        )
        if rc != 0:
            raise RuntimeError(f"main_gdn_prep_bwd failed rc={rc}")

        dq_nw_ret = dq_nw_out.to(q_nw_dt) if has_norm else None
        dk_nw_ret = dk_nw_out.to(k_nw_dt) if has_norm else None
        return dqkv, dkconv, dq_nw_ret, dk_nw_ret, None, None, None, None, None


# ---------------------------------------------------------------------------
# Autograd Function 3: Bidirectional Main GDN Scan (Phase A + B + C + OutDiv)
# ---------------------------------------------------------------------------
class _FusedBidiMainScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_rot, q_pos, k_rot, k_pos, v, beta, decay, B: int, H: int, T: int, S: int, eps: float):
        # q_rot, q_pos, k_rot, k_pos, v: (BH, N, D) [fp32 or bf16]; beta: (B, H, T, S) fp32; decay: (B, H, T) fp32
        BH = B * H
        D = q_rot.shape[-1]
        M = BH * T
        dt = q_rot.dtype
        use_bf16 = (dt == torch.bfloat16)
        dev = q_rot.device

        qr = q_rot.view(M, S, D)
        qp = q_pos.view(M, S, D)
        kr = k_rot.view(M, S, D)
        kp = k_pos.view(M, S, D)
        vv = v.view(M, S, D)
        beta_m = beta.float().reshape(M, S, 1)
        decay_m = decay.float().reshape(M, 1, 1)

        # Phase A (computed ONCE for both directions!)
        beta_dt = beta_m.to(dt) if use_bf16 else beta_m
        kr_b = kr * beta_dt
        kp_b = kp * beta_dt
        v_b  = vv * beta_dt

        P_kv = torch.bmm(kr_b.transpose(-1, -2), kr).float()   # (M, D, D)
        U_kv = torch.bmm(v_b.transpose(-1, -2),  kr).float()   # (M, D, D)
        P_z  = torch.bmm(kp_b.transpose(-1, -2), kp).float()   # (M, D, D)
        U_z  = kp_b.float().sum(dim=1)                         # (M, D)

        I_D = _eye_cache(D, dev, torch.float32)
        I_m_Pkv = I_D - P_kv
        I_m_Pz  = I_D - P_z
        W_kv = (decay_m * I_m_Pkv).contiguous()
        W_z  = (decay_m * I_m_Pz).contiguous()
        U_kv = U_kv.contiguous()
        U_z  = U_z.contiguous()

        # Phase B (fused MUSA C++ bidirectional scan!)
        S_fwd = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        S_rev = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        Sz_fwd = torch.empty((BH, T, D), device=dev, dtype=torch.float32)
        Sz_rev = torch.empty((BH, T, D), device=dev, dtype=torch.float32)

        lib = _load_lib()
        st = _stream(dev)
        rc = lib.launch_bidi_state_scan_fwd(W_kv.data_ptr(), U_kv.data_ptr(), S_fwd.data_ptr(), S_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"bidi_state_scan_fwd failed rc={rc}")
        rc = lib.launch_bidi_z_scan_fwd(W_z.data_ptr(), U_z.data_ptr(), Sz_fwd.data_ptr(), Sz_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"bidi_z_scan_fwd failed rc={rc}")

        S_tot = (S_fwd + S_rev).view(M, D, D)
        Sz_tot = (Sz_fwd + Sz_rev).view(M, D)

        # Phase C (computed ONCE on S_tot = S_fwd + S_rev!)
        S_tot_dt = S_tot.to(dt) if use_bf16 else S_tot
        num = torch.bmm(qr, S_tot_dt.transpose(-1, -2))                      # (M, S, D)
        den = torch.bmm(qp.float(), Sz_tot.unsqueeze(-1)).squeeze(-1).contiguous()  # (M, S) fp32

        out = torch.empty((B, T * S, H * D), device=dev, dtype=torch.bfloat16)
        rc = lib.launch_out_div_fwd(
            num.data_ptr(), den.data_ptr(), out.data_ptr(),
            B, T * S, H, D, float(eps), 1 if use_bf16 else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"out_div_fwd failed rc={rc}")

        ctx.save_for_backward(
            q_rot, q_pos, k_rot, k_pos, v, beta_m, decay_m,
            I_m_Pkv, I_m_Pz, W_kv, W_z, S_fwd, S_rev, Sz_fwd, Sz_rev, num, den,
        )
        ctx.meta = (B, H, T, S, D, float(eps), use_bf16, beta.dtype, decay.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        (q_rot, q_pos, k_rot, k_pos, v, beta_m, decay_m,
         I_m_Pkv, I_m_Pz, W_kv, W_z, S_fwd, S_rev, Sz_fwd, Sz_rev, num, den) = ctx.saved_tensors
        B, H, T, S, D, eps, use_bf16, beta_orig_dt, decay_orig_dt = ctx.meta
        BH = B * H
        M = BH * T
        dt = q_rot.dtype
        dev = q_rot.device
        lib = _load_lib()
        st = _stream(dev)

        dout_c = dout.contiguous()
        dnum = torch.empty_like(num)
        dden = torch.empty_like(den)
        rc = lib.launch_out_div_bwd(
            dout_c.data_ptr(), num.data_ptr(), den.data_ptr(), dnum.data_ptr(), dden.data_ptr(),
            B, T * S, H, D, eps, 1 if use_bf16 else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"out_div_bwd failed rc={rc}")

        qr = q_rot.view(M, S, D)
        qp = q_pos.view(M, S, D)
        kr = k_rot.view(M, S, D)
        kp = k_pos.view(M, S, D)
        vv = v.view(M, S, D)

        S_tot = (S_fwd + S_rev).view(M, D, D)
        Sz_tot = (Sz_fwd + Sz_rev).view(M, D)

        # Phase C_bwd
        S_tot_dt = S_tot.to(dt) if use_bf16 else S_tot
        dQ_rot = torch.bmm(dnum, S_tot_dt)                                 # (M, S, D)
        dS_tot = torch.bmm(dnum.transpose(-1, -2), qr).float().contiguous()# (M, D, D)
        dQ_pos = (dden.unsqueeze(-1) * Sz_tot.unsqueeze(1)).to(dt)         # (M, S, D)
        dSz_tot = torch.bmm(dden.unsqueeze(1), qp.float()).squeeze(1).contiguous()  # (M, D)

        # Phase B_bwd (fused MUSA C++ bidirectional backward scan!)
        g_fwd = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        g_rev = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        gz_fwd = torch.empty((BH, T, D), device=dev, dtype=torch.float32)
        gz_rev = torch.empty((BH, T, D), device=dev, dtype=torch.float32)

        rc = lib.launch_bidi_state_scan_bwd(W_kv.data_ptr(), dS_tot.data_ptr(), g_fwd.data_ptr(), g_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"bidi_state_scan_bwd failed rc={rc}")
        rc = lib.launch_bidi_z_scan_bwd(W_z.data_ptr(), dSz_tot.data_ptr(), gz_fwd.data_ptr(), gz_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"bidi_z_scan_bwd failed rc={rc}")

        # Combine fwd + rev contributions to dU_kv, dW_kv, dU_z, dW_z
        dU_kv = g_fwd.clone()
        dU_kv[:, 1:].add_(g_rev[:, :-1])
        dW_kv = torch.zeros_like(g_fwd)
        if T > 1:
            n_m1 = BH * (T - 1)
            dW_kv[:, 1:] = (
                torch.bmm(S_fwd[:, :-1].reshape(n_m1, D, D).transpose(-1, -2), g_fwd[:, 1:].reshape(n_m1, D, D))
                + torch.bmm(S_rev[:, 1:].reshape(n_m1, D, D).transpose(-1, -2), g_rev[:, :-1].reshape(n_m1, D, D))
            ).view(BH, T - 1, D, D)

        dU_z = gz_fwd.clone()
        dU_z[:, 1:].add_(gz_rev[:, :-1])
        dW_z = torch.zeros_like(g_fwd)
        if T > 1:
            dW_z[:, 1:] = (
                Sz_fwd[:, :-1].unsqueeze(-1) * gz_fwd[:, 1:].unsqueeze(-2)
                + Sz_rev[:, 1:].unsqueeze(-1) * gz_rev[:, :-1].unsqueeze(-2)
            )

        dW_kv_m = dW_kv.view(M, D, D)
        dU_kv_m = dU_kv.view(M, D, D)
        dW_z_m  = dW_z.view(M, D, D)
        dU_z_m  = dU_z.view(M, D)

        # Phase A_bwd
        d_decay = (dW_kv_m * I_m_Pkv).sum(dim=(-2, -1)) + (dW_z_m * I_m_Pz).sum(dim=(-2, -1))
        dP_kv = (-decay_m * dW_kv_m)
        dP_z  = (-decay_m * dW_z_m)

        dP_kv_dt = dP_kv.to(dt) if use_bf16 else dP_kv
        dU_kv_dt = dU_kv_m.to(dt) if use_bf16 else dU_kv_m
        dP_z_dt  = dP_z.to(dt) if use_bf16 else dP_z
        beta_dt  = beta_m.to(dt) if use_bf16 else beta_m

        kr_b = kr * beta_dt
        kp_b = kp * beta_dt
        v_b  = vv * beta_dt

        dKr_b = torch.bmm(kr, dP_kv_dt.transpose(-1, -2))
        dV_b  = torch.bmm(kr, dU_kv_dt.transpose(-1, -2))
        dK_rot = torch.bmm(kr_b, dP_kv_dt) + torch.bmm(v_b, dU_kv_dt) + dKr_b * beta_dt
        dV     = dV_b * beta_dt

        dKp_b = torch.bmm(kp, dP_z_dt.transpose(-1, -2)) + dU_z_m.to(dt).unsqueeze(1)
        dK_pos = torch.bmm(kp_b, dP_z_dt) + dKp_b * beta_dt

        dbeta = (
            (dKr_b.float() * kr.float()).sum(dim=-1)
            + (dV_b.float() * vv.float()).sum(dim=-1)
            + (dKp_b.float() * kp.float()).sum(dim=-1)
        )

        return (
            dQ_rot.view(BH, T * S, D),
            dQ_pos.view(BH, T * S, D),
            dK_rot.view(BH, T * S, D),
            dK_pos.view(BH, T * S, D),
            dV.view(BH, T * S, D),
            dbeta.view(B, H, T, S).to(beta_orig_dt),
            d_decay.view(B, H, T).to(decay_orig_dt),
            None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Autograd Function 4: Fused Cam Prep
# ---------------------------------------------------------------------------
class _FusedCamPrepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_raw, k_raw, v_raw, q_nw, k_nw, P_T, P_inv, rope_cos, rope_sin,
                H: int, D: int, k_scale: float, norm_eps: float, use_bf16: bool):
        q_c = q_raw.contiguous()
        k_c = k_raw.contiguous()
        v_c = v_raw.contiguous()
        q_nw_f = q_nw.float().contiguous()
        k_nw_f = k_nw.float().contiguous()
        pt_f = P_T.float().contiguous()
        pinv_f = P_inv.float().contiguous()
        rc_f = rope_cos.float().contiguous()
        rs_f = rope_sin.float().contiguous()
        B, N, _ = q_c.shape
        BH = B * H
        dev = q_c.device
        out_dt = torch.bfloat16 if use_bf16 else torch.float32

        q_out = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        k_out = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        v_out = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        inf_sq = torch.empty((BH, N), device=dev, dtype=torch.float32)
        q_ir = torch.empty((B * N,), device=dev, dtype=torch.float32)
        k_ir = torch.empty((B * N,), device=dev, dtype=torch.float32)
        pre_ksq = torch.empty((BH, N), device=dev, dtype=torch.float32)
        post_ksq = torch.empty((BH, N), device=dev, dtype=torch.float32)

        rc = _load_lib().launch_cam_prep_fwd(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(),
            q_nw_f.data_ptr(), k_nw_f.data_ptr(),
            pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
            q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), inf_sq.data_ptr(),
            q_ir.data_ptr(), k_ir.data_ptr(), pre_ksq.data_ptr(), post_ksq.data_ptr(),
            B, N, H, D, float(k_scale), float(norm_eps), 1 if use_bf16 else 0, _stream(dev),
        )
        if rc != 0:
            raise RuntimeError(f"cam_prep_fwd failed rc={rc}")

        ctx.save_for_backward(q_c, k_c, q_nw_f, k_nw_f, pt_f, pinv_f, rc_f, rs_f, q_ir, k_ir, pre_ksq, post_ksq)
        ctx.meta = (B, N, H, D, float(k_scale), use_bf16, q_nw.dtype, k_nw.dtype)
        return q_out, k_out, v_out, inf_sq

    @staticmethod
    def backward(ctx, dq_out, dk_out, dv_out, d_inf_sq):
        q_c, k_c, q_nw_f, k_nw_f, pt_f, pinv_f, rc_f, rs_f, q_ir, k_ir, pre_ksq, post_ksq = ctx.saved_tensors
        B, N, H, D, k_scale, use_bf16, q_nw_dt, k_nw_dt = ctx.meta
        C = H * D
        dev = q_c.device
        dq_c = dq_out.contiguous()
        dk_c = dk_out.contiguous()
        dv_c = dv_out.contiguous()
        dinf_c = d_inf_sq.float().contiguous()

        dq_raw = torch.empty_like(q_c)
        dk_raw = torch.empty_like(k_c)
        dv_raw = torch.empty_like(q_c)
        n_chunks = 128
        dq_nw_part = torch.empty((n_chunks, C), device=dev, dtype=torch.float32)
        dk_nw_part = torch.empty((n_chunks, C), device=dev, dtype=torch.float32)
        dq_nw_out = torch.empty((C,), device=dev, dtype=torch.float32)
        dk_nw_out = torch.empty((C,), device=dev, dtype=torch.float32)

        rc = _load_lib().launch_cam_prep_bwd(
            q_c.data_ptr(), k_c.data_ptr(), q_nw_f.data_ptr(), k_nw_f.data_ptr(),
            pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
            q_ir.data_ptr(), k_ir.data_ptr(), pre_ksq.data_ptr(), post_ksq.data_ptr(),
            dq_c.data_ptr(), dk_c.data_ptr(), dv_c.data_ptr(), dinf_c.data_ptr(),
            dq_raw.data_ptr(), dk_raw.data_ptr(), dv_raw.data_ptr(),
            dq_nw_part.data_ptr(), dk_nw_part.data_ptr(), dq_nw_out.data_ptr(), dk_nw_out.data_ptr(),
            B, N, H, D, k_scale, 1 if use_bf16 else 0, n_chunks, _stream(dev),
        )
        if rc != 0:
            raise RuntimeError(f"cam_prep_bwd failed rc={rc}")
        return (
            dq_raw, dk_raw, dv_raw,
            dq_nw_out.to(q_nw_dt), dk_nw_out.to(k_nw_dt),
            None, None, None, None, None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Autograd Function 5: Fused Bidirectional Cam Scan + Fused apply_fn_o
# ---------------------------------------------------------------------------
class _FusedBidiCamScanAndOutFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, decay, P, rope_cos, rope_sin, B: int, H: int, T: int, S: int):
        # q, k, v: (BH, N, D) [fp32 or bf16]; beta: (B, H, T, S) or (B, H, T); decay: (B, H, T)
        BH = B * H
        D = q.shape[-1]
        M = BH * T
        dt = q.dtype
        use_bf16 = (dt == torch.bfloat16)
        dev = q.device

        qq = q.view(M, S, D)
        kk = k.view(M, S, D)
        vv = v.view(M, S, D)
        beta_is_4d = (beta.ndim == 4)
        beta_m = beta.float().reshape(M, S, 1) if beta_is_4d else beta.float().reshape(M, 1, 1)
        decay_m = decay.float().reshape(M, 1, 1)
        beta_dt = beta_m.to(dt) if use_bf16 else beta_m

        k_b = kk * beta_dt
        v_b = vv * beta_dt
        P_cam = torch.bmm(k_b.transpose(-1, -2), kk).float()
        U_cam = torch.bmm(v_b.transpose(-1, -2), kk).float().contiguous()
        I_D = _eye_cache(D, dev, torch.float32)
        I_m_P = I_D - P_cam
        W_cam = (decay_m * I_m_P).contiguous()

        S_fwd = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        S_rev = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        lib = _load_lib()
        st = _stream(dev)
        rc = lib.launch_bidi_state_scan_fwd(W_cam.data_ptr(), U_cam.data_ptr(), S_fwd.data_ptr(), S_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"cam bidi_state_scan_fwd failed rc={rc}")

        S_tot = (S_fwd + S_rev).view(M, D, D)
        S_tot_dt = S_tot.to(dt) if use_bf16 else S_tot
        raw_out = torch.bmm(qq, S_tot_dt.transpose(-1, -2)).contiguous()  # (M, S, D)

        P_f = P.float().contiguous()
        rc_f = rope_cos.float().contiguous()
        rs_f = rope_sin.float().contiguous()
        out = torch.empty((B, T * S, H * D), device=dev, dtype=torch.bfloat16)
        rc = lib.launch_cam_out_apply_o_fwd(
            raw_out.data_ptr(), P_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(), out.data_ptr(),
            B, T * S, H, D, 1 if use_bf16 else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"cam_out_apply_o_fwd failed rc={rc}")

        ctx.save_for_backward(q, k, v, beta_m, decay_m, I_m_P, W_cam, S_fwd, S_rev, P_f, rc_f, rs_f)
        ctx.meta = (B, H, T, S, D, use_bf16, beta_is_4d, beta.dtype, decay.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, beta_m, decay_m, I_m_P, W_cam, S_fwd, S_rev, P_f, rc_f, rs_f = ctx.saved_tensors
        B, H, T, S, D, use_bf16, beta_is_4d, beta_orig_dt, decay_orig_dt = ctx.meta
        BH = B * H
        M = BH * T
        dt = q.dtype
        dev = q.device
        lib = _load_lib()
        st = _stream(dev)

        dout_c = dout.contiguous()
        d_raw = torch.empty((M, S, D), device=dev, dtype=dt)
        rc = lib.launch_cam_out_apply_o_bwd(
            dout_c.data_ptr(), P_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(), d_raw.data_ptr(),
            B, T * S, H, D, 1 if use_bf16 else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"cam_out_apply_o_bwd failed rc={rc}")

        qq = q.view(M, S, D)
        kk = k.view(M, S, D)
        vv = v.view(M, S, D)
        S_tot = (S_fwd + S_rev).view(M, D, D)
        S_tot_dt = S_tot.to(dt) if use_bf16 else S_tot

        dQ = torch.bmm(d_raw, S_tot_dt)
        dS_tot = torch.bmm(d_raw.transpose(-1, -2), qq).float().contiguous()

        g_fwd = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        g_rev = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        rc = lib.launch_bidi_state_scan_bwd(W_cam.data_ptr(), dS_tot.data_ptr(), g_fwd.data_ptr(), g_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"cam bidi_state_scan_bwd failed rc={rc}")

        dU = g_fwd.clone()
        dU[:, 1:].add_(g_rev[:, :-1])
        dW = torch.zeros_like(g_fwd)
        if T > 1:
            n_m1 = BH * (T - 1)
            dW[:, 1:] = (
                torch.bmm(S_fwd[:, :-1].reshape(n_m1, D, D).transpose(-1, -2), g_fwd[:, 1:].reshape(n_m1, D, D))
                + torch.bmm(S_rev[:, 1:].reshape(n_m1, D, D).transpose(-1, -2), g_rev[:, :-1].reshape(n_m1, D, D))
            ).view(BH, T - 1, D, D)

        dW_m = dW.view(M, D, D)
        dU_m = dU.view(M, D, D)
        d_decay = (dW_m * I_m_P).sum(dim=(-2, -1))
        dP = (-decay_m * dW_m)

        dP_dt = dP.to(dt) if use_bf16 else dP
        dU_dt = dU_m.to(dt) if use_bf16 else dU_m
        beta_dt = beta_m.to(dt) if use_bf16 else beta_m

        k_b = kk * beta_dt
        v_b = vv * beta_dt
        dK_b = torch.bmm(kk, dP_dt.transpose(-1, -2))
        dV_b = torch.bmm(kk, dU_dt.transpose(-1, -2))
        dK = torch.bmm(k_b, dP_dt) + torch.bmm(v_b, dU_dt) + dK_b * beta_dt
        dV = dV_b * beta_dt

        dbeta_s = (dK_b.float() * kk.float()).sum(dim=-1) + (dV_b.float() * vv.float()).sum(dim=-1)
        if beta_is_4d:
            dbeta_ret = dbeta_s.view(B, H, T, S).to(beta_orig_dt)
        else:
            dbeta_ret = dbeta_s.sum(dim=-1).view(B, H, T).to(beta_orig_dt)

        return (
            dQ.view(BH, T * S, D),
            dK.view(BH, T * S, D),
            dV.view(BH, T * S, D),
            dbeta_ret,
            d_decay.view(B, H, T).to(decay_orig_dt),
            None, None, None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Autograd Function 6: Ultra-Lean Combined Main ShortConv4 + Prep + Bidi Scan (v2)
#   - Absorbs short_conv4 (directly reads qkv[:,:,1], eliminates k_raw & k_conv save: -197 MB/layer)
#   - Eliminates num save (-98.6 MB/layer) by using already-computed out(B,N,C) in backward
#   - Uses fused Phase A/C kernels (eliminates FP32 CastOp, ReduceABC, broadcast MulOp, AddOp)
#   - Uses single-pass main_gdn_prep_bwd with n_chunks=240
# ---------------------------------------------------------------------------
class _FusedMainPrepAndScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_or_qkv, qkv_w, qkv_b, conv_k_w, conv_k_b, q_nw, k_nw, rope_cos, rope_sin, beta, decay,
                B: int, H: int, T: int, S: int, k_scale: float, norm_eps: float, eps: float, use_bf16: bool):
        recompute_qkv = qkv_w is not None
        has_qkv_bias = recompute_qkv and (qkv_b is not None)
        N = T * S
        if recompute_qkv:
            x_c = x_or_qkv.contiguous()
            C = x_c.shape[-1]
            D = C // H
            qkv_c = torch.nn.functional.linear(x_c.view(B * N, C), qkv_w, qkv_b).view(B, N, 3, H, D).contiguous()
        else:
            x_c = None
            qkv_c = x_or_qkv.contiguous()
            D = qkv_c.shape[-1]
            C = H * D
        has_kconv = conv_k_w is not None
        has_kbias = has_kconv and (conv_k_b is not None)
        ckw_c = conv_k_w.detach().contiguous().clone() if has_kconv else None
        ckb_c = conv_k_b.detach().contiguous().clone() if has_kbias else None
        BH = B * H
        M = BH * T
        has_norm = q_nw is not None
        q_nw_f = q_nw.float().contiguous() if has_norm else None
        k_nw_f = k_nw.float().contiguous() if has_norm else None
        rc_c = rope_cos.float().contiguous()
        rs_c = rope_sin.float().contiguous()
        dev = qkv_c.device
        out_dt = torch.bfloat16 if use_bf16 else torch.float32
        lib = _load_lib()
        st = _stream(dev)

        kc_tmp = None
        if has_kconv:
            kc_tmp = torch.empty((B, N, C), device=dev, dtype=qkv_c.dtype)
            rc = lib.launch_bidi_short_conv4_qkv_k_fwd(
                qkv_c.data_ptr(), ckw_c.data_ptr(), ckb_c.data_ptr() if has_kbias else 0,
                kc_tmp.data_ptr(), B, T, S, C, 1 if has_kbias else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"bidi_short_conv4_qkv_k_fwd failed rc={rc}")

        beta_m = beta.float().reshape(M, S).contiguous()
        decay_m = decay.float().reshape(M).contiguous()
        q_rot = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        q_pos = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        k_rot = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        k_pos = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        q_ir = torch.empty((B * N,), device=dev, dtype=torch.float32)
        k_ir = torch.empty((B * N,), device=dev, dtype=torch.float32)

        use_fused_u4 = use_bf16 and has_norm and (H == 20) and (D == 112)
        if use_fused_u4:
            kr_b = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            kp_b = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            v_b  = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            U_z  = torch.empty((M, D), device=dev, dtype=torch.float32)
            rc = lib.launch_main_gdn_prep_scale_fwd_u4(
                qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0,
                q_nw_f.data_ptr(), k_nw_f.data_ptr(),
                rc_c.data_ptr(), rs_c.data_ptr(), beta_m.data_ptr(),
                q_rot.data_ptr(), q_pos.data_ptr(), k_rot.data_ptr(), k_pos.data_ptr(),
                kr_b.data_ptr(), kp_b.data_ptr(), v_b.data_ptr(), 0, U_z.data_ptr(),
                q_ir.data_ptr(), k_ir.data_ptr(),
                B, T, S, float(k_scale), float(norm_eps), st,
            )
            if rc != 0:
                raise RuntimeError(f"main_gdn_prep_scale_fwd_u4 failed rc={rc}")
            del kc_tmp
            if recompute_qkv:
                del qkv_c
            qr = q_rot.view(M, S, D)
            qp = q_pos.view(M, S, D)
            kr = k_rot.view(M, S, D)
            kp = k_pos.view(M, S, D)
            P_kv = torch.bmm(kr_b.transpose(-1, -2), kr).float().contiguous()
            U_kv = torch.bmm(v_b.transpose(-1, -2),  kr).float().contiguous()
            P_z  = torch.bmm(kp_b.transpose(-1, -2), kp).float().contiguous()
            del kr_b, kp_b, v_b
            W_kv = torch.empty_like(P_kv)
            W_z  = torch.empty_like(P_z)
            rc = lib.launch_gdn_make_W(
                P_kv.data_ptr(), P_z.data_ptr(), decay_m.data_ptr(), W_kv.data_ptr(), W_z.data_ptr(), M, D, st,
            )
            if rc != 0:
                raise RuntimeError(f"gdn_make_W failed rc={rc}")
            del P_kv, P_z
        else:
            v_out = torch.empty((BH, N, D), device=dev, dtype=out_dt)
            rc = lib.launch_main_gdn_prep_fwd(
                qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0,
                q_nw_f.data_ptr() if has_norm else 0, k_nw_f.data_ptr() if has_norm else 0,
                rc_c.data_ptr(), rs_c.data_ptr(),
                q_rot.data_ptr(), q_pos.data_ptr(), k_rot.data_ptr(), k_pos.data_ptr(), v_out.data_ptr(),
                q_ir.data_ptr(), k_ir.data_ptr(),
                B, N, H, D, float(k_scale), float(norm_eps), 1 if has_norm else 0, 1 if use_bf16 else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"main_gdn_prep_fwd failed rc={rc}")
            del kc_tmp
            if recompute_qkv:
                # Discard 282 MiB qkv immediately; recompute in 1.69 ms in backward!
                del qkv_c

            qr = q_rot.view(M, S, D)
            qp = q_pos.view(M, S, D)
            kr = k_rot.view(M, S, D)
            kp = k_pos.view(M, S, D)
            vv = v_out.view(M, S, D)

            if use_bf16:
                kr_b = torch.empty_like(kr)
                kp_b = torch.empty_like(kp)
                v_b  = torch.empty_like(vv)
                U_z  = torch.empty((M, D), device=dev, dtype=torch.float32)
                rc = lib.launch_gdn_main_phase_a_scale(
                    kr.data_ptr(), kp.data_ptr(), vv.data_ptr(), beta_m.data_ptr(),
                    kr_b.data_ptr(), kp_b.data_ptr(), v_b.data_ptr(), U_z.data_ptr(),
                    M, S, D, st,
                )
                if rc != 0:
                    raise RuntimeError(f"gdn_main_phase_a_scale failed rc={rc}")
                P_kv = torch.bmm(kr_b.transpose(-1, -2), kr).float().contiguous()
                U_kv = torch.bmm(v_b.transpose(-1, -2),  kr).float().contiguous()
                P_z  = torch.bmm(kp_b.transpose(-1, -2), kp).float().contiguous()
                del kr_b, kp_b, v_b
                W_kv = torch.empty_like(P_kv)
                W_z  = torch.empty_like(P_z)
                rc = lib.launch_gdn_make_W(
                    P_kv.data_ptr(), P_z.data_ptr(), decay_m.data_ptr(), W_kv.data_ptr(), W_z.data_ptr(), M, D, st,
                )
                if rc != 0:
                    raise RuntimeError(f"gdn_make_W failed rc={rc}")
                del P_kv, P_z
            else:
                b3 = beta_m.unsqueeze(-1)
                d3 = decay_m.view(M, 1, 1)
                kr_b, kp_b, v_b = kr * b3, kp * b3, vv * b3
                P_kv = torch.bmm(kr_b.transpose(-1, -2), kr).float()
                U_kv = torch.bmm(v_b.transpose(-1, -2),  kr).float().contiguous()
                P_z  = torch.bmm(kp_b.transpose(-1, -2), kp).float()
                U_z  = kp_b.float().sum(dim=1).contiguous()
                I_D = _eye_cache(D, dev, torch.float32)
                W_kv = (d3 * (I_D - P_kv)).contiguous()
                W_z  = (d3 * (I_D - P_z)).contiguous()

        S_fwd = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        S_rev = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        Sz_fwd = torch.empty((BH, T, D), device=dev, dtype=torch.float32)
        Sz_rev = torch.empty((BH, T, D), device=dev, dtype=torch.float32)

        rc = lib.launch_bidi_state_scan_fwd(W_kv.data_ptr(), U_kv.data_ptr(), S_fwd.data_ptr(), S_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"bidi_state_scan_fwd failed rc={rc}")
        rc = lib.launch_bidi_z_scan_fwd(W_z.data_ptr(), U_z.data_ptr(), Sz_fwd.data_ptr(), Sz_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"bidi_z_scan_fwd failed rc={rc}")
        del U_kv, U_z

        if use_bf16:
            S_tot_dt = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            Sz_tot = torch.empty((M, D), device=dev, dtype=torch.float32)
            rc = lib.launch_gdn_add_cast_S(
                S_fwd.data_ptr(), S_rev.data_ptr(), Sz_fwd.data_ptr(), Sz_rev.data_ptr(),
                S_tot_dt.data_ptr(), Sz_tot.data_ptr(), M, D, st,
            )
            if rc != 0:
                raise RuntimeError(f"gdn_add_cast_S failed rc={rc}")
            num = torch.bmm(qr, S_tot_dt.transpose(-1, -2))
            den = torch.empty((M, S), device=dev, dtype=torch.float32)
            rc = lib.launch_gdn_qp_sz_dot(qp.data_ptr(), Sz_tot.data_ptr(), den.data_ptr(), M, S, D, st)
            if rc != 0:
                raise RuntimeError(f"gdn_qp_sz_dot failed rc={rc}")
        else:
            S_tot = (S_fwd + S_rev).view(M, D, D)
            Sz_tot = (Sz_fwd + Sz_rev).view(M, D)
            num = torch.bmm(qr, S_tot.transpose(-1, -2))
            den = torch.bmm(qp.float(), Sz_tot.unsqueeze(-1)).squeeze(-1).contiguous()

        out = torch.empty((B, N, C), device=dev, dtype=torch.bfloat16)
        rc = lib.launch_out_div_fwd(
            num.data_ptr(), den.data_ptr(), out.data_ptr(),
            B, N, H, D, float(eps), 1 if use_bf16 else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"out_div_fwd failed rc={rc}")

        empty = torch.empty(0, device=dev, dtype=torch.float32)
        ctx.save_for_backward(
            x_c if recompute_qkv else qkv_c,
            qkv_w if recompute_qkv else empty,
            qkv_b if has_qkv_bias else empty,
            ckw_c if has_kconv else empty,
            ckb_c if has_kbias else empty,
            q_nw_f if has_norm else empty,
            k_nw_f if has_norm else empty,
            q_ir, k_ir, rc_c, rs_c,
            beta_m, decay_m, W_kv, W_z, S_fwd, S_rev, Sz_fwd, Sz_rev,
            out if use_bf16 else num, den,
        )
        ctx.meta = (
            B, H, T, S, D, float(k_scale), float(norm_eps), float(eps),
            has_norm, has_kconv, has_kbias, use_bf16, recompute_qkv, has_qkv_bias,
            conv_k_w.shape if has_kconv else None,
            conv_k_b.shape if has_kbias else None,
            q_nw.dtype if has_norm else None, k_nw.dtype if has_norm else None,
            beta.dtype, decay.dtype,
        )
        return out

    @staticmethod
    def backward(ctx, dout):
        (x_or_qkv_saved, qkv_w_saved, qkv_b_saved, ckw_saved, ckb_saved, q_nw_f, k_nw_f, q_ir, k_ir, rc_c, rs_c,
         beta_m, decay_m, W_kv, W_z, S_fwd, S_rev, Sz_fwd, Sz_rev, out_or_num, den) = ctx.saved_tensors
        (B, H, T, S, D, k_scale, norm_eps, eps,
         has_norm, has_kconv, has_kbias, use_bf16, recompute_qkv, has_qkv_bias, ckw_shape, ckb_shape,
         q_nw_dt, k_nw_dt, beta_orig_dt, decay_orig_dt) = ctx.meta
        N = T * S
        BH = B * H
        M = BH * T
        C = H * D
        dt = torch.bfloat16 if use_bf16 else torch.float32
        dev = x_or_qkv_saved.device
        lib = _load_lib()
        st = _stream(dev)

        if recompute_qkv:
            x_2d = x_or_qkv_saved.view(B * N, C)
            qkv_c = torch.nn.functional.linear(
                x_2d, qkv_w_saved, qkv_b_saved if has_qkv_bias else None
            ).view(B, N, 3, H, D).contiguous()
        else:
            qkv_c = x_or_qkv_saved

        kc_tmp = None
        if has_kconv:
            kc_tmp = torch.empty((B, N, C), device=dev, dtype=qkv_c.dtype)
            rc = lib.launch_bidi_short_conv4_qkv_k_fwd(
                qkv_c.data_ptr(), ckw_saved.data_ptr(), ckb_saved.data_ptr() if has_kbias else 0,
                kc_tmp.data_ptr(), B, T, S, C, 1 if has_kbias else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"bidi_short_conv4_qkv_k_fwd (bwd recompute) failed rc={rc}")

        q_rot = torch.empty((BH, N, D), device=dev, dtype=dt)
        q_pos = torch.empty((BH, N, D), device=dev, dtype=dt)
        k_rot = torch.empty((BH, N, D), device=dev, dtype=dt)
        k_pos = torch.empty((BH, N, D), device=dev, dtype=dt)
        v_out = torch.empty((BH, N, D), device=dev, dtype=dt)
        use_fused_u4 = use_bf16 and has_norm and (H == 20) and (D == 112)
        if use_fused_u4:
            kr_b = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            kp_b = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            v_b  = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            rc = lib.launch_main_gdn_prep_scale_fwd_u4(
                qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0,
                q_nw_f.data_ptr(), k_nw_f.data_ptr(),
                rc_c.data_ptr(), rs_c.data_ptr(), beta_m.data_ptr(),
                q_rot.data_ptr(), q_pos.data_ptr(), k_rot.data_ptr(), k_pos.data_ptr(),
                kr_b.data_ptr(), kp_b.data_ptr(), v_b.data_ptr(), v_out.data_ptr(), 0,
                0, 0,
                B, T, S, float(k_scale), float(norm_eps), st,
            )
            if rc != 0:
                raise RuntimeError(f"main_gdn_prep_scale_fwd_u4 (bwd) failed rc={rc}")
        else:
            kr_b = kp_b = v_b = None
            q_ir_tmp = torch.empty((B * N,), device=dev, dtype=torch.float32)
            k_ir_tmp = torch.empty((B * N,), device=dev, dtype=torch.float32)
            rc = lib.launch_main_gdn_prep_fwd(
                qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0,
                q_nw_f.data_ptr() if has_norm else 0, k_nw_f.data_ptr() if has_norm else 0,
                rc_c.data_ptr(), rs_c.data_ptr(),
                q_rot.data_ptr(), q_pos.data_ptr(), k_rot.data_ptr(), k_pos.data_ptr(), v_out.data_ptr(),
                q_ir_tmp.data_ptr(), k_ir_tmp.data_ptr(),
                B, N, H, D, float(k_scale), float(norm_eps), 1 if has_norm else 0, 1 if use_bf16 else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"main_gdn_prep_fwd (bwd recompute) failed rc={rc}")
            del q_ir_tmp, k_ir_tmp

        dout_c = dout.contiguous()
        qr = q_rot.view(M, S, D)
        qp = q_pos.view(M, S, D)
        kr = k_rot.view(M, S, D)
        kp = k_pos.view(M, S, D)
        vv = v_out.view(M, S, D)

        if use_bf16:
            S_tot_dt = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            Sz_tot = torch.empty((M, D), device=dev, dtype=torch.float32)
            rc = lib.launch_gdn_add_cast_S(
                S_fwd.data_ptr(), S_rev.data_ptr(), Sz_fwd.data_ptr(), Sz_rev.data_ptr(),
                S_tot_dt.data_ptr(), Sz_tot.data_ptr(), M, D, st,
            )
            if rc != 0:
                raise RuntimeError(f"gdn_add_cast_S (bwd) failed rc={rc}")
            dnum = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            dden = torch.empty((M, S), device=dev, dtype=torch.float32)
            dQ_pos = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            rc = lib.launch_out_div_bwd_from_out_dqpos(
                dout_c.data_ptr(), out_or_num.data_ptr(), den.data_ptr(), Sz_tot.data_ptr(),
                dnum.data_ptr(), dden.data_ptr(), dQ_pos.data_ptr(),
                BH, H, T, S, D, eps, st,
            )
            if rc != 0:
                raise RuntimeError(f"out_div_bwd_from_out_dqpos failed rc={rc}")
            dQ_rot = torch.bmm(dnum, S_tot_dt)
            dS_tot = torch.bmm(dnum.transpose(-1, -2), qr).float().contiguous()
            dSz_tot = torch.empty((M, D), device=dev, dtype=torch.float32)
            rc = lib.launch_gdn_dden_qp_bwd(dden.data_ptr(), qp.data_ptr(), dSz_tot.data_ptr(), M, S, D, st)
            if rc != 0:
                raise RuntimeError(f"gdn_dden_qp_bwd failed rc={rc}")
            del dnum, dden, q_rot, q_pos, qr, qp, S_tot_dt, Sz_tot
        else:
            dnum = torch.empty_like(out_or_num)
            dden = torch.empty_like(den)
            rc = lib.launch_out_div_bwd(
                dout_c.data_ptr(), out_or_num.data_ptr(), den.data_ptr(), dnum.data_ptr(), dden.data_ptr(),
                B, N, H, D, eps, 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"out_div_bwd failed rc={rc}")
            S_tot = (S_fwd + S_rev).view(M, D, D)
            Sz_tot = (Sz_fwd + Sz_rev).view(M, D)
            dQ_rot = torch.bmm(dnum, S_tot)
            dS_tot = torch.bmm(dnum.transpose(-1, -2), qr).float().contiguous()
            dQ_pos = (dden.unsqueeze(-1) * Sz_tot.unsqueeze(1))
            dSz_tot = torch.bmm(dden.unsqueeze(1), qp.float()).squeeze(1).contiguous()
            del dnum, dden, q_rot, q_pos, qr, qp

        g_fwd = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        g_rev = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        gz_fwd = torch.empty((BH, T, D), device=dev, dtype=torch.float32)
        gz_rev = torch.empty((BH, T, D), device=dev, dtype=torch.float32)
        rc = lib.launch_bidi_state_scan_bwd(W_kv.data_ptr(), dS_tot.data_ptr(), g_fwd.data_ptr(), g_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"bidi_state_scan_bwd failed rc={rc}")
        rc = lib.launch_bidi_z_scan_bwd(W_z.data_ptr(), dSz_tot.data_ptr(), gz_fwd.data_ptr(), gz_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"bidi_z_scan_bwd failed rc={rc}")

        dW_kv = torch.zeros_like(g_fwd)
        if T > 1:
            n_m1 = BH * (T - 1)
            dW_kv[:, 1:] = (
                torch.bmm(S_fwd[:, :-1].reshape(n_m1, D, D).transpose(-1, -2), g_fwd[:, 1:].reshape(n_m1, D, D))
                + torch.bmm(S_rev[:, 1:].reshape(n_m1, D, D).transpose(-1, -2), g_rev[:, :-1].reshape(n_m1, D, D))
            ).view(BH, T - 1, D, D)
        dW_z = torch.zeros_like(g_fwd)
        if T > 1:
            dW_z[:, 1:] = (
                Sz_fwd[:, :-1].unsqueeze(-1) * gz_fwd[:, 1:].unsqueeze(-2)
                + Sz_rev[:, 1:].unsqueeze(-1) * gz_rev[:, :-1].unsqueeze(-2)
            )

        if use_bf16:
            d_decay = torch.empty((M,), device=dev, dtype=torch.float32)
            dP_kv_dt = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            dU_kv_dt = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            dP_z_dt  = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            dU_z_m   = torch.empty((M, D), device=dev, dtype=torch.float32)
            rc = lib.launch_gdn_ddecay_dp_du_bwd(
                W_kv.data_ptr(), dW_kv.data_ptr(), g_fwd.data_ptr(), g_rev.data_ptr(),
                W_z.data_ptr(), dW_z.data_ptr(), gz_fwd.data_ptr(), gz_rev.data_ptr(),
                decay_m.data_ptr(), d_decay.data_ptr(),
                dP_kv_dt.data_ptr(), dU_kv_dt.data_ptr(), dP_z_dt.data_ptr(), dU_z_m.data_ptr(),
                M, T, D, st,
            )
            if rc != 0:
                raise RuntimeError(f"gdn_ddecay_dp_du_bwd failed rc={rc}")
            del g_fwd, g_rev, gz_fwd, gz_rev, dW_kv, dW_z

            if kr_b is None:
                kr_b = torch.empty_like(kr)
                kp_b = torch.empty_like(kp)
                v_b  = torch.empty_like(vv)
                rc = lib.launch_gdn_main_phase_a_scale(
                    kr.data_ptr(), kp.data_ptr(), vv.data_ptr(), beta_m.data_ptr(),
                    kr_b.data_ptr(), kp_b.data_ptr(), v_b.data_ptr(), 0,
                    M, S, D, st,
                )
                if rc != 0:
                    raise RuntimeError(f"gdn_main_phase_a_scale (bwd) failed rc={rc}")

            dKr_b = torch.bmm(kr, dP_kv_dt.transpose(-1, -2))
            dV_b  = torch.bmm(kr, dU_kv_dt.transpose(-1, -2))
            dKr_bmm1 = torch.bmm(kr_b, dP_kv_dt)
            dKr_bmm2 = torch.bmm(v_b, dU_kv_dt)
            dKp_b_raw = torch.bmm(kp, dP_z_dt.transpose(-1, -2))
            dKp_bmm   = torch.bmm(kp_b, dP_z_dt)
            del kr_b, kp_b, v_b, dP_kv_dt, dU_kv_dt, dP_z_dt

            dK_rot = torch.empty_like(kr)
            dK_pos = torch.empty_like(kp)
            dV     = torch.empty_like(vv)
            dbeta  = torch.empty((M, S), device=dev, dtype=torch.float32)
            rc = lib.launch_gdn_main_phase_a_epilogue_bwd(
                kr.data_ptr(), kp.data_ptr(), vv.data_ptr(), beta_m.data_ptr(),
                dKr_b.data_ptr(), dV_b.data_ptr(), dKp_b_raw.data_ptr(), dU_z_m.data_ptr(),
                dKr_bmm1.data_ptr(), dKr_bmm2.data_ptr(), dKp_bmm.data_ptr(),
                dK_rot.data_ptr(), dK_pos.data_ptr(), dV.data_ptr(), dbeta.data_ptr(),
                M, S, D, st,
            )
            if rc != 0:
                raise RuntimeError(f"gdn_main_phase_a_epilogue_bwd failed rc={rc}")
            del dKr_b, dV_b, dKp_b_raw, dU_z_m, dKr_bmm1, dKr_bmm2, dKp_bmm, k_rot, k_pos, v_out, kr, kp, vv
        else:
            dU_kv = g_fwd.clone(); dU_kv[:, 1:].add_(g_rev[:, :-1])
            dU_z  = gz_fwd.clone(); dU_z[:, 1:].add_(gz_rev[:, :-1])
            del g_fwd, g_rev, gz_fwd, gz_rev
            dW_kv_m, dU_kv_m = dW_kv.view(M, D, D), dU_kv.view(M, D, D)
            dW_z_m, dU_z_m = dW_z.view(M, D, D), dU_z.view(M, D)
            d3 = decay_m.view(M, 1, 1)
            inv_decay = 1.0 / d3.clamp_min(1e-12)
            d_decay = ((dW_kv_m * W_kv.view(M, D, D)).sum(dim=(-2, -1)) + (dW_z_m * W_z.view(M, D, D)).sum(dim=(-2, -1))) * inv_decay.squeeze(-1).squeeze(-1)
            dP_kv, dP_z = -d3 * dW_kv_m, -d3 * dW_z_m
            b3 = beta_m.unsqueeze(-1)
            kr_b, kp_b, v_b = kr * b3, kp * b3, vv * b3
            dKr_b = torch.bmm(kr, dP_kv.transpose(-1, -2))
            dV_b  = torch.bmm(kr, dU_kv_m.transpose(-1, -2))
            dK_rot = torch.bmm(kr_b, dP_kv) + torch.bmm(v_b, dU_kv_m) + dKr_b * b3
            dV = dV_b * b3
            dKp_b = torch.bmm(kp, dP_z.transpose(-1, -2)) + dU_z_m.unsqueeze(1)
            dK_pos = torch.bmm(kp_b, dP_z) + dKp_b * b3
            dbeta = (dKr_b * kr).sum(-1) + (dV_b * vv).sum(-1) + (dKp_b * kp).sum(-1)
            del kr_b, kp_b, v_b, dKr_b, dV_b, dKp_b, k_rot, k_pos, v_out, kr, kp, vv

        dq_rot_c = dQ_rot.view(BH, N, D).contiguous()
        dq_pos_c = dQ_pos.view(BH, N, D).contiguous()
        dk_rot_c = dK_rot.view(BH, N, D).contiguous()
        dk_pos_c = dK_pos.view(BH, N, D).contiguous()
        dv_c     = dV.view(BH, N, D).contiguous()

        dqkv = torch.empty_like(qkv_c)
        dkconv = torch.empty((B, N, C), device=dev, dtype=qkv_c.dtype) if has_kconv else None
        n_chunks = 120
        dq_nw_part = torch.empty((n_chunks, C), device=dev, dtype=torch.float32) if has_norm else None
        dk_nw_part = torch.empty((n_chunks, C), device=dev, dtype=torch.float32) if has_norm else None
        dq_nw_out = torch.empty((C,), device=dev, dtype=torch.float32) if has_norm else None
        dk_nw_out = torch.empty((C,), device=dev, dtype=torch.float32) if has_norm else None

        rc = lib.launch_main_gdn_prep_bwd(
            qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0,
            q_nw_f.data_ptr() if has_norm else 0, k_nw_f.data_ptr() if has_norm else 0,
            q_ir.data_ptr(), k_ir.data_ptr(), rc_c.data_ptr(), rs_c.data_ptr(),
            dq_rot_c.data_ptr(), dq_pos_c.data_ptr(), dk_rot_c.data_ptr(), dk_pos_c.data_ptr(), dv_c.data_ptr(),
            dqkv.data_ptr(), dkconv.data_ptr() if has_kconv else 0,
            dq_nw_part.data_ptr() if has_norm else 0, dk_nw_part.data_ptr() if has_norm else 0,
            dq_nw_out.data_ptr() if has_norm else 0, dk_nw_out.data_ptr() if has_norm else 0,
            B, N, H, D, k_scale, 1 if has_norm else 0, 1 if use_bf16 else 0, n_chunks, st,
        )
        if rc != 0:
            raise RuntimeError(f"main_gdn_prep_bwd failed rc={rc}")
        del kc_tmp

        dckw = None
        dckb = None
        if has_kconv:
            n_bs_blk = (B * S + 7) // 8
            dw_part = torch.empty((n_bs_blk, C, 4), device=dev, dtype=torch.float32)
            dckw = torch.empty(ckw_shape, device=dev, dtype=ckw_saved.dtype)
            db_part = torch.empty((n_bs_blk, C), device=dev, dtype=torch.float32) if has_kbias else None
            dckb = torch.empty(ckb_shape, device=dev, dtype=ckw_saved.dtype) if has_kbias else None
            rc = lib.launch_bidi_short_conv4_qkv_k_bwd(
                qkv_c.data_ptr(), dkconv.data_ptr(), ckw_saved.data_ptr(), dqkv.data_ptr(),
                dw_part.data_ptr(), db_part.data_ptr() if has_kbias else 0,
                dckw.data_ptr(), dckb.data_ptr() if has_kbias else 0,
                B, T, S, C, 1 if has_kbias else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"bidi_short_conv4_qkv_k_bwd failed rc={rc}")

        dq_nw_ret = dq_nw_out.to(q_nw_dt) if has_norm else None
        dk_nw_ret = dk_nw_out.to(k_nw_dt) if has_norm else None
        if recompute_qkv:
            from diffusion.model.musa_ops.block_fusion import fast_col_sum_bf16
            del qkv_c
            dqkv_2d = dqkv.view(B * N, 3 * C)
            dx_in = torch.mm(dqkv_2d, qkv_w_saved).view(B, N, C) if ctx.needs_input_grad[0] else None
            d_qkv_w = torch.mm(dqkv_2d.t(), x_2d) if ctx.needs_input_grad[1] else None
            if has_qkv_bias and ctx.needs_input_grad[2]:
                d_qkv_b = fast_col_sum_bf16(dqkv_2d)
                if d_qkv_b.dtype != qkv_b_saved.dtype:
                    d_qkv_b = d_qkv_b.to(qkv_b_saved.dtype)
            else:
                d_qkv_b = None
            del dqkv, dqkv_2d
            return (
                dx_in, d_qkv_w, d_qkv_b, dckw, dckb, dq_nw_ret, dk_nw_ret, None, None,
                dbeta.view(B, H, T, S).to(beta_orig_dt),
                d_decay.view(B, H, T).to(decay_orig_dt),
                None, None, None, None, None, None, None, None,
            )
        return (
            dqkv, None, None, dckw, dckb, dq_nw_ret, dk_nw_ret, None, None,
            dbeta.view(B, H, T, S).to(beta_orig_dt),
            d_decay.view(B, H, T).to(decay_orig_dt),
            None, None, None, None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Autograd Function 7: Ultra-Lean Combined Cam ShortConv4 + Prep + Bidi Scan + Apply O (v2)
#   - Takes packed qkv_cam(B,N,3*C) directly: zero chunk/contiguous/cat copies!
#   - Absorbs conv_k_cam short_conv4: avoids saving k_conv_cam (-98.6 MB/layer)
#   - Uses fused Phase A/C kernels + single-pass cam_prep_bwd (n_chunks=240)
# ---------------------------------------------------------------------------
class _FusedCamPrepAndScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv_cam, conv_k_w, conv_k_b, q_nw, k_nw, beta_raw, decay,
                P, P_T, P_inv, rope_cos, rope_sin,
                B: int, H: int, T: int, S: int, D: int, k_scale: float, norm_eps: float, use_bf16: bool):
        qkv_c = qkv_cam.contiguous()
        has_kconv = conv_k_w is not None
        has_kbias = has_kconv and (conv_k_b is not None)
        ckw_c = conv_k_w.detach().contiguous().clone() if has_kconv else None
        ckb_c = conv_k_b.detach().contiguous().clone() if has_kbias else None
        q_nw_f = q_nw.float().contiguous()
        k_nw_f = k_nw.float().contiguous()
        P_f = P.float().contiguous()
        pt_f = P_T.float().contiguous()
        pinv_f = P_inv.float().contiguous()
        rc_f = rope_cos.float().contiguous()
        rs_f = rope_sin.float().contiguous()
        N = T * S
        C = H * D
        BH = B * H
        M = BH * T
        dev = qkv_c.device
        out_dt = torch.bfloat16 if use_bf16 else torch.float32
        lib = _load_lib()
        st = _stream(dev)

        kc_tmp = None
        if has_kconv:
            kc_tmp = torch.empty((B, N, C), device=dev, dtype=qkv_c.dtype)
            rc = lib.launch_bidi_short_conv4_qkv_k_fwd(
                qkv_c.data_ptr(), ckw_c.data_ptr(), ckb_c.data_ptr() if has_kbias else 0,
                kc_tmp.data_ptr(), B, T, S, C, 1 if has_kbias else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"cam bidi_short_conv4_qkv_k_fwd failed rc={rc}")

        q_out = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        k_out = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        v_out = torch.empty((BH, N, D), device=dev, dtype=out_dt)
        inf_sq = torch.empty((BH, N), device=dev, dtype=torch.float32)
        q_ir = torch.empty((B * N,), device=dev, dtype=torch.float32)
        k_ir = torch.empty((B * N,), device=dev, dtype=torch.float32)
        pre_ksq = torch.empty((BH, N), device=dev, dtype=torch.float32)
        post_ksq = torch.empty((BH, N), device=dev, dtype=torch.float32)

        use_cam_u4 = use_bf16 and (H == 20) and (D == 112)
        if use_cam_u4:
            rc = lib.launch_cam_prep_fwd_u4(
                qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0, 0,
                q_nw_f.data_ptr(), k_nw_f.data_ptr(),
                pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
                0,
                q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), 0, 0,
                inf_sq.data_ptr(), q_ir.data_ptr(), k_ir.data_ptr(), pre_ksq.data_ptr(), post_ksq.data_ptr(),
                B, T, S, 0, float(k_scale), float(norm_eps), st,
            )
            if rc != 0:
                raise RuntimeError(f"cam_prep_fwd_u4 failed rc={rc}")
        else:
            # Packed mode: v_ptr=0 signals qkv_c is (B,N,3*C); k_ptr=kc_tmp (or 0 for slot 1)
            rc = lib.launch_cam_prep_fwd(
                qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0, 0,
                q_nw_f.data_ptr(), k_nw_f.data_ptr(),
                pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
                q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), inf_sq.data_ptr(),
                q_ir.data_ptr(), k_ir.data_ptr(), pre_ksq.data_ptr(), post_ksq.data_ptr(),
                B, N, H, D, float(k_scale), float(norm_eps), 1 if use_bf16 else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"cam_prep_fwd failed rc={rc}")
        del kc_tmp

        beta_is_4d = (beta_raw.ndim == 4)
        beta_raw_f = beta_raw.float().contiguous()
        decay_m = decay.float().reshape(M).contiguous()
        qq = q_out.view(M, S, D)
        kk = k_out.view(M, S, D)
        vv = v_out.view(M, S, D)

        if use_cam_u4:
            k_b = torch.empty_like(kk)
            v_b = torch.empty_like(vv)
            beta_m = torch.empty((M, S) if beta_is_4d else (M,), device=dev, dtype=torch.float32)
            frame_inf = torch.empty((B, H, T), device=dev, dtype=torch.float32)
            rc = lib.launch_cam_phase_a_scale_fused_u4(
                kk.data_ptr(), vv.data_ptr(), inf_sq.data_ptr(), beta_raw_f.data_ptr(),
                k_b.data_ptr(), v_b.data_ptr(), beta_m.data_ptr(), frame_inf.data_ptr(),
                M, S, 1 if beta_is_4d else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"cam_phase_a_scale_fused_u4 failed rc={rc}")
            P_cam = torch.bmm(k_b.transpose(-1, -2), kk).float().contiguous()
            U_cam = torch.bmm(v_b.transpose(-1, -2), kk).float().contiguous()
            del k_b, v_b
            W_cam = torch.empty_like(P_cam)
            rc = lib.launch_gdn_make_W(P_cam.data_ptr(), 0, decay_m.data_ptr(), W_cam.data_ptr(), 0, M, D, st)
            if rc != 0:
                raise RuntimeError(f"cam gdn_make_W failed rc={rc}")
            del P_cam
        else:
            frame_inf = inf_sq.view(B, H, T, S).mean(dim=-1)
            denom = frame_inf.clamp_min(1.0)
            if beta_is_4d:
                beta_m = (beta_raw_f / denom.unsqueeze(-1)).reshape(M, S).contiguous()
            else:
                beta_m = (beta_raw_f / denom).reshape(M).contiguous()

            if use_bf16:
                k_b = torch.empty_like(kk)
                v_b = torch.empty_like(vv)
                rc = lib.launch_gdn_cam_phase_a_scale(
                    kk.data_ptr(), vv.data_ptr(), beta_m.data_ptr(),
                    k_b.data_ptr(), v_b.data_ptr(), M, S, D, 1 if beta_is_4d else 0, st,
                )
                if rc != 0:
                    raise RuntimeError(f"gdn_cam_phase_a_scale failed rc={rc}")
                P_cam = torch.bmm(k_b.transpose(-1, -2), kk).float().contiguous()
                U_cam = torch.bmm(v_b.transpose(-1, -2), kk).float().contiguous()
                del k_b, v_b
                W_cam = torch.empty_like(P_cam)
                rc = lib.launch_gdn_make_W(P_cam.data_ptr(), 0, decay_m.data_ptr(), W_cam.data_ptr(), 0, M, D, st)
                if rc != 0:
                    raise RuntimeError(f"cam gdn_make_W failed rc={rc}")
                del P_cam
            else:
                b3 = beta_m.unsqueeze(-1) if beta_is_4d else beta_m.view(M, 1, 1)
                k_b, v_b = kk * b3, vv * b3
                P_cam = torch.bmm(k_b.transpose(-1, -2), kk).float()
                U_cam = torch.bmm(v_b.transpose(-1, -2), kk).float().contiguous()
                I_D = _eye_cache(D, dev, torch.float32)
                W_cam = (decay_m.view(M, 1, 1) * (I_D - P_cam)).contiguous()

        S_fwd = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        S_rev = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        rc = lib.launch_bidi_state_scan_fwd(W_cam.data_ptr(), U_cam.data_ptr(), S_fwd.data_ptr(), S_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"cam bidi_state_scan_fwd failed rc={rc}")
        del U_cam

        if use_bf16:
            S_tot_dt = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            rc = lib.launch_gdn_add_cast_S(S_fwd.data_ptr(), S_rev.data_ptr(), 0, 0, S_tot_dt.data_ptr(), 0, M, D, st)
            if rc != 0:
                raise RuntimeError(f"cam gdn_add_cast_S failed rc={rc}")
        else:
            S_tot_dt = (S_fwd + S_rev).view(M, D, D)

        raw_out = torch.bmm(qq, S_tot_dt.transpose(-1, -2)).contiguous()
        out = torch.empty((B, N, C), device=dev, dtype=torch.bfloat16)
        rc = lib.launch_cam_out_apply_o_fwd(
            raw_out.data_ptr(), P_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(), out.data_ptr(),
            B, N, H, D, 1 if use_bf16 else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"cam_out_apply_o_fwd failed rc={rc}")

        empty = torch.empty(0, device=dev, dtype=torch.float32)
        ctx.save_for_backward(
            qkv_c,
            ckw_c if has_kconv else empty,
            ckb_c if has_kbias else empty,
            q_nw_f, k_nw_f,
            beta_raw_f, frame_inf, decay_m, W_cam, S_fwd, S_rev,
            P_f, pt_f, pinv_f, rc_f, rs_f, q_ir, k_ir, pre_ksq, post_ksq,
        )
        ctx.meta = (
            B, H, T, S, D, float(k_scale), float(norm_eps), use_bf16, beta_is_4d,
            has_kconv, has_kbias,
            conv_k_w.shape if has_kconv else None,
            conv_k_b.shape if has_kbias else None,
            q_nw.dtype, k_nw.dtype, beta_raw.dtype, decay.dtype,
        )
        return out

    @staticmethod
    def backward(ctx, dout):
        (qkv_c, ckw_saved, ckb_saved, q_nw_f, k_nw_f,
         beta_raw_f, frame_inf, decay_m, W_cam, S_fwd, S_rev,
         P_f, pt_f, pinv_f, rc_f, rs_f, q_ir, k_ir, pre_ksq, post_ksq) = ctx.saved_tensors
        (B, H, T, S, D, k_scale, norm_eps, use_bf16, beta_is_4d,
         has_kconv, has_kbias, ckw_shape, ckb_shape,
         q_nw_dt, k_nw_dt, beta_orig_dt, decay_orig_dt) = ctx.meta
        N = T * S
        BH = B * H
        M = BH * T
        C = H * D
        dt = torch.bfloat16 if use_bf16 else torch.float32
        dev = qkv_c.device
        lib = _load_lib()
        st = _stream(dev)

        kc_tmp = None
        if has_kconv:
            kc_tmp = torch.empty((B, N, C), device=dev, dtype=qkv_c.dtype)
            rc = lib.launch_bidi_short_conv4_qkv_k_fwd(
                qkv_c.data_ptr(), ckw_saved.data_ptr(), ckb_saved.data_ptr() if has_kbias else 0,
                kc_tmp.data_ptr(), B, T, S, C, 1 if has_kbias else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"cam bidi_short_conv4_qkv_k_fwd (bwd recompute) failed rc={rc}")

        denom = frame_inf.clamp_min(1.0)
        if beta_is_4d:
            beta_m = (beta_raw_f / denom.unsqueeze(-1)).reshape(M, S).contiguous()
        else:
            beta_m = (beta_raw_f / denom).reshape(M).contiguous()

        q_out = torch.empty((BH, N, D), device=dev, dtype=dt)
        k_out = torch.empty((BH, N, D), device=dev, dtype=dt)
        v_out = torch.empty((BH, N, D), device=dev, dtype=dt)
        use_cam_u4 = use_bf16 and (H == 20) and (D == 112)
        if use_cam_u4:
            k_b = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            v_b = torch.empty((M, S, D), device=dev, dtype=torch.bfloat16)
            rc = lib.launch_cam_prep_fwd_u4(
                qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0, 0,
                q_nw_f.data_ptr(), k_nw_f.data_ptr(),
                pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
                beta_m.data_ptr(),
                q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), k_b.data_ptr(), v_b.data_ptr(),
                0, 0, 0, 0, 0,
                B, T, S, 1 if beta_is_4d else 0, float(k_scale), float(norm_eps), st,
            )
            if rc != 0:
                raise RuntimeError(f"cam_prep_fwd_u4 (bwd recompute) failed rc={rc}")
        else:
            k_b = v_b = None
            inf_tmp = torch.empty((BH, N), device=dev, dtype=torch.float32)
            q_ir_t = torch.empty((B * N,), device=dev, dtype=torch.float32)
            k_ir_t = torch.empty((B * N,), device=dev, dtype=torch.float32)
            pre_t = torch.empty((BH, N), device=dev, dtype=torch.float32)
            post_t = torch.empty((BH, N), device=dev, dtype=torch.float32)
            rc = lib.launch_cam_prep_fwd(
                qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0, 0,
                q_nw_f.data_ptr(), k_nw_f.data_ptr(),
                pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
                q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), inf_tmp.data_ptr(),
                q_ir_t.data_ptr(), k_ir_t.data_ptr(), pre_t.data_ptr(), post_t.data_ptr(),
                B, N, H, D, float(k_scale), float(norm_eps), 1 if use_bf16 else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"cam_prep_fwd (bwd recompute) failed rc={rc}")
            del inf_tmp, q_ir_t, k_ir_t, pre_t, post_t

        dout_c = dout.contiguous()
        d_raw = torch.empty((M, S, D), device=dev, dtype=dt)
        rc = lib.launch_cam_out_apply_o_bwd(
            dout_c.data_ptr(), P_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(), d_raw.data_ptr(),
            B, N, H, D, 1 if use_bf16 else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"cam_out_apply_o_bwd failed rc={rc}")

        qq = q_out.view(M, S, D)
        kk = k_out.view(M, S, D)
        vv = v_out.view(M, S, D)
        if use_bf16:
            S_tot_dt = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            rc = lib.launch_gdn_add_cast_S(S_fwd.data_ptr(), S_rev.data_ptr(), 0, 0, S_tot_dt.data_ptr(), 0, M, D, st)
            if rc != 0:
                raise RuntimeError(f"cam gdn_add_cast_S (bwd) failed rc={rc}")
        else:
            S_tot_dt = (S_fwd + S_rev).view(M, D, D)

        dQ = torch.bmm(d_raw, S_tot_dt)
        dS_tot = torch.bmm(d_raw.transpose(-1, -2), qq).float().contiguous()
        del d_raw, q_out, qq, S_tot_dt

        g_fwd = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        g_rev = torch.empty((BH, T, D, D), device=dev, dtype=torch.float32)
        rc = lib.launch_bidi_state_scan_bwd(W_cam.data_ptr(), dS_tot.data_ptr(), g_fwd.data_ptr(), g_rev.data_ptr(), BH, T, st)
        if rc != 0:
            raise RuntimeError(f"cam bidi_state_scan_bwd failed rc={rc}")

        dW = torch.zeros_like(g_fwd)
        if T > 1:
            n_m1 = BH * (T - 1)
            dW[:, 1:] = (
                torch.bmm(S_fwd[:, :-1].reshape(n_m1, D, D).transpose(-1, -2), g_fwd[:, 1:].reshape(n_m1, D, D))
                + torch.bmm(S_rev[:, 1:].reshape(n_m1, D, D).transpose(-1, -2), g_rev[:, :-1].reshape(n_m1, D, D))
            ).view(BH, T - 1, D, D)

        if use_bf16:
            d_decay = torch.empty((M,), device=dev, dtype=torch.float32)
            dP_dt = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            dU_dt = torch.empty((M, D, D), device=dev, dtype=torch.bfloat16)
            rc = lib.launch_gdn_ddecay_dp_du_bwd(
                W_cam.data_ptr(), dW.data_ptr(), g_fwd.data_ptr(), g_rev.data_ptr(),
                0, 0, 0, 0,
                decay_m.data_ptr(), d_decay.data_ptr(),
                dP_dt.data_ptr(), dU_dt.data_ptr(), 0, 0,
                M, T, D, st,
            )
            if rc != 0:
                raise RuntimeError(f"cam gdn_ddecay_dp_du_bwd failed rc={rc}")
            del g_fwd, g_rev, dW

            if k_b is None:
                k_b = torch.empty_like(kk)
                v_b = torch.empty_like(vv)
                rc = lib.launch_gdn_cam_phase_a_scale(
                    kk.data_ptr(), vv.data_ptr(), beta_m.data_ptr(),
                    k_b.data_ptr(), v_b.data_ptr(), M, S, D, 1 if beta_is_4d else 0, st,
                )
                if rc != 0:
                    raise RuntimeError(f"cam gdn_cam_phase_a_scale (bwd) failed rc={rc}")

            dK_b = torch.bmm(kk, dP_dt.transpose(-1, -2))
            dV_b = torch.bmm(kk, dU_dt.transpose(-1, -2))
            dK_bmm1 = torch.bmm(k_b, dP_dt)
            dK_bmm2 = torch.bmm(v_b, dU_dt)
            del k_b, v_b, dP_dt, dU_dt

            dK = torch.empty_like(kk)
            dV = torch.empty_like(vv)
            dbeta_s = torch.empty((M, S), device=dev, dtype=torch.float32)
            rc = lib.launch_gdn_cam_phase_a_epilogue_bwd(
                kk.data_ptr(), vv.data_ptr(), beta_m.data_ptr(),
                dK_b.data_ptr(), dV_b.data_ptr(), dK_bmm1.data_ptr(), dK_bmm2.data_ptr(),
                dK.data_ptr(), dV.data_ptr(), dbeta_s.data_ptr(),
                M, S, D, 1 if beta_is_4d else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"cam gdn_cam_phase_a_epilogue_bwd failed rc={rc}")
            del dK_b, dV_b, dK_bmm1, dK_bmm2, k_out, v_out, kk, vv
        else:
            dU = g_fwd.clone(); dU[:, 1:].add_(g_rev[:, :-1])
            del g_fwd, g_rev
            dW_m, dU_m = dW.view(M, D, D), dU.view(M, D, D)
            d3 = decay_m.view(M, 1, 1)
            inv_decay = 1.0 / d3.clamp_min(1e-12)
            d_decay = (dW_m * W_cam.view(M, D, D)).sum(dim=(-2, -1)) * inv_decay.squeeze(-1).squeeze(-1)
            dP = -d3 * dW_m
            b3 = beta_m.unsqueeze(-1) if beta_is_4d else beta_m.view(M, 1, 1)
            k_b, v_b = kk * b3, vv * b3
            dK_b = torch.bmm(kk, dP.transpose(-1, -2))
            dV_b = torch.bmm(kk, dU_m.transpose(-1, -2))
            dK = torch.bmm(k_b, dP) + torch.bmm(v_b, dU_m) + dK_b * b3
            dV = dV_b * b3
            dbeta_s = (dK_b * kk).sum(-1) + (dV_b * vv).sum(-1)
            del k_b, v_b, dK_b, dV_b, k_out, v_out, kk, vv

        if beta_is_4d:
            dbeta_scaled = dbeta_s.view(B, H, T, S)
            denom_u = denom.unsqueeze(-1)
            dbeta_raw = dbeta_scaled / denom_u
            d_denom = -(dbeta_scaled * beta_raw_f / (denom_u * denom_u)).sum(dim=-1)
        else:
            dbeta_scaled = dbeta_s.sum(dim=-1).view(B, H, T)
            dbeta_raw = dbeta_scaled / denom
            d_denom = -(dbeta_scaled * beta_raw_f / (denom * denom))

        d_frame_inf = d_denom * (frame_inf >= 1.0).float()
        dinf_c = (d_frame_inf / float(S)).unsqueeze(-1).expand(B, H, T, S).reshape(BH, N).contiguous()

        dq_c = dQ.view(BH, N, D).contiguous()
        dk_c = dK.view(BH, N, D).contiguous()
        dv_c = dV.view(BH, N, D).contiguous()

        dqkv_cam = torch.empty_like(qkv_c)
        dkconv = torch.empty((B, N, C), device=dev, dtype=qkv_c.dtype) if has_kconv else None
        n_chunks = 240
        dq_nw_part = torch.empty((n_chunks, C), device=dev, dtype=torch.float32)
        dk_nw_part = torch.empty((n_chunks, C), device=dev, dtype=torch.float32)
        dq_nw_out = torch.empty((C,), device=dev, dtype=torch.float32)
        dk_nw_out = torch.empty((C,), device=dev, dtype=torch.float32)

        # Packed mode: dv_raw=0 signals dqkv_cam is (B,N,3*C); k_ptr=kc_tmp, dk_raw=dkconv
        rc = lib.launch_cam_prep_bwd(
            qkv_c.data_ptr(), kc_tmp.data_ptr() if has_kconv else 0,
            q_nw_f.data_ptr(), k_nw_f.data_ptr(),
            pt_f.data_ptr(), pinv_f.data_ptr(), rc_f.data_ptr(), rs_f.data_ptr(),
            q_ir.data_ptr(), k_ir.data_ptr(), pre_ksq.data_ptr(), post_ksq.data_ptr(),
            dq_c.data_ptr(), dk_c.data_ptr(), dv_c.data_ptr(), dinf_c.data_ptr(),
            dqkv_cam.data_ptr(), dkconv.data_ptr() if has_kconv else 0, 0,
            dq_nw_part.data_ptr(), dk_nw_part.data_ptr(), dq_nw_out.data_ptr(), dk_nw_out.data_ptr(),
            B, N, H, D, k_scale, 1 if use_bf16 else 0, n_chunks, st,
        )
        if rc != 0:
            raise RuntimeError(f"cam_prep_bwd failed rc={rc}")
        del kc_tmp

        dckw = None
        dckb = None
        if has_kconv:
            n_bs_blk = (B * S + 7) // 8
            dw_part = torch.empty((n_bs_blk, C, 4), device=dev, dtype=torch.float32)
            dckw = torch.empty(ckw_shape, device=dev, dtype=ckw_saved.dtype)
            db_part = torch.empty((n_bs_blk, C), device=dev, dtype=torch.float32) if has_kbias else None
            dckb = torch.empty(ckb_shape, device=dev, dtype=ckw_saved.dtype) if has_kbias else None
            rc = lib.launch_bidi_short_conv4_qkv_k_bwd(
                qkv_c.data_ptr(), dkconv.data_ptr(), ckw_saved.data_ptr(), dqkv_cam.data_ptr(),
                dw_part.data_ptr(), db_part.data_ptr() if has_kbias else 0,
                dckw.data_ptr(), dckb.data_ptr() if has_kbias else 0,
                B, T, S, C, 1 if has_kbias else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"cam bidi_short_conv4_qkv_k_bwd failed rc={rc}")

        return (
            dqkv_cam, dckw, dckb,
            dq_nw_out.to(q_nw_dt), dk_nw_out.to(k_nw_dt),
            dbeta_raw.to(beta_orig_dt),
            d_decay.view(B, H, T).to(decay_orig_dt),
            None, None, None, None, None,
            None, None, None, None, None, None, None, None,
        )


def can_use_fused_bidi_gdn(x: torch.Tensor, H: int, D: int) -> bool:
    if os.environ.get("SANA_OPT_FUSED_GDN", "1") == "0":
        return False
    return (
        x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and D == 112
        and (H * D) % 64 == 0
        and H <= 32
    )


def gdn_use_bf16_bmm() -> bool:
    return os.environ.get("SANA_GDN_BMM_DTYPE", "bf16").strip().lower() == "bf16"
