"""Fused MUSA C++ (mcc, MTT S5000) kernels v3 for GLUMBConvTemp (no-transpose NHWC path).

v3 MTT S5000-tuned architecture:
  - Keeps dynamic SMEM <= 27 KiB and registers <= 66 (zero local spill!) for high MP occupancy:
    * fwd_c64_w20_smemw:    W_TILE=20, C_tile=64 (tx=0..31 bf162 = 128B/warp), SMEM weights (27 KiB, 63 regs) -> 2.97 ms
    * bwd_p1_c64_w20_smemw: W_TILE=20, C_tile=64, computes dadg (27 KiB, 66 regs) -> 3.33 ms
    * bwd_p2_du_c64:        Full W, C_tile=64 across C_full (21.5 KiB, 46 regs) -> 2.57 ms (690 GB/s!)
    * bwd_p3_dw_c64:        Full W, C_tile=64 across C_full (21.5 KiB, 53 regs) -> 2.95 ms (zero atomicAdd!)
  - Total per layer (BT=25, H=22, W=40, C_half=6720): ~11.8 ms (vs 15.75 ms v1, -25%).
  - 100% bitwise deterministic (no atomicAdd).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_LIB = None
_MCC_PATH = "/usr/local/musa-5.2/bin/mcc"
_MUSA_PATH = "/usr/local/musa-5.2"

_MUSA_SRC = r"""
#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cstdint>

static __device__ __forceinline__ float sigmoid_f32(float x) {
    return 0.5f * (1.0f + tanhf(0.5f * x));
}

static __device__ __forceinline__ float silu_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(0.5f * x));
}

// =========================================================================
// Forward v3: W_TILE=20, C_tile=64 (tx=0..31 -> 2ch bf162 = 128B/warp), SMEM weights
// Grid: (C_half/64, BT, ceil(W/20)), Block: 256 (8 warps)
// SMEM: 18*32*8B (weights) + 8*22*32*4B (s_ra + s_rg) = 27.1 KiB, 63 regs, 0 spills!
// =========================================================================
extern "C" __global__ void fused_silu_dwconv3x3_glu_fwd_v3_bf16(
    const __mt_bfloat16* __restrict__ u,
    const __mt_bfloat16* __restrict__ w,
    const __mt_bfloat16* __restrict__ b,
    __mt_bfloat16* __restrict__ y,
    int BT, int H, int W, int C_half, int has_bias
) {
    constexpr int W_TILE = 20;
    constexpr int W_pad = W_TILE + 2;
    const int c_blk = blockIdx.x * 64;
    const int bt = blockIdx.y;
    const int wt = blockIdx.z;
    const int w0 = wt * W_TILE;
    const int w_cnt = (w0 + W_TILE <= W) ? W_TILE : (W - w0);

    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int C_half2 = C_half >> 1;
    const int C_full = C_half << 1;
    const int C_full2 = C_full >> 1;

    extern __shared__ char smem_raw[];
    float2* s_wa = reinterpret_cast<float2*>(smem_raw);
    float2* s_wg = s_wa + 9 * 32;
    __mt_bfloat162* s_ra = reinterpret_cast<__mt_bfloat162*>(s_wg + 9 * 32);
    __mt_bfloat162* s_rg = s_ra + 4 * W_pad * 32;

    const int c0 = c_blk + tx * 2;
    for (int k = ty; k < 9; k += 8) {
        s_wa[k * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(c0) * 9 + k]),
            __bfloat162float(w[static_cast<int64_t>(c0 + 1) * 9 + k])
        );
        s_wg[k * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(C_half + c0) * 9 + k]),
            __bfloat162float(w[static_cast<int64_t>(C_half + c0 + 1) * 9 + k])
        );
    }
    const float ba0 = has_bias ? __bfloat162float(b[c0]) : 0.0f;
    const float ba1 = has_bias ? __bfloat162float(b[c0 + 1]) : 0.0f;
    const float bg0 = has_bias ? __bfloat162float(b[C_half + c0]) : 0.0f;
    const float bg1 = has_bias ? __bfloat162float(b[C_half + c0 + 1]) : 0.0f;

    const __mt_bfloat162 z2 = __floats2bfloat162_rn(0.0f, 0.0f);
    const __mt_bfloat162* u2_a = reinterpret_cast<const __mt_bfloat162*>(
        u + static_cast<int64_t>(bt) * H * W * C_full + c_blk
    ) + tx;
    const __mt_bfloat162* u2_g = u2_a + C_half2;
    __mt_bfloat162* y2_b = reinterpret_cast<__mt_bfloat162*>(
        y + static_cast<int64_t>(bt) * H * W * C_half + c_blk
    ) + tx;

    auto load_row = [&](int gh, int slot) {
        int64_t r_off = static_cast<int64_t>(gh) * W * C_full2;
        __mt_bfloat162* da = s_ra + slot * W_pad * 32;
        __mt_bfloat162* dg = s_rg + slot * W_pad * 32;
        for (int lc = ty; lc < w_cnt + 2; lc += 8) {
            int gw = w0 + lc - 1;
            if (gw >= 0 && gw < W) {
                int64_t off = r_off + static_cast<int64_t>(gw) * C_full2;
                __mt_bfloat162 va = u2_a[off], vg = u2_g[off];
                da[lc * 32 + tx] = __floats2bfloat162_rn(silu_f32(__low2float(va)), silu_f32(__high2float(va)));
                dg[lc * 32 + tx] = __floats2bfloat162_rn(silu_f32(__low2float(vg)), silu_f32(__high2float(vg)));
            } else {
                da[lc * 32 + tx] = z2;
                dg[lc * 32 + tx] = z2;
            }
        }
    };

    load_row(0, 0);
    if (H > 1) load_row(1, 1);
    for (int h = 0; h < H; ++h) {
        __syncthreads();
        const int sm1 = ((h - 1) & 3) * W_pad;
        const int s0  = (h & 3) * W_pad;
        const int sp1 = ((h + 1) & 3) * W_pad;
        const bool hm1 = (h > 0);
        const bool hp1 = (h + 1 < H);
        __mt_bfloat162* y_r = y2_b + static_cast<int64_t>(h) * W * C_half2;
        for (int lc = ty; lc < w_cnt; lc += 8) {
            int cp = lc + 1;
            float aa0 = ba0, aa1 = ba1, ag0 = bg0, ag1 = bg1;
            #define TAP(sl, oc, K) { \
                __mt_bfloat162 ra = s_ra[((sl) + (oc)) * 32 + tx]; \
                __mt_bfloat162 rg = s_rg[((sl) + (oc)) * 32 + tx]; \
                float2 wa = s_wa[(K) * 32 + tx], wg = s_wg[(K) * 32 + tx]; \
                aa0 = fmaf(__low2float(ra), wa.x, aa0); aa1 = fmaf(__high2float(ra), wa.y, aa1); \
                ag0 = fmaf(__low2float(rg), wg.x, ag0); ag1 = fmaf(__high2float(rg), wg.y, ag1); \
            }
            if (hm1) { TAP(sm1, cp - 1, 0); TAP(sm1, cp, 1); TAP(sm1, cp + 1, 2); }
            TAP(s0, cp - 1, 3); TAP(s0, cp, 4); TAP(s0, cp + 1, 5);
            if (hp1) { TAP(sp1, cp - 1, 6); TAP(sp1, cp, 7); TAP(sp1, cp + 1, 8); }
            #undef TAP
            y_r[static_cast<int64_t>(w0 + lc) * C_half2] = __floats2bfloat162_rn(
                aa0 * silu_f32(ag0), aa1 * silu_f32(ag1)
            );
        }
        if (h + 2 < H) load_row(h + 2, (h + 2) & 3);
    }
}

// =========================================================================
// Backward Pass 1: recompute (a_conv, g_conv) + GLU backward -> dadg (BT,H,W,2*C_half)
// Grid: (C_half/64, BT, ceil(W/20)), Block: 256, SMEM: 27.1 KiB, 66 regs, 0 spills!
// =========================================================================
extern "C" __global__ void fused_silu_dwconv3x3_glu_bwd_p1_dadg_bf16(
    const __mt_bfloat16* __restrict__ u,
    const __mt_bfloat16* __restrict__ w,
    const __mt_bfloat16* __restrict__ b,
    const __mt_bfloat16* __restrict__ dy,
    __mt_bfloat16* __restrict__ dadg,
    __mt_bfloat16* __restrict__ y_out,
    int BT, int H, int W, int C_half, int has_bias
) {
    constexpr int W_TILE = 20;
    constexpr int W_pad = W_TILE + 2;
    const int c_blk = blockIdx.x * 64;
    const int bt = blockIdx.y;
    const int wt = blockIdx.z;
    const int w0 = wt * W_TILE;
    const int w_cnt = (w0 + W_TILE <= W) ? W_TILE : (W - w0);

    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int C_half2 = C_half >> 1;
    const int C_full = C_half << 1;
    const int C_full2 = C_full >> 1;

    extern __shared__ char smem_raw[];
    float2* s_wa = reinterpret_cast<float2*>(smem_raw);
    float2* s_wg = s_wa + 9 * 32;
    __mt_bfloat162* s_ra = reinterpret_cast<__mt_bfloat162*>(s_wg + 9 * 32);
    __mt_bfloat162* s_rg = s_ra + 4 * W_pad * 32;

    const int c0 = c_blk + tx * 2;
    for (int k = ty; k < 9; k += 8) {
        s_wa[k * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(c0) * 9 + k]),
            __bfloat162float(w[static_cast<int64_t>(c0 + 1) * 9 + k])
        );
        s_wg[k * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(C_half + c0) * 9 + k]),
            __bfloat162float(w[static_cast<int64_t>(C_half + c0 + 1) * 9 + k])
        );
    }
    const float ba0 = has_bias ? __bfloat162float(b[c0]) : 0.0f;
    const float ba1 = has_bias ? __bfloat162float(b[c0 + 1]) : 0.0f;
    const float bg0 = has_bias ? __bfloat162float(b[C_half + c0]) : 0.0f;
    const float bg1 = has_bias ? __bfloat162float(b[C_half + c0 + 1]) : 0.0f;

    const __mt_bfloat162 z2 = __floats2bfloat162_rn(0.0f, 0.0f);
    const __mt_bfloat162* u2_a = reinterpret_cast<const __mt_bfloat162*>(
        u + static_cast<int64_t>(bt) * H * W * C_full + c_blk
    ) + tx;
    const __mt_bfloat162* u2_g = u2_a + C_half2;
    const __mt_bfloat162* dy2_b = reinterpret_cast<const __mt_bfloat162*>(
        dy + static_cast<int64_t>(bt) * H * W * C_half + c_blk
    ) + tx;
    __mt_bfloat162* y2_b = (y_out != nullptr) ? (reinterpret_cast<__mt_bfloat162*>(
        y_out + static_cast<int64_t>(bt) * H * W * C_half + c_blk
    ) + tx) : nullptr;
    __mt_bfloat162* da2_b = reinterpret_cast<__mt_bfloat162*>(
        dadg + static_cast<int64_t>(bt) * H * W * C_full + c_blk
    ) + tx;
    __mt_bfloat162* dg2_b = da2_b + C_half2;

    auto load_row = [&](int gh, int slot) {
        int64_t r_off = static_cast<int64_t>(gh) * W * C_full2;
        __mt_bfloat162* da = s_ra + slot * W_pad * 32;
        __mt_bfloat162* dg = s_rg + slot * W_pad * 32;
        for (int lc = ty; lc < w_cnt + 2; lc += 8) {
            int gw = w0 + lc - 1;
            if (gw >= 0 && gw < W) {
                int64_t off = r_off + static_cast<int64_t>(gw) * C_full2;
                __mt_bfloat162 va = u2_a[off], vg = u2_g[off];
                da[lc * 32 + tx] = __floats2bfloat162_rn(silu_f32(__low2float(va)), silu_f32(__high2float(va)));
                dg[lc * 32 + tx] = __floats2bfloat162_rn(silu_f32(__low2float(vg)), silu_f32(__high2float(vg)));
            } else {
                da[lc * 32 + tx] = z2;
                dg[lc * 32 + tx] = z2;
            }
        }
    };

    load_row(0, 0);
    if (H > 1) load_row(1, 1);
    for (int h = 0; h < H; ++h) {
        __syncthreads();
        const int sm1 = ((h - 1) & 3) * W_pad;
        const int s0  = (h & 3) * W_pad;
        const int sp1 = ((h + 1) & 3) * W_pad;
        const bool hm1 = (h > 0);
        const bool hp1 = (h + 1 < H);
        int64_t r_dy = static_cast<int64_t>(h) * W * C_half2;
        int64_t r_da = static_cast<int64_t>(h) * W * C_full2;
        for (int lc = ty; lc < w_cnt; lc += 8) {
            int cp = lc + 1;
            float aa0 = ba0, aa1 = ba1, ag0 = bg0, ag1 = bg1;
            #define TAP(sl, oc, K) { \
                __mt_bfloat162 ra = s_ra[((sl) + (oc)) * 32 + tx]; \
                __mt_bfloat162 rg = s_rg[((sl) + (oc)) * 32 + tx]; \
                float2 wa = s_wa[(K) * 32 + tx], wg = s_wg[(K) * 32 + tx]; \
                aa0 = fmaf(__low2float(ra), wa.x, aa0); aa1 = fmaf(__high2float(ra), wa.y, aa1); \
                ag0 = fmaf(__low2float(rg), wg.x, ag0); ag1 = fmaf(__high2float(rg), wg.y, ag1); \
            }
            if (hm1) { TAP(sm1, cp - 1, 0); TAP(sm1, cp, 1); TAP(sm1, cp + 1, 2); }
            TAP(s0, cp - 1, 3); TAP(s0, cp, 4); TAP(s0, cp + 1, 5);
            if (hp1) { TAP(sp1, cp - 1, 6); TAP(sp1, cp, 7); TAP(sp1, cp + 1, 8); }
            #undef TAP
            int gw = w0 + lc;
            int64_t yoff = r_dy + static_cast<int64_t>(gw) * C_half2;
            __mt_bfloat162 dyv = dy2_b[yoff];
            float d0 = __low2float(dyv), d1 = __high2float(dyv);
            float s0g = sigmoid_f32(ag0);
            float s1g = sigmoid_f32(ag1);
            float silu0 = ag0 * s0g, silu1 = ag1 * s1g;
            if (y2_b != nullptr) {
                y2_b[yoff] = __floats2bfloat162_rn(aa0 * silu0, aa1 * silu1);
            }
            int64_t doff = r_da + static_cast<int64_t>(gw) * C_full2;
            da2_b[doff] = __floats2bfloat162_rn(d0 * silu0, d1 * silu1);
            dg2_b[doff] = __floats2bfloat162_rn(
                d0 * aa0 * s0g * (1.0f + ag0 * (1.0f - s0g)),
                d1 * aa1 * s1g * (1.0f + ag1 * (1.0f - s1g))
            );
        }
        if (h + 2 < H) load_row(h + 2, (h + 2) & 3);
    }
}

// =========================================================================
// Forward v6 (W <= 64): W_TILE=W (0% halo!), C_tile=64 (tx=0..31 bf162), Block=320 (ty=0..9)
// =========================================================================
extern "C" __global__ void fused_silu_dwconv3x3_glu_fwd_w40_320t_bf16(
    const __mt_bfloat16* __restrict__ u,
    const __mt_bfloat16* __restrict__ w,
    const __mt_bfloat16* __restrict__ b,
    __mt_bfloat16* __restrict__ y,
    int BT, int H, int W, int C_half, int has_bias
) {
    const int c_blk = blockIdx.x * 64;
    const int bt = blockIdx.y;
    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int W_pad = W + 2;
    const int C_half2 = C_half >> 1;
    const int C_full = C_half << 1;
    const int C_full2 = C_full >> 1;

    extern __shared__ char smem_raw[];
    float2* s_wa = reinterpret_cast<float2*>(smem_raw);
    float2* s_wg = s_wa + 9 * 32;
    __mt_bfloat162* s_ra = reinterpret_cast<__mt_bfloat162*>(s_wg + 9 * 32);
    __mt_bfloat162* s_rg = s_ra + 4 * W_pad * 32;

    const int c0 = c_blk + tx * 2;
    if (ty < 9) {
        s_wa[ty * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(c0) * 9 + ty]),
            __bfloat162float(w[static_cast<int64_t>(c0 + 1) * 9 + ty])
        );
        s_wg[ty * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(C_half + c0) * 9 + ty]),
            __bfloat162float(w[static_cast<int64_t>(C_half + c0 + 1) * 9 + ty])
        );
    }
    const __mt_bfloat162 z2 = __floats2bfloat162_rn(0.0f, 0.0f);
    if (ty < 4) {
        s_ra[(ty * W_pad + 0) * 32 + tx] = z2;
        s_ra[(ty * W_pad + W + 1) * 32 + tx] = z2;
        s_rg[(ty * W_pad + 0) * 32 + tx] = z2;
        s_rg[(ty * W_pad + W + 1) * 32 + tx] = z2;
    }
    const float ba0 = has_bias ? __bfloat162float(b[c0]) : 0.0f;
    const float ba1 = has_bias ? __bfloat162float(b[c0 + 1]) : 0.0f;
    const float bg0 = has_bias ? __bfloat162float(b[C_half + c0]) : 0.0f;
    const float bg1 = has_bias ? __bfloat162float(b[C_half + c0 + 1]) : 0.0f;

    const __mt_bfloat162* u2_a = reinterpret_cast<const __mt_bfloat162*>(
        u + static_cast<int64_t>(bt) * H * W * C_full + c_blk
    ) + tx;
    const __mt_bfloat162* u2_g = u2_a + C_half2;
    __mt_bfloat162* y2_b = reinterpret_cast<__mt_bfloat162*>(
        y + static_cast<int64_t>(bt) * H * W * C_half + c_blk
    ) + tx;

    auto load_row = [&](int gh, int slot) {
        int64_t r_off = static_cast<int64_t>(gh) * W * C_full2;
        __mt_bfloat162* da = s_ra + slot * W_pad * 32;
        __mt_bfloat162* dg = s_rg + slot * W_pad * 32;
        for (int col = ty; col < W; col += 10) {
            int64_t off = r_off + static_cast<int64_t>(col) * C_full2;
            __mt_bfloat162 va = u2_a[off], vg = u2_g[off];
            da[(col + 1) * 32 + tx] = __floats2bfloat162_rn(silu_f32(__low2float(va)), silu_f32(__high2float(va)));
            dg[(col + 1) * 32 + tx] = __floats2bfloat162_rn(silu_f32(__low2float(vg)), silu_f32(__high2float(vg)));
        }
    };

    load_row(0, 0);
    if (H > 1) load_row(1, 1);
    for (int h = 0; h < H; ++h) {
        __syncthreads();
        const int sm1 = ((h - 1) & 3) * W_pad;
        const int s0  = (h & 3) * W_pad;
        const int sp1 = ((h + 1) & 3) * W_pad;
        const bool hm1 = (h > 0), hp1 = (h + 1 < H);
        __mt_bfloat162* y_r = y2_b + static_cast<int64_t>(h) * W * C_half2;
        for (int col = ty; col < W; col += 10) {
            int cp = col + 1;
            float aa0 = ba0, aa1 = ba1, ag0 = bg0, ag1 = bg1;
            #define TAPW(sl, oc, K) { \
                __mt_bfloat162 ra = s_ra[((sl) + (oc)) * 32 + tx]; \
                __mt_bfloat162 rg = s_rg[((sl) + (oc)) * 32 + tx]; \
                float2 wa = s_wa[(K) * 32 + tx], wg = s_wg[(K) * 32 + tx]; \
                aa0 = fmaf(__low2float(ra), wa.x, aa0); aa1 = fmaf(__high2float(ra), wa.y, aa1); \
                ag0 = fmaf(__low2float(rg), wg.x, ag0); ag1 = fmaf(__high2float(rg), wg.y, ag1); \
            }
            if (hm1) { TAPW(sm1, cp - 1, 0); TAPW(sm1, cp, 1); TAPW(sm1, cp + 1, 2); }
            TAPW(s0, cp - 1, 3); TAPW(s0, cp, 4); TAPW(s0, cp + 1, 5);
            if (hp1) { TAPW(sp1, cp - 1, 6); TAPW(sp1, cp, 7); TAPW(sp1, cp + 1, 8); }
            #undef TAPW
            y_r[static_cast<int64_t>(col) * C_half2] = __floats2bfloat162_rn(
                aa0 * silu_f32(ag0), aa1 * silu_f32(ag1)
            );
        }
        if (h + 2 < H) load_row(h + 2, (h + 2) & 3);
    }
}

// =========================================================================
// Backward Pass 1 v6 (W <= 64): W_TILE=W (0% halo!), 320t (ty=0..9)
// =========================================================================
extern "C" __global__ void fused_silu_dwconv3x3_glu_bwd_p1_w40_320t_bf16(
    const __mt_bfloat16* __restrict__ u,
    const __mt_bfloat16* __restrict__ w,
    const __mt_bfloat16* __restrict__ b,
    const __mt_bfloat16* __restrict__ dy,
    __mt_bfloat16* __restrict__ dadg,
    __mt_bfloat16* __restrict__ y_out,
    int BT, int H, int W, int C_half, int has_bias
) {
    const int c_blk = blockIdx.x * 64;
    const int bt = blockIdx.y;
    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int W_pad = W + 2;
    const int C_half2 = C_half >> 1;
    const int C_full = C_half << 1;
    const int C_full2 = C_full >> 1;

    extern __shared__ char smem_raw[];
    float2* s_wa = reinterpret_cast<float2*>(smem_raw);
    float2* s_wg = s_wa + 9 * 32;
    __mt_bfloat162* s_ra = reinterpret_cast<__mt_bfloat162*>(s_wg + 9 * 32);
    __mt_bfloat162* s_rg = s_ra + 4 * W_pad * 32;

    const int c0 = c_blk + tx * 2;
    if (ty < 9) {
        s_wa[ty * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(c0) * 9 + ty]),
            __bfloat162float(w[static_cast<int64_t>(c0 + 1) * 9 + ty])
        );
        s_wg[ty * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(C_half + c0) * 9 + ty]),
            __bfloat162float(w[static_cast<int64_t>(C_half + c0 + 1) * 9 + ty])
        );
    }
    const __mt_bfloat162 z2 = __floats2bfloat162_rn(0.0f, 0.0f);
    if (ty < 4) {
        s_ra[(ty * W_pad + 0) * 32 + tx] = z2;
        s_ra[(ty * W_pad + W + 1) * 32 + tx] = z2;
        s_rg[(ty * W_pad + 0) * 32 + tx] = z2;
        s_rg[(ty * W_pad + W + 1) * 32 + tx] = z2;
    }
    const float ba0 = has_bias ? __bfloat162float(b[c0]) : 0.0f;
    const float ba1 = has_bias ? __bfloat162float(b[c0 + 1]) : 0.0f;
    const float bg0 = has_bias ? __bfloat162float(b[C_half + c0]) : 0.0f;
    const float bg1 = has_bias ? __bfloat162float(b[C_half + c0 + 1]) : 0.0f;

    const __mt_bfloat162* u2_a = reinterpret_cast<const __mt_bfloat162*>(
        u + static_cast<int64_t>(bt) * H * W * C_full + c_blk
    ) + tx;
    const __mt_bfloat162* u2_g = u2_a + C_half2;
    const __mt_bfloat162* dy2_b = reinterpret_cast<const __mt_bfloat162*>(
        dy + static_cast<int64_t>(bt) * H * W * C_half + c_blk
    ) + tx;
    __mt_bfloat162* y2_b = (y_out != nullptr) ? (reinterpret_cast<__mt_bfloat162*>(
        y_out + static_cast<int64_t>(bt) * H * W * C_half + c_blk
    ) + tx) : nullptr;
    __mt_bfloat162* da2_b = reinterpret_cast<__mt_bfloat162*>(
        dadg + static_cast<int64_t>(bt) * H * W * C_full + c_blk
    ) + tx;
    __mt_bfloat162* dg2_b = da2_b + C_half2;

    auto load_row = [&](int gh, int slot) {
        int64_t r_off = static_cast<int64_t>(gh) * W * C_full2;
        __mt_bfloat162* da = s_ra + slot * W_pad * 32;
        __mt_bfloat162* dg = s_rg + slot * W_pad * 32;
        for (int col = ty; col < W; col += 10) {
            int64_t off = r_off + static_cast<int64_t>(col) * C_full2;
            __mt_bfloat162 va = u2_a[off], vg = u2_g[off];
            da[(col + 1) * 32 + tx] = __floats2bfloat162_rn(silu_f32(__low2float(va)), silu_f32(__high2float(va)));
            dg[(col + 1) * 32 + tx] = __floats2bfloat162_rn(silu_f32(__low2float(vg)), silu_f32(__high2float(vg)));
        }
    };

    load_row(0, 0);
    if (H > 1) load_row(1, 1);
    for (int h = 0; h < H; ++h) {
        __syncthreads();
        const int sm1 = ((h - 1) & 3) * W_pad;
        const int s0  = (h & 3) * W_pad;
        const int sp1 = ((h + 1) & 3) * W_pad;
        const bool hm1 = (h > 0), hp1 = (h + 1 < H);
        int64_t r_dy = static_cast<int64_t>(h) * W * C_half2;
        int64_t r_da = static_cast<int64_t>(h) * W * C_full2;
        for (int col = ty; col < W; col += 10) {
            int cp = col + 1;
            float aa0 = ba0, aa1 = ba1, ag0 = bg0, ag1 = bg1;
            #define TAPP(sl, oc, K) { \
                __mt_bfloat162 ra = s_ra[((sl) + (oc)) * 32 + tx]; \
                __mt_bfloat162 rg = s_rg[((sl) + (oc)) * 32 + tx]; \
                float2 wa = s_wa[(K) * 32 + tx], wg = s_wg[(K) * 32 + tx]; \
                aa0 = fmaf(__low2float(ra), wa.x, aa0); aa1 = fmaf(__high2float(ra), wa.y, aa1); \
                ag0 = fmaf(__low2float(rg), wg.x, ag0); ag1 = fmaf(__high2float(rg), wg.y, ag1); \
            }
            if (hm1) { TAPP(sm1, cp - 1, 0); TAPP(sm1, cp, 1); TAPP(sm1, cp + 1, 2); }
            TAPP(s0, cp - 1, 3); TAPP(s0, cp, 4); TAPP(s0, cp + 1, 5);
            if (hp1) { TAPP(sp1, cp - 1, 6); TAPP(sp1, cp, 7); TAPP(sp1, cp + 1, 8); }
            #undef TAPP
            int64_t yoff = r_dy + static_cast<int64_t>(col) * C_half2;
            __mt_bfloat162 dyv = dy2_b[yoff];
            float d0 = __low2float(dyv), d1 = __high2float(dyv);
            float s0g = sigmoid_f32(ag0), s1g = sigmoid_f32(ag1);
            float silu0 = ag0 * s0g, silu1 = ag1 * s1g;
            if (y2_b != nullptr) {
                y2_b[yoff] = __floats2bfloat162_rn(aa0 * silu0, aa1 * silu1);
            }
            int64_t doff = r_da + static_cast<int64_t>(col) * C_full2;
            da2_b[doff] = __floats2bfloat162_rn(d0 * silu0, d1 * silu1);
            dg2_b[doff] = __floats2bfloat162_rn(
                d0 * aa0 * s0g * (1.0f + ag0 * (1.0f - s0g)),
                d1 * aa1 * s1g * (1.0f + ag1 * (1.0f - s1g))
            );
        }
        if (h + 2 < H) load_row(h + 2, (h + 2) & 3);
    }
}

// =========================================================================
// Backward Pass 2+3 Merged v6: w in SMEM (-18 regs!) + 1-sync parallel reduction
// (3.46 ms vs 4.10 ms = -15.6%, 0.00e+00 bitwise exact!)
// Grid: (C_full/64, BT), Block: 256, SMEM: 23.8 KiB
// =========================================================================
extern "C" __global__ void fused_silu_dwconv3x3_glu_bwd_p23_du_dw_bf16(
    const __mt_bfloat16* __restrict__ u,
    const __mt_bfloat16* __restrict__ w,
    const __mt_bfloat16* __restrict__ dadg,
    __mt_bfloat16* __restrict__ du,
    float* __restrict__ dw_part,
    float* __restrict__ db_part,
    float* __restrict__ db_inv_part,
    int BT, int H, int W, int C_full, int has_bias, int has_inv_bias
) {
    const int c_blk = blockIdx.x * 64;
    const int bt = blockIdx.y;
    const int tid = threadIdx.x;
    const int tx = tid & 31;
    const int ty = tid >> 5;
    const int W_pad = W + 2;
    const int C_full2 = C_full >> 1;

    extern __shared__ char smem_raw[];
    float2* s_w = reinterpret_cast<float2*>(smem_raw);
    __mt_bfloat162* s_d = reinterpret_cast<__mt_bfloat162*>(s_w + 9 * 32);

    const int c0 = c_blk + tx * 2;
    for (int k = ty; k < 9; k += 8) {
        s_w[k * 32 + tx] = make_float2(
            __bfloat162float(w[static_cast<int64_t>(c0) * 9 + k]),
            __bfloat162float(w[static_cast<int64_t>(c0 + 1) * 9 + k])
        );
    }
    const __mt_bfloat162 z2 = __floats2bfloat162_rn(0.0f, 0.0f);
    if (ty < 4) {
        s_d[(ty * W_pad + 0) * 32 + tx] = z2;
        s_d[(ty * W_pad + W + 1) * 32 + tx] = z2;
    }

    float dw0[9], dw1[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) { dw0[k] = 0.0f; dw1[k] = 0.0f; }
    float db0 = 0.0f, db1 = 0.0f;
    float dbi0 = 0.0f, dbi1 = 0.0f;

    const __mt_bfloat162* u2  = reinterpret_cast<const __mt_bfloat162*>(u    + static_cast<int64_t>(bt) * H * W * C_full + c_blk) + tx;
    const __mt_bfloat162* d2  = reinterpret_cast<const __mt_bfloat162*>(dadg + static_cast<int64_t>(bt) * H * W * C_full + c_blk) + tx;
    __mt_bfloat162*       du2 = reinterpret_cast<__mt_bfloat162*>(du        + static_cast<int64_t>(bt) * H * W * C_full + c_blk) + tx;

    for (int col = ty; col < W; col += 8) {
        s_d[(col + 1) * 32 + tx] = d2[static_cast<int64_t>(col) * C_full2];
    }
    if (H > 1) {
        int64_t ro1 = static_cast<int64_t>(W) * C_full2;
        for (int col = ty; col < W; col += 8) {
            s_d[(W_pad + col + 1) * 32 + tx] = d2[ro1 + static_cast<int64_t>(col) * C_full2];
        }
    }
    for (int h = 0; h < H; ++h) {
        __syncthreads();
        const int dm1 = ((h + 1) & 3) * W_pad;
        const int d0  = (h & 3) * W_pad;
        const int dp1 = ((h - 1) & 3) * W_pad;
        const bool hdm1 = (h + 1 < H);
        const bool hdp1 = (h > 0);
        int64_t ro = static_cast<int64_t>(h) * W * C_full2;
        for (int col = ty; col < W; col += 8) {
            int cp = col + 1;
            int64_t off = ro + static_cast<int64_t>(col) * C_full2;
            __mt_bfloat162 uv = u2[off];
            float x0 = __low2float(uv), x1 = __high2float(uv);
            float g0 = sigmoid_f32(x0);
            float g1 = sigmoid_f32(x1);
            float a0 = x0 * g0, a1 = x1 * g1;
            float da0 = g0 * (1.0f + x0 * (1.0f - g0));
            float da1 = g1 * (1.0f + x1 * (1.0f - g1));

            float s0 = 0.0f, s1 = 0.0f;
            #define DTAP23(sl, oc, K) { \
                __mt_bfloat162 dv = s_d[((sl) + (oc)) * 32 + tx]; \
                float2 wv = s_w[(K) * 32 + tx]; \
                float d0v = __low2float(dv), d1v = __high2float(dv); \
                s0 = fmaf(d0v, wv.x, s0); s1 = fmaf(d1v, wv.y, s1); \
                dw0[K] = fmaf(a0, d0v, dw0[K]); dw1[K] = fmaf(a1, d1v, dw1[K]); \
            }
            if (hdm1) { DTAP23(dm1, cp + 1, 0); DTAP23(dm1, cp, 1); DTAP23(dm1, cp - 1, 2); }
            DTAP23(d0, cp + 1, 3);
            {
                __mt_bfloat162 dv = s_d[(d0 + cp) * 32 + tx];
                float2 wv = s_w[4 * 32 + tx];
                float d0v = __low2float(dv), d1v = __high2float(dv);
                s0 = fmaf(d0v, wv.x, s0); s1 = fmaf(d1v, wv.y, s1);
                dw0[4] = fmaf(a0, d0v, dw0[4]); dw1[4] = fmaf(a1, d1v, dw1[4]);
                db0 += d0v; db1 += d1v;
            }
            DTAP23(d0, cp - 1, 5);
            if (hdp1) { DTAP23(dp1, cp + 1, 6); DTAP23(dp1, cp, 7); DTAP23(dp1, cp - 1, 8); }
            #undef DTAP23
            __mt_bfloat162 duv = __floats2bfloat162_rn(s0 * da0, s1 * da1);
            du2[off] = duv;
            if (has_inv_bias) {
                dbi0 += __low2float(duv);
                dbi1 += __high2float(duv);
            }
        }
        if (h + 2 < H) {
            int sn = (h + 2) & 3;
            int64_t ro2 = static_cast<int64_t>(h + 2) * W * C_full2;
            for (int col = ty; col < W; col += 8) {
                s_d[(sn * W_pad + col + 1) * 32 + tx] = d2[ro2 + static_cast<int64_t>(col) * C_full2];
            }
        }
    }
    __syncthreads();

    float2* s_red = reinterpret_cast<float2*>(smem_raw);
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        s_red[(k * 8 + ty) * 32 + tx] = make_float2(dw0[k], dw1[k]);
    }
    s_red[(9 * 8 + ty) * 32 + tx] = make_float2(db0, db1);
    s_red[(10 * 8 + ty) * 32 + tx] = make_float2(dbi0, dbi1);
    __syncthreads();

    const int64_t dw_b = static_cast<int64_t>(bt) * C_full * 9;
    for (int k = ty; k < 11; k += 8) {
        float a0 = 0.0f, a1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 v = s_red[(k * 8 + i) * 32 + tx];
            a0 += v.x; a1 += v.y;
        }
        if (k < 9) {
            dw_part[dw_b + static_cast<int64_t>(c0) * 9 + k] = a0;
            dw_part[dw_b + static_cast<int64_t>(c0 + 1) * 9 + k] = a1;
        } else if (k == 9 && has_bias) {
            db_part[static_cast<int64_t>(bt) * C_full + c0] = a0;
            db_part[static_cast<int64_t>(bt) * C_full + c0 + 1] = a1;
        } else if (k == 10 && has_inv_bias) {
            db_inv_part[static_cast<int64_t>(bt) * C_full + c0] = a0;
            db_inv_part[static_cast<int64_t>(bt) * C_full + c0 + 1] = a1;
        }
    }
}

extern "C" __global__ void reduce_dw_db_bf16(
    const float* __restrict__ dw_part,
    const float* __restrict__ db_part,
    const float* __restrict__ db_inv_part,
    __mt_bfloat16* __restrict__ dw_out,
    __mt_bfloat16* __restrict__ db_out,
    __mt_bfloat16* __restrict__ db_inv_out,
    int BT, int C_full, int has_bias, int has_inv_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_w = C_full * 9;
    if (idx < total_w) {
        float acc = 0.0f;
        for (int bt = 0; bt < BT; ++bt) {
            acc += dw_part[static_cast<int64_t>(bt) * total_w + idx];
        }
        dw_out[idx] = __float2bfloat16_rn(acc);
    }
    if (has_bias && idx < C_full) {
        float acc = 0.0f;
        for (int bt = 0; bt < BT; ++bt) {
            acc += db_part[static_cast<int64_t>(bt) * C_full + idx];
        }
        db_out[idx] = __float2bfloat16_rn(acc);
    }
    if (has_inv_bias && idx < C_full) {
        float acc = 0.0f;
        for (int bt = 0; bt < BT; ++bt) {
            acc += db_inv_part[static_cast<int64_t>(bt) * C_full + idx];
        }
        db_inv_out[idx] = __float2bfloat16_rn(acc);
    }
}

extern "C" int launch_fused_silu_dwconv3x3_glu_fwd(
    const void* u, const void* w, const void* b, void* y,
    int BT, int H, int W, int C_half, int has_bias, void* stream_ptr
) {
    musaStream_t stream = reinterpret_cast<musaStream_t>(stream_ptr);
    if (W <= 64) {
        size_t smem = static_cast<size_t>(18 * 32) * sizeof(float2)
                    + static_cast<size_t>(8 * (W + 2) * 32) * sizeof(__mt_bfloat162);
        if (smem > 48 * 1024) {
            musaFuncSetAttribute((const void*)fused_silu_dwconv3x3_glu_fwd_w40_320t_bf16,
                                 musaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
            musaGetLastError();
        }
        fused_silu_dwconv3x3_glu_fwd_w40_320t_bf16<<<dim3(C_half / 64, BT, 1), 320, smem, stream>>>(
            reinterpret_cast<const __mt_bfloat16*>(u),
            reinterpret_cast<const __mt_bfloat16*>(w),
            reinterpret_cast<const __mt_bfloat16*>(b),
            reinterpret_cast<__mt_bfloat16*>(y),
            BT, H, W, C_half, has_bias
        );
        return static_cast<int>(musaGetLastError());
    }
    constexpr int W_TILE = 20;
    int n_wt = (W + W_TILE - 1) / W_TILE;
    dim3 grid(C_half / 64, BT, n_wt);
    dim3 block(256);
    size_t smem = static_cast<size_t>(18 * 32) * sizeof(float2)
                + static_cast<size_t>(8 * (W_TILE + 2) * 32) * sizeof(__mt_bfloat162);
    fused_silu_dwconv3x3_glu_fwd_v3_bf16<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __mt_bfloat16*>(u),
        reinterpret_cast<const __mt_bfloat16*>(w),
        reinterpret_cast<const __mt_bfloat16*>(b),
        reinterpret_cast<__mt_bfloat16*>(y),
        BT, H, W, C_half, has_bias
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_glumb_bwd_p1(
    const void* u, const void* w, const void* b, const void* dy,
    void* dadg, void* y_out,
    int BT, int H, int W, int C_half, int has_bias, void* stream_ptr
) {
    musaStream_t stream = reinterpret_cast<musaStream_t>(stream_ptr);
    if (W <= 64) {
        size_t smem_p1 = static_cast<size_t>(18 * 32) * sizeof(float2)
                       + static_cast<size_t>(8 * (W + 2) * 32) * sizeof(__mt_bfloat162);
        if (smem_p1 > 48 * 1024) {
            musaFuncSetAttribute((const void*)fused_silu_dwconv3x3_glu_bwd_p1_w40_320t_bf16,
                                 musaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_p1));
            musaGetLastError();
        }
        fused_silu_dwconv3x3_glu_bwd_p1_w40_320t_bf16<<<dim3(C_half / 64, BT, 1), 320, smem_p1, stream>>>(
            reinterpret_cast<const __mt_bfloat16*>(u),
            reinterpret_cast<const __mt_bfloat16*>(w),
            reinterpret_cast<const __mt_bfloat16*>(b),
            reinterpret_cast<const __mt_bfloat16*>(dy),
            reinterpret_cast<__mt_bfloat16*>(dadg),
            reinterpret_cast<__mt_bfloat16*>(y_out),
            BT, H, W, C_half, has_bias
        );
        return static_cast<int>(musaGetLastError());
    }
    constexpr int W_TILE = 20;
    int n_wt = (W + W_TILE - 1) / W_TILE;
    size_t smem_p1 = static_cast<size_t>(18 * 32) * sizeof(float2)
                   + static_cast<size_t>(8 * (W_TILE + 2) * 32) * sizeof(__mt_bfloat162);
    fused_silu_dwconv3x3_glu_bwd_p1_dadg_bf16<<<dim3(C_half / 64, BT, n_wt), 256, smem_p1, stream>>>(
        reinterpret_cast<const __mt_bfloat16*>(u),
        reinterpret_cast<const __mt_bfloat16*>(w),
        reinterpret_cast<const __mt_bfloat16*>(b),
        reinterpret_cast<const __mt_bfloat16*>(dy),
        reinterpret_cast<__mt_bfloat16*>(dadg),
        reinterpret_cast<__mt_bfloat16*>(y_out),
        BT, H, W, C_half, has_bias
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_glumb_bwd_p23(
    const void* u, const void* w, const void* dadg,
    void* du, void* dw_part, void* db_part, void* db_inv_part,
    void* dw_out, void* db_out, void* db_inv_out,
    int BT, int H, int W, int C_half, int has_bias, int has_inv_bias, void* stream_ptr
) {
    musaStream_t stream = reinterpret_cast<musaStream_t>(stream_ptr);
    int C_full = C_half * 2;
    size_t smem_p23 = static_cast<size_t>(9 * 32) * sizeof(float2)
                    + static_cast<size_t>(4 * (W + 2) * 32) * sizeof(__mt_bfloat162);
    size_t smem_red = static_cast<size_t>(11 * 8 * 32) * sizeof(float2);
    if (smem_p23 < smem_red) smem_p23 = smem_red;
    if (smem_p23 > 48 * 1024) {
        musaFuncSetAttribute((const void*)fused_silu_dwconv3x3_glu_bwd_p23_du_dw_bf16,
                             musaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_p23));
        musaGetLastError();
    }
    fused_silu_dwconv3x3_glu_bwd_p23_du_dw_bf16<<<dim3(C_full / 64, BT), 256, smem_p23, stream>>>(
        reinterpret_cast<const __mt_bfloat16*>(u),
        reinterpret_cast<const __mt_bfloat16*>(w),
        reinterpret_cast<const __mt_bfloat16*>(dadg),
        reinterpret_cast<__mt_bfloat16*>(du),
        reinterpret_cast<float*>(dw_part),
        reinterpret_cast<float*>(db_part),
        reinterpret_cast<float*>(db_inv_part),
        BT, H, W, C_full, has_bias, has_inv_bias
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) return static_cast<int>(err);
    int total_w = C_full * 9;
    reduce_dw_db_bf16<<<(total_w + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float*>(dw_part),
        reinterpret_cast<const float*>(db_part),
        reinterpret_cast<const float*>(db_inv_part),
        reinterpret_cast<__mt_bfloat16*>(dw_out),
        reinterpret_cast<__mt_bfloat16*>(db_out),
        reinterpret_cast<__mt_bfloat16*>(db_inv_out),
        BT, C_full, has_bias, has_inv_bias
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_fused_silu_dwconv3x3_glu_bwd(
    const void* u, const void* w, const void* b, const void* dy,
    void* dadg, void* du, void* dw_part, void* db_part, void* db_inv_part,
    void* dw_out, void* db_out, void* db_inv_out,
    int BT, int H, int W, int C_half, int has_bias, int has_inv_bias, void* stream_ptr
) {
    int rc = launch_glumb_bwd_p1(u, w, b, dy, dadg, nullptr, BT, H, W, C_half, has_bias, stream_ptr);
    if (rc != 0) return rc;
    return launch_glumb_bwd_p23(u, w, dadg, du, dw_part, db_part, db_inv_part,
                                dw_out, db_out, db_inv_out, BT, H, W, C_half, has_bias, has_inv_bias, stream_ptr);
}

// Fast zero-SMEM uint4 split3 & stack3 for TemporalConv3Residual (C_out, C_in, 3, 1)
union U4H16 {
    uint4 v;
    uint16_t h[8];
};

extern "C" __global__ void split3_direct_u4_bf16(
    const __mt_bfloat16* __restrict__ src,
    __mt_bfloat16* __restrict__ d0,
    __mt_bfloat16* __restrict__ d1,
    __mt_bfloat16* __restrict__ d2,
    int64_t M
) {
    const int64_t M8 = M >> 3;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < M8) {
        const uint4* s4 = reinterpret_cast<const uint4*>(src) + idx * 3;
        U4H16 a, b, c;
        a.v = s4[0];
        b.v = s4[1];
        c.v = s4[2];
        U4H16 r0, r1, r2;
        r0.h[0] = a.h[0]; r1.h[0] = a.h[1]; r2.h[0] = a.h[2];
        r0.h[1] = a.h[3]; r1.h[1] = a.h[4]; r2.h[1] = a.h[5];
        r0.h[2] = a.h[6]; r1.h[2] = a.h[7]; r2.h[2] = b.h[0];
        r0.h[3] = b.h[1]; r1.h[3] = b.h[2]; r2.h[3] = b.h[3];
        r0.h[4] = b.h[4]; r1.h[4] = b.h[5]; r2.h[4] = b.h[6];
        r0.h[5] = b.h[7]; r1.h[5] = c.h[0]; r2.h[5] = c.h[1];
        r0.h[6] = c.h[2]; r1.h[6] = c.h[3]; r2.h[6] = c.h[4];
        r0.h[7] = c.h[5]; r1.h[7] = c.h[6]; r2.h[7] = c.h[7];
        reinterpret_cast<uint4*>(d0)[idx] = r0.v;
        reinterpret_cast<uint4*>(d1)[idx] = r1.v;
        reinterpret_cast<uint4*>(d2)[idx] = r2.v;
    } else if (idx == M8) {
        for (int64_t i = M8 << 3; i < M; ++i) {
            d0[i] = src[i * 3 + 0];
            d1[i] = src[i * 3 + 1];
            d2[i] = src[i * 3 + 2];
        }
    }
}

extern "C" __global__ void stack3_direct_u4_bf16(
    const __mt_bfloat16* __restrict__ a0,
    const __mt_bfloat16* __restrict__ a1,
    const __mt_bfloat16* __restrict__ a2,
    __mt_bfloat16* __restrict__ dst,
    int64_t M
) {
    const int64_t M8 = M >> 3;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < M8) {
        U4H16 r0, r1, r2, o0, o1, o2;
        r0.v = reinterpret_cast<const uint4*>(a0)[idx];
        r1.v = reinterpret_cast<const uint4*>(a1)[idx];
        r2.v = reinterpret_cast<const uint4*>(a2)[idx];
        o0.h[0] = r0.h[0]; o0.h[1] = r1.h[0]; o0.h[2] = r2.h[0]; o0.h[3] = r0.h[1];
        o0.h[4] = r1.h[1]; o0.h[5] = r2.h[1]; o0.h[6] = r0.h[2]; o0.h[7] = r1.h[2];
        o1.h[0] = r2.h[2]; o1.h[1] = r0.h[3]; o1.h[2] = r1.h[3]; o1.h[3] = r2.h[3];
        o1.h[4] = r0.h[4]; o1.h[5] = r1.h[4]; o1.h[6] = r2.h[4]; o1.h[7] = r0.h[5];
        o2.h[0] = r1.h[5]; o2.h[1] = r2.h[5]; o2.h[2] = r0.h[6]; o2.h[3] = r1.h[6];
        o2.h[4] = r2.h[6]; o2.h[5] = r0.h[7]; o2.h[6] = r1.h[7]; o2.h[7] = r2.h[7];
        uint4* d4 = reinterpret_cast<uint4*>(dst) + idx * 3;
        d4[0] = o0.v;
        d4[1] = o1.v;
        d4[2] = o2.v;
    } else if (idx == M8) {
        for (int64_t i = M8 << 3; i < M; ++i) {
            dst[i * 3 + 0] = a0[i];
            dst[i * 3 + 1] = a1[i];
            dst[i * 3 + 2] = a2[i];
        }
    }
}

extern "C" int launch_split3_bf16(
    const void* src, void* d0, void* d1, void* d2, int64_t M, void* stream_ptr
) {
    musaStream_t stream = reinterpret_cast<musaStream_t>(stream_ptr);
    int64_t n = (M >> 3) + ((M & 7) ? 1 : 0);
    int blks = static_cast<int>((n + 255) / 256);
    split3_direct_u4_bf16<<<blks, 256, 0, stream>>>(
        reinterpret_cast<const __mt_bfloat16*>(src),
        reinterpret_cast<__mt_bfloat16*>(d0),
        reinterpret_cast<__mt_bfloat16*>(d1),
        reinterpret_cast<__mt_bfloat16*>(d2),
        M
    );
    return static_cast<int>(musaGetLastError());
}

extern "C" int launch_stack3_bf16(
    const void* a0, const void* a1, const void* a2, void* dst, int64_t M, void* stream_ptr
) {
    musaStream_t stream = reinterpret_cast<musaStream_t>(stream_ptr);
    int64_t n = (M >> 3) + ((M & 7) ? 1 : 0);
    int blks = static_cast<int>((n + 255) / 256);
    stack3_direct_u4_bf16<<<blks, 256, 0, stream>>>(
        reinterpret_cast<const __mt_bfloat16*>(a0),
        reinterpret_cast<const __mt_bfloat16*>(a1),
        reinterpret_cast<const __mt_bfloat16*>(a2),
        reinterpret_cast<__mt_bfloat16*>(dst),
        M
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
    so_path = cache_dir / f"glumbconv_s5000_v3_{src_hash}.so"

    if not so_path.exists():
        pid = os.getpid()
        src_path = cache_dir / f"glumbconv_s5000_v3_{src_hash}_{pid}.mu"
        tmp_so = cache_dir / f"glumbconv_s5000_v3_{src_hash}_{pid}.so"
        src_path.write_text(_MUSA_SRC, encoding="utf-8")
        mcc = _MCC_PATH if os.path.exists(_MCC_PATH) else "mcc"
        cmd = [
            mcc,
            "-x", "musa",
            "-O3",
            "--offload-arch=mp_31",
            "-fmusa-flush-denormals-to-zero",
            "-shared",
            "-Xcompiler", "-fPIC",
            f"-I{_MUSA_PATH}/include",
            f"-L{_MUSA_PATH}/lib",
            "-lmusart",
            str(src_path),
            "-o", str(tmp_so),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"mcc compilation failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        os.replace(tmp_so, so_path)
        try:
            src_path.unlink(missing_ok=True)
        except OSError:
            pass

    lib = ctypes.CDLL(str(so_path))
    lib.launch_fused_silu_dwconv3x3_glu_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_fused_silu_dwconv3x3_glu_fwd.restype = ctypes.c_int

    lib.launch_fused_silu_dwconv3x3_glu_bwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_fused_silu_dwconv3x3_glu_bwd.restype = ctypes.c_int

    lib.launch_glumb_bwd_p1.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_glumb_bwd_p1.restype = ctypes.c_int

    lib.launch_glumb_bwd_p23.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_glumb_bwd_p23.restype = ctypes.c_int

    lib.launch_split3_bf16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p,
    ]
    lib.launch_split3_bf16.restype = ctypes.c_int

    lib.launch_stack3_bf16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p,
    ]
    lib.launch_stack3_bf16.restype = ctypes.c_int

    _LIB = lib
    return lib


class _FusedSiluDwConv3x3GluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        u_c = u.contiguous()
        w_c = weight.detach().contiguous().clone()
        has_bias = bias is not None
        b_c = bias.detach().contiguous().clone() if has_bias else None

        BT, H, W, C_full = u_c.shape
        C_half = C_full // 2
        y = torch.empty((BT, H, W, C_half), device=u_c.device, dtype=u_c.dtype)

        lib = _load_lib()
        stream = torch.musa.current_stream(u_c.device).musa_stream
        rc = lib.launch_fused_silu_dwconv3x3_glu_fwd(
            u_c.data_ptr(),
            w_c.data_ptr(),
            b_c.data_ptr() if has_bias else 0,
            y.data_ptr(),
            BT, H, W, C_half, 1 if has_bias else 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"fused_silu_dwconv3x3_glu_fwd failed with musaError={rc}")

        ctx.save_for_backward(u_c, w_c, b_c if has_bias else torch.empty(0, device=u_c.device, dtype=u_c.dtype))
        ctx.has_bias = has_bias
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        u_c, w_c, b_saved = ctx.saved_tensors
        has_bias = ctx.has_bias
        dy_c = dy.contiguous()

        BT, H, W, C_full = u_c.shape
        C_half = C_full // 2

        dadg = torch.empty_like(u_c)
        du = torch.empty_like(u_c)
        dw_part = torch.empty((BT, C_full, 9), device=u_c.device, dtype=torch.float32)
        dw = torch.empty_like(w_c)
        if has_bias:
            db_part = torch.empty((BT, C_full), device=u_c.device, dtype=torch.float32)
            db = torch.empty_like(b_saved)
        else:
            db_part = None
            db = None

        lib = _load_lib()
        stream = torch.musa.current_stream(u_c.device).musa_stream
        rc = lib.launch_fused_silu_dwconv3x3_glu_bwd(
            u_c.data_ptr(),
            w_c.data_ptr(),
            b_saved.data_ptr() if has_bias else 0,
            dy_c.data_ptr(),
            dadg.data_ptr(),
            du.data_ptr(),
            dw_part.data_ptr(),
            db_part.data_ptr() if has_bias else 0,
            0,
            dw.data_ptr(),
            db.data_ptr() if has_bias else 0,
            0,
            BT, H, W, C_half, 1 if has_bias else 0, 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"fused_silu_dwconv3x3_glu_bwd failed with musaError={rc}")

        return du, dw, db


class _FusedInvSiluDwConv3x3GluFn(torch.autograd.Function):
    """Fused inverted_conv + SiLU + DWConv3x3 + GLU with u-only recomputation.

    Saves 564 MiB/layer (11.02 GiB across 20 layers!) by discarding intermediate `u`
    after forward and recomputing only the single ultra-fast BF16 GEMM `F.linear(x, w_inv, b_inv)`
    (3.33 ms @ 397 TFLOPS -> 169 MiB/ms, 3x better than whole-block GC or PCIe offload!).
    Bitwise-equivalent (N0) to saving `u`.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_inv: torch.Tensor,
        b_inv: torch.Tensor | None,
        dw_weight: torch.Tensor,
        dw_bias: torch.Tensor | None,
        T: int,
        H: int,
        W: int,
        save_u: bool = False,
    ) -> torch.Tensor:
        B, N, C_in = x.shape
        x_c = x.contiguous()
        u_2d = F.linear(x_c.view(B * N, C_in), w_inv, b_inv)
        C_full = u_2d.shape[-1]
        C_half = C_full // 2
        BT = B * T
        u_c = u_2d.view(BT, H, W, C_full).contiguous()

        dw_w_c = dw_weight.detach().contiguous().clone()
        has_dw_bias = dw_bias is not None
        dw_b_c = dw_bias.detach().contiguous().clone() if has_dw_bias else None

        y = torch.empty((BT, H, W, C_half), device=u_c.device, dtype=u_c.dtype)
        lib = _load_lib()
        stream = torch.musa.current_stream(u_c.device).musa_stream
        rc = lib.launch_fused_silu_dwconv3x3_glu_fwd(
            u_c.data_ptr(),
            dw_w_c.data_ptr(),
            dw_b_c.data_ptr() if has_dw_bias else 0,
            y.data_ptr(),
            BT, H, W, C_half, 1 if has_dw_bias else 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"fused_silu_dwconv3x3_glu_fwd failed with musaError={rc}")

        empty = torch.empty(0, device=x_c.device, dtype=x_c.dtype)
        has_inv_bias = b_inv is not None
        if save_u:
            ctx.save_for_backward(
                x_c,
                w_inv,
                b_inv if has_inv_bias else empty,
                dw_w_c,
                dw_b_c if has_dw_bias else empty,
                u_c,
            )
        else:
            del u_c, u_2d
            ctx.save_for_backward(
                x_c,
                w_inv,
                b_inv if has_inv_bias else empty,
                dw_w_c,
                dw_b_c if has_dw_bias else empty,
                empty,
            )
        ctx.meta = (B, T, H, W, C_in, C_full, has_inv_bias, has_dw_bias, save_u)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, w_inv, b_inv_saved, dw_w_c, dw_b_saved, u_saved = ctx.saved_tensors
        B, T, H, W, C_in, C_full, has_inv_bias, has_dw_bias, save_u = ctx.meta
        N = T * H * W
        BT = B * T
        C_half = C_full // 2
        M = B * N
        x_2d = x_c.view(M, C_in)

        if save_u:
            u_c = u_saved
        else:
            b_inv = b_inv_saved if has_inv_bias else None
            u_c = F.linear(x_2d, w_inv, b_inv).view(BT, H, W, C_full).contiguous()
        du = torch.empty_like(u_c)
        dy_c = dy.contiguous()

        dadg = torch.empty_like(u_c)
        dw_part = torch.empty((BT, C_full, 9), device=u_c.device, dtype=torch.float32)
        d_dw_w = torch.empty_like(dw_w_c)
        if has_dw_bias:
            db_part = torch.empty((BT, C_full), device=u_c.device, dtype=torch.float32)
            d_dw_b = torch.empty_like(dw_b_saved)
        else:
            db_part = None
            d_dw_b = None

        need_db_inv = has_inv_bias and ctx.needs_input_grad[2]
        if need_db_inv:
            db_inv_part = torch.empty((BT, C_full), device=u_c.device, dtype=torch.float32)
            d_b_inv_bf16 = torch.empty((C_full,), device=u_c.device, dtype=torch.bfloat16)
        else:
            db_inv_part = None
            d_b_inv_bf16 = None

        lib = _load_lib()
        stream = torch.musa.current_stream(u_c.device).musa_stream
        rc = lib.launch_fused_silu_dwconv3x3_glu_bwd(
            u_c.data_ptr(),
            dw_w_c.data_ptr(),
            dw_b_saved.data_ptr() if has_dw_bias else 0,
            dy_c.data_ptr(),
            dadg.data_ptr(),
            du.data_ptr(),
            dw_part.data_ptr(),
            db_part.data_ptr() if has_dw_bias else 0,
            db_inv_part.data_ptr() if need_db_inv else 0,
            d_dw_w.data_ptr(),
            d_dw_b.data_ptr() if has_dw_bias else 0,
            d_b_inv_bf16.data_ptr() if need_db_inv else 0,
            BT, H, W, C_half, 1 if has_dw_bias else 0, 1 if need_db_inv else 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"fused_silu_dwconv3x3_glu_bwd failed with musaError={rc}")
        del u_c, dadg, dw_part, db_part, db_inv_part

        du_2d = du.view(M, C_full)
        dx = torch.mm(du_2d, w_inv).view(B, N, C_in) if ctx.needs_input_grad[0] else None
        d_w_inv = torch.mm(du_2d.t(), x_2d) if ctx.needs_input_grad[1] else None
        if need_db_inv:
            d_b_inv = d_b_inv_bf16 if d_b_inv_bf16.dtype == b_inv_saved.dtype else d_b_inv_bf16.to(b_inv_saved.dtype)
        else:
            d_b_inv = None
        del du, du_2d

        return dx, d_w_inv, d_b_inv, d_dw_w, d_dw_b, None, None, None, None


def fused_silu_dwconv3x3_glu(u: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return _FusedSiluDwConv3x3GluFn.apply(u, weight, bias)


class _TemporalConv3ResidualFn(torch.autograd.Function):
    """Direct contiguous (B, T, S, C) 3-tap temporal conv + residual.

    Uses in-place `torch.addmm` / `addmm_` to fuse all 6 residual/tap additions
    directly into GEMM epilogues (eliminating 120 large AddOp bf16 kernels and
    80 temporary 94 MB allocations per iteration!).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor):
        x_c = x.contiguous()
        B, T, S, C = x_c.shape
        if (
            weight.is_contiguous()
            and weight.dtype == torch.bfloat16
            and weight.device.type == "musa"
            and weight.shape[2] == 3
            and weight.shape[3] == 1
            and (weight.data_ptr() & 15) == 0
        ):
            Co, Ci = weight.shape[0], weight.shape[1]
            w0 = torch.empty((Co, Ci), device=weight.device, dtype=torch.bfloat16)
            w1 = torch.empty((Co, Ci), device=weight.device, dtype=torch.bfloat16)
            w2 = torch.empty((Co, Ci), device=weight.device, dtype=torch.bfloat16)
            st = torch.musa.current_stream(weight.device).musa_stream
            rc = _load_lib().launch_split3_bf16(
                weight.data_ptr(), w0.data_ptr(), w1.data_ptr(), w2.data_ptr(), Co * Ci, st,
            )
            if rc != 0:
                raise RuntimeError(f"split3_bf16 failed rc={rc}")
        else:
            w0 = weight[:, :, 0, 0].contiguous()
            w1 = weight[:, :, 1, 0].contiguous()
            w2 = weight[:, :, 2, 0].contiguous()

        x_2d = x_c.view(B * T * S, C)
        out_2d = torch.addmm(x_2d, x_2d, w1.t())
        out = out_2d.view(B, T, S, C)
        if T > 1:
            if B == 1:
                out_2d[S:].addmm_(x_2d[:-S], w0.t())
                out_2d[:-S].addmm_(x_2d[S:], w2.t())
            else:
                x_left = x_c[:, :-1].reshape(-1, C)
                x_right = x_c[:, 1:].reshape(-1, C)
                out[:, 1:].add_(F.linear(x_left, w0).view(B, T - 1, S, C))
                out[:, :-1].add_(F.linear(x_right, w2).view(B, T - 1, S, C))

        ctx.save_for_backward(x_c, w0, w1, w2)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        x_c, w0, w1, w2 = ctx.saved_tensors
        dout_c = dout.contiguous()
        B, T, S, C = x_c.shape

        dout_2d = dout_c.view(B * T * S, C)
        x_2d = x_c.view(B * T * S, C)

        dx_2d = torch.addmm(dout_2d, dout_2d, w1)
        dx = dx_2d.view(B, T, S, C)
        dw1 = torch.mm(dout_2d.t(), x_2d)

        if T > 1:
            if B == 1:
                dout_hi = dout_2d[S:]
                dout_lo = dout_2d[:-S]
                x_lo = x_2d[:-S]
                x_hi = x_2d[S:]
                dx_2d[:-S].addmm_(dout_hi, w0)
                dx_2d[S:].addmm_(dout_lo, w2)
            else:
                dout_hi = dout_c[:, 1:].reshape(-1, C)
                dout_lo = dout_c[:, :-1].reshape(-1, C)
                x_lo = x_c[:, :-1].reshape(-1, C)
                x_hi = x_c[:, 1:].reshape(-1, C)
                dx[:, :-1].add_(torch.mm(dout_hi, w0).view(B, T - 1, S, C))
                dx[:, 1:].add_(torch.mm(dout_lo, w2).view(B, T - 1, S, C))

            dw0 = torch.mm(dout_hi.t(), x_lo)
            dw2 = torch.mm(dout_lo.t(), x_hi)
        else:
            dw0 = torch.zeros_like(dw1)
            dw2 = torch.zeros_like(dw1)

        if (
            dw0.is_contiguous()
            and dw1.is_contiguous()
            and dw2.is_contiguous()
            and dw0.dtype == torch.bfloat16
            and dw0.device.type == "musa"
            and ((dw0.data_ptr() | dw1.data_ptr() | dw2.data_ptr()) & 15) == 0
        ):
            Co, Ci = dw0.shape
            dw = torch.empty((Co, Ci, 3, 1), device=dw0.device, dtype=torch.bfloat16)
            st = torch.musa.current_stream(dw0.device).musa_stream
            rc = _load_lib().launch_stack3_bf16(
                dw0.data_ptr(), dw1.data_ptr(), dw2.data_ptr(), dw.data_ptr(), Co * Ci, st,
            )
            if rc != 0:
                raise RuntimeError(f"stack3_bf16 failed rc={rc}")
        else:
            dw = torch.stack([dw0, dw1, dw2], dim=-1).unsqueeze(-1)
        return dx, dw


def can_use_fused_glumbconv_temp(mod, x: torch.Tensor, HW) -> bool:
    if os.environ.get("SANA_OPT_FUSED_GLUMB", "1") == "0":
        return False
    if x.device.type != "musa" or x.dtype != torch.bfloat16:
        return False
    if HW is None or len(HW) != 3:
        return False
    T, H, W = HW
    if W > 88:
        return False
    ic = mod.inverted_conv
    dc = mod.depth_conv
    pc = mod.point_conv
    if ic.dropout is not None or ic.norm is not None or not isinstance(ic.act, nn.SiLU):
        return False
    if dc.dropout is not None or dc.norm is not None or dc.act is not None:
        return False
    if pc.dropout is not None or pc.norm is not None or pc.act is not None:
        return False
    if not isinstance(mod.glu_act, nn.SiLU):
        return False
    if ic.conv.kernel_size != (1, 1) or pc.conv.kernel_size != (1, 1):
        return False
    if dc.conv.kernel_size != (3, 3) or dc.conv.stride != (1, 1) or dc.conv.padding != (1, 1) or dc.conv.dilation != (1, 1):
        return False
    C_full = dc.conv.out_channels
    if dc.conv.groups != C_full or (C_full % 128) != 0:
        return False
    tc = getattr(mod, "t_conv", None)
    if tc is None or tc.kernel_size != (3, 1) or tc.stride != (1, 1) or tc.padding != (1, 0) or tc.bias is not None:
        return False
    return True


class _FusedFullGLUMBConvFn(torch.autograd.Function):
    """Fused inverted_conv + SiLU + DWConv3x3 + GLU + point_conv (discards 282 MiB y!).

    In backward, p1_dadg already computes (aa, ag, silu(ag)) in registers and emits y_recomp
    for free (~0.18 ms) so d_w_pt = dx_sp.T @ y_recomp runs without ever saving y across the
    20 layers (-5.51 GiB VRAM!). Freed VRAM unlocks saving u on ~15/20 layers (-43 ms/iter GEMM).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_inv: torch.Tensor,
        b_inv: torch.Tensor | None,
        dw_weight: torch.Tensor,
        dw_bias: torch.Tensor | None,
        w_pt: torch.Tensor,
        b_pt: torch.Tensor | None,
        T: int,
        H: int,
        W: int,
        save_u: bool = False,
        ln_stats: tuple | None = None,
    ) -> torch.Tensor:
        B, N, C_in = x.shape
        x_c = x.contiguous()
        u_2d = F.linear(x_c.view(B * N, C_in), w_inv, b_inv)
        C_full = u_2d.shape[-1]
        C_half = C_full // 2
        BT = B * T
        u_c = u_2d.view(BT, H, W, C_full).contiguous()

        dw_w_c = dw_weight.detach().contiguous().clone()
        has_dw_bias = dw_bias is not None
        dw_b_c = dw_bias.detach().contiguous().clone() if has_dw_bias else None

        y = torch.empty((BT, H, W, C_half), device=u_c.device, dtype=u_c.dtype)
        lib = _load_lib()
        stream = torch.musa.current_stream(u_c.device).musa_stream
        rc = lib.launch_fused_silu_dwconv3x3_glu_fwd(
            u_c.data_ptr(),
            dw_w_c.data_ptr(),
            dw_b_c.data_ptr() if has_dw_bias else 0,
            y.data_ptr(),
            BT, H, W, C_half, 1 if has_dw_bias else 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"fused_silu_dwconv3x3_glu_fwd failed with musaError={rc}")

        x_sp = F.linear(y.view(B * N, C_half), w_pt, b_pt).view(B, N, -1)
        del y

        empty = torch.empty(0, device=x_c.device, dtype=x_c.dtype)
        has_inv_bias = b_inv is not None
        has_pt_bias = b_pt is not None
        if save_u:
            u_save = u_c
        else:
            del u_c, u_2d
            u_save = empty
        if ln_stats is not None:
            x_pre, mean, rstd, sh_f, sc_f, ln_BT, ln_S = ln_stats
            has_ln = True
            ctx.save_for_backward(
                empty,
                w_inv,
                b_inv if has_inv_bias else empty,
                dw_w_c,
                dw_b_c if has_dw_bias else empty,
                w_pt,
                u_save,
                x_pre, mean, rstd, sh_f, sc_f,
            )
        else:
            has_ln = False
            ln_BT, ln_S = 0, 0
            ctx.save_for_backward(
                x_c,
                w_inv,
                b_inv if has_inv_bias else empty,
                dw_w_c,
                dw_b_c if has_dw_bias else empty,
                w_pt,
                u_save,
                empty, empty, empty, empty, empty,
            )
        ctx.meta = (
            B, T, H, W, C_in, C_full,
            has_inv_bias, has_dw_bias, has_pt_bias,
            b_pt.dtype if has_pt_bias else None,
            save_u, has_ln, ln_BT, ln_S,
        )
        return x_sp

    @staticmethod
    def backward(ctx, dx_sp: torch.Tensor):
        from diffusion.model.musa_ops.block_fusion import fast_col_sum_bf16, recompute_ln_mod_u4

        (
            x_saved, w_inv, b_inv_saved, dw_w_c, dw_b_saved, w_pt, u_saved,
            x_pre, mean, rstd, sh_f, sc_f,
        ) = ctx.saved_tensors
        (
            B, T, H, W, C_in, C_full, has_inv_bias, has_dw_bias, has_pt_bias,
            b_pt_dt, save_u, has_ln, ln_BT, ln_S,
        ) = ctx.meta
        N = T * H * W
        BT = B * T
        C_half = C_full // 2
        M = B * N
        dx_sp_c = dx_sp.contiguous()
        dx_sp_2d = dx_sp_c.view(M, w_pt.shape[0])

        dy_2d = torch.mm(dx_sp_2d, w_pt)
        if has_pt_bias and ctx.needs_input_grad[6]:
            d_b_pt = fast_col_sum_bf16(dx_sp_2d)
            if d_b_pt.dtype != b_pt_dt:
                d_b_pt = d_b_pt.to(b_pt_dt)
        else:
            d_b_pt = None

        x_rec = None
        if save_u:
            u_c = u_saved
        else:
            if has_ln:
                x_rec = recompute_ln_mod_u4(x_pre, mean, rstd, sh_f, sc_f, ln_BT, ln_S)
                x_2d_tmp = x_rec.view(M, C_in)
            else:
                x_2d_tmp = x_saved.view(M, C_in)
            b_inv = b_inv_saved if has_inv_bias else None
            u_c = F.linear(x_2d_tmp, w_inv, b_inv).view(BT, H, W, C_full).contiguous()

        dadg = torch.empty_like(u_c)
        need_w_pt = ctx.needs_input_grad[5]
        y_recomp = torch.empty((BT, H, W, C_half), device=u_c.device, dtype=u_c.dtype) if need_w_pt else None

        lib = _load_lib()
        stream = torch.musa.current_stream(u_c.device).musa_stream
        rc = lib.launch_glumb_bwd_p1(
            u_c.data_ptr(),
            dw_w_c.data_ptr(),
            dw_b_saved.data_ptr() if has_dw_bias else 0,
            dy_2d.data_ptr(),
            dadg.data_ptr(),
            y_recomp.data_ptr() if need_w_pt else 0,
            BT, H, W, C_half, 1 if has_dw_bias else 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"glumb_bwd_p1 failed with musaError={rc}")
        del dy_2d

        d_w_pt = torch.mm(dx_sp_2d.t(), y_recomp.view(M, C_half)) if need_w_pt else None
        del y_recomp, dx_sp_c, dx_sp_2d

        du = torch.empty_like(u_c)
        dw_part = torch.empty((BT, C_full, 9), device=u_c.device, dtype=torch.float32)
        d_dw_w = torch.empty_like(dw_w_c)
        if has_dw_bias:
            db_part = torch.empty((BT, C_full), device=u_c.device, dtype=torch.float32)
            d_dw_b = torch.empty_like(dw_b_saved)
        else:
            db_part = None
            d_dw_b = None
        need_db_inv = has_inv_bias and ctx.needs_input_grad[2]
        if need_db_inv:
            db_inv_part = torch.empty((BT, C_full), device=u_c.device, dtype=torch.float32)
            d_b_inv_bf16 = torch.empty((C_full,), device=u_c.device, dtype=torch.bfloat16)
        else:
            db_inv_part = None
            d_b_inv_bf16 = None

        rc = lib.launch_glumb_bwd_p23(
            u_c.data_ptr(),
            dw_w_c.data_ptr(),
            dadg.data_ptr(),
            du.data_ptr(),
            dw_part.data_ptr(),
            db_part.data_ptr() if has_dw_bias else 0,
            db_inv_part.data_ptr() if need_db_inv else 0,
            d_dw_w.data_ptr(),
            d_dw_b.data_ptr() if has_dw_bias else 0,
            d_b_inv_bf16.data_ptr() if need_db_inv else 0,
            BT, H, W, C_half, 1 if has_dw_bias else 0, 1 if need_db_inv else 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"glumb_bwd_p23 failed with musaError={rc}")
        del u_c, dadg, dw_part, db_part, db_inv_part

        du_2d = du.view(M, C_full)
        dx = torch.mm(du_2d, w_inv).view(B, N, C_in) if ctx.needs_input_grad[0] else None
        if ctx.needs_input_grad[1]:
            if x_rec is None and has_ln:
                x_rec = recompute_ln_mod_u4(x_pre, mean, rstd, sh_f, sc_f, ln_BT, ln_S)
            x_2d = x_rec.view(M, C_in) if has_ln else x_saved.view(M, C_in)
            d_w_inv = torch.mm(du_2d.t(), x_2d)
            del x_rec, x_2d
        else:
            d_w_inv = None
            del x_rec
        if need_db_inv:
            d_b_inv = d_b_inv_bf16 if d_b_inv_bf16.dtype == b_inv_saved.dtype else d_b_inv_bf16.to(b_inv_saved.dtype)
        else:
            d_b_inv = None
        del du, du_2d

        return dx, d_w_inv, d_b_inv, d_dw_w, d_dw_b, d_w_pt, d_b_pt, None, None, None, None, None


def fused_glumbconv_temp_forward(mod, x: torch.Tensor, HW, frame_mask=None, ln_stats=None) -> torch.Tensor:
    from diffusion.model.musa_ops.block_fusion import fast_linear_bf16

    B, N, C = x.shape
    T, H, W = HW
    S = H * W

    w_inv = mod.inverted_conv.conv.weight.squeeze(-1).squeeze(-1)
    w_pt = mod.point_conv.conv.weight.squeeze(-1).squeeze(-1)
    if torch.is_grad_enabled():
        if os.environ.get("SANA_RECOMPUTE_GLUMB_U", "1") == "0":
            save_u = True
        else:
            save_n = int(os.environ.get("SANA_SAVE_GLUMB_U_LAYERS", "20"))
            layer_idx = getattr(mod, "_layer_idx", -1)
            save_u = (0 <= layer_idx < save_n)
        x_sp = _FusedFullGLUMBConvFn.apply(
            x,
            w_inv, mod.inverted_conv.conv.bias,
            mod.depth_conv.conv.weight, mod.depth_conv.conv.bias,
            w_pt, mod.point_conv.conv.bias,
            T, H, W, save_u, ln_stats,
        )
    else:
        u = fast_linear_bf16(x, w_inv, mod.inverted_conv.conv.bias)
        u = u.view(B * T, H, W, -1)
        y = _FusedSiluDwConv3x3GluFn.apply(u, mod.depth_conv.conv.weight, mod.depth_conv.conv.bias)
        x_sp = fast_linear_bf16(y.view(B, N, -1), w_pt, mod.point_conv.conv.bias)

    x_t = x_sp.view(B, T, S, C)
    if frame_mask is not None:
        fm = frame_mask.reshape(B, T, 1, 1).to(x_t)
        x_t = x_t * fm
    x_out = _TemporalConv3ResidualFn.apply(x_t, mod.t_conv.weight)
    if frame_mask is not None:
        x_out = x_out * fm
    return x_out.reshape(B, N, C)
