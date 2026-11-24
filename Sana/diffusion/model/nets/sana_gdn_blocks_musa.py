"""Pure-PyTorch (MUSA-compatible, Triton-free) Bidirectional GDN + UCPE camera blocks.

Implements the exact same forward math, module structure, and checkpoint state-dict keys
as ``sana_gdn_blocks_triton.py`` (BidirectionalGDNTriton,
BidirectionalGDNUCPESinglePathLiteLATriton, BidirectionalGDNUCPESinglePathLiteLABothTriton)
using native PyTorch tensor ops (muBLAS on Moore Threads MTT S5000) with standard autograd.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.nets.sana_camctrl_blocks import (
    _maybe_drop_cam_branch,
    _prepare_ray_apply_fns,
    compute_fov_from_fx_xi,
    ucm_unproject_grid_fov,
    world_to_ray_mats,
)
from diffusion.model.nets.sana_gdn_blocks import (
    BidirectionalGDN,
    ChunkCausalGDN,
    flip_and_shift,
)
from diffusion.model.nets.sana_gdn_camctrl_blocks import (
    BidirectionalGDNUCPESinglePathLiteLA,
    ChunkCausalGDNUCPESinglePathLiteLA,
)
from diffusion.model.registry import ATTENTION_BLOCKS


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _process_camera_conditions_raymats_only(
    camera_conditions: torch.Tensor,
    B: int,
    HW: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    F_dim = camera_conditions.shape[1]
    C_to_W = camera_conditions[..., :16].view(B, F_dim, 4, 4)
    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]
    H_dim, W_dim = HW[1], HW[2]
    image_width = W_dim * patch_size[2]
    image_height = H_dim * patch_size[1]
    xi = torch.zeros((B, F_dim), device=camera_conditions.device, dtype=camera_conditions.dtype)
    x_fov = compute_fov_from_fx_xi(fx, xi, image_width, device=camera_conditions.device, dtype=camera_conditions.dtype).view(B, F_dim)
    y_fov = compute_fov_from_fx_xi(fy, xi, image_height, device=camera_conditions.device, dtype=camera_conditions.dtype).view(B, F_dim)
    d_cam = ucm_unproject_grid_fov(
        x_fov, y_fov, xi, H_dim, W_dim,
        cx / patch_size[2], cy / patch_size[1],
        device=camera_conditions.device, dtype=camera_conditions.dtype,
    )
    if d_cam.ndim == 4 and d_cam.shape[0] == B * F_dim:
        d_cam = d_cam.view(B, F_dim, H_dim, W_dim, 3)
    return world_to_ray_mats(d_cam, C_to_W)


def _prepare_rope_tables(rotary_emb, N: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_emb is None:
        return (
            torch.ones(N, D, device=device, dtype=torch.float32),
            torch.zeros(N, D, device=device, dtype=torch.float32),
        )
    freqs = rotary_emb.squeeze(0).squeeze(0)  # (N, D//2) complex
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1).contiguous()
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(N, D).contiguous()
    return rope_cos, rope_sin


def _pair_swap(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return x.unflatten(-1, (d // 2, 2)).flip(-1).flatten(-2)


def _main_gdn_scan_single_dir(
    q_rot: torch.Tensor,  # (B, H, T, D, S) fp32
    q_raw: torch.Tensor,  # (B, H, T, D, S) fp32
    k_rot: torch.Tensor,  # (B, H, T, D, S) fp32
    k_raw: torch.Tensor,  # (B, H, T, D, S) fp32
    v: torch.Tensor,      # (B, H, T, D, S) fp32
    beta: torch.Tensor,   # (B, H, T, S) fp32
    decay: torch.Tensor,  # (B, H, T) fp32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-direction exact frame-wise GDN scan matching _fused_gdn_kernel."""
    B, H, T, D, S = q_rot.shape
    beta_v = beta.unsqueeze(3)          # (B, H, T, 1, S)
    decay_v = decay.view(B, H, T, 1, 1)
    eye = torch.eye(D, device=q_rot.device, dtype=torch.float32).view(1, 1, 1, D, D)

    # Numerator state: (B, H, D_v, D_k)
    k_rot_beta = k_rot * beta_v
    W_kv = decay_v * (eye - torch.matmul(k_rot_beta, k_rot.transpose(-1, -2)))
    U_kv = torch.matmul(v * beta_v, k_rot.transpose(-1, -2))

    # Denominator state: (B, H, 1, D_k)
    k_raw_beta = k_raw * beta_v
    W_z = decay_v * (eye - torch.matmul(k_raw_beta, k_raw.transpose(-1, -2)))
    U_z = k_raw_beta.sum(dim=-1).unsqueeze(-2)  # (B, H, T, 1, D)

    s_kv = torch.zeros(B, H, D, D, device=q_rot.device, dtype=torch.float32)
    s_z = torch.zeros(B, H, 1, D, device=q_rot.device, dtype=torch.float32)
    s_kv_list = []
    s_z_list = []
    for t in range(T):
        s_kv = torch.matmul(s_kv, W_kv[:, :, t]) + U_kv[:, :, t]
        s_z = torch.matmul(s_z, W_z[:, :, t]) + U_z[:, :, t]
        s_kv_list.append(s_kv)
        s_z_list.append(s_z)

    s_kv_all = torch.stack(s_kv_list, dim=2)  # (B, H, T, D, D)
    s_z_all = torch.stack(s_z_list, dim=2)    # (B, H, T, 1, D)

    num_t = torch.matmul(s_kv_all, q_rot)                 # (B, H, T, D, S)
    den_t = torch.matmul(s_z_all, q_raw).squeeze(-2)      # (B, H, T, S)
    return num_t, den_t


def _cam_scan_single_chunk(
    q: torch.Tensor,      # (B, H, D, N) fp32
    k: torch.Tensor,      # (B, H, D, N) fp32
    v: torch.Tensor,      # (B, H, D, N) fp32
    beta: torch.Tensor,   # (B, H, T, S) or (B, H, T) fp32
    decay: torch.Tensor,  # (B, H, T) fp32
) -> torch.Tensor:
    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def to_ft(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_t, k_t, v_t = to_ft(q), to_ft(k), to_ft(v)
    beta_v = beta.unsqueeze(3) if beta.ndim == 4 else beta.view(B, H, T, 1, 1)
    decay_v = decay.view(B, H, T, 1, 1)
    eye = torch.eye(D, device=q.device, dtype=torch.float32).view(1, 1, 1, D, D)

    k_beta = k_t * beta_v
    W = decay_v * (eye - torch.matmul(k_beta, k_t.transpose(-1, -2)))
    U = torch.matmul(v_t * beta_v, k_t.transpose(-1, -2))

    state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    s_list = []
    for t in range(T):
        state = torch.matmul(state, W[:, :, t]) + U[:, :, t]
        s_list.append(state)
    s_all = torch.stack(s_list, dim=2)
    out_t = torch.matmul(s_all, q_t)  # (B, H, T, D, S)
    return out_t.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)


def _cam_scan_bidi(q, k, v, beta, decay) -> torch.Tensor:
    out_fwd = _cam_scan_single_chunk(q, k, v, beta, decay)
    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def to_ft(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    def from_ft(x):
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)

    q_bwd = torch.flip(to_ft(q), dims=[2])
    k_bwd = flip_and_shift(to_ft(k), dim=2, shift_val=0.0)
    v_bwd = flip_and_shift(to_ft(v), dim=2, shift_val=0.0)
    beta_bwd = flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = flip_and_shift(decay, dim=2, shift_val=1.0)
    out_bwd = _cam_scan_single_chunk(from_ft(q_bwd), from_ft(k_bwd), from_ft(v_bwd), beta_bwd, decay_bwd)
    out_bwd = torch.flip(out_bwd.view(B, H, D, T, S), dims=[3]).reshape(B, H, D, N)
    return out_fwd + out_bwd


def _cam_prep_torch(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_scale: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, H, D = q_raw.shape
    C = H * D
    D_half = D // 2
    n_groups = D_half // 4

    q32, k32, v32 = q_raw.float(), k_raw.float(), v_raw.float()
    q_inv_rms = torch.rsqrt((q32 * q32).sum(dim=(-1, -2)) / C + norm_eps)
    k_inv_rms = torch.rsqrt((k32 * k32).sum(dim=(-1, -2)) / C + norm_eps)
    q_nw = q_norm_weight.float().view(1, 1, H, D)
    k_nw = k_norm_weight.float().view(1, 1, H, D)

    q_normed = torch.relu(q32 * q_inv_rms.view(B, N, 1, 1) * q_nw)
    k_normed = torch.relu(k32 * k_inv_rms.view(B, N, 1, 1) * k_nw) * k_scale
    pre_k_sq_BNH = (k_normed * k_normed).sum(dim=-1)

    q_first = q_normed[..., :D_half].reshape(B, N, H, n_groups, 4)
    k_first = k_normed[..., :D_half].reshape(B, N, H, n_groups, 4)
    v_first = v32[..., :D_half].reshape(B, N, H, n_groups, 4)

    pq, pkv = proj_q.float(), proj_kv.float()
    q_first_proj = torch.einsum("bnij,bnhgj->bnhgi", pq, q_first).reshape(B, N, H, D_half)
    k_first_proj = torch.einsum("bnij,bnhgj->bnhgi", pkv, k_first).reshape(B, N, H, D_half)
    v_first_proj = torch.einsum("bnij,bnhgj->bnhgi", pkv, v_first).reshape(B, N, H, D_half)

    q_second, k_second, v_second = q_normed[..., D_half:], k_normed[..., D_half:], v32[..., D_half:]
    cos_b = rope_cos.float().view(1, N, 1, D_half)
    sin_b = rope_sin.float().view(1, N, 1, D_half)
    q_rope = q_second * cos_b + _pair_swap(q_second) * sin_b
    k_rope = k_second * cos_b + _pair_swap(k_second) * sin_b
    v_rope = v_second * cos_b + _pair_swap(v_second) * sin_b

    q_out_BNHD = torch.cat([q_first_proj, q_rope], dim=-1)
    k_out_BNHD = torch.cat([k_first_proj, k_rope], dim=-1)
    v_out_BNHD = torch.cat([v_first_proj, v_rope], dim=-1)

    post_k_sq_BNH = (k_out_BNHD * k_out_BNHD).sum(dim=-1)
    out_dtype = q_raw.dtype
    q_out = q_out_BNHD.to(out_dtype).permute(0, 2, 3, 1).contiguous()
    k_out = k_out_BNHD.to(out_dtype).permute(0, 2, 3, 1).contiguous()
    v_out = v_out_BNHD.to(out_dtype).permute(0, 2, 3, 1).contiguous()

    pre_k_sq = pre_k_sq_BNH.permute(0, 2, 1).contiguous()
    post_k_sq = post_k_sq_BNH.permute(0, 2, 1).contiguous()
    inflation_sq = post_k_sq.clamp_min(1e-12) / pre_k_sq.clamp_min(1e-12)
    return q_out, k_out, v_out, inflation_sq


def _bidi_gdn_main_forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    HW: tuple[int, int, int] | None = None,
    rotary_emb: torch.Tensor | None = None,
    block_mask: torch.Tensor | None = None,
    apply_output_gate: bool = True,
    **kwargs: object,
) -> torch.Tensor:
    del mask, block_mask
    B, N, C = x.shape
    T, H_s, W_s = HW
    S = H_s * W_s
    H, D = self.heads, self.dim

    qkv = self.qkv(x).reshape(B, N, 3, H, D)
    q_raw = qkv[:, :, 0]
    k_raw = qkv[:, :, 1]
    v_raw = qkv[:, :, 2]

    if self.conv_k is not None:
        k_conv = self._apply_temporal_short_conv(k_raw.contiguous().reshape(B, N, C), self.conv_k, HW)
        k_raw = k_conv.reshape(B, N, H, D)

    precomputed_gates = kwargs.get("precomputed_gates", None)
    if precomputed_gates is not None:
        beta, decay = precomputed_gates
    else:
        beta, decay = self._compute_frame_gates(x, HW)
    beta = beta.float()
    decay = decay.float()

    q32, k32, v32 = q_raw.float(), k_raw.float(), v_raw.float()
    if not isinstance(self.q_norm, nn.Identity):
        norm_eps = float(getattr(self.q_norm, "eps", 1e-5))
        q_inv_rms = torch.rsqrt((q32 * q32).sum(dim=(-1, -2)) / C + norm_eps)
        k_inv_rms = torch.rsqrt((k32 * k32).sum(dim=(-1, -2)) / C + norm_eps)
        q_nw = self.q_norm.weight.float().view(1, 1, H, D)
        k_nw = self.k_norm.weight.float().view(1, 1, H, D)
        q32 = q32 * q_inv_rms.view(B, N, 1, 1) * q_nw
        k32 = k32 * k_inv_rms.view(B, N, 1, 1) * k_nw

    k_scale = (D ** -0.5) * (S ** -0.5)
    q_pos = torch.relu(q32)
    k_pos = torch.relu(k32) * k_scale

    rope_cos, rope_sin = _prepare_rope_tables(rotary_emb, N, D, x.device)
    cos_b = rope_cos.view(1, N, 1, D)
    sin_b = rope_sin.view(1, N, 1, D)
    q_rot = q_pos * cos_b + _pair_swap(q_pos) * sin_b
    k_rot = k_pos * cos_b + _pair_swap(k_pos) * sin_b

    # Convert to (B, H, T, D, S)
    def to_ft(t):
        return t.view(B, T, S, H, D).permute(0, 3, 1, 4, 2).contiguous()

    q_rot_t, q_pos_t = to_ft(q_rot), to_ft(q_pos)
    k_rot_t, k_pos_t = to_ft(k_rot), to_ft(k_pos)
    v_t = to_ft(v32)

    num_fwd, den_fwd = _main_gdn_scan_single_dir(q_rot_t, q_pos_t, k_rot_t, k_pos_t, v_t, beta, decay)

    # Reverse exclusive scan (flip_and_shift on k, v, beta, decay)
    q_rot_b = torch.flip(q_rot_t, dims=[2])
    q_pos_b = torch.flip(q_pos_t, dims=[2])
    k_rot_b = flip_and_shift(k_rot_t, dim=2, shift_val=0.0)
    k_pos_b = flip_and_shift(k_pos_t, dim=2, shift_val=0.0)
    v_b = flip_and_shift(v_t, dim=2, shift_val=0.0)
    beta_b = flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_b = flip_and_shift(decay, dim=2, shift_val=1.0)
    num_bwd, den_bwd = _main_gdn_scan_single_dir(q_rot_b, q_pos_b, k_rot_b, k_pos_b, v_b, beta_b, decay_b)
    num_bwd = torch.flip(num_bwd, dims=[2])
    den_bwd = torch.flip(den_bwd, dims=[2])

    num = num_fwd + num_bwd  # (B, H, T, D, S)
    den = den_fwd + den_bwd  # (B, H, T, S)
    out = num / (den.unsqueeze(3) + self.eps)
    out = out.permute(0, 2, 4, 1, 3).reshape(B, N, C).to(x.dtype)

    if apply_output_gate:
        out = self._apply_output_gate(out, x)
        out = self.proj(out.to(self.proj.weight.dtype))
    return out


class BidirectionalGDNMusa(BidirectionalGDN):
    def __init__(self, *args, use_autograd_kernel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autograd_kernel = False

    forward = _bidi_gdn_main_forward


class BidirectionalGDNUCPEMusa(BidirectionalGDNUCPESinglePathLiteLA):
    def __init__(self, *args, use_autograd_kernel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autograd_kernel = False

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        dtype_orig = x.dtype
        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = F.linear(x, qkv_w, qkv_b)
        q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)

        if self.conv_k_cam is not None:
            k_raw = self._apply_temporal_short_conv(k_raw, self.conv_k_cam, HW)

        q_raw = q_raw.contiguous().view(B, N, H_heads, D_head)
        k_raw = k_raw.contiguous().view(B, N, H_heads, D_head)
        v_raw = v_raw.contiguous().view(B, N, H_heads, D_head)

        raymats = _process_camera_conditions_raymats_only(camera_conditions, B, HW, self.patch_size)
        P = raymats.reshape(B, -1, 4, 4)
        P_T = P.transpose(-1, -2).contiguous()
        P_inv = _invert_SE3(P).contiguous()

        if rotary_emb is not None:
            head_dim = D_head
            orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
            orig_h_size = head_dim // 6
            new_head_dim = head_dim // 2
            new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
            new_h_size = new_head_dim // 6
            new_w_size = new_head_dim // 6
            t_part = rotary_emb[..., :new_t_size]
            h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
            w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
            rotary_emb_cam = torch.cat([t_part, h_part, w_part], dim=-1)
            rope_cos, rope_sin = _prepare_rope_tables(rotary_emb_cam, N, D_head // 2, x.device)
        else:
            rotary_emb_cam = None
            rope_cos = torch.ones(N, D_head // 2, device=x.device, dtype=torch.float32)
            rope_sin = torch.zeros(N, D_head // 2, device=x.device, dtype=torch.float32)

        q_norm_w = self.q_norm_cam.weight.float()
        k_norm_w = self.k_norm_cam.weight.float()
        k_scale = (D_head ** -0.5) * (S ** -0.5)
        norm_eps = float(getattr(self.q_norm_cam, "eps", getattr(self.q_norm_cam, "variance_epsilon", 1e-6)))

        q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq = _cam_prep_torch(
            q_raw, k_raw, v_raw,
            q_norm_weight=q_norm_w, k_norm_weight=k_norm_w,
            proj_q=P_T, proj_kv=P_inv,
            rope_cos=rope_cos, rope_sin=rope_sin,
            k_scale=k_scale, norm_eps=norm_eps,
        )

        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        inflation_sq_spatial = inflation_sq.view(B, H_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        out = _cam_scan_bidi(
            q_cam_trans.float(), k_cam_trans.float(), v_cam_trans.float(),
            beta.float(), decay.float(),
        )
        out = out.to(dtype_orig)

        _, _, apply_fn_o = _prepare_ray_apply_fns(
            head_dim=D_head, P=P, P_T=P_T, P_inv=P_inv, rotary_emb=rotary_emb_cam,
        )
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        out = out.reshape(B, self.cam_dim, -1).permute(0, 2, 1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        precomputed_gates = self._compute_frame_gates(x, HW) if HW is not None else None
        main_raw = _bidi_gdn_main_forward(
            self, x, mask=mask, HW=HW, rotary_emb=rotary_emb, block_mask=block_mask,
            apply_output_gate=False, chunk_size=chunk_size, precomputed_gates=precomputed_gates, **kwargs,
        )
        cam_contrib: torch.Tensor | int = 0
        camera_conditions = _maybe_drop_cam_branch(
            camera_conditions, kwargs.get("cam_branch_drop_prob", 0.0), self.training, x.device,
        )
        if camera_conditions is not None:
            cam_raw = self._forward_cam_branch(
                x, HW, camera_conditions, rotary_emb,
                chunk_size=chunk_size, precomputed_gates=precomputed_gates, **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)
        combined = self._apply_output_gate(main_raw + cam_contrib, x)
        return self.proj(combined.to(self.proj.weight.dtype))


def register_musa_gdn_fallbacks():
    for name in ("BidirectionalGDN", "BidirectionalGDNTriton", "ChunkCausalGDN", "ChunkCausalGDNTriton"):
        ATTENTION_BLOCKS.register_module(name=name, force=True, module=BidirectionalGDNMusa)
    for name in (
        "BidirectionalGDNUCPESinglePathLiteLA",
        "BidirectionalGDNUCPESinglePathLiteLATriton",
        "BidirectionalGDNUCPESinglePathLiteLABothTriton",
        "ChunkCausalGDNUCPESinglePathLiteLA",
        "ChunkCausalGDNUCPESinglePathLiteLATriton",
        "ChunkCausalGDNUCPESinglePathLiteLABothTriton",
    ):
        ATTENTION_BLOCKS.register_module(name=name, force=True, module=BidirectionalGDNUCPEMusa)
