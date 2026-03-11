"""
cuTile GDN Prefill Kernel for Qwen3 Next GDN - Full Implementation (v2)

This module implements the cuTile-based GDN chunked prefill kernels,
replacing the 6-stage Triton pipeline with 4 fused cuTile kernels:

1. cutile_chunk_cumsum: Per-chunk cumulative sum of g
2. cutile_chunk_prepare: Fused KKT + solve_tril + WY recompute (3-in-1)
3. cutile_chunk_recurrence: Hidden state recurrence (h update)
4. cutile_chunk_output: Output computation (o = q@h + intra-chunk)

Based on the Triton implementation in:
- sglang/python/sglang/srt/layers/attention/fla/chunk.py

Target: Match Triton precision, achieve 50%+ speedup on B200
"""

import math
from typing import Optional, Tuple

import torch
import numpy as np

# Pre-import solve_tril and l2norm from Triton for hybrid approach
try:
    from sglang.srt.layers.attention.fla.solve_tril import solve_tril as triton_solve_tril
    from sglang.srt.layers.attention.fla.solve_tril import (
        solve_tril_16x16_kernel,
        merge_16x16_to_64x64_inverse_kernel,
    )
    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd as triton_l2norm_fwd
    from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd as triton_recompute_w_u_fwd
except ImportError:
    triton_solve_tril = None
    solve_tril_16x16_kernel = None
    merge_16x16_to_64x64_inverse_kernel = None
    triton_l2norm_fwd = None
    triton_recompute_w_u_fwd = None

try:
    import cuda.tile as ct
    from cuda.tile import Constant as Const
    CUTILE_AVAILABLE = True
except ImportError:
    CUTILE_AVAILABLE = False
    ct = None
    Const = None

ConstInt = ct.Constant[int] if ct else None
ConstFloat = ct.Constant[float] if ct else None
ConstBool = ct.Constant[bool] if ct else None
PAD_ZERO = ct.PaddingMode.ZERO if ct else None


# ============================================================================
# Kernel 4: Output computation
# o = scale * (q @ h + causal_attn(q, k, v_new))
# where causal_attn = (q @ k^T) * exp(g_i - g_j) * causal_mask @ v_new
#
# Grid: (cdiv(V, BV), NT, B*H)
# ============================================================================
@ct.kernel
def cutile_output_kernel(
    q,          # [B, T, H, K]
    k,          # [B, T, H, K]
    v_new,      # [B, T, H, V]
    h,          # [B, NT, H, K, V]
    g_cumsum,   # [B, T, H] (float32)
    o,          # [B, T, H, V] output
    scale: ConstFloat,
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Compute output: o = scale * (q @ h + causal_attn(q, k, v_new)).

    Grid: (cdiv(V, BV), NT, B*H)
    """
    i_v = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // H
    i_h = i_bh % H

    # Accumulators
    b_o = ct.zeros((BT, BV), dtype=ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)

    for i_k in range(ct.cdiv(K, BK)):
        # Load q: [1, BT, 1, BK] -> [BT, BK]
        q_tile = ct.load(
            q, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))

        # Load k: [1, BT, 1, BK] -> [BT, BK], then transpose -> [BK, BT]
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        k_tile_t = ct.transpose(k_tile)  # [BK, BT]

        # Load h: [1, 1, 1, BK, BV] -> [BK, BV]
        h_tile = ct.load(
            h, index=(i_b, i_t, i_h, i_k, i_v), shape=(1, 1, 1, BK, BV),
            padding_mode=PAD_ZERO,
        ).reshape((BK, BV))

        # o += q @ h: [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o = ct.mma(q_tile, h_tile, b_o)
        # A += q @ k^T: [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A = ct.mma(q_tile, k_tile_t, b_A)

    # Apply gating
    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1),
        padding_mode=PAD_ZERO,
    ).reshape((BT,)).astype(ct.float32)

    # o *= exp(g)
    b_o = b_o * ct.exp(b_g)[:, None]

    # A *= exp(g_i - g_j) (safe)
    g_i = b_g[:, None]
    g_j = b_g[None, :]
    g_diff = g_i - g_j
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff)

    # Causal mask: row >= col
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    causal_mask = row_idx >= col_idx
    b_A = ct.where(causal_mask, b_A, 0.0)

    # Load v_new: [1, BT, 1, BV] -> [BT, BV]
    v_tile = ct.load(
        v_new, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
        padding_mode=PAD_ZERO,
    ).reshape((BT, BV))

    # o = scale * (o + A @ v_new)
    b_o = ct.mma(b_A.astype(v_tile.dtype), v_tile, b_o) * scale

    # Store output: [BT, BV] -> [1, BT, 1, BV]
    ct.store(
        o, index=(i_b, i_t, i_h, i_v),
        tile=b_o.astype(o.dtype).reshape((1, BT, 1, BV)),
    )


# ============================================================================
# Kernel 4b: Output computation with occupancy=2 (better for large grids)
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_output_kernel_occ2(
    q,          # [B, T, H, K]
    k,          # [B, T, H, K]
    v_new,      # [B, T, H, V]
    h,          # [B, NT, H, K, V]
    g_cumsum,   # [B, T, H] (float32)
    o,          # [B, T, H, V] output
    scale: ConstFloat,
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Output kernel with occupancy=2 hint for better large-grid performance."""
    i_v = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // H
    i_h = i_bh % H

    b_o = ct.zeros((BT, BV), dtype=ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)

    for i_k in range(ct.cdiv(K, BK)):
        q_tile = ct.load(q, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        h_tile = ct.load(h, index=(i_b, i_t, i_h, i_k, i_v), shape=(1, 1, 1, BK, BV),
                         padding_mode=PAD_ZERO).reshape((BK, BV))
        b_o = ct.mma(q_tile, h_tile, b_o)
        b_A = ct.mma(q_tile, ct.transpose(k_tile), b_A)

    b_g = ct.load(g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                  padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_o = b_o * ct.exp(b_g)[:, None]
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff)
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    b_A = ct.where(row_idx >= col_idx, b_A, 0.0)

    v_tile = ct.load(v_new, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                     padding_mode=PAD_ZERO).reshape((BT, BV))
    b_o = ct.mma(b_A.astype(v_tile.dtype), v_tile, b_o) * scale

    ct.store(o, index=(i_b, i_t, i_h, i_v),
             tile=b_o.astype(o.dtype).reshape((1, BT, 1, BV)))

# ============================================================================
# Kernel 2aa: Output with Q-cached in registers
# Same grid as multiV: (NT, B*H). Caches all Q K-block tiles in registers
# (32KB bf16 for K=256, BK=64) to eliminate Q re-loads across V-blocks.
# Saves 16 q-tile reloads per CTA (from 20 to 4) — 33% bandwidth reduction.
# Uses occ=1 to give the kernel full 256KB register budget.
# ============================================================================
@ct.kernel(occupancy=1)
def cutile_output_qcached_kernel(
    q,          # [B, T, H, K]
    k,          # [B, T, H, K]
    v_new,      # [B, T, H, V]
    h,          # [B, NT, H, K, V]
    g_cumsum,   # [B, T, H] (float32)
    o,          # [B, T, H, V] output
    scale: ConstFloat,
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Output kernel with Q cached in registers (occ=1).
    Grid: (NT, B*H). Loads Q once upfront and reuses for both QK^T and Q@H.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load g_cumsum and precompute g-gate matrices
    b_g = ct.load(g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                  padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_exp_g_diff = ct.exp(g_diff)         # [BT, BT] — applied to QK^T
    b_exp_g     = ct.exp(b_g)             # [BT]     — applied to Q@H output
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]

    # ── Cache all Q K-block tiles in registers (32 KB for K=256) ──────────
    q1 = ct.load(q, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BK),
                 padding_mode=PAD_ZERO).reshape((BT, BK))
    if K > 64:
        q2 = ct.load(q, index=(i_b, i_t, i_h, 1), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
    if K > 128:
        q3 = ct.load(q, index=(i_b, i_t, i_h, 2), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
    if K > 192:
        q4 = ct.load(q, index=(i_b, i_t, i_h, 3), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))

    # ── Compute QK^T using cached Q (load K on-the-fly) ───────────────────
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    k1 = ct.load(k, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BK),
                 padding_mode=PAD_ZERO).reshape((BT, BK))
    b_A = ct.mma(q1, ct.transpose(k1), b_A)
    if K > 64:
        k2 = ct.load(k, index=(i_b, i_t, i_h, 1), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(q2, ct.transpose(k2), b_A)
    if K > 128:
        k3 = ct.load(k, index=(i_b, i_t, i_h, 2), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(q3, ct.transpose(k3), b_A)
    if K > 192:
        k4 = ct.load(k, index=(i_b, i_t, i_h, 3), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(q4, ct.transpose(k4), b_A)

    # Apply g-gating and causal mask to QK^T
    b_A = b_A * b_exp_g_diff
    b_A = ct.where(row_idx >= col_idx, b_A, 0.0)

    # ── For each V-tile: Q@H (cached Q) + A@v_new ─────────────────────────
    for i_v in range(ct.cdiv(V, BV)):
        b_o = ct.zeros((BT, BV), dtype=ct.float32)

        # Q@H: load H tiles, use cached Q
        h1 = ct.load(h, index=(i_b, i_t, i_h, 0, i_v), shape=(1, 1, 1, BK, BV),
                     padding_mode=PAD_ZERO).reshape((BK, BV))
        b_o = ct.mma(q1, h1, b_o)
        if K > 64:
            h2 = ct.load(h, index=(i_b, i_t, i_h, 1, i_v), shape=(1, 1, 1, BK, BV),
                         padding_mode=PAD_ZERO).reshape((BK, BV))
            b_o = ct.mma(q2, h2, b_o)
        if K > 128:
            h3 = ct.load(h, index=(i_b, i_t, i_h, 2, i_v), shape=(1, 1, 1, BK, BV),
                         padding_mode=PAD_ZERO).reshape((BK, BV))
            b_o = ct.mma(q3, h3, b_o)
        if K > 192:
            h4 = ct.load(h, index=(i_b, i_t, i_h, 3, i_v), shape=(1, 1, 1, BK, BV),
                         padding_mode=PAD_ZERO).reshape((BK, BV))
            b_o = ct.mma(q4, h4, b_o)

        b_o = b_o * b_exp_g[:, None]

        # Add causal intra-chunk attention: A @ v_new
        v_tile = ct.load(v_new, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                         padding_mode=PAD_ZERO).reshape((BT, BV))
        b_o = ct.mma(b_A.astype(v_tile.dtype), v_tile, b_o) * scale

        ct.store(o, index=(i_b, i_t, i_h, i_v),
                 tile=b_o.astype(o.dtype).reshape((1, BT, 1, BV)))

# ============================================================================
# Kernel 2aa-occ2: Output with Q cached in registers (occupancy=2)
# Same body as cutile_output_qcached_kernel but with occ=2 for medium grids
# (512 CTAs → 1.73 waves vs 3.46 at occ=1). Budget: 128KB. Q=32KB fits.
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_output_qcached_occ2_kernel(
    q,          # [B, T, H, K]
    k,          # [B, T, H, K]
    v_new,      # [B, T, H, V]
    h,          # [B, NT, H, K, V]
    g_cumsum,   # [B, T, H] (float32)
    o,          # [B, T, H, V] output
    scale: ConstFloat,
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Output kernel with Q cached in registers (occ=2).
    Grid: (NT, B*H). Loads Q once upfront and reuses for both QK^T and Q@H.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load g_cumsum and precompute g-gate matrices
    b_g = ct.load(g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                  padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_exp_g_diff = ct.exp(g_diff)
    b_exp_g     = ct.exp(b_g)
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]

    # Cache all Q K-block tiles in registers (32 KB for K=256)
    q1 = ct.load(q, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BK),
                 padding_mode=PAD_ZERO).reshape((BT, BK))
    if K > 64:
        q2 = ct.load(q, index=(i_b, i_t, i_h, 1), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
    if K > 128:
        q3 = ct.load(q, index=(i_b, i_t, i_h, 2), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
    if K > 192:
        q4 = ct.load(q, index=(i_b, i_t, i_h, 3), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))

    # Compute QK^T using cached Q (load K on-the-fly)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    k1 = ct.load(k, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BK),
                 padding_mode=PAD_ZERO).reshape((BT, BK))
    b_A = ct.mma(q1, ct.transpose(k1), b_A)
    if K > 64:
        k2 = ct.load(k, index=(i_b, i_t, i_h, 1), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(q2, ct.transpose(k2), b_A)
    if K > 128:
        k3 = ct.load(k, index=(i_b, i_t, i_h, 2), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(q3, ct.transpose(k3), b_A)
    if K > 192:
        k4 = ct.load(k, index=(i_b, i_t, i_h, 3), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(q4, ct.transpose(k4), b_A)

    # Apply g-gating and causal mask to QK^T
    b_A = b_A * b_exp_g_diff
    b_A = ct.where(row_idx >= col_idx, b_A, 0.0)

    # For each V-tile: Q@H (cached Q) + A@v_new
    for i_v in range(ct.cdiv(V, BV)):
        b_o = ct.zeros((BT, BV), dtype=ct.float32)

        # Q@H: load H tiles, use cached Q
        h1 = ct.load(h, index=(i_b, i_t, i_h, 0, i_v), shape=(1, 1, 1, BK, BV),
                     padding_mode=PAD_ZERO).reshape((BK, BV))
        b_o = ct.mma(q1, h1, b_o)
        if K > 64:
            h2 = ct.load(h, index=(i_b, i_t, i_h, 1, i_v), shape=(1, 1, 1, BK, BV),
                         padding_mode=PAD_ZERO).reshape((BK, BV))
            b_o = ct.mma(q2, h2, b_o)
        if K > 128:
            h3 = ct.load(h, index=(i_b, i_t, i_h, 2, i_v), shape=(1, 1, 1, BK, BV),
                         padding_mode=PAD_ZERO).reshape((BK, BV))
            b_o = ct.mma(q3, h3, b_o)
        if K > 192:
            h4 = ct.load(h, index=(i_b, i_t, i_h, 3, i_v), shape=(1, 1, 1, BK, BV),
                         padding_mode=PAD_ZERO).reshape((BK, BV))
            b_o = ct.mma(q4, h4, b_o)

        b_o = b_o * b_exp_g[:, None]

        # Add causal intra-chunk attention: A @ v_new
        v_tile = ct.load(v_new, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                         padding_mode=PAD_ZERO).reshape((BT, BV))
        b_o = ct.mma(b_A.astype(v_tile.dtype), v_tile, b_o) * scale

        ct.store(o, index=(i_b, i_t, i_h, i_v),
                 tile=b_o.astype(o.dtype).reshape((1, BT, 1, BV)))


# ============================================================================
# L2 norm kernel
# ============================================================================
@ct.kernel
def cutile_l2norm_kernel(
    x,      # [M, D] input
    y,      # [M, D] output
    M: ConstInt,
    D: ConstInt,
    BD: ConstInt,
    eps: ConstFloat,
):
    """L2 normalize each row of x.
    Grid: (M,)
    """
    i_m = ct.bid(0)

    # Load row: [1, BD] -> [BD]
    b_x = ct.load(x, index=(i_m, 0), shape=(1, BD), padding_mode=PAD_ZERO).reshape((BD,)).astype(ct.float32)
    # Compute L2 norm
    b_var = ct.sum(b_x * b_x)
    b_rstd = 1.0 / ct.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    # Store: [BD] -> [1, BD]
    ct.store(y, index=(i_m, 0), tile=b_y.astype(y.dtype).reshape((1, BD)))


def cutile_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize the last dimension of x."""
    x_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    M, D = x_flat.shape
    BD = 1 << (D - 1).bit_length()  # next power of 2
    y = torch.empty_like(x_flat)
    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        cutile_l2norm_kernel,
        (x_flat, y, M, D, BD, eps),
    )
    return y.reshape(x_shape)


# ============================================================================
# Tensor pre-allocation cache (reduces allocation overhead ~10-15us per call)
# ============================================================================
_tensor_cache = {}
_cache_config = None  # (B, T, H, K, V, device, dtype) tuple for invalidation

def _init_cache(B, T, H, K, V, device, dtype_k, dtype_v):
    """Pre-allocate all intermediate tensors for the given config."""
    global _tensor_cache, _cache_config
    key = (B, T, H, K, V, device, dtype_k, dtype_v)
    if _cache_config == key:
        return  # Already initialized for this config
    _cache_config = key
    BT = 64
    NT = (T + BT - 1) // BT
    _tensor_cache = {
        'g_cumsum': torch.empty(B, T, H, dtype=torch.float32, device=device),
        'A': torch.empty(B, T, H, BT, dtype=torch.float32, device=device),
        'A_inv': torch.empty(B, T, H, BT, dtype=dtype_k, device=device),
        'Ad': torch.empty(B, T, H, 16, dtype=torch.float32, device=device),  # For cached solve
        'h': torch.empty(B, NT, H, K, V, dtype=dtype_k, device=device),
        'v_new': torch.empty(B, T, H, V, dtype=dtype_v, device=device),
        'o': torch.empty(B, T, H, V, dtype=dtype_v, device=device),
        'w': torch.empty(B, T, H, K, dtype=dtype_k, device=device),
        'u': torch.empty(B, T, H, V, dtype=dtype_v, device=device),
    }


def _solve_tril_cached(A, Ad, Ai, output_dtype):
    """Cached solve_tril using .run() for lower Python overhead."""
    import triton
    B, T, H, BT = A.shape
    BH = B * H
    solve_tril_16x16_kernel.run(
        A, Ad, None, None,
        T, H, BT, False,
        grid=(triton.cdiv(T, 16), BH),
        warmup=False,
        num_warps=1, num_stages=4,
    )
    merge_16x16_to_64x64_inverse_kernel.run(
        A, Ad, Ai, None, None,
        T, H, BT, False,
        grid=(triton.cdiv(T, BT), BH),
        warmup=False,
        num_warps=4, num_stages=3,
    )
    return Ai

def _recompute_w_u_cached(k, v, beta, g_cumsum, A_inv, w_out, u_out):
    """Cached WY recompute using .run() for lower Python overhead."""
    import triton
    from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd_kernel
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A_inv.shape[-1]
    recompute_w_u_fwd_kernel.run(
        k, v, beta, w_out, u_out, A_inv, g_cumsum,
        None, None,
        T, H, k.shape[-2], K, V, BT, 64, 64,
        False,
        grid=(triton.cdiv(T, BT), B * H),
        warmup=False,
        num_warps=8, num_stages=4,
    )
    return w_out, u_out

# ============================================================================
# Main entry: chunk_gated_delta_rule_cutile
# ============================================================================
# Cached stream to avoid repeated torch.cuda.current_stream() calls
_cached_stream = None

def chunk_gated_delta_rule_cutile(
    q: torch.Tensor,        # [B, T, H, K]
    k: torch.Tensor,        # [B, T, H, K]
    v: torch.Tensor,        # [B, T, H, V]
    g: torch.Tensor,        # [B, T, H]
    beta: torch.Tensor,     # [B, T, H]
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,  # [N, H, K, V]
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """cuTile implementation of chunk_gated_delta_rule for GDN prefill."""
    global _cached_stream
    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]
    BH = B * H
    NT = (T + 63) >> 6  # T // 64, BT=64 always

    if scale is None:
        scale = K ** -0.5

    # L2 normalize q and k if requested
    if use_qk_l2norm_in_kernel:
        q = triton_l2norm_fwd(q) if triton_l2norm_fwd is not None else cutile_l2norm(q)
        k = triton_l2norm_fwd(k) if triton_l2norm_fwd is not None else cutile_l2norm(k)

    if _cached_stream is None:
        _cached_stream = torch.cuda.current_stream()
    stream = _cached_stream
    BK = 64 if K >= 64 else K
    total_blocks = NT * BH
    USE_INITIAL = initial_state is not None

    # Pre-allocated intermediates
    _init_cache(B, T, H, K, V, k.device, k.dtype, v.dtype)
    c = _tensor_cache
    g_cumsum = c['g_cumsum']
    A_inv = c['A_inv']
    h = c['h']
    v_new = c['v_new']

    o = c['o']

    # Stage 1+2+3: fused cumsum+KKT+10-step squaring solve
    # occ=3 for large grids (NT*BH >= 512, 1.15 waves on B200 444-cap)
    # occ=2 for medium grids (NT*BH >= 256)
    if NT * BH >= 512:
        ckkt_kernel = cutile_fused_ckkt_solve_v2_occ3_kernel
    elif NT * BH >= 256:
        ckkt_kernel = cutile_fused_ckkt_solve_v2_occ2_kernel
    else:
        ckkt_kernel = cutile_fused_ckkt_solve_v2_kernel

    def _dispatch_output():
        if (V + 63) // 64 * NT * BH >= 512:
            # BV=128 when occ=2 applies or BH≥64 (2 V-iters, 128KB budget); BV=64 otherwise
            out_BV = 128 if (BH >= 64 or NT * BH >= 256) else 64
            # occ=2 for NT*BH>=256 (halves waves); BV=128 fits in 128KB budget
            out_kernel = cutile_output_qcached_occ2_kernel if NT * BH >= 256 else cutile_output_qcached_kernel
            ct.launch(stream, (NT, BH), out_kernel,
                      (q, k, v_new, h, g_cumsum, o, scale, T, H, K, V, 64, 64, out_BV))
        else:
            out_blocks = (V + 63) // 64 * NT * BH
            ct.launch(stream, ((V + 63) // 64, NT, BH),
                      cutile_output_kernel_occ2 if out_blocks >= 256 else cutile_output_kernel,
                      (q, k, v_new, h, g_cumsum, o, scale, T, H, K, V, 64, 128, 64))

    # Stage 4+5+6: TritonWY + adaptive-BV rec + output
    # Adaptive BV: tune CTA count for optimal rec throughput on B200 (148 SMs, occ2 = 296 capacity)
    # For BH≥64 (large batch): allow up to 512 CTAs (BV=32 is more efficient than BV=64 for large BH)
    # For BH≤8 (small batch, e.g. B=2,H=4): cap at 128 CTAs forcing BV=16 (empirically 13% faster)
    # For 8<BH<64: keep CTAs ≤ 256 (empirically optimal for BH=16..32)
    rec_BV = 8
    # For BH≤4 with long sequences (NT>32, T>2048): cap at 64 CTAs to force BV=16, using occ2
    # V<=128: smaller V means BV=32 at 256 CTAs is feasible; V>128: allow 512 CTAs for BV=32
    # V<=128: BV=32 at 256 CTAs is optimal (0.87 waves vs 1.73 at 512); V>128: need 512 for BV=32
    # V<=128: BV=32 at 256 CTAs is optimal (0.87 waves vs 1.73 at 512); V>128: need 512 for BV=32
    max_rec_CTAs = (256 if V <= 128 else 512) if BH >= 64 else (64 if (BH <= 4 and NT > 32) else (128 if BH <= 8 else 256))
    while (V + rec_BV - 1) // rec_BV * BH > max_rec_CTAs and rec_BV < 64:
        rec_BV *= 2
    n_v = (V + rec_BV - 1) // rec_BV
    rec_CTAs = n_v * BH
    # occ3 helps for BH≤4 with small BV (≤8) at ≤128 CTAs (B=1 short sequences)
    # BV≥16 configs use occ2 (register spilling occurs with occ3/occ4 even at BV=16)
    rec_kernel = cutile_recurrence_kernel_bv16_occ3 if (rec_CTAs <= 128 and BH <= 4 and rec_BV <= 8) else cutile_recurrence_kernel_bv16

    ct.launch(stream, (NT, BH), ckkt_kernel,
              (g, k, beta, g_cumsum, A_inv, T, H, K, 64, BK))

    # For fused path: if BV>32 but BV=32 with more CTAs is possible, prefer that
    fused_BV = rec_BV
    fused_n_v = n_v
    if K <= 128 and BH >= 64 and rec_BV > 32 and V <= 128:
        fused_BV = 32
        fused_n_v = (V + 31) // 32
    use_fused = K <= 128 and BH >= 64 and fused_BV <= 32 and NT * BH >= 512
    if use_fused:
        # Fused WY+rec: w, u computed on-the-fly per chunk, never in HBM.
        # Saves w+u write+read bandwidth (~128MB for large configs at 8TB/s = ~16us).
        ct.launch(stream, (fused_n_v, BH), cutile_fused_wy_rec_kernel,
                  (k, v, beta, g_cumsum, A_inv, h, v_new,
                   initial_state, initial_state_indices, T, H, K, V, 64, fused_BV, NT, USE_INITIAL))
    else:
        _recompute_w_u_cached(k, v, beta, g_cumsum, A_inv, c['w'], c['u'])
        ct.launch(stream, (n_v, BH), rec_kernel,
                  (k, c['w'], c['u'], g_cumsum, h, v_new,
                   initial_state, initial_state_indices, T, H, K, V, 64, rec_BV, NT, USE_INITIAL))
    _dispatch_output()

    return o, None, h


# ============================================================================
# Kernel 3 v4: BV=16 recurrence for better SM utilization on small grids
# When CTA count is low (e.g. B=1,H=4,V=256 -> 32 CTAs with BV=32),
# using BV=16 doubles CTAs (64) for better occupancy on B200 (192 SMs).
# 1.30x speedup for B=1,T=4096; 1.27x for B=2,T=2048.
# Worse for high-CTA configs (>256 CTAs) due to overhead.
# Grid: (cdiv(V, 16), B*H)
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_recurrence_kernel_bv16(
    k,              # [B, T, H, K]
    w,              # [B, T, H, K]
    u,              # [B, T, H, V]
    g_cumsum,       # [B, T, H] (float32)
    h,              # [B, NT, H, K, V] output hidden states
    v_new,          # [B, T, H, V] output v_new
    initial_state,  # [N, H, K, V]
    initial_state_indices,  # [B] int32
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BV: ConstInt,    # block size for V dimension (16)
    NT: ConstInt,
    USE_INITIAL_STATE: ConstBool,
):
    """BV=16 recurrence for more CTA parallelism on small grids."""
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // H
    i_h = i_nh % H
    BK = 64
    b_h1 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h2 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h3 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h4 = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
        b_h1 = b_h1 + ct.load(initial_state, index=(idx, i_h, 0, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 64:
            b_h2 = b_h2 + ct.load(initial_state, index=(idx, i_h, 1, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 128:
            b_h3 = b_h3 + ct.load(initial_state, index=(idx, i_h, 2, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 192:
            b_h4 = b_h4 + ct.load(initial_state, index=(idx, i_h, 3, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
    for i_t in range(NT):
        ct.store(h, index=(i_n, i_t, i_h, 0, i_v), tile=b_h1.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 64:
            ct.store(h, index=(i_n, i_t, i_h, 1, i_v), tile=b_h2.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 128:
            ct.store(h, index=(i_n, i_t, i_h, 2, i_v), tile=b_h3.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 192:
            ct.store(h, index=(i_n, i_t, i_h, 3, i_v), tile=b_h4.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        w1 = ct.load(w, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
        b_v = ct.zeros((BT, BV), dtype=ct.float32)
        b_v = ct.mma(w1, b_h1.astype(w1.dtype), b_v)
        if K > 64:
            w2 = ct.load(w, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w2, b_h2.astype(w2.dtype), b_v)
        if K > 128:
            w3 = ct.load(w, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w3, b_h3.astype(w3.dtype), b_v)
        if K > 192:
            w4 = ct.load(w, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w4, b_h4.astype(w4.dtype), b_v)
        u_tile = ct.load(u, index=(i_n, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=PAD_ZERO).reshape((BT, BV))
        b_v_new = u_tile.astype(ct.float32) - b_v
        ct.store(v_new, index=(i_n, i_t, i_h, i_v), tile=b_v_new.astype(v_new.dtype).reshape((1, BT, 1, BV)))
        b_g = ct.load(g_cumsum, index=(i_n, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_g_last = ct.extract(b_g, index=(BT - 1,), shape=(1,))
        g_diff = b_g_last - b_g
        g_diff = ct.minimum(g_diff, 20.0)
        b_v_new_scaled = (b_v_new * ct.exp(g_diff)[:, None]).astype(k.dtype)
        b_g_last_exp = ct.exp(b_g_last)
        b_h1 = b_h1 * b_g_last_exp
        if K > 64:
            b_h2 = b_h2 * b_g_last_exp
        if K > 128:
            b_h3 = b_h3 * b_g_last_exp
        if K > 192:
            b_h4 = b_h4 * b_g_last_exp
        k1 = ct.load(k, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
        b_h1 = ct.mma(ct.transpose(k1), b_v_new_scaled, b_h1)
        if K > 64:
            k2 = ct.load(k, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h2 = ct.mma(ct.transpose(k2), b_v_new_scaled, b_h2)
        if K > 128:
            k3 = ct.load(k, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h3 = ct.mma(ct.transpose(k3), b_v_new_scaled, b_h3)
        if K > 192:
            k4 = ct.load(k, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h4 = ct.mma(ct.transpose(k4), b_v_new_scaled, b_h4)
    idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
    ct.store(initial_state, index=(idx, i_h, 0, i_v), tile=b_h1.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 64:
        ct.store(initial_state, index=(idx, i_h, 1, i_v), tile=b_h2.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 128:
        ct.store(initial_state, index=(idx, i_h, 2, i_v), tile=b_h3.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 192:
        ct.store(initial_state, index=(idx, i_h, 3, i_v), tile=b_h4.astype(initial_state.dtype).reshape((1, 1, BK, BV)))


# ============================================================================
# Kernel: Fused WY+rec — eliminates w, u HBM roundtrip
# Computes w = A_inv @ (k * beta * exp_g) and u = A_inv @ (v * beta) on-the-fly
# per chunk, so w and u never touch HBM.
# Savings: w(B*T*H*K*2) + u(B*T*H*V*2) write + read = 128MB for typical config
# Grid: (cdiv(V, BV), B*H)
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_fused_wy_rec_kernel(
    k,              # [B, T, H, K]
    v,              # [B, T, H, V]
    beta,           # [B, T, H] (float32)
    g_cumsum,       # [B, T, H] (float32)
    A_inv,          # [B, T, H, BT] — stored as [B, NT*BT, H, BT]
    h,              # [B, NT, H, K, V] output hidden states
    v_new,          # [B, T, H, V] output v_new
    initial_state,  # [N, H, K, V]
    initial_state_indices,  # [B] int32
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BV: ConstInt,
    NT: ConstInt,
    USE_INITIAL_STATE: ConstBool,
):
    """Fused WY+rec: w and u computed on-the-fly per chunk, never written to HBM."""
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // H
    i_h = i_nh % H
    BK = 64

    b_h1 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h2 = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
        b_h1 = b_h1 + ct.load(initial_state, index=(idx, i_h, 0, i_v),
                               shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 64:
            b_h2 = b_h2 + ct.load(initial_state, index=(idx, i_h, 1, i_v),
                                   shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)

    for i_t in range(NT):
        # Store h[t] for output kernel
        ct.store(h, index=(i_n, i_t, i_h, 0, i_v), tile=b_h1.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 64:
            ct.store(h, index=(i_n, i_t, i_h, 1, i_v), tile=b_h2.astype(h.dtype).reshape((1, 1, 1, BK, BV)))

        # Load A_inv for this chunk: [BT, BT]
        b_Ai = ct.load(A_inv, index=(i_n, i_t, i_h, 0),
                       shape=(1, BT, 1, BT), padding_mode=PAD_ZERO).reshape((BT, BT))

        # Load beta and g_cumsum for this chunk
        b_beta = ct.load(beta, index=(i_n, i_t, i_h),
                        shape=(1, BT, 1), padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_g = ct.load(g_cumsum, index=(i_n, i_t, i_h),
                     shape=(1, BT, 1), padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_g_exp = ct.exp(b_g)

        # --- Compute w = A_inv @ (k * beta * exp_g) per K-block, then w @ h ---
        b_v_corr = ct.zeros((BT, BV), dtype=ct.float32)

        k1 = ct.load(k, index=(i_n, i_t, i_h, 0),
                    shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
        kb1 = (k1.astype(ct.float32) * (b_beta * b_g_exp)[:, None]).astype(k.dtype)
        w1 = ct.mma(b_Ai.astype(k.dtype), kb1, ct.zeros((BT, BK), dtype=ct.float32)).astype(k.dtype)
        b_v_corr = ct.mma(w1, b_h1.astype(w1.dtype), b_v_corr)

        if K > 64:
            k2 = ct.load(k, index=(i_n, i_t, i_h, 1),
                        shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            kb2 = (k2.astype(ct.float32) * (b_beta * b_g_exp)[:, None]).astype(k.dtype)
            w2 = ct.mma(b_Ai.astype(k.dtype), kb2, ct.zeros((BT, BK), dtype=ct.float32)).astype(k.dtype)
            b_v_corr = ct.mma(w2, b_h2.astype(w2.dtype), b_v_corr)

        # --- Compute u = A_inv @ (v * beta) for this V-block ---
        v_tile = ct.load(v, index=(i_n, i_t, i_h, i_v),
                        shape=(1, BT, 1, BV), padding_mode=PAD_ZERO).reshape((BT, BV))
        vb = (v_tile.astype(ct.float32) * b_beta[:, None]).astype(v.dtype)
        u_tile = ct.mma(b_Ai.astype(v.dtype), vb, ct.zeros((BT, BV), dtype=ct.float32))

        # v_new = u - w @ h
        b_v_new = u_tile - b_v_corr
        ct.store(v_new, index=(i_n, i_t, i_h, i_v),
                tile=b_v_new.astype(v_new.dtype).reshape((1, BT, 1, BV)))

        # --- Update h: h[t+1] = h[t] * exp(g_last) + k^T @ (v_new * exp(g_last - g)) ---
        b_g_last = ct.extract(b_g, index=(BT - 1,), shape=(1,))
        g_diff = b_g_last - b_g
        g_diff = ct.minimum(g_diff, 20.0)
        b_v_new_scaled = (b_v_new * ct.exp(g_diff)[:, None]).astype(k.dtype)
        b_g_last_exp = ct.exp(b_g_last)
        b_h1 = b_h1 * b_g_last_exp
        if K > 64:
            b_h2 = b_h2 * b_g_last_exp
        # k was already loaded above
        b_h1 = ct.mma(ct.transpose(k1), b_v_new_scaled, b_h1)
        if K > 64:
            b_h2 = ct.mma(ct.transpose(k2), b_v_new_scaled, b_h2)

    # Store final state
    idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
    ct.store(initial_state, index=(idx, i_h, 0, i_v),
             tile=b_h1.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 64:
        ct.store(initial_state, index=(idx, i_h, 1, i_v),
                 tile=b_h2.astype(initial_state.dtype).reshape((1, 1, BK, BV)))


# ============================================================================
# Kernel 3 v4b: BV=16 recurrence with occupancy=3 for small grids (<=256 CTAs)
# Higher occupancy allows better pipelining when SM isn't saturated.
# 4-10% speedup for B=1,T=4096; B=2,T=2048; B=1,T=2048,H=8
# Grid: (cdiv(V, 16), B*H)
# ============================================================================
@ct.kernel(occupancy=3)
def cutile_recurrence_kernel_bv16_occ3(
    k,              # [B, T, H, K]
    w,              # [B, T, H, K]
    u,              # [B, T, H, V]
    g_cumsum,       # [B, T, H] (float32)
    h,              # [B, NT, H, K, V] output hidden states
    v_new,          # [B, T, H, V] output v_new
    initial_state,  # [N, H, K, V]
    initial_state_indices,  # [B] int32
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BV: ConstInt,    # block size for V dimension (16)
    NT: ConstInt,
    USE_INITIAL_STATE: ConstBool,
):
    """BV=16 recurrence with occupancy=3 for small grids."""
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // H
    i_h = i_nh % H
    BK = 64
    b_h1 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h2 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h3 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h4 = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
        b_h1 = b_h1 + ct.load(initial_state, index=(idx, i_h, 0, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 64:
            b_h2 = b_h2 + ct.load(initial_state, index=(idx, i_h, 1, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 128:
            b_h3 = b_h3 + ct.load(initial_state, index=(idx, i_h, 2, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 192:
            b_h4 = b_h4 + ct.load(initial_state, index=(idx, i_h, 3, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
    for i_t in range(NT):
        ct.store(h, index=(i_n, i_t, i_h, 0, i_v), tile=b_h1.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 64:
            ct.store(h, index=(i_n, i_t, i_h, 1, i_v), tile=b_h2.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 128:
            ct.store(h, index=(i_n, i_t, i_h, 2, i_v), tile=b_h3.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 192:
            ct.store(h, index=(i_n, i_t, i_h, 3, i_v), tile=b_h4.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        w1 = ct.load(w, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
        b_v = ct.zeros((BT, BV), dtype=ct.float32)
        b_v = ct.mma(w1, b_h1.astype(w1.dtype), b_v)
        if K > 64:
            w2 = ct.load(w, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w2, b_h2.astype(w2.dtype), b_v)
        if K > 128:
            w3 = ct.load(w, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w3, b_h3.astype(w3.dtype), b_v)
        if K > 192:
            w4 = ct.load(w, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w4, b_h4.astype(w4.dtype), b_v)
        u_tile = ct.load(u, index=(i_n, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=PAD_ZERO).reshape((BT, BV))
        b_v_new = u_tile.astype(ct.float32) - b_v
        ct.store(v_new, index=(i_n, i_t, i_h, i_v), tile=b_v_new.astype(v_new.dtype).reshape((1, BT, 1, BV)))
        b_g = ct.load(g_cumsum, index=(i_n, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_g_last = ct.extract(b_g, index=(BT - 1,), shape=(1,))
        g_diff = b_g_last - b_g
        g_diff = ct.minimum(g_diff, 20.0)
        b_v_new_scaled = (b_v_new * ct.exp(g_diff)[:, None]).astype(k.dtype)
        b_g_last_exp = ct.exp(b_g_last)
        b_h1 = b_h1 * b_g_last_exp
        if K > 64:
            b_h2 = b_h2 * b_g_last_exp
        if K > 128:
            b_h3 = b_h3 * b_g_last_exp
        if K > 192:
            b_h4 = b_h4 * b_g_last_exp
        k1 = ct.load(k, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
        b_h1 = ct.mma(ct.transpose(k1), b_v_new_scaled, b_h1)
        if K > 64:
            k2 = ct.load(k, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h2 = ct.mma(ct.transpose(k2), b_v_new_scaled, b_h2)
        if K > 128:
            k3 = ct.load(k, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h3 = ct.mma(ct.transpose(k3), b_v_new_scaled, b_h3)
        if K > 192:
            k4 = ct.load(k, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h4 = ct.mma(ct.transpose(k4), b_v_new_scaled, b_h4)
    idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
    ct.store(initial_state, index=(idx, i_h, 0, i_v), tile=b_h1.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 64:
        ct.store(initial_state, index=(idx, i_h, 1, i_v), tile=b_h2.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 128:
        ct.store(initial_state, index=(idx, i_h, 2, i_v), tile=b_h3.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 192:
        ct.store(initial_state, index=(idx, i_h, 3, i_v), tile=b_h4.astype(initial_state.dtype).reshape((1, 1, BK, BV)))


# ============================================================================
# Kernel: Fused cumsum + KKT + 64×64 squaring-trick solve (v2)
#
# Replaces the 16×16 hierarchical block inversion with:
#   (I+A)⁻¹ = (I-A)(I+A²)(I+A⁴)(I+A⁸)(I+A¹⁶)(I+A³²)
# for the 64×64 strictly lower-triangular nilpotent matrix A (A⁶⁴=0).
#
# Uses 10 sequential 64×64 MMAs vs ~36 serial 16×16 MMAs in v1.
# Stores A_inv as flat [1, BT, 1, BT] compatible with fused_wy_rec.
# Grid: (NT, B*H)
# ============================================================================
@ct.kernel(occupancy=1)
def cutile_fused_ckkt_solve_v2_kernel(
    g_in,       # [B, T, H] input gates
    k,          # [B, T, H, K] keys
    beta,       # [B, T, H] beta values
    g_out,      # [B, T, H] output cumsum (float32)
    A_inv_out,  # [B, T, H, BT] output A_inv (bf16), stored as flat [BT,BT]
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
):
    """Fused cumsum + KKT + full 64×64 squaring-trick solve.
    (I+A)⁻¹ = (I-A)(I+A²)(I+A⁴)(I+A⁸)(I+A¹⁶)(I+A³²) — exact for A^64=0.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # ---- cumsum ----
    b_g_raw = ct.load(g_in, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                      padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_g = ct.cumsum(b_g_raw, axis=0)
    ct.store(g_out, index=(i_b, i_t, i_h),
             tile=b_g.astype(g_out.dtype).reshape((1, BT, 1)))

    # ---- KKT ----
    b_beta = ct.load(beta, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                     padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(k_tile, ct.transpose(k_tile), b_A)
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff) * b_beta[:, None]
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    b_A = ct.where(row_idx > col_idx, b_A, 0.0)

    # ---- squaring trick: (I+A)⁻¹ for 64×64 nilpotent A ----
    bt_idx = ct.arange(BT, dtype=ct.int32)
    I_BT = ct.where(bt_idx[:, None] == bt_idx[None, :], 1.0, 0.0)
    X = (-b_A).astype(ct.bfloat16)
    I_bf16 = I_BT.astype(ct.bfloat16)

    X2 = ct.mma(X, X, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t12 = ct.mma(I_bf16 + X, I_bf16 + X2, ct.zeros((BT, BT), dtype=ct.float32))

    X4 = ct.mma(X2, X2, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    X8 = ct.mma(X4, X4, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t48 = ct.mma(I_bf16 + X4, I_bf16 + X8, ct.zeros((BT, BT), dtype=ct.float32))

    t1248 = ct.mma(t12.astype(ct.bfloat16), t48.astype(ct.bfloat16),
                   ct.zeros((BT, BT), dtype=ct.float32))

    X16 = ct.mma(X8, X8, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    X32 = ct.mma(X16, X16, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t1632 = ct.mma(I_bf16 + X16, I_bf16 + X32, ct.zeros((BT, BT), dtype=ct.float32))

    b_Ai = ct.mma(t1248.astype(ct.bfloat16), t1632.astype(ct.bfloat16),
                  ct.zeros((BT, BT), dtype=ct.float32))

    ct.store(A_inv_out, index=(i_b, i_t, i_h, 0),
             tile=b_Ai.astype(A_inv_out.dtype).reshape((1, BT, 1, BT)))


# ============================================================================
# Fused WY + Recurrence kernel (Iteration 9)
# Eliminates w/u intermediate tensors by computing on-the-fly:
#   v_new = A_inv @ (v*beta - (k*beta*exp(g)) @ h)
# Grid: (cdiv(V, BV), B*H)
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_fused_wy_rec_kernel(
    k, v, beta, g_cumsum, A_inv,
    h, v_new, initial_state, initial_state_indices,
    T: ConstInt, H: ConstInt, K: ConstInt, V: ConstInt,
    BT: ConstInt, BV: ConstInt, NT: ConstInt,
    USE_INITIAL_STATE: ConstBool,
):
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // H
    i_h = i_nh % H
    BK = 64
    b_h1 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h2 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h3 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h4 = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
        b_h1 = b_h1 + ct.load(initial_state, index=(idx, i_h, 0, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 64:
            b_h2 = b_h2 + ct.load(initial_state, index=(idx, i_h, 1, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 128:
            b_h3 = b_h3 + ct.load(initial_state, index=(idx, i_h, 2, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 192:
            b_h4 = b_h4 + ct.load(initial_state, index=(idx, i_h, 3, i_v), shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
    for i_t in range(NT):
        ct.store(h, index=(i_n, i_t, i_h, 0, i_v), tile=b_h1.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 64:
            ct.store(h, index=(i_n, i_t, i_h, 1, i_v), tile=b_h2.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 128:
            ct.store(h, index=(i_n, i_t, i_h, 2, i_v), tile=b_h3.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 192:
            ct.store(h, index=(i_n, i_t, i_h, 3, i_v), tile=b_h4.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        b_beta = ct.load(beta, index=(i_n, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_g = ct.load(g_cumsum, index=(i_n, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_exp_g = ct.exp(b_g)
        v_tile = ct.load(v, index=(i_n, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=PAD_ZERO).reshape((BT, BV))
        b_temp_f = v_tile.astype(ct.float32) * b_beta[:, None]
        k1 = ct.load(k, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
        k1_scaled = (k1.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k1.dtype)
        b_temp_f = b_temp_f - ct.mma(k1_scaled, b_h1.astype(k1_scaled.dtype), ct.zeros((BT, BV), dtype=ct.float32))
        if K > 64:
            k2 = ct.load(k, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k2_scaled = (k2.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k2.dtype)
            b_temp_f = b_temp_f - ct.mma(k2_scaled, b_h2.astype(k2_scaled.dtype), ct.zeros((BT, BV), dtype=ct.float32))
        if K > 128:
            k3 = ct.load(k, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k3_scaled = (k3.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k3.dtype)
            b_temp_f = b_temp_f - ct.mma(k3_scaled, b_h3.astype(k3_scaled.dtype), ct.zeros((BT, BV), dtype=ct.float32))
        if K > 192:
            k4 = ct.load(k, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k4_scaled = (k4.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k4.dtype)
            b_temp_f = b_temp_f - ct.mma(k4_scaled, b_h4.astype(k4_scaled.dtype), ct.zeros((BT, BV), dtype=ct.float32))
        b_Ai = ct.load(A_inv, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BT), padding_mode=PAD_ZERO).reshape((BT, BT))
        b_v_new = ct.mma(b_Ai, b_temp_f.astype(b_Ai.dtype), ct.zeros((BT, BV), dtype=ct.float32))
        ct.store(v_new, index=(i_n, i_t, i_h, i_v), tile=b_v_new.astype(v_new.dtype).reshape((1, BT, 1, BV)))
        b_g_last = ct.extract(b_g, index=(BT - 1,), shape=(1,))
        g_diff = b_g_last - b_g
        g_diff = ct.minimum(g_diff, 20.0)
        b_v_new_scaled = (b_v_new * ct.exp(g_diff)[:, None]).astype(k.dtype)
        b_g_last_exp = ct.exp(b_g_last)
        b_h1 = b_h1 * b_g_last_exp
        if K > 64:
            b_h2 = b_h2 * b_g_last_exp
        if K > 128:
            b_h3 = b_h3 * b_g_last_exp
        if K > 192:
            b_h4 = b_h4 * b_g_last_exp
        b_h1 = ct.mma(ct.transpose(k1), b_v_new_scaled, b_h1)
        if K > 64:
            b_h2 = ct.mma(ct.transpose(k2), b_v_new_scaled, b_h2)
        if K > 128:
            b_h3 = ct.mma(ct.transpose(k3), b_v_new_scaled, b_h3)
        if K > 192:
            b_h4 = ct.mma(ct.transpose(k4), b_v_new_scaled, b_h4)
    idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
    ct.store(initial_state, index=(idx, i_h, 0, i_v), tile=b_h1.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 64:
        ct.store(initial_state, index=(idx, i_h, 1, i_v), tile=b_h2.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 128:
        ct.store(initial_state, index=(idx, i_h, 2, i_v), tile=b_h3.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 192:
        ct.store(initial_state, index=(idx, i_h, 3, i_v), tile=b_h4.astype(initial_state.dtype).reshape((1, 1, BK, BV)))


# ============================================================================
# Kernel: Fused cumsum + KKT + solve v2 — occupancy=2 variant
# Same as cutile_fused_ckkt_solve_v2_kernel but with occupancy=2 hint.
# With 1024 CTAs on 148 SMs: 3.5 waves (vs 7 waves at occ=1), ~2x faster.
# Register analysis: peak ~96KB (< 128KB budget for occ=2 on 256KB reg file).
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_fused_ckkt_solve_v2_occ2_kernel(
    g_in,       # [B, T, H] input gates
    k,          # [B, T, H, K] keys
    beta,       # [B, T, H] beta values
    g_out,      # [B, T, H] output cumsum (float32)
    A_inv_out,  # [B, T, H, BT] output A_inv (bf16), stored as flat [BT,BT]
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
):
    """Fused cumsum + KKT + full 64×64 squaring-trick solve (occupancy=2).
    (I+A)⁻¹ = (I-A)(I+A²)(I+A⁴)(I+A⁸)(I+A¹⁶)(I+A³²) — exact for A^64=0.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # ---- cumsum ----
    b_g_raw = ct.load(g_in, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                      padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_g = ct.cumsum(b_g_raw, axis=0)
    ct.store(g_out, index=(i_b, i_t, i_h),
             tile=b_g.astype(g_out.dtype).reshape((1, BT, 1)))

    # ---- KKT ----
    b_beta = ct.load(beta, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                     padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(k_tile, ct.transpose(k_tile), b_A)
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff) * b_beta[:, None]
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    b_A = ct.where(row_idx > col_idx, b_A, 0.0)

    # ---- squaring trick: (I+A)⁻¹ for 64×64 nilpotent A ----
    bt_idx = ct.arange(BT, dtype=ct.int32)
    I_BT = ct.where(bt_idx[:, None] == bt_idx[None, :], 1.0, 0.0)
    X = (-b_A).astype(ct.bfloat16)
    I_bf16 = I_BT.astype(ct.bfloat16)

    X2 = ct.mma(X, X, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t12 = ct.mma(I_bf16 + X, I_bf16 + X2, ct.zeros((BT, BT), dtype=ct.float32))

    X4 = ct.mma(X2, X2, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    X8 = ct.mma(X4, X4, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t48 = ct.mma(I_bf16 + X4, I_bf16 + X8, ct.zeros((BT, BT), dtype=ct.float32))

    t1248 = ct.mma(t12.astype(ct.bfloat16), t48.astype(ct.bfloat16),
                   ct.zeros((BT, BT), dtype=ct.float32))

    X16 = ct.mma(X8, X8, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    X32 = ct.mma(X16, X16, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t1632 = ct.mma(I_bf16 + X16, I_bf16 + X32, ct.zeros((BT, BT), dtype=ct.float32))

    b_Ai = ct.mma(t1248.astype(ct.bfloat16), t1632.astype(ct.bfloat16),
                  ct.zeros((BT, BT), dtype=ct.float32))

    ct.store(A_inv_out, index=(i_b, i_t, i_h, 0),
             tile=b_Ai.astype(A_inv_out.dtype).reshape((1, BT, 1, BT)))


# ============================================================================
# Kernel: Fused cumsum + KKT + solve v2 — occupancy=3 variant
# Same body as occ=2 but with occupancy=3. Budget: 85KB. May cause minor spill
# if peak >85KB, but reduces waves: 1024 CTAs → 2.31 waves (vs 3.46 at occ=2).
# ============================================================================
@ct.kernel(occupancy=3)
def cutile_fused_ckkt_solve_v2_occ3_kernel(
    g_in,       # [B, T, H] input gates
    k,          # [B, T, H, K] keys
    beta,       # [B, T, H] beta values
    g_out,      # [B, T, H] output cumsum (float32)
    A_inv_out,  # [B, T, H, BT] output A_inv (bf16), stored as flat [BT,BT]
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
):
    """Fused cumsum + KKT + full 64×64 squaring-trick solve (occupancy=3).
    (I+A)⁻¹ = (I-A)(I+A²)(I+A⁴)(I+A⁸)(I+A¹⁶)(I+A³²) — exact for A^64=0.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # ---- cumsum ----
    b_g_raw = ct.load(g_in, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                      padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_g = ct.cumsum(b_g_raw, axis=0)
    ct.store(g_out, index=(i_b, i_t, i_h),
             tile=b_g.astype(g_out.dtype).reshape((1, BT, 1)))

    # ---- KKT ----
    b_beta = ct.load(beta, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                     padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(k_tile, ct.transpose(k_tile), b_A)
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff) * b_beta[:, None]
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    b_A = ct.where(row_idx > col_idx, b_A, 0.0)

    # ---- squaring trick: (I+A)⁻¹ for 64×64 nilpotent A ----
    bt_idx = ct.arange(BT, dtype=ct.int32)
    I_BT = ct.where(bt_idx[:, None] == bt_idx[None, :], 1.0, 0.0)
    X = (-b_A).astype(ct.bfloat16)
    I_bf16 = I_BT.astype(ct.bfloat16)

    X2 = ct.mma(X, X, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t12 = ct.mma(I_bf16 + X, I_bf16 + X2, ct.zeros((BT, BT), dtype=ct.float32))

    X4 = ct.mma(X2, X2, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    X8 = ct.mma(X4, X4, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t48 = ct.mma(I_bf16 + X4, I_bf16 + X8, ct.zeros((BT, BT), dtype=ct.float32))

    t1248 = ct.mma(t12.astype(ct.bfloat16), t48.astype(ct.bfloat16),
                   ct.zeros((BT, BT), dtype=ct.float32))

    X16 = ct.mma(X8, X8, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    X32 = ct.mma(X16, X16, ct.zeros((BT, BT), dtype=ct.float32)).astype(ct.bfloat16)
    t1632 = ct.mma(I_bf16 + X16, I_bf16 + X32, ct.zeros((BT, BT), dtype=ct.float32))

    b_Ai = ct.mma(t1248.astype(ct.bfloat16), t1632.astype(ct.bfloat16),
                  ct.zeros((BT, BT), dtype=ct.float32))

    ct.store(A_inv_out, index=(i_b, i_t, i_h, 0),
             tile=b_Ai.astype(A_inv_out.dtype).reshape((1, BT, 1, BT)))

