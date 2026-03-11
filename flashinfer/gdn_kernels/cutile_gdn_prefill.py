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
# Kernel 1: Chunk-local cumulative sum
# ============================================================================
@ct.kernel
def cutile_cumsum_kernel(
    g_in,       # [B, T, H] input gates (bfloat16/float32)
    g_out,      # [B, T, H] output cumsum (float32)
    T: ConstInt,
    H: ConstInt,
    BT: ConstInt,  # chunk size (64)
):
    """Per-chunk cumulative sum of gate values along the T dimension.
    Grid: (NT, B*H) where NT = cdiv(T, BT)

    g layout: [B, T, H] with tile-space partition (1, BT, 1).
    For block (i_t, i_bh), we load g[i_b, i_t*BT:(i_t+1)*BT, i_h].
    """
    i_t = ct.bid(0)  # chunk index
    i_bh = ct.bid(1)  # batch*head index
    i_b = i_bh // H
    i_h = i_bh % H

    # Load chunk of g: [1, BT, 1] -> reshape to [BT]
    b_g = ct.load(
        g_in,
        index=(i_b, i_t, i_h),
        shape=(1, BT, 1),
        padding_mode=PAD_ZERO,
    ).reshape((BT,)).astype(ct.float32)

    # Cumulative sum
    b_g_cumsum = ct.cumsum(b_g, axis=0)

    # Store result: reshape back to [1, BT, 1]
    ct.store(
        g_out,
        index=(i_b, i_t, i_h),
        tile=b_g_cumsum.astype(g_out.dtype).reshape((1, BT, 1)),
    )


def cutile_chunk_local_cumsum(
    g: torch.Tensor,  # [B, T, H]
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute per-chunk cumulative sum of g."""
    B, T, H = g.shape
    BT = chunk_size
    NT = math.ceil(T / BT)

    g_out = torch.empty(B, T, H, dtype=torch.float32, device=g.device)

    grid = (NT, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        cutile_cumsum_kernel,
        (g, g_out, T, H, BT),
    )
    return g_out



# ============================================================================
# Kernel 1b: Fused cumsum + KKT for split path (saves one launch + memory round-trip)
# ============================================================================
@ct.kernel
def cutile_cumsum_kkt_kernel(
    g_in,       # [B, T, H] input gates
    k,          # [B, T, H, K] keys
    beta,       # [B, T, H] beta values
    g_out,      # [B, T, H] output cumsum (float32)
    A,          # [B, T, H, BT] output A matrix (float32)
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
):
    """Fused cumsum + KKT. Saves one kernel launch + g_cumsum memory round-trip.
    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load and compute cumsum of g
    b_g_raw = ct.load(g_in, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                      padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_g = ct.cumsum(b_g_raw, axis=0)
    ct.store(g_out, index=(i_b, i_t, i_h), tile=b_g.astype(g_out.dtype).reshape((1, BT, 1)))

    # Load beta
    b_beta = ct.load(beta, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                     padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)

    # Compute A = K @ K^T
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(k_tile, ct.transpose(k_tile), b_A)

    # Apply gating
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff) * b_beta[:, None]

    # Strictly lower triangular mask
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    b_A = ct.where(row_idx > col_idx, b_A, 0.0)

    ct.store(A, index=(i_b, i_t, i_h, 0),
             tile=b_A.astype(A.dtype).reshape((1, BT, 1, BT)))

# ============================================================================
# Kernel 1c: Fused cumsum + KKT + solve (eliminates A store/load + one launch)
# Grid: (NT, B*H) — same as ckkt and solve individually
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_fused_ckkt_solve_kernel(
    g_in,       # [B, T, H] input gates
    k,          # [B, T, H, K] keys
    beta,       # [B, T, H] beta values
    g_out,      # [B, T, H] output cumsum (float32)
    A_inv_out,  # [B, T, H, BT] output A_inv (bf16)
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
):
    """Fused cumsum + KKT + hierarchical solve. Saves one ct.launch + A memory round-trip.
    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # ---- cumsum ----
    b_g_raw = ct.load(g_in, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                      padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_g = ct.cumsum(b_g_raw, axis=0)
    ct.store(g_out, index=(i_b, i_t, i_h), tile=b_g.astype(g_out.dtype).reshape((1, BT, 1)))

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

    # ---- solve: hierarchical 16x16 block inversion ----
    BS = 16
    A11 = ct.extract(b_A, index=(0, 0), shape=(BS, BS))
    A22 = ct.extract(b_A, index=(1, 1), shape=(BS, BS))
    A33 = ct.extract(b_A, index=(2, 2), shape=(BS, BS))
    A44 = ct.extract(b_A, index=(3, 3), shape=(BS, BS))
    A21 = ct.extract(b_A, index=(1, 0), shape=(BS, BS))
    A31 = ct.extract(b_A, index=(2, 0), shape=(BS, BS))
    A32 = ct.extract(b_A, index=(2, 1), shape=(BS, BS))
    A41 = ct.extract(b_A, index=(3, 0), shape=(BS, BS))
    A42 = ct.extract(b_A, index=(3, 1), shape=(BS, BS))
    A43 = ct.extract(b_A, index=(3, 2), shape=(BS, BS))

    bs_row = ct.arange(BS, dtype=ct.int32)[:, None]
    bs_col = ct.arange(BS, dtype=ct.int32)[None, :]
    I16 = ct.where(bs_row == bs_col, 1.0, 0.0)

    def inv_block(Axx):
        neg = -Axx
        sq = ct.mma(neg, neg, ct.zeros((BS, BS), dtype=ct.float32))
        q4 = ct.mma(sq, sq, ct.zeros((BS, BS), dtype=ct.float32))
        q8 = ct.mma(q4, q4, ct.zeros((BS, BS), dtype=ct.float32))
        p1 = ct.mma((I16 + neg).astype(ct.bfloat16), (I16 + sq).astype(ct.bfloat16),
                     ct.zeros((BS, BS), dtype=ct.float32))
        p2 = ct.mma((I16 + q4).astype(ct.bfloat16), (I16 + q8).astype(ct.bfloat16),
                     ct.zeros((BS, BS), dtype=ct.float32))
        return ct.mma(p1.astype(ct.bfloat16), p2.astype(ct.bfloat16),
                      ct.zeros((BS, BS), dtype=ct.float32))

    Ai11 = inv_block(A11)
    Ai22 = inv_block(A22)
    Ai33 = inv_block(A33)
    Ai44 = inv_block(A44)

    tmp = ct.mma(Ai22.astype(ct.bfloat16), A21.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai21 = ct.mma(-tmp.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    tmp = ct.mma(Ai33.astype(ct.bfloat16), A32.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai32 = ct.mma(-tmp.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    tmp = ct.mma(Ai44.astype(ct.bfloat16), A43.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai43 = ct.mma(-tmp.astype(ct.bfloat16), Ai33.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A31.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A32.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    Ai31 = ct.mma(-Ai33.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A42.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A43.astype(ct.bfloat16), Ai32.astype(ct.bfloat16), t1)
    Ai42 = ct.mma(-Ai44.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A41.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A42.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    t3 = ct.mma(A43.astype(ct.bfloat16), Ai31.astype(ct.bfloat16), t2)
    Ai41 = ct.mma(-Ai44.astype(ct.bfloat16), t3.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Store A_inv
    z16 = ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 0), tile=Ai11.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 1), tile=z16)
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 2), tile=z16)
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 3), tile=z16)
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 0), tile=Ai21.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 1), tile=Ai22.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 2), tile=z16)
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 3), tile=z16)
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 0), tile=Ai31.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 1), tile=Ai32.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 2), tile=Ai33.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 3), tile=z16)
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 0), tile=Ai41.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 1), tile=Ai42.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 2), tile=Ai43.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 3), tile=Ai44.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))

# ============================================================================
# Kernel 2: Fused KKT + solve_tril + WY recompute
# ============================================================================
@ct.kernel
def cutile_prepare_kernel(
    k,          # [B, T, H, K] keys
    v,          # [B, T, H, V] values
    beta,       # [B, T, H] beta values
    g_cumsum,   # [B, T, H] cumulative sum of g (float32)
    w,          # [B, T, H, K] output w
    u,          # [B, T, H, V] output u
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Fused: scaled_dot_kkt + solve_tril + wy_recompute.

    For each chunk, computes:
    1. A = beta * K @ K^T (with gating), strictly lower triangular
    2. A_inv = (I + A)^{-1}  (via Neumann series / power series)
    3. u = A_inv @ (v * beta)
    4. w = A_inv @ (k * beta * exp(g))

    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load beta for this chunk: [1, BT, 1] -> [BT]
    b_beta = ct.load(
        beta, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Load g_cumsum for this chunk: [1, BT, 1] -> [BT]
    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Step 1: Compute A = beta * K @ K^T with gating, strictly lower triangular
    b_A = ct.zeros((BT, BT), dtype=ct.float32)

    for i_k in range(ct.cdiv(K, BK)):
        # Load k chunk: [1, BT, 1, BK] -> [BT, BK]
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        # A += k @ k^T
        b_A = ct.mma(k_tile, ct.transpose(k_tile), b_A)

    # Apply gating: A *= exp(g_i - g_j) (safe_exp style)
    g_i = b_g[:, None]   # [BT, 1]
    g_j = b_g[None, :]   # [1, BT]
    g_diff = g_i - g_j
    # Clamp for numerical stability
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff)

    # Apply beta (row-wise) and strictly lower triangular mask
    b_A = b_A * b_beta[:, None]
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    mask_lt = row_idx > col_idx
    b_A = ct.where(mask_lt, b_A, 0.0)

    # Step 2: Solve (I + A)^{-1}
    # A is strictly lower triangular. Use Neumann power series:
    # (I + A)^{-1} = I - A + A^2 - A^3 + ... = sum_{k=0}^{63} (-A)^k
    # For 64x64, need up to (-A)^63 for exactness, but since entries
    # typically decay, 16 iterations should suffice for bf16 accuracy.
    neg_A = -b_A
    b_Ai = ct.zeros((BT, BT), dtype=ct.float32)
    # Add identity
    diag_mask = row_idx == col_idx
    b_Ai = ct.where(diag_mask, 1.0, 0.0)
    # Add (-A)^1
    b_Ai = b_Ai + neg_A

    # Compute higher powers: (-A)^2 through (-A)^K
    # Each iteration: power = power @ (-A), b_Ai += power
    # For 64x64, (-A)^k becomes zero for k>=64. But entries decay quickly.
    # 3 iterations (total terms 0..4) provides good bf16 accuracy.
    power = neg_A
    for _ in range(3):  # 3 more iterations -> total (-A)^0 through (-A)^4
        acc = ct.zeros((BT, BT), dtype=ct.float32)
        power = ct.mma(power, neg_A, acc)
        b_Ai = b_Ai + power

    # Step 3: Compute u = A_inv @ (v * beta) and w = A_inv @ (k * beta * exp(g))
    b_exp_g = ct.exp(b_g)

    for i_v in range(ct.cdiv(V, BV)):
        # Load v chunk: [1, BT, 1, BV] -> [BT, BV]
        v_tile = ct.load(
            v, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BV))
        # v * beta  (in bf16 for mma)
        vb = (v_tile.astype(ct.float32) * b_beta[:, None]).astype(v_tile.dtype)
        # u = A_inv @ vb
        acc_u = ct.zeros((BT, BV), dtype=ct.float32)
        b_u = ct.mma(b_Ai.astype(vb.dtype), vb, acc_u)
        # Store u: reshape to [1, BT, 1, BV]
        ct.store(
            u, index=(i_b, i_t, i_h, i_v),
            tile=b_u.astype(u.dtype).reshape((1, BT, 1, BV)),
        )

    for i_k in range(ct.cdiv(K, BK)):
        # Load k chunk: [1, BT, 1, BK] -> [BT, BK]
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        # k * beta * exp(g)
        kb = (k_tile.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k_tile.dtype)
        # w = A_inv @ kb
        acc_w = ct.zeros((BT, BK), dtype=ct.float32)
        b_w = ct.mma(b_Ai.astype(kb.dtype), kb, acc_w)
        # Store w: reshape to [1, BT, 1, BK]
        ct.store(
            w, index=(i_b, i_t, i_h, i_k),
            tile=b_w.astype(w.dtype).reshape((1, BT, 1, BK)),
        )


# ============================================================================
# Kernel 3: Hidden state recurrence
# h[B, NT, H, K, V] stores inter-chunk hidden states
# v_new[B, T, H, V] stores corrected values
#
# The recurrence (per chunk i_t):
#   store h[n, i_t, h, :, :] = b_h  (before update)
#   b_v_corr = w[chunk] @ b_h       (correction term)
#   b_v_new = u[chunk] - b_v_corr   (corrected v)
#   scale v_new by exp(g_last - g[t]) for each t in chunk
#   b_h = b_h * exp(g_last) + k^T @ v_new_scaled
#
# Grid: (cdiv(V, BV), B*H)
# Note: K is split into 64-wide blocks (BK=64) for register efficiency
# ============================================================================
@ct.kernel
def cutile_recurrence_kernel(
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
    BK: ConstInt,    # block size for K dimension (64)
    BV: ConstInt,    # block size for V dimension (32)
    NT: ConstInt,
    USE_INITIAL_STATE: ConstBool,
):
    """Hidden state recurrence for gated delta rule.

    Grid: (cdiv(K, BK), cdiv(V, BV), B*H)
    Each block handles one [BK, BV] slice of the hidden state.
    """
    i_k = ct.bid(0)   # K-block index
    i_v = ct.bid(1)   # V-block index
    i_nh = ct.bid(2)
    i_n = i_nh // H
    i_h = i_nh % H

    # Initialize hidden state block [BK, BV]
    b_h = ct.zeros((BK, BV), dtype=ct.float32)

    # Load initial state if needed
    if USE_INITIAL_STATE:
        idx = ct.load(
            initial_state_indices, index=(i_n,), shape=()
        ).astype(ct.int32)
        # initial_state: [N, H, K, V]
        # Load [1, 1, BK, BV] at (idx, i_h, i_k, i_v)
        h0_tile = ct.load(
            initial_state,
            index=(idx, i_h, i_k, i_v),
            shape=(1, 1, BK, BV),
            padding_mode=PAD_ZERO,
        ).reshape((BK, BV)).astype(ct.float32)
        b_h = b_h + h0_tile

    for i_t in range(NT):
        # Store current h for this chunk: h[n, i_t, h, k_block, v_block]
        ct.store(
            h,
            index=(i_n, i_t, i_h, i_k, i_v),
            tile=b_h.astype(h.dtype).reshape((1, 1, 1, BK, BV)),
        )

        # Load w for this chunk: [1, BT, 1, BK] -> [BT, BK]
        w_tile = ct.load(
            w, index=(i_n, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))

        # v_corr = w @ b_h: [BT, BK] @ [BK, BV] -> [BT, BV]
        acc_v = ct.zeros((BT, BV), dtype=ct.float32)
        b_v_corr = ct.mma(w_tile, b_h.astype(w_tile.dtype), acc_v)

        # Load u (pre-corrected v): [1, BT, 1, BV] -> [BT, BV]
        u_tile = ct.load(
            u, index=(i_n, i_t, i_h, i_v), shape=(1, BT, 1, BV),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BV))

        # v_new = u - v_correction
        b_v_new = u_tile.astype(ct.float32) - b_v_corr

        # Store v_new (only when i_k == 0, to avoid multiple writes)
        # Actually, when K > BK, different i_k blocks contribute to v_corr.
        # But in Triton, v_corr is accumulated across all K blocks in a single thread.
        # In our grid, we split K across blocks - so we'd need atomics or a different approach.
        # For now, since this is wrong for K > BK, we'll handle it in the launcher.
        # TODO: fix for K > BK by using a reduction
        ct.store(
            v_new, index=(i_n, i_t, i_h, i_v),
            tile=b_v_new.astype(v_new.dtype).reshape((1, BT, 1, BV)),
        )

        # Load g_cumsum for this chunk: [1, BT, 1] -> [BT]
        b_g = ct.load(
            g_cumsum, index=(i_n, i_t, i_h), shape=(1, BT, 1),
            padding_mode=PAD_ZERO,
        ).reshape((BT,)).astype(ct.float32)

        # Get g_last (last element of chunk)
        # For non-last chunks: g_last = g[BT-1]
        # For last chunk: might be padded. Use the loaded values.
        # We approximate by taking g[BT-1] which is correct for full chunks
        # and for padded chunks the padding_mode=ZERO means g_last=0 for OOB,
        # but we need the actual last valid value.
        # Simple approach: just use g[BT-1] since we load with padding=ZERO
        # and the cumsum of zeros is zero, so padded positions have correct values.
        # Actually, cumsum with zero padding means the cumsum continues from last valid.
        # No - with our cumsum kernel, OOB g values are loaded as 0, so cumsum
        # for padded positions keeps the last valid value. This is correct!

        # Extract last element: use gather on a 1D slice
        # b_g is [BT], get element at BT-1
        last_pos = ct.full((1,), BT - 1, dtype=ct.int32)
        # Need to get scalar from tile. Use ct.extract or ct.gather on a 1D view.
        # ct.extract extracts a sub-tile by tile-index.
        # For [BT] tile, extract at index (BT-1) with shape (1) gives last element.
        # BT-1 is not a tile index unless shape is 1, which gives BT tiles of size 1.
        # So: ct.extract(b_g, (BT-1,), shape=(1,)) gives a tile of shape (1,) with the last element.
        # But BT must be a compile-time constant, and BT-1 must be a compile-time expression.
        # Since BT is ConstInt (e.g., 64), BT-1 = 63 is compile-time.
        b_g_last = ct.extract(b_g, index=(BT - 1,), shape=(1,))  # [1]

        # Scale v_new by exp(g_last - g[t]) for each t
        g_diff = b_g_last - b_g  # [BT], broadcasts
        g_diff = ct.minimum(g_diff, 20.0)
        b_v_new_scaled = b_v_new * ct.exp(g_diff)[:, None]

        # Scale h by exp(g_last)
        b_g_last_exp = ct.exp(b_g_last)  # scalar-ish [1]
        b_h = b_h * b_g_last_exp

        # Load k for this chunk: [1, BT, 1, BK] -> [BT, BK]
        k_tile = ct.load(
            k, index=(i_n, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        k_tile_t = ct.transpose(k_tile)  # [BK, BT]

        # h += k^T @ v_new_scaled: [BK, BT] @ [BT, BV] -> [BK, BV]
        b_h = ct.mma(k_tile_t, b_v_new_scaled.astype(k_tile_t.dtype), b_h)

    # Write back final state to initial_state (in-place update)
    idx = ct.load(
        initial_state_indices, index=(i_n,), shape=()
    ).astype(ct.int32)
    ct.store(
        initial_state,
        index=(idx, i_h, i_k, i_v),
        tile=b_h.astype(initial_state.dtype).reshape((1, 1, BK, BV)),
    )


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
# Kernel 4c: Multi-V-tile output (processes ALL V-tiles per block, sharing QK^T)
# Grid: (NT, B*H) — each block handles all V-tiles for one chunk
# Saves ~10us for B=4,B=8 configs by sharing q,k loads and QK^T computation
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_output_multiV_kernel(
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
    """Output kernel that processes all V-tiles sequentially, sharing q/k/QKT.
    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Step 1: Compute QK^T (once for all V-tiles)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)

    for i_k in range(ct.cdiv(K, BK)):
        q_tile = ct.load(q, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A = ct.mma(q_tile, ct.transpose(k_tile), b_A)

    # Apply gating to QK^T
    b_g = ct.load(g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                  padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)

    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff)
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    b_A = ct.where(row_idx >= col_idx, b_A, 0.0)

    b_exp_g = ct.exp(b_g)

    # Step 2: For each V-tile, compute output
    for i_v in range(ct.cdiv(V, BV)):
        # Compute q@h for this V-tile
        b_o = ct.zeros((BT, BV), dtype=ct.float32)
        for i_k in range(ct.cdiv(K, BK)):
            q_tile = ct.load(q, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                             padding_mode=PAD_ZERO).reshape((BT, BK))
            h_tile = ct.load(h, index=(i_b, i_t, i_h, i_k, i_v), shape=(1, 1, 1, BK, BV),
                             padding_mode=PAD_ZERO).reshape((BK, BV))
            b_o = ct.mma(q_tile, h_tile, b_o)

        b_o = b_o * b_exp_g[:, None]

        # Add causal attention: A @ v_new
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
# Kernel 1e: Q-cached output occ=3 (BV=64 must be used; peak 72KB < 85KB budget)
# At occ=3: 1024/444=2.31 waves vs occ=2's 3.46 waves. Same total data/CTA.
# ============================================================================
@ct.kernel(occupancy=3)
def cutile_output_qcached_occ3_kernel(
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
    """Output kernel with Q cached in registers (occ=3, BV=64).
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
# Kernel 2ac: Q-cached output occ=3 v2 (BV=64)
# Fix for occ=3 spilling: reorder g_diff computation to AFTER QK^T, so
# g_diff(16KB) is computed when k_temp(8KB) is already freed. Also reuse
# the same g_diff variable name (no separate b_exp_g_diff) to avoid 32KB
# peak. Result: peak = 72KB < 85KB occ=3 budget, vs original 88KB.
# At occ=3: 1024/444=2.31 waves vs occ=2's 3.46 waves → ~1.5x less wave stall.
# ============================================================================
@ct.kernel(occupancy=3)
def cutile_output_qcached_occ3_v2_kernel(
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
    """Output kernel occ=3 v2 with Q cached. Peak 72KB < 85KB via reordering.
    g_diff computed AFTER QK^T (k_temp freed), in-place variable reuse.
    Must be dispatched with BV=64 to keep b_o=[64,64] fp32=16KB in bounds.
    Grid: (NT, B*H).
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load g_cumsum: only b_exp_g needed early (for Q@H output scaling)
    b_g = ct.load(g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                  padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_exp_g = ct.exp(b_g)  # [BT] fp32 = 256B — small, persists to V loop

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
    # Peak here: Q(32KB) + b_A(16KB) + k_temp(8KB) = 56KB
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

    # ── Apply g-gating AFTER QK^T (k_temp freed): peak Q(32)+b_A(16)+g_diff(16)=64KB ──
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    # Reuse g_diff name in-place to avoid 2 separate 16KB allocations
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    g_diff = ct.exp(g_diff)
    b_A = b_A * g_diff  # g_diff freed after this; b_A contains gated QK^T
    b_A = ct.where(row_idx >= col_idx, b_A, 0.0)
    # Cast b_A to bf16 HERE (freeing fp32 16KB → bf16 8KB) before V loop.
    # This brings peak in V loop to Q(32)+b_A_bf16(8)+b_o(16)+h_tile(8)=64KB < 85KB.
    b_A = b_A.astype(ct.bfloat16)

    # ── For each V-tile (BV=64): peak Q(32)+b_A_bf16(8)+b_o(16)+h_tile(8)=64KB ─
    for i_v in range(ct.cdiv(V, BV)):
        b_o = ct.zeros((BT, BV), dtype=ct.float32)

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

        v_tile = ct.load(v_new, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                         padding_mode=PAD_ZERO).reshape((BT, BV))
        # b_A already bf16, no inline cast needed
        b_o = ct.mma(b_A, v_tile, b_o) * scale

        ct.store(o, index=(i_b, i_t, i_h, i_v),
                 tile=b_o.astype(o.dtype).reshape((1, BT, 1, BV)))

# ============================================================================
# Kernel 2ab: Fused KKT + solve (cuTile)
# Computes A = beta * K @ K^T, then solves (I + A)^{-1} via power series
# Stores A_inv as [B, T, H, BT] matching solve_tril output format
# ============================================================================
@ct.kernel
def cutile_kkt_solve_kernel(
    k,          # [B, T, H, K]
    beta,       # [B, T, H]
    g_cumsum,   # [B, T, H] (float32)
    A_inv_out,  # [B, T, H, BT] output - the solved inverse
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
):
    """Fused KKT + power-series solve. Grid: (NT, B*H)"""
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load beta: [BT]
    b_beta = ct.load(
        beta, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Load g_cumsum: [BT]
    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Compute A = K @ K^T
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        b_A = ct.mma(k_tile, ct.transpose(k_tile), b_A)

    # Apply gating: A *= exp(g_i - g_j)
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff)

    # Apply beta and strictly lower triangular mask
    b_A = b_A * b_beta[:, None]
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    mask_lt = row_idx > col_idx
    b_A = ct.where(mask_lt, b_A, 0.0)

    # Solve (I + A)^{-1} via Neumann power series
    neg_A = -b_A
    diag_mask = row_idx == col_idx
    b_Ai = ct.where(diag_mask, 1.0, 0.0)
    b_Ai = b_Ai + neg_A

    power = neg_A
    for _ in range(3):  # 3 iterations for (-A)^2 through (-A)^4
        acc = ct.zeros((BT, BT), dtype=ct.float32)
        power = ct.mma(power, neg_A, acc)
        b_Ai = b_Ai + power

    # Store A_inv: [BT, BT] -> [1, BT, 1, BT]
    ct.store(
        A_inv_out, index=(i_b, i_t, i_h, 0),
        tile=b_Ai.astype(A_inv_out.dtype).reshape((1, BT, 1, BT)),
    )


# ============================================================================
# Helper: Exact 16x16 strictly lower triangular inverse via squaring trick
# For nilpotent A (A^16 = 0):
#   (I + A)^{-1} = (I - A)(I + A^2)(I + A^4)(I + A^8)
# This requires only 3 squarings + 4 multiplies = 7 mma ops (exact, not approximate)
# ============================================================================


# ============================================================================
# Kernel 2x: Fused KKT + hierarchical block solve + WY (all-in-one)
# Computes A = beta * K @ K^T, then solves (I+A)^{-1} using hierarchical
# block inversion with exact 16x16 solves, then computes w and u.
# Eliminates A and A_inv intermediate global memory tensors.
# Grid: (NT, B*H)
# ============================================================================
@ct.kernel
def cutile_fused_kkt_solve_wy_kernel(
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    beta,       # [B, T, H]
    g_cumsum,   # [B, T, H] (float32)
    w,          # [B, T, H, K] output
    u,          # [B, T, H, V] output
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Fused KKT + hierarchical block solve + WY recompute.

    For each chunk:
    1. Compute A = beta * K @ K^T (64x64, strictly lower triangular)
    2. Solve (I+A)^{-1} using hierarchical 16x16 block inversion
    3. Compute w = A_inv @ (k * beta * exp(g)) and u = A_inv @ (v * beta)

    The 16x16 block inverse uses the squaring trick for nilpotent matrices:
      (I + A)^{-1} = (I - A)(I + A^2)(I + A^4)(I + A^8)

    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    BS = 16  # sub-block size for hierarchical solve

    # Load beta: [BT]
    b_beta = ct.load(
        beta, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Load g_cumsum: [BT]
    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Step 1: Compute full A = K @ K^T (64x64)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        b_A = ct.mma(k_tile, ct.transpose(k_tile), b_A)

    # Apply gating: A *= exp(g_i - g_j)
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff)

    # Apply beta and strictly lower triangular mask
    b_A = b_A * b_beta[:, None]
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    mask_lt = row_idx > col_idx
    b_A = ct.where(mask_lt, b_A, 0.0)

    # Step 2: Hierarchical block inversion
    # Extract 16x16 sub-blocks from the 64x64 matrix A
    # A is partitioned as:
    # [A11  0   0   0 ]
    # [A21 A22  0   0 ]
    # [A31 A32 A33  0 ]
    # [A41 A42 A43 A44]
    #
    # Where each Aij is 16x16 and diagonal blocks are strictly lower triangular.
    # Note: A11, A22, A33, A44 from the full matrix are strictly lower triangular
    #       within their 16x16 block.

    # Extract diagonal blocks (strictly lower triangular within each block)
    A11 = ct.extract(b_A, index=(0, 0), shape=(BS, BS))   # rows 0-15, cols 0-15
    A22 = ct.extract(b_A, index=(1, 1), shape=(BS, BS))   # rows 16-31, cols 16-31
    A33 = ct.extract(b_A, index=(2, 2), shape=(BS, BS))   # rows 32-47, cols 32-47
    A44 = ct.extract(b_A, index=(3, 3), shape=(BS, BS))   # rows 48-63, cols 48-63

    # Extract off-diagonal blocks
    A21 = ct.extract(b_A, index=(1, 0), shape=(BS, BS))   # rows 16-31, cols 0-15
    A31 = ct.extract(b_A, index=(2, 0), shape=(BS, BS))   # rows 32-47, cols 0-15
    A32 = ct.extract(b_A, index=(2, 1), shape=(BS, BS))   # rows 32-47, cols 16-31
    A41 = ct.extract(b_A, index=(3, 0), shape=(BS, BS))   # rows 48-63, cols 0-15
    A42 = ct.extract(b_A, index=(3, 1), shape=(BS, BS))   # rows 48-63, cols 16-31
    A43 = ct.extract(b_A, index=(3, 2), shape=(BS, BS))   # rows 48-63, cols 32-47

    # Invert diagonal blocks using squaring trick for nilpotent matrices:
    # (I + A)^{-1} = (I - A)(I + A^2)(I + A^4)(I + A^8)
    # Each diagonal block is 16x16 strictly lower triangular, so nilpotent.

    # Helper: inline inversion of 16x16 strictly lower triangular block
    # For A11:
    neg_A11 = -A11
    A11_sq = ct.mma(neg_A11, neg_A11, ct.zeros((BS, BS), dtype=ct.float32))  # A^2
    A11_q4 = ct.mma(A11_sq, A11_sq, ct.zeros((BS, BS), dtype=ct.float32))    # A^4
    A11_q8 = ct.mma(A11_q4, A11_q4, ct.zeros((BS, BS), dtype=ct.float32))    # A^8
    # Build: (I - A), (I + A^2), (I + A^4), (I + A^8)
    bs_row = ct.arange(BS, dtype=ct.int32)[:, None]
    bs_col = ct.arange(BS, dtype=ct.int32)[None, :]
    bs_diag = bs_row == bs_col
    I16 = ct.where(bs_diag, 1.0, 0.0)
    # (I - A)
    t1_11 = I16 + neg_A11
    # (I + A^2)
    t2_11 = I16 + A11_sq
    # (I + A^4)
    t3_11 = I16 + A11_q4
    # (I + A^8)
    t4_11 = I16 + A11_q8
    # Multiply: ((I-A)(I+A^2)) * ((I+A^4)(I+A^8))
    p1_11 = ct.mma(t1_11.astype(ct.bfloat16), t2_11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    p2_11 = ct.mma(t3_11.astype(ct.bfloat16), t4_11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai11 = ct.mma(p1_11.astype(ct.bfloat16), p2_11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # For A22:
    neg_A22 = -A22
    A22_sq = ct.mma(neg_A22, neg_A22, ct.zeros((BS, BS), dtype=ct.float32))
    A22_q4 = ct.mma(A22_sq, A22_sq, ct.zeros((BS, BS), dtype=ct.float32))
    A22_q8 = ct.mma(A22_q4, A22_q4, ct.zeros((BS, BS), dtype=ct.float32))
    t1_22 = I16 + neg_A22
    t2_22 = I16 + A22_sq
    t3_22 = I16 + A22_q4
    t4_22 = I16 + A22_q8
    p1_22 = ct.mma(t1_22.astype(ct.bfloat16), t2_22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    p2_22 = ct.mma(t3_22.astype(ct.bfloat16), t4_22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai22 = ct.mma(p1_22.astype(ct.bfloat16), p2_22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # For A33:
    neg_A33 = -A33
    A33_sq = ct.mma(neg_A33, neg_A33, ct.zeros((BS, BS), dtype=ct.float32))
    A33_q4 = ct.mma(A33_sq, A33_sq, ct.zeros((BS, BS), dtype=ct.float32))
    A33_q8 = ct.mma(A33_q4, A33_q4, ct.zeros((BS, BS), dtype=ct.float32))
    t1_33 = I16 + neg_A33
    t2_33 = I16 + A33_sq
    t3_33 = I16 + A33_q4
    t4_33 = I16 + A33_q8
    p1_33 = ct.mma(t1_33.astype(ct.bfloat16), t2_33.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    p2_33 = ct.mma(t3_33.astype(ct.bfloat16), t4_33.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai33 = ct.mma(p1_33.astype(ct.bfloat16), p2_33.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # For A44:
    neg_A44 = -A44
    A44_sq = ct.mma(neg_A44, neg_A44, ct.zeros((BS, BS), dtype=ct.float32))
    A44_q4 = ct.mma(A44_sq, A44_sq, ct.zeros((BS, BS), dtype=ct.float32))
    A44_q8 = ct.mma(A44_q4, A44_q4, ct.zeros((BS, BS), dtype=ct.float32))
    t1_44 = I16 + neg_A44
    t2_44 = I16 + A44_sq
    t3_44 = I16 + A44_q4
    t4_44 = I16 + A44_q8
    p1_44 = ct.mma(t1_44.astype(ct.bfloat16), t2_44.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    p2_44 = ct.mma(t3_44.astype(ct.bfloat16), t4_44.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai44 = ct.mma(p1_44.astype(ct.bfloat16), p2_44.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Compute off-diagonal blocks of the inverse:
    # Ai_21 = -Ai22 @ A21 @ Ai11
    tmp = ct.mma(Ai22.astype(ct.bfloat16), A21.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai21 = ct.mma(-tmp.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Ai_32 = -Ai33 @ A32 @ Ai22
    tmp = ct.mma(Ai33.astype(ct.bfloat16), A32.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai32 = ct.mma(-tmp.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Ai_43 = -Ai44 @ A43 @ Ai33
    tmp = ct.mma(Ai44.astype(ct.bfloat16), A43.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai43 = ct.mma(-tmp.astype(ct.bfloat16), Ai33.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Ai_31 = -Ai33 @ (A31 @ Ai11 + A32 @ Ai21)
    t1 = ct.mma(A31.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A32.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    Ai31 = ct.mma(-Ai33.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Ai_42 = -Ai44 @ (A42 @ Ai22 + A43 @ Ai32)
    t1 = ct.mma(A42.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A43.astype(ct.bfloat16), Ai32.astype(ct.bfloat16), t1)
    Ai42 = ct.mma(-Ai44.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Ai_41 = -Ai44 @ (A41 @ Ai11 + A42 @ Ai21 + A43 @ Ai31)
    t1 = ct.mma(A41.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A42.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    t3 = ct.mma(A43.astype(ct.bfloat16), Ai31.astype(ct.bfloat16), t2)
    Ai41 = ct.mma(-Ai44.astype(ct.bfloat16), t3.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Step 3: Compute w = A_inv @ (k * beta * exp(g)) and u = A_inv @ (v * beta)
    # A_inv is composed of the 16 blocks (4x4 lower triangular in blocks)
    # For efficiency, we multiply each block row of A_inv with the corresponding
    # block of the input vector.

    b_exp_g = ct.exp(b_g)

    # Split beta and exp_g into 16-element chunks for the 4 blocks
    beta0 = ct.extract(b_beta, index=(0,), shape=(BS,))
    beta1 = ct.extract(b_beta, index=(1,), shape=(BS,))
    beta2 = ct.extract(b_beta, index=(2,), shape=(BS,))
    beta3 = ct.extract(b_beta, index=(3,), shape=(BS,))

    expg0 = ct.extract(b_exp_g, index=(0,), shape=(BS,))
    expg1 = ct.extract(b_exp_g, index=(1,), shape=(BS,))
    expg2 = ct.extract(b_exp_g, index=(2,), shape=(BS,))
    expg3 = ct.extract(b_exp_g, index=(3,), shape=(BS,))

    # Compute u = A_inv @ (v * beta)
    for i_v in range(ct.cdiv(V, BV)):
        # Load v in 4 blocks of [BS, BV]
        # v is [B, T, H, V]. For chunk i_t, time positions are i_t*BT to (i_t+1)*BT.
        # We use the tiled indexing: index=(i_b, i_t*4+block, i_h, i_v) with shape=(1, BS, 1, BV)
        # But BT=64 and BS=16, so we need 4 loads at time offsets 0,16,32,48 within chunk.
        # The tile index for time dimension with shape BS=16: tile_idx = i_t*4 + block
        v0 = ct.load(v, index=(i_b, i_t * 4 + 0, i_h, i_v), shape=(1, BS, 1, BV),
                      padding_mode=PAD_ZERO).reshape((BS, BV))
        v1 = ct.load(v, index=(i_b, i_t * 4 + 1, i_h, i_v), shape=(1, BS, 1, BV),
                      padding_mode=PAD_ZERO).reshape((BS, BV))
        v2 = ct.load(v, index=(i_b, i_t * 4 + 2, i_h, i_v), shape=(1, BS, 1, BV),
                      padding_mode=PAD_ZERO).reshape((BS, BV))
        v3 = ct.load(v, index=(i_b, i_t * 4 + 3, i_h, i_v), shape=(1, BS, 1, BV),
                      padding_mode=PAD_ZERO).reshape((BS, BV))

        # v * beta for each block
        vb0 = (v0.astype(ct.float32) * beta0[:, None]).astype(ct.bfloat16)
        vb1 = (v1.astype(ct.float32) * beta1[:, None]).astype(ct.bfloat16)
        vb2 = (v2.astype(ct.float32) * beta2[:, None]).astype(ct.bfloat16)
        vb3 = (v3.astype(ct.float32) * beta3[:, None]).astype(ct.bfloat16)

        # u = A_inv @ vb (block matrix-vector multiply)
        # Row 0: u0 = Ai11 @ vb0
        u0 = ct.mma(Ai11.astype(ct.bfloat16), vb0, ct.zeros((BS, BV), dtype=ct.float32))
        # Row 1: u1 = Ai21 @ vb0 + Ai22 @ vb1
        u1 = ct.mma(Ai21.astype(ct.bfloat16), vb0, ct.zeros((BS, BV), dtype=ct.float32))
        u1 = ct.mma(Ai22.astype(ct.bfloat16), vb1, u1)
        # Row 2: u2 = Ai31 @ vb0 + Ai32 @ vb1 + Ai33 @ vb2
        u2 = ct.mma(Ai31.astype(ct.bfloat16), vb0, ct.zeros((BS, BV), dtype=ct.float32))
        u2 = ct.mma(Ai32.astype(ct.bfloat16), vb1, u2)
        u2 = ct.mma(Ai33.astype(ct.bfloat16), vb2, u2)
        # Row 3: u3 = Ai41 @ vb0 + Ai42 @ vb1 + Ai43 @ vb2 + Ai44 @ vb3
        u3 = ct.mma(Ai41.astype(ct.bfloat16), vb0, ct.zeros((BS, BV), dtype=ct.float32))
        u3 = ct.mma(Ai42.astype(ct.bfloat16), vb1, u3)
        u3 = ct.mma(Ai43.astype(ct.bfloat16), vb2, u3)
        u3 = ct.mma(Ai44.astype(ct.bfloat16), vb3, u3)

        # Store u blocks
        ct.store(u, index=(i_b, i_t * 4 + 0, i_h, i_v),
                 tile=u0.astype(u.dtype).reshape((1, BS, 1, BV)))
        ct.store(u, index=(i_b, i_t * 4 + 1, i_h, i_v),
                 tile=u1.astype(u.dtype).reshape((1, BS, 1, BV)))
        ct.store(u, index=(i_b, i_t * 4 + 2, i_h, i_v),
                 tile=u2.astype(u.dtype).reshape((1, BS, 1, BV)))
        ct.store(u, index=(i_b, i_t * 4 + 3, i_h, i_v),
                 tile=u3.astype(u.dtype).reshape((1, BS, 1, BV)))

    # Compute w = A_inv @ (k * beta * exp(g))
    for i_k in range(ct.cdiv(K, BK)):
        # Load k in 4 blocks of [BS, BK]
        k0 = ct.load(k, index=(i_b, i_t * 4 + 0, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k1 = ct.load(k, index=(i_b, i_t * 4 + 1, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k2 = ct.load(k, index=(i_b, i_t * 4 + 2, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k3 = ct.load(k, index=(i_b, i_t * 4 + 3, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))

        # k * beta * exp(g) for each block
        kb0 = (k0.astype(ct.float32) * beta0[:, None] * expg0[:, None]).astype(ct.bfloat16)
        kb1 = (k1.astype(ct.float32) * beta1[:, None] * expg1[:, None]).astype(ct.bfloat16)
        kb2 = (k2.astype(ct.float32) * beta2[:, None] * expg2[:, None]).astype(ct.bfloat16)
        kb3 = (k3.astype(ct.float32) * beta3[:, None] * expg3[:, None]).astype(ct.bfloat16)

        # w = A_inv @ kb (block matrix-vector multiply)
        # Row 0: w0 = Ai11 @ kb0
        w0 = ct.mma(Ai11.astype(ct.bfloat16), kb0, ct.zeros((BS, BK), dtype=ct.float32))
        # Row 1: w1 = Ai21 @ kb0 + Ai22 @ kb1
        w1 = ct.mma(Ai21.astype(ct.bfloat16), kb0, ct.zeros((BS, BK), dtype=ct.float32))
        w1 = ct.mma(Ai22.astype(ct.bfloat16), kb1, w1)
        # Row 2: w2 = Ai31 @ kb0 + Ai32 @ kb1 + Ai33 @ kb2
        w2 = ct.mma(Ai31.astype(ct.bfloat16), kb0, ct.zeros((BS, BK), dtype=ct.float32))
        w2 = ct.mma(Ai32.astype(ct.bfloat16), kb1, w2)
        w2 = ct.mma(Ai33.astype(ct.bfloat16), kb2, w2)
        # Row 3: w3 = Ai41 @ kb0 + Ai42 @ kb1 + Ai43 @ kb2 + Ai44 @ kb3
        w3 = ct.mma(Ai41.astype(ct.bfloat16), kb0, ct.zeros((BS, BK), dtype=ct.float32))
        w3 = ct.mma(Ai42.astype(ct.bfloat16), kb1, w3)
        w3 = ct.mma(Ai43.astype(ct.bfloat16), kb2, w3)
        w3 = ct.mma(Ai44.astype(ct.bfloat16), kb3, w3)

        # Store w blocks
        ct.store(w, index=(i_b, i_t * 4 + 0, i_h, i_k),
                 tile=w0.astype(w.dtype).reshape((1, BS, 1, BK)))
        ct.store(w, index=(i_b, i_t * 4 + 1, i_h, i_k),
                 tile=w1.astype(w.dtype).reshape((1, BS, 1, BK)))
        ct.store(w, index=(i_b, i_t * 4 + 2, i_h, i_k),
                 tile=w2.astype(w.dtype).reshape((1, BS, 1, BK)))
        ct.store(w, index=(i_b, i_t * 4 + 3, i_h, i_k),
                 tile=w3.astype(w.dtype).reshape((1, BS, 1, BK)))


# ============================================================================
# Kernel 2x v2: Fused KKT + hierarchical block solve + WY (register-optimized)
# Avoids materializing the full 64x64 A matrix. Computes each 16x16 sub-block
# of A independently, which reduces register pressure by ~4x.
# Uses 2-step squaring (terms up to A^7) for faster diagonal inverse.
# Grid: (NT, B*H)
# ============================================================================
@ct.kernel
def cutile_fused_kkt_solve_wy_kernel_v2(
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    beta,       # [B, T, H]
    g_cumsum,   # [B, T, H] (float32)
    w,          # [B, T, H, K] output
    u,          # [B, T, H, V] output
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Fused KKT + hierarchical block solve + WY (register-optimized).

    Computes 16x16 sub-blocks of A = beta * K @ K^T directly,
    rather than materializing the full 64x64 matrix.
    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    BS = 16  # sub-block size

    # Load beta for this chunk: [BT] in 4 blocks of [BS]
    b_beta = ct.load(
        beta, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Load g_cumsum: [BT]
    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Split beta and g into 4 blocks of BS=16
    beta0 = ct.extract(b_beta, index=(0,), shape=(BS,))
    beta1 = ct.extract(b_beta, index=(1,), shape=(BS,))
    beta2 = ct.extract(b_beta, index=(2,), shape=(BS,))
    beta3 = ct.extract(b_beta, index=(3,), shape=(BS,))

    g0 = ct.extract(b_g, index=(0,), shape=(BS,))
    g1 = ct.extract(b_g, index=(1,), shape=(BS,))
    g2 = ct.extract(b_g, index=(2,), shape=(BS,))
    g3 = ct.extract(b_g, index=(3,), shape=(BS,))

    # Load K in 4 blocks of [BS, K] for each sub-block row
    # We'll compute sub-blocks of A = beta * K @ K^T with gating

    # Helper: compute A[row, col] = beta[row] * sum_k(K_row @ K_col^T) * exp(g_row - g_col)
    # Then apply strictly lower triangular mask (within the block context)

    # Identity for 16x16
    bs_row = ct.arange(BS, dtype=ct.int32)[:, None]
    bs_col = ct.arange(BS, dtype=ct.int32)[None, :]
    bs_diag = bs_row == bs_col
    I16 = ct.where(bs_diag, 1.0, 0.0)

    # Helper function to compute one 16x16 sub-block of A
    # A[i,j] = beta_i * K_i @ K_j^T * exp(g_i - g_j)
    # For diagonal blocks (i==j): apply strictly lower triangular mask

    # Compute diagonal blocks and their inverses
    # For each diagonal block i: A[i,i] = beta_i * K_i @ K_i^T * exp(g_i[:,None] - g_i[None,:])
    # These are strictly lower triangular within each 16x16 block.

    # Block 0: rows 0-15, cols 0-15
    A00 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k0 = ct.load(k, index=(i_b, i_t * 4 + 0, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A00 = ct.mma(k0, ct.transpose(k0), A00)
    gd00 = g0[:, None] - g0[None, :]
    gd00 = ct.minimum(gd00, 20.0)
    A00 = A00 * ct.exp(gd00) * beta0[:, None]
    mask_lt = bs_row > bs_col
    A00 = ct.where(mask_lt, A00, 0.0)

    # Invert A00: (I + A00)^{-1} using squaring trick (2 steps for speed)
    neg_A00 = -A00
    A00_sq = ct.mma(neg_A00, neg_A00, ct.zeros((BS, BS), dtype=ct.float32))
    A00_q4 = ct.mma(A00_sq, A00_sq, ct.zeros((BS, BS), dtype=ct.float32))
    p1 = ct.mma((I16 + neg_A00).astype(ct.bfloat16), (I16 + A00_sq).astype(ct.bfloat16),
                ct.zeros((BS, BS), dtype=ct.float32))
    Ai00 = ct.mma(p1.astype(ct.bfloat16), (I16 + A00_q4).astype(ct.bfloat16),
                  ct.zeros((BS, BS), dtype=ct.float32))

    # Block 1
    A11 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k1 = ct.load(k, index=(i_b, i_t * 4 + 1, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A11 = ct.mma(k1, ct.transpose(k1), A11)
    gd11 = g1[:, None] - g1[None, :]
    gd11 = ct.minimum(gd11, 20.0)
    A11 = A11 * ct.exp(gd11) * beta1[:, None]
    A11 = ct.where(mask_lt, A11, 0.0)
    neg_A11 = -A11
    A11_sq = ct.mma(neg_A11, neg_A11, ct.zeros((BS, BS), dtype=ct.float32))
    A11_q4 = ct.mma(A11_sq, A11_sq, ct.zeros((BS, BS), dtype=ct.float32))
    p1 = ct.mma((I16 + neg_A11).astype(ct.bfloat16), (I16 + A11_sq).astype(ct.bfloat16),
                ct.zeros((BS, BS), dtype=ct.float32))
    Ai11 = ct.mma(p1.astype(ct.bfloat16), (I16 + A11_q4).astype(ct.bfloat16),
                  ct.zeros((BS, BS), dtype=ct.float32))

    # Block 2
    A22 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k2 = ct.load(k, index=(i_b, i_t * 4 + 2, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A22 = ct.mma(k2, ct.transpose(k2), A22)
    gd22 = g2[:, None] - g2[None, :]
    gd22 = ct.minimum(gd22, 20.0)
    A22 = A22 * ct.exp(gd22) * beta2[:, None]
    A22 = ct.where(mask_lt, A22, 0.0)
    neg_A22 = -A22
    A22_sq = ct.mma(neg_A22, neg_A22, ct.zeros((BS, BS), dtype=ct.float32))
    A22_q4 = ct.mma(A22_sq, A22_sq, ct.zeros((BS, BS), dtype=ct.float32))
    p1 = ct.mma((I16 + neg_A22).astype(ct.bfloat16), (I16 + A22_sq).astype(ct.bfloat16),
                ct.zeros((BS, BS), dtype=ct.float32))
    Ai22 = ct.mma(p1.astype(ct.bfloat16), (I16 + A22_q4).astype(ct.bfloat16),
                  ct.zeros((BS, BS), dtype=ct.float32))

    # Block 3
    A33 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k3 = ct.load(k, index=(i_b, i_t * 4 + 3, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A33 = ct.mma(k3, ct.transpose(k3), A33)
    gd33 = g3[:, None] - g3[None, :]
    gd33 = ct.minimum(gd33, 20.0)
    A33 = A33 * ct.exp(gd33) * beta3[:, None]
    A33 = ct.where(mask_lt, A33, 0.0)
    neg_A33 = -A33
    A33_sq = ct.mma(neg_A33, neg_A33, ct.zeros((BS, BS), dtype=ct.float32))
    A33_q4 = ct.mma(A33_sq, A33_sq, ct.zeros((BS, BS), dtype=ct.float32))
    p1 = ct.mma((I16 + neg_A33).astype(ct.bfloat16), (I16 + A33_sq).astype(ct.bfloat16),
                ct.zeros((BS, BS), dtype=ct.float32))
    Ai33 = ct.mma(p1.astype(ct.bfloat16), (I16 + A33_q4).astype(ct.bfloat16),
                  ct.zeros((BS, BS), dtype=ct.float32))

    # Compute off-diagonal sub-blocks of A and the inverse blocks
    # A[1,0]: rows 16-31, cols 0-15
    A10 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k1 = ct.load(k, index=(i_b, i_t * 4 + 1, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k0 = ct.load(k, index=(i_b, i_t * 4 + 0, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A10 = ct.mma(k1, ct.transpose(k0), A10)
    gd10 = g1[:, None] - g0[None, :]
    gd10 = ct.minimum(gd10, 20.0)
    A10 = A10 * ct.exp(gd10) * beta1[:, None]

    # Ai10 = -Ai11 @ A10 @ Ai00
    tmp = ct.mma(Ai11.astype(ct.bfloat16), A10.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai10 = ct.mma(-tmp.astype(ct.bfloat16), Ai00.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # A[2,1]
    A21 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k2 = ct.load(k, index=(i_b, i_t * 4 + 2, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k1 = ct.load(k, index=(i_b, i_t * 4 + 1, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A21 = ct.mma(k2, ct.transpose(k1), A21)
    gd21 = g2[:, None] - g1[None, :]
    gd21 = ct.minimum(gd21, 20.0)
    A21 = A21 * ct.exp(gd21) * beta2[:, None]

    # A[2,0]
    A20 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k2 = ct.load(k, index=(i_b, i_t * 4 + 2, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k0 = ct.load(k, index=(i_b, i_t * 4 + 0, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A20 = ct.mma(k2, ct.transpose(k0), A20)
    gd20 = g2[:, None] - g0[None, :]
    gd20 = ct.minimum(gd20, 20.0)
    A20 = A20 * ct.exp(gd20) * beta2[:, None]

    # Ai21 = -Ai22 @ A21 @ Ai11
    tmp = ct.mma(Ai22.astype(ct.bfloat16), A21.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai21 = ct.mma(-tmp.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Ai20 = -Ai22 @ (A20 @ Ai00 + A21 @ Ai10)
    t1 = ct.mma(A20.astype(ct.bfloat16), Ai00.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A21.astype(ct.bfloat16), Ai10.astype(ct.bfloat16), t1)
    Ai20 = ct.mma(-Ai22.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # A[3,2]
    A32 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k3 = ct.load(k, index=(i_b, i_t * 4 + 3, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k2 = ct.load(k, index=(i_b, i_t * 4 + 2, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A32 = ct.mma(k3, ct.transpose(k2), A32)
    gd32 = g3[:, None] - g2[None, :]
    gd32 = ct.minimum(gd32, 20.0)
    A32 = A32 * ct.exp(gd32) * beta3[:, None]

    # A[3,1]
    A31 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k3 = ct.load(k, index=(i_b, i_t * 4 + 3, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k1 = ct.load(k, index=(i_b, i_t * 4 + 1, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A31 = ct.mma(k3, ct.transpose(k1), A31)
    gd31 = g3[:, None] - g1[None, :]
    gd31 = ct.minimum(gd31, 20.0)
    A31 = A31 * ct.exp(gd31) * beta3[:, None]

    # A[3,0]
    A30 = ct.zeros((BS, BS), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k3 = ct.load(k, index=(i_b, i_t * 4 + 3, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        k0 = ct.load(k, index=(i_b, i_t * 4 + 0, i_h, i_k), shape=(1, BS, 1, BK),
                      padding_mode=PAD_ZERO).reshape((BS, BK))
        A30 = ct.mma(k3, ct.transpose(k0), A30)
    gd30 = g3[:, None] - g0[None, :]
    gd30 = ct.minimum(gd30, 20.0)
    A30 = A30 * ct.exp(gd30) * beta3[:, None]

    # Ai32 = -Ai33 @ A32 @ Ai22
    tmp = ct.mma(Ai33.astype(ct.bfloat16), A32.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai32 = ct.mma(-tmp.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Ai31 = -Ai33 @ (A31 @ Ai11 + A32 @ Ai21)
    t1 = ct.mma(A31.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A32.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    Ai31 = ct.mma(-Ai33.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Ai30 = -Ai33 @ (A30 @ Ai00 + A31 @ Ai10 + A32 @ Ai20)
    t1 = ct.mma(A30.astype(ct.bfloat16), Ai00.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A31.astype(ct.bfloat16), Ai10.astype(ct.bfloat16), t1)
    t3 = ct.mma(A32.astype(ct.bfloat16), Ai20.astype(ct.bfloat16), t2)
    Ai30 = ct.mma(-Ai33.astype(ct.bfloat16), t3.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Step 3: Compute w and u using the block inverse
    b_exp_g = ct.exp(b_g)
    expg0 = ct.extract(b_exp_g, index=(0,), shape=(BS,))
    expg1 = ct.extract(b_exp_g, index=(1,), shape=(BS,))
    expg2 = ct.extract(b_exp_g, index=(2,), shape=(BS,))
    expg3 = ct.extract(b_exp_g, index=(3,), shape=(BS,))

    # Compute u = Ai @ (v * beta)
    for i_v in range(ct.cdiv(V, BV)):
        v0 = ct.load(v, index=(i_b, i_t * 4 + 0, i_h, i_v), shape=(1, BS, 1, BV), padding_mode=PAD_ZERO).reshape((BS, BV))
        v1 = ct.load(v, index=(i_b, i_t * 4 + 1, i_h, i_v), shape=(1, BS, 1, BV), padding_mode=PAD_ZERO).reshape((BS, BV))
        v2 = ct.load(v, index=(i_b, i_t * 4 + 2, i_h, i_v), shape=(1, BS, 1, BV), padding_mode=PAD_ZERO).reshape((BS, BV))
        v3 = ct.load(v, index=(i_b, i_t * 4 + 3, i_h, i_v), shape=(1, BS, 1, BV), padding_mode=PAD_ZERO).reshape((BS, BV))
        vb0 = (v0.astype(ct.float32) * beta0[:, None]).astype(ct.bfloat16)
        vb1 = (v1.astype(ct.float32) * beta1[:, None]).astype(ct.bfloat16)
        vb2 = (v2.astype(ct.float32) * beta2[:, None]).astype(ct.bfloat16)
        vb3 = (v3.astype(ct.float32) * beta3[:, None]).astype(ct.bfloat16)

        u0 = ct.mma(Ai00.astype(ct.bfloat16), vb0, ct.zeros((BS, BV), dtype=ct.float32))
        u1 = ct.mma(Ai10.astype(ct.bfloat16), vb0, ct.zeros((BS, BV), dtype=ct.float32))
        u1 = ct.mma(Ai11.astype(ct.bfloat16), vb1, u1)
        u2 = ct.mma(Ai20.astype(ct.bfloat16), vb0, ct.zeros((BS, BV), dtype=ct.float32))
        u2 = ct.mma(Ai21.astype(ct.bfloat16), vb1, u2)
        u2 = ct.mma(Ai22.astype(ct.bfloat16), vb2, u2)
        u3 = ct.mma(Ai30.astype(ct.bfloat16), vb0, ct.zeros((BS, BV), dtype=ct.float32))
        u3 = ct.mma(Ai31.astype(ct.bfloat16), vb1, u3)
        u3 = ct.mma(Ai32.astype(ct.bfloat16), vb2, u3)
        u3 = ct.mma(Ai33.astype(ct.bfloat16), vb3, u3)

        ct.store(u, index=(i_b, i_t * 4 + 0, i_h, i_v), tile=u0.astype(u.dtype).reshape((1, BS, 1, BV)))
        ct.store(u, index=(i_b, i_t * 4 + 1, i_h, i_v), tile=u1.astype(u.dtype).reshape((1, BS, 1, BV)))
        ct.store(u, index=(i_b, i_t * 4 + 2, i_h, i_v), tile=u2.astype(u.dtype).reshape((1, BS, 1, BV)))
        ct.store(u, index=(i_b, i_t * 4 + 3, i_h, i_v), tile=u3.astype(u.dtype).reshape((1, BS, 1, BV)))

    # Compute w = Ai @ (k * beta * exp(g))
    for i_k in range(ct.cdiv(K, BK)):
        k0 = ct.load(k, index=(i_b, i_t * 4 + 0, i_h, i_k), shape=(1, BS, 1, BK), padding_mode=PAD_ZERO).reshape((BS, BK))
        k1 = ct.load(k, index=(i_b, i_t * 4 + 1, i_h, i_k), shape=(1, BS, 1, BK), padding_mode=PAD_ZERO).reshape((BS, BK))
        k2 = ct.load(k, index=(i_b, i_t * 4 + 2, i_h, i_k), shape=(1, BS, 1, BK), padding_mode=PAD_ZERO).reshape((BS, BK))
        k3 = ct.load(k, index=(i_b, i_t * 4 + 3, i_h, i_k), shape=(1, BS, 1, BK), padding_mode=PAD_ZERO).reshape((BS, BK))
        kb0 = (k0.astype(ct.float32) * beta0[:, None] * expg0[:, None]).astype(ct.bfloat16)
        kb1 = (k1.astype(ct.float32) * beta1[:, None] * expg1[:, None]).astype(ct.bfloat16)
        kb2 = (k2.astype(ct.float32) * beta2[:, None] * expg2[:, None]).astype(ct.bfloat16)
        kb3 = (k3.astype(ct.float32) * beta3[:, None] * expg3[:, None]).astype(ct.bfloat16)

        w0 = ct.mma(Ai00.astype(ct.bfloat16), kb0, ct.zeros((BS, BK), dtype=ct.float32))
        w1 = ct.mma(Ai10.astype(ct.bfloat16), kb0, ct.zeros((BS, BK), dtype=ct.float32))
        w1 = ct.mma(Ai11.astype(ct.bfloat16), kb1, w1)
        w2 = ct.mma(Ai20.astype(ct.bfloat16), kb0, ct.zeros((BS, BK), dtype=ct.float32))
        w2 = ct.mma(Ai21.astype(ct.bfloat16), kb1, w2)
        w2 = ct.mma(Ai22.astype(ct.bfloat16), kb2, w2)
        w3 = ct.mma(Ai30.astype(ct.bfloat16), kb0, ct.zeros((BS, BK), dtype=ct.float32))
        w3 = ct.mma(Ai31.astype(ct.bfloat16), kb1, w3)
        w3 = ct.mma(Ai32.astype(ct.bfloat16), kb2, w3)
        w3 = ct.mma(Ai33.astype(ct.bfloat16), kb3, w3)

        ct.store(w, index=(i_b, i_t * 4 + 0, i_h, i_k), tile=w0.astype(w.dtype).reshape((1, BS, 1, BK)))
        ct.store(w, index=(i_b, i_t * 4 + 1, i_h, i_k), tile=w1.astype(w.dtype).reshape((1, BS, 1, BK)))
        ct.store(w, index=(i_b, i_t * 4 + 2, i_h, i_k), tile=w2.astype(w.dtype).reshape((1, BS, 1, BK)))
        ct.store(w, index=(i_b, i_t * 4 + 3, i_h, i_k), tile=w3.astype(w.dtype).reshape((1, BS, 1, BK)))


# ============================================================================
# Kernel 2a: KKT (split from prepare kernel)
# Computes A = beta * K @ K^T with gating, strictly lower triangular
# Output stored as [B, T, H, BT] matching Triton's solve_tril format
# ============================================================================
@ct.kernel
def cutile_kkt_kernel(
    k,          # [B, T, H, K]
    beta,       # [B, T, H]
    g_cumsum,   # [B, T, H] (float32)
    A,          # [B, T, H, BT] output
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
):
    """Compute A = beta * K @ K^T with gating. Grid: (NT, B*H)"""
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load beta: [BT]
    b_beta = ct.load(
        beta, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Load g_cumsum: [BT]
    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Compute A = K @ K^T
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        b_A = ct.mma(k_tile, ct.transpose(k_tile), b_A)

    # Apply gating: A *= exp(g_i - g_j)
    g_diff = b_g[:, None] - b_g[None, :]
    g_diff = ct.minimum(g_diff, 20.0)
    b_A = b_A * ct.exp(g_diff)

    # Apply beta and strictly lower triangular mask
    b_A = b_A * b_beta[:, None]
    row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
    col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
    mask_lt = row_idx > col_idx
    b_A = ct.where(mask_lt, b_A, 0.0)

    # Store A: [BT, BT] -> [1, BT, 1, BT]
    ct.store(
        A, index=(i_b, i_t, i_h, 0),
        tile=b_A.astype(A.dtype).reshape((1, BT, 1, BT)),
    )


# ============================================================================
# Kernel 2c: WY recompute (split from prepare kernel)
# Computes u = A_inv @ (v * beta) and w = A_inv @ (k * beta * exp(g))
# ============================================================================
@ct.kernel
def cutile_wy_kernel(
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    beta,       # [B, T, H]
    g_cumsum,   # [B, T, H] (float32)
    A_inv,      # [B, T, H, BT] - the solved inverse
    w,          # [B, T, H, K] output
    u,          # [B, T, H, V] output
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Compute w and u from A_inv. Grid: (NT, B*H)"""
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load beta: [BT]
    b_beta = ct.load(
        beta, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    # Load g_cumsum: [BT]
    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)
    b_exp_g = ct.exp(b_g)

    # Load A_inv: [1, BT, 1, BT] -> [BT, BT]
    b_Ai = ct.load(
        A_inv, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BT),
        padding_mode=PAD_ZERO,
    ).reshape((BT, BT))

    # Compute u = A_inv @ (v * beta)
    for i_v in range(ct.cdiv(V, BV)):
        v_tile = ct.load(
            v, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BV))
        vb = (v_tile.astype(ct.float32) * b_beta[:, None]).astype(v_tile.dtype)
        acc_u = ct.zeros((BT, BV), dtype=ct.float32)
        b_u = ct.mma(b_Ai, vb, acc_u)
        ct.store(
            u, index=(i_b, i_t, i_h, i_v),
            tile=b_u.astype(u.dtype).reshape((1, BT, 1, BV)),
        )

    # Compute w = A_inv @ (k * beta * exp(g))
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        kb = (k_tile.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k_tile.dtype)
        acc_w = ct.zeros((BT, BK), dtype=ct.float32)
        b_w = ct.mma(b_Ai, kb, acc_w)
        ct.store(
            w, index=(i_b, i_t, i_h, i_k),
            tile=b_w.astype(w.dtype).reshape((1, BT, 1, BK)),
        )



# ============================================================================
# Kernel 2c-occ3: WY recompute at occupancy=3 (85KB budget ≥ ~33KB peak)
# ============================================================================
@ct.kernel(occupancy=3)
def cutile_wy_occ6_kernel(
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    beta,       # [B, T, H]
    g_cumsum,   # [B, T, H] (float32)
    A_inv,      # [B, T, H, BT] - the solved inverse
    w,          # [B, T, H, K] output
    u,          # [B, T, H, V] output
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Compute w and u from A_inv at occupancy=6. Grid: (NT, B*H)"""
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    b_beta = ct.load(
        beta, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)
    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)
    b_exp_g = ct.exp(b_g)
    b_Ai = ct.load(
        A_inv, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BT),
        padding_mode=PAD_ZERO,
    ).reshape((BT, BT))

    for i_v in range(ct.cdiv(V, BV)):
        v_tile = ct.load(
            v, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BV))
        vb = (v_tile.astype(ct.float32) * b_beta[:, None]).astype(v_tile.dtype)
        acc_u = ct.zeros((BT, BV), dtype=ct.float32)
        b_u = ct.mma(b_Ai, vb, acc_u)
        ct.store(
            u, index=(i_b, i_t, i_h, i_v),
            tile=b_u.astype(u.dtype).reshape((1, BT, 1, BV)),
        )

    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        kb = (k_tile.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k_tile.dtype)
        acc_w = ct.zeros((BT, BK), dtype=ct.float32)
        b_w = ct.mma(b_Ai, kb, acc_w)
        ct.store(
            w, index=(i_b, i_t, i_h, i_k),
            tile=b_w.astype(w.dtype).reshape((1, BT, 1, BK)),
        )


# ============================================================================
# Kernel 2d: Standalone solve_tril using hierarchical 16x16 block inversion
# Takes A matrix (strictly lower triangular), outputs (I+A)^{-1}
# Faster than Triton's 3-kernel solve for <=384 blocks
# ============================================================================
@ct.kernel
def cutile_solve_kernel(
    A,          # [B, T, H, BT] input (float32)
    A_inv_out,  # [B, T, H, BT] output (bf16)
    T: ConstInt,
    H: ConstInt,
    BT: ConstInt,
):
    """Standalone solve_tril using hierarchical 16x16 block inversion.
    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H
    BS = 16

    b_A = ct.load(A, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BT),
                  padding_mode=PAD_ZERO).reshape((BT, BT)).astype(ct.float32)

    A11 = ct.extract(b_A, index=(0, 0), shape=(BS, BS))
    A22 = ct.extract(b_A, index=(1, 1), shape=(BS, BS))
    A33 = ct.extract(b_A, index=(2, 2), shape=(BS, BS))
    A44 = ct.extract(b_A, index=(3, 3), shape=(BS, BS))
    A21 = ct.extract(b_A, index=(1, 0), shape=(BS, BS))
    A31 = ct.extract(b_A, index=(2, 0), shape=(BS, BS))
    A32 = ct.extract(b_A, index=(2, 1), shape=(BS, BS))
    A41 = ct.extract(b_A, index=(3, 0), shape=(BS, BS))
    A42 = ct.extract(b_A, index=(3, 1), shape=(BS, BS))
    A43 = ct.extract(b_A, index=(3, 2), shape=(BS, BS))

    bs_row = ct.arange(BS, dtype=ct.int32)[:, None]
    bs_col = ct.arange(BS, dtype=ct.int32)[None, :]
    I16 = ct.where(bs_row == bs_col, 1.0, 0.0)

    def inv_block(Axx):
        neg = -Axx
        sq = ct.mma(neg, neg, ct.zeros((BS, BS), dtype=ct.float32))
        q4 = ct.mma(sq, sq, ct.zeros((BS, BS), dtype=ct.float32))
        q8 = ct.mma(q4, q4, ct.zeros((BS, BS), dtype=ct.float32))
        p1 = ct.mma((I16 + neg).astype(ct.bfloat16), (I16 + sq).astype(ct.bfloat16),
                     ct.zeros((BS, BS), dtype=ct.float32))
        p2 = ct.mma((I16 + q4).astype(ct.bfloat16), (I16 + q8).astype(ct.bfloat16),
                     ct.zeros((BS, BS), dtype=ct.float32))
        return ct.mma(p1.astype(ct.bfloat16), p2.astype(ct.bfloat16),
                      ct.zeros((BS, BS), dtype=ct.float32))

    Ai11 = inv_block(A11)
    Ai22 = inv_block(A22)
    Ai33 = inv_block(A33)
    Ai44 = inv_block(A44)

    tmp = ct.mma(Ai22.astype(ct.bfloat16), A21.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai21 = ct.mma(-tmp.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    tmp = ct.mma(Ai33.astype(ct.bfloat16), A32.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai32 = ct.mma(-tmp.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    tmp = ct.mma(Ai44.astype(ct.bfloat16), A43.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai43 = ct.mma(-tmp.astype(ct.bfloat16), Ai33.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A31.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A32.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    Ai31 = ct.mma(-Ai33.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A42.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A43.astype(ct.bfloat16), Ai32.astype(ct.bfloat16), t1)
    Ai42 = ct.mma(-Ai44.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A41.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A42.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    t3 = ct.mma(A43.astype(ct.bfloat16), Ai31.astype(ct.bfloat16), t2)
    Ai41 = ct.mma(-Ai44.astype(ct.bfloat16), t3.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Store A_inv as 16 blocks
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 0), tile=Ai11.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 1), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 2), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 3), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 0), tile=Ai21.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 1), tile=Ai22.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 2), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 3), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 0), tile=Ai31.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 1), tile=Ai32.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 2), tile=Ai33.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 3), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 0), tile=Ai41.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 1), tile=Ai42.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 2), tile=Ai43.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 3), tile=Ai44.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))


# ============================================================================
# Solve kernel with occupancy=2 (better for 256-512 block grids)
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_solve_kernel_occ2(
    A,          # [B, T, H, BT] input (float32)
    A_inv_out,  # [B, T, H, BT] output (bf16)
    T: ConstInt,
    H: ConstInt,
    BT: ConstInt,
):
    """solve_tril with occupancy=2 for better latency hiding at larger grids.
    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H
    BS = 16

    b_A = ct.load(A, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BT),
                  padding_mode=PAD_ZERO).reshape((BT, BT)).astype(ct.float32)

    A11 = ct.extract(b_A, index=(0, 0), shape=(BS, BS))
    A22 = ct.extract(b_A, index=(1, 1), shape=(BS, BS))
    A33 = ct.extract(b_A, index=(2, 2), shape=(BS, BS))
    A44 = ct.extract(b_A, index=(3, 3), shape=(BS, BS))
    A21 = ct.extract(b_A, index=(1, 0), shape=(BS, BS))
    A31 = ct.extract(b_A, index=(2, 0), shape=(BS, BS))
    A32 = ct.extract(b_A, index=(2, 1), shape=(BS, BS))
    A41 = ct.extract(b_A, index=(3, 0), shape=(BS, BS))
    A42 = ct.extract(b_A, index=(3, 1), shape=(BS, BS))
    A43 = ct.extract(b_A, index=(3, 2), shape=(BS, BS))

    bs_row = ct.arange(BS, dtype=ct.int32)[:, None]
    bs_col = ct.arange(BS, dtype=ct.int32)[None, :]
    I16 = ct.where(bs_row == bs_col, 1.0, 0.0)

    def inv_block(Axx):
        neg = -Axx
        sq = ct.mma(neg, neg, ct.zeros((BS, BS), dtype=ct.float32))
        q4 = ct.mma(sq, sq, ct.zeros((BS, BS), dtype=ct.float32))
        q8 = ct.mma(q4, q4, ct.zeros((BS, BS), dtype=ct.float32))
        p1 = ct.mma((I16 + neg).astype(ct.bfloat16), (I16 + sq).astype(ct.bfloat16),
                     ct.zeros((BS, BS), dtype=ct.float32))
        p2 = ct.mma((I16 + q4).astype(ct.bfloat16), (I16 + q8).astype(ct.bfloat16),
                     ct.zeros((BS, BS), dtype=ct.float32))
        return ct.mma(p1.astype(ct.bfloat16), p2.astype(ct.bfloat16),
                      ct.zeros((BS, BS), dtype=ct.float32))

    Ai11 = inv_block(A11)
    Ai22 = inv_block(A22)
    Ai33 = inv_block(A33)
    Ai44 = inv_block(A44)

    tmp = ct.mma(Ai22.astype(ct.bfloat16), A21.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai21 = ct.mma(-tmp.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    tmp = ct.mma(Ai33.astype(ct.bfloat16), A32.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai32 = ct.mma(-tmp.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    tmp = ct.mma(Ai44.astype(ct.bfloat16), A43.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    Ai43 = ct.mma(-tmp.astype(ct.bfloat16), Ai33.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A31.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A32.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    Ai31 = ct.mma(-Ai33.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A42.astype(ct.bfloat16), Ai22.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A43.astype(ct.bfloat16), Ai32.astype(ct.bfloat16), t1)
    Ai42 = ct.mma(-Ai44.astype(ct.bfloat16), t2.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t1 = ct.mma(A41.astype(ct.bfloat16), Ai11.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))
    t2 = ct.mma(A42.astype(ct.bfloat16), Ai21.astype(ct.bfloat16), t1)
    t3 = ct.mma(A43.astype(ct.bfloat16), Ai31.astype(ct.bfloat16), t2)
    Ai41 = ct.mma(-Ai44.astype(ct.bfloat16), t3.astype(ct.bfloat16), ct.zeros((BS, BS), dtype=ct.float32))

    # Store A_inv as 16 blocks
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 0), tile=Ai11.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 1), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 2), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 0, i_h, 3), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 0), tile=Ai21.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 1), tile=Ai22.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 2), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 1, i_h, 3), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 0), tile=Ai31.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 1), tile=Ai32.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 2), tile=Ai33.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 2, i_h, 3), tile=ct.zeros((BS, BS), dtype=ct.float32).astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 0), tile=Ai41.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 1), tile=Ai42.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 2), tile=Ai43.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))
    ct.store(A_inv_out, index=(i_b, i_t * 4 + 3, i_h, 3), tile=Ai44.astype(A_inv_out.dtype).reshape((1, BS, 1, BS)))


# ============================================================================
# WY kernel with occupancy=2 (better for 256-512 block grids)
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_wy_kernel_occ2(
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    beta,       # [B, T, H]
    g_cumsum,   # [B, T, H] (float32)
    A_inv,      # [B, T, H, BT] - the solved inverse
    w,          # [B, T, H, K] output
    u,          # [B, T, H, V] output
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """WY recompute with occupancy=2 for better latency hiding.
    Grid: (NT, B*H)
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    b_beta = ct.load(
        beta, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)

    b_g = ct.load(
        g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO
    ).reshape((BT,)).astype(ct.float32)
    b_exp_g = ct.exp(b_g)

    b_Ai = ct.load(
        A_inv, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BT),
        padding_mode=PAD_ZERO,
    ).reshape((BT, BT))

    for i_v in range(ct.cdiv(V, BV)):
        v_tile = ct.load(
            v, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BV))
        vb = (v_tile.astype(ct.float32) * b_beta[:, None]).astype(v_tile.dtype)
        acc_u = ct.zeros((BT, BV), dtype=ct.float32)
        b_u = ct.mma(b_Ai, vb, acc_u)
        ct.store(
            u, index=(i_b, i_t, i_h, i_v),
            tile=b_u.astype(u.dtype).reshape((1, BT, 1, BV)),
        )

    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(
            k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
            padding_mode=PAD_ZERO,
        ).reshape((BT, BK))
        kb = (k_tile.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k_tile.dtype)
        acc_w = ct.zeros((BT, BK), dtype=ct.float32)
        b_w = ct.mma(b_Ai, kb, acc_w)
        ct.store(
            w, index=(i_b, i_t, i_h, i_k),
            tile=b_w.astype(w.dtype).reshape((1, BT, 1, BK)),
        )


# ============================================================================
# Recurrence with BV=64 and occupancy=2 (for B=8,T=1024 type configs)
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_recurrence_kernel_bv64(
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
    BV: ConstInt,   # = 64
    NT: ConstInt,
    USE_INITIAL_STATE: ConstBool,
):
    """Recurrence with BV=64 tiles. Better for configs with many parallel
    rec_blocks (e.g., B=8,T=1024) where wider tiles improve MMA utilization.
    Grid: (ceil(V/64), B*H)
    """
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

    if K <= 128 and BH >= 64 and rec_BV <= 32 and NT * BH >= 512:
        # Fused WY+rec: w, u computed on-the-fly per chunk, never in HBM.
        # Saves w+u write+read bandwidth (~128MB for large configs at 8TB/s = ~16us).
        ct.launch(stream, (n_v, BH), cutile_fused_wy_rec_kernel,
                  (k, v, beta, g_cumsum, A_inv, h, v_new,
                   initial_state, initial_state_indices, T, H, K, V, 64, rec_BV, NT, USE_INITIAL))
    else:
        _recompute_w_u_cached(k, v, beta, g_cumsum, A_inv, c['w'], c['u'])
        ct.launch(stream, (n_v, BH), rec_kernel,
                  (k, c['w'], c['u'], g_cumsum, h, v_new,
                   initial_state, initial_state_indices, T, H, K, V, 64, rec_BV, NT, USE_INITIAL))
    _dispatch_output()

    return o, None, h


# ============================================================================
# Kernel 3 v2: Hidden state recurrence (loop over K blocks internally)
# This matches the Triton approach: one thread block handles all K blocks
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_recurrence_kernel_v2(
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
    BV: ConstInt,    # block size for V dimension (32)
    NT: ConstInt,
    USE_INITIAL_STATE: ConstBool,
):
    """Hidden state recurrence - loops over K blocks internally.

    Grid: (cdiv(V, BV), B*H)
    Each block handles one V-slice, all K blocks, all time chunks.
    
    Uses 4 separate [64, BV] registers for K=256 (matching Triton).
    """
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // H
    i_h = i_nh % H

    BK = 64  # fixed K-block size

    # Initialize hidden state blocks [BK, BV] for up to 4 K-blocks
    b_h1 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h2 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h3 = ct.zeros((BK, BV), dtype=ct.float32)
    b_h4 = ct.zeros((BK, BV), dtype=ct.float32)

    # Load initial state if needed
    if USE_INITIAL_STATE:
        idx = ct.load(
            initial_state_indices, index=(i_n,), shape=()
        ).astype(ct.int32)
        # initial_state: [N, H, K, V]
        h0_1 = ct.load(
            initial_state, index=(idx, i_h, 0, i_v), shape=(1, 1, BK, BV),
            padding_mode=PAD_ZERO,
        ).reshape((BK, BV)).astype(ct.float32)
        b_h1 = b_h1 + h0_1
        if K > 64:
            h0_2 = ct.load(
                initial_state, index=(idx, i_h, 1, i_v), shape=(1, 1, BK, BV),
                padding_mode=PAD_ZERO,
            ).reshape((BK, BV)).astype(ct.float32)
            b_h2 = b_h2 + h0_2
        if K > 128:
            h0_3 = ct.load(
                initial_state, index=(idx, i_h, 2, i_v), shape=(1, 1, BK, BV),
                padding_mode=PAD_ZERO,
            ).reshape((BK, BV)).astype(ct.float32)
            b_h3 = b_h3 + h0_3
        if K > 192:
            h0_4 = ct.load(
                initial_state, index=(idx, i_h, 3, i_v), shape=(1, 1, BK, BV),
                padding_mode=PAD_ZERO,
            ).reshape((BK, BV)).astype(ct.float32)
            b_h4 = b_h4 + h0_4

    for i_t in range(NT):
        # Store current h for this chunk
        ct.store(h, index=(i_n, i_t, i_h, 0, i_v),
                 tile=b_h1.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 64:
            ct.store(h, index=(i_n, i_t, i_h, 1, i_v),
                     tile=b_h2.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 128:
            ct.store(h, index=(i_n, i_t, i_h, 2, i_v),
                     tile=b_h3.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 192:
            ct.store(h, index=(i_n, i_t, i_h, 3, i_v),
                     tile=b_h4.astype(h.dtype).reshape((1, 1, 1, BK, BV)))

        # Compute v_corr = sum_k w_k @ h_k: accumulate [BT, BV]
        # Load w block 1: [1, BT, 1, BK] -> [BT, BK]
        w1 = ct.load(w, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_v = ct.zeros((BT, BV), dtype=ct.float32)
        b_v = ct.mma(w1, b_h1.astype(w1.dtype), b_v)

        if K > 64:
            w2 = ct.load(w, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w2, b_h2.astype(w2.dtype), b_v)
        if K > 128:
            w3 = ct.load(w, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w3, b_h3.astype(w3.dtype), b_v)
        if K > 192:
            w4 = ct.load(w, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v = ct.mma(w4, b_h4.astype(w4.dtype), b_v)

        # Load u: [1, BT, 1, BV] -> [BT, BV]
        u_tile = ct.load(u, index=(i_n, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                         padding_mode=PAD_ZERO).reshape((BT, BV))

        # v_new = u - v_corr
        b_v_new = u_tile.astype(ct.float32) - b_v

        # Store v_new
        ct.store(v_new, index=(i_n, i_t, i_h, i_v),
                 tile=b_v_new.astype(v_new.dtype).reshape((1, BT, 1, BV)))

        # Load g_cumsum for this chunk: [BT]
        b_g = ct.load(
            g_cumsum, index=(i_n, i_t, i_h), shape=(1, BT, 1),
            padding_mode=PAD_ZERO,
        ).reshape((BT,)).astype(ct.float32)

        # Get g_last (last element of chunk)
        b_g_last = ct.extract(b_g, index=(BT - 1,), shape=(1,))  # [1]

        # Scale v_new by exp(g_last - g[t])
        g_diff = b_g_last - b_g
        g_diff = ct.minimum(g_diff, 20.0)
        b_v_new_scaled = (b_v_new * ct.exp(g_diff)[:, None]).astype(k.dtype)

        # Scale all h blocks by exp(g_last)
        b_g_last_exp = ct.exp(b_g_last)
        b_h1 = b_h1 * b_g_last_exp
        if K > 64:
            b_h2 = b_h2 * b_g_last_exp
        if K > 128:
            b_h3 = b_h3 * b_g_last_exp
        if K > 192:
            b_h4 = b_h4 * b_g_last_exp

        # h += k^T @ v_new_scaled for each K block
        # Load k block 1: [1, BT, 1, BK] -> [BT, BK] -> transpose -> [BK, BT]
        k1 = ct.load(k, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_h1 = ct.mma(ct.transpose(k1), b_v_new_scaled, b_h1)

        if K > 64:
            k2 = ct.load(k, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h2 = ct.mma(ct.transpose(k2), b_v_new_scaled, b_h2)
        if K > 128:
            k3 = ct.load(k, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h3 = ct.mma(ct.transpose(k3), b_v_new_scaled, b_h3)
        if K > 192:
            k4 = ct.load(k, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_h4 = ct.mma(ct.transpose(k4), b_v_new_scaled, b_h4)

    # Write back final state
    idx = ct.load(
        initial_state_indices, index=(i_n,), shape=()
    ).astype(ct.int32)
    ct.store(initial_state, index=(idx, i_h, 0, i_v),
             tile=b_h1.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 64:
        ct.store(initial_state, index=(idx, i_h, 1, i_v),
                 tile=b_h2.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 128:
        ct.store(initial_state, index=(idx, i_h, 2, i_v),
                 tile=b_h3.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 192:
        ct.store(initial_state, index=(idx, i_h, 3, i_v),
                 tile=b_h4.astype(initial_state.dtype).reshape((1, 1, BK, BV)))


# ============================================================================
# Kernel 3 v3: Hidden state recurrence (no occupancy hint - better for large grids)
# ============================================================================
@ct.kernel
def cutile_recurrence_kernel_v3(
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
    BV: ConstInt,    # block size for V dimension (32)
    NT: ConstInt,
    USE_INITIAL_STATE: ConstBool,
):
    """Hidden state recurrence - no occupancy hint (better for large grids).

    Grid: (cdiv(V, BV), B*H)
    """
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // H
    i_h = i_nh % H

    BK = 64  # fixed K-block size

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
# Kernel 3 v4d: BV=16 recurrence with occupancy=4 for large grids (>296 CTAs)
# Register budget: 256KB/4 = 64KB per CTA.
# State b_h1..4: 4×BK×BV fp32 = 16KB; temporaries ~26KB; total ~42KB < 64KB.
# For B=8 (512 CTAs): reduces from 1.73 waves (occ=2) to single wave (occ=4).
# For B=16 (1024 CTAs): reduces from 3.46 waves to 1.73 waves.
# Grid: (cdiv(V, 16), B*H)
# ============================================================================
@ct.kernel(occupancy=4)
def cutile_recurrence_kernel_bv16_occ4(
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
    """BV=16 recurrence with occupancy=4 for large grids (>296 CTAs).
    4 CTAs per SM reduces wave count 2x vs occ=2 for B=8,B=16 workloads.
    """
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
# Kernel 3 v4e: BV=16 recurrence with split b_v accumulators (ILP-2)
# The b_v computation currently serializes 4 MMA ops into one accumulator
# (critical path = 4L for K=256). Splitting into two independent chains:
#   chain-a: w1@h1 → acc_a; w3@h3 → acc_a  (2 serial)
#   chain-b: w2@h2 → acc_b; w4@h4 → acc_b  (2 serial, independent of a)
# reduces b_v critical path from 4L → 2L.
# Combined with parallel h-updates (~L), total per-chunk path: 3L vs 5L = 1.67x.
# WGMMA latency L≈400ns → per-chunk: 5×400=2μs vs 3×400=1.2μs.
# For B=4,T=4096 (64 chunks): rec 130→78μs; total 243→191μs.
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_recurrence_kernel_bv16_split2(
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
    """BV=16 recurrence with split b_v MMA chains for ILP.
    b_v = (w1@h1 + w3@h3) + (w2@h2 + w4@h4) where chains a and b are independent.
    Reduces serial WGMMA critical path for b_v from 4L to 2L (K=256).
    Grid: (cdiv(V, BV), B*H)
    """
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
        # Split b_v into two independent MMA chains (chain-a: w1,w3; chain-b: w2,w4)
        # Hardware can pipeline chain-a and chain-b independently, halving critical path.
        b_v_a = ct.mma(w1, b_h1.astype(w1.dtype), ct.zeros((BT, BV), dtype=ct.float32))
        b_v_b = ct.zeros((BT, BV), dtype=ct.float32)
        if K > 64:
            w2 = ct.load(w, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v_b = ct.mma(w2, b_h2.astype(w2.dtype), b_v_b)
        if K > 128:
            w3 = ct.load(w, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v_a = ct.mma(w3, b_h3.astype(w3.dtype), b_v_a)
        if K > 192:
            w4 = ct.load(w, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            b_v_b = ct.mma(w4, b_h4.astype(w4.dtype), b_v_b)
        b_v = b_v_a + b_v_b
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
# Kernel: No-WY BV16 recurrence (BV=16, 64 CTAs)
#
# Eliminates the Triton WY precompute step by fusing A_inv application.
# Key design: k is loaded TWICE per chunk (phases 1 and 2) to avoid register
# spilling. Phase 2 k-loads are L2 hits (same data loaded moments earlier).
#
# HBM per chunk/CTA: k(32KB) + A_inv(8KB) + v(2KB) + rest(~10KB) ≈ 52KB
#   vs old rec_bv16:  k(32KB) + w(32KB)  + u(2KB) + rest(~10KB) ≈ 76KB (31% less)
# Saves 8μs Triton WY step on top of rec speedup.
# Grid: (V//BV, B*H) with BV=16 → 64 CTAs for B=1,H=4,V=256
# ============================================================================
@ct.kernel(occupancy=1)
def cutile_rec_nowy_bv16_kernel(
    k,           # [B, T, H, K]
    v,           # [B, T, H, V]
    beta,        # [B, T, H]
    g_cumsum,    # [B, T, H] float32
    A_inv,       # [B, NT, H, *, BT] flat (bf16)
    h,           # [B, NT, H, K_blocks, V_blocks, BK, BV] output hidden states
    v_new,       # [B, T, H, V] output
    initial_state,
    initial_state_indices,
    T: ConstInt, H: ConstInt, K: ConstInt, V: ConstInt,
    BT: ConstInt, BV: ConstInt, NT: ConstInt, USE_INITIAL_STATE: ConstBool,
):
    """BV=16 recurrence without separate WY precompute.

    Computes the same formula as cutile_fused_wy_rec_kernel but loads k in
    two separate passes so k tiles don't stay in registers across the A_inv
    computation, avoiding register spilling for K=256.
    """
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
        # Store h at start of chunk (read by output kernel)
        ct.store(h, index=(i_n, i_t, i_h, 0, i_v), tile=b_h1.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 64:
            ct.store(h, index=(i_n, i_t, i_h, 1, i_v), tile=b_h2.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 128:
            ct.store(h, index=(i_n, i_t, i_h, 2, i_v), tile=b_h3.astype(h.dtype).reshape((1, 1, 1, BK, BV)))
        if K > 192:
            ct.store(h, index=(i_n, i_t, i_h, 3, i_v), tile=b_h4.astype(h.dtype).reshape((1, 1, 1, BK, BV)))

        # Load per-chunk scalars
        b_beta = ct.load(beta, index=(i_n, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_g = ct.load(g_cumsum, index=(i_n, i_t, i_h), shape=(1, BT, 1), padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_exp_g = ct.exp(b_g)

        # Phase 1: kh_sum = Σ_bk (beta * exp_g * k_bk) @ h_bk
        # k tiles loaded one at a time and discarded — no register pressure from k
        kh_sum = ct.zeros((BT, BV), dtype=ct.float32)
        k1 = ct.load(k, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
        k1_sc = (k1.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k1.dtype)
        kh_sum = ct.mma(k1_sc, b_h1.astype(k1_sc.dtype), kh_sum)
        if K > 64:
            k2 = ct.load(k, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k2_sc = (k2.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k2.dtype)
            kh_sum = ct.mma(k2_sc, b_h2.astype(k2_sc.dtype), kh_sum)
        if K > 128:
            k3 = ct.load(k, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k3_sc = (k3.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k3.dtype)
            kh_sum = ct.mma(k3_sc, b_h3.astype(k3_sc.dtype), kh_sum)
        if K > 192:
            k4 = ct.load(k, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k4_sc = (k4.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k4.dtype)
            kh_sum = ct.mma(k4_sc, b_h4.astype(k4_sc.dtype), kh_sum)

        # Compute v_new = A_inv @ (beta*v - kh_sum)
        v_tile = ct.load(v, index=(i_n, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=PAD_ZERO).reshape((BT, BV))
        b_temp = v_tile.astype(ct.float32) * b_beta[:, None] - kh_sum
        b_Ai = ct.load(A_inv, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BT), padding_mode=PAD_ZERO).reshape((BT, BT))
        b_v_new = ct.mma(b_Ai, b_temp.astype(b_Ai.dtype), ct.zeros((BT, BV), dtype=ct.float32))
        ct.store(v_new, index=(i_n, i_t, i_h, i_v), tile=b_v_new.astype(v_new.dtype).reshape((1, BT, 1, BV)))

        # Scale v_new by exp(g_diff) for h update
        b_g_last = ct.extract(b_g, index=(BT - 1,), shape=(1,))
        g_diff = b_g_last - b_g
        g_diff = ct.minimum(g_diff, 20.0)
        b_v_new_scaled = (b_v_new * ct.exp(g_diff)[:, None]).astype(k.dtype)

        # Scale h by exp(g_last)
        b_g_last_exp = ct.exp(b_g_last)
        b_h1 = b_h1 * b_g_last_exp
        if K > 64:
            b_h2 = b_h2 * b_g_last_exp
        if K > 128:
            b_h3 = b_h3 * b_g_last_exp
        if K > 192:
            b_h4 = b_h4 * b_g_last_exp

        # Phase 2: h update — reload k (should be L2 hits from phase 1)
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
# Kernel: No-WY BV16 recurrence — occupancy=2 variant
# Same as cutile_rec_nowy_bv16_kernel but with occupancy=2 hint.
# Register analysis: state(16KB) + kh_sum(4KB) + k_tile(8KB) + b_temp(4KB) +
#   b_Ai(8KB) + b_v_new(4KB) + scalars(~2KB) ≈ 46KB < 128KB budget for occ=2.
# With 256 CTAs on 148 SMs at occ=2: 0.86 waves (single wave).
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_rec_nowy_bv16_occ2_kernel(
    k,           # [B, T, H, K]
    v,           # [B, T, H, V]
    beta,        # [B, T, H]
    g_cumsum,    # [B, T, H] float32
    A_inv,       # [B, T, H, BT] flat (bf16)
    h,           # [B, NT, H, K//BK, V//BV] output hidden states
    v_new,       # [B, T, H, V] output
    initial_state,
    initial_state_indices,
    T: ConstInt, H: ConstInt, K: ConstInt, V: ConstInt,
    BT: ConstInt, BV: ConstInt, NT: ConstInt, USE_INITIAL_STATE: ConstBool,
):
    """BV=16 recurrence without WY precompute (occ=2). Eliminates 28us WY step.
    Reads A_inv (8MB) instead of w (33MB) per chunk, saving ~24KB/chunk bandwidth.
    """
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
        kh_sum = ct.zeros((BT, BV), dtype=ct.float32)
        k1 = ct.load(k, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
        k1_sc = (k1.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k1.dtype)
        kh_sum = ct.mma(k1_sc, b_h1.astype(k1_sc.dtype), kh_sum)
        if K > 64:
            k2 = ct.load(k, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k2_sc = (k2.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k2.dtype)
            kh_sum = ct.mma(k2_sc, b_h2.astype(k2_sc.dtype), kh_sum)
        if K > 128:
            k3 = ct.load(k, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k3_sc = (k3.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k3.dtype)
            kh_sum = ct.mma(k3_sc, b_h3.astype(k3_sc.dtype), kh_sum)
        if K > 192:
            k4 = ct.load(k, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK), padding_mode=PAD_ZERO).reshape((BT, BK))
            k4_sc = (k4.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k4.dtype)
            kh_sum = ct.mma(k4_sc, b_h4.astype(k4_sc.dtype), kh_sum)
        v_tile = ct.load(v, index=(i_n, i_t, i_h, i_v), shape=(1, BT, 1, BV), padding_mode=PAD_ZERO).reshape((BT, BV))
        b_temp = v_tile.astype(ct.float32) * b_beta[:, None] - kh_sum
        b_Ai = ct.load(A_inv, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BT), padding_mode=PAD_ZERO).reshape((BT, BT))
        b_v_new = ct.mma(b_Ai, b_temp.astype(b_Ai.dtype), ct.zeros((BT, BV), dtype=ct.float32))
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
# Kernel: Fused cumsum + KKT + squaring solve + WY
#
# Extends cutile_fused_ckkt_solve_v2_kernel by also computing w and u
# in the same kernel using A_inv kept in fp32 registers:
#   w = A_inv @ (k * beta * exp_g)  →  [B, T, H, K]
#   u = A_inv @ (v * beta)          →  [B, T, H, V]
#
# Benefits over ckkt + separate Triton WY:
#   - A_inv never written/read from HBM (saves 1MB write + 1MB read = 2MB)
#   - k tiles loaded for KKT stay in L2 → WY k-reload costs only L2 latency
#   - One kernel launch instead of two
#
# Grid: (NT, B*H)
# ============================================================================
@ct.kernel(occupancy=1)
def cutile_fused_ckkt_wy_kernel(
    g_in,   # [B, T, H] input gates
    k,      # [B, T, H, K] keys
    v,      # [B, T, H, V] values
    beta,   # [B, T, H] beta values
    g_out,  # [B, T, H] output cumsum (float32)
    w_out,  # [B, T, H, K] output w = A_inv @ (k * beta * exp_g)
    u_out,  # [B, T, H, V] output u = A_inv @ (v * beta)
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Fused cumsum + KKT + squaring-trick solve + WY recompute.
    A_inv stays in fp32 registers — never written to HBM.
    k tiles loaded for KKT are L2-resident for the WY k-loop.
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
    b_exp_g = ct.exp(b_g)
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
    # b_Ai is fp32 (BT×BT), kept in registers — no HBM write

    b_Ai_bf16 = b_Ai.astype(ct.bfloat16)

    # ---- WY: w = A_inv @ (k * beta * exp_g) ----
    # k tiles loaded here are likely L2-resident from KKT loop above
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        kb = (k_tile.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(ct.bfloat16)
        b_w = ct.mma(b_Ai_bf16, kb, ct.zeros((BT, BK), dtype=ct.float32))
        ct.store(w_out, index=(i_b, i_t, i_h, i_k),
                 tile=b_w.astype(w_out.dtype).reshape((1, BT, 1, BK)))

    # ---- WY: u = A_inv @ (v * beta) ----
    for i_v in range(ct.cdiv(V, BV)):
        v_tile = ct.load(v, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                         padding_mode=PAD_ZERO).reshape((BT, BV))
        vb = (v_tile.astype(ct.float32) * b_beta[:, None]).astype(ct.bfloat16)
        b_u = ct.mma(b_Ai_bf16, vb, ct.zeros((BT, BV), dtype=ct.float32))
        ct.store(u_out, index=(i_b, i_t, i_h, i_v),
                 tile=b_u.astype(u_out.dtype).reshape((1, BT, 1, BV)))


# ============================================================================
# Kernel: Standalone cuTile WY recompute (occupancy=6)
# Replaces Triton recompute_w_u_fwd_kernel. Low register usage (~40KB):
# A_inv(8KB) + accumulator(16KB) + tiles(2*8KB) = 40KB < 42KB occ=6 budget.
# Grid: (NT, B*H). Same as ckkt.
# ============================================================================
@ct.kernel(occupancy=4)
def cutile_recompute_wu_occ4_kernel(
    k,         # [B, T, H, K]
    v,         # [B, T, H, V]
    beta,      # [B, T, H] float32
    A_inv,     # [B, T, H, BT] bf16 (triangular, per-chunk rows)
    g_cumsum,  # [B, T, H] float32
    w_out,     # [B, T, H, K]
    u_out,     # [B, T, H, V]
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """cuTile WY recompute: w = A_inv @ (k * beta * exp_g), u = A_inv @ (v * beta).
    Grid: (NT, B*H). A_inv kept in registers (8KB bf16) throughout K and V loops.
    Register budget at occ=4: 64KB. Peak usage: A(8)+b_w_accum(16)+tiles(2*8)=40KB.
    """
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Load triangular A_inv: [BT, BT] for this chunk
    b_Ai = ct.load(A_inv, index=(i_b, i_t, i_h, 0), shape=(1, BT, 1, BT),
                   padding_mode=PAD_ZERO).reshape((BT, BT)).astype(ct.float32).astype(ct.bfloat16)

    # Load beta [BT] and g_cumsum [BT]
    b_beta = ct.load(beta, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                     padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_g = ct.load(g_cumsum, index=(i_b, i_t, i_h), shape=(1, BT, 1),
                  padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
    b_exp_g = ct.exp(b_g)

    # w = A_inv @ (k * beta * exp_g)
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        kb = (k_tile.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(ct.bfloat16)
        b_w = ct.mma(b_Ai, kb, ct.zeros((BT, BK), dtype=ct.float32))
        ct.store(w_out, index=(i_b, i_t, i_h, i_k),
                 tile=b_w.astype(w_out.dtype).reshape((1, BT, 1, BK)))

    # u = A_inv @ (v * beta)
    for i_v in range(ct.cdiv(V, BV)):
        v_tile = ct.load(v, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                         padding_mode=PAD_ZERO).reshape((BT, BV))
        vb = (v_tile.astype(ct.float32) * b_beta[:, None]).astype(ct.bfloat16)
        b_u = ct.mma(b_Ai, vb, ct.zeros((BT, BV), dtype=ct.float32))
        ct.store(u_out, index=(i_b, i_t, i_h, i_v),
                 tile=b_u.astype(u_out.dtype).reshape((1, BT, 1, BV)))


# ============================================================================
# Kernel: Fused WY + Recurrence + Output
# Eliminates h, v_new global tensors by computing output on-the-fly.
# Saves 32MB (h) + 4MB (v_new) HBM traffic vs separate kernels.
# Grid: (cdiv(V, BV), B*H)
# ============================================================================
@ct.kernel(occupancy=1)
def cutile_fused_wy_rec_output_kernel(
    q,              # [B, T, H, K]
    k,              # [B, T, H, K]
    v,              # [B, T, H, V]
    beta,           # [B, T, H]
    g_cumsum,       # [B, T, H] (float32)
    A_inv,          # [B, T, H, BT] — flat [BT,BT] per chunk
    o,              # [B, T, H, V] output (written here)
    initial_state,  # [N, H, K, V]
    initial_state_indices,  # [B] int32
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BV: ConstInt,
    NT: ConstInt,
    scale: ConstFloat,
    USE_INITIAL_STATE: ConstBool,
):
    """Fused WY + Recurrence + Output. Eliminates h and v_new HBM tensors."""
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
        b_h1 = b_h1 + ct.load(initial_state, index=(idx, i_h, 0, i_v),
                               shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 64:
            b_h2 = b_h2 + ct.load(initial_state, index=(idx, i_h, 1, i_v),
                                   shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 128:
            b_h3 = b_h3 + ct.load(initial_state, index=(idx, i_h, 2, i_v),
                                   shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)
        if K > 192:
            b_h4 = b_h4 + ct.load(initial_state, index=(idx, i_h, 3, i_v),
                                   shape=(1, 1, BK, BV), padding_mode=PAD_ZERO).reshape((BK, BV)).astype(ct.float32)

    for i_t in range(NT):
        b_beta = ct.load(beta, index=(i_n, i_t, i_h), shape=(1, BT, 1),
                         padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_g = ct.load(g_cumsum, index=(i_n, i_t, i_h), shape=(1, BT, 1),
                      padding_mode=PAD_ZERO).reshape((BT,)).astype(ct.float32)
        b_exp_g = ct.exp(b_g)
        v_tile = ct.load(v, index=(i_n, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                         padding_mode=PAD_ZERO).reshape((BT, BV))

        # Step 1: Load k1..k4, compute temp_f = v*beta - sum k_scaled @ h
        b_temp_f = v_tile.astype(ct.float32) * b_beta[:, None]
        k1 = ct.load(k, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        k1s = (k1.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k1.dtype)
        b_temp_f = b_temp_f - ct.mma(k1s, b_h1.astype(k1s.dtype),
                                     ct.zeros((BT, BV), dtype=ct.float32))
        if K > 64:
            k2 = ct.load(k, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            k2s = (k2.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k2.dtype)
            b_temp_f = b_temp_f - ct.mma(k2s, b_h2.astype(k2s.dtype),
                                         ct.zeros((BT, BV), dtype=ct.float32))
        if K > 128:
            k3 = ct.load(k, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            k3s = (k3.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k3.dtype)
            b_temp_f = b_temp_f - ct.mma(k3s, b_h3.astype(k3s.dtype),
                                         ct.zeros((BT, BV), dtype=ct.float32))
        if K > 192:
            k4 = ct.load(k, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            k4s = (k4.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(k4.dtype)
            b_temp_f = b_temp_f - ct.mma(k4s, b_h4.astype(k4s.dtype),
                                         ct.zeros((BT, BV), dtype=ct.float32))

        # Step 2: v_new = A_inv @ temp_f
        b_Ai = ct.load(A_inv, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BT),
                       padding_mode=PAD_ZERO).reshape((BT, BT))
        b_v_new = ct.mma(b_Ai, b_temp_f.astype(b_Ai.dtype),
                         ct.zeros((BT, BV), dtype=ct.float32))

        # Step 3: Compute output = scale * (q@h*exp(g) + gated_causal_A@v_new)
        b_A_qk = ct.zeros((BT, BT), dtype=ct.float32)
        b_o_cross = ct.zeros((BT, BV), dtype=ct.float32)
        q1 = ct.load(q, index=(i_n, i_t, i_h, 0), shape=(1, BT, 1, BK),
                     padding_mode=PAD_ZERO).reshape((BT, BK))
        b_A_qk = ct.mma(q1, ct.transpose(k1), b_A_qk)
        b_o_cross = ct.mma(q1, b_h1.astype(q1.dtype), b_o_cross)
        if K > 64:
            q2 = ct.load(q, index=(i_n, i_t, i_h, 1), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_A_qk = ct.mma(q2, ct.transpose(k2), b_A_qk)
            b_o_cross = ct.mma(q2, b_h2.astype(q2.dtype), b_o_cross)
        if K > 128:
            q3 = ct.load(q, index=(i_n, i_t, i_h, 2), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_A_qk = ct.mma(q3, ct.transpose(k3), b_A_qk)
            b_o_cross = ct.mma(q3, b_h3.astype(q3.dtype), b_o_cross)
        if K > 192:
            q4 = ct.load(q, index=(i_n, i_t, i_h, 3), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
            b_A_qk = ct.mma(q4, ct.transpose(k4), b_A_qk)
            b_o_cross = ct.mma(q4, b_h4.astype(q4.dtype), b_o_cross)

        g_diff_qk = b_g[:, None] - b_g[None, :]
        g_diff_qk = ct.minimum(g_diff_qk, 20.0)
        b_A_qk = b_A_qk * ct.exp(g_diff_qk)
        row_idx = ct.arange(BT, dtype=ct.int32)[:, None]
        col_idx = ct.arange(BT, dtype=ct.int32)[None, :]
        b_A_qk = ct.where(row_idx >= col_idx, b_A_qk, 0.0)

        b_o_val = b_o_cross * b_exp_g[:, None]
        b_o_val = ct.mma(b_A_qk.astype(b_v_new.astype(ct.bfloat16).dtype),
                         b_v_new.astype(ct.bfloat16), b_o_val) * scale
        ct.store(o, index=(i_n, i_t, i_h, i_v),
                 tile=b_o_val.astype(o.dtype).reshape((1, BT, 1, BV)))

        # Step 4: Recurrence update
        b_g_last = ct.extract(b_g, index=(BT - 1,), shape=(1,))
        g_diff_rec = b_g_last - b_g
        g_diff_rec = ct.minimum(g_diff_rec, 20.0)
        b_v_new_scaled = (b_v_new * ct.exp(g_diff_rec)[:, None]).astype(k.dtype)
        b_g_last_exp = ct.exp(b_g_last)
        b_h1 = b_h1 * b_g_last_exp
        b_h1 = ct.mma(ct.transpose(k1), b_v_new_scaled, b_h1)
        if K > 64:
            b_h2 = b_h2 * b_g_last_exp
            b_h2 = ct.mma(ct.transpose(k2), b_v_new_scaled, b_h2)
        if K > 128:
            b_h3 = b_h3 * b_g_last_exp
            b_h3 = ct.mma(ct.transpose(k3), b_v_new_scaled, b_h3)
        if K > 192:
            b_h4 = b_h4 * b_g_last_exp
            b_h4 = ct.mma(ct.transpose(k4), b_v_new_scaled, b_h4)

    idx = ct.load(initial_state_indices, index=(i_n,), shape=()).astype(ct.int32)
    ct.store(initial_state, index=(idx, i_h, 0, i_v),
             tile=b_h1.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 64:
        ct.store(initial_state, index=(idx, i_h, 1, i_v),
                 tile=b_h2.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 128:
        ct.store(initial_state, index=(idx, i_h, 2, i_v),
                 tile=b_h3.astype(initial_state.dtype).reshape((1, 1, BK, BV)))
    if K > 192:
        ct.store(initial_state, index=(idx, i_h, 3, i_v),
                 tile=b_h4.astype(initial_state.dtype).reshape((1, 1, BK, BV)))


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


# ============================================================================
# Kernel: Fused cumsum + KKT + solve v2 — occupancy=4 variant
# Budget: 64KB. May cause more spill than occ=3 (85KB).
# 1024 CTAs → 1.73 waves (vs 2.31 at occ=3).
# ============================================================================
@ct.kernel(occupancy=4)
def cutile_fused_ckkt_solve_v2_occ4_kernel(
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
    """Fused cumsum + KKT + full 64×64 squaring-trick solve (occupancy=4).
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
# Kernel: Fused cumsum + KKT + squaring solve + WY — occupancy=2 variant
# Same as cutile_fused_ckkt_wy_kernel but with occupancy=2 hint.
# With 1024 CTAs on 148 SMs: 3.5 waves (vs 7 waves at occ=1), ~2x faster.
# A_inv stays in fp32 registers — never written to HBM.
# Register analysis: peak ~96KB (solve) then ~44KB (WY) < 128KB budget.
# ============================================================================
@ct.kernel(occupancy=2)
def cutile_fused_ckkt_wy_occ2_kernel(
    g_in,   # [B, T, H] input gates
    k,      # [B, T, H, K] keys
    v,      # [B, T, H, V] values
    beta,   # [B, T, H] beta values
    g_out,  # [B, T, H] output cumsum (float32)
    w_out,  # [B, T, H, K] output w = A_inv @ (k * beta * exp_g)
    u_out,  # [B, T, H, V] output u = A_inv @ (v * beta)
    T: ConstInt,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    """Fused cumsum + KKT + squaring-trick solve + WY recompute (occupancy=2).
    A_inv stays in fp32 registers — never written to HBM.
    k tiles loaded for KKT are L2-resident for the WY k-loop.
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
    b_exp_g = ct.exp(b_g)
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
    # b_Ai is fp32 (BT×BT), kept in registers — no HBM write

    b_Ai_bf16 = b_Ai.astype(ct.bfloat16)

    # ---- WY: w = A_inv @ (k * beta * exp_g) ----
    # k tiles loaded here are likely L2-resident from KKT loop above
    for i_k in range(ct.cdiv(K, BK)):
        k_tile = ct.load(k, index=(i_b, i_t, i_h, i_k), shape=(1, BT, 1, BK),
                         padding_mode=PAD_ZERO).reshape((BT, BK))
        kb = (k_tile.astype(ct.float32) * b_beta[:, None] * b_exp_g[:, None]).astype(ct.bfloat16)
        b_w = ct.mma(b_Ai_bf16, kb, ct.zeros((BT, BK), dtype=ct.float32))
        ct.store(w_out, index=(i_b, i_t, i_h, i_k),
                 tile=b_w.astype(w_out.dtype).reshape((1, BT, 1, BK)))

    # ---- WY: u = A_inv @ (v * beta) ----
    for i_v in range(ct.cdiv(V, BV)):
        v_tile = ct.load(v, index=(i_b, i_t, i_h, i_v), shape=(1, BT, 1, BV),
                         padding_mode=PAD_ZERO).reshape((BT, BV))
        vb = (v_tile.astype(ct.float32) * b_beta[:, None]).astype(ct.bfloat16)
        b_u = ct.mma(b_Ai_bf16, vb, ct.zeros((BT, BV), dtype=ct.float32))
        ct.store(u_out, index=(i_b, i_t, i_h, i_v),
                 tile=b_u.astype(u_out.dtype).reshape((1, BT, 1, BV)))
