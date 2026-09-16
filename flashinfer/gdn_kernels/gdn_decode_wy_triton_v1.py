"""
Triton GDN WY-parallel MTP decode kernel — v1 (fused state+output).

================================================================================
PROBLEM IT SOLVES
================================================================================
Same functional contract as v2 in `gdn_decode_wy_triton.py`: single-pass
T-tokens-from-state GDN MTP. ONE Triton kernel computes outputs AND
(optionally) writes h_T back to the state pool in the same launch.

v2 imports v1 as a small-BS fallback when the output grid is smaller
than ~10 waves of CTAs (BS ≤ 11 on qwen3.5 HV=64). Deleting this file
breaks the v2 import.

================================================================================
INPUT / OUTPUT CONTRACT
================================================================================
    A_log:                  [HV]              fp32
    a:                      [B, T, HV]        bf16
    dt_bias:                [HV]              fp32
    q:                      [B, T, H,  K]     bf16
    k:                      [B, T, H,  K]     bf16
    v:                      [B, T, HV, V]     bf16
    b:                      [B, T, HV]        bf16
    initial_state_source:   [pool, HV, V, K]  bf16  (in-place when state update)
    initial_state_indices:  [B]               int32
    output:                 [B, T, HV, V]     bf16

================================================================================
ALGORITHM (per batch, hv_head, v_tile)
================================================================================
Follows GDN paper §3.3 (arxiv 2412.06464).

    KKT   = K @ K^T                                         # [T, T]
    IKK[i,j] = delta_ij + (i>j) * beta[i] * exp(gamma[i]-gamma[j]) * KKT[i,j]
    T_mat = inv(IKK) @ diag(beta)                           # [T, T]
    QKTm  = (Q @ K^T) * exp(gamma[i]-gamma[j]) * (i>=j)     # [T, T]
    W     = T_mat @ (exp(gamma) * K)                        # [T, K]
    u     = T_mat @ V                                       # [T, BLOCK_V]
    WH    = W  @ h0^T                                       # [T, BLOCK_V]
    OI    = (exp(gamma) * Q) @ h0^T                         # [T, BLOCK_V]
    new_v = u - WH                                          # [T, BLOCK_V]
    out   = OI + QKTm @ new_v                               # [T, BLOCK_V]
    # optional in-place state update:
    h_new = exp(gamma[T-1]) * h0 + (exp(gamma[T-1]-gamma) * K)^T @ new_v

inv(IKK) is computed as a truncated Neumann series: since L = IKK - I is
strictly lower triangular, L^T = 0, so inv(I + L) = sum_{k=0}^{T-1} (-L)^k.
For T=8 that's 7 small (T, T) matmuls.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x, beta: tl.constexpr, threshold: tl.constexpr):
    # Numerically-stable softplus with a linear tail past `threshold`.
    bx = beta * x
    sp = tl.log(1.0 + tl.exp(bx)) / beta
    return tl.where(bx <= threshold, sp, x)


@triton.jit
def _gdn_wy_kernel_v1(
    q_ptr,  # [B, T, H,  K]  bf16
    k_ptr,  # [B, T, H,  K]  bf16
    v_ptr,  # [B, T, HV, V]  bf16
    a_ptr,  # [B, T, HV]     bf16
    b_ptr,  # [B, T, HV]     bf16
    a_log_ptr,  # [HV]           fp32
    dt_bias_ptr,  # [HV]           fp32
    h0_ptr,  # [pool, HV, V, K] bf16
    h0_idx_ptr,  # [B]            int32
    out_ptr,  # [B, T, HV, V]  bf16
    # scale / gate constants
    scale,  # fp32 (runtime)
    # --- strides (in elements) ---
    sq_b,
    sq_t,
    sq_h,
    sq_k,
    sk_b,
    sk_t,
    sk_h,
    sk_k,
    sv_b,
    sv_t,
    sv_hv,
    sv_v,
    sa_b,
    sa_t,
    sa_hv,
    sb_b,
    sb_t,
    sb_hv,
    sh_pool,
    sh_hv,
    sh_v,
    sh_k,
    so_b,
    so_t,
    so_hv,
    so_v,
    # --- sizes ---
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    # --- tile / pad ---
    BLOCK_V: tl.constexpr,
    T_PAD: tl.constexpr,  # max(next_pow2(T), 16) — required for tl.dot
    # --- options ---
    DISABLE_STATE_UPDATE: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_b = tl.program_id(2)

    # GVA head mapping: each v-head picks a q/k head.
    i_h = pid_hv // (HV // H)

    # Cache slot for h0 (per-batch indirection).
    cache_idx = tl.load(h0_idx_ptr + pid_b).to(tl.int64)

    # ------------------------------------------------------------------
    # Index vectors
    # ------------------------------------------------------------------
    offs_t = tl.arange(0, T_PAD)  # [T_PAD]
    offs_k = tl.arange(0, K)  # [K]
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)  # [BLOCK_V]
    t_mask = offs_t < T  # [T_PAD]
    v_mask = offs_v < V  # [BLOCK_V]

    # ------------------------------------------------------------------
    # Phase 0 — Load Q, K, V, gate inputs; build alpha, beta, gamma.
    # ------------------------------------------------------------------
    # Q: [T_PAD, K]  (rows T..T_PAD-1 are zero → contribute nothing)
    q_ptrs = (
        q_ptr
        + pid_b * sq_b
        + i_h * sq_h
        + offs_t[:, None] * sq_t
        + offs_k[None, :] * sq_k
    )
    Q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    # K: [T_PAD, K]
    k_ptrs = (
        k_ptr
        + pid_b * sk_b
        + i_h * sk_h
        + offs_t[:, None] * sk_t
        + offs_k[None, :] * sk_k
    )
    Km = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    # V: [T_PAD, BLOCK_V]
    v_ptrs = (
        v_ptr
        + pid_b * sv_b
        + pid_hv * sv_hv
        + offs_t[:, None] * sv_t
        + offs_v[None, :] * sv_v
    )
    Vm = tl.load(v_ptrs, mask=t_mask[:, None] & v_mask[None, :], other=0.0).to(
        tl.float32
    )

    # L2-norm + scale on Q, K along the K axis. Padding rows are zero so
    # their norm is zero — guard with eps.
    if USE_L2NORM:
        q_norm = tl.sqrt(tl.sum(Q * Q, axis=1) + EPS)  # [T_PAD]
        k_norm = tl.sqrt(tl.sum(Km * Km, axis=1) + EPS)
        Q = Q * (scale / q_norm)[:, None]
        Km = Km / k_norm[:, None]
    else:
        Q = Q * scale

    # Zero out the padded rows again (norm path leaves them as 0/eps * 0 = 0,
    # but be explicit in case scale != 0).
    Q = tl.where(t_mask[:, None], Q, 0.0)
    Km = tl.where(t_mask[:, None], Km, 0.0)
    Vm = tl.where(t_mask[:, None], Vm, 0.0)

    # Gate scalars — per-token a[t] and b[t], per-head A_log and dt_bias.
    a_ptrs = a_ptr + pid_b * sa_b + offs_t * sa_t + pid_hv * sa_hv
    b_ptrs = b_ptr + pid_b * sb_b + offs_t * sb_t + pid_hv * sb_hv
    a_raw = tl.load(a_ptrs, mask=t_mask, other=0.0).to(tl.float32)
    b_raw = tl.load(b_ptrs, mask=t_mask, other=0.0).to(tl.float32)
    A_log = tl.load(a_log_ptr + pid_hv).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)

    sp = _softplus(a_raw + dt_bias, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD)  # [T_PAD]
    log_alpha = -tl.exp(A_log) * sp  # [T_PAD]
    # Padded rows must contribute log_alpha=0 (so alpha=1, zero decay increment).
    log_alpha = tl.where(t_mask, log_alpha, 0.0)
    gamma = tl.cumsum(log_alpha, axis=0)  # [T_PAD]
    exp_g = tl.exp(gamma)  # [T_PAD]
    beta = tl.sigmoid(b_raw)  # [T_PAD]
    beta = tl.where(t_mask, beta, 0.0)

    # gamma[T-1] — grab via mask-select so T is compile-time and we don't
    # index a dynamic position.
    last_mask = offs_t == (T - 1)
    gamma_T = tl.sum(tl.where(last_mask, gamma, 0.0), axis=0)  # scalar
    exp_gT = tl.exp(gamma_T)

    # ------------------------------------------------------------------
    # Phase 1 — WY control math (all small T×T matrices).
    # ------------------------------------------------------------------
    # KKT[i,j] = K[i,:] · K[j,:]
    KKT = tl.dot(Km.to(tl.bfloat16), tl.trans(Km).to(tl.bfloat16))  # [T_PAD, T_PAD]

    row = offs_t[:, None]
    col = offs_t[None, :]
    strict_lower = row > col
    diag = row == col
    causal = row >= col
    gmat = gamma[:, None] - gamma[None, :]
    exp_gij = tl.exp(gmat)

    # L = beta_i * exp(gamma_i - gamma_j) * KKT[i,j] on strict lower triangle,
    # 0 elsewhere. IKK = I + L. Want inv(I + L) = sum_{k=0}^{T-1} (-L)^k.
    L = tl.where(strict_lower, beta[:, None] * exp_gij * KKT, 0.0)
    negL = -L

    # Neumann series. L is zero outside the T×T top-left block (padded rows
    # have beta=0), so L^T = 0 and T-1 multiplies suffice.
    inv = tl.where(diag, 1.0, 0.0)  # [T_PAD, T_PAD]
    powk = tl.where(diag, 1.0, 0.0)
    for _ in tl.static_range(T - 1):
        powk = tl.dot(negL.to(tl.bfloat16), powk.to(tl.bfloat16))
        inv = inv + powk

    # T_matrix = inv(IKK) @ diag(beta). Diagonal scaling by column.
    Tmat = inv * beta[None, :]  # [T_PAD, T_PAD]
    # Valid rows/cols only (otherwise pad rows produce spurious outputs
    # when they happen to have nonzero negL entries from overflow).
    Tmat = tl.where(t_mask[:, None] & t_mask[None, :], Tmat, 0.0)

    # QKTm = (Q @ K^T) * exp(gamma_i - gamma_j), causal mask (i >= j).
    QKT = tl.dot(Q.to(tl.bfloat16), tl.trans(Km).to(tl.bfloat16))  # [T_PAD, T_PAD]
    QKTm = tl.where(causal & t_mask[:, None] & t_mask[None, :], QKT * exp_gij, 0.0)

    # W = T_matrix @ (exp(gamma) * K)  — shape [T_PAD, K]
    eK = exp_g[:, None] * Km
    W = tl.dot(Tmat.to(tl.bfloat16), eK.to(tl.bfloat16))

    # u = T_matrix @ V — shape [T_PAD, BLOCK_V]
    u = tl.dot(Tmat.to(tl.bfloat16), Vm.to(tl.bfloat16))

    # GQ = exp(gamma) * Q — shape [T_PAD, K]
    GQ = exp_g[:, None] * Q

    # ------------------------------------------------------------------
    # Phase 2 — the two K=128 matmuls against h0.
    # h0 layout is [pool, HV, V, K] → for a given (cache, hv) it's [V, K].
    # We need WH = W @ h0^T and OI = GQ @ h0^T, both [T_PAD, BLOCK_V].
    # Load h0 as [BLOCK_V, K] (bf16), reuse across both dots.
    # ------------------------------------------------------------------
    h0_ptrs = (
        h0_ptr
        + cache_idx * sh_pool
        + pid_hv * sh_hv
        + offs_v[:, None] * sh_v
        + offs_k[None, :] * sh_k
    )
    H0 = tl.load(h0_ptrs, mask=v_mask[:, None], other=0.0).to(
        tl.float32
    )  # [BLOCK_V, K]

    WH = tl.dot(W.to(tl.bfloat16), tl.trans(H0).to(tl.bfloat16))  # [T_PAD, BLOCK_V]
    OI = tl.dot(GQ.to(tl.bfloat16), tl.trans(H0).to(tl.bfloat16))  # [T_PAD, BLOCK_V]

    # ------------------------------------------------------------------
    # Phase 3 — output + optional state update.
    # ------------------------------------------------------------------
    new_v = u - WH  # [T_PAD, BLOCK_V]
    o_intra = tl.dot(QKTm.to(tl.bfloat16), new_v.to(tl.bfloat16))  # [T_PAD, BLOCK_V]
    out = OI + o_intra

    out_ptrs = (
        out_ptr
        + pid_b * so_b
        + pid_hv * so_hv
        + offs_t[:, None] * so_t
        + offs_v[None, :] * so_v
    )
    tl.store(
        out_ptrs,
        out.to(out_ptrs.dtype.element_ty),
        mask=t_mask[:, None] & v_mask[None, :],
    )

    if not DISABLE_STATE_UPDATE:
        # K_dec = exp(gamma[T-1] - gamma)[:, None] * K   — shape [T_PAD, K]
        exp_kdec = tl.exp(gamma_T - gamma)  # [T_PAD]
        exp_kdec = tl.where(t_mask, exp_kdec, 0.0)
        K_dec = exp_kdec[:, None] * Km  # [T_PAD, K]
        # h_new[v, k] = exp_gT * h0[v, k] + sum_t K_dec[t, k] * new_v[t, v]
        #            = exp_gT * h0 + new_v^T @ K_dec    (shape [BLOCK_V, K])
        update = tl.dot(
            tl.trans(new_v).to(tl.bfloat16), K_dec.to(tl.bfloat16)
        )  # [BLOCK_V, K]
        h_new = exp_gT * H0 + update
        tl.store(h0_ptrs, h_new.to(h0_ptrs.dtype.element_ty), mask=v_mask[:, None])


def _select_block_v_v1(V: int) -> int:
    # For K=V=128 heads a tile of 64 is a good default; 128 if the grid is huge.
    if V <= 32:
        return V
    if V <= 64:
        return V
    return 64


def gated_delta_rule_mtp_wy_triton_v1(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    block_v: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Triton WY-parallel GDN MTP. API-compatible with `gated_delta_rule_mtp_wy`."""
    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    pool_size = initial_state_source.shape[0]
    assert q.shape == k.shape == (B, T, H, K)
    assert v.shape == (B, T, HV, V)
    assert b.shape == a.shape == (B, T, HV)
    assert A_log.shape == (HV,) and dt_bias.shape == (HV,)
    assert initial_state_source.shape == (pool_size, HV, V, K)
    assert initial_state_source.dtype == torch.bfloat16
    assert 1 <= T <= 16, f"Only T in [1, 16] supported, got T={T}"
    assert HV % H == 0, "HV must be divisible by H"

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=q.device)
    if output is None:
        output = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)

    BLOCK_V = block_v if block_v is not None else _select_block_v_v1(V)
    T_PAD = max(triton.next_power_of_2(T), 16)

    grid = (triton.cdiv(V, BLOCK_V), HV, B)

    _gdn_wy_kernel_v1[grid](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state_source,
        initial_state_indices,
        output,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        initial_state_source.stride(0),
        initial_state_source.stride(1),
        initial_state_source.stride(2),
        initial_state_source.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BLOCK_V=BLOCK_V,
        T_PAD=T_PAD,
        DISABLE_STATE_UPDATE=disable_state_update,
        USE_L2NORM=use_qk_l2norm_in_kernel,
        SOFTPLUS_BETA=softplus_beta,
        SOFTPLUS_THRESHOLD=softplus_threshold,
        EPS=1e-6,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
