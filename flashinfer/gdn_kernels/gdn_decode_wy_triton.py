"""
Triton GDN WY-parallel MTP decode kernel — v2 (split output + state).

================================================================================
PROBLEM IT SOLVES
================================================================================
Single-pass T-tokens-from-state GDN MTP. Given initial state h0 and T input
tokens (q, k, v, gates), compute T outputs in one call and optionally write
the final state h_T back to the state pool.

Public API: `gated_delta_rule_mtp_wy_triton(...)`.

Two internal Triton kernels are used:
  * Output kernel — computes outputs. When state update is requested,
    also writes `new_v` to a scratch tensor for the state kernel.
  * State kernel — reads (K, gates, h0, scratch new_v), produces
    `h_new = exp(γ_T)·h0 + new_v^T @ K_dec`, writes back to h0.
    Uses BLOCK_V=32 (vs BLOCK_V=64 in the output kernel).
When `disable_state_update=True`, only the output kernel runs.

At very small batch the public function delegates to
`gdn_decode_wy_triton_v1.py` (fused single-kernel path).

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
    initial_state_source:   [pool, HV, V, K]  bf16   (in-place when state update)
    initial_state_indices:  [B]               int32
    output:                 [B, T, HV, V]     bf16

================================================================================
ALGORITHM
================================================================================
    KKT   = K @ K^T                                         # [T, T]
    IKK[i,j] = delta_ij + (i>j) * beta[i] * exp(gamma[i]-gamma[j]) * KKT[i,j]
    T_mat = inv(IKK) @ diag(beta)                           # [T, T]  via Neumann
    QKTm  = (Q @ K^T) * exp(gamma[i]-gamma[j]) * (i>=j)     # [T, T]
    W     = T_mat @ (exp(gamma) * K)                        # [T, K]
    u     = T_mat @ V                                       # [T, BLOCK_V]
    WH    = W  @ h0^T                                       # [T, BLOCK_V]
    OI    = (exp(gamma) * Q) @ h0^T                         # [T, BLOCK_V]
    new_v = u - WH                                          # [T, BLOCK_V]
    out   = OI + QKTm @ new_v                               # [T, BLOCK_V]
    # in a second kernel, if state_update:
    update = new_v^T @ (exp(gamma[T-1] - gamma) * K)        # [BLOCK_V, K]
    h_new  = exp(gamma[T-1]) * h0 + update                  # [BLOCK_V, K]

`inv(IKK)` is computed as a truncated Neumann series: L = IKK - I is
strictly lower triangular, so L^T = 0 and inv(I + L) = sum_{k=0}^{T-1} (-L)^k.
For T=8 that's 7 small T×T matmuls.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x, beta: tl.constexpr, threshold: tl.constexpr):
    # log(1 + exp(bx)) / beta, with a linear tail past `threshold` for
    # numerical stability.
    bx = beta * x
    sp = tl.log(1.0 + tl.exp(bx)) / beta
    return tl.where(bx <= threshold, sp, x)


# ============================================================================
# Output kernel — one CTA per (batch, hv, v-tile). When WRITE_NEW_V is true,
# also writes `new_v` to a scratch tensor for the state kernel.
# ============================================================================
@triton.jit
def _gdn_wy_output_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    h0_ptr,
    h0_idx_ptr,
    out_ptr,
    new_v_ptr,  # scratch [B, T, HV, V] bf16; unused when !WRITE_NEW_V
    scale,
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
    snv_b,
    snv_t,
    snv_hv,
    snv_v,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    T_PAD: tl.constexpr,
    WRITE_NEW_V: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_b = tl.program_id(2)

    i_h = pid_hv // (HV // H)
    cache_idx = tl.load(h0_idx_ptr + pid_b).to(tl.int64)

    offs_t = tl.arange(0, T_PAD)
    offs_k = tl.arange(0, K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    t_mask = offs_t < T
    v_mask = offs_v < V

    # --- Load Q, K, V ---
    q_ptrs = (
        q_ptr
        + pid_b * sq_b
        + i_h * sq_h
        + offs_t[:, None] * sq_t
        + offs_k[None, :] * sq_k
    )
    Q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    k_ptrs = (
        k_ptr
        + pid_b * sk_b
        + i_h * sk_h
        + offs_t[:, None] * sk_t
        + offs_k[None, :] * sk_k
    )
    Km = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

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

    if USE_L2NORM:
        q_norm = tl.sqrt(tl.sum(Q * Q, axis=1) + EPS)
        k_norm = tl.sqrt(tl.sum(Km * Km, axis=1) + EPS)
        Q = Q * (scale / q_norm)[:, None]
        Km = Km / k_norm[:, None]
    else:
        Q = Q * scale

    Q = tl.where(t_mask[:, None], Q, 0.0)
    Km = tl.where(t_mask[:, None], Km, 0.0)
    Vm = tl.where(t_mask[:, None], Vm, 0.0)

    # --- Gates: α, β, γ ---
    a_raw = tl.load(
        a_ptr + pid_b * sa_b + offs_t * sa_t + pid_hv * sa_hv, mask=t_mask, other=0.0
    ).to(tl.float32)
    b_raw = tl.load(
        b_ptr + pid_b * sb_b + offs_t * sb_t + pid_hv * sb_hv, mask=t_mask, other=0.0
    ).to(tl.float32)
    A_log = tl.load(a_log_ptr + pid_hv).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)

    sp = _softplus(a_raw + dt_bias, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD)
    log_alpha = -tl.exp(A_log) * sp
    log_alpha = tl.where(t_mask, log_alpha, 0.0)
    gamma = tl.cumsum(log_alpha, axis=0)
    exp_g = tl.exp(gamma)
    beta = tl.sigmoid(b_raw)
    beta = tl.where(t_mask, beta, 0.0)

    # --- WY control math ---
    KKT = tl.dot(Km.to(tl.bfloat16), tl.trans(Km).to(tl.bfloat16))
    row, col = offs_t[:, None], offs_t[None, :]
    strict_lower = row > col
    diag = row == col
    causal = row >= col
    exp_gij = tl.exp(gamma[:, None] - gamma[None, :])
    L = tl.where(strict_lower, beta[:, None] * exp_gij * KKT, 0.0)
    negL = -L

    inv = tl.where(diag, 1.0, 0.0)
    powk = tl.where(diag, 1.0, 0.0)
    for _ in tl.static_range(T - 1):
        powk = tl.dot(negL.to(tl.bfloat16), powk.to(tl.bfloat16))
        inv = inv + powk
    Tmat = inv * beta[None, :]
    Tmat = tl.where(t_mask[:, None] & t_mask[None, :], Tmat, 0.0)

    QKT = tl.dot(Q.to(tl.bfloat16), tl.trans(Km).to(tl.bfloat16))
    QKTm = tl.where(causal & t_mask[:, None] & t_mask[None, :], QKT * exp_gij, 0.0)

    eK = exp_g[:, None] * Km
    W = tl.dot(Tmat.to(tl.bfloat16), eK.to(tl.bfloat16))
    u = tl.dot(Tmat.to(tl.bfloat16), Vm.to(tl.bfloat16))
    GQ = exp_g[:, None] * Q

    # --- MMAs against h0 ---
    h0_ptrs = (
        h0_ptr
        + cache_idx * sh_pool
        + pid_hv * sh_hv
        + offs_v[:, None] * sh_v
        + offs_k[None, :] * sh_k
    )
    H0 = tl.load(h0_ptrs, mask=v_mask[:, None], other=0.0).to(tl.float32)

    WH = tl.dot(W.to(tl.bfloat16), tl.trans(H0).to(tl.bfloat16))
    OI = tl.dot(GQ.to(tl.bfloat16), tl.trans(H0).to(tl.bfloat16))

    # --- Output tail ---
    new_v = u - WH
    o_intra = tl.dot(QKTm.to(tl.bfloat16), new_v.to(tl.bfloat16))
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

    # --- Optional: write new_v scratch for the state kernel ---
    if WRITE_NEW_V:
        new_v_ptrs = (
            new_v_ptr
            + pid_b * snv_b
            + pid_hv * snv_hv
            + offs_t[:, None] * snv_t
            + offs_v[None, :] * snv_v
        )
        tl.store(
            new_v_ptrs,
            new_v.to(new_v_ptrs.dtype.element_ty),
            mask=t_mask[:, None] & v_mask[None, :],
        )


# ============================================================================
# State kernel — tiny focused kernel that does the final h_T update.
# Reads: K, gates, new_v scratch, h0.  Writes: h_new → h0.
# Uses a smaller BLOCK_V_STATE so register footprint allows 4-8 blocks/SM.
# ============================================================================
@triton.jit
def _gdn_wy_state_kernel(
    k_ptr,  # [B, T, H, K]   bf16
    a_ptr,
    b_ptr,  # [B, T, HV]     bf16
    a_log_ptr,  # [HV]           fp32
    dt_bias_ptr,  # [HV]           fp32
    h0_ptr,
    h0_idx_ptr,
    new_v_ptr,  # [B, T, HV, V]  bf16
    sk_b,
    sk_t,
    sk_h,
    sk_k,
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
    snv_b,
    snv_t,
    snv_hv,
    snv_v,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V_STATE: tl.constexpr,
    T_PAD: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_b = tl.program_id(2)

    i_h = pid_hv // (HV // H)
    cache_idx = tl.load(h0_idx_ptr + pid_b).to(tl.int64)

    offs_t = tl.arange(0, T_PAD)
    offs_k = tl.arange(0, K)
    offs_v = pid_v * BLOCK_V_STATE + tl.arange(0, BLOCK_V_STATE)
    t_mask = offs_t < T
    v_mask = offs_v < V

    # --- Gate scalars (recomputed: cheap, keeps this kernel stateless) ---
    a_raw = tl.load(
        a_ptr + pid_b * sa_b + offs_t * sa_t + pid_hv * sa_hv, mask=t_mask, other=0.0
    ).to(tl.float32)
    A_log = tl.load(a_log_ptr + pid_hv).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
    sp = _softplus(a_raw + dt_bias, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD)
    log_alpha = -tl.exp(A_log) * sp
    log_alpha = tl.where(t_mask, log_alpha, 0.0)
    gamma = tl.cumsum(log_alpha, axis=0)
    last_mask = offs_t == (T - 1)
    gamma_T = tl.sum(tl.where(last_mask, gamma, 0.0), axis=0)
    exp_gT = tl.exp(gamma_T)
    exp_kdec = tl.exp(gamma_T - gamma)
    exp_kdec = tl.where(t_mask, exp_kdec, 0.0)

    # --- Load K and re-normalize (same formula as output kernel) ---
    k_ptrs = (
        k_ptr
        + pid_b * sk_b
        + i_h * sk_h
        + offs_t[:, None] * sk_t
        + offs_k[None, :] * sk_k
    )
    Km = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    if USE_L2NORM:
        k_norm = tl.sqrt(tl.sum(Km * Km, axis=1) + EPS)
        Km = Km / k_norm[:, None]
    Km = tl.where(t_mask[:, None], Km, 0.0)

    # --- K_dec = exp(γ_T - γ)[:, None] * K_normalized ---
    K_dec = exp_kdec[:, None] * Km  # [T_PAD, K]

    # --- Load new_v from scratch ---
    new_v_ptrs = (
        new_v_ptr
        + pid_b * snv_b
        + pid_hv * snv_hv
        + offs_t[:, None] * snv_t
        + offs_v[None, :] * snv_v
    )
    new_v = tl.load(new_v_ptrs, mask=t_mask[:, None] & v_mask[None, :], other=0.0).to(
        tl.float32
    )  # [T_PAD, BLOCK_V_STATE]

    # --- update = new_v^T @ K_dec ---
    update = tl.dot(
        tl.trans(new_v).to(tl.bfloat16), K_dec.to(tl.bfloat16)
    )  # [BLOCK_V_STATE, K]

    # --- Load h0, compute h_new, store ---
    h0_ptrs = (
        h0_ptr
        + cache_idx * sh_pool
        + pid_hv * sh_hv
        + offs_v[:, None] * sh_v
        + offs_k[None, :] * sh_k
    )
    H0 = tl.load(h0_ptrs, mask=v_mask[:, None], other=0.0).to(tl.float32)
    h_new = exp_gT * H0 + update
    tl.store(h0_ptrs, h_new.to(h0_ptrs.dtype.element_ty), mask=v_mask[:, None])


def _select_block_v(V: int) -> int:
    # Output kernel: larger tile → better data reuse inside each CTA.
    if V <= 32:
        return V
    if V <= 64:
        return V
    return 64


def _select_block_v_state(V: int) -> int:
    if V <= 32:
        return V
    return 32


# Dispatch threshold for state update: below this many output-kernel CTAs, the
# extra launch cost of the state kernel outweighs its register-pressure savings.
# Empirical crossover on B200 (148 SMs, qwen3.5 HV=64, BLOCK_V=64) is at
# num_output_ctas ≈ 1500 (BS ≈ 12 for qwen3.5).  Use 10 waves as the threshold.
_STATE_SPLIT_MIN_CTAS = 10 * 148  # ~10 waves


def gated_delta_rule_mtp_wy_triton(
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
    block_v_state: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    num_warps_state: int = 4,
    num_stages_state: int = 2,
) -> torch.Tensor:
    """v2 Triton WY GDN MTP — split output / state kernels. Same public API."""
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

    BLOCK_V = block_v if block_v is not None else _select_block_v(V)
    T_PAD = max(triton.next_power_of_2(T), 16)

    # For state_update with tiny grids, falling back to the v1 fused kernel
    # beats the 2-kernel split (extra launch cost > register-pressure savings).
    num_output_ctas = triton.cdiv(V, BLOCK_V) * HV * B
    use_split = (not disable_state_update) and (
        num_output_ctas >= _STATE_SPLIT_MIN_CTAS
    )
    if (not disable_state_update) and (not use_split):
        # Delegate to the v1 fused path for small grids.
        from .gdn_decode_wy_triton_v1 import gated_delta_rule_mtp_wy_triton_v1

        return gated_delta_rule_mtp_wy_triton_v1(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            disable_state_update=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            output=output,
            block_v=block_v,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    # Allocate new_v scratch only if state update is enabled (zero-copy dummy otherwise).
    if disable_state_update:
        # Dummy scratch; pointer passed to kernel but never written.
        new_v_scratch = output  # any bf16 tensor with compatible strides suffices
        snv_strides = (
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
        )
    else:
        new_v_scratch = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)
        snv_strides = (
            new_v_scratch.stride(0),
            new_v_scratch.stride(1),
            new_v_scratch.stride(2),
            new_v_scratch.stride(3),
        )

    grid_out = (triton.cdiv(V, BLOCK_V), HV, B)
    _gdn_wy_output_kernel[grid_out](
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
        new_v_scratch,
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
        snv_strides[0],
        snv_strides[1],
        snv_strides[2],
        snv_strides[3],
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BLOCK_V=BLOCK_V,
        T_PAD=T_PAD,
        WRITE_NEW_V=(not disable_state_update),
        USE_L2NORM=use_qk_l2norm_in_kernel,
        SOFTPLUS_BETA=softplus_beta,
        SOFTPLUS_THRESHOLD=softplus_threshold,
        EPS=1e-6,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if not disable_state_update:
        BLOCK_V_STATE = (
            block_v_state if block_v_state is not None else _select_block_v_state(V)
        )
        grid_state = (triton.cdiv(V, BLOCK_V_STATE), HV, B)
        _gdn_wy_state_kernel[grid_state](
            k,
            a,
            b,
            A_log,
            dt_bias,
            initial_state_source,
            initial_state_indices,
            new_v_scratch,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
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
            new_v_scratch.stride(0),
            new_v_scratch.stride(1),
            new_v_scratch.stride(2),
            new_v_scratch.stride(3),
            B=B,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BLOCK_V_STATE=BLOCK_V_STATE,
            T_PAD=T_PAD,
            USE_L2NORM=use_qk_l2norm_in_kernel,
            SOFTPLUS_BETA=softplus_beta,
            SOFTPLUS_THRESHOLD=softplus_threshold,
            EPS=1e-6,
            num_warps=num_warps_state,
            num_stages=num_stages_state,
        )

    return output
