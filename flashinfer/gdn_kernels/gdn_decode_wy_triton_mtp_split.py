"""
Triton GDN WY-parallel MTP spec-cycle kernel — split + PDL.

================================================================================
PROBLEM IT SOLVES
================================================================================
Same public contract as the fused spec-cycle kernel in `_mtp_fused.py`:
  * Input A: K accepted draft tokens (per-batch variable via `num_accepted[B]`)
  * Input B: T new draft tokens
  * Output: T new-token outputs; state tensor written in place from h0 → h_K
            (NOT h_{K+T} — draft updates aren't committed yet).

Implemented as TWO Triton kernels launched back-to-back with Programmatic
Dependent Launch (PDL):

  Kernel A (state-step): reads h0 + K accepted tokens, computes and writes
    h_K back to the state pool. Skips Q loads, QKT, GQ, OI matmul, and
    output writes.
  Kernel B (output-only): reads h_K as initial state, writes outputs for
    the T new tokens. No state mutation here.

PDL lets Kernel B's prolog (load Q/K/V, compute gates, Phase-1 WY control
math) overlap Kernel A's tail; Kernel B's h_K-dependent MMAs wait for A's
h_K write via the PDL barrier (`gdc_wait()`).

Invoked by `gated_delta_rule_mtp_auto` from `gdn_decode_wy_triton_mtp.py`.

Public entry: `gated_delta_rule_mtp_split(...)`.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x, beta: tl.constexpr, threshold: tl.constexpr):
    bx = beta * x
    sp = tl.log(1.0 + tl.exp(bx)) / beta
    return tl.where(bx <= threshold, sp, x)


# ============================================================================
# Kernel A — state-step (no Q, no QKT, no output).
# Per-batch K via `num_accepted[pid_b]` mask.
# ============================================================================
@triton.jit
def _gdn_mtp_state_step_kernel(
    k_acc_ptr,
    v_acc_ptr,
    a_acc_ptr,
    b_acc_ptr,
    num_accepted_ptr,
    a_log_ptr,
    dt_bias_ptr,
    h0_ptr,
    h0_idx_ptr,
    # strides — accepted
    ska_b,
    ska_t,
    ska_h,
    ska_k,
    sva_b,
    sva_t,
    sva_hv,
    sva_v,
    saa_b,
    saa_t,
    saa_hv,
    sba_b,
    sba_t,
    sba_hv,
    # h0 strides
    sh_pool,
    sh_hv,
    sh_v,
    sh_k,
    # sizes
    B: tl.constexpr,
    K_MAX: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    K_PAD: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    EPS: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_b = tl.program_id(2)

    i_h = pid_hv // (HV // H)
    cache_idx = tl.load(h0_idx_ptr + pid_b).to(tl.int64)
    num_acc = tl.load(num_accepted_ptr + pid_b).to(tl.int32)

    offs_k_vec = tl.arange(0, K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_kt = tl.arange(0, K_PAD)
    v_mask = offs_v < V
    k_mask = offs_kt < num_acc

    A_log = tl.load(a_log_ptr + pid_hv).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)

    # Load K_acc, V_acc
    k_acc_ptrs = (
        k_acc_ptr
        + pid_b * ska_b
        + i_h * ska_h
        + offs_kt[:, None] * ska_t
        + offs_k_vec[None, :] * ska_k
    )
    K_acc = tl.load(k_acc_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

    v_acc_ptrs = (
        v_acc_ptr
        + pid_b * sva_b
        + pid_hv * sva_hv
        + offs_kt[:, None] * sva_t
        + offs_v[None, :] * sva_v
    )
    V_acc = tl.load(v_acc_ptrs, mask=k_mask[:, None] & v_mask[None, :], other=0.0).to(
        tl.float32
    )

    if USE_L2NORM:
        k_norm = tl.sqrt(tl.sum(K_acc * K_acc, axis=1) + EPS)
        K_acc = K_acc / k_norm[:, None]
    K_acc = tl.where(k_mask[:, None], K_acc, 0.0)
    V_acc = tl.where(k_mask[:, None], V_acc, 0.0)

    a_acc = tl.load(
        a_acc_ptr + pid_b * saa_b + offs_kt * saa_t + pid_hv * saa_hv,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    b_acc = tl.load(
        b_acc_ptr + pid_b * sba_b + offs_kt * sba_t + pid_hv * sba_hv,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    sp = _softplus(a_acc + dt_bias, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD)
    log_alpha = -tl.exp(A_log) * sp
    log_alpha = tl.where(k_mask, log_alpha, 0.0)
    gamma_a = tl.cumsum(log_alpha, axis=0)
    exp_ga = tl.exp(gamma_a)
    beta_a = tl.sigmoid(b_acc)
    beta_a = tl.where(k_mask, beta_a, 0.0)

    last_mask = offs_kt == (num_acc - 1)
    gamma_K = tl.sum(tl.where(last_mask, gamma_a, 0.0), axis=0)
    exp_gK = tl.exp(gamma_K)
    exp_kdec = tl.exp(gamma_K - gamma_a)
    exp_kdec = tl.where(k_mask, exp_kdec, 0.0)

    row, col = offs_kt[:, None], offs_kt[None, :]
    diag = row == col
    strict_lower = row > col
    exp_gij = tl.exp(gamma_a[:, None] - gamma_a[None, :])

    KKT = tl.dot(K_acc.to(tl.bfloat16), tl.trans(K_acc).to(tl.bfloat16))
    L = tl.where(strict_lower, beta_a[:, None] * exp_gij * KKT, 0.0)
    negL = -L
    inv = tl.where(diag, 1.0, 0.0)
    powk = tl.where(diag, 1.0, 0.0)
    for _ in tl.static_range(K_PAD - 1):
        powk = tl.dot(negL.to(tl.bfloat16), powk.to(tl.bfloat16))
        inv = inv + powk
    Tmat = inv * beta_a[None, :]
    Tmat = tl.where(k_mask[:, None] & k_mask[None, :], Tmat, 0.0)

    eK = exp_ga[:, None] * K_acc
    W = tl.dot(Tmat.to(tl.bfloat16), eK.to(tl.bfloat16))
    u = tl.dot(Tmat.to(tl.bfloat16), V_acc.to(tl.bfloat16))

    # Load h0, do WH = W @ h0^T
    h0_ptrs = (
        h0_ptr
        + cache_idx * sh_pool
        + pid_hv * sh_hv
        + offs_v[:, None] * sh_v
        + offs_k_vec[None, :] * sh_k
    )
    H0 = tl.load(h0_ptrs, mask=v_mask[:, None], other=0.0).to(tl.float32)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    WH = tl.dot(W.to(tl.bfloat16), tl.trans(H0).to(tl.bfloat16))
    new_v = u - WH
    K_dec = exp_kdec[:, None] * K_acc
    update = tl.dot(tl.trans(new_v).to(tl.bfloat16), K_dec.to(tl.bfloat16))
    h_new = exp_gK * H0 + update

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    tl.store(h0_ptrs, h_new.to(h0_ptrs.dtype.element_ty), mask=v_mask[:, None])


# ============================================================================
# Kernel B — output-only over T new tokens.
# Same as v2 output kernel with WRITE_NEW_V=False.
# ============================================================================
@triton.jit
def _gdn_mtp_output_kernel(
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
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    T_PAD: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    EPS: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
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

    row, col = offs_t[:, None], offs_t[None, :]
    diag = row == col
    strict_lower = row > col
    causal = row >= col
    exp_gij = tl.exp(gamma[:, None] - gamma[None, :])

    KKT = tl.dot(Km.to(tl.bfloat16), tl.trans(Km).to(tl.bfloat16))
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

    h0_ptrs = (
        h0_ptr
        + cache_idx * sh_pool
        + pid_hv * sh_hv
        + offs_v[:, None] * sh_v
        + offs_k[None, :] * sh_k
    )

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    H0 = tl.load(h0_ptrs, mask=v_mask[:, None], other=0.0).to(tl.float32)

    WH = tl.dot(W.to(tl.bfloat16), tl.trans(H0).to(tl.bfloat16))
    OI = tl.dot(GQ.to(tl.bfloat16), tl.trans(H0).to(tl.bfloat16))

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


def _select_block_v(V: int) -> int:
    if V <= 32:
        return V
    if V <= 64:
        return V
    return 64


def gated_delta_rule_mtp_split(
    k_accepted: torch.Tensor,
    v_accepted: torch.Tensor,
    a_accepted: torch.Tensor,
    b_accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    a_new: torch.Tensor,
    b_new: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: Optional[torch.Tensor] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    output: Optional[torch.Tensor] = None,
    block_v: Optional[int] = None,
    block_v_state: Optional[int] = None,
    launch_with_pdl: bool = True,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Two-kernel MTP with PDL: state-step then output-only."""
    B, T, H, K = q_new.shape
    HV = v_new.shape[2]
    V = v_new.shape[3]
    K_MAX = k_accepted.shape[1]

    assert k_accepted.shape == (B, K_MAX, H, K)
    assert v_accepted.shape == (B, K_MAX, HV, V)
    assert a_accepted.shape == b_accepted.shape == (B, K_MAX, HV)
    assert q_new.shape == k_new.shape == (B, T, H, K)
    assert v_new.shape == (B, T, HV, V)
    assert a_new.shape == b_new.shape == (B, T, HV)
    assert num_accepted.shape == (B,) and num_accepted.dtype == torch.int32
    assert initial_state_source.dtype == torch.bfloat16
    assert 1 <= T <= 16 and 0 <= K_MAX <= 16

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=q_new.device)
    if output is None:
        output = torch.empty(B, T, HV, V, device=q_new.device, dtype=q_new.dtype)

    BLOCK_V = block_v if block_v is not None else _select_block_v(V)
    BLOCK_V_STATE = (
        block_v_state if block_v_state is not None else (32 if V > 32 else V)
    )
    T_PAD = max(triton.next_power_of_2(T), 16)
    K_PAD = max(triton.next_power_of_2(K_MAX), 16)

    # ---- Kernel A: state-step ----
    # Always launch — per-batch masking (num_accepted[pid_b]) inside the
    # kernel handles K=0 batches (state is simply written back unchanged).
    # Avoid any host-side sync (no .item() calls in the hot path).
    grid_state = (triton.cdiv(V, BLOCK_V_STATE), HV, B)
    if True:
        _gdn_mtp_state_step_kernel[grid_state](
            k_accepted,
            v_accepted,
            a_accepted,
            b_accepted,
            num_accepted,
            A_log,
            dt_bias,
            initial_state_source,
            initial_state_indices,
            k_accepted.stride(0),
            k_accepted.stride(1),
            k_accepted.stride(2),
            k_accepted.stride(3),
            v_accepted.stride(0),
            v_accepted.stride(1),
            v_accepted.stride(2),
            v_accepted.stride(3),
            a_accepted.stride(0),
            a_accepted.stride(1),
            a_accepted.stride(2),
            b_accepted.stride(0),
            b_accepted.stride(1),
            b_accepted.stride(2),
            initial_state_source.stride(0),
            initial_state_source.stride(1),
            initial_state_source.stride(2),
            initial_state_source.stride(3),
            B=B,
            K_MAX=K_MAX,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BLOCK_V=BLOCK_V_STATE,
            K_PAD=K_PAD,
            USE_L2NORM=use_qk_l2norm_in_kernel,
            SOFTPLUS_BETA=softplus_beta,
            SOFTPLUS_THRESHOLD=softplus_threshold,
            EPS=1e-6,
            LAUNCH_WITH_PDL=False,  # first kernel; no predecessor to wait on
            num_warps=num_warps,
            num_stages=num_stages,
            launch_pdl=launch_with_pdl,  # allows our launch to predecessor-wait
        )

    # ---- Kernel B: output-only with PDL wait on Kernel A ----
    grid_out = (triton.cdiv(V, BLOCK_V), HV, B)
    _gdn_mtp_output_kernel[grid_out](
        q_new,
        k_new,
        v_new,
        a_new,
        b_new,
        A_log,
        dt_bias,
        initial_state_source,
        initial_state_indices,
        output,
        scale,
        q_new.stride(0),
        q_new.stride(1),
        q_new.stride(2),
        q_new.stride(3),
        k_new.stride(0),
        k_new.stride(1),
        k_new.stride(2),
        k_new.stride(3),
        v_new.stride(0),
        v_new.stride(1),
        v_new.stride(2),
        v_new.stride(3),
        a_new.stride(0),
        a_new.stride(1),
        a_new.stride(2),
        b_new.stride(0),
        b_new.stride(1),
        b_new.stride(2),
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
        USE_L2NORM=use_qk_l2norm_in_kernel,
        SOFTPLUS_BETA=softplus_beta,
        SOFTPLUS_THRESHOLD=softplus_threshold,
        EPS=1e-6,
        LAUNCH_WITH_PDL=launch_with_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=launch_with_pdl,
    )

    return output
