"""
Triton GDN WY-parallel MTP spec-cycle kernel — fused.

================================================================================
PROBLEM IT SOLVES
================================================================================
Handles a full MTP speculation cycle in ONE Triton launch per call:
  Phase A: replay K accepted draft tokens to update state  h0 → h_K
  Phase B: compute outputs for T new draft tokens starting from h_K
  Final  : write h_K (NOT h_{K+T}) back to the state pool —
           draft-token updates aren't committed until accepted next cycle.

Invoked by `gated_delta_rule_mtp_auto` from `gdn_decode_wy_triton_mtp.py`.

================================================================================
INPUTS
================================================================================
Two disjoint token sets per call:

  accepted-token side (state replay only — no output compute):
    k_acc: [B, K_MAX, H,  K]   bf16
    v_acc: [B, K_MAX, HV, V]   bf16
    a_acc: [B, K_MAX, HV]      bf16
    b_acc: [B, K_MAX, HV]      bf16
    num_accepted: [B]          int32   (per-batch K, 0..K_MAX)

  new-token side (output compute):
    q_new: [B, T, H,  K]       bf16
    k_new: [B, T, H,  K]       bf16
    v_new: [B, T, HV, V]       bf16
    a_new: [B, T, HV]          bf16
    b_new: [B, T, HV]          bf16

  shared:
    A_log, dt_bias:        [HV]              fp32
    initial_state_source:  [pool, HV, V, K]  bf16  (written: h0 → h_K)
    initial_state_indices: [B]               int32

Outputs:
    output: [B, T, HV, V] bf16   (outputs for the T new tokens)
    state updated in place         (h_K; NOT h_{K+T})

================================================================================
K=0 BEHAVIOR
================================================================================
Per-batch variable K is masked via `offs_k < num_accepted[pid_b]`. For
K=0 the entire Phase A collapses to h_K = h0 automatically (β=0 ⇒
update=0; γ_K=0 ⇒ scale=1) — no separate code path needed.
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


@triton.jit
def _gdn_mtp_fused_kernel(
    # accepted-token tensors (state update)
    k_acc_ptr,
    v_acc_ptr,
    a_acc_ptr,
    b_acc_ptr,
    num_accepted_ptr,
    # new-token tensors (output)
    q_new_ptr,
    k_new_ptr,
    v_new_ptr,
    a_new_ptr,
    b_new_ptr,
    # shared
    a_log_ptr,
    dt_bias_ptr,
    h0_ptr,
    h0_idx_ptr,
    out_ptr,
    scale,
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
    # strides — new
    sqn_b,
    sqn_t,
    sqn_h,
    sqn_k,
    skn_b,
    skn_t,
    skn_h,
    skn_k,
    svn_b,
    svn_t,
    svn_hv,
    svn_v,
    san_b,
    san_t,
    san_hv,
    sbn_b,
    sbn_t,
    sbn_hv,
    # strides — shared
    sh_pool,
    sh_hv,
    sh_v,
    sh_k,
    so_b,
    so_t,
    so_hv,
    so_v,
    # sizes
    B: tl.constexpr,
    T: tl.constexpr,
    K_MAX: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    T_PAD: tl.constexpr,
    K_PAD: tl.constexpr,  # max(next_pow2(K_MAX), 16)
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
    num_acc = tl.load(num_accepted_ptr + pid_b).to(tl.int32)

    offs_k_vec = tl.arange(0, K)  # K (feature axis, i.e. 128)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_t = tl.arange(0, T_PAD)
    offs_kt = tl.arange(0, K_PAD)

    v_mask = offs_v < V
    t_mask = offs_t < T
    k_mask = offs_kt < num_acc  # dynamic per-batch

    # ================================================================
    # Load h0 (state before any accepted tokens)
    # ================================================================
    h0_ptrs = (
        h0_ptr
        + cache_idx * sh_pool
        + pid_hv * sh_hv
        + offs_v[:, None] * sh_v
        + offs_k_vec[None, :] * sh_k
    )
    H = tl.load(h0_ptrs, mask=v_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_V, K]

    # Per-head gate parameters
    A_log = tl.load(a_log_ptr + pid_hv).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)

    # ================================================================
    # PHASE A — WY-replay K accepted tokens  (h0 → h_K)
    # ================================================================
    # K_acc: [K_PAD, K]
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
        k_acc_norm = tl.sqrt(tl.sum(K_acc * K_acc, axis=1) + EPS)
        K_acc = K_acc / k_acc_norm[:, None]
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

    sp_a = _softplus(a_acc + dt_bias, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD)
    log_alpha_a = -tl.exp(A_log) * sp_a
    log_alpha_a = tl.where(k_mask, log_alpha_a, 0.0)
    gamma_a = tl.cumsum(log_alpha_a, axis=0)  # [K_PAD]
    exp_ga = tl.exp(gamma_a)
    beta_a = tl.sigmoid(b_acc)
    beta_a = tl.where(k_mask, beta_a, 0.0)

    # γ_K = γ_a[num_acc - 1]  (if num_acc==0 → γ_K = 0 → exp_gK = 1)
    last_mask_a = offs_kt == (num_acc - 1)
    gamma_K = tl.sum(tl.where(last_mask_a, gamma_a, 0.0), axis=0)
    exp_gK = tl.exp(gamma_K)
    exp_kdec_a = tl.exp(gamma_K - gamma_a)
    exp_kdec_a = tl.where(k_mask, exp_kdec_a, 0.0)

    row_a, col_a = offs_kt[:, None], offs_kt[None, :]
    diag_a = row_a == col_a
    strict_lower_a = row_a > col_a
    exp_gij_a = tl.exp(gamma_a[:, None] - gamma_a[None, :])

    KKT_a = tl.dot(K_acc.to(tl.bfloat16), tl.trans(K_acc).to(tl.bfloat16))
    L_a = tl.where(strict_lower_a, beta_a[:, None] * exp_gij_a * KKT_a, 0.0)
    negL_a = -L_a

    # Neumann series for T_matrix.  K_PAD - 1 iterations safely covers any
    # num_acc ≤ K_MAX ≤ K_PAD, since L_a has zero rows for masked positions.
    inv_a = tl.where(diag_a, 1.0, 0.0)
    powk_a = tl.where(diag_a, 1.0, 0.0)
    for _ in tl.static_range(K_PAD - 1):
        powk_a = tl.dot(negL_a.to(tl.bfloat16), powk_a.to(tl.bfloat16))
        inv_a = inv_a + powk_a
    Tmat_a = inv_a * beta_a[None, :]
    Tmat_a = tl.where(k_mask[:, None] & k_mask[None, :], Tmat_a, 0.0)

    eK_a = exp_ga[:, None] * K_acc
    W_a = tl.dot(Tmat_a.to(tl.bfloat16), eK_a.to(tl.bfloat16))  # [K_PAD, K]
    u_a = tl.dot(Tmat_a.to(tl.bfloat16), V_acc.to(tl.bfloat16))  # [K_PAD, BLOCK_V]

    WH_a = tl.dot(W_a.to(tl.bfloat16), tl.trans(H).to(tl.bfloat16))  # [K_PAD, BLOCK_V]
    new_v_a = u_a - WH_a  # [K_PAD, BLOCK_V]

    K_dec_a = exp_kdec_a[:, None] * K_acc  # [K_PAD, K]
    update_K = tl.dot(
        tl.trans(new_v_a).to(tl.bfloat16), K_dec_a.to(tl.bfloat16)
    )  # [BLOCK_V, K]

    # h_K = exp(γ_K) * h0 + update_K
    H = exp_gK * H + update_K  # reuse H register

    # ================================================================
    # PHASE B — WY over T new tokens using h_K as initial state
    # ================================================================
    # Q_new, K_new, V_new
    q_ptrs = (
        q_new_ptr
        + pid_b * sqn_b
        + i_h * sqn_h
        + offs_t[:, None] * sqn_t
        + offs_k_vec[None, :] * sqn_k
    )
    Q = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    k_ptrs = (
        k_new_ptr
        + pid_b * skn_b
        + i_h * skn_h
        + offs_t[:, None] * skn_t
        + offs_k_vec[None, :] * skn_k
    )
    Km = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    v_ptrs = (
        v_new_ptr
        + pid_b * svn_b
        + pid_hv * svn_hv
        + offs_t[:, None] * svn_t
        + offs_v[None, :] * svn_v
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

    a_new = tl.load(
        a_new_ptr + pid_b * san_b + offs_t * san_t + pid_hv * san_hv,
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)
    b_new = tl.load(
        b_new_ptr + pid_b * sbn_b + offs_t * sbn_t + pid_hv * sbn_hv,
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)

    sp_n = _softplus(a_new + dt_bias, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD)
    log_alpha_n = -tl.exp(A_log) * sp_n
    log_alpha_n = tl.where(t_mask, log_alpha_n, 0.0)
    gamma_n = tl.cumsum(log_alpha_n, axis=0)
    exp_gn = tl.exp(gamma_n)
    beta_n = tl.sigmoid(b_new)
    beta_n = tl.where(t_mask, beta_n, 0.0)

    row_n, col_n = offs_t[:, None], offs_t[None, :]
    diag_n = row_n == col_n
    strict_lower_n = row_n > col_n
    causal_n = row_n >= col_n
    exp_gij_n = tl.exp(gamma_n[:, None] - gamma_n[None, :])

    KKT_n = tl.dot(Km.to(tl.bfloat16), tl.trans(Km).to(tl.bfloat16))
    L_n = tl.where(strict_lower_n, beta_n[:, None] * exp_gij_n * KKT_n, 0.0)
    negL_n = -L_n
    inv_n = tl.where(diag_n, 1.0, 0.0)
    powk_n = tl.where(diag_n, 1.0, 0.0)
    for _ in tl.static_range(T - 1):
        powk_n = tl.dot(negL_n.to(tl.bfloat16), powk_n.to(tl.bfloat16))
        inv_n = inv_n + powk_n
    Tmat_n = inv_n * beta_n[None, :]
    Tmat_n = tl.where(t_mask[:, None] & t_mask[None, :], Tmat_n, 0.0)

    QKT_n = tl.dot(Q.to(tl.bfloat16), tl.trans(Km).to(tl.bfloat16))
    QKTm_n = tl.where(
        causal_n & t_mask[:, None] & t_mask[None, :], QKT_n * exp_gij_n, 0.0
    )

    eK_n = exp_gn[:, None] * Km
    W_n = tl.dot(Tmat_n.to(tl.bfloat16), eK_n.to(tl.bfloat16))
    u_n = tl.dot(Tmat_n.to(tl.bfloat16), Vm.to(tl.bfloat16))
    GQ_n = exp_gn[:, None] * Q

    # Phase-B MMAs against h_K (`H` register).
    WH_n = tl.dot(W_n.to(tl.bfloat16), tl.trans(H).to(tl.bfloat16))
    OI_n = tl.dot(GQ_n.to(tl.bfloat16), tl.trans(H).to(tl.bfloat16))

    new_v_n = u_n - WH_n
    o_intra = tl.dot(QKTm_n.to(tl.bfloat16), new_v_n.to(tl.bfloat16))
    out = OI_n + o_intra

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

    # ================================================================
    # Write final state = h_K (NOT h_{K+T})
    # ================================================================
    tl.store(h0_ptrs, H.to(h0_ptrs.dtype.element_ty), mask=v_mask[:, None])


def _select_block_v(V: int) -> int:
    if V <= 32:
        return V
    if V <= 64:
        return V
    return 64


def gated_delta_rule_mtp_fused(
    # accepted tokens
    k_accepted: torch.Tensor,
    v_accepted: torch.Tensor,
    a_accepted: torch.Tensor,
    b_accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    # new tokens
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    a_new: torch.Tensor,
    b_new: torch.Tensor,
    # shared
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
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Fused Triton MTP GDN: replay K accepted tokens → compute outputs for T new tokens.

    Saves h_K (state after K accepted tokens) back to the state tensor — NOT h_{K+T}.
    """
    assert q_new.shape == k_new.shape, "q_new and k_new must have same shape"
    B, T, H, K = q_new.shape
    HV = v_new.shape[2]
    V = v_new.shape[3]
    K_MAX = k_accepted.shape[1]
    pool_size = initial_state_source.shape[0]

    assert k_accepted.shape == (B, K_MAX, H, K)
    assert v_accepted.shape == (B, K_MAX, HV, V)
    assert a_accepted.shape == b_accepted.shape == (B, K_MAX, HV)
    assert q_new.shape == k_new.shape == (B, T, H, K)
    assert v_new.shape == (B, T, HV, V)
    assert a_new.shape == b_new.shape == (B, T, HV)
    assert num_accepted.shape == (B,) and num_accepted.dtype == torch.int32
    assert A_log.shape == (HV,) and dt_bias.shape == (HV,)
    assert initial_state_source.shape == (pool_size, HV, V, K)
    assert initial_state_source.dtype == torch.bfloat16
    assert 1 <= T <= 16 and 0 <= K_MAX <= 16
    assert HV % H == 0

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=q_new.device)
    if output is None:
        output = torch.empty(B, T, HV, V, device=q_new.device, dtype=q_new.dtype)

    BLOCK_V = block_v if block_v is not None else _select_block_v(V)
    T_PAD = max(triton.next_power_of_2(T), 16)
    K_PAD = max(triton.next_power_of_2(K_MAX), 16)

    grid = (triton.cdiv(V, BLOCK_V), HV, B)
    _gdn_mtp_fused_kernel[grid](
        k_accepted,
        v_accepted,
        a_accepted,
        b_accepted,
        num_accepted,
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
        K_MAX=K_MAX,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BLOCK_V=BLOCK_V,
        T_PAD=T_PAD,
        K_PAD=K_PAD,
        USE_L2NORM=use_qk_l2norm_in_kernel,
        SOFTPLUS_BETA=softplus_beta,
        SOFTPLUS_THRESHOLD=softplus_threshold,
        EPS=1e-6,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
