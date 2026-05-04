"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

KDA chunk-prefill correctness tests against a vendored PyTorch reference.
"""

from __future__ import annotations

import math

import pytest
import torch

from flashinfer.kda import chunk_kda_fwd, prepare_chunk_indices

from .conftest import skip_if_not_sm100
from .reference_kda import reference_chunk_kda_fwd


# Tolerances tuned against bf16 accumulation in the optimized kernels.
# softplus mode produces wider state dynamic range and hits the bf16 floor;
# safe_gate is tighter because the sigmoid bounds the gate.
ATOL_SOFTPLUS, RTOL_SOFTPLUS = 1.5e-2, 1.5e-2
ATOL_SAFE_GATE, RTOL_SAFE_GATE = 1e-2, 1e-2
ATOL_VARLEN_SOFTPLUS, RTOL_VARLEN_SOFTPLUS = 1e-1, 1e-1
ATOL_FINAL_STATE, RTOL_FINAL_STATE = 1e-1, 1e-1


def _l2_normalize_last(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2.0, dim=-1)


def _make_inputs(B, T, H, K, *, seed=0xC0FFEE, device="cuda"):
    torch.manual_seed(seed)
    q = _l2_normalize_last(torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device))
    k = _l2_normalize_last(torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device))
    v = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    g = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.5
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=device) * 0.1
    h0 = torch.randn(B, H, K, K, dtype=torch.float32, device=device) * 0.01
    return q, k, v, g, beta, A_log, dt_bias, h0, K ** -0.5


@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize("with_dt_bias", [False, True])
@pytest.mark.parametrize("T", [256, 512])
def test_eqlen_chunk_kda(safe_gate: bool, with_dt_bias: bool, T: int) -> None:
    skip_if_not_sm100()

    B, H, K = 1, 4, 128
    q, k, v, g, beta, A_log, dt_bias, h0, scale = _make_inputs(B, T, H, K)
    bias = dt_bias if with_dt_bias else None
    lower_bound = -5.0 if safe_gate else None

    o_opt, s_opt = chunk_kda_fwd(
        q, k, v, g, beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        A_log=A_log,
        dt_bias=bias,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    o_ref, s_ref = reference_chunk_kda_fwd(
        q, k, v, g, beta,
        scale=scale,
        A_log=A_log,
        dt_bias=bias,
        initial_state=h0,
        output_final_state=True,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    atol, rtol = (ATOL_SAFE_GATE, RTOL_SAFE_GATE) if safe_gate else (ATOL_SOFTPLUS, RTOL_SOFTPLUS)
    torch.testing.assert_close(o_opt.float(), o_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        s_opt.float(), s_ref.float(), atol=ATOL_FINAL_STATE, rtol=RTOL_FINAL_STATE
    )


@pytest.mark.parametrize("safe_gate", [False, True])
def test_eqlen_partial_chunk(safe_gate: bool) -> None:
    """T not a multiple of 64 — host-pad path. B=1 only."""
    skip_if_not_sm100()

    B, T, H, K = 1, 320 + 31, 4, 128  # 351, not divisible by 64
    q, k, v, g, beta, A_log, dt_bias, h0, scale = _make_inputs(B, T, H, K)

    o_opt, s_opt = chunk_kda_fwd(
        q, k, v, g, beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        A_log=A_log,
        dt_bias=dt_bias,
        safe_gate=safe_gate,
    )

    # Reference also expects T-padding to chunk_size; emulate by padding to BT.
    BT = 64
    Tpad = ((T + BT - 1) // BT) * BT
    pad_zeros = lambda x, sz: torch.cat([x, torch.zeros(B, sz, *x.shape[2:], dtype=x.dtype, device=x.device)], dim=1)
    qp = pad_zeros(q, Tpad - T); kp = pad_zeros(k, Tpad - T)
    vp = pad_zeros(v, Tpad - T); gp_zero = pad_zeros(g, Tpad - T)
    # g tail must saturate to zero gate -> use -1e3 sentinel.
    gp = gp_zero.clone()
    if Tpad > T:
        gp[:, T:] = -1000.0
    bp = pad_zeros(beta, Tpad - T)

    o_ref_pad, s_ref = reference_chunk_kda_fwd(
        qp, kp, vp, gp, bp,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        safe_gate=safe_gate,
    )
    o_ref = o_ref_pad[:, :T]

    atol, rtol = (ATOL_SAFE_GATE, RTOL_SAFE_GATE) if safe_gate else (ATOL_SOFTPLUS, RTOL_SOFTPLUS)
    torch.testing.assert_close(o_opt.float(), o_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        s_opt.float(), s_ref.float(), atol=ATOL_FINAL_STATE, rtol=RTOL_FINAL_STATE
    )


@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [256],          # single seq, aligned
        [256, 256],     # two seqs, aligned
        [320, 192],     # two seqs, mixed but aligned
    ],
)
def test_varlen_chunk_kda(safe_gate: bool, seq_lens: list[int]) -> None:
    """Varlen with all seqs 64-aligned (covers the VARLEN_PURE fast path)."""
    skip_if_not_sm100()

    H, K = 4, 128
    T_total = sum(seq_lens)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device="cuda")

    q, k, v, g, beta, A_log, dt_bias, _, scale = _make_inputs(1, T_total, H, K)
    h0 = torch.zeros(len(seq_lens), H, K, K, dtype=torch.float32, device="cuda")

    o_opt, s_opt = chunk_kda_fwd(
        q, k, v, g, beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=prepare_chunk_indices(cu_seqlens, 64),
        A_log=A_log,
        dt_bias=dt_bias,
        safe_gate=safe_gate,
    )

    o_ref, s_ref = reference_chunk_kda_fwd(
        q, k, v, g, beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
    )

    atol, rtol = (ATOL_SAFE_GATE, RTOL_SAFE_GATE) if safe_gate else (ATOL_SOFTPLUS, RTOL_SOFTPLUS)
    torch.testing.assert_close(o_opt.float(), o_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        s_opt.float(), s_ref.float(), atol=ATOL_FINAL_STATE, rtol=RTOL_FINAL_STATE
    )


def test_varlen_non_aligned_single_seq() -> None:
    """Single varlen seq with length not divisible by 64 — Phase 2.1 sentinel
    path (caller pre-pads q/k/v/beta; we sentinel-pad g and run varlen-pure)."""
    skip_if_not_sm100()

    H, K, BT = 4, 128, 64
    seq_len = 511
    Tpad = ((seq_len + BT - 1) // BT) * BT
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int64, device="cuda")

    q, k, v, g, beta, A_log, dt_bias, _, scale = _make_inputs(1, Tpad, H, K)
    # Caller convention: zero-pad q/k/v/beta to BT-multiple. g tail is also
    # zero (kernel will sentinel-pad internally).
    q[:, seq_len:] = 0
    k[:, seq_len:] = 0
    v[:, seq_len:] = 0
    beta[:, seq_len:] = 0
    h0 = torch.zeros(1, H, K, K, dtype=torch.float32, device="cuda")

    o_opt, s_opt = chunk_kda_fwd(
        q, k, v, g, beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        safe_gate=False,
    )

    # Reference handles varlen via per-seq slice; truncate output before
    # comparison to drop the padded tail.
    o_ref, s_ref = reference_chunk_kda_fwd(
        q, k, v, g, beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        safe_gate=False,
    )

    torch.testing.assert_close(
        o_opt[:, :seq_len].float(), o_ref[:, :seq_len].float(),
        atol=ATOL_VARLEN_SOFTPLUS, rtol=RTOL_VARLEN_SOFTPLUS,
    )
    torch.testing.assert_close(
        s_opt.float(), s_ref.float(),
        atol=ATOL_FINAL_STATE, rtol=RTOL_FINAL_STATE,
    )
