# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""The SM103 V-tile schedules of Cake KDA prefill against an FP64 recurrence.

On SM103 ``recurrent_kda(backend="cake")`` routes two shapes to the V-tile
M128 kernels: the direct one-wave body for a fixed-layout H96 batch of long
sequences, and the persistent body for packed equal-length H64/H96 sequences
whose task count fills four or six tasks per worker.  Each case checks that
the call launched a generated V-tile module and that its output and final
state match the token-by-token FP64 recurrence.
"""

import importlib

import pytest
import torch
import torch.nn.functional as F

from flashinfer.kda import recurrent_kda

kda_prefill_api = importlib.import_module("flashinfer.kda_prefill")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the V-tile schedules are SM103a only",
)

HEAD_DIM = 128
LOWER_BOUND = -5.0
VTILE_ROUTE = "vtile_m128"

CASES = [
    pytest.param((8192,), 96, True, 7101, id="direct_h96_1x8192"),
    pytest.param((2048,) * 8, 64, False, 7102, id="persistent_h64_8x2048"),
]


def _inputs(lengths, heads, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    count, tokens = len(lengths), lengths[0]

    def randn(*size, scale=1.0, shift=0.0):
        return torch.randn(size, generator=gen, device="cuda") * scale + shift

    shape = (count, tokens, heads, HEAD_DIM)
    return dict(
        q=randn(*shape).bfloat16(),
        k=randn(*shape).bfloat16(),
        v=randn(*shape).bfloat16(),
        g=randn(*shape, scale=0.5).bfloat16(),
        beta=randn(*shape[:3]).bfloat16(),
        A_log=randn(heads, scale=0.2),
        dt_bias=randn(heads, HEAD_DIM, scale=0.5, shift=-4.0),
        state=randn(count, heads, HEAD_DIM, HEAD_DIM, scale=0.05),
    )


def _fp64_reference(inp):
    """Token-by-token bounded-gate KDA recurrence in FP64, all sequences at once."""
    qn = F.normalize(inp["q"].float(), dim=-1).bfloat16().double()
    kn = F.normalize(inp["k"].float(), dim=-1).bfloat16().double()
    rate = inp["A_log"].double().exp().reshape(1, 1, -1, 1)
    biased = inp["g"].double() + inp["dt_bias"].double()
    decay = (LOWER_BOUND * torch.sigmoid(rate * biased)).exp()
    active = inp["beta"].float().sigmoid().double()
    v = inp["v"].double()
    state = inp["state"].double().clone()
    out = torch.empty(qn.shape, dtype=torch.float64, device=qn.device)
    for t in range(qn.shape[1]):
        state = state * decay[:, t, :, None, :]
        residual = (v[:, t] - (state * kn[:, t, :, None, :]).sum(-1)) * active[
            :, t, :, None
        ]
        state = state + residual[..., None] * kn[:, t, :, None, :]
        out[:, t] = (state * qn[:, t, :, None, :]).sum(-1) * HEAD_DIM**-0.5
    return out, state


def _check(name, actual, reference, tol=0.01):
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"
    rel = (actual.double() - reference).norm() / reference.norm()
    assert rel < tol, f"{name}: rel L2 {rel:.4g} >= {tol}"


@pytest.mark.parametrize("state", ["bf16", "fp32_pool"])
@pytest.mark.parametrize(("lengths", "heads", "fixed", "seed"), CASES)
def test_vtile_prefill_matches_fp64_recurrence(
    lengths, heads, fixed, seed, state, monkeypatch
):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    route = kda_prefill_api.select_bf16_schedule_route(
        compute_capability=(10, 3),
        sm_count=sm_count,
        fixed_layout=fixed,
        sequence_lengths=lengths,
        num_heads=heads,
        state_dtype_is_fp32=state == "fp32_pool",
    )
    assert route == VTILE_ROUTE, route
    inp = _inputs(lengths, heads, seed)
    expected_out, expected_final = _fp64_reference(inp)

    launched = []
    run_generated = kda_prefill_api._run_generated_single_route

    def record(**kwargs):
        ok = run_generated(**kwargs)
        launched.append((kwargs["route"], bool(ok)))
        return ok

    monkeypatch.setattr(kda_prefill_api, "_run_generated_single_route", record)

    n = len(lengths)
    operands = {name: inp[name] for name in ("q", "k", "v", "g", "beta")}
    cu_seqlens = None
    if not fixed:
        # the packed layout is [1, total tokens, heads, dim] with cu_seqlens
        operands = {
            name: t.reshape(1, -1, *t.shape[2:]) for name, t in operands.items()
        }
        cu_seqlens = torch.arange(
            0, n * lengths[0] + 1, lengths[0], device="cuda", dtype=torch.int64
        )
    kwargs = {}
    if state == "bf16":
        initial = inp["state"].bfloat16()
    else:
        initial = torch.zeros(
            (n + 2, heads, HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.float32
        )
        indices = torch.arange(n, device="cuda", dtype=torch.int32) + 1
        initial[indices.long()] = inp["state"]
        kwargs["ssm_state_indices"] = indices
    out, final_state = recurrent_kda(
        **operands,
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        beta_is_logit=True,
        backend="cake",
        **kwargs,
    )
    torch.cuda.synchronize()
    assert (VTILE_ROUTE, True) in launched, launched
    if state == "fp32_pool":
        final_state = initial[kwargs["ssm_state_indices"].long()]
    _check("output", out.reshape(expected_out.shape), expected_out)
    _check("final_state", final_state, expected_final)
