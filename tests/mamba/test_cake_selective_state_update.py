# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness coverage for the Cake selective-state-update backend."""

from __future__ import annotations

import pytest
import torch

from flashinfer.jit.core import MissingJITCacheError
from flashinfer.jit.mamba import cake_selective_state_update as cake
from flashinfer.mamba import cake_selective_state_update, selective_state_update

_CASES = (
    ("stp_ratio1", 32, 16, 16, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio8", 64, 64, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio8_sat", 128, 64, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio16", 32, 128, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_persistent", 257, 128, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_fp32", 64, 64, 8, 128, 128, 0, torch.float32, "auto", False, None),
    ("mtp_short1", 64, 64, 8, 128, 128, 1, torch.bfloat16, "auto", False, None),
    ("mtp_short2", 64, 64, 8, 128, 128, 2, torch.bfloat16, "auto", False, None),
    ("mtp_short2_h32", 64, 32, 8, 128, 128, 2, torch.bfloat16, "auto", False, None),
    ("mtp_short1_b16", 16, 64, 8, 128, 128, 1, torch.bfloat16, "auto", False, None),
    ("mtp_short2_b16", 16, 64, 8, 128, 128, 2, torch.bfloat16, "auto", False, None),
    ("mtp_short2_b256", 256, 64, 8, 128, 128, 2, torch.bfloat16, "auto", False, None),
    ("mtp_cache", 1, 64, 8, 64, 128, 6, torch.bfloat16, "vertical", True, None),
    ("mtp_cache_g1", 1, 64, 1, 64, 128, 6, torch.bfloat16, "vertical", True, None),
    ("mtp_horizontal", 32, 64, 8, 64, 128, 6, torch.bfloat16, "horizontal", True, None),
    (
        "mtp_horizontal_b64",
        64,
        64,
        8,
        64,
        128,
        6,
        torch.bfloat16,
        "horizontal",
        True,
        None,
    ),
    ("dynamic0", 1, 16, 1, 64, 128, 1, torch.float32, "simple", False, 0),
    ("dynamic0_b8", 8, 16, 1, 64, 128, 2, torch.float32, "simple", False, 0),
    ("dynamic1", 1, 16, 1, 64, 128, 4, torch.float32, "simple", False, 1),
    ("dynamic3", 1, 16, 1, 64, 128, 8, torch.float32, "simple", False, 3),
    ("dynamic7", 1, 16, 1, 64, 128, 8, torch.float32, "simple", False, 7),
    # Headdim-64 single-token decode (Nemotron-H / granite-4.0-h): row-owner tiles and paired batch programs.
    ("stp_hd64_rows_bf16", 4, 128, 8, 64, 128, 0, torch.bfloat16, "auto", False, None),
    (
        "stp_hd64_rows_bf16_b1",
        1,
        128,
        8,
        64,
        128,
        0,
        torch.bfloat16,
        "auto",
        False,
        None,
    ),
    (
        "stp_hd64_paired_rows",
        64,
        128,
        8,
        64,
        128,
        0,
        torch.bfloat16,
        "auto",
        False,
        None,
    ),
    (
        "stp_hd64_paired_rows_b320",
        320,
        128,
        8,
        64,
        128,
        0,
        torch.bfloat16,
        "auto",
        False,
        None,
    ),
    ("stp_hd64_fp32", 64, 128, 8, 64, 128, 0, torch.float32, "auto", False, None),
    ("stp_hd64_odd_pairs", 8, 24, 8, 64, 128, 0, torch.bfloat16, "auto", False, None),
)


def _supported_device() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


requires_blackwell = pytest.mark.skipif(
    not _supported_device(),
    reason="Cake selective state update requires SM100 or SM103",
)


def _make_case(case, destination_columns=None):
    (
        _name,
        batch_size,
        nheads,
        ngroups,
        dim,
        dstate,
        token_steps,
        state_dtype,
        algorithm,
        cache_intermediate,
        checkpoint_step,
    ) = case
    generator = torch.Generator(device="cuda").manual_seed(0)
    state = (
        torch.randn(
            (batch_size, nheads, dim, dstate), generator=generator, device="cuda"
        )
        * 0.05
    ).to(state_dtype)
    x_shape = (
        (batch_size, nheads, dim)
        if token_steps == 0
        else (batch_size, token_steps, nheads, dim)
    )
    x = (torch.randn(x_shape, generator=generator, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    dt_base = torch.randn(x_shape[:-1], generator=generator, device="cuda")
    dt = dt_base.as_strided(x_shape, (*dt_base.stride(), 0))
    A_base = -torch.rand((nheads,), generator=generator, device="cuda") - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    bc_shape = (
        (batch_size, ngroups, dstate)
        if token_steps == 0
        else (batch_size, token_steps, ngroups, dstate)
    )
    B = (torch.randn(bc_shape, generator=generator, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    C = (torch.randn(bc_shape, generator=generator, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    D_base = torch.randn((nheads,), generator=generator, device="cuda")
    D = D_base.as_strided((nheads, dim), (1, 0))
    bias_base = torch.rand((nheads,), generator=generator, device="cuda") - 4.0
    dt_bias = bias_base.as_strided((nheads, dim), (1, 0))
    source = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    destination = None
    if checkpoint_step is not None:
        destination = torch.full(
            (batch_size, token_steps), -1, dtype=torch.int64, device="cuda"
        )
        if destination_columns is None:
            destination[:, checkpoint_step] = source
        else:
            for row, column in enumerate(destination_columns):
                if column is not None:
                    destination[row, column] = source[row]
    intermediate = None
    intermediate_indices = None
    if cache_intermediate:
        intermediate = torch.empty(
            (batch_size, token_steps, nheads, dim, dstate),
            dtype=state_dtype,
            device="cuda",
        )
        intermediate_indices = source
    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "state_batch_indices": source,
        "dst_state_batch_indices": destination,
        "cache_steps": token_steps,
        "algorithm": algorithm,
        "dt_softplus": checkpoint_step is not None or cache_intermediate,
        "disable_state_update": cache_intermediate,
        "intermediate_states_buffer": intermediate,
        "intermediate_state_indices": intermediate_indices,
    }


def _split_arms(inputs):
    reference = dict(inputs)
    candidate = dict(inputs)
    for key in ("state", "intermediate_states_buffer"):
        if inputs[key] is not None:
            reference[key] = inputs[key].clone()
            candidate[key] = inputs[key].clone()
    return reference, candidate


def _assert_arms_close(candidate, out_candidate, reference, out_reference):
    torch.testing.assert_close(out_candidate, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        candidate["state"], reference["state"], atol=1e-2, rtol=1e-2
    )
    if candidate["intermediate_states_buffer"] is not None:
        torch.testing.assert_close(
            candidate["intermediate_states_buffer"],
            reference["intermediate_states_buffer"],
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.fixture
def cake_hits(monkeypatch):
    """Record whether every Cake call ran a Cake program (no silent fallback)."""
    hits = []
    original = cake.try_cake_selective_state_update

    def strict(**kwargs):
        hit = original(**kwargs)
        hits.append(hit)
        return hit

    monkeypatch.setattr(cake, "try_cake_selective_state_update", strict)
    return hits


@requires_blackwell
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case[0])
def test_cake_selective_state_update_matches_flashinfer(case, cake_hits) -> None:
    reference, candidate = _split_arms(_make_case(case))
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True], "promoted test row fell back instead of running Cake"
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
@pytest.mark.parametrize("name", ("mtp_short2", "dynamic0_b8"))
def test_softplus_is_the_identity_above_the_threshold(name, cake_hits) -> None:
    """``dt`` above 20 passes through softplus unchanged (``exp`` overflows past ~88); the reference thresholds too."""
    inputs = _make_case(_case(name))
    inputs["dt_softplus"] = True
    dt_rows = torch.full(inputs["dt"].shape[:-1], 100.0, device="cuda")
    dt_rows.flatten()[::3] = 30.0
    dt_rows.flatten()[1::3] = -0.5
    inputs["dt"] = dt_rows.as_strided(inputs["dt"].shape, (*dt_rows.stride(), 0))
    reference, candidate = _split_arms(inputs)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True]
    assert torch.isfinite(out_candidate).all()
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
def test_destination_table_with_per_sequence_columns(cake_hits) -> None:
    """Every non-pad entry of the destination table is a checkpoint; rows may differ."""
    case = ("dynamic_columns", 4, 16, 1, 64, 128, 8, torch.float32, "simple", False, 3)
    reference, candidate = _split_arms(
        _make_case(case, destination_columns=(1, 7, None, 0))
    )
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True]
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
def test_checkpoint_route_captures_into_cuda_graph(cake_hits) -> None:
    """The destination table is read on the device, so the route records into a CUDA graph."""
    case = ("dynamic_graph", 8, 16, 1, 64, 128, 2, torch.float32, "simple", False, 0)
    reference, candidate = _split_arms(_make_case(case))
    initial_state = candidate["state"].clone()
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = torch.empty_like(candidate["x"])
    cake_selective_state_update(**candidate, out=out_candidate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_selective_state_update(**candidate, out=out_candidate)
    assert cake_hits == [True, True], (
        "the checkpoint route must run on Cake inside stream capture"
    )
    candidate["state"].copy_(initial_state)
    out_candidate.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


def _raw_projection_case(batch_size: int):
    """Six-token projection views with padded batch/step strides, BF16 coefficients and int32 slot tables."""
    generator = torch.Generator(device="cuda").manual_seed(batch_size)
    nheads, ngroups, dim, dstate, steps = 64, 1, 64, 128, 6

    def strided(shape, strides, dtype, scale):
        span = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(shape, strides, strict=True)
            if size > 1
        )
        storage = torch.zeros((span,), dtype=dtype, device="cuda")
        view = storage.as_strided(shape, strides)
        view.copy_(
            (torch.randn(shape, generator=generator, device="cuda") * scale).to(dtype)
        )
        return view

    x = strided(
        (batch_size, steps, nheads, dim), (26112, 4352, 64, 1), torch.bfloat16, 0.1
    )
    dt_rows = strided(
        (batch_size, steps, nheads), (51072, 8512, 1), torch.bfloat16, 1.0
    )
    dt = dt_rows.as_strided((batch_size, steps, nheads, dim), (51072, 8512, 1, 0))
    B = strided(
        (batch_size, steps, ngroups, dstate), (26112, 4352, 128, 1), torch.bfloat16, 0.1
    )
    C = strided(
        (batch_size, steps, ngroups, dstate), (26112, 4352, 128, 1), torch.bfloat16, 0.1
    )
    A_base = -torch.rand((nheads,), generator=generator, device="cuda") - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    D_base = torch.randn((nheads,), generator=generator, device="cuda").to(
        torch.bfloat16
    )
    D = D_base.as_strided((nheads, dim), (1, 0))
    bias_base = (torch.rand((nheads,), generator=generator, device="cuda") - 4.0).to(
        torch.bfloat16
    )
    dt_bias = bias_base.as_strided((nheads, dim), (1, 0))
    state = (
        torch.randn((65, nheads, dim, dstate), generator=generator, device="cuda")
        * 0.05
    ).to(torch.bfloat16)
    source = torch.arange(batch_size, dtype=torch.int32, device="cuda") + 7
    intermediate = torch.empty(
        (5, steps, nheads, dim, dstate), dtype=torch.bfloat16, device="cuda"
    )
    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "state_batch_indices": source,
        "cache_steps": steps,
        "algorithm": "auto",
        "dt_softplus": True,
        "disable_state_update": True,
        "intermediate_states_buffer": intermediate,
        "intermediate_state_indices": torch.arange(
            batch_size, dtype=torch.int32, device="cuda"
        ),
    }


@requires_blackwell
@pytest.mark.parametrize("batch_size", (1, 2, 4))
def test_raw_projection_layout_matches_flashinfer(batch_size, cake_hits) -> None:
    """Padded projection views, BF16 dt/D/dt_bias and int32 slot tables run on the same program as the canonical layout."""
    candidate = _raw_projection_case(batch_size)
    nheads, dim = candidate["D"].shape

    def rows_over_dim(view, shape):
        # The reference backend takes the per-head coefficients as float views
        # broadcast over ``dim`` with a zero stride (the canonical layout).
        rows = view[..., 0].float().contiguous()
        return rows.as_strided(shape, (*rows.stride(), 0))

    reference = dict(
        candidate,
        state=candidate["state"].clone(),
        x=candidate["x"].contiguous(),
        dt=rows_over_dim(candidate["dt"], candidate["dt"].shape),
        B=candidate["B"].contiguous(),
        C=candidate["C"].contiguous(),
        D=rows_over_dim(candidate["D"], (nheads, dim)),
        dt_bias=rows_over_dim(candidate["dt_bias"], (nheads, dim)),
        state_batch_indices=candidate["state_batch_indices"].to(torch.int64),
        intermediate_states_buffer=candidate["intermediate_states_buffer"].clone(),
        intermediate_state_indices=candidate["intermediate_state_indices"].to(
            torch.int64
        ),
    )
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = torch.empty(
        (batch_size, 6, 64, 64), dtype=torch.bfloat16, device="cuda"
    )
    cake_selective_state_update(**candidate, out=out_candidate)
    assert cake_hits == [True], "the raw projection layout must run on Cake"
    torch.testing.assert_close(out_candidate, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        candidate["intermediate_states_buffer"][:batch_size],
        reference["intermediate_states_buffer"][:batch_size],
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        candidate["state"], reference["state"], atol=1e-2, rtol=1e-2
    )


def _raw_decode_case(batch_size: int, state_dtype, nheads: int = 128, ngroups: int = 8):
    """SGLang Mamba2 decode views: x/B/C column slices of the fused ``xBC`` conv output, dt the trailing
    columns of the fused ``zxbcdt`` projection broadcast over dim, BF16 coefficients, int32 slot tables."""
    generator = torch.Generator(device="cuda").manual_seed(batch_size)
    dim, dstate = 64, 128
    xbc_width = nheads * dim + 2 * ngroups * dstate
    zxbcdt_width = 2 * nheads * dim + 2 * ngroups * dstate + nheads
    xbc = (
        torch.randn((batch_size, xbc_width), generator=generator, device="cuda") * 0.1
    ).to(torch.bfloat16)
    x = xbc[:, : nheads * dim].view(batch_size, nheads, dim)
    B = xbc[:, nheads * dim : nheads * dim + ngroups * dstate].view(
        batch_size, ngroups, dstate
    )
    C = xbc[:, nheads * dim + ngroups * dstate :].view(batch_size, ngroups, dstate)
    zxbcdt = torch.randn(
        (batch_size, zxbcdt_width), generator=generator, device="cuda"
    ).to(torch.bfloat16)
    dt = (
        zxbcdt[:, zxbcdt_width - nheads :].unsqueeze(-1).expand(batch_size, nheads, dim)
    )
    A_base = -torch.rand((nheads,), generator=generator, device="cuda") - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    D_base = torch.randn((nheads,), generator=generator, device="cuda").to(
        torch.bfloat16
    )
    D = D_base.as_strided((nheads, dim), (1, 0))
    bias_base = (torch.rand((nheads,), generator=generator, device="cuda") - 4.0).to(
        torch.bfloat16
    )
    dt_bias = bias_base.as_strided((nheads, dim), (1, 0))
    state = (
        torch.randn(
            (batch_size + 3, nheads, dim, dstate), generator=generator, device="cuda"
        )
        * 0.05
    ).to(state_dtype)
    source = torch.arange(batch_size, dtype=torch.int32, device="cuda") + 2
    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "state_batch_indices": source,
        "dst_state_batch_indices": None,
        "cache_steps": 0,
        "algorithm": "auto",
        "dt_softplus": True,
        "disable_state_update": False,
        "intermediate_states_buffer": None,
        "intermediate_state_indices": None,
    }


def _canonical_arm(candidate):
    """The same call on the canonical layout (contiguous rows, FP32 coefficients, int64 tables) for the reference backend."""
    nheads, dim = candidate["D"].shape

    def rows_over_dim(view, shape):
        rows = view[..., 0].float().contiguous()
        return rows.as_strided(shape, (*rows.stride(), 0))

    return dict(
        candidate,
        state=candidate["state"].clone(),
        x=candidate["x"].contiguous(),
        dt=rows_over_dim(candidate["dt"], candidate["dt"].shape),
        B=candidate["B"].contiguous(),
        C=candidate["C"].contiguous(),
        D=rows_over_dim(candidate["D"], (nheads, dim)),
        dt_bias=rows_over_dim(candidate["dt_bias"], (nheads, dim)),
        state_batch_indices=candidate["state_batch_indices"].to(torch.int64),
    )


def _routes_as_measured() -> bool:
    """The paired row-major programs are chosen by program heads per SM: the expected names of the test cases
    hold on B200 (sm_100a, 148 SMs) and GB300 (sm_103a, 152 SMs)."""
    device = torch.cuda.current_device()
    return (torch.cuda.get_device_capability(device), cake._sm_count(device)) in (
        ((10, 0), 148),
        ((10, 3), 152),
    )


def _build(program, plan) -> str:
    """The build ``plan`` runs for the measured ``program`` (a name, or a name per architecture): its ``_lean`` build
    where the architecture lists the program or the launch's CTAs fall inside its lean band
    (:func:`cake.lean_program`), else the program."""
    if isinstance(program, dict):
        program = program[plan.arch]
    return cake.lean_program(
        program, plan.arch, plan.grid[0], cake._sm_count(plan.device_index)
    )


# (name, batch, state dtype, program, groups): 128 heads of 64 rows, 64 program heads when paired.
_RAW_DECODE_CASES = (
    ("rows_bf16", 2, torch.bfloat16, "stp_hd64_rows_bf16_dt8_bf16_i32", 8),
    (
        "split_bf16_one_head",
        4,
        torch.bfloat16,
        "stp_hd64_paired_hpc1_ratio8_s4w2_bf16_i32",
        8,
    ),
    (
        "split_bf16_one_head_g4",
        4,
        torch.bfloat16,
        "stp_hd64_paired_hpc1_s4w2_bf16_i32",
        4,
    ),
    ("paired_bf16", 64, torch.bfloat16, "stp_hd64_paired_hpc2_bf16_i32", 8),
    (
        "split_bf16_two_heads",
        5,
        torch.bfloat16,
        "stp_hd64_paired_hpc2_ratio8_s2w4_bf16_i32",
        8,
    ),
    (
        "split_bf16_two_heads_g4",
        5,
        torch.bfloat16,
        "stp_hd64_paired_hpc2_s2w4_bf16_i32",
        4,
    ),
    (
        "paired_bf16_two_ctas",
        8,
        torch.bfloat16,
        {
            "sm_100a": "stp_hd64_paired_hpc2_mb2_bf16_i32",
            "sm_103a": "stp_hd64_paired_hpc2_ratio8_s2w4_bf16_i32",
        },
        8,
    ),
    (
        "split_bf16_four_heads",
        16,
        torch.bfloat16,
        "stp_hd64_paired_hpc4_s2w4_bf16_i32",
        8,
    ),
    (
        "split_bf16_four_heads_s4",
        20,
        torch.bfloat16,
        "stp_hd64_paired_hpc4_s4w2_bf16_i32",
        8,
    ),
    ("split_bf16_mb5", 32, torch.bfloat16, "stp_hd64_paired_hpc2_s2w4_mb5_bf16_i32", 8),
    ("rows_fp32_dt8", 1, torch.float32, "stp_hd64_rows_fp32_dt8_bf16_i32", 8),
    ("rows_fp32_dt4", 2, torch.float32, "stp_hd64_rows_fp32_dt4_bf16_i32", 8),
    ("rows_fp32_dt2", 4, torch.float32, "stp_hd64_rows_fp32_dt2_bf16_i32", 8),
    ("rows_fp32", 8, torch.float32, "stp_hd64_rows_fp32_dt1_bf16_i32", 8),
    ("rows_fp32_2048", 16, torch.float32, "stp_hd64_rows_fp32_dt1_bf16_i32", 8),
    (
        "rows_fp32_l2_prefetch",
        32,
        torch.float32,
        "stp_hd64_rows_fp32_dt1_l2_prefetch_bf16_i32",
        8,
    ),
    ("rows_fp32_between", 48, torch.float32, "stp_hd64_rows_fp32", 8),
    ("rows_fp32_large", 64, torch.float32, "stp_hd64_rows_fp32_dt1_large_bf16_i32", 8),
)


@requires_blackwell
@pytest.mark.parametrize("case", _RAW_DECODE_CASES, ids=lambda case: case[0])
def test_raw_decode_layout_matches_flashinfer(case, cake_hits) -> None:
    """Padded fused-projection views, BF16 dt/D/dt_bias and int32 slot tables run on the headdim-64 programs."""
    _name, batch_size, state_dtype, program, ngroups = case
    candidate = _raw_decode_case(batch_size, state_dtype, ngroups=ngroups)
    reference = _canonical_arm(candidate)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    plan = _route(dict(candidate, x=candidate["x"]), output_like=candidate["x"])
    assert plan is not None
    if _routes_as_measured():
        assert plan.program == _build(program, plan)
    _assert_split_grid(plan)
    out_candidate = torch.empty(
        (batch_size, 128, 64), dtype=torch.bfloat16, device="cuda"
    )
    cake_selective_state_update(**candidate, out=out_candidate)
    assert cake_hits == [True], "the raw decode layout must run on Cake"
    torch.testing.assert_close(out_candidate, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        candidate["state"], reference["state"], atol=1e-2, rtol=1e-2
    )


# (name, raw ABI, nheads, batch, {arch: program}): BF16-state headdim-64 rows of one head per group (no head
# pairing) at 512 batch-heads, FlashInfer's four-tile band on both devices.
_BF16_DT4_CASES = (
    (
        "canonical_h128",
        False,
        128,
        4,
        {
            "sm_100a": "stp_hd64_rows_bf16_dt4_xa_od",
            "sm_103a": "stp_hd64_rows_bf16_dt4_od",
        },
    ),
    (
        "canonical_h32",
        False,
        32,
        16,
        {
            "sm_100a": "stp_hd64_rows_bf16_dt4_xa_od",
            "sm_103a": "stp_hd64_rows_bf16_dt4_od",
        },
    ),
    (
        "raw_h128",
        True,
        128,
        4,
        {
            "sm_100a": "stp_hd64_rows_bf16_dt4_od_bf16_i32",
            "sm_103a": "stp_hd64_rows_bf16_dt4_od_bf16_i32",
        },
    ),
    (
        "raw_h64",
        True,
        64,
        8,
        {
            "sm_100a": "stp_hd64_rows_bf16_dt4_od_bf16_i32",
            "sm_103a": "stp_hd64_rows_bf16_dt4_od_bf16_i32",
        },
    ),
    (
        "raw_h32",
        True,
        32,
        16,
        {
            "sm_100a": "stp_hd64_rows_bf16_dt4_od_bf16_i32",
            "sm_103a": "stp_hd64_rows_bf16_dt4_xa_bf16_i32",
        },
    ),
)


@requires_blackwell
@pytest.mark.parametrize("case", _BF16_DT4_CASES, ids=lambda case: case[0])
def test_bf16_four_tile_rows_pick_the_build_of_the_device(case, cake_hits) -> None:
    """The four-tile BF16 band runs the build of the device's architecture and, in the BF16/int32 ABI, of the
    head count: the outputs and state match FlashInfer on every build."""
    _name, raw_abi, nheads, batch_size, programs = case
    raw = _raw_decode_case(batch_size, torch.bfloat16, nheads=nheads, ngroups=nheads)
    reference = _canonical_arm(raw)
    candidate = raw if raw_abi else dict(reference, state=reference["state"].clone())
    out_reference = selective_state_update(**reference, backend="flashinfer")
    plan = _route(dict(candidate), output_like=candidate["x"])
    assert plan is not None
    if _routes_as_measured():
        arch = cake._device_arch(torch.cuda.current_device())
        assert (plan.program, plan.grid) == (programs[arch], (4, nheads, batch_size))
    out_candidate = torch.empty(
        (batch_size, nheads, 64), dtype=torch.bfloat16, device="cuda"
    )
    cake_selective_state_update(**candidate, out=out_candidate)
    assert cake_hits == [True], "the four-tile BF16 rows must run on Cake"
    torch.testing.assert_close(out_candidate, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        candidate["state"], reference["state"], atol=1e-2, rtol=1e-2
    )


# (name, batch, program, (DIM_TILES, PREFETCH_ROWS) of the carried program or None): canonical-ABI FP32 rows of
# 128 heads of 64 rows. The bands without measured canonical-ABI rows (four and two tiles, and 4096 to 8192
# batch-heads at one tile) keep the carried row-owner program under the tile plan's defines.
_CANONICAL_FP32_ROWS_CASES = (
    ("dt8", 1, "stp_hd64_rows_fp32_dt8", None),
    ("dt4", 2, "stp_hd64_rows_fp32", (4, 1)),
    ("dt2", 4, "stp_hd64_rows_fp32", (2, 4)),
    ("dt1", 8, "stp_hd64_rows_fp32_dt1", None),
    ("dt1_between", 48, "stp_hd64_rows_fp32", (1, 4)),
    ("dt1_large", 64, "stp_hd64_rows_fp32_dt1_large", None),
)


@requires_blackwell
@pytest.mark.parametrize("case", _CANONICAL_FP32_ROWS_CASES, ids=lambda case: case[0])
def test_canonical_fp32_rows_keep_the_carried_program_outside_the_measured_bands(
    case, cake_hits
) -> None:
    """Canonical FP32 headdim-64 rows run their band's rendered build where it has one; a canonical call at four
    or two tiles, or between 4096 and 8192 batch-heads, runs the carried row-owner program as before."""
    _name, batch_size, program, carried = case
    candidate = _canonical_arm(_raw_decode_case(batch_size, torch.float32))
    reference = dict(candidate, state=candidate["state"].clone())
    out_reference = selective_state_update(**reference, backend="flashinfer")
    plan = _route(candidate)
    assert plan is not None
    if _routes_as_measured():
        assert plan.program == program
        if carried is None:
            assert plan.defines == ()
        else:
            dim_tiles, prefetch_rows = carried
            assert plan.defines == cake.rows_defines(
                torch.float32, torch.int64, prefetch_rows, dim_tiles
            )
            assert plan.grid == (dim_tiles, 128, batch_size)
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True], "the canonical FP32 rows must run on Cake"
    torch.testing.assert_close(out_candidate, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        candidate["state"], reference["state"], atol=1e-2, rtol=1e-2
    )


@requires_blackwell
@pytest.mark.parametrize("batch_size", (2, 64))
def test_hd64_pad_rows_write_the_zero_state_output(batch_size, cake_hits) -> None:
    """Rows whose slot equals ``pad_slot_id`` (SGLang CUDA-graph padding) read a zero state, write that row's output
    like the FlashInfer STP kernel, and leave the state pool untouched."""
    candidate = _raw_decode_case(batch_size, torch.bfloat16)
    candidate["state_batch_indices"][0] = -1
    if batch_size > 2:
        candidate["state_batch_indices"][batch_size // 2] = -1
    live = candidate["state_batch_indices"] >= 0
    initial_state = candidate["state"].clone()
    reference = _canonical_arm(candidate)
    out_reference = selective_state_update(
        **reference, backend="flashinfer", pad_slot_id=-1
    )
    out_candidate = torch.full(
        (batch_size, 128, 64), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    cake_selective_state_update(**candidate, out=out_candidate, pad_slot_id=-1)
    assert cake_hits == [True]
    torch.testing.assert_close(out_candidate, out_reference, atol=1e-2, rtol=1e-2)
    assert bool((out_candidate[~live] != 7.0).any()), (
        "pad rows carry the zero-state output row"
    )
    torch.testing.assert_close(
        candidate["state"], reference["state"], atol=1e-2, rtol=1e-2
    )
    untouched = torch.ones(candidate["state"].shape[0], dtype=torch.bool, device="cuda")
    untouched[candidate["state_batch_indices"][live].long()] = False
    assert torch.equal(candidate["state"][untouched], initial_state[untouched])


@requires_blackwell
@pytest.mark.parametrize(
    "case_name", ("stp_hd64_rows_bf16", "stp_hd64_paired_rows", "stp_hd64_fp32")
)
def test_hd64_routes_capture_into_cuda_graph(case_name, cake_hits) -> None:
    """The headdim-64 routes read every slot on the device: no host synchronization, graph-capturable."""
    reference, candidate = _split_arms(_make_case(_case(case_name)))
    initial_state = candidate["state"].clone()
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = torch.empty_like(candidate["x"])
    cake_selective_state_update(**candidate, out=out_candidate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_selective_state_update(**candidate, out=out_candidate)
    assert cake_hits == [True, True], (
        "the headdim-64 routes must run on Cake inside stream capture"
    )
    candidate["state"].copy_(initial_state)
    out_candidate.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
def test_hd64_route_falls_back_for_non_dense_rows(cake_hits) -> None:
    """Rows with a non-unit inner stride are outside the programs' addressing; the call runs on FlashInfer."""
    inputs = _make_case(_case("stp_hd64_paired_rows"))
    plan = _route(inputs)
    assert plan.program == _build("stp_hd64_paired_hpc2", plan)
    batch_size, nheads, dim = inputs["x"].shape
    wide = torch.zeros(
        (batch_size, nheads, 2 * dim), dtype=torch.bfloat16, device="cuda"
    )
    wide[..., ::2].copy_(inputs["x"])
    inputs["x"] = wide[..., ::2]
    assert _route(inputs) is None
    _assert_falls_back(inputs, cake_hits)


@requires_blackwell
def test_hd64_saturated_odd_pairing_falls_back(cake_hits) -> None:
    """An odd head pairing at saturating work has no delivered tile (the four-warp BF16 row owner); FlashInfer serves it."""
    inputs = _make_case(
        ("odd_saturated", 256, 24, 8, 64, 128, 0, torch.bfloat16, "auto", False, None)
    )
    assert _route(inputs) is None
    _assert_falls_back(inputs, cake_hits)


def test_hd64_defines_follow_the_storage_dtypes() -> None:
    assert cake.rows_defines(torch.float32, torch.int64, 1, 8) == (
        ("COEFFICIENT_BF16", 0),
        ("INDEX_I32", 0),
        ("PREFETCH_ROWS", 1),
        ("DIM_TILES", 8),
    )
    assert cake.rows_defines(torch.bfloat16, torch.int32, 4, 2) == (
        ("COEFFICIENT_BF16", 1),
        ("INDEX_I32", 1),
        ("PREFETCH_ROWS", 4),
        ("DIM_TILES", 2),
    )


def test_paired_rows_programs_follow_the_work() -> None:
    """Program heads per SM select the work tile of the paired headdim-64 route; the gate and the head ratio
    select the 128x128 build."""

    def hd64(row_groups, sm_count, program_heads_per_group=8, raw_abi=False):
        return cake.hd64_paired_program(
            row_groups, program_heads_per_group, sm_count, raw_abi
        )

    # (batch x groups, SMs[, program heads per group]); comments count program heads and CTAs.
    assert hd64(55, 148) == ("stp_hd64_paired_hpc1_ratio8", 1)  # 440: below 3 per SM
    assert hd64(110, 148, 4) == ("stp_hd64_paired_hpc1", 1)  # runtime head ratio
    assert hd64(111, 148, 4) == ("stp_hd64_paired_hpc2_mb2", 2)  # 444: 222 CTAs
    assert hd64(111, 152, 4) == ("stp_hd64_paired_hpc1", 1)  # 444: 2.9 per SM
    assert hd64(296, 148, 2) == ("stp_hd64_paired_hpc2_mb2", 2)  # 296 CTAs: two per SM
    assert hd64(297, 148, 2) == ("stp_hd64_paired_hpc2", 2)  # 297 CTAs
    assert hd64(76, 152) == ("stp_hd64_paired_hpc2_mb2", 2)  # 304 CTAs
    # An odd head count per group leaves a part-filled tile in every group: the CTAs, not the heads, decide.
    assert hd64(148, 148, 3) == ("stp_hd64_paired_hpc2_mb2", 2)  # 444: 296 CTAs
    assert hd64(149, 148, 3) == ("stp_hd64_paired_hpc2", 2)  # 447: 298 CTAs
    assert hd64(160, 148, 3) == ("stp_hd64_paired_hpc2", 2)  # 480: 320 CTAs
    # From 6 per SM: four heads per CTA unless two put fewer head slots on the busiest SM.
    assert hd64(111, 148) == ("stp_hd64_paired_hpc2", 2)  # busiest SM 6 vs 8
    assert hd64(114, 148) == ("stp_hd64_paired_hpc4", 4)  # 8 vs 8
    assert hd64(150, 148) == ("stp_hd64_paired_hpc2", 2)  # 10 vs 12
    assert hd64(150, 152) == ("stp_hd64_paired_hpc4", 4)  # 8 vs 8
    assert hd64(400, 148, 3) == ("stp_hd64_paired_hpc4", 4)  # 800 / 400 CTAs: 12 vs 12
    assert hd64(221, 148) == ("stp_hd64_paired_hpc4", 4)  # 12 vs 12
    assert hd64(222, 148) == ("stp_hd64_paired_hpc2", 2)  # 1776: 12 per SM
    assert hd64(512, 152, raw_abi=True) == ("stp_hd64_paired_hpc2_bf16_i32", 2)
    assert cake.dim128_program(8, gated=True) == ("stp_dim128_z_gated", 2)
    assert cake.dim128_program(1, gated=True) == ("stp_dim128_z_gated", 2)
    assert cake.dim128_program(1, gated=False) == ("stp_dim128_hpc1_mb4", 1)
    assert cake.dim128_program(8, gated=False) == ("stp_dim128_hpc2", 2)


def test_balanced_units_follow_the_measured_intervals() -> None:
    """A z-gated 128x128 call runs the persistent build inside the measured ``batch x nheads`` intervals of B200
    (148 SMs) and GB300 (152 SMs), edges included; every other call, SM count or architecture keeps the z-gated
    build."""

    def units(batch_size, nheads=64, sm_count=148, arch="sm_100a"):
        return cake.dim128_balanced_units(batch_size, nheads, sm_count, arch)

    gb300 = {"sm_count": 152, "arch": "sm_103a"}
    assert units(1) == 64 and units(1, 63) == 0  # the smallest measured call
    assert units(6) == 384 and units(7) == 0  # three CTAs per SM: 444 on B200
    assert units(7, **gb300) == 448 and units(8, **gb300) == 0
    assert units(19) == 0 and units(20) == 1280 and units(20, **gb300) == 1280
    assert units(26) == 1664 and units(27) == 0  # 1728 heads
    assert units(28, **gb300) == 1792 and units(29, **gb300) == 0
    assert units(39) == 0 and units(40) == 2560 and units(44) == 2816
    assert units(45) == 0 and units(45, **gb300) == 2880
    assert units(52, **gb300) == 3328 and units(53, **gb300) == 0
    assert units(51) == 0 and units(52) == 3328 and units(64) == 4096
    assert units(65) == 0 and units(59, **gb300) == 0 and units(60, **gb300) == 3840
    assert units(68, **gb300) == 4352 and units(69, **gb300) == 0
    assert units(300, 1) == 300 and units(97, 6) == 0  # 582 heads
    assert units(48, 30) == 1440 and units(48, 30, **gb300) == 1440
    assert units(4, sm_count=132) == 0  # an SM count without measurements
    assert units(4, arch="sm_103a") == 0 and units(4, sm_count=152) == 0


def test_split_programs_follow_the_work() -> None:
    """The P-split rule: ``(program, heads_per_cta, p_split)`` of the split build a launch runs, or ``None`` for
    the build of :func:`hd64_paired_program` / :func:`dim128_program`."""

    def hd64(rg, hpg, sms=148, arch="sm_100a", raw=False):
        return cake.hd64_split_program(rg, hpg, sms, arch, raw)

    hpc1_s4 = ("stp_hd64_paired_hpc1_s4w2", 1, 4)
    hpc2_s2 = ("stp_hd64_paired_hpc2_s2w4", 2, 2)
    hpc4_s2 = ("stp_hd64_paired_hpc4_s2w4", 4, 2)
    hpc4_s4 = ("stp_hd64_paired_hpc4_s4w2", 4, 4)
    mb5 = ("stp_hd64_paired_hpc2_s2w4_mb5", 2, 2)
    ratio8_s4 = ("stp_hd64_paired_hpc1_ratio8_s4w2", 1, 4)
    # (batch x groups, program heads per group[, SMs, arch]); comments count program heads and CTAs.
    # Below three program heads per SM: one to two per SM split four ways, above two split two ways.
    assert hd64(36, 4) is None  # 144: below one per SM, the row-owner tile
    assert hd64(37, 4) == hpc1_s4  # 148
    assert hd64(74, 4) == hpc1_s4  # 296
    assert hd64(50, 3) == hpc1_s4  # 150 in groups of three program heads
    assert hd64(75, 4) == hpc2_s2  # 300
    assert hd64(110, 4) == hpc2_s2  # 440
    assert hd64(100, 3) is None  # 300 in groups of three: no head pairs to split
    assert hd64(16, 8) is None  # 128
    assert hd64(19, 8) == ratio8_s4  # 152: the folded ratio at eight per group
    assert hd64(40, 8) == ("stp_hd64_paired_hpc2_ratio8_s2w4", 2, 2)  # 320
    assert hd64(19, 8, raw=True) == (f"{ratio8_s4[0]}_bf16_i32", 1, 4)
    # From three per SM, by the unsplit build and its CTAs.
    assert hd64(111, 4) is None  # 444: two heads per CTA at two CTAs per SM
    assert hd64(111, 4, 152, "sm_103a") == hpc2_s2  # 444: below three per SM
    assert hd64(128, 8) == hpc4_s2  # 1024: four heads per CTA, 256 CTAs
    assert hd64(221, 8) is None  # 1768: four heads per CTA, 442 CTAs
    assert hd64(80, 8) == hpc4_s2  # 640 / 320 CTAs: up to three CTAs per SM
    assert hd64(100, 6) is None  # 600 / 300 CTAs in groups of six program heads
    assert hd64(160, 8) == hpc4_s4  # 1280 / 640 CTAs
    assert hd64(166, 8) == hpc4_s4  # 1328: at most nine program heads per SM
    assert hd64(170, 8) is None  # 1360
    assert hd64(222, 8) == mb5  # 888 CTAs: 6 per SM on sm_100a
    assert hd64(244, 8) == mb5  # 976 CTAs
    assert hd64(277, 8) == mb5  # 1108 CTAs: below 7.5 per SM
    assert hd64(278, 8) is None  # 1112 CTAs
    assert hd64(227, 8, 152, "sm_103a") is None  # 1816: below the band on sm_103a too
    assert hd64(228, 8, 152, "sm_103a") == mb5  # 912 CTAs: 6 per SM on sm_103a
    assert hd64(244, 8, 152, "sm_103a") == mb5  # 976 CTAs
    assert hd64(251, 8, 152, "sm_103a") == mb5  # 1004 CTAs
    assert hd64(284, 8, 152, "sm_103a") == mb5  # 1136 CTAs: below 7.5 per SM
    assert hd64(285, 8, 152, "sm_103a") is None  # 1140 CTAs
    assert hd64(887, 8) is None  # 3548 CTAs
    assert hd64(888, 8) == mb5  # 3552 CTAs: 24 per SM
    assert hd64(888, 8, 152, "sm_103a") is None
    assert hd64(1824, 8, 152, "sm_103a") == mb5  # 7296 CTAs: 48 per SM
    # The two-head two-CTA build on sm_103a from 1.68 to 1.90 of its CTAs per SM: split two ways (one wave of four
    # CTAs per SM); every other launch of it, and sm_100a, keep the build.
    c2_ratio8 = ("stp_hd64_paired_hpc2_ratio8_s2w4", 2, 2)
    assert hd64(60, 8, 152, "sm_103a") is None  # 480 / 240 CTAs: 1.58 per SM
    assert hd64(64, 8, 152, "sm_103a") == c2_ratio8  # 512 / 256 CTAs: 1.68 per SM
    assert hd64(72, 8, 152, "sm_103a") == c2_ratio8  # 576 / 288 CTAs: 1.89 per SM
    assert hd64(73, 8, 152, "sm_103a") is None  # 584 / 292 CTAs: 1.92 per SM
    assert hd64(8, 64, 152, "sm_103a") == hpc2_s2  # 512 in one group of 64
    assert hd64(64, 8, 152, "sm_103a", raw=True) == (f"{c2_ratio8[0]}_bf16_i32", 2, 2)
    assert hd64(64, 8) is None  # sm_100a: 512 / 256 CTAs keep the two-CTA build

    def dim128(rg, hpg, arch="sm_100a", gated=False):
        return cake.dim128_split_program(rg, hpg, 148, arch, gated)

    assert dim128(512, 1, gated=True) is None
    assert dim128(518, 1) == ("stp_dim128_hpc1_s2w4", 1, 2)  # 518 CTAs: 3.5 per SM
    assert dim128(519, 1) is None
    assert dim128(1776, 8) == ("stp_dim128_hpc2_s2w4_mb5", 2, 2)  # 7104 CTAs: 48 per SM
    assert dim128(1775, 8) is None  # 7100 CTAs: the two-head build
    assert dim128(888, 8) is None  # 3552 CTAs: 24 per SM, 8 waves
    assert dim128(1776, 8, "sm_103a") is None
    assert cake.dim128_split_program(4096, 8, 152, "sm_103a", False) is None


def test_four_head_split_takes_the_nearly_full_waves() -> None:
    """A launch of the two-head build at three CTAs per SM (groups of a multiple of four program heads) runs four
    heads per CTA split two ways when its CTAs end in the last quarter (sm_100a) or the last 40 % (sm_103a) of a
    third or fourth wave of four CTAs per SM; a fifth wave, groups of six heads and the other launches keep their
    build."""

    def hd64(rg, hpg=8, sms=148, arch="sm_100a", raw=False):
        return cake.hd64_split_program(rg, hpg, sms, arch, raw)

    c4 = ("stp_hd64_paired_hpc4_s2w4", 4, 2)
    gb300 = {"sms": 152, "arch": "sm_103a"}
    # (batch x groups[, program heads per group]); comments count the two-head build's CTAs.
    assert hd64(407) is None  # 1628: 2.75 waves of four CTAs per SM
    assert hd64(408) == c4  # 1632
    assert hd64(444) == c4  # 1776: three whole waves
    assert hd64(445) is None  # 1780
    assert hd64(555) is None and hd64(556) == c4  # 2220 / 2224
    assert hd64(592) == c4 and hd64(593) is None  # 2368 / 2372
    assert hd64(740) is None  # 2960: five whole waves
    assert hd64(395, **gb300) is None and hd64(396, **gb300) == c4  # 1580 / 1584
    assert hd64(456, **gb300) == c4 and hd64(457, **gb300) is None  # 1824 / 1828
    assert hd64(547, **gb300) is None and hd64(548, **gb300) == c4  # 2188 / 2192
    assert hd64(608, **gb300) == c4 and hd64(609, **gb300) is None  # 2432 / 2436
    assert hd64(880, 4) == c4  # 1760 CTAs of four program heads per group
    assert hd64(587, 6) is None  # 1761 CTAs in groups of six program heads
    assert hd64(440, raw=True) == (f"{c4[0]}_bf16_i32", 4, 2)


def test_unsplit_program_takes_four_heads_in_the_band() -> None:
    """The two-head build at three CTAs per SM runs four heads per CTA from 3.05 to 3.46 waves of three CTAs per
    SM on sm_100a and from 3.09 to 3.37 on sm_103a, for groups of a multiple of four program heads; every other
    launch keeps :func:`hd64_paired_program`'s build."""

    def unsplit(rg, hpg=8, sms=148, arch="sm_100a", raw=False):
        return cake.hd64_unsplit_program(rg, hpg, sms, arch, raw)

    hpc2, hpc4 = ("stp_hd64_paired_hpc2", 2), ("stp_hd64_paired_hpc4", 4)
    gb300 = {"sms": 152, "arch": "sm_103a"}
    # (batch x groups[, program heads per group]); comments count the two-head build's CTAs.
    # sm_100a: 1352 / 1356 (3.05 waves: 1354.2), 1536 / 1540 (3.46 waves: 1536.2), 1600.
    assert unsplit(338) == hpc2 and unsplit(339) == hpc4
    assert unsplit(384) == hpc4 and unsplit(385) == hpc2
    assert unsplit(400) == hpc2
    # sm_103a: 1408 / 1412 (3.09 waves: 1409.0), 1536 / 1540 (3.37 waves: 1536.7).
    assert unsplit(352, **gb300) == hpc2 and unsplit(353, **gb300) == hpc4
    assert unsplit(384, **gb300) == hpc4 and unsplit(385, **gb300) == hpc2
    assert unsplit(700, 4) == hpc4  # 1400 CTAs of four program heads per group
    assert unsplit(233, 6) == ("stp_hd64_paired_hpc2", 2)  # 699 CTAs in groups of six
    assert unsplit(360, raw=True) == ("stp_hd64_paired_hpc4_bf16_i32", 4)
    # An architecture without the band.
    assert unsplit(360, sms=148, arch="sm_120a") == hpc2
    # Below six program heads per SM the band is not reached: hd64_paired_program's build.
    for rg in (40, 111, 114, 150, 221, 222):
        assert unsplit(rg) == cake.hd64_paired_program(rg, 8, 148, False)


def test_lean_builds_follow_the_architecture() -> None:
    """A launch runs the ``_lean`` build (consumer proxy fence after the full-stage wait dropped) of the programs its
    architecture lists, in either ABI; the loader ships every ``_lean`` build for exactly those architectures and
    never the plain build an architecture replaces."""
    lean = cake.lean_program
    assert lean("stp_hd64_paired_hpc2", "sm_100a") == "stp_hd64_paired_hpc2_lean"
    assert (
        lean("stp_hd64_paired_hpc2_bf16_i32", "sm_100a")
        == "stp_hd64_paired_hpc2_lean_bf16_i32"
    )
    assert lean("stp_hd64_paired_hpc2", "sm_103a") == "stp_hd64_paired_hpc2"
    # sm_103a runs the two-head build's lean form from 6.10 to 26.95 of its CTAs per SM.
    gb300_hpc2 = (
        ("stp_hd64_paired_hpc2", "sm_103a"),
        ("stp_hd64_paired_hpc2_bf16_i32", "sm_103a"),
    )
    for (program, arch), suffix in zip(gb300_hpc2, ("", "_bf16_i32"), strict=True):
        assert lean(program, arch, 927, 152) == program
        assert lean(program, arch, 928, 152) == f"stp_hd64_paired_hpc2_lean{suffix}"
        assert lean(program, arch, 4096, 152) == f"stp_hd64_paired_hpc2_lean{suffix}"
        assert lean(program, arch, 4100, 152) == program
    assert (
        lean("stp_hd64_paired_hpc2", "sm_100a", 4100, 148)
        == "stp_hd64_paired_hpc2_lean"
    )
    assert lean("stp_hd64_paired_hpc1_s4w2", "sm_100a") == "stp_hd64_paired_hpc1_s4w2"
    assert (
        lean("stp_hd64_paired_hpc1_s4w2", "sm_103a") == "stp_hd64_paired_hpc1_s4w2_lean"
    )
    assert lean("stp_dim128_hpc1_s2w4", "sm_100a") == "stp_dim128_hpc1_s2w4"
    assert lean("stp_dim128_hpc1_s2w4", "sm_103a") == "stp_dim128_hpc1_s2w4_lean"
    for arch in ("sm_100a", "sm_103a"):
        assert lean("stp_dim128_z_balanced", arch) == "stp_dim128_z_balanced_lean"
        assert (
            lean("stp_hd64_paired_hpc4_s2w4", arch) == "stp_hd64_paired_hpc4_s2w4_lean"
        )
        for kept in (
            "stp_hd64_paired_hpc1",
            "stp_hd64_paired_hpc1_ratio8_bf16_i32",
            "stp_dim128_hpc1_mb4",
            "stp_dim128_z_gated",
            "stp_hd64_rows_bf16_dt8",
            "dynamic_lanes16",
        ):
            assert lean(kept, arch) == kept
        for name, module in cake.PROGRAMS.items():
            arches = cake.MODULES[module].get("arches", ["sm_100a", "sm_103a"])
            base = name.removesuffix("_bf16_i32")
            if base.endswith("_lean"):
                takes = base.removesuffix("_lean") in cake._LEAN_PROGRAMS[arch] or (
                    base.removesuffix("_lean")
                    in cake._LEAN_BAND_HUNDREDTHS.get(arch, {})
                )
                assert (arch in arches) == takes, (name, arch)
            elif base in cake._LEAN_PROGRAMS[arch]:
                assert arch not in arches, (name, arch)
    assert lean("stp_hd64_paired_hpc4_s2w4", "sm_120a") == "stp_hd64_paired_hpc4_s2w4"


def test_dynamic_program_follows_the_token_count() -> None:
    """The dynamic program's build: 16 lanes per row below eight batch rows of a multi-token call, its
    pipelined-reduction build for one batch row at 4 to 8 tokens on sm_100a and 5 to 8 on sm_103a."""
    pr = {"sm_100a": (4, 5, 6, 7, 8), "sm_103a": (5, 6, 7, 8)}
    for arch, steps in pr.items():
        for batch in range(1, 12):
            for tokens in range(1, 9):
                if tokens < 2 or batch >= 8:
                    want = "dynamic_lanes8"
                elif batch == 1 and tokens in steps:
                    want = "dynamic_lanes16_pr"
                else:
                    want = "dynamic_lanes16"
                assert cake.dynamic_program(batch, tokens, arch) == want


def test_every_route_build_ships_for_its_architecture() -> None:
    """Every build the single-token and dynamic routes can name on an architecture is a program the loader
    delivers there (a narrowed or retired build is never routed)."""
    for arch, sms in (("sm_100a", 148), ("sm_103a", 152)):
        named = set()
        for hpg in (1, 2, 3, 4, 6, 8, 16, 64):
            for rg in range(1, 3000):
                if rg * hpg < sms:
                    continue  # the row-owner tile
                for raw in (False, True):
                    split = cake.hd64_split_program(rg, hpg, sms, arch, raw)
                    named.add(
                        split[0]
                        if split
                        else cake.hd64_unsplit_program(rg, hpg, sms, arch, raw)[0]
                    )
        for hpg in (1, 2, 8):
            for rg in range(1, 3000):
                split = cake.dim128_split_program(rg, hpg, sms, arch, False)
                named.add(split[0] if split else cake.dim128_program(hpg, False)[0])
        named |= {"stp_dim128_z_gated", "stp_dim128_z_balanced"}
        named |= {cake.dim128_fp32_program(bh) for bh in range(1, 9000)}
        if any(cake.dim128_stp_ring(b, 64, 8, sms, arch) for b in range(1, 3000)):
            named.add(_STP_RING)
        named |= {
            cake.mtp_short_program(b, nheads, 8, t, sms, arch)[0]
            for b in range(1, 3000)
            for t in (1, 2)
            for nheads in (32, 64)
        }
        named |= {
            cake.dynamic_program(b, t, arch) for b in range(1, 9) for t in range(1, 9)
        }
        for program in named:
            builds = {cake.lean_program(program, arch)}
            band = cake._LEAN_BAND_HUNDREDTHS.get(arch, {}).get(
                program.removesuffix("_bf16_i32")
            )
            if band is not None:
                builds.add(
                    cake.lean_program(program, arch, band[0] * sms // 100 + 1, sms)
                )
            for build in builds:
                module = cake.PROGRAMS[build]
                assert arch in cake.MODULES[module].get("arches", [arch]), (build, arch)


@requires_blackwell
@pytest.mark.parametrize("batch_size", (221, 222))
def test_dim128_two_head_split_starts_at_sixteen_waves(batch_size) -> None:
    """On B200 the two-head 128x128 build runs split two ways at five CTAs per SM from 48 two-head CTAs per SM
    (batch 222 of 64 heads in 8 groups: 7104 CTAs); batch 221 keeps the two-head build.  GB300 never splits it."""
    bf16 = torch.bfloat16
    case = ("waves16", batch_size, 64, 8, 128, 128, 0, bf16, "auto", False, None)
    plan = _route(_make_case(case))
    assert plan is not None
    if not _routes_as_measured():
        pytest.skip("the split floor is measured on B200 and GB300 only")
    b200 = cake._sm_count(plan.device_index) == 148 and plan.arch == "sm_100a"
    split = b200 and batch_size == 222
    program = "stp_dim128_hpc2_s2w4_mb5" if split else "stp_dim128_hpc2"
    assert plan.program == cake.lean_program(program, plan.arch)
    assert plan.grid == (batch_size * 8 * 4 * (2 if split else 1), 1, 1)


# (case, z, the program the route selects): canonical layouts on every paired row-major program.
_PAIRED_ROWS_CASES = (
    # Batch 48: no batch the one-token ring takes.
    (("dim128_hpc2", 48, 64, 8, 128, 128, 0), False, "stp_dim128_hpc2"),
    (("dim128_hpc1", 32, 16, 16, 128, 128, 0), False, "stp_dim128_hpc1_s2w4"),
    (("dim128_hpc1_mb4", 40, 16, 16, 128, 128, 0), False, "stp_dim128_hpc1_mb4"),
    # Batch 64: the z-gated one-token ring on B200, the persistent build on GB300.
    (
        ("dim128_z", 64, 64, 8, 128, 128, 0),
        True,
        {
            "sm_100a": "mtp_short_ring_heads64_groups8_t1_stp_z",
            "sm_103a": "stp_dim128_z_balanced",
        },
    ),
    (("dim128_z_b16", 16, 64, 8, 128, 128, 0), True, "stp_dim128_z_gated"),
    (("dim128_z_hpc1", 32, 16, 16, 128, 128, 0), True, "stp_dim128_z_gated"),
    (
        ("hd64_one_head", 4, 128, 8, 64, 128, 0),
        False,
        "stp_hd64_paired_hpc1_ratio8_s4w2",
    ),
    (("hd64_one_head_g4", 4, 128, 4, 64, 128, 0), False, "stp_hd64_paired_hpc1_s4w2"),
    (
        ("hd64_two_heads", 5, 128, 8, 64, 128, 0),
        False,
        "stp_hd64_paired_hpc2_ratio8_s2w4",
    ),
    (("hd64_two_heads_g4", 5, 128, 4, 64, 128, 0), False, "stp_hd64_paired_hpc2_s2w4"),
    (
        ("hd64_two_ctas", 8, 128, 8, 64, 128, 0),
        False,
        {
            "sm_100a": "stp_hd64_paired_hpc2_mb2",
            "sm_103a": "stp_hd64_paired_hpc2_ratio8_s2w4",
        },
    ),
    (("hd64_one_head_odd", 20, 32, 16, 64, 128, 0), False, "stp_hd64_paired_hpc1"),
    (("hd64_four_split", 10, 128, 8, 64, 128, 0), False, "stp_hd64_paired_hpc4_s2w4"),
    (("hd64_four_heads", 16, 128, 8, 64, 128, 0), False, "stp_hd64_paired_hpc4_s2w4"),
    (("hd64_four_heads_z", 16, 128, 8, 64, 128, 0), True, "stp_hd64_paired_hpc4_s2w4"),
    (("hd64_four_s4", 20, 128, 8, 64, 128, 0), False, "stp_hd64_paired_hpc4_s4w2"),
    (("hd64_two_heads_wide", 64, 128, 8, 64, 128, 0), False, "stp_hd64_paired_hpc2"),
    (("hd64_mb5", 32, 128, 8, 64, 128, 0), False, "stp_hd64_paired_hpc2_s2w4_mb5"),
    # 1760 two-head CTAs: a nearly full third wave of four CTAs per SM (c4), and 1472 in the four-head band.
    (("hd64_c4", 55, 128, 8, 64, 128, 0), False, "stp_hd64_paired_hpc4_s2w4"),
    (("hd64_h4", 46, 128, 8, 64, 128, 0), False, "stp_hd64_paired_hpc4"),
)


@requires_blackwell
@pytest.mark.parametrize(
    "case, gated, program",
    _PAIRED_ROWS_CASES,
    ids=[case[0][0] for case in _PAIRED_ROWS_CASES],
)
def test_paired_rows_routes_match_flashinfer(case, gated, program, cake_hits) -> None:
    """Every paired row-major program serves its route: same outputs and state as FlashInfer."""
    inputs = _make_case((*case, torch.bfloat16, "auto", False, None))
    if gated:
        generator = torch.Generator(device="cuda").manual_seed(1)
        inputs["z"] = (
            torch.randn(inputs["x"].shape, generator=generator, device="cuda") * 0.5
        ).to(torch.bfloat16)
    plan = _route(inputs)
    assert plan is not None
    if _routes_as_measured():
        assert plan.program == _build(program, plan)
    _assert_split_grid(plan)
    reference, candidate = _split_arms(inputs)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True], "the paired row-major route must run on Cake"
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


# Each P-split build: the unsplit build it is bitwise equal to, its heads per CTA and its CTAs per work tile.
_SPLIT_UNSPLIT = {
    "stp_hd64_paired_hpc1_ratio8_s4w2": ("stp_hd64_paired_hpc1_ratio8", 1, 4),
    "stp_hd64_paired_hpc1_s4w2": ("stp_hd64_paired_hpc1", 1, 4),
    "stp_hd64_paired_hpc2_ratio8_s2w4": ("stp_hd64_paired_hpc2", 2, 2),
    "stp_hd64_paired_hpc2_s2w4": ("stp_hd64_paired_hpc2", 2, 2),
    "stp_hd64_paired_hpc2_s2w4_mb5": ("stp_hd64_paired_hpc2", 2, 2),
    "stp_hd64_paired_hpc4_s2w4": ("stp_hd64_paired_hpc4", 4, 2),
    "stp_hd64_paired_hpc4_s4w2": ("stp_hd64_paired_hpc4", 4, 4),
    "stp_dim128_hpc1_s2w4": ("stp_dim128_hpc1_mb4", 1, 2),
    "stp_dim128_hpc2_s2w4_mb5": ("stp_dim128_hpc2", 2, 2),
}
# (nheads, ngroups, dim) each build is checked on: eight program heads per group for the folded ratio, a
# runtime ratio (two or four program heads per group) otherwise; every head ratio is one FlashInfer's
# reference kernel dispatches.
_SPLIT_HEADS = {
    "stp_hd64_paired_hpc1_ratio8_s4w2": (128, 8, 64),
    "stp_hd64_paired_hpc1_s4w2": (64, 16, 64),
    "stp_hd64_paired_hpc2_ratio8_s2w4": (128, 8, 64),
    "stp_hd64_paired_hpc2_s2w4": (128, 16, 64),
    "stp_hd64_paired_hpc2_s2w4_mb5": (128, 8, 64),
    "stp_hd64_paired_hpc4_s2w4": (128, 8, 64),
    "stp_hd64_paired_hpc4_s4w2": (128, 8, 64),
    "stp_dim128_hpc1_s2w4": (16, 16, 128),
    "stp_dim128_hpc2_s2w4_mb5": (64, 8, 128),
}
# ``z`` and the raw ABI reach the headdim-64 builds only (a gated headdim-128 call runs the z-gated build).
_SPLIT_VARIANTS = ("plain", "pad", "dst", "disable", "padded", "z", "raw")
_SPLIT_CASES = [
    (program, variant)
    for program in _SPLIT_UNSPLIT
    for variant in _SPLIT_VARIANTS
    if program.startswith("stp_hd64") or variant not in ("z", "raw")
]


def _assert_split_grid(plan) -> None:
    """A P-split plan launches ``p_split`` CTAs on every work tile of its heads per CTA."""
    split = _SPLIT_UNSPLIT.get(
        plan.program.removesuffix("_bf16_i32").removesuffix("_lean")
    )
    if split is None:
        return
    _, heads_per_cta, p_split = split
    bindings = plan.bindings
    heads_per_group = bindings["nheads"] // bindings["ngroups"]
    assert bindings["head_tiles"] == -(-heads_per_group // heads_per_cta)
    tiles = bindings["x"].shape[0] * bindings["ngroups"] * bindings["head_tiles"]
    assert plan.grid == (tiles * p_split, 1, 1)


def _padded_batch_stride(rows):
    """The ``(batch, n, row)`` tensor as dense rows at a batch stride 128 elements wider."""
    batch_size, n, row = rows.shape
    wide = torch.zeros(
        (batch_size, n * row + 128), dtype=rows.dtype, device=rows.device
    )
    view = wide[:, : n * row].view(batch_size, n, row)
    view.copy_(rows)
    return view


def _split_inputs(program, variant):
    """One P-split check: eight batch rows on the build's head geometry (enough program heads for the paired
    programs' bindings on every SM100 / SM103 part), changed as ``variant`` says."""
    nheads, ngroups, dim = _SPLIT_HEADS[program]
    batch = 8
    if variant == "raw":
        return _raw_decode_case(batch, torch.bfloat16, nheads, ngroups)
    bf16 = torch.bfloat16
    case = ("split", batch, nheads, ngroups, dim, 128, 0, bf16, "auto", False, None)
    return _vary(_make_case(case), variant)


def _vary(inputs, variant):
    """``inputs`` changed as ``variant`` says: a pad slot, a destination table into a second set of slots,
    no state update, ``x`` / ``B`` / ``C`` at a wider batch stride, or a gate ``z``."""
    batch = inputs["x"].shape[0]
    generator = torch.Generator(device="cuda").manual_seed(1)
    if variant == "pad":
        inputs["state_batch_indices"][0] = -1
    elif variant == "dst":
        # Read the batch's slots and write a second set of slots of a pool twice the batch.
        state = inputs["state"]
        written = torch.randn(state.shape, generator=generator, device="cuda") * 0.05
        inputs["state"] = torch.cat((state, written.to(state.dtype)))
        inputs["dst_state_batch_indices"] = inputs["state_batch_indices"] + batch
    elif variant == "disable":
        inputs["disable_state_update"] = True
    elif variant == "padded":
        for key in ("x", "B", "C"):
            inputs[key] = _padded_batch_stride(inputs[key])
    elif variant == "z":
        z = torch.randn(inputs["x"].shape, generator=generator, device="cuda") * 0.5
        inputs["z"] = z.to(torch.bfloat16)
    return inputs


def _delivered(program, arch):
    """The ``PROGRAMS`` key of the build a launch of ``program`` runs on ``arch`` (:func:`cake.lean_program`), or
    ``None`` when no build of it ships for that architecture."""
    name = cake.lean_program(program, arch)
    module = cake.PROGRAMS.get(name)
    if module is None or arch not in cake.MODULES[module].get("arches", [arch]):
        return None
    return name


def _launch_paired(plan, program, heads_per_cta, p_split) -> None:
    """Launch one paired row-major build on a route's bindings: ``p_split`` CTAs per work tile of
    ``heads_per_cta`` program heads of one (batch, group)."""
    bindings = plan.bindings
    ngroups = bindings["ngroups"]
    head_tiles = -(-(bindings["nheads"] // ngroups) // heads_per_cta)
    grid = (bindings["x"].shape[0] * ngroups * head_tiles * p_split, 1, 1)
    with torch.cuda.device(plan.device_index):
        cake._program(program, plan.arch, ()).launch(
            grid, dict(bindings, head_tiles=head_tiles)
        )


@requires_blackwell
@pytest.mark.parametrize(
    "program, variant", _SPLIT_CASES, ids=[f"{p}-{v}" for p, v in _SPLIT_CASES]
)
def test_split_programs_are_bitwise_the_unsplit_builds(program, variant) -> None:
    """Each P-split build writes the output and state of the unsplit build it splits, bit for bit, on the same
    bindings, and matches FlashInfer: pad slot, destination table, disabled state update, padded batch
    strides, ``z`` and the raw ABI (BF16 coefficients, int32 slot tables)."""
    unsplit, heads_per_cta, p_split = _SPLIT_UNSPLIT[program]
    inputs = _split_inputs(program, variant)
    if variant == "raw":
        program, unsplit = f"{program}_bf16_i32", f"{unsplit}_bf16_i32"
    arch = cake._device_arch(torch.cuda.current_device())
    build = _delivered(program, arch)
    if build is None:
        # Every build of the program the loader records ships for another architecture only.
        builds = [
            name for name in (program, f"{program}_lean") if name in cake.PROGRAMS
        ]
        assert builds
        for name in builds:
            with pytest.raises(ValueError, match="not delivered"):
                cake.jit_spec(cake.PROGRAMS[name], arch)
        pytest.skip(f"{program} is not delivered for {arch}")
    program, unsplit = build, _delivered(unsplit, arch)
    assert unsplit is not None
    initial_state = inputs["state"].clone()
    split_arm, unsplit_arm = _split_arms(inputs)
    if variant == "raw":
        reference = _canonical_arm(inputs)
    else:
        reference = dict(inputs, state=inputs["state"].clone())
        for key in ("x", "B", "C"):
            reference[key] = reference[key].contiguous()
    out_reference = selective_state_update(
        **reference, backend="flashinfer", pad_slot_id=-1
    )
    plans = [_route(arm) for arm in (split_arm, unsplit_arm)]
    builds = ((program, p_split), (unsplit, 1))
    for plan, (build, ctas_per_tile) in zip(plans, builds, strict=True):
        assert plan is not None and "head_tiles" in plan.bindings, (
            "the call must plan the paired programs' bindings"
        )
        _launch_paired(plan, build, heads_per_cta, ctas_per_tile)
    out_split, out_unsplit = (plan.bindings["output"] for plan in plans)
    assert torch.equal(out_split, out_unsplit)
    assert torch.equal(split_arm["state"], unsplit_arm["state"])
    torch.testing.assert_close(out_split, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        split_arm["state"], reference["state"], atol=1e-2, rtol=1e-2
    )
    if variant == "disable":
        assert torch.equal(split_arm["state"], initial_state)


# (name, batch, nheads, ngroups, the build the route takes, its heads per CTA and CTAs per work tile): launches of
# the two-head build at three CTAs per SM that the four-head rules take on B200 and GB300 (eight program heads per
# group: 1760 two-head CTAs, a nearly full third wave of four CTAs per SM; 1472 in the four-head band).
_FOUR_HEAD_CASES = (
    ("c4", 55, 128, 8, "stp_hd64_paired_hpc4_s2w4", 4, 2),
    ("c4_raw", 55, 128, 8, "stp_hd64_paired_hpc4_s2w4_bf16_i32", 4, 2),
    ("h4", 46, 128, 8, "stp_hd64_paired_hpc4", 4, 1),
    ("h4_raw", 46, 128, 8, "stp_hd64_paired_hpc4_bf16_i32", 4, 1),
)


@requires_blackwell
@pytest.mark.parametrize("case", _FOUR_HEAD_CASES, ids=lambda case: case[0])
def test_four_head_routes_are_bitwise_the_two_head_build(case) -> None:
    """The four-head builds the c4 / h4 rules route a launch to write the output and state of the two-head build
    the launch ran before, bit for bit, on the same bindings."""
    name, batch, nheads, ngroups, program, heads_per_cta, p_split = case
    if not _routes_as_measured():
        pytest.skip("the four-head bands are measured on B200 and GB300 only")
    raw = name.endswith("_raw")
    bf16 = torch.bfloat16
    if raw:
        inputs = _raw_decode_case(batch, bf16, nheads, ngroups)
    else:
        inputs = _make_case(
            (name, batch, nheads, ngroups, 64, 128, 0, bf16, "auto", False, None)
        )
    four_arm, two_arm = _split_arms(inputs)
    four_plan, two_plan = _route(four_arm), _route(two_arm)
    assert four_plan is not None and two_plan is not None
    arch = four_plan.arch
    assert four_plan.program == cake.lean_program(program, arch)
    assert four_plan.bindings["head_tiles"] == -(
        -(nheads // 2 // ngroups) // heads_per_cta
    )
    _launch_paired(four_plan, four_plan.program, heads_per_cta, p_split)
    two = cake.lean_program(f"stp_hd64_paired_hpc2{'_bf16_i32' if raw else ''}", arch)
    _launch_paired(two_plan, two, 2, 1)
    assert torch.equal(four_plan.bindings["output"], two_plan.bindings["output"])
    assert torch.equal(four_arm["state"], two_arm["state"])


@requires_blackwell
@pytest.mark.parametrize("token_steps", (4, 5, 6, 7, 8))
def test_pipelined_reduction_build_is_bitwise_the_dynamic_build(token_steps) -> None:
    """The dynamic program's pipelined-reduction build writes the output and state of the 16-lane build, bit
    for bit, on the same bindings (one batch row, a destination table), and matches FlashInfer."""
    case = ("pr", 1, 16, 1, 64, 128, token_steps, torch.float32, "simple", False, 1)
    inputs = _make_case(case)
    reference = dict(inputs, state=inputs["state"].clone())
    out_reference = selective_state_update(**reference, backend="flashinfer")
    pr_arm, plain_arm = _split_arms(inputs)
    pr_plan, plain_plan = _route(pr_arm), _route(plain_arm)
    assert pr_plan is not None and plain_plan is not None
    arch = pr_plan.arch
    assert pr_plan.program == cake.dynamic_program(1, token_steps, arch)
    with torch.cuda.device(pr_plan.device_index):
        for plan, program in (
            (pr_plan, "dynamic_lanes16_pr"),
            (plain_plan, "dynamic_lanes16"),
        ):
            cake._program(program, arch, ()).launch(plan.grid, plan.bindings)
    out_pr, out_plain = pr_plan.bindings["output"], plain_plan.bindings["output"]
    assert torch.equal(out_pr, out_plain)
    assert torch.equal(pr_arm["state"], plain_arm["state"])
    torch.testing.assert_close(out_pr, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        pr_arm["state"], reference["state"], atol=1e-2, rtol=1e-2
    )


# (batch, nheads, ngroups) of z-gated 128x128 calls inside the persistent build's domain on B200 and GB300: one
# head per CTA, several heads per CTA (a grid of three CTAs per SM), an odd batch whose heads are no multiple of
# the grid, and one head per group (head ratios FlashInfer's reference kernel dispatches).
_BALANCED_SHAPES = ((4, 64, 8), (64, 64, 8), (45, 32, 4), (300, 1, 1))


@requires_blackwell
@pytest.mark.parametrize("variant", ("plain", "pad", "dst", "disable", "padded"))
@pytest.mark.parametrize(
    "shape", _BALANCED_SHAPES, ids=["x".join(map(str, s)) for s in _BALANCED_SHAPES]
)
def test_balanced_program_is_bitwise_the_z_gated_build(
    shape, variant, monkeypatch
) -> None:
    """The persistent build writes the output and state of the z-gated build, bit for bit, on the same inputs
    with ``z`` (pad slot, destination table, disabled state update, padded batch strides), and matches
    FlashInfer; its launch hands every head of the batch to at most three CTAs per SM.  The z-gated one-token
    ring stays off its batches here (it has a test of its own)."""
    monkeypatch.setattr(cake, "_DIM128_STP_RING_Z_BATCHES", {})
    batch, nheads, ngroups = shape
    bf16 = torch.bfloat16
    case = ("balanced", batch, nheads, ngroups, 128, 128, 0, bf16, "auto", False, None)
    inputs = _vary(_make_case(case), variant)
    generator = torch.Generator(device="cuda").manual_seed(2)
    z = torch.randn((batch, nheads, 128), generator=generator, device="cuda") * 0.5
    inputs["z"] = z.to(bf16)
    if variant == "padded":
        inputs["z"] = _padded_batch_stride(inputs["z"])
    initial_state = inputs["state"].clone()
    reference = dict(inputs, state=inputs["state"].clone())
    for key in ("x", "B", "C", "z"):
        reference[key] = reference[key].contiguous()
    out_reference = selective_state_update(
        **reference, backend="flashinfer", pad_slot_id=-1
    )
    balanced_arm, gated_arm = _split_arms(inputs)
    balanced_plan, gated_plan = _route(balanced_arm), _route(gated_arm)
    assert balanced_plan is not None and gated_plan is not None
    if balanced_plan.program != cake.lean_program(
        "stp_dim128_z_balanced", balanced_plan.arch
    ):
        pytest.skip(f"the route takes {balanced_plan.program} on this device")
    units = batch * nheads
    sm_count = cake._sm_count(balanced_plan.device_index)
    assert balanced_plan.grid == (min(units, 3 * sm_count), 1, 1)
    assert balanced_plan.bindings["units"] == units
    assert "head_tiles" not in balanced_plan.bindings
    with torch.cuda.device(balanced_plan.device_index):
        cake._program(balanced_plan.program, balanced_plan.arch, ()).launch(
            balanced_plan.grid, balanced_plan.bindings
        )
    _launch_paired(gated_plan, "stp_dim128_z_gated", 2, 1)
    out_balanced = balanced_plan.bindings["output"]
    out_gated = gated_plan.bindings["output"]
    assert torch.equal(out_balanced, out_gated)
    assert torch.equal(balanced_arm["state"], gated_arm["state"])
    torch.testing.assert_close(out_balanced, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        balanced_arm["state"], reference["state"], atol=1e-2, rtol=1e-2
    )
    if variant == "disable":
        assert torch.equal(balanced_arm["state"], initial_state)


@requires_blackwell
def test_balanced_program_stays_off_the_raw_abi() -> None:
    """A call in the persistent build's domain with BF16 coefficients and int32 slot tables (the raw ABI) never
    plans the persistent build: the build reads the canonical ABI only."""
    bf16 = torch.bfloat16
    case = ("balanced", 4, 64, 8, 128, 128, 0, bf16, "auto", False, None)
    inputs = _make_case(case)
    generator = torch.Generator(device="cuda").manual_seed(2)
    z = torch.randn(inputs["x"].shape, generator=generator, device="cuda") * 0.5
    inputs["z"] = z.to(bf16)
    plan = _route(inputs)
    assert plan is not None
    balanced = cake.lean_program("stp_dim128_z_balanced", plan.arch)
    if plan.program != balanced:
        pytest.skip(f"the route takes {plan.program} on this device")
    nheads, dim = inputs["D"].shape
    raw = dict(inputs)
    dt_rows = inputs["dt"][..., 0].to(bf16).contiguous()
    raw["dt"] = dt_rows.as_strided(inputs["dt"].shape, (*dt_rows.stride(), 0))
    for key in ("D", "dt_bias"):
        rows = inputs[key][:, 0].to(bf16).contiguous()
        raw[key] = rows.as_strided((nheads, dim), (1, 0))
    raw["state_batch_indices"] = inputs["state_batch_indices"].to(torch.int32)
    raw_plan = _route(raw)
    assert raw_plan is None or raw_plan.program != balanced


@requires_blackwell
def test_hd64_split_layout_outside_the_paired_programs_runs_the_row_owner_tile(
    cake_hits,
) -> None:
    """A launch the P-split rule takes (one to two program heads per SM) whose ``D`` steps heads at stride two
    is outside the paired programs' addressing: it runs the direct row-owner tile."""
    inputs = _make_case(_case("stp_hd64_rows_bf16"))
    if _routes_as_measured():
        plan = _route(inputs)
        assert plan.program == cake.lean_program(
            "stp_hd64_paired_hpc1_ratio8_s4w2", plan.arch
        )
    nheads, dim = inputs["D"].shape
    D_rows = inputs["D"][:, 0].contiguous()
    D_wide = torch.zeros(2 * nheads, dtype=D_rows.dtype, device="cuda")
    D_wide[::2].copy_(D_rows)
    inputs["D"] = D_wide.as_strided((nheads, dim), (2, 0))
    plan = _route(inputs)
    assert plan is not None and plan.program.startswith("stp_hd64_rows_bf16")
    reference, candidate = _split_arms(inputs)
    reference["D"] = D_rows.as_strided((nheads, dim), (1, 0))
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True], "the row-owner tile must serve the call"
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


def test_hd64_rows_plan_follows_the_work() -> None:
    sm_count = 148
    assert cake.hd64_rows_plan(1, 128, torch.bfloat16, sm_count) == (8, 1)
    assert cake.hd64_rows_plan(4, 128, torch.bfloat16, sm_count) == (4, 1)
    assert cake.hd64_rows_plan(64, 24, torch.bfloat16, sm_count) is None
    assert cake.hd64_rows_plan(1, 128, torch.float32, sm_count) == (8, 1)
    assert cake.hd64_rows_plan(2, 128, torch.float32, sm_count) == (4, 1)
    assert cake.hd64_rows_plan(4, 128, torch.float32, sm_count) == (2, 4)
    assert cake.hd64_rows_plan(8, 128, torch.float32, sm_count) == (1, 4)
    assert cake.hd64_rows_plan(64, 128, torch.float32, sm_count) == (1, 4)


def test_hd64_rows_program_follows_the_bands() -> None:
    """The rendered row-owner build of each band; ``None`` keeps the carried FP32 program."""
    f32, bf16 = torch.float32, torch.bfloat16
    program = cake.hd64_rows_program
    for arch in ("sm_100a", "sm_103a"):
        assert program(bf16, False, 8, 128, arch, 128) == ("stp_hd64_rows_bf16_dt8", 4)
        assert program(bf16, True, 8, 128, arch, 128) == (
            "stp_hd64_rows_bf16_dt8_bf16_i32",
            4,
        )
        assert program(f32, False, 8, 128, arch, 128) == ("stp_hd64_rows_fp32_dt8", 4)
        assert program(f32, True, 8, 128, arch, 128) == (
            "stp_hd64_rows_fp32_dt8_bf16_i32",
            4,
        )
    # Four BF16 tiles: one build per architecture in the canonical ABI; in the raw ABI the one-division build,
    # but on sm_103a below 64 heads the x-in-all-lanes build.
    for nheads in (32, 63, 64, 128):
        assert program(bf16, False, 4, 512, "sm_100a", nheads) == (
            "stp_hd64_rows_bf16_dt4_xa_od",
            4,
        )
        assert program(bf16, False, 4, 512, "sm_103a", nheads) == (
            "stp_hd64_rows_bf16_dt4_od",
            4,
        )
        assert program(bf16, True, 4, 512, "sm_100a", nheads) == (
            "stp_hd64_rows_bf16_dt4_od_bf16_i32",
            4,
        )
    for nheads, build in ((32, "xa"), (63, "xa"), (64, "od"), (128, "od")):
        assert program(bf16, True, 4, 512, "sm_103a", nheads) == (
            f"stp_hd64_rows_bf16_dt4_{build}_bf16_i32",
            4,
        )
    for arch in ("sm_100a", "sm_103a"):
        # Four and two FP32 tiles: raw-ABI builds only; a canonical call keeps the carried program.
        assert program(f32, True, 4, 256, arch, 128) == (
            "stp_hd64_rows_fp32_dt4_bf16_i32",
            4,
        )
        assert program(f32, True, 2, 512, arch, 128) == (
            "stp_hd64_rows_fp32_dt2_bf16_i32",
            4,
        )
        assert program(f32, False, 4, 256, arch, 128) is None
        assert program(f32, False, 2, 512, arch, 128) is None
        # One tile per head, by batch-heads.
        assert program(f32, False, 1, 4096, arch, 128) == ("stp_hd64_rows_fp32_dt1", 4)
        assert program(f32, True, 1, 2048, arch, 128) == (
            "stp_hd64_rows_fp32_dt1_bf16_i32",
            2,
        )
        l2_prefetch = ("stp_hd64_rows_fp32_dt1_l2_prefetch_bf16_i32", 4)
        assert program(f32, True, 1, 2049, arch, 128) == l2_prefetch
        assert program(f32, True, 1, 4096, arch, 128) == l2_prefetch
        for raw_abi in (False, True):
            assert program(f32, raw_abi, 1, 4097, arch, 128) is None
            assert program(f32, raw_abi, 1, 8191, arch, 128) is None
        # From 8192 batch-heads: the four-tile read-ahead build in either ABI.
        assert program(f32, False, 1, 8192, arch, 128) == (
            "stp_hd64_rows_fp32_dt1_large",
            4,
        )
        assert program(f32, True, 1, 28671, arch, 128) == (
            "stp_hd64_rows_fp32_dt1_large_bf16_i32",
            4,
        )
        assert program(f32, False, 1, 40961, arch, 128) == (
            "stp_hd64_rows_fp32_dt1_large",
            4,
        )
    # sm_103a from 28672 to 40960 batch-heads: the read-ahead build that stores the state with an L2 evict-first
    # policy; sm_100a keeps the read-ahead build.
    for raw_abi, suffix in ((False, ""), (True, "_bf16_i32")):
        for batch_heads in (28672, 32768, 40960):
            assert program(f32, raw_abi, 1, batch_heads, "sm_103a", 128) == (
                f"stp_hd64_rows_fp32_dt1_large_sl2first{suffix}",
                4,
            )
            assert program(f32, raw_abi, 1, batch_heads, "sm_100a", 128) == (
                f"stp_hd64_rows_fp32_dt1_large{suffix}",
                4,
            )


def test_cache_program_follows_the_projection_layout() -> None:
    """The three prebuilt projection layouts run their rendered builds, the canonical ones on exactly 64 heads of
    the layout's group count with dense slots the folded builds; any other layout keeps the carried program."""

    def layout(raw_abi, nheads, ngroups, x_stride, dt_stride, bc_stride):
        coefficient = torch.bfloat16 if raw_abi else torch.float32
        index = torch.int32 if raw_abi else torch.int64
        return cake.cache_defines(
            coefficient,
            index,
            nheads,
            ngroups,
            x_stride,
            dt_stride,
            bc_stride,
            bc_stride,
        )

    def program(defines, nheads=64, ngroups=1, dense_slots=True):
        return cake.mtp_cache_program(
            defines, nheads=nheads, ngroups=ngroups, dense_slots=dense_slots
        )

    groups1 = layout(False, 64, 1, (24576, 4096), (384, 64), (768, 128))
    groups8 = layout(False, 64, 8, (24576, 4096), (384, 64), (6144, 1024))
    raw = layout(True, 64, 1, (26112, 4352), (51072, 8512), (26112, 4352))
    assert program(groups1) == ("mtp_cache_groups1_heads64", ())
    assert program(groups8, ngroups=8) == ("mtp_cache_groups8_heads64", ())
    assert program(raw) == ("mtp_cache_bf16_i32", ())
    assert program(raw, dense_slots=False) == ("mtp_cache_bf16_i32", ())
    # The canonical layouts off the folded counts or slots keep the layout's build: fewer heads in a 64-head
    # projection, fewer groups in an 8-group one, a slot stride other than the dense one.
    assert program(groups1, nheads=32) == ("mtp_cache_groups1", ())
    assert program(groups8, nheads=32, ngroups=8) == ("mtp_cache_groups8", ())
    assert program(groups8, ngroups=4) == ("mtp_cache_groups8", ())
    assert program(groups1, dense_slots=False) == ("mtp_cache_groups1", ())
    assert program(groups8, ngroups=8, dense_slots=False) == ("mtp_cache_groups8", ())
    # They are the carried program's prebuilt instantiations.
    listed = cake.MODULES[cake.PROGRAMS["mtp_cache_c4_t6"]]["instantiations"]
    assert {cake.define_digest(d) for d in (groups1, groups8, raw)} == set(listed)
    for defines, nheads, ngroups in (
        (layout(False, 64, 1, (26112, 4352), (384, 64), (768, 128)), 64, 1),  # padded x
        (layout(False, 32, 1, (12288, 2048), (192, 32), (768, 128)), 32, 1),  # 32 heads
        (
            layout(False, 64, 4, (24576, 4096), (384, 64), (3072, 512)),
            64,
            4,
        ),  # 4 groups
        (layout(True, 64, 2, (26112, 4352), (51072, 8512), (26112, 4352)), 64, 2),
    ):
        assert program(defines, nheads, ngroups) == ("mtp_cache_c4_t6", defines)


def _case(name):
    return next(case for case in _CASES if case[0] == name)


def _route(inputs, output_like=None):
    """``plan_route`` of one normalized input set (the keyword arguments the public API forwards)."""
    forwarded = (
        "state",
        "x",
        "dt",
        "A",
        "B",
        "C",
        "D",
        "dt_bias",
        "state_batch_indices",
        "dst_state_batch_indices",
        "disable_state_update",
        "intermediate_states_buffer",
        "intermediate_state_indices",
        "cache_steps",
        "algorithm",
        "dt_softplus",
    )
    return cake.plan_route(
        **{key: inputs[key] for key in forwarded},
        z=inputs.get("z"),
        output=torch.empty(
            tuple((inputs["x"] if output_like is None else output_like).shape),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        pad_slot_id=-1,
        state_scale=None,
        intermediate_state_scales=None,
        rand_seed=None,
        cu_seqlens=None,
        num_accepted_tokens=None,
    )


@requires_blackwell
def test_dynamic_route_falls_back_for_non_contiguous_inputs(cake_hits) -> None:
    """A padded projection view reaches FlashInfer instead of the binding's contiguity check."""
    inputs = _make_case(_case("dynamic0_b8"))
    assert _route(inputs).program == "dynamic_lanes8"
    batch_size, token_steps, nheads, dim = inputs["x"].shape
    padded = torch.zeros(
        (batch_size, token_steps + 1, nheads, dim), dtype=torch.bfloat16, device="cuda"
    )
    padded[:, :token_steps].copy_(inputs["x"])
    inputs["x"] = padded[:, :token_steps]
    assert not inputs["x"].is_contiguous()
    # The shapes still plan the launch; the binding rejects the padded rows before launching.
    assert _route(inputs).program == "dynamic_lanes8"
    reference, candidate = _split_arms(inputs)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [False], "non-contiguous inputs must fall back to FlashInfer"
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
def test_dynamic_route_falls_back_for_mismatched_shapes() -> None:
    """Shapes the fixed-index program does not index fall back before the launch."""
    inputs = _make_case(_case("dynamic0_b8"))
    batch_size, token_steps, nheads, dim = inputs["x"].shape
    half_heads = torch.zeros(
        (batch_size, token_steps, nheads // 2, dim), dtype=torch.bfloat16, device="cuda"
    )
    assert _route(dict(inputs, x=half_heads)) is None
    assert _route(dict(inputs, B=inputs["B"][..., :64].contiguous())) is None


def _padded_steps(x):
    """``x`` as a view with a padded batch stride (one spare token step per sequence): same shape, not contiguous."""
    batch_size, token_steps, nheads, dim = x.shape
    storage = torch.zeros(
        (batch_size, token_steps + 1, nheads, dim), dtype=x.dtype, device="cuda"
    )
    storage[:, :token_steps].copy_(x)
    return storage[:, :token_steps]


def _assert_falls_back(inputs, cake_hits):
    """The public API hands ``inputs`` to FlashInfer (no Cake launch) and both backends agree."""
    reference, candidate = _split_arms(inputs)
    try:
        out_reference = selective_state_update(**reference, backend="flashinfer")
    except Exception as error:  # FlashInfer rejects the layout too: the Cake backend must surface the same rejection
        with pytest.raises(type(error)):
            cake_selective_state_update(**candidate)
    else:
        out_candidate = cake_selective_state_update(**candidate)
        _assert_arms_close(candidate, out_candidate, reference, out_reference)
    assert cake_hits == [False], "the layout must fall back to FlashInfer"


@requires_blackwell
def test_identity_route_falls_back_for_non_broadcast_coefficients(cake_hits) -> None:
    """A dense A (one value per element instead of one per head) never reaches the 128x128 FP32 program."""
    inputs = _make_case(_case("stp_fp32"))
    assert _route(inputs).program == "stp_dim128_fp32_l2_prefetch"
    inputs["A"] = inputs["A"].contiguous()
    assert _route(inputs) is None
    _assert_falls_back(inputs, cake_hits)


@requires_blackwell
def test_identity_route_falls_back_for_non_contiguous_inputs(cake_hits) -> None:
    inputs = _make_case(_case("stp_fp32"))
    batch_size, nheads, dim = inputs["x"].shape
    storage = torch.zeros(
        (batch_size, 2, nheads, dim), dtype=torch.bfloat16, device="cuda"
    )
    storage[:, 0].copy_(inputs["x"])
    inputs["x"] = storage[:, 0]
    assert not inputs["x"].is_contiguous()
    assert _route(inputs) is None  # the route serves contiguous rows only
    _assert_falls_back(inputs, cake_hits)


# (batch x nheads, build): both edges of the 128x128 FP32 program's L2 read-ahead window.
_DIM128_FP32_WINDOW_EDGES = (
    (1407, "stp_dim128_fp32"),
    (1408, "stp_dim128_fp32_l2_prefetch"),
    (6656, "stp_dim128_fp32_l2_prefetch"),
    (6657, "stp_dim128_fp32"),
)


@pytest.mark.parametrize("batch_heads, program", _DIM128_FP32_WINDOW_EDGES)
def test_dim128_fp32_program_follows_the_l2_prefetch_window(
    batch_heads, program
) -> None:
    assert cake.dim128_fp32_program(batch_heads) == program


@requires_blackwell
@pytest.mark.parametrize("batch_size", (21, 22, 104, 105, 112))
def test_fp32_route_takes_the_l2_prefetch_build_inside_its_window(
    batch_size, cake_hits
) -> None:
    """A 128x128 FP32 single-token call of 64 heads runs the L2 read-ahead build from 1408 to 6656 batch-heads
    (batches 22 to 104) and the plain build outside them (batch 21: 1344; 105: 6720; 112: 7168), on the same
    grid, and matches FlashInfer."""
    fp32 = torch.float32
    case = ("fp32_window", batch_size, 64, 8, 128, 128, 0, fp32, "auto", False, None)
    inputs = _make_case(case)
    plan = _route(inputs)
    if batch_size * 64 < 8 * cake._sm_count(torch.cuda.current_device()):
        pytest.skip("below the FP32 program's eight batch-heads per SM")
    assert plan is not None
    inside = 22 <= batch_size <= 104
    program = "stp_dim128_fp32_l2_prefetch" if inside else "stp_dim128_fp32"
    assert (plan.program, plan.grid) == (program, (8, 64, batch_size))
    reference, candidate = _split_arms(inputs)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True]
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
@pytest.mark.parametrize("variant", ("plain", "pad"))
def test_fp32_l2_prefetch_build_is_bitwise_the_plain_build(variant) -> None:
    """The L2 read-ahead build writes the output and state of the plain 128x128 FP32 build, bit for bit, on the
    same bindings (a pad slot keeps its state)."""
    inputs = _make_case(_case("stp_fp32"))
    if variant == "pad":
        inputs["state_batch_indices"][3] = -1
    prefetch_arm, plain_arm = _split_arms(inputs)
    plans = [_route(arm) for arm in (prefetch_arm, plain_arm)]
    assert plans[0] is not None and plans[1] is not None
    builds = ("stp_dim128_fp32_l2_prefetch", "stp_dim128_fp32")
    with torch.cuda.device(plans[0].device_index):
        for plan, build in zip(plans, builds, strict=True):
            cake._program(build, plan.arch, ()).launch(plan.grid, plan.bindings)
    outputs = [plan.bindings["output"] for plan in plans]
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(prefetch_arm["state"], plain_arm["state"])


@requires_blackwell
@pytest.mark.parametrize(
    "case_name, program",
    [
        (
            "mtp_short2",
            {
                "sm_100a": "mtp_short_ring_heads64_groups8_t2_jr",
                "sm_103a": "mtp_short_ring_heads64_groups8_t2_jr_hb",
            },
        ),
        ("mtp_short2_b16", "mtp_short_prefetch_heads64_groups8"),
    ],
)
def test_short_route_falls_back_for_non_contiguous_inputs(
    case_name, program, cake_hits
) -> None:
    inputs = _make_case(_case(case_name))
    plan = _route(inputs)
    planned = plan.program
    if isinstance(program, dict):
        program = program[plan.arch]
    if _routes_as_measured():
        assert planned == program
    inputs["x"] = _padded_steps(inputs["x"])
    assert not inputs["x"].is_contiguous()
    # Planned; the binding rejects the padded rows.
    assert _route(inputs).program == planned
    _assert_falls_back(inputs, cake_hits)


# (batch, tokens, SMs, architecture, ring): the short program's TMA-ring domain on 64 heads of 8 groups.
_JR_RING = "mtp_short_ring_heads64_groups8_t2_jr"
# (batch, tokens, SMs, architecture, ring): the ring build the call runs (``None``: the build folding 64 heads of 8
# groups).
_SHORT_RING_DOMAIN_CASES = (
    # sm_100a, one token: batches 32 to 96, the last wave of 296 eight-head CTAs at least half full.
    (31, 1, 148, "sm_100a", None),
    (32, 1, 148, "sm_100a", "t1"),
    (33, 1, 148, "sm_100a", "t1"),  # a joint-reduction batch of two tokens
    (37, 1, 148, "sm_100a", "t1"),  # one whole wave
    (38, 1, 148, "sm_100a", None),
    (55, 1, 148, "sm_100a", None),
    (56, 1, 148, "sm_100a", "t1"),
    (96, 1, 148, "sm_100a", "t1"),
    (97, 1, 148, "sm_100a", None),
    # sm_100a, two tokens: from batch 32, the last wave at least a quarter full; the joint-reduction ring at its
    # measured batches.
    (31, 2, 148, "sm_100a", None),
    (46, 2, 148, "sm_100a", None),
    (47, 2, 148, "sm_100a", "t2"),
    (192, 2, 148, "sm_100a", None),
    (256, 2, 148, "sm_100a", "t2_jr"),
    (257, 2, 148, "sm_100a", "t2"),
    # The same batches on another SM count of sm_100a: the two-token ring.
    (256, 2, 152, "sm_100a", "t2"),
    (64, 2, 132, "sm_100a", "t2"),
    # sm_103a, one token: batches 64 to 96, the last wave of 304 CTAs at least half full.
    (63, 1, 152, "sm_103a", None),
    (64, 1, 152, "sm_103a", "t1"),
    (76, 1, 152, "sm_103a", "t1"),  # two whole waves
    (77, 1, 152, "sm_103a", None),
    (94, 1, 152, "sm_103a", None),
    (95, 1, 152, "sm_103a", "t1"),  # exactly half
    (97, 1, 152, "sm_103a", None),
    # sm_103a, two tokens: every batch from 32; the joint-reduction ring with the per-head state base at its
    # measured batches, the two-token ring at the others.
    (31, 2, 152, "sm_103a", None),
    (32, 2, 152, "sm_103a", "t2_jr_hb"),
    (37, 2, 152, "sm_103a", "t2_jr_hb"),
    (38, 2, 152, "sm_103a", "t2_jr_hb"),
    (47, 2, 152, "sm_103a", "t2"),
    (75, 2, 152, "sm_103a", "t2"),
    (77, 2, 152, "sm_103a", "t2_jr_hb"),
    (256, 2, 152, "sm_103a", "t2_jr_hb"),
    (319, 2, 152, "sm_103a", "t2"),
    (320, 2, 152, "sm_103a", "t2_jr_hb"),
    (321, 2, 152, "sm_103a", "t2"),
    (4096, 2, 152, "sm_103a", "t2"),
    # The same batches on another SM count of sm_103a: the two-token ring.
    (32, 2, 148, "sm_103a", "t2"),
    (320, 2, 132, "sm_103a", "t2"),
)


@pytest.mark.parametrize(
    "batch_size, token_steps, sm_count, arch, ring", _SHORT_RING_DOMAIN_CASES
)
def test_short_program_follows_the_ring_domain(
    batch_size, token_steps, sm_count, arch, ring
) -> None:
    program, head_tiles = cake.mtp_short_program(
        batch_size, 64, 8, token_steps, sm_count, arch
    )
    if ring:
        assert (program, head_tiles) == (f"mtp_short_ring_heads64_groups8_{ring}", 1)
    else:
        assert (program, head_tiles) == ("mtp_short_prefetch_heads64_groups8", 0)
    for nheads, ngroups in ((32, 8), (64, 4), (128, 8)):
        other = cake.mtp_short_program(
            batch_size, nheads, ngroups, token_steps, sm_count, arch
        )
        assert other == ("mtp_short_prefetch", 0)


@requires_blackwell
@pytest.mark.parametrize("token_steps", (1, 2))
@pytest.mark.parametrize("variant", ("plain", "pad"))
def test_ring_programs_are_bitwise_the_pass_build(token_steps, variant) -> None:
    """Each TMA-ring build writes the output and state of the pass build folding 64 heads of 8 groups, bit for
    bit, on the same inputs (a pad slot keeps its state and writes a zero output), and matches FlashInfer."""
    inputs = _make_case(_case(f"mtp_short{token_steps}"))
    if variant == "pad":
        inputs["state_batch_indices"][3] = -1
    reference = dict(inputs, state=inputs["state"].clone())
    out_reference = selective_state_update(
        **reference, backend="flashinfer", pad_slot_id=-1
    )
    ring_arm, pass_arm = _split_arms(inputs)
    ring_plan, pass_plan = _route(ring_arm), _route(pass_arm)
    if not ring_plan.program.startswith("mtp_short_ring"):
        pytest.skip(f"the route takes {ring_plan.program} on this device")
    batch_size = inputs["x"].shape[0]
    assert ring_plan.grid == (batch_size * 8, 1, 1)
    assert ring_plan.bindings["head_tiles"] == 1
    with torch.cuda.device(ring_plan.device_index):
        cake._program(ring_plan.program, ring_plan.arch, ()).launch(
            ring_plan.grid, ring_plan.bindings
        )
        bindings = {
            key: value
            for key, value in pass_plan.bindings.items()
            if key not in ("state_tma", "head_tiles")
        }
        state = pass_arm["state"]
        bindings.update(state=state, state_stride_slot=state.stride(0))
        cake._program("mtp_short_prefetch_heads64_groups8", pass_plan.arch, ()).launch(
            (batch_size * 64, 1, 1), bindings
        )
    out_ring, out_pass = ring_plan.bindings["output"], pass_plan.bindings["output"]
    assert torch.equal(out_ring, out_pass)
    assert torch.equal(ring_arm["state"], pass_arm["state"])
    torch.testing.assert_close(out_ring, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        ring_arm["state"], reference["state"], atol=1e-2, rtol=1e-2
    )


@requires_blackwell
def test_short_route_falls_back_for_mismatched_shapes() -> None:
    inputs = _make_case(_case("mtp_short2"))
    assert _route(dict(inputs, B=inputs["B"][..., :64].contiguous())) is None
    batch_size, token_steps, nheads, dim = inputs["x"].shape
    half_heads = torch.zeros(
        (batch_size, token_steps, nheads // 2, dim), dtype=torch.bfloat16, device="cuda"
    )
    assert _route(dict(inputs, x=half_heads)) is None


# The two-token ring reducing both tokens' projections together, which two-token calls of 64 heads in 8 groups
# take at the measured batches of sm_100a at 148 SMs: every batch of the set and its neighbours.
_JR_BATCH_LIST = (
    "33-35, 51-65, 68, 86, 88-101, 104, 107-108, 110-111, 122-138, 140-141, 158-168, 170, "
    "172-178, 195-202, 204-206, 232-237, 239-245, 248, 254-256, 270, 285, 296, 306"
)
_JR_BATCHES = frozenset(
    batch
    for part in _JR_BATCH_LIST.split(", ")
    for batch in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)
)


def test_jr_ring_follows_the_measured_batches() -> None:
    """On sm_100a at 148 SMs a two-token call of 64 heads in 8 groups runs the joint-reduction ring exactly at its
    109 measured batches and the two-token ring at every other batch the ring serves; nothing else changes."""
    assert len(_JR_BATCHES) == 109
    for batch_size in range(1, 4097):
        program, head_tiles = cake.mtp_short_program(
            batch_size, 64, 8, 2, 148, "sm_100a"
        )
        if batch_size in _JR_BATCHES:
            assert (program, head_tiles) == (_JR_RING, 1), batch_size
        else:
            assert program != _JR_RING, batch_size


def test_jr_ring_stays_off_other_devices_and_token_counts() -> None:
    """The joint-reduction batches are measured on sm_100a at 148 SMs: another architecture or SM count, one token,
    or another head or group count never runs it."""
    devices = (("sm_103a", 152), ("sm_103a", 148), ("sm_100a", 152), ("sm_100a", 132))
    for batch_size in range(1, 4097):
        for arch, sm_count in devices:
            for token_steps in (1, 2):
                program = cake.mtp_short_program(
                    batch_size, 64, 8, token_steps, sm_count, arch
                )[0]
                assert program != _JR_RING
        assert (
            cake.mtp_short_program(batch_size, 64, 8, 1, 148, "sm_100a")[0] != _JR_RING
        )
        for nheads, ngroups in ((32, 8), (64, 4), (128, 8)):
            program = cake.mtp_short_program(
                batch_size, nheads, ngroups, 2, 148, "sm_100a"
            )[0]
            assert program != _JR_RING


# The sm_103a joint-reduction ring with each consumer's state base set before the head's first wait, which
# two-token calls of 64 heads in 8 groups take at the measured batches of sm_103a at 152 SMs.
_JR_HB_RING = "mtp_short_ring_heads64_groups8_t2_jr_hb"
_JR_HB_BATCH_LIST = (
    "32-46, 50, 54-74, 76-78, 81, 83-106, 108, 110, 112-118, 123-148, 154, "
    "156-184, 186-188, 190-256, 260, 266, 270, 280, 285, 296, 304, 320"
)
_JR_HB_BATCHES = frozenset(
    batch
    for part in _JR_HB_BATCH_LIST.split(", ")
    for batch in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)
)


def test_jr_hb_ring_follows_the_measured_batches() -> None:
    """On sm_103a at 152 SMs a two-token call of 64 heads in 8 groups runs the joint-reduction ring with the
    per-head state base exactly at its 208 measured batches and the two-token ring at every other batch the ring
    serves; nothing else changes."""
    assert len(_JR_HB_BATCHES) == 208
    for batch_size in range(1, 4097):
        program, head_tiles = cake.mtp_short_program(
            batch_size, 64, 8, 2, 152, "sm_103a"
        )
        if batch_size in _JR_HB_BATCHES:
            assert (program, head_tiles) == (_JR_HB_RING, 1), batch_size
        elif batch_size >= 32:
            assert (program, head_tiles) == (
                "mtp_short_ring_heads64_groups8_t2",
                1,
            ), batch_size


def test_jr_hb_ring_stays_off_other_devices_and_token_counts() -> None:
    """The batches are measured on sm_103a at 152 SMs: another architecture or SM count, one token, or another
    head or group count never runs the joint-reduction ring with the per-head state base."""
    devices = (("sm_100a", 148), ("sm_100a", 152), ("sm_103a", 148), ("sm_103a", 132))
    for batch_size in range(1, 4097):
        for arch, sm_count in devices:
            for token_steps in (1, 2):
                program = cake.mtp_short_program(
                    batch_size, 64, 8, token_steps, sm_count, arch
                )[0]
                assert program != _JR_HB_RING
        assert (
            cake.mtp_short_program(batch_size, 64, 8, 1, 152, "sm_103a")[0]
            != _JR_HB_RING
        )
        for nheads, ngroups in ((32, 8), (64, 4), (128, 8)):
            program = cake.mtp_short_program(
                batch_size, nheads, ngroups, 2, 152, "sm_103a"
            )[0]
            assert program != _JR_HB_RING


def _jr_case(batch_size):
    bf16 = torch.bfloat16
    return _make_case(
        ("mtp_short2", batch_size, 64, 8, 128, 128, 2, bf16, "auto", False, None)
    )


@requires_blackwell
@pytest.mark.parametrize(
    "batch_size", (32, 33, 34, 38, 64, 66, 296, 306, 307, 320, 321)
)
def test_jr_ring_route_takes_the_measured_batches(batch_size, cake_hits) -> None:
    """A two-token call of 64 heads in 8 groups runs the device's joint-reduction ring (one CTA per batch row and
    group) at its measured batches, the two-token ring at the ring's other batches, and matches FlashInfer."""
    inputs = _jr_case(batch_size)
    plan = _route(inputs)
    assert plan is not None
    sm_count = cake._sm_count(plan.device_index)
    expected = cake.mtp_short_program(batch_size, 64, 8, 2, sm_count, plan.arch)[0]
    assert plan.program == expected
    if (plan.arch, sm_count) == ("sm_100a", 148):
        assert (plan.program == _JR_RING) is (batch_size in _JR_BATCHES)
    if (plan.arch, sm_count) == ("sm_103a", 152):
        assert (plan.program == _JR_HB_RING) is (batch_size in _JR_HB_BATCHES)
    if plan.program in (_JR_RING, _JR_HB_RING):
        assert plan.grid == (batch_size * 8, 1, 1)
        assert (plan.bindings["head_tiles"], plan.bindings["token_steps"]) == (1, 2)
    reference, candidate = _split_arms(inputs)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True]
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
@pytest.mark.parametrize("variant", ("plain", "pad"))
@pytest.mark.parametrize("batch_size", (32, 33, 34, 38, 64, 296, 320))
def test_jr_ring_is_bitwise_the_two_token_ring(
    batch_size, variant, monkeypatch
) -> None:
    """The device's joint-reduction ring writes the output and state of the two-token ring bit for bit on the
    same inputs (a pad slot keeps its state and writes a zero output), and matches FlashInfer."""
    inputs = _jr_case(batch_size)
    if variant == "pad":
        inputs["state_batch_indices"][3] = -1
    reference = dict(inputs, state=inputs["state"].clone())
    out_reference = selective_state_update(
        **reference, backend="flashinfer", pad_slot_id=-1
    )
    jr_arm, ring_arm = _split_arms(inputs)
    jr_plan = _route(jr_arm)
    if jr_plan is None or jr_plan.program not in (_JR_RING, _JR_HB_RING):
        pytest.skip("the route takes no joint-reduction ring on this device")
    monkeypatch.setattr(cake, "_MTP_SHORT_RING_JR_BATCHES", {})
    ring_plan = _route(ring_arm)
    assert ring_plan is not None
    assert ring_plan.program == "mtp_short_ring_heads64_groups8_t2"
    assert ring_plan.grid == jr_plan.grid
    with torch.cuda.device(jr_plan.device_index):
        for plan in (jr_plan, ring_plan):
            cake._program(plan.program, plan.arch, plan.defines).launch(
                plan.grid, plan.bindings
            )
    out_jr, out_ring = jr_plan.bindings["output"], ring_plan.bindings["output"]
    assert torch.equal(out_jr, out_ring)
    assert torch.equal(jr_arm["state"], ring_arm["state"])
    torch.testing.assert_close(out_jr, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        jr_arm["state"], reference["state"], atol=1e-2, rtol=1e-2
    )


# The one-token ring in the single-token programs' arithmetic, which BF16 single-token calls of 64 heads in 8
# groups take at the measured batches.
_STP_RING = "mtp_short_ring_heads64_groups8_t1_stp"
# (batch, ring): both edges of every batch interval in which such a call runs the ring on sm_100a at 148 SMs.
_STP_RING_EDGES = (
    (15, False),
    (16, True),
    (18, True),
    (19, False),
    (27, False),
    (28, True),
    (29, True),
    (30, False),
    (31, True),
    (37, True),
    (38, False),
    (55, False),
    (56, True),
    (67, True),
    (68, False),
    (69, False),
    (70, True),
    (71, True),
    (72, False),
)


@pytest.mark.parametrize("batch_size, ring", _STP_RING_EDGES)
def test_stp_ring_follows_the_measured_batches(batch_size, ring) -> None:
    assert cake.dim128_stp_ring(batch_size, 64, 8, 148, "sm_100a") is ring


def test_stp_ring_stays_off_other_devices_and_head_counts() -> None:
    """The ring's batches are measured on sm_100a at 148 SMs: another architecture or SM count, or another head
    or group count, never runs it."""
    devices = (("sm_103a", 152), ("sm_103a", 148), ("sm_100a", 152), ("sm_100a", 132))
    shapes = ((32, 8), (64, 4), (64, 16), (128, 8), (128, 16))
    for batch_size in range(1, 4097):
        for arch, sm_count in devices:
            assert not cake.dim128_stp_ring(batch_size, 64, 8, sm_count, arch)
        for nheads, ngroups in shapes:
            assert not cake.dim128_stp_ring(batch_size, nheads, ngroups, 148, "sm_100a")


def _stp_ring_case(batch_size):
    bf16 = torch.bfloat16
    return _make_case(
        ("stp_ring", batch_size, 64, 8, 128, 128, 0, bf16, "auto", False, None)
    )


@requires_blackwell
@pytest.mark.parametrize("batch_size, ring", _STP_RING_EDGES)
def test_stp_ring_route_takes_the_measured_batches(batch_size, ring, cake_hits) -> None:
    """A canonical BF16 single-token call of 64 heads in 8 groups runs the one-token ring (one CTA per batch row
    and group) at the device's measured batches, a single-token program at the others, and matches
    FlashInfer."""
    inputs = _stp_ring_case(batch_size)
    plan = _route(inputs)
    assert plan is not None
    sm_count = cake._sm_count(plan.device_index)
    if cake.dim128_stp_ring(batch_size, 64, 8, sm_count, plan.arch):
        assert (plan.program, plan.grid) == (_STP_RING, (batch_size * 8, 1, 1))
        assert (plan.bindings["head_tiles"], plan.bindings["token_steps"]) == (1, 1)
    else:
        assert plan.program != _STP_RING
    if (plan.arch, sm_count) == ("sm_100a", 148):
        assert (plan.program == _STP_RING) is ring
    reference, candidate = _split_arms(inputs)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True]
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
@pytest.mark.parametrize("variant", ("plain", "pad", "softplus"))
@pytest.mark.parametrize("batch_size", (16, 33, 64))
def test_stp_ring_is_bitwise_the_single_token_build(
    batch_size, variant, monkeypatch
) -> None:
    """The one-token ring writes the output and state of the single-token program the call runs off the ring's
    batches, bit for bit, on the same inputs (a pad slot keeps its state; softplus), and matches FlashInfer."""
    inputs = _stp_ring_case(batch_size)
    if variant == "pad":
        inputs["state_batch_indices"][3] = -1
    elif variant == "softplus":
        inputs["dt_softplus"] = True
    reference = dict(inputs, state=inputs["state"].clone())
    out_reference = selective_state_update(
        **reference, backend="flashinfer", pad_slot_id=-1
    )
    ring_arm, rows_arm = _split_arms(inputs)
    ring_plan = _route(ring_arm)
    if ring_plan is None or ring_plan.program != _STP_RING:
        pytest.skip("the route takes no one-token ring on this device")
    monkeypatch.setattr(cake, "_DIM128_STP_RING_BATCHES", {})
    rows_plan = _route(rows_arm)
    assert rows_plan is not None and rows_plan.program != _STP_RING
    with torch.cuda.device(ring_plan.device_index):
        for plan in (ring_plan, rows_plan):
            cake._program(plan.program, plan.arch, plan.defines).launch(
                plan.grid, plan.bindings
            )
    out_rows = rows_plan.bindings["output"]
    out_ring = ring_plan.bindings["output"].view(out_rows.shape)
    assert torch.equal(out_ring, out_rows)
    assert torch.equal(ring_arm["state"], rows_arm["state"])
    torch.testing.assert_close(out_ring, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        ring_arm["state"], reference["state"], atol=1e-2, rtol=1e-2
    )


@requires_blackwell
@pytest.mark.parametrize("variant", ("z", "dst", "disable", "padded", "dense_dt"))
def test_stp_ring_route_keeps_the_single_token_programs_off_its_layout(
    variant,
) -> None:
    """A call at a ring batch that the ring does not read -- a gate ``z``, a destination table, no state update,
    ``x`` / ``B`` / ``C`` at a wider batch stride, one ``dt`` per element -- never plans the ring."""
    inputs = _stp_ring_case(64)
    plan = _route(inputs)
    if plan is None or plan.program != _STP_RING:
        pytest.skip("the route takes no one-token ring on this device")
    if variant == "dense_dt":
        inputs["dt"] = inputs["dt"].contiguous()
    else:
        inputs = _vary(inputs, variant)
    plan = _route(inputs)
    assert plan is None or plan.program != _STP_RING


# The same ring with the gate ``z`` (64 heads in 8 groups): BF16 single-token calls of that shape with ``z`` take
# it at the measured batch on sm_100a at 148 SMs.
_STP_RING_Z = "mtp_short_ring_heads64_groups8_t1_stp_z"


def test_stp_ring_programs_follow_the_measured_shapes() -> None:
    """The z-gated ring runs 64 heads in 8 groups with ``z`` at batch 64, on sm_100a at 148 SMs alone; a call of
    64 heads without ``z`` keeps the 64-head ring's intervals."""
    devices = (("sm_103a", 152), ("sm_103a", 148), ("sm_100a", 152), ("sm_100a", 132))
    for batch_size in range(1, 4097):
        program = cake.dim128_stp_ring_program
        gated = program(batch_size, 64, 8, 148, "sm_100a", True)
        assert gated == (_STP_RING_Z if batch_size == 64 else None)
        ring = cake.dim128_stp_ring(batch_size, 64, 8, 148, "sm_100a")
        assert program(batch_size, 64, 8, 148, "sm_100a", False) == (
            _STP_RING if ring else None
        )
        for arch, sm_count in devices:
            assert program(batch_size, 64, 8, sm_count, arch, False) is None
            assert program(batch_size, 64, 8, sm_count, arch, True) is None
        for nheads, ngroups in ((128, 8), (128, 16), (128, 4), (32, 8), (64, 4)):
            assert program(batch_size, nheads, ngroups, 148, "sm_100a", False) is None
        for nheads, ngroups in ((128, 8), (32, 8), (64, 16)):
            assert program(batch_size, nheads, ngroups, 148, "sm_100a", True) is None


def _stp_ring_gated_case(variant):
    """A canonical BF16 single-token call at the z-gated ring's measured batch: 64 rows of 64 heads in 8 groups
    with ``z``; ``variant`` a pad slot or softplus."""
    bf16 = torch.bfloat16
    inputs = _make_case(("stp_ring", 64, 64, 8, 128, 128, 0, bf16, "auto", False, None))
    inputs = _vary(inputs, "z")
    if variant == "pad":
        inputs["state_batch_indices"][3] = -1
    elif variant == "softplus":
        inputs["dt_softplus"] = True
    return inputs


@requires_blackwell
@pytest.mark.parametrize("variant", ("plain", "pad", "softplus"))
def test_stp_ring_z_is_bitwise_the_single_token_build(
    variant, monkeypatch, cake_hits
) -> None:
    """The z-gated ring writes the output and state of the single-token program the call runs off its batches,
    bit for bit, on the same inputs (a pad slot keeps its state; softplus), every output position written, and
    matches FlashInfer."""
    program, table = _STP_RING_Z, "_DIM128_STP_RING_Z_BATCHES"
    inputs = _stp_ring_gated_case(variant)
    reference = dict(inputs, state=inputs["state"].clone())
    out_reference = selective_state_update(
        **reference, backend="flashinfer", pad_slot_id=-1
    )
    ring_arm, rows_arm = _split_arms(inputs)
    ring_plan = _route(ring_arm)
    if ring_plan is None or ring_plan.program != program:
        pytest.skip("the route takes no z-gated one-token ring on this device")
    heads_per_group = inputs["x"].shape[1] // 8
    assert ring_plan.grid == (inputs["x"].shape[0] * 8 * (heads_per_group // 8), 1, 1)
    assert ring_plan.bindings["token_steps"] == 1
    measured = getattr(cake, table)
    monkeypatch.setattr(cake, table, {})
    rows_plan = _route(rows_arm)
    assert rows_plan is not None and rows_plan.program != program
    for plan in (ring_plan, rows_plan):
        plan.bindings["output"].fill_(float("nan"))
    with torch.cuda.device(ring_plan.device_index):
        for plan in (ring_plan, rows_plan):
            cake._program(plan.program, plan.arch, plan.defines).launch(
                plan.grid, plan.bindings
            )
    out_rows = rows_plan.bindings["output"]
    out_ring = ring_plan.bindings["output"].view(out_rows.shape)
    assert not out_ring.isnan().any(), "the ring left output positions unwritten"
    assert torch.equal(out_ring, out_rows)
    assert torch.equal(ring_arm["state"], rows_arm["state"])
    torch.testing.assert_close(out_ring, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        ring_arm["state"], reference["state"], atol=1e-2, rtol=1e-2
    )
    monkeypatch.setattr(cake, table, measured)
    candidate = dict(inputs, state=inputs["state"].clone())
    out_candidate = cake_selective_state_update(**candidate, pad_slot_id=-1)
    assert cake_hits == [True]
    assert torch.equal(out_candidate, out_ring.view(out_candidate.shape))
    assert torch.equal(candidate["state"], ring_arm["state"])


@requires_blackwell
@pytest.mark.parametrize("variant", ("dst", "disable", "padded", "padded_z"))
def test_stp_ring_z_route_keeps_the_single_token_programs_off_its_layout(
    variant,
) -> None:
    """A z-gated call at the ring's batch that the ring does not read -- a destination table, no state update,
    ``x`` / ``B`` / ``C`` or ``z`` at a wider batch stride -- never plans the z-gated ring."""
    inputs = _stp_ring_gated_case("plain")
    plan = _route(inputs)
    if plan is None or plan.program != _STP_RING_Z:
        pytest.skip("the route takes no z-gated one-token ring on this device")
    if variant == "padded_z":
        inputs["z"] = _padded_batch_stride(inputs["z"])
    else:
        inputs = _vary(inputs, variant)
    plan = _route(inputs)
    assert plan is None or plan.program != _STP_RING_Z


@requires_blackwell
def test_horizontal_route_serves_padded_rows(cake_hits) -> None:
    """x, B, C and the state reach the horizontal program through tensor maps built from their strides: a padded x stays on it."""
    inputs = _make_case(_case("mtp_horizontal"))
    inputs["x"] = _padded_steps(inputs["x"])
    assert not inputs["x"].is_contiguous()
    assert _route(inputs).program == "mtp_horizontal_early"
    reference, candidate = _split_arms(inputs)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True], "a padded x must stay on the horizontal program"
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
def test_horizontal_route_falls_back_for_non_contiguous_inputs(cake_hits) -> None:
    """The slot tables are read directly (the binding checks their contiguity): a strided table falls back."""
    inputs = _make_case(_case("mtp_horizontal"))
    assert _route(inputs).program == "mtp_horizontal_early"
    table = torch.zeros((inputs["x"].shape[0], 2), dtype=torch.int64, device="cuda")
    table[:, 0] = inputs["state_batch_indices"]
    inputs["state_batch_indices"] = table[:, 0]
    inputs["intermediate_state_indices"] = table[:, 0]
    assert not inputs["state_batch_indices"].is_contiguous()
    assert (
        _route(inputs).program == "mtp_horizontal_early"
    )  # planned; the binding rejects the strided table
    _assert_falls_back(inputs, cake_hits)


@requires_blackwell
def test_horizontal_route_falls_back_for_mismatched_shapes() -> None:
    inputs = _make_case(_case("mtp_horizontal"))
    batch_size, token_steps, nheads, dim = inputs["x"].shape
    half_heads = torch.zeros(
        (batch_size, token_steps, nheads // 2, dim), dtype=torch.bfloat16, device="cuda"
    )
    assert _route(dict(inputs, x=half_heads)) is None
    assert _route(dict(inputs, C=inputs["C"][..., :64].contiguous())) is None


# (case, program or {arch: program}, grid): the multi-token routes' builds and launch grids.
_MULTI_TOKEN_ROUTES = (
    ("dynamic0", "dynamic_lanes8", (1, 16, 4)),
    (
        "dynamic1",
        {"sm_100a": "dynamic_lanes16_pr", "sm_103a": "dynamic_lanes16"},
        (1, 16, 8),
    ),
    ("dynamic3", "dynamic_lanes16_pr", (1, 16, 8)),
    ("dynamic0_b8", "dynamic_lanes8", (8, 16, 4)),
    ("mtp_cache", "mtp_cache_groups8_heads64", (1, 64, 4)),
    ("mtp_cache_g1", "mtp_cache_groups1_heads64", (1, 64, 4)),
    ("mtp_short1", "mtp_short_ring_heads64_groups8_t1", (64 * 8, 1, 1)),
    (
        "mtp_short2",
        {"sm_100a": _JR_RING, "sm_103a": _JR_HB_RING},
        (64 * 8, 1, 1),
    ),
    (
        "mtp_short2_b256",
        {"sm_100a": _JR_RING, "sm_103a": _JR_HB_RING},
        (256 * 8, 1, 1),
    ),
    ("mtp_short1_b16", "mtp_short_prefetch_heads64_groups8", (16 * 64, 1, 1)),
    ("mtp_short2_h32", "mtp_short_prefetch", (64 * 32, 1, 1)),
    ("mtp_horizontal", "mtp_horizontal_early", (32 * 64 // 2, 1, 1)),
    ("mtp_horizontal_b64", "mtp_horizontal_direct", (64 * 64 // 2, 1, 1)),
    ("stp_fp32", "stp_dim128_fp32_l2_prefetch", (8, 64, 64)),
)


@requires_blackwell
@pytest.mark.parametrize(
    "case_name, program, grid",
    _MULTI_TOKEN_ROUTES,
    ids=[route[0] for route in _MULTI_TOKEN_ROUTES],
)
def test_routes_pick_the_build_and_grid(case_name, program, grid) -> None:
    """The dynamic program runs 16 lanes per row below eight batch rows of a multi-token call, one batch row at
    the token counts of the architecture its pipelined-reduction build; the horizontal producer runs one CTA per
    work group, the early-index build up to eight work groups per SM; the short program's TMA ring runs one CTA
    per (batch row, group) inside its domain, the build folding 64 heads of 8 groups one CTA per (batch row,
    head) outside it."""
    plan = _route(_make_case(_case(case_name)))
    assert plan is not None
    if isinstance(program, dict):
        program = program[plan.arch]
    if _routes_as_measured():
        assert (plan.program, plan.grid, plan.defines) == (program, grid, ())


def _unlisted_projection_layout(x):
    """``x`` with one spare head of padding per step: a projection layout no listed cache instantiation has."""
    batch_size, token_steps, nheads, dim = x.shape
    padded = torch.empty(
        (batch_size, token_steps, nheads + 1, dim), dtype=x.dtype, device=x.device
    )
    padded[:, :, :nheads].copy_(x)
    return padded[:, :, :nheads]


@requires_blackwell
@pytest.mark.parametrize(
    "case_name, program",
    [("dynamic0_b8", "dynamic_lanes8"), ("mtp_cache", "mtp_cache_groups8_heads64")],
)
def test_routes_fall_back_when_jit_is_disabled_and_the_build_is_missing(
    case_name, program, monkeypatch, cake_hits
) -> None:
    """With JIT builds disabled and no build of the program in the cache, the call runs on FlashInfer.

    Control first: with JIT enabled the same call plans and runs on the Cake program.
    """
    reference, candidate = _split_arms(_make_case(_case(case_name)))
    out_reference = selective_state_update(**reference, backend="flashinfer")
    assert _route(candidate).program == program
    # The control arm gets its own state copies (the coefficient views keep their broadcast strides).
    _, control_candidate = _split_arms(candidate)
    out_control = cake_selective_state_update(**control_candidate)
    assert cake_hits == [True], "with JIT enabled the route must run on Cake"
    _assert_arms_close(control_candidate, out_control, reference, out_reference)

    def missing_build(program, arch, defines):
        raise MissingJITCacheError(
            "JIT compilation is disabled and the module is not in the JIT cache"
        )

    monkeypatch.setattr(cake, "_program", missing_build)
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True, False], (
        "a missing build must fall back when JIT builds are disabled"
    )
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
def test_unlisted_cache_instantiation_falls_back_when_jit_is_disabled(
    monkeypatch, cake_hits
) -> None:
    """A projection layout outside the cache program's listed instantiations has no build to load when
    JIT is disabled; the loader's own ``MissingJITCacheError`` path hands the call to FlashInfer."""
    reference, candidate = _split_arms(_make_case(_case("mtp_cache")))
    # Load the FlashInfer program before JIT builds are disabled; the fallback must not build anything.
    out_reference = selective_state_update(**reference, backend="flashinfer")
    candidate = dict(candidate, x=_unlisted_projection_layout(candidate["x"]))
    plan = _route(candidate)
    assert plan.program == "mtp_cache_c4_t6"
    assert (
        cake.define_digest(plan.defines)
        not in cake.MODULES[cake.PROGRAMS[plan.program]]["instantiations"]
    )
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [False], (
        "an unlisted instantiation must fall back when JIT builds are disabled"
    )
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


def test_cache_defines_follow_the_storage_dtypes_and_projection_layout() -> None:
    canonical = cake.cache_defines(
        torch.float32,
        torch.int64,
        64,
        1,
        (24576, 4096),
        (384, 64),
        (768, 128),
        (768, 128),
    )
    assert canonical == (
        ("COEFFICIENT_BF16", 0),
        ("INDEX_I32", 0),
        ("NHEADS_STATIC", 0),
        ("NGROUPS_STATIC", 0),
        ("X_BATCH_STRIDE", 24576),
        ("X_STEP_STRIDE", 4096),
        ("DT_BATCH_STRIDE", 384),
        ("DT_STEP_STRIDE", 64),
        ("B_BATCH_STRIDE", 768),
        ("B_STEP_STRIDE", 128),
        ("C_BATCH_STRIDE", 768),
        ("C_STEP_STRIDE", 128),
    )
    raw = cake.cache_defines(
        torch.bfloat16,
        torch.int32,
        64,
        1,
        (26112, 4352),
        (51072, 8512),
        (26112, 4352),
        (26112, 4352),
    )
    assert raw == (
        ("COEFFICIENT_BF16", 1),
        ("INDEX_I32", 1),
        ("NHEADS_STATIC", 64),
        ("NGROUPS_STATIC", 1),
        ("X_BATCH_STRIDE", 26112),
        ("X_STEP_STRIDE", 4352),
        ("DT_BATCH_STRIDE", 51072),
        ("DT_STEP_STRIDE", 8512),
        ("B_BATCH_STRIDE", 26112),
        ("B_STEP_STRIDE", 4352),
        ("C_BATCH_STRIDE", 26112),
        ("C_STEP_STRIDE", 4352),
    )


def test_jit_names_stay_within_the_file_name_limit() -> None:
    """Every delivered instantiation is named by its define-set digest (never the spelled-out defines)."""
    seen = set()
    for module, record in cake.MODULES.items():
        instantiations = record["instantiations"] or {"": {}}
        for digest, values in instantiations.items():
            defines = tuple((name, int(values[name])) for name in record["defines"])
            assert digest == ("" if not defines else cake.define_digest(defines))
            for arch in ("sm_100a", "sm_103a"):
                if arch not in record.get("arches", (arch,)):
                    with pytest.raises(ValueError, match="not delivered"):
                        cake.jit_spec(module, arch, defines)
                    continue
                spec = cake.jit_spec(module, arch, defines)
                assert len(spec.name.encode()) < 200
                assert spec.name not in seen
                seen.add(spec.name)
                assert all(
                    f"-D{name}={value}" in spec.extra_cuda_cflags
                    for name, value in defines
                )
    raw = cake.cache_defines(
        torch.bfloat16,
        torch.int32,
        64,
        1,
        (26112, 4352),
        (51072, 8512),
        (26112, 4352),
        (26112, 4352),
    )
    assert (
        cake.define_digest(raw)
        in cake.MODULES[cake.PROGRAMS["mtp_cache_c4_t6"]]["instantiations"]
    )


def test_aot_table_names_every_delivered_build() -> None:
    """The AOT specifications are exactly the loader's build specifications: every program instantiation, on
    each architecture the program is delivered for."""
    for arch in ("sm_100a", "sm_103a"):
        expected = set()
        for module, record in cake.MODULES.items():
            if arch not in record.get("arches", (arch,)):
                continue
            for values in (record["instantiations"] or {"": {}}).values():
                defines = tuple((name, int(values[name])) for name in record["defines"])
                expected.add(cake.jit_spec(module, arch, defines).name)
        specs = cake.gen_cake_selective_state_update_modules(arch)
        assert len(specs) == len(expected)
        assert {spec.name for spec in specs} == expected


def test_registry_names_every_program() -> None:
    both, b200, gb300 = ["sm_100a", "sm_103a"], ["sm_100a"], ["sm_103a"]
    # The builds of the paired row-major module by the architectures they ship for: the plain build where the
    # architecture runs it, the ``_lean`` build where it runs that one; the headdim-64 builds in both ABIs.
    hd64 = {
        "hpc1_ratio8": (both, None),
        "hpc1": (both, None),
        "hpc2_mb2": (None, both),
        "hpc2": (gb300, both),
        "hpc4": (None, both),
        "hpc1_ratio8_s4w2": (b200, gb300),
        "hpc1_s4w2": (b200, gb300),
        "hpc2_ratio8_s2w4": (None, both),
        "hpc2_s2w4": (None, both),
        "hpc4_s2w4": (None, both),
        "hpc4_s4w2": (None, both),
        "hpc2_s2w4_mb5": (None, both),
    }
    dim128 = {
        "hpc2": (None, both),
        "hpc1_mb4": (both, None),
        "z_gated": (both, None),
        "z_balanced": (None, both),
        "hpc1_s2w4": (b200, gb300),
        "hpc2_s2w4_mb5": (None, b200),
    }
    paired_rows = {}
    for variant, (plain, lean) in hd64.items():
        for abi in ("", "_bf16_i32"):
            if plain:
                paired_rows[f"stp_hd64_paired_{variant}{abi}"] = plain
            if lean:
                paired_rows[f"stp_hd64_paired_{variant}_lean{abi}"] = lean
    for variant, (plain, lean) in dim128.items():
        if plain:
            paired_rows[f"stp_dim128_{variant}"] = plain
        if lean:
            paired_rows[f"stp_dim128_{variant}_lean"] = lean
    assert len(paired_rows) == 37
    for arch in both:
        lean = {
            name
            for name, arches in paired_rows.items()
            if "_lean" in name and arch in arches
        }
        assert len(lean) == {"sm_100a": 19, "sm_103a": 23}[arch]
    rows = {
        f"stp_hd64_rows_{band}{abi}"
        for band in ("bf16_dt8", "bf16_dt4_od", "fp32_dt8", "fp32_dt1_large")
        for abi in ("", "_bf16_i32")
    } | {
        "stp_hd64_rows_bf16_dt4_xa_od",
        "stp_hd64_rows_bf16_dt4_xa_bf16_i32",
        "stp_hd64_rows_fp32_dt1",
        "stp_hd64_rows_fp32_dt4_bf16_i32",
        "stp_hd64_rows_fp32_dt2_bf16_i32",
        "stp_hd64_rows_fp32_dt1_bf16_i32",
        "stp_hd64_rows_fp32_dt1_l2_prefetch_bf16_i32",
        "stp_hd64_rows_fp32_dt1_large_sl2first",
        "stp_hd64_rows_fp32_dt1_large_sl2first_bf16_i32",
        "stp_dim128_fp32",
        "stp_dim128_fp32_l2_prefetch",
    }
    multi_token = {
        "dynamic_lanes8",
        "dynamic_lanes16",
        "dynamic_lanes16_pr",
        "mtp_cache_groups1",
        "mtp_cache_groups8",
        "mtp_cache_bf16_i32",
        "mtp_cache_groups1_heads64",
        "mtp_cache_groups8_heads64",
        "mtp_short_prefetch",
        "mtp_short_prefetch_heads64_groups8",
        "mtp_short_ring_heads64_groups8_t1",
        "mtp_short_ring_heads64_groups8_t2",
        _STP_RING,
        _STP_RING_Z,
        _JR_RING,
        _JR_HB_RING,
        "mtp_horizontal_early",
        "mtp_horizontal_direct",
    }
    # The carried programs: the FP32 row-owner tile under the tile plan's defines (bands without a rendered
    # build) and the six-token cache producer instantiated per projection layout (layouts beyond the rendered
    # three).
    carried = {"stp_hd64_rows_fp32", "mtp_cache_c4_t6"}
    assert set(cake.PROGRAMS) == set(paired_rows) | rows | multi_token | carried
    assert len(cake.PROGRAMS) == 76
    # A build shipping on one architecture spells it; every other record serves both.
    expected_arches = dict(
        paired_rows,
        **{
            _STP_RING: b200,
            _STP_RING_Z: b200,
            _JR_RING: b200,
            _JR_HB_RING: gb300,
        },
        **{
            f"stp_hd64_rows_fp32_dt1_large_sl2first{abi}": gb300
            for abi in ("", "_bf16_i32")
        },
    )
    for program, module in cake.PROGRAMS.items():
        arches = cake.MODULES[module].get("arches", both)
        assert arches == expected_arches.get(program, both), program
    for program in carried:
        assert cake.MODULES[cake.PROGRAMS[program]]["defines"]
    for program in set(cake.PROGRAMS) - carried:
        assert cake.MODULES[cake.PROGRAMS[program]]["defines"] == []
        assert cake.MODULES[cake.PROGRAMS[program]]["instantiations"] == {}
    assert set(cake.PROGRAMS.values()) == set(cake.MODULES)
    for record in cake.MODULES.values():
        assert len(record["sources"]) == 2
        assert all((cake._source_dir() / path).is_file() for path in record["sources"])
