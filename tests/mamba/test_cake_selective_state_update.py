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
    ("mtp_cache", 1, 64, 8, 64, 128, 6, torch.bfloat16, "vertical", True, None),
    ("mtp_horizontal", 32, 64, 8, 64, 128, 6, torch.bfloat16, "horizontal", True, None),
    ("dynamic0", 1, 16, 1, 64, 128, 1, torch.float32, "simple", False, 0),
    ("dynamic0_b8", 8, 16, 1, 64, 128, 2, torch.float32, "simple", False, 0),
    ("dynamic1", 1, 16, 1, 64, 128, 4, torch.float32, "simple", False, 1),
    ("dynamic3", 1, 16, 1, 64, 128, 8, torch.float32, "simple", False, 3),
    ("dynamic7", 1, 16, 1, 64, 128, 8, torch.float32, "simple", False, 7),
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


def _case(name):
    return next(case for case in _CASES if case[0] == name)


def _route(inputs):
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
        z=None,
        output=torch.empty_like(inputs["x"]),
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
    assert _route(inputs).program == "dynamic"
    batch_size, token_steps, nheads, dim = inputs["x"].shape
    padded = torch.zeros(
        (batch_size, token_steps + 1, nheads, dim), dtype=torch.bfloat16, device="cuda"
    )
    padded[:, :token_steps].copy_(inputs["x"])
    inputs["x"] = padded[:, :token_steps]
    assert not inputs["x"].is_contiguous()
    # The shapes still plan the launch; the binding rejects the padded rows before launching.
    assert _route(inputs).program == "dynamic"
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
    """A dense A (one value per element instead of one per head) never reaches the FP32 identity program."""
    inputs = _make_case(_case("stp_fp32"))
    assert _route(inputs).program == "stp_fp32_identity"
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
    assert (
        _route(inputs).program == "stp_fp32_identity"
    )  # planned; the binding rejects the padded rows
    _assert_falls_back(inputs, cake_hits)


@requires_blackwell
def test_short_route_falls_back_for_non_contiguous_inputs(cake_hits) -> None:
    inputs = _make_case(_case("mtp_short2"))
    assert _route(inputs).program == "mtp_short"
    inputs["x"] = _padded_steps(inputs["x"])
    assert not inputs["x"].is_contiguous()
    assert (
        _route(inputs).program == "mtp_short"
    )  # planned; the binding rejects the padded rows
    _assert_falls_back(inputs, cake_hits)


@requires_blackwell
def test_short_route_falls_back_for_mismatched_shapes() -> None:
    inputs = _make_case(_case("mtp_short2"))
    assert _route(dict(inputs, B=inputs["B"][..., :64].contiguous())) is None
    batch_size, token_steps, nheads, dim = inputs["x"].shape
    half_heads = torch.zeros(
        (batch_size, token_steps, nheads // 2, dim), dtype=torch.bfloat16, device="cuda"
    )
    assert _route(dict(inputs, x=half_heads)) is None


@requires_blackwell
def test_horizontal_route_serves_padded_rows(cake_hits) -> None:
    """x, B, C and the state reach the horizontal program through tensor maps built from their strides: a padded x stays on it."""
    inputs = _make_case(_case("mtp_horizontal"))
    inputs["x"] = _padded_steps(inputs["x"])
    assert not inputs["x"].is_contiguous()
    assert _route(inputs).program == "mtp_horizontal"
    reference, candidate = _split_arms(inputs)
    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True], "a padded x must stay on the horizontal program"
    _assert_arms_close(candidate, out_candidate, reference, out_reference)


@requires_blackwell
def test_horizontal_route_falls_back_for_non_contiguous_inputs(cake_hits) -> None:
    """The slot tables are read directly (the binding checks their contiguity): a strided table falls back."""
    inputs = _make_case(_case("mtp_horizontal"))
    assert _route(inputs).program == "mtp_horizontal"
    table = torch.zeros((inputs["x"].shape[0], 2), dtype=torch.int64, device="cuda")
    table[:, 0] = inputs["state_batch_indices"]
    inputs["state_batch_indices"] = table[:, 0]
    inputs["intermediate_state_indices"] = table[:, 0]
    assert not inputs["state_batch_indices"].is_contiguous()
    assert (
        _route(inputs).program == "mtp_horizontal"
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
    "case_name, program", [("dynamic0_b8", "dynamic"), ("mtp_cache", "mtp_cache_c4_t6")]
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


def test_shipped_stp_program_follows_the_head_ratio_and_grid() -> None:
    """The BF16 single-token clones keep their selection rules, and every one of them is present as source."""
    sm_count = 148
    assert cake.shipped_stp_program(1, 512, sm_count) == "stp_bf16_direct"
    assert cake.shipped_stp_program(4, 512, sm_count) == "stp_bf16_direct"
    assert cake.shipped_stp_program(8, 1024, sm_count) == "stp_bf16_ratio8"
    assert cake.shipped_stp_program(8, 2048, sm_count) == "stp_bf16_ratio8_saturated"
    assert cake.shipped_stp_program(16, 2048, sm_count) == "stp_bf16_ratio16"
    # The direct cap is nine resident waves of three CTAs per SM; above it the persistent program runs.
    assert cake.shipped_stp_program(16, 27 * sm_count, sm_count) == "stp_bf16_ratio16"
    assert (
        cake.shipped_stp_program(16, 27 * sm_count + 1, sm_count)
        == "stp_bf16_persistent"
    )
    assert (
        cake.shipped_stp_program(1, 27 * sm_count + 1, sm_count)
        == "stp_bf16_persistent"
    )
    for name in cake._SHIPPED_STP:
        assert (
            cake._source_dir() / "cuda" / f"cake_selective_state_update_{name}.cu"
        ).is_file()
        assert (
            cake._source_dir() / "host" / f"cake_selective_state_update_{name}.cc"
        ).is_file()


def test_jit_names_stay_within_the_file_name_limit() -> None:
    """Every delivered instantiation is named by its define-set digest (never the spelled-out defines)."""
    seen = set()
    for module, record in cake.MODULES.items():
        instantiations = record["instantiations"] or {"": {}}
        for digest, values in instantiations.items():
            defines = tuple((name, int(values[name])) for name in record["defines"])
            assert digest == ("" if not defines else cake.define_digest(defines))
            for arch in ("sm_100a", "sm_103a"):
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
    """The AOT specifications are exactly the loader's build specifications, one per program instantiation."""
    for arch in ("sm_100a", "sm_103a"):
        expected = set()
        for module, record in cake.MODULES.items():
            for values in (record["instantiations"] or {"": {}}).values():
                defines = tuple((name, int(values[name])) for name in record["defines"])
                expected.add(cake.jit_spec(module, arch, defines).name)
        specs = cake.gen_cake_selective_state_update_modules(arch)
        assert len(specs) == len(expected)
        assert {spec.name for spec in specs} == expected


def test_registry_names_every_program() -> None:
    assert set(cake.PROGRAMS) == {
        "stp_fp32_identity",
        "mtp_short",
        "mtp_cache_c4_t6",
        "mtp_horizontal",
        "dynamic",
    }
    assert set(cake.PROGRAMS.values()) == set(cake.MODULES)
    for record in cake.MODULES.values():
        assert len(record["sources"]) == 2
        assert all((cake._source_dir() / path).is_file() for path in record["sources"])
