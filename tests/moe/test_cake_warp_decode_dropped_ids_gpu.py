"""GPU tests for routing slots whose expert id lies outside ``[0, num_experts)``.

The Cake warp-decode kernels drop such a slot: it adds nothing to the output,
its weight (even NaN or Inf) never enters the sum, and a token without a valid
slot outputs +0. Direct rows compute a dropped slot as expert 0 and the
finalizer discards it; packed rows never give it a tile. No host code reads the
ids, so graph replays may switch between valid and dropped ids freely.

Every geometry runs on exact SM100 (the 15 parity geometries and the clamped
E256 geometry) and exact SM103 (the 15 parity geometries). Each row checks:

* ``padded``: some slots are ``-1`` and the last token has no valid slot. The
  workspace is filled with 0xFF before preparation, so a stale route slot or
  partial would show. The output matches the public reference (where one
  exists), the empty token is exactly +0, and the output is bitwise equal to
  the same routing with valid ids and zero weights on the dropped slots.
* ``collision``: every slot except slot 0 (expert 0) is dropped, so on direct
  rows the dropped slots compute the same expert as the live one.
* ``values``: the other out-of-range values (``E``, ``E + 7``, ``INT32_MAX``,
  ``INT32_MIN``) and NaN / Inf weights on dropped slots give the bits of the
  ``-1`` row.
* ``all_dropped``: a batch without any valid slot outputs +0, and the next
  launch on the same workspace with valid ids is still correct.
* ``graph_*``: one CUDA graph, captured with valid or with padded ids, replays
  both routings from the same buffers.
* ``layer``: the public ``MoELayer`` path, eager and graph replay.
"""

from __future__ import annotations

import functools

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CakeWarpDecodeConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
)
from tests.moe.cake_warp_decode_reference import (
    CLAMPED_GEOMETRY,
    MAX_TOKENS,
    QUANT,
    TARGETS,
    Fixture,
    Geometry,
    assert_close,
    cake_module,
    nondefault_streams,
    prepare_fixture,
    reference,
    run_cake,
)
from tests.moe.test_cake_warp_decode import _GPU_GEOMETRIES, _gpu_fixture

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)
_CLAMPED_SEED = 20260924

_GEOMETRIES: tuple[Geometry, ...] = (*_GPU_GEOMETRIES, CLAMPED_GEOMETRY)
_SM100_ONLY = {CLAMPED_GEOMETRY.name}


def _cases(geometry: Geometry, num_tokens: int) -> list[str]:
    cases = ["padded", "collision"]
    boundary = num_tokens in geometry.token_counts
    if boundary:
        cases += ["values", "all_dropped", "graph_valid_first", "graph_padded_first"]
    if num_tokens in (geometry.token_counts[0], geometry.token_counts[-1]):
        cases.append("layer")
    return cases


# Geometry-major, so one cached fixture serves all rows of a geometry.
_ROWS = [
    pytest.param(
        geometry, num_tokens, case, id=f"{geometry.name}-t{num_tokens:02d}-{case}"
    )
    for geometry in _GEOMETRIES
    for num_tokens in range(1, MAX_TOKENS + 1)
    for case in _cases(geometry, num_tokens)
]


def _target(geometry: Geometry) -> str:
    if not torch.cuda.is_available():
        pytest.skip("Cake warp decode GPU tests require a CUDA device")
    capability = torch.cuda.get_device_capability(torch.device("cuda"))
    if capability not in TARGETS:
        pytest.skip(
            "Cake warp decode GPU tests require exact SM100 or SM103, "
            f"got SM{capability[0]}{capability[1]}"
        )
    target = TARGETS[capability]
    if geometry.name in _SM100_ONLY and target != "sm100a":
        pytest.skip(f"{geometry.name} is served on exact SM100 only")
    return target


@functools.lru_cache(maxsize=1)
def _clamped_fixture() -> Fixture:
    return prepare_fixture(CLAMPED_GEOMETRY, _CLAMPED_SEED)


def _fixture(geometry_name: str) -> Fixture:
    """The parity matrix's cached fixture of one geometry, or the clamped one."""
    if geometry_name == CLAMPED_GEOMETRY.name:
        return _clamped_fixture()
    return _gpu_fixture(geometry_name)


def _padding_mask(num_tokens: int, top_k: int, device: torch.device) -> torch.Tensor:
    """Dropped slots: a fixed pattern that keeps slot 0, plus a last token with none."""
    tokens = torch.arange(num_tokens, device=device)[:, None]
    ranks = torch.arange(top_k, device=device)[None, :]
    mask = (tokens * 5 + ranks * 3) % 4 == 1
    mask[:, 0] = False
    if num_tokens > 1:
        mask[-1, :] = True
    return mask


def _routing(
    fixture: Fixture,
    num_tokens: int,
    mask: torch.Tensor,
    dropped_id: int | None,
    dropped_weight: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixture routing with ``dropped_id`` (None: keep the id) on the masked slots."""
    ids = fixture.topk_ids[:num_tokens].clone()
    weights = fixture.topk_weights[:num_tokens].clone()
    if dropped_id is not None:
        ids.masked_fill_(mask, dropped_id)
    if dropped_weight is not None:
        weights.masked_fill_(mask, dropped_weight)
    return ids.contiguous(), weights.contiguous()


def _prepare(module, fixture: Fixture, num_tokens: int, *, poison: bool = True):
    """Allocate a workspace (0xFF-filled when ``poison``), prepare it, return it."""
    shape = fixture.shape(num_tokens)
    size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
    workspace = torch.empty(size, dtype=torch.uint8, device=fixture.device)
    if poison:
        workspace.fill_(0xFF)
    output = torch.full(
        (num_tokens, fixture.geometry.hidden_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=fixture.device,
    )
    receipt = int(
        module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
    )
    assert receipt > 0
    return output, workspace, receipt


def _run(module, fixture: Fixture, num_tokens: int, ids, weights, *, poison=True):
    """One launch on a fresh workspace; returns a copy of the output."""
    output, workspace, receipt = _prepare(module, fixture, num_tokens, poison=poison)
    try:
        run_cake(
            module,
            fixture,
            num_tokens,
            output,
            workspace,
            receipt,
            topk_ids=ids,
            topk_weights=weights,
        )
        result = output.clone()
        torch.cuda.synchronize(fixture.device)
    finally:
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
    return result


def _assert_positive_zero(rows: torch.Tensor) -> None:
    assert (rows.contiguous().view(torch.int16) == 0).all(), "expected exact +0"


def _check_reference(fixture: Fixture, num_tokens: int, actual, ids, weights) -> None:
    if fixture.geometry.has_public_reference:
        expected = reference(fixture, num_tokens, topk_ids=ids, topk_weights=weights)
        torch.cuda.synchronize(fixture.device)
        assert_close(actual, expected)


def _check_padded(module, fixture: Fixture, num_tokens: int, mask) -> torch.Tensor:
    ids, weights = _routing(fixture, num_tokens, mask, -1)
    actual = _run(module, fixture, num_tokens, ids, weights)
    assert torch.isfinite(actual.float()).all()
    if num_tokens > 1:
        _assert_positive_zero(actual[-1])
    # Oracle: the same slots with their valid ids and zero weights.
    oracle_ids, oracle_weights = _routing(fixture, num_tokens, mask, None, 0.0)
    oracle = _run(module, fixture, num_tokens, oracle_ids, oracle_weights, poison=False)
    assert torch.equal(actual, oracle), "dropped slots differ from zero-weight slots"
    _check_reference(fixture, num_tokens, actual, ids, weights)
    return actual


def _check_values(module, fixture: Fixture, num_tokens: int, mask) -> None:
    baseline = _check_padded(module, fixture, num_tokens, mask)
    num_experts = fixture.geometry.num_experts
    for dropped_id, dropped_weight in (
        (num_experts, float("nan")),
        (num_experts + 7, float("inf")),
        (INT32_MAX, float("-inf")),
        (INT32_MIN, None),
        (-1, float("nan")),
    ):
        ids, weights = _routing(fixture, num_tokens, mask, dropped_id, dropped_weight)
        actual = _run(module, fixture, num_tokens, ids, weights)
        assert torch.equal(actual, baseline), (dropped_id, dropped_weight)


def _check_all_dropped(module, fixture: Fixture, num_tokens: int) -> None:
    """All slots dropped, then valid ids, then dropped again on one workspace."""
    everything = torch.ones(
        num_tokens, fixture.geometry.top_k, dtype=torch.bool, device=fixture.device
    )
    dropped_ids, dropped_weights = _routing(
        fixture, num_tokens, everything, -1, float("nan")
    )
    dropped_ids[:, 1::2] = fixture.geometry.num_experts
    valid_ids, valid_weights = _routing(fixture, num_tokens, everything, None)
    expected_valid = _run(module, fixture, num_tokens, valid_ids, valid_weights)
    output, workspace, receipt = _prepare(module, fixture, num_tokens)
    try:
        results = []
        for ids, weights in (
            (dropped_ids, dropped_weights),
            (valid_ids, valid_weights),
            (dropped_ids, dropped_weights),
        ):
            output.fill_(float("nan"))
            run_cake(
                module,
                fixture,
                num_tokens,
                output,
                workspace,
                receipt,
                topk_ids=ids,
                topk_weights=weights,
            )
            results.append(output.clone())
        torch.cuda.synchronize(fixture.device)
    finally:
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
    _assert_positive_zero(results[0])
    assert torch.equal(results[1], expected_valid), "valid launch after an empty one"
    _assert_positive_zero(results[2])
    _check_reference(fixture, num_tokens, results[1], valid_ids, valid_weights)


def _check_collision(module, fixture: Fixture, num_tokens: int) -> None:
    mask = torch.ones(
        num_tokens, fixture.geometry.top_k, dtype=torch.bool, device=fixture.device
    )
    mask[:, 0] = False
    assert (fixture.topk_ids[:num_tokens, 0] == 0).all()
    ids, weights = _routing(fixture, num_tokens, mask, -1)
    actual = _run(module, fixture, num_tokens, ids, weights)
    assert torch.isfinite(actual.float()).all()
    oracle_ids, oracle_weights = _routing(fixture, num_tokens, mask, None, 0.0)
    oracle = _run(module, fixture, num_tokens, oracle_ids, oracle_weights, poison=False)
    assert torch.equal(actual, oracle), "dropped slots changed the expert-0 route"
    _check_reference(fixture, num_tokens, actual, ids, weights)


def _check_graph(module, fixture: Fixture, num_tokens: int, mask, padded_first: bool):
    """Capture once and replay valid and padded routings from the same buffers."""
    valid = _routing(fixture, num_tokens, mask, None)
    padded = _routing(fixture, num_tokens, mask, -1, float("nan"))
    expected = {
        "valid": _run(module, fixture, num_tokens, *valid),
        "padded": _run(module, fixture, num_tokens, *padded),
    }
    order = ("padded", "valid") if padded_first else ("valid", "padded")
    routings = {"valid": valid, "padded": padded}
    ids = routings[order[0]][0].clone()
    weights = routings[order[0]][1].clone()
    capture_stream, replay_stream = nondefault_streams(fixture.device, 2)
    with torch.cuda.stream(capture_stream):
        output, workspace, receipt = _prepare(module, fixture, num_tokens)
    try:
        with torch.cuda.stream(capture_stream):
            run_cake(
                module,
                fixture,
                num_tokens,
                output,
                workspace,
                receipt,
                topk_ids=ids,
                topk_weights=weights,
            )
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            run_cake(
                module,
                fixture,
                num_tokens,
                output,
                workspace,
                receipt,
                topk_ids=ids,
                topk_weights=weights,
            )
        replays = []
        with torch.cuda.stream(replay_stream):
            for name in (*order, order[0]):
                ids.copy_(routings[name][0])
                weights.copy_(routings[name][1])
                output.fill_(float("nan"))
                graph.replay()
                replays.append((name, output.clone()))
        replay_stream.synchronize()
    finally:
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
    for name, actual in replays:
        assert torch.equal(actual, expected[name]), f"{name} replay"


def _check_layer(fixture: Fixture, num_tokens: int, mask) -> None:
    """MoELayer with the explicit Cake config: eager, then graph replay."""
    geometry = fixture.geometry
    device = fixture.device
    ids, weights = _routing(fixture, num_tokens, mask, -1, float("nan"))
    activations = MoEActivationPack(
        hidden_states_q=fixture.hidden_states_q[:num_tokens],
        hidden_states_scale=fixture.hidden_states_scale[:num_tokens],
        topk_ids=ids,
        topk_weights=weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    pack = MoEWeightPack()
    pack.prepare_for("cake", dict(fixture.weight_view))
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=geometry.num_experts,
            top_k=geometry.top_k,
            method=RoutingMethodType.TopK,
        ),
        quant=QUANT,
        experts=ExpertConfig(intermediate_size=geometry.intermediate_size),
        activation=geometry.activation,
        backend=BackendOptions((CakeWarpDecodeConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=MAX_TOKENS, enable_pdl=True),
    )
    layer = MoELayer(config, device=device)
    layer.tuner.clear_cache()
    warmup_stream, capture_stream, replay_stream = nondefault_streams(device, 3)
    with autotune(True, tuning_buckets=(num_tokens,)), torch.cuda.stream(warmup_stream):
        eager = layer(activations, pack).clone()
    warmup_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured = layer(activations, pack)
    with torch.cuda.stream(replay_stream):
        graph.replay()
        replayed = captured.clone()
    replay_stream.synchronize()
    if num_tokens > 1:
        _assert_positive_zero(eager[-1])
    assert torch.equal(replayed, eager), "graph replay differs from the eager call"
    _check_reference(fixture, num_tokens, eager, ids, weights)


@pytest.mark.gpu
@pytest.mark.parametrize("geometry,num_tokens,case", _ROWS)
def test_dropped_expert_ids(geometry, num_tokens, case):
    target = _target(geometry)
    fixture = _fixture(geometry.name)
    module = cake_module(fixture.device)
    assert TARGETS[torch.cuda.get_device_capability(fixture.device)] == target
    mask = _padding_mask(num_tokens, geometry.top_k, fixture.device)
    if case == "padded":
        _check_padded(module, fixture, num_tokens, mask)
    elif case == "collision":
        _check_collision(module, fixture, num_tokens)
    elif case == "values":
        _check_values(module, fixture, num_tokens, mask)
    elif case == "all_dropped":
        _check_all_dropped(module, fixture, num_tokens)
    elif case == "graph_valid_first":
        _check_graph(module, fixture, num_tokens, mask, padded_first=False)
    elif case == "graph_padded_first":
        _check_graph(module, fixture, num_tokens, mask, padded_first=True)
    elif case == "layer":
        _check_layer(fixture, num_tokens, mask)
    else:
        raise AssertionError(case)
