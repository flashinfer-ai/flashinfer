# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""GPU tests of the Cake single-GPU norm package (``flashinfer.cake_muse_spark_norm_backend``) behind
``gated_residual_next_rmsnorm(..., backend="cake")``: every routed program against the reference on both sides of
its window edges, the tensor-alpha-without-post_weight form run through the post_weight programs exactly, the per-device
zero vector, CUDA graph capture and replay, the launch grids and the input checks."""

import pytest
import torch

import flashinfer.cake_muse_spark_norm_backend as pkg
import flashinfer.norm as norm
from flashinfer.cake_muse_spark_norm_backend import cake_jit as loader
from flashinfer.norm import (
    gated_residual_next_rmsnorm,
    supports_gated_residual_next_rmsnorm,
)

HIDDEN_SIZE = pkg.HIDDEN_SIZE
POST_EPS = 1e-8
NEXT_EPS = 1e-5
NEXT_GAIN_CENTER = 1.0
OP = pkg.GATED_RESIDUAL_NEXT_RMSNORM
# (tensor_alpha, has_post_weight): the three exported variants and the fallback form.
FORMS = ((True, True), (False, False), (False, True), (True, False))


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _arch():
    return pkg._arch(_device()) if torch.cuda.is_available() else None


def _supported() -> bool:
    return _arch() is not None and all(
        pkg.is_cake_gated_residual_next_rmsnorm_supported(
            _device(), tensor_alpha=tensor_alpha, has_post_weight=has_post_weight
        )
        for tensor_alpha, has_post_weight in FORMS
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_muse_spark_norm needs an SM100 / SM103 / SM107 device with every route exported",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before the first test, through the
    # spec objects the package's loader itself builds on first use.
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


def _inputs(rows: int, *, tensor_alpha: bool = True, alpha_scalar: float = 0.5):
    branch = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    next_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.float32).uniform_(-3, 3)
    beta = torch.sigmoid(gate)
    alpha = torch.sqrt(torch.clamp(1.0 - beta.square(), min=1e-3))
    return [
        branch,
        residual,
        next_weight,
        alpha if tensor_alpha else alpha_scalar,
        beta,
    ]


def _reference(inputs, post_weight=None, post_gain_center=0.0):
    branch, residual, next_weight, alpha, beta = inputs
    branch_fp32 = branch.float()
    if post_weight is not None:
        branch_fp32 *= post_weight.float() + post_gain_center
    branch_fp32 *= torch.rsqrt(
        branch_fp32.square().mean(dim=-1, keepdim=True) + POST_EPS
    )
    residual_out = alpha * residual + beta * branch_fp32
    ffn_input = residual_out * torch.rsqrt(
        residual_out.square().mean(dim=-1, keepdim=True) + NEXT_EPS
    )
    ffn_input *= next_weight.float() + NEXT_GAIN_CENTER
    return residual_out, ffn_input.to(torch.bfloat16)


def _call(inputs, backend="cake", **kwargs):
    return gated_residual_next_rmsnorm(
        *inputs,
        post_eps=POST_EPS,
        next_eps=NEXT_EPS,
        next_gain_center=NEXT_GAIN_CENTER,
        backend=backend,
        **kwargs,
    )


def _assert_close(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.015625, atol=0.03125)


def _kwargs(inputs, has_post_weight: bool):
    if not has_post_weight:
        return {}
    return {"post_weight": torch.randn_like(inputs[2]) * 0.1, "post_gain_center": 1.0}


def _edge_rows(route: str) -> list[int]:
    """1, both sides of every window edge of the route on this device (the rows of a gap between windows
    included), and the two contract rows."""
    rows = {1, 17, 2437}
    for name in loader.route_modules(_arch(), route):
        for above, at_most in loader.MODULES[name]["select_rule"]["windows"][_arch()]:
            rows.add(int(above) + 1)
            if at_most is not None:
                rows.update((int(at_most), int(at_most) + 1))
    return sorted(rows)


def _admits(tensor_alpha: bool, has_post_weight: bool, rows: int) -> bool:
    """Whether the Cake package serves ``rows`` rows of the form on this device (its routes may leave a window of
    row counts to the CuTe DSL kernel, as the scalar-alpha forms do below 256 rows on sm_107a)."""
    return pkg.is_cake_gated_residual_next_rmsnorm_supported(
        _device(), tensor_alpha=tensor_alpha, has_post_weight=has_post_weight, rows=rows
    )


def _admitted_rows(tensor_alpha: bool, has_post_weight: bool, *candidates: int) -> int:
    """The first of ``candidates`` the Cake package serves for the form on this device."""
    for rows in candidates:
        if _admits(tensor_alpha, has_post_weight, rows):
            return rows
    raise AssertionError((tensor_alpha, has_post_weight, candidates))


@pytest.mark.parametrize(
    "form",
    FORMS,
    ids=lambda f: f"alpha_{'tensor' if f[0] else 'scalar'}_{'pw' if f[1] else 'nopw'}",
)
def test_every_window_matches_the_reference(form):
    tensor_alpha, has_post_weight = form
    route = pkg.gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
    launched = []
    original = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append((name, rows))
        return original(name, values, rows=rows, device_index=device_index)

    try:
        loader.run_module = spy
        checked = 0
        for rows in _edge_rows(route):
            torch.manual_seed(1000 + rows)
            inputs = _inputs(rows, tensor_alpha=tensor_alpha)
            kwargs = _kwargs(inputs, has_post_weight)
            assert supports_gated_residual_next_rmsnorm(
                *inputs, post_weight=kwargs.get("post_weight"), backend="cake"
            )
            if not _admits(tensor_alpha, has_post_weight, rows):
                # a declared gap of the route on this device (the kernel lane left the row count to the CuTe DSL
                # kernel): the package refuses the explicit call and ``auto`` runs that kernel
                with pytest.raises(NotImplementedError, match=f"admits {rows} rows"):
                    loader.selected_module(_arch(), route, rows)
                with pytest.raises(ValueError, match=f"admits {rows} rows"):
                    _call(inputs, **kwargs)
                if supports_gated_residual_next_rmsnorm(
                    *inputs, post_weight=kwargs.get("post_weight"), backend="cute"
                ):
                    before = len(launched)
                    _assert_close(
                        _call(inputs, backend="auto", **kwargs),
                        _reference(inputs, **kwargs),
                    )
                    assert len(launched) == before
                continue
            checked += 1
            _assert_close(_call(inputs, **kwargs), _reference(inputs, **kwargs))
            assert launched[-1] == (loader.selected_module(_arch(), route, rows), rows)
        assert checked, f"route {route} admits none of its edge rows on {_arch()}"
    finally:
        loader.run_module = original


def test_empty_batch_launches_nothing():
    launched = []
    original = loader.run_module
    try:
        loader.run_module = lambda *a, **k: launched.append(a)
        inputs = _inputs(0)
        residual_out, ffn_input = _call(inputs)
        assert residual_out.shape == (0, HIDDEN_SIZE) and ffn_input.shape == (
            0,
            HIDDEN_SIZE,
        )
        assert not launched
    finally:
        loader.run_module = original


def test_tensor_alpha_without_post_weight_runs_the_post_weight_program_exactly():
    # The fallback form: the post_weight route with a zero vector and gain 1.0. Its bits equal the same route fed
    # a ones vector with gain 0.0 (branch * (w + c) is exact for both), it is within the contract of the
    # reference and of the CuTe kernel, and it launches a record of the post_weight route.
    torch.manual_seed(4125)
    inputs = _inputs(257)
    fallback = _call(inputs)
    ones = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    zeros = torch.zeros(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    for post_weight, gain in ((ones, 0.0), (zeros, 1.0)):
        explicit = _call(inputs, post_weight=post_weight, post_gain_center=gain)
        assert torch.equal(fallback[0], explicit[0]) and torch.equal(
            fallback[1], explicit[1]
        )
    _assert_close(fallback, _reference(inputs))
    if supports_gated_residual_next_rmsnorm(*inputs, backend="cute"):
        _assert_close(fallback, _call(inputs, backend="cute"))
    route = pkg.gated_residual_next_rmsnorm_route(True, False)
    assert route == pkg.route_key(OP, "alpha_tensor_pw")
    name = loader.selected_module(_arch(), route, 257)
    assert loader.MODULES[name]["build"]["post_weight"] is True
    assert loader.MODULES[name]["build"]["tensor_alpha"] is True


def test_zero_post_weight_is_one_tensor_per_device():
    device = _device()
    pkg._zero_post_weights.clear()
    first = pkg.zero_post_weight(device)
    assert first.dtype == torch.bfloat16 and tuple(first.shape) == (HIDDEN_SIZE,)
    assert first.device == device and not first.any()
    for rows in (1, 17, 129):
        _call(_inputs(rows))
    assert list(pkg._zero_post_weights) == [device.index]
    assert pkg.zero_post_weight(device) is first
    assert pkg._zero_post_weights[device.index] is first


def test_zero_post_weight_is_never_allocated_during_capture(monkeypatch):
    pkg._zero_post_weights.clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="allocated on first use"):
        pkg.zero_post_weight(_device())
    assert not pkg._zero_post_weights
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    pkg.warm_up(_device(), tensor_alpha=True, has_post_weight=False)
    assert list(pkg._zero_post_weights) == [_device().index]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert pkg.zero_post_weight(_device()) is pkg._zero_post_weights[_device().index]


@pytest.mark.parametrize(
    "form",
    FORMS,
    ids=lambda f: f"alpha_{'tensor' if f[0] else 'scalar'}_{'pw' if f[1] else 'nopw'}",
)
def test_cuda_graph_replay_uses_changed_inputs(form):
    tensor_alpha, has_post_weight = form
    pkg._zero_post_weights.clear()
    pkg.warm_up(_device(), tensor_alpha=tensor_alpha, has_post_weight=has_post_weight)
    rows = _admitted_rows(tensor_alpha, has_post_weight, 17, 257, 513, 2437)
    inputs = _inputs(rows, tensor_alpha=tensor_alpha)
    kwargs = _kwargs(inputs, has_post_weight)
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _call(inputs, **kwargs)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        actual = _call(inputs, **kwargs)
    for seed in (3001, 4001):
        torch.manual_seed(seed)
        inputs[0].copy_(torch.randn_like(inputs[0]))
        inputs[1].copy_(torch.randn_like(inputs[1]))
        if has_post_weight:
            kwargs["post_weight"].copy_(torch.randn_like(inputs[2]) * 0.1)
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(actual, _reference(inputs, **kwargs))


def test_warm_up_loads_every_route():
    device = _device()
    for tensor_alpha, has_post_weight in FORMS:
        pkg.warm_up(device, tensor_alpha=tensor_alpha, has_post_weight=has_post_weight)
        route = pkg.gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
        for name in loader.route_modules(_arch(), route):
            assert loader.load(name, _arch()) is loader.load(name, _arch())
            if loader.MODULES[name]["grid_rule"]["kind"] == "persistent_rows":
                assert loader.cluster_capacity(name, _arch(), device.index) >= 1


def _tokens_heads_launch_cluster(rule) -> tuple[int, int, int]:
    """The launch cluster of a ``tokens_heads`` record: ``cluster`` CTAs over the head
    blocks (the grid's first dimension) when the rule names one above one; otherwise the
    token cluster along the token dimension, the first grid dimension in token-major
    order and the second in head-major order."""
    cluster, tcluster = int(rule["cluster"]), int(rule.get("tcluster", 1))
    if cluster > 1:
        return (cluster, 1, 1)
    if rule.get("order") == "heads":
        return (1, tcluster, 1)
    return (tcluster, 1, 1)


def _tokens_heads_grid(rule, tokens: int, heads: int) -> tuple[int, int, int]:
    """The grid a ``tokens_heads`` rule prescribes: a CTA of ``tpc`` tokens covers
    ``rows_per_cta / tpc`` heads each; otherwise a CTA covers ``rows_per_cta`` heads of
    one token, the token count rounded up to the token cluster (or the interleave), the
    head blocks first under a cluster over them or the head-major order."""
    rows_per_cta, tpc = int(rule["rows_per_cta"]), int(rule.get("tpc", 1))
    if tpc > 1:
        return (-(-tokens // tpc), -(-heads // (rows_per_cta // tpc)), 1)
    blocks = -(-heads // rows_per_cta)
    tcluster, interleave = int(rule.get("tcluster", 1)), int(rule.get("interleave", 1))
    if int(rule["cluster"]) > 1 or rule.get("order") == "heads":
        return (blocks, -(-tokens // tcluster) * tcluster, 1)
    step = max(tcluster, interleave)
    return (-(-tokens // step) * step, blocks, 1)


def test_launch_grid_follows_the_record_rule():
    """Every record delivered for this device launches the grid its ``grid_rule`` spells: ``rows x row_ctas`` CTAs
    for a ``rows`` rule (``row_ctas`` the rule's CTAs per row, the cluster when it names none), ``min(rows, co-resident
    clusters) x cluster`` for the persistent row loop, and the kernel's own two-dimensional arithmetic for a
    ``tokens_heads`` rule (``ceil(heads / rows_per_cta)`` head blocks beside the token count rounded up to the token
    cluster, in the order the rule names).  The launch cluster is the rule's: ``cluster`` CTAs along the head-block
    axis, or for a ``tokens_heads`` rule without one the token cluster ``tcluster`` along the token axis; every grid
    is a whole number of clusters."""
    device = _device()
    checked = 0
    for name, record in loader.MODULES.items():
        if _arch() not in record["arches"]:
            continue
        rule = record["grid_rule"]
        cluster = int(rule["cluster"])
        kind = rule["kind"]
        if kind == "tokens_heads":
            launch_cluster = _tokens_heads_launch_cluster(rule)
        else:
            launch_cluster = (cluster, 1, 1)
        assert tuple(record["launch"]["cluster"]) == launch_cluster, (
            name,
            rule,
            record["launch"]["cluster"],
        )
        for rows in (1, 7, 148, 4096):
            if kind == "rows":
                grid = loader.launch_grid(record, rows, device_index=device.index)
                assert grid == (rows * int(rule.get("row_ctas", cluster)), 1, 1), (
                    name,
                    rows,
                    grid,
                )
            elif kind == "persistent_rows":
                grid = loader.launch_grid(record, rows, device_index=device.index)
                capacity = loader.cluster_capacity(name, _arch(), device.index)
                assert capacity >= 1, name
                assert grid == (min(rows, capacity) * cluster, 1, 1), (name, rows, grid)
            else:
                assert kind == "tokens_heads", (name, rule)
                # the two-dimensional grid of ``rows`` tokens and the given heads; a launch without a head count
                # is refused
                with pytest.raises(ValueError, match="must carry 'heads'"):
                    loader.launch_grid(record, rows, device_index=device.index)
                # the head counts the record's windows name on any device (the only ones its head blocks fill a
                # whole cluster for), or its CTA's own when it names none
                heads_named = {
                    int(heads)
                    for windows in (record.get("select_rule") or {})
                    .get("windows", {})
                    .values()
                    for heads, _above, _at_most in windows
                } or {int(rule["rows_per_cta"])}
                for heads in sorted({1, 16, 32, 128} | heads_named):
                    grid = loader.launch_grid(
                        record, rows, device_index=device.index, values={"heads": heads}
                    )
                    assert grid == _tokens_heads_grid(rule, rows, heads), (
                        name,
                        rows,
                        heads,
                        grid,
                    )
                    if heads in heads_named:
                        assert all(
                            extent % size == 0
                            for extent, size in zip(grid, launch_cluster, strict=True)
                        ), (name, rows, heads, grid, launch_cluster)
            checked += 1
    assert checked


def test_refuses_inputs_outside_the_contract():
    inputs = _inputs(2)
    call = pkg.cake_gated_residual_next_rmsnorm
    common = dict(
        post_eps=POST_EPS, next_eps=NEXT_EPS, next_gain_center=NEXT_GAIN_CENTER
    )
    branch, residual, next_weight, alpha, beta = inputs
    out = (torch.empty_like(residual), torch.empty_like(branch))
    call(branch, residual, next_weight, alpha, beta, *out, **common)
    with pytest.raises(ValueError, match="alpha tensor or alpha_scalar"):
        call(branch, residual, next_weight, None, beta, *out, **common)
    with pytest.raises(ValueError, match="residual must be"):
        call(branch, residual.half(), next_weight, alpha, beta, *out, **common)
    with pytest.raises(ValueError, match="branch must be"):
        call(branch[:, :4096], residual, next_weight, alpha, beta, *out, **common)
    with pytest.raises(ValueError, match="next_weight must be"):
        call(branch, residual, next_weight.float(), alpha, beta, *out, **common)
    with pytest.raises(ValueError, match="post_weight must be"):
        call(
            branch,
            residual,
            next_weight,
            alpha,
            beta,
            *out,
            post_weight=next_weight.float(),
            **common,
        )
    misaligned = torch.empty(HIDDEN_SIZE + 8, device="cuda", dtype=torch.float32)[
        1 : 1 + HIDDEN_SIZE
    ]
    with pytest.raises(ValueError, match="beta must be"):
        call(branch, residual, next_weight, alpha, misaligned, *out, **common)
    with pytest.raises(ValueError, match="ffn_input must be"):
        call(
            branch, residual, next_weight, alpha, beta, out[0], out[1].float(), **common
        )
    with pytest.raises(ValueError, match="supports compute capability"):
        pkg.warm_up(torch.device("cpu"))
    assert norm.supports_gated_residual_next_rmsnorm(*inputs, backend="cake")
    assert not norm.supports_gated_residual_next_rmsnorm(
        branch.cpu(),
        residual.cpu(),
        next_weight.cpu(),
        alpha.cpu(),
        beta.cpu(),
        backend="cake",
    )
