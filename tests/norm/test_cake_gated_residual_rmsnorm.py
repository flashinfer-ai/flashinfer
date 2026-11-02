# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""GPU tests of the Cake gated residual RMSNorm (``flashinfer.cake_muse_spark_norm_backend``) behind
``gated_residual_rmsnorm(..., backend="cake")``: every routed program against the reference on both sides of its
window edges in both modes and both launch forms (a PDL route's windows name a PDL build, the plain build launched
without the attribute, or no program: the CuTe DSL kernel under PDL; a plain route's windows may leave a row count
to the CuTe DSL kernel too where the kernel lane measured it ahead, as the sm_107a routes do), the Cake result
against the CuTe DSL kernel's, the output aliasing the residual, the empty batch, CUDA graph replay, the fall-back
at a gap of the windows and the input checks."""

import pytest
import torch

import flashinfer.cake_muse_spark_norm_backend as pkg
from flashinfer.cake_muse_spark_norm_backend import cake_jit as loader
from flashinfer.norm import (
    gated_residual_rmsnorm,
    supports_gated_residual_rmsnorm,
)

HIDDEN_SIZE = pkg.HIDDEN_SIZE
EPS = 1e-5
OP = pkg.GATED_RESIDUAL_RMSNORM
# (weight_before_norm, enable_pdl): the four exported variants.
FORMS = ((False, False), (False, True), (True, False), (True, True))


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _arch():
    return pkg._arch(_device()) if torch.cuda.is_available() else None


def _supported() -> bool:
    return _arch() is not None and all(
        pkg.is_cake_gated_residual_rmsnorm_supported(
            _device(), weight_before_norm=weight_before_norm, enable_pdl=enable_pdl
        )
        for weight_before_norm, enable_pdl in FORMS
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the Cake gated residual RMSNorm needs an SM100 / SM103 device with every route exported",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries of this operator's routes that have no prebuilt file before the
    # first test, through the spec objects the package's loader itself builds on first use.
    if _supported():
        names = {
            name
            for route, names in loader.ROUTES[_arch()].items()
            if route.startswith(OP)
            for name in names
        }
        for spec in loader.specs_for_arch(_arch()):
            if any(spec.name.startswith(name) for name in names):
                spec.build_and_load()
    yield


def _inputs(rows: int, seed: int):
    # Drawn on the CPU so that the inputs do not depend on the device's SM count.
    generator = torch.Generator().manual_seed(seed)
    branch = torch.randn(rows, HIDDEN_SIZE, dtype=torch.bfloat16, generator=generator)
    residual = torch.randn(rows, HIDDEN_SIZE, dtype=torch.float32, generator=generator)
    weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, generator=generator) * 0.1
    gate = torch.empty(HIDDEN_SIZE, dtype=torch.float32).uniform_(
        -3, 3, generator=generator
    )
    beta = torch.sigmoid(gate)
    alpha = torch.sqrt(torch.clamp(1.0 - beta.square(), min=1e-3))
    return [tensor.cuda() for tensor in (branch, residual, weight, alpha, beta)]


def _reference(inputs, weight_before_norm: bool):
    branch, residual, weight, alpha, beta = inputs
    values = branch.float()
    if weight_before_norm:
        values *= weight.float() + 1.0
    values *= torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPS)
    return alpha * residual + beta * values


def _call(inputs, weight_before_norm: bool, enable_pdl: bool, backend="cake", **kw):
    branch, residual, weight, alpha, beta = inputs
    return gated_residual_rmsnorm(
        branch,
        residual,
        weight if weight_before_norm else None,
        alpha,
        beta,
        eps=EPS,
        gain_center=1.0 if weight_before_norm else 0.0,
        enable_pdl=enable_pdl,
        backend=backend,
        **kw,
    )


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _windows(name: str, route: str) -> list:
    """The record's windows on this device as a call of ``route`` reads them (its borrowed ones first)."""
    rule = loader.MODULES[name]["select_rule"]
    return list(rule.get("routes", {}).get(route, rule["windows"]).get(_arch(), ()))


def _edge_rows(route: str) -> list[int]:
    """1, both sides of every window edge of the route on this device (a PDL route's gaps included), and two
    off-grid row counts."""
    rows = {1, 17, 2437}
    for name in loader.route_modules(_arch(), route):
        for above, at_most in _windows(name, route):
            rows.add(int(above) + 1)
            if int(above) > 0:
                rows.add(
                    int(above)
                )  # the last row below the window: the top of a gap, if there is one
            if at_most is not None:
                rows.update((int(at_most), int(at_most) + 1))
    return sorted(rows)


def _admits(weight_before_norm: bool, enable_pdl: bool, rows: int) -> bool:
    return pkg.is_cake_gated_residual_rmsnorm_supported(
        _device(),
        weight_before_norm=weight_before_norm,
        enable_pdl=enable_pdl,
        rows=rows,
    )


def _uses_pdl(name: str, route: str, enable_pdl: bool) -> bool:
    """The launch attribute a call of ``route`` gets from the record: a PDL build's, or none for the plain build a
    PDL route borrows (its ``routes`` name the route)."""
    borrowed = route in loader.MODULES[name]["select_rule"].get("routes", {})
    return bool(enable_pdl and not borrowed)


def _cute_available(inputs) -> bool:
    return supports_gated_residual_rmsnorm(*inputs, backend="cute")


def _form_id(form):
    return f"mode{int(form[0])}{'_pdl' if form[1] else ''}"


@pytest.mark.parametrize("form", FORMS, ids=_form_id)
def test_every_window_matches_the_reference(form):
    weight_before_norm, enable_pdl = form
    route = pkg.gated_residual_rmsnorm_route(weight_before_norm, enable_pdl)
    launched = []
    original = pkg._launch

    def spy(plan, args):
        launched.append((plan.name, plan.rows))
        return original(plan, args)

    try:
        pkg._launch = spy
        for rows in _edge_rows(route):
            inputs = _inputs(rows, seed=1000 + rows)
            assert supports_gated_residual_rmsnorm(
                inputs[0],
                inputs[1],
                inputs[2] if weight_before_norm else None,
                inputs[3],
                inputs[4],
                enable_pdl=enable_pdl,
                backend="cake",
            )
            if not _admits(weight_before_norm, enable_pdl, rows):
                # a declared gap of the route (a PDL route's, or on this device the plain route's where the kernel
                # lane left the row count to the CuTe DSL kernel): the package refuses the call and ``auto`` runs
                # that kernel.  A plain route's gap is a gap of its PDL route too (the PDL route borrows the plain
                # programs where it has none of its own).
                assert enable_pdl or not _admits(weight_before_norm, True, rows), (
                    route,
                    rows,
                )
                with pytest.raises(NotImplementedError, match=f"admits {rows} rows"):
                    loader.selected_module(_arch(), route, rows)
                with pytest.raises(ValueError, match=f"admits {rows} rows"):
                    _call(inputs, weight_before_norm, enable_pdl, backend="cake")
                if _cute_available(inputs):
                    before = len(launched)
                    out = _call(inputs, weight_before_norm, enable_pdl, backend="auto")
                    _assert_close(out, _reference(inputs, weight_before_norm))
                    assert len(launched) == before
                continue
            out = _call(inputs, weight_before_norm, enable_pdl)
            _assert_close(out, _reference(inputs, weight_before_norm))
            name = loader.selected_module(_arch(), route, rows)
            assert launched[-1] == (name, rows)
            assert loader.MODULES[name]["launch"]["use_pdl"] is _uses_pdl(
                name, route, enable_pdl
            )
    finally:
        pkg._launch = original


@pytest.mark.parametrize("form", FORMS, ids=_form_id)
def test_matches_the_cute_kernel(form):
    weight_before_norm, enable_pdl = form
    inputs = _inputs(1, seed=7)
    if not supports_gated_residual_rmsnorm(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], backend="cute"
    ):
        pytest.skip("the CuTe DSL kernel is not available on this device")
    compared = 0
    for rows in (1, 17, 128, 1024, 2437):
        if not _admits(weight_before_norm, enable_pdl, rows):
            continue  # a declared gap of the route: no Cake program to compare
        inputs = _inputs(rows, seed=2000 + rows)
        cake = _call(inputs, weight_before_norm, enable_pdl, backend="cake")
        cute = _call(inputs, weight_before_norm, enable_pdl, backend="cute")
        _assert_close(cake, cute)
        compared += 1
    assert compared


@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_a_borrowed_window_launches_the_plain_build_without_the_attribute(
    weight_before_norm,
):
    """A window of the PDL route that names the plain build: the call runs a record of the plain route, whose
    launch carries no programmatic-stream-serialization attribute."""
    route = pkg.gated_residual_rmsnorm_route(weight_before_norm, True)
    borrowed = [
        (name, window)
        for name in loader.route_modules(_arch(), route)
        for window in loader.MODULES[name]["select_rule"]
        .get("routes", {})
        .get(route, {})
        .get(_arch(), ())
    ]
    if not borrowed:
        pytest.skip("no window of the PDL route borrows the plain build on this device")
    launched = []
    original = pkg._launch

    def spy(plan, args):
        launched.append((plan.name, plan.rows))
        return original(plan, args)

    try:
        pkg._launch = spy
        for name, (above, _at_most) in borrowed:
            rows = int(above) + 1
            assert loader.MODULES[name]["launch"]["use_pdl"] is False
            assert name in loader.route_modules(
                _arch(), pkg.gated_residual_rmsnorm_route(weight_before_norm, False)
            )
            inputs = _inputs(rows, seed=7000 + rows)
            out = _call(inputs, weight_before_norm, True)
            _assert_close(out, _reference(inputs, weight_before_norm))
            assert launched[-1] == (name, rows)
            assert loader.selected_module(_arch(), route, rows) == name
    finally:
        pkg._launch = original


@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_a_gap_of_the_pdl_route_falls_to_the_cute_kernel(weight_before_norm):
    """A row count the PDL route leaves to the CuTe DSL kernel while the plain route serves it: ``auto`` under PDL
    launches no Cake program, an explicit ``"cake"`` raises, and ``enable_pdl=False`` runs the plain route."""
    gaps = [
        rows
        for rows in (769, 1000, 2437, 4000, 5000, 8000)
        if _admits(weight_before_norm, False, rows)
        and not _admits(weight_before_norm, True, rows)
    ]
    if not gaps:
        pytest.skip("the PDL route leaves no gap the plain route serves on this device")
    launched = []
    original = pkg._launch

    def spy(plan, args):
        launched.append((plan.name, plan.rows))
        return original(plan, args)

    try:
        pkg._launch = spy
        for rows in gaps:
            inputs = _inputs(rows, seed=8000 + rows)
            expected = _reference(inputs, weight_before_norm)
            with pytest.raises(ValueError, match=f"admits {rows} rows"):
                _call(inputs, weight_before_norm, True, backend="cake")
            if _cute_available(inputs):
                _assert_close(
                    _call(inputs, weight_before_norm, True, backend="auto"), expected
                )
                assert launched == []
            _assert_close(
                _call(inputs, weight_before_norm, False, backend="cake"), expected
            )
            plain = pkg.gated_residual_rmsnorm_route(weight_before_norm, False)
            assert launched[-1] == (loader.selected_module(_arch(), plain, rows), rows)
            launched.clear()
    finally:
        pkg._launch = original


@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_output_may_alias_the_residual(weight_before_norm):
    inputs = _inputs(96, seed=31)
    expected = _reference(inputs, weight_before_norm)
    out = _call(inputs, weight_before_norm, True, out=inputs[1])
    assert out is inputs[1]
    _assert_close(inputs[1], expected)


def test_empty_batch_launches_nothing():
    launched = []
    original = pkg._launch
    try:
        pkg._launch = lambda *a: launched.append(a)
        inputs = _inputs(0, seed=1)
        out = _call(inputs, True, True)
        assert out.shape == (0, HIDDEN_SIZE) and out.dtype == torch.float32
    finally:
        pkg._launch = original
    assert launched == []


@pytest.mark.parametrize("form", FORMS, ids=_form_id)
def test_cuda_graph_replay_uses_changed_inputs(form):
    weight_before_norm, enable_pdl = form
    inputs = _inputs(17, seed=3)
    _call(inputs, weight_before_norm, enable_pdl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(inputs, weight_before_norm, enable_pdl)
    for seed in (3001, 4001):
        fresh = _inputs(17, seed=seed)
        inputs[0].copy_(fresh[0])
        inputs[1].copy_(fresh[1])
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(out, _reference(inputs, weight_before_norm))


def test_a_gap_of_the_windows_falls_to_the_cute_kernel(monkeypatch):
    """A row count no record of the route admits: ``auto`` runs the CuTe DSL kernel and launches no Cake program;
    an explicit ``"cake"`` raises."""
    inputs = _inputs(65, seed=65)
    if not supports_gated_residual_rmsnorm(*inputs, backend="cute"):
        pytest.skip("the CuTe DSL kernel is not available on this device")
    route = pkg.gated_residual_rmsnorm_route(True, True)
    gapped = {
        "arches": [_arch()],
        "route": route,
        "select_rule": {"kind": "rows_between", "windows": {_arch(): [[0, 64]]}},
    }
    monkeypatch.setitem(loader.MODULES, "gapped_record", gapped)
    monkeypatch.setitem(loader.ROUTES[_arch()], route, ["gapped_record"])
    monkeypatch.setattr(
        pkg, "_ADMIT_INDEX", {}
    )  # the replaced route rebuilds its admit index
    pkg.clear_launch_plans()  # plans prepared under the real registry must not serve the gapped one
    launched = []
    monkeypatch.setattr(pkg, "_launch", lambda *a: launched.append(a))
    assert supports_gated_residual_rmsnorm(*inputs, backend="cake")
    assert not pkg.is_cake_gated_residual_rmsnorm_supported(
        _device(), weight_before_norm=True, enable_pdl=True, rows=65
    )
    _assert_close(_call(inputs, True, True, backend="auto"), _reference(inputs, True))
    assert launched == []
    with pytest.raises(ValueError, match="admits 65 rows"):
        _call(inputs, True, True, backend="cake")


def test_refuses_inputs_outside_the_contract():
    branch, residual, weight, alpha, beta = _inputs(2, seed=5)
    out = torch.empty_like(residual)
    call = pkg.cake_gated_residual_rmsnorm
    call(branch, residual, weight, alpha, beta, out, eps=EPS, gain_center=1.0)
    call(branch, residual, None, alpha, beta, out, eps=EPS)
    with pytest.raises(ValueError, match="residual must be"):
        call(branch, residual.half(), weight, alpha, beta, out, eps=EPS)
    with pytest.raises(ValueError, match="branch must be"):
        call(branch[:, :4096], residual, weight, alpha, beta, out, eps=EPS)
    with pytest.raises(ValueError, match="weight must be"):
        call(branch, residual, weight.float(), alpha, beta, out, eps=EPS)
    misaligned = torch.empty(HIDDEN_SIZE + 8, device="cuda", dtype=torch.float32)[
        1 : 1 + HIDDEN_SIZE
    ]
    with pytest.raises(ValueError, match="beta must be"):
        call(branch, residual, weight, alpha, misaligned, out, eps=EPS)
    # aligned to 16 bytes but not to the package's 32: the programs read and write the FP32 planes in 256-bit vectors
    half_aligned = torch.empty(HIDDEN_SIZE + 8, device="cuda", dtype=torch.float32)[
        4 : 4 + HIDDEN_SIZE
    ]
    assert half_aligned.data_ptr() % 16 == 0 and half_aligned.data_ptr() % 32 != 0
    with pytest.raises(ValueError, match=r"beta must be .*32-byte aligned storage"):
        call(branch, residual, weight, alpha, half_aligned, out, eps=EPS)
    with pytest.raises(ValueError, match="out must be"):
        call(branch, residual, weight, alpha, beta, out.bfloat16(), eps=EPS)
    with pytest.raises(ValueError, match="supports compute capability"):
        call(
            branch.cpu(),
            residual.cpu(),
            weight.cpu(),
            alpha.cpu(),
            beta.cpu(),
            out.cpu(),
            eps=EPS,
        )
    assert supports_gated_residual_rmsnorm(
        branch, residual, weight, alpha, beta, backend="cake"
    )
    assert not supports_gated_residual_rmsnorm(
        branch.cpu(),
        residual.cpu(),
        weight.cpu(),
        alpha.cpu(),
        beta.cpu(),
        backend="cake",
    )


def test_storage_the_package_does_not_serve_falls_to_the_cute_kernel(monkeypatch):
    """Rows aligned to 16 bytes but not 32 -- storage the facade's contract admits but the package does not (its
    programs read and write the FP32 planes in 256-bit vectors) -- run the CuTe DSL kernel under ``auto`` and launch
    no Cake program, whether the residual or the output sits there; an explicit ``"cake"`` raises."""
    rows = next(
        (
            count
            for count in (17, 16, 32, 64, 128)
            if all(_admits(*form, count) for form in FORMS)
        ),
        None,
    )
    if rows is None:
        pytest.skip("no probe row count lies in every route's windows on this device")
    inputs = _inputs(rows, seed=rows)
    if not supports_gated_residual_rmsnorm(*inputs, backend="cute"):
        pytest.skip("the CuTe DSL kernel is not available on this device")
    launched = []
    monkeypatch.setattr(pkg, "_launch", lambda *a: launched.append(a))
    branch, residual, weight, alpha, beta = inputs

    def half_aligned_like(tensor: torch.Tensor) -> torch.Tensor:
        storage = torch.empty(tensor.numel() + 8, dtype=tensor.dtype, device="cuda")
        view = storage[4 : 4 + tensor.numel()].view_as(tensor)
        assert view.data_ptr() % 16 == 0 and view.data_ptr() % 32 != 0
        return view

    shifted = half_aligned_like(residual)
    shifted.copy_(residual)
    shifted_inputs = [branch, shifted, weight, alpha, beta]
    assert supports_gated_residual_rmsnorm(*shifted_inputs, backend="auto")
    assert not supports_gated_residual_rmsnorm(*shifted_inputs, backend="cake")
    for weight_before_norm, enable_pdl in FORMS:
        out = _call(shifted_inputs, weight_before_norm, enable_pdl, backend="auto")
        torch.cuda.synchronize()
        _assert_close(out, _reference(shifted_inputs, weight_before_norm))
        assert launched == []
        with pytest.raises(ValueError, match="not available"):
            _call(shifted_inputs, weight_before_norm, enable_pdl, backend="cake")
    shifted_out = half_aligned_like(residual)
    for weight_before_norm, enable_pdl in FORMS:
        _call(inputs, weight_before_norm, enable_pdl, backend="auto", out=shifted_out)
        torch.cuda.synchronize()
        _assert_close(shifted_out, _reference(inputs, weight_before_norm))
        assert launched == []
        with pytest.raises(ValueError, match="not available"):
            _call(
                inputs, weight_before_norm, enable_pdl, backend="cake", out=shifted_out
            )
    # the same calls on storage the package serves launch its programs
    _call(inputs, True, False, backend="auto")
    assert len(launched) == 1
