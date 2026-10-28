# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""GPU tests of the Cake per-head RMSNorm times the sigmoid gate (``flashinfer.cake_muse_spark_norm_backend``)
behind ``rmsnorm_sigmoid_gate(..., backend="cake")``: every routed program against the reference on both sides of
its window edges for the three routed head counts in both launch forms (the plain route, and the PDL route whose
windows name a PDL build, the plain build launched without the attribute, or no program), the Cake result bit for
bit against the CuTe DSL kernel's, the strided gate view of a wider projection, the 2-D call (one head, which the
package does not route), the empty batch, CUDA graph replay, the PDL route and its declared gaps, the fall-back at a
gap of the plain windows and the input checks."""

import pytest
import torch

import flashinfer.cake_muse_spark_norm_backend as pkg
from flashinfer.cake_muse_spark_norm_backend import cake_jit as loader
from flashinfer.norm import (
    rmsnorm_sigmoid_gate,
    supports_rmsnorm_sigmoid_gate,
)

HEAD_DIM = pkg.HEAD_DIM
OP = pkg.RMSNORM_SIGMOID_GATE
HEADS = (128, 32, 16)
KV_HEADS = 2
ATOL = RTOL = 0.015625
PDL_FORMS = (False, True)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _arch():
    return pkg._arch(_device()) if torch.cuda.is_available() else None


def _supported(enable_pdl: bool = False) -> bool:
    return _arch() is not None and all(
        pkg.is_cake_rmsnorm_sigmoid_gate_supported(
            _device(), heads=heads, enable_pdl=enable_pdl
        )
        for heads in HEADS
    )


def _route(enable_pdl: bool) -> str:
    return pkg.rmsnorm_sigmoid_gate_route(enable_pdl)


def _skip_without_pdl_route() -> None:
    if not pkg.is_cake_rmsnorm_sigmoid_gate_supported(_device(), enable_pdl=True):
        pytest.skip("the package ships no PDL route of the sigmoid gate on this device")


def _admits(heads: int, tokens: int, enable_pdl: bool) -> bool:
    return pkg.is_cake_rmsnorm_sigmoid_gate_supported(
        _device(), heads=heads, tokens=tokens, enable_pdl=enable_pdl
    )


def _windows(name: str, route: str) -> list:
    """The record's windows on this device as a call of ``route`` reads them (its borrowed ones first)."""
    rule = loader.MODULES[name]["select_rule"]
    return list(rule.get("routes", {}).get(route, rule["windows"]).get(_arch(), ()))


def _cute_available() -> bool:
    return supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM, backend="cute")


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the Cake RMSNorm sigmoid gate needs an SM100 / SM103 device with its route exported",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        names = set()
        for enable_pdl in PDL_FORMS:
            if loader.route_available(_arch(), _route(enable_pdl)):
                names.update(loader.route_modules(_arch(), _route(enable_pdl)))
        for spec in loader.specs_for_arch(_arch()):
            if any(spec.name.startswith(name) for name in names):
                spec.build_and_load()
    yield


def _inputs(tokens: int, heads: int, seed: int):
    """``x`` contiguous ``(tokens, heads, 64)``; ``gate`` the head rows of a ``[q, k, v, gate]`` projection of width
    ``2 * heads * 64 + 2 * KV_HEADS * 64`` (a strided view); both drawn on the CPU."""
    generator = torch.Generator().manual_seed(seed)
    q_size = heads * HEAD_DIM
    kv_size = KV_HEADS * HEAD_DIM
    projection = (
        torch.empty((tokens, 2 * q_size + 2 * kv_size), dtype=torch.bfloat16)
        .uniform_(-8.0, 8.0, generator=generator)
        .cuda()
    )
    gate = projection[:, q_size + 2 * kv_size : 2 * q_size + 2 * kv_size].view(
        tokens, heads, HEAD_DIM
    )
    x = torch.randn(
        (tokens, heads, HEAD_DIM), dtype=torch.bfloat16, generator=generator
    ).cuda()
    return x, gate, projection


def _reference(x: torch.Tensor, gate: torch.Tensor, eps: float):
    values = x.float()
    inv_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (values * inv_rms * torch.sigmoid(gate.float())).bfloat16()


def _edge_tokens(heads: int, enable_pdl: bool) -> list[int]:
    """1, 2, 17 and both sides of every window edge of the head count on this device for the launch form's
    route (a PDL route's edges include the token counts just inside and outside its gaps)."""
    tokens = {1, 2, 17}
    route = _route(enable_pdl)
    for name in loader.route_modules(_arch(), route):
        for count, above, at_most in _windows(name, route):
            if int(count) != heads:
                continue
            tokens.update((int(above) + 1,))
            if at_most is not None:
                tokens.update((int(at_most), int(at_most) + 1))
    return sorted(tokens)


def _uses_pdl(name: str, enable_pdl: bool) -> bool:
    """The launch attribute a call of the form's route gets from the record: a PDL build's, or none for the plain
    build a PDL route borrows (its ``routes`` name the route)."""
    record = loader.MODULES[name]
    borrowed = _route(enable_pdl) in record["select_rule"].get("routes", {})
    return bool(enable_pdl and not borrowed)


@pytest.mark.parametrize("enable_pdl", PDL_FORMS, ids=["plain", "pdl"])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
@pytest.mark.parametrize("heads", HEADS)
def test_every_window_matches_the_reference(heads, eps, enable_pdl):
    if enable_pdl:
        _skip_without_pdl_route()
    route = _route(enable_pdl)
    launched = []
    original = pkg._launch

    def spy(plan, args):
        launched.append((plan.name, plan.rows))
        return original(plan, args)

    try:
        pkg._launch = spy
        for tokens in _edge_tokens(heads, enable_pdl):
            x, gate, projection = _inputs(tokens, heads, seed=919 + tokens)
            assert gate.stride()[1:] == (HEAD_DIM, 1)
            if tokens > 1:
                # one token's view takes the compact stride; from two on it is the projection's
                assert gate.stride(0) == projection.stride(0)
            expected = _reference(x, gate, eps)
            if not _admits(heads, tokens, enable_pdl):
                # a declared gap of the route (the PDL route's, or the plain route's where its table leaves the
                # token count to the CuTe DSL kernel): the package refuses the call and ``auto`` runs that kernel
                with pytest.raises(
                    ValueError, match=f"admits {tokens} tokens of {heads} heads"
                ):
                    rmsnorm_sigmoid_gate(
                        x, gate, eps, enable_pdl=enable_pdl, backend="cake"
                    )
                if _cute_available():
                    before = len(launched)
                    rmsnorm_sigmoid_gate(
                        x, gate, eps, enable_pdl=enable_pdl, backend="auto"
                    )
                    torch.cuda.synchronize()
                    torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
                    assert len(launched) == before
                continue
            pointer = x.data_ptr()
            rmsnorm_sigmoid_gate(x, gate, eps, enable_pdl=enable_pdl, backend="cake")
            torch.cuda.synchronize()
            assert x.data_ptr() == pointer
            torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
            name = loader.selected_module(_arch(), route, tokens, heads)
            assert launched[-1] == (name, tokens)
            assert loader.MODULES[name]["launch"]["use_pdl"] is _uses_pdl(
                name, enable_pdl
            )
    finally:
        pkg._launch = original


@pytest.mark.parametrize("enable_pdl", PDL_FORMS, ids=["plain", "pdl"])
@pytest.mark.parametrize("heads", HEADS)
def test_matches_the_cute_kernel_bit_for_bit(heads, enable_pdl):
    if not _cute_available():
        pytest.skip("the CuTe DSL kernel is not available on this device")
    if enable_pdl:
        _skip_without_pdl_route()
    compared = 0
    for tokens in (1, 8, 9, 17, 256, 257, 1024, 1025):
        if not _admits(heads, tokens, enable_pdl):
            continue  # a gap of the PDL route: no Cake program to compare
        x, gate, _ = _inputs(tokens, heads, seed=5000 + tokens)
        cake, cute = x.clone(), x.clone()
        rmsnorm_sigmoid_gate(cake, gate, 1e-5, enable_pdl=enable_pdl, backend="cake")
        rmsnorm_sigmoid_gate(cute, gate, 1e-5, enable_pdl=enable_pdl, backend="cute")
        torch.cuda.synchronize()
        assert torch.equal(cake, cute), (heads, tokens)
        compared += 1
    assert compared


def test_a_borrowed_window_launches_the_plain_build_without_the_attribute():
    """A window of the PDL route that names the plain build: the call runs the plain route's record, whose launch
    carries no programmatic-stream-serialization attribute."""
    _skip_without_pdl_route()
    route = _route(True)
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
        for name, (count, above, _at_most) in borrowed:
            heads, tokens = int(count), int(above) + 1
            assert loader.MODULES[name]["launch"]["use_pdl"] is False
            assert name in loader.route_modules(_arch(), _route(False))
            x, gate, _ = _inputs(tokens, heads, seed=7000 + tokens)
            expected = _reference(x, gate, 1e-5)
            rmsnorm_sigmoid_gate(x, gate, 1e-5, enable_pdl=True, backend="cake")
            torch.cuda.synchronize()
            torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
            assert launched[-1] == (name, tokens)
            assert loader.selected_module(_arch(), route, tokens, heads) == name
    finally:
        pkg._launch = original


def _gaps(heads: int) -> list[tuple[int, int]]:
    """The PDL route's declared gaps of the head count on this device: the ``(above, at_most]`` token ranges no
    record's window admits, read from the registry (the windows and gaps partition the positive token counts)."""
    route = _route(True)
    windows = sorted(
        (int(above), None if at_most is None else int(at_most))
        for name in loader.route_modules(_arch(), route)
        for count, above, at_most in _windows(name, route)
        if int(count) == heads
    )
    gaps: list[tuple[int, int]] = []
    edge = 0
    for above, at_most in windows:
        assert edge is not None and above >= edge, (heads, windows)
        if above > edge:
            gaps.append((edge, above))
        edge = at_most
    assert edge is None, (heads, windows)
    return gaps


def test_the_pdl_route_admits_every_token_count_the_plain_route_admits_but_its_gaps():
    """The PDL route's windows and its declared gaps partition the token count for every head count the plain
    route names: an explicit ``"cake"`` call under PDL runs a Cake program wherever ``enable_pdl=False`` does,
    from the launch form the window names (its PDL build, its late-signal PDL build or the plain build without
    the attribute) -- except in a gap, where the predicate answers False and the package refuses the call
    (``auto`` keeps the CuTe DSL kernel under PDL there).  Where the plain route itself declares a gap (a table
    may leave token counts to the CuTe DSL kernel on both routes), its predicate answers False too; the PDL
    table still partitions every positive token count into windows and declared gaps."""
    _skip_without_pdl_route()
    launched = []
    original = pkg._launch

    def spy(plan, args):
        launched.append((plan.name, plan.rows))
        return original(plan, args)

    try:
        pkg._launch = spy
        for heads in HEADS:
            gaps = _gaps(heads)
            edges = {
                tokens for low, high in gaps for tokens in (low + 1, high, high + 1)
            }
            for tokens in sorted(
                {1, 2, 7, 8, 9, 64, 65, 255, 256, 257, 1023, 1024, 1025, 4096} | edges
            ):
                in_gap = any(low < tokens <= high for low, high in gaps)
                assert _admits(heads, tokens, True) is (not in_gap), (heads, tokens)
            for tokens in (2, 258, 1025):
                assert _admits(heads, tokens, True), (heads, tokens)
                x, gate, _ = _inputs(tokens, heads, seed=8000 + tokens)
                expected = _reference(x, gate, 1e-5)
                rmsnorm_sigmoid_gate(x, gate, 1e-5, enable_pdl=True, backend="cake")
                torch.cuda.synchronize()
                torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
                name = loader.selected_module(_arch(), _route(True), tokens, heads)
                assert launched[-1] == (name, tokens)
                assert loader.MODULES[name]["launch"]["use_pdl"] is _uses_pdl(
                    name, True
                )
                launched.clear()
            for low, high in gaps:
                for tokens in (low + 1, high):
                    x, gate, _ = _inputs(tokens, heads, seed=9000 + tokens)
                    with pytest.raises(
                        ValueError, match=f"admits {tokens} tokens of {heads} heads"
                    ):
                        rmsnorm_sigmoid_gate(
                            x, gate, 1e-5, enable_pdl=True, backend="cake"
                        )
                    assert not launched
    finally:
        pkg._launch = original


def test_contiguous_gate_and_differing_token_strides():
    # the first token count from 300 the plain route admits at 32 heads on this device (a table may leave 300
    # to the CuTe DSL kernel)
    tokens = next((t for t in range(300, 4097) if _admits(32, t, False)), None)
    if tokens is None:
        pytest.skip("the plain route admits no token count in 300..4096 at 32 heads")
    x, gate, _ = _inputs(tokens, 32, seed=11)
    dense = gate.contiguous()
    expected = _reference(x, dense, 1e-5)
    strided, contiguous = x.clone(), x.clone()
    rmsnorm_sigmoid_gate(strided, gate, 1e-5, enable_pdl=False, backend="cake")
    rmsnorm_sigmoid_gate(contiguous, dense, 1e-5, enable_pdl=False, backend="cake")
    torch.cuda.synchronize()
    torch.testing.assert_close(contiguous, expected, atol=ATOL, rtol=RTOL)
    assert torch.equal(strided, contiguous)


def test_a_two_d_call_is_one_head_and_falls_to_the_cute_kernel(monkeypatch):
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM, backend="cute"):
        pytest.skip("the CuTe DSL kernel is not available on this device")
    x, gate, _ = _inputs(64, 32, seed=13)
    rows, rows_gate = x.view(-1, HEAD_DIM).clone(), gate.reshape(-1, HEAD_DIM).clone()
    expected = _reference(rows, rows_gate, 1e-5)
    launched = []
    monkeypatch.setattr(pkg, "_launch", lambda *a: launched.append(a))
    for enable_pdl in PDL_FORMS:
        assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
            _device(), heads=1, enable_pdl=enable_pdl
        )
        probe = rows.clone()
        rmsnorm_sigmoid_gate(
            probe, rows_gate, 1e-5, enable_pdl=enable_pdl, backend="auto"
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(probe, expected, atol=ATOL, rtol=RTOL)
        assert launched == []
        if enable_pdl and not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
            _device(), enable_pdl=True
        ):
            continue  # no PDL route at all: the facade refuses the backend before the package sees the call
        with pytest.raises(ValueError, match="admits 2048 tokens of 1 heads"):
            rmsnorm_sigmoid_gate(
                rows, rows_gate, 1e-5, enable_pdl=enable_pdl, backend="cake"
            )


def test_empty_batch_launches_nothing():
    launched = []
    original = pkg._launch
    try:
        pkg._launch = lambda *a: launched.append(a)
        x, gate, _ = _inputs(0, 32, seed=1)
        rmsnorm_sigmoid_gate(x, gate, 1e-5, enable_pdl=False, backend="cake")
        if pkg.is_cake_rmsnorm_sigmoid_gate_supported(_device(), enable_pdl=True):
            rmsnorm_sigmoid_gate(x, gate, 1e-5, enable_pdl=True, backend="cake")
    finally:
        pkg._launch = original
    assert launched == []


@pytest.mark.parametrize("enable_pdl", PDL_FORMS, ids=["plain", "pdl"])
@pytest.mark.parametrize("heads", HEADS)
def test_cuda_graph_replay_uses_changed_inputs(heads, enable_pdl):
    if enable_pdl:
        _skip_without_pdl_route()
    admitted = [t for t in (64, 256, 1024, 8, 2) if _admits(heads, t, enable_pdl)]
    if not admitted:
        pytest.skip("no window of the route admits the probe token counts")
    tokens = admitted[0]
    static_x, static_gate, projection = _inputs(tokens, heads, seed=1001)
    rmsnorm_sigmoid_gate(
        static_x, static_gate, 1e-5, enable_pdl=enable_pdl, backend="cake"
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsnorm_sigmoid_gate(
            static_x, static_gate, 1e-5, enable_pdl=enable_pdl, backend="cake"
        )
    for seed in (2003, 3001):
        next_x, next_gate, next_projection = _inputs(tokens, heads, seed=seed)
        expected = _reference(next_x, next_gate, 1e-5)
        static_x.copy_(next_x)
        projection.copy_(next_projection)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_x, expected, atol=ATOL, rtol=RTOL)


def test_a_gap_of_the_windows_falls_to_the_cute_kernel(monkeypatch):
    """A token count no record of the plain route admits for the head count: ``auto`` runs the CuTe DSL kernel
    and launches no Cake program; an explicit ``"cake"`` raises."""
    if not _cute_available():
        pytest.skip("the CuTe DSL kernel is not available on this device")
    route = _route(False)
    gapped = {
        "arches": [_arch()],
        "route": route,
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {_arch(): [[32, 0, 8]]},
        },
    }
    monkeypatch.setitem(loader.MODULES, "gapped_record", gapped)
    monkeypatch.setitem(loader.ROUTES[_arch()], route, ["gapped_record"])
    monkeypatch.setattr(
        pkg, "_ADMIT_INDEX", {}
    )  # the replaced route rebuilds its admit index
    pkg.clear_launch_plans()  # plans prepared under the real registry must not serve the gapped one
    launched = []
    monkeypatch.setattr(pkg, "_launch", lambda *a: launched.append(a))
    x, gate, _ = _inputs(9, 32, seed=9)
    expected = _reference(x, gate, 1e-5)
    assert supports_rmsnorm_sigmoid_gate(
        torch.bfloat16, HEAD_DIM, backend="cake", enable_pdl=False
    )
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(_device(), heads=32, tokens=9)
    rmsnorm_sigmoid_gate(x, gate, 1e-5, enable_pdl=False, backend="auto")
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
    assert launched == []
    with pytest.raises(ValueError, match="admits 9 tokens of 32 heads"):
        rmsnorm_sigmoid_gate(x, gate, 1e-5, enable_pdl=False, backend="cake")


def test_refuses_inputs_outside_the_contract():
    x, gate, _ = _inputs(2, 32, seed=2)
    call = pkg.cake_rmsnorm_sigmoid_gate
    call(x.clone(), gate, eps=1e-5)
    with pytest.raises(ValueError, match="must be a bfloat16"):
        call(x.half(), gate.half(), eps=1e-5)
    with pytest.raises(ValueError, match="gate must have the shape"):
        call(x, gate[:1], eps=1e-5)
    wide = torch.empty(2, 32, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"must be \[tokens, 64\]"):
        call(wide, wide, eps=1e-5)
    storage = torch.empty(2 * 32 * HEAD_DIM + 16, dtype=torch.bfloat16, device="cuda")
    misaligned = storage[1 : 1 + 2 * 32 * HEAD_DIM].view(2, 32, HEAD_DIM)
    with pytest.raises(ValueError, match="aligned storage"):
        call(misaligned, gate, eps=1e-5)
    # aligned to 16 bytes but not to the package's 32: the 16-head programs read and write 256-bit vectors
    half_aligned = storage[8 : 8 + 2 * 32 * HEAD_DIM].view(2, 32, HEAD_DIM)
    assert half_aligned.data_ptr() % 16 == 0 and half_aligned.data_ptr() % 32 != 0
    with pytest.raises(ValueError, match="32-byte aligned storage"):
        call(half_aligned, gate, eps=1e-5)
    odd = torch.empty(2, 68, dtype=torch.bfloat16, device="cuda")[:, :HEAD_DIM]
    with pytest.raises(ValueError, match="multiple of 16 elements"):
        call(odd, odd, eps=1e-5)
    # a token stride of 16-byte multiples only (72 elements): the package serves 32-byte multiples
    padded = torch.empty(2, 72, dtype=torch.bfloat16, device="cuda")[:, :HEAD_DIM]
    with pytest.raises(ValueError, match="multiple of 16 elements"):
        call(padded, padded, eps=1e-5)
    with pytest.raises(ValueError, match="supports compute capability"):
        call(x.cpu(), gate.cpu(), eps=1e-5)
    assert supports_rmsnorm_sigmoid_gate(
        torch.bfloat16, HEAD_DIM, backend="cake", enable_pdl=False
    )
    assert supports_rmsnorm_sigmoid_gate(
        torch.bfloat16, HEAD_DIM, backend="cake", enable_pdl=True
    ) is pkg.is_cake_rmsnorm_sigmoid_gate_supported(_device(), enable_pdl=True)
    assert not supports_rmsnorm_sigmoid_gate(
        torch.bfloat16, HEAD_DIM, device=torch.device("cpu"), backend="cake"
    )


@pytest.mark.parametrize("enable_pdl", PDL_FORMS, ids=["plain", "pdl"])
def test_storage_the_package_does_not_serve_falls_to_the_cute_kernel(
    monkeypatch, enable_pdl
):
    """Storage the facade's contract admits but the package does not -- rows aligned to 16 bytes but not 32, or a
    token stride that is a 16-byte multiple but not a 32-byte one -- runs the CuTe DSL kernel under ``auto`` and
    launches no Cake program (the package's 16-head programs of sm_107a read and write 256-bit vectors; the package
    admits one alignment for every program); an explicit ``"cake"`` raises.  Every head count, at a token count of
    each window class the route admits (the 16-head 256-bit programs of sm_107a sit above 2047 tokens)."""
    if not _cute_available():
        pytest.skip("the CuTe DSL kernel is not available on this device")
    if enable_pdl:
        _skip_without_pdl_route()
    launched = []
    monkeypatch.setattr(pkg, "_launch", lambda *a: launched.append(a))
    checked = 0
    for heads in HEADS:
        for tokens in (17, 4096, 9000):
            if not _admits(heads, tokens, enable_pdl):
                continue
            x, gate, _ = _inputs(tokens, heads, seed=tokens + heads)
            numel = tokens * heads * HEAD_DIM
            storage = torch.empty(numel + 16, dtype=torch.bfloat16, device="cuda")
            shifted = storage[8 : 8 + numel].view(tokens, heads, HEAD_DIM)
            assert shifted.data_ptr() % 16 == 0 and shifted.data_ptr() % 32 != 0
            padded = torch.empty(
                (tokens, heads * HEAD_DIM + 8), dtype=torch.bfloat16, device="cuda"
            )
            strided = padded[:, : heads * HEAD_DIM].view(tokens, heads, HEAD_DIM)
            assert strided.stride(0) % 8 == 0 and strided.stride(0) % 16 != 0
            for view in (shifted, strided):
                view.copy_(x)
                expected = _reference(view, gate, 1e-5)
                rmsnorm_sigmoid_gate(
                    view, gate, 1e-5, enable_pdl=enable_pdl, backend="auto"
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(view, expected, atol=ATOL, rtol=RTOL)
                assert launched == []
                with pytest.raises(ValueError, match="not available"):
                    rmsnorm_sigmoid_gate(
                        view, gate, 1e-5, enable_pdl=enable_pdl, backend="cake"
                    )
                checked += 1
            # the same call on storage the package serves launches its program
            rmsnorm_sigmoid_gate(x, gate, 1e-5, enable_pdl=enable_pdl, backend="auto")
            assert len(launched) == 1
            launched.clear()
    assert checked >= 2 * len(HEADS)
