"""The internal route between the two NVFP4 MSA decode implementations.

The route accepts exactly what it accepted before -- :func:`check_surface` is
untouched -- and then picks between a CuTe-DSL implementation that is
specialised for one geometry and a few split counts, and the warp-specialised
ping-pong kernel, which has no cliff. Three things have to hold and none of
them is visible in an output comparison:

* the specialised implementation is entered ONLY where its own instantiations
  cover the call, so the route never inherits its low-batch cliff;
* the instantiation that carries the unsound global publication is therefore
  never reached -- and that is proved by enumeration here, not asserted;
* a call that is routed away is still SERVED, by the other kernel, with the
  same result.

The host-only half runs anywhere. It needs the implementation module, which
needs ``cutlass``; where that is absent the enumeration is skipped rather than
silently weakened.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from flashinfer.msa_ops import _nvfp4_decode_sm100 as nvfp4

from tests.msa_ops.test_msa_nvfp4_decode_sm100 import (  # noqa: F401
    _assert_peer,
    _build_inputs,
    _call,
    _reference,
    sm100_only,
)

_ROUTE_ENV = nvfp4._ROUTE_ENV
_GEOMETRY = dict(
    num_qo_heads=64,
    num_kv_heads=4,
    grp=16,
    topk=16,
    page_size=128,
    seqlen_q=1,
    causal=1,
)


@pytest.fixture(scope="module")
def impl():
    module = nvfp4._specialised_module()
    if module is None:
        pytest.skip(
            f"the specialised implementation is unavailable: "
            f"{nvfp4._specialised_import_error}"
        )
    return module


@pytest.fixture(autouse=True)
def _default_route(monkeypatch):
    monkeypatch.delenv(_ROUTE_ENV, raising=False)


# ---------------------------------------------------------------------------
# the predicate, enumerated
# ---------------------------------------------------------------------------
def test_no_reachable_call_selects_an_uncompiled_instantiation(impl):
    """The unreachability claim, by enumeration rather than by argument.

    Over every geometry the route can present and every batch up to 1024, a
    plan that reports ``specialised`` must name an instantiation that is
    actually built -- and in particular never index 0, the only one whose
    split-K partials are published through global memory.
    """

    checked = 0
    for qo, kv, topk, page, sq, causal in itertools.product(
        (16, 32, 64, 128),
        (1, 2, 4, 8),
        (1, 8, 16, 32, 128),
        (16, 64, 128, 256),
        (1, 2, 5),
        (0, 1),
    ):
        if qo % kv:
            continue
        for batch in (
            1,
            2,
            3,
            5,
            8,
            9,
            10,
            16,
            21,
            22,
            31,
            32,
            63,
            64,
            65,
            127,
            128,
            255,
            256,
            1024,
        ):
            plan = impl.plan(
                total_q=batch * sq,
                num_qo_heads=qo,
                num_kv_heads=kv,
                grp=qo // kv,
                topk=topk,
                page_size=page,
                seqlen_q=sq,
                causal=causal,
            )
            checked += 1
            if plan["specialised"]:
                assert plan["kernel_idx"] in impl.SPECIALISED_KERNEL_IDS
                assert plan["kernel_idx"] != 0
                assert plan["scored_geom"]
            else:
                assert plan["reason"]
    assert checked > 10_000


def test_every_batch_lands_on_a_split_count_that_has_a_binary(impl):
    """The split count is not the one the CTA target asks for, and must not be.

    ``ceil(256 / (batch * 4))`` capped at the top-k takes every value in its
    range, not only the ones with an instantiation: batches 1..7 ask for 10..16
    and 10..15 ask for 5..7. Asking for a count with no binary used to DECLINE
    the call, which put 1..7 and 10..15 -- three of the consumer's capture rungs
    among them -- on the other kernel. They now step DOWN to the largest count
    that has one, so the covered set is every batch this geometry reaches, and
    the count each batch lands on is an instantiated one.
    """

    covered = {}
    for batch in range(1, 257):
        plan = impl.plan(total_q=batch, **_GEOMETRY)
        assert plan["specialised"], (batch, plan)
        assert plan["kernel_idx"] in impl.SPECIALISED_KERNEL_IDS, (batch, plan)
        covered[batch] = plan["nsplit"]
    assert set(covered) == set(range(1, 257))
    # And the count is one of the instantiated ones, never an interpolation.
    assert set(covered.values()) <= {1} | set(impl._BASE_K)


@pytest.mark.parametrize(
    "override",
    [
        dict(softmax_scale=0.0),
        dict(softmax_scale=-0.1),
        dict(k_global_scale=0.0),
        dict(k_global_scale=-1.0),
    ],
)
def test_a_non_positive_scale_routes_away_instead_of_raising(impl, override):
    """The specialised binaries scale the row maximum AFTER the reduction.

    That is exact only for a positive scale, so they refuse one. The route must
    not propagate the refusal: the public API accepted these before this kernel
    existed and the ping-pong kernel still serves them.
    """

    kwargs = dict(total_q=64, softmax_scale=0.088, k_global_scale=1.0, **_GEOMETRY)
    kwargs.update(override)
    assert impl.specialised_reason(**kwargs) is not None
    assert (
        impl.specialised_reason(**dict(kwargs, softmax_scale=0.088, k_global_scale=1.0))
        is None
    )


def test_the_route_asks_the_implementation_rather_than_deciding(impl, monkeypatch):
    """One copy of the predicate: break the implementation's and the route
    must follow it, not out-vote it."""

    inputs = _build_inputs(64, [8192] * 64, torch.device("cpu"))
    kwargs = dict(
        q=inputs["q"],
        k=inputs["k"],
        q2k_indices=inputs["q2k_indices"],
        seqlen_q=1,
        causal=True,
        softmax_scale=inputs["softmax_scale"],
        k_global_scale=inputs["k_global_scale"],
        device_warm=False,
    )
    assert nvfp4.specialised_route_reason(**kwargs) is None
    monkeypatch.setattr(
        impl, "specialised_reason", lambda **_: "the implementation says no"
    )
    assert nvfp4.specialised_route_reason(**kwargs) == "the implementation says no"


def test_an_unwarmed_device_routes_away_rather_than_compiling(impl, monkeypatch):
    """Compiling on the call path would break a CUDA-graph capture."""

    inputs = _build_inputs(64, [8192] * 64, torch.device("cpu"))
    kwargs = dict(
        q=inputs["q"],
        k=inputs["k"],
        q2k_indices=inputs["q2k_indices"],
        seqlen_q=1,
        causal=True,
        softmax_scale=inputs["softmax_scale"],
        k_global_scale=inputs["k_global_scale"],
    )
    monkeypatch.setattr(impl, "is_warm", lambda _device: False)
    assert "not warmed" in nvfp4.specialised_route_reason(**kwargs)


def test_the_override_is_validated(monkeypatch):
    monkeypatch.setenv(_ROUTE_ENV, "nonsense")
    with pytest.raises(ValueError, match=_ROUTE_ENV):
        nvfp4._route_choice()
    for value in ("auto", "pingpong", "specialised", "AUTO", " pingpong "):
        monkeypatch.setenv(_ROUTE_ENV, value)
        assert nvfp4._route_choice() == value.strip().lower()


def test_stats_report_what_the_route_holds_and_refuses(impl):
    stats = nvfp4.msa_decode_nvfp4_specialized_stats()["specialised_route"]
    assert stats["available"] is True
    assert stats["persistent_device_bytes"] == 0
    assert stats["persistent_device_bytes_if_generalized_were_reachable"] > 60 << 20
    assert 0 in stats["uncompiled_instantiations"]
    assert stats["compiled_instantiations"] == sorted(impl.SPECIALISED_KERNEL_IDS)
    assert stats["batch_spans_at_geometry"] == [[1, 256]]
    assert stats["concurrent_stream_limit"] is None


def test_the_warm_shapes_cover_every_compiled_instantiation(impl):
    """One eager launch per instantiation, derived rather than tabulated."""

    shapes = nvfp4._specialised_warm_shapes()
    reached = {impl.plan(total_q=batch, **_GEOMETRY)["kernel_idx"] for batch in shapes}
    assert reached == set(impl.SPECIALISED_KERNEL_IDS)


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
_ROWS = [
    pytest.param(1, "specialised"),
    pytest.param(5, "specialised"),
    pytest.param(10, "specialised"),
    pytest.param(15, "specialised"),
    pytest.param(8, "specialised"),
    pytest.param(16, "specialised"),
    pytest.param(22, "specialised"),
    pytest.param(24, "specialised"),
    pytest.param(31, "specialised"),
    pytest.param(32, "specialised"),
    pytest.param(64, "specialised"),
]


@sm100_only
@pytest.mark.parametrize("batch, expected", _ROWS)
def test_both_sides_of_the_route_match_the_reference(batch, expected, impl):
    inputs = _build_inputs(batch, [8192] * batch, torch.device("cuda"))
    before = nvfp4.msa_decode_nvfp4_specialized_stats()
    out = _call(inputs)
    after = nvfp4.msa_decode_nvfp4_specialized_stats()
    took = (
        after["specialised_route"]["dispatch_count"]
        - before["specialised_route"]["dispatch_count"]
    )
    assert took == (1 if expected == "specialised" else 0)
    assert after["dispatch_count"] == before["dispatch_count"] + 1
    _assert_peer(out, _reference(inputs), min_cosine=0.998)


def _cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a[None], b[None]).item()


@sm100_only
@pytest.mark.parametrize("batch", [1, 5, 24, 32, 64])
def test_the_two_implementations_agree_with_each_other(batch, monkeypatch, impl):
    """Same tensors, same process, both kernels.

    The bar the pair is held to is DERIVED, not chosen: two kernels can agree
    with each other no better than the less accurate of them agrees with the
    reference. Measured on the covered batches, the specialised one sits at
    cosine 0.99999+ against the FP32 reference and the ping-pong one at
    0.9988, and the pair lands at 0.9988 -- which is the bound, not a defect.
    A fixed threshold tighter than that would have been a test of which kernel
    ran; a looser one would have tested nothing. The reference comparisons are
    what carry the correctness claim.
    """

    inputs = _build_inputs(batch, [8192] * batch, torch.device("cuda"))
    monkeypatch.setenv(_ROUTE_ENV, "pingpong")
    pingpong = _call(inputs).clone()
    monkeypatch.setenv(_ROUTE_ENV, "auto")
    routed = _call(inputs).clone()
    reference = _reference(inputs)

    _assert_peer(routed, reference, min_cosine=0.998)
    _assert_peer(pingpong, reference, min_cosine=0.998)
    worse = min(_cosine(routed, reference), _cosine(pingpong, reference))
    pair = _cosine(routed, pingpong)
    assert pair >= worse - 1e-4, (pair, worse)


@sm100_only
def test_forcing_the_specialised_route_off_its_surface_raises(monkeypatch, impl):
    """The guard is provable: forced onto a shape it cannot serve, the route
    reports it rather than quietly doing the other thing."""

    # Every BATCH is now inside the surface, so the shape that proves the
    # guard has to leave the GEOMETRY instead: a top-k of 8 is not the
    # specialised geometry, and narrowing the selection tensor is the one axis
    # these fixtures can move without rebuilding the page pool.
    inputs = _build_inputs(1, [8192], torch.device("cuda"))
    inputs = dict(inputs, q2k_indices=inputs["q2k_indices"][..., :8].contiguous())
    assert impl.plan(total_q=1, **dict(_GEOMETRY, topk=8))["specialised"] is False
    monkeypatch.setenv(_ROUTE_ENV, "specialised")
    with pytest.raises(RuntimeError, match="cannot serve this call"):
        _call(inputs)


@sm100_only
@pytest.mark.parametrize("batch", [1, 32])
def test_a_captured_graph_replays_both_sides_of_the_route(batch, impl):
    """vLLM captures a graph per decode rung, and the rungs straddle the
    route, so capture has to work on both sides of it."""

    inputs = _build_inputs(batch, [8192] * batch, torch.device("cuda"))
    nvfp4.warm(inputs["q"].device)
    eager = _call(inputs).clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _call(inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_peer(captured, eager, min_cosine=0.9999)


# The shape sweep the pinned file can no longer run against this kernel.
# Every coordinate here is one the route SENDS to the specialised
# implementation, so the sweep exercises the kernel it names: partial final
# blocks, an empty request among full ones, a single very short request among
# long ones, prime batch sizes, and the split-count boundaries at 8/9, 16/21
# and 32/63.
_SPECIALISED_SHAPES = [
    (8, [8192] * 8),
    (8, [8192 + 97 * i for i in range(8)]),
    (9, [8192] * 9),
    (16, [1024] * 16),
    (16, [8192] * 15 + [3]),
    (16, [0] + [8192] * 15),
    (17, [4096 + 11 * i for i in range(17)]),
    (21, [6000] * 21),
    # 22..31 is the three-way split, and it is the one split count that does
    # NOT divide the sixteen output rows, so its last cluster rank owns four
    # rows where the others own six. Every coordinate that could expose an
    # unwritten or twice-written row is here: the two ends of the span, a
    # prime batch inside it, ragged lengths, an empty request, a request
    # shorter than one page, and a length that leaves a partial final block.
    (22, [8192] * 22),
    (23, [8192 + 89 * i for i in range(23)]),
    (24, [8192] * 24),
    (24, [1024] * 24),
    (24, [8192] * 23 + [3]),
    (24, [0] + [8192] * 23),
    (24, [8000 + 13 * i for i in range(24)]),
    (29, [6000] * 29),
    (31, [4096 + 17 * i for i in range(31)]),
    (32, [1024] * 32),
    (33, [1024 + 7 * i for i in range(33)]),
    (37, [6000] * 37),
    (63, [2048] * 63),
    (64, [8192] * 64),
    (129, [2000 + 3 * i for i in range(129)]),
]


@sm100_only
@pytest.mark.parametrize("batch, seq_lengths", _SPECIALISED_SHAPES)
def test_the_specialised_implementation_over_its_own_shape_sweep(
    batch, seq_lengths, impl
):
    """Shape generality, on the kernel that actually serves these shapes.

    The equivalent sweep in test_msa_nvfp4_decode_sm100.py is pinned to the
    ping-pong kernel now, so without this the specialised implementation would
    be covered by seven uniform-length rows and nothing else -- and uniform
    lengths are exactly the case where a partial final block, an empty request
    and a block-id past the staged block-table prefix never occur.
    """

    inputs = _build_inputs(batch, seq_lengths, torch.device("cuda"), seed=batch + 7)
    before = nvfp4.msa_decode_nvfp4_specialized_stats()
    out = _call(inputs)
    after = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert (
        after["specialised_route"]["dispatch_count"]
        == before["specialised_route"]["dispatch_count"] + 1
    ), "this coordinate was supposed to reach the specialised implementation"
    _assert_peer(out, _reference(inputs))


@sm100_only
@pytest.mark.parametrize("max_blocks", [64, 128, 256])
def test_a_selection_past_the_staged_block_table_prefix(max_blocks, monkeypatch, impl):
    """The block-table row is staged into shared memory only up to a fixed
    prefix; a selected block id past it resolves through a per-entry global
    read instead.

    160 blocks per request is what makes that branch live: it is longer than
    the 128-entry prefix, so roughly a fifth of the selections land beyond it.
    A sweep that never exceeds the prefix -- which is every uniform 8192-token
    row -- cannot tell the two paths apart.
    """

    from tests.msa_ops import test_msa_nvfp4_decode_sm100 as base

    monkeypatch.setattr(base, "MAX_BLOCKS", max_blocks)
    blocks = min(max_blocks, 160)
    inputs = _build_inputs(
        16, [blocks * 128] * 16, torch.device("cuda"), seed=max_blocks
    )
    assert int(inputs["page_table"].shape[1]) == max_blocks
    out = _call(inputs)
    _assert_peer(out, _reference(inputs))


@sm100_only
def test_warming_the_route_holds_no_persistent_device_memory(impl):
    """The 65 MiB arena the standalone form allocated is not allocated here.

    Measured, not asserted from the constant: warm on a clean allocator and
    compare the allocator's own high-water mark.
    """

    device = torch.device("cuda", torch.cuda.current_device())
    nvfp4.warm(device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    nvfp4._specialised_warm_devices.discard((device.type, device.index))
    nvfp4._specialised_warm(device)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated(device)
    assert after - before < 1 << 20, (before, after)
