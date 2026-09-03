"""Host-only unit tests for NcclEpHandle's host-path caching layer.

Covers the fleet-anchored hot cache (``NcclEpFleet._hot_cache``) that
``NcclEpHandle`` populates and consumes: the ``_wrap`` FFI-descriptor memo,
cross-handle recv-buffer reuse, cache invalidation on dtype change and
``update_topology``, the HT forward-time token cap, and the
``HandleAlgoKnobNumReceivedTokens`` (recv-count) opt-in.

Everything runs on CPU tensors against the fake ``nccl.ep`` from
``conftest.py`` — the fake handle records calls instead of communicating, so
no GPU or nccl4py wheel is needed.
"""

from __future__ import annotations

import pytest


def _make_fleet(fake_nccl_ep, *, algorithm=None, world=4, max_tokens=128, hidden=2048):
    from flashinfer.moe_ep.config import BootstrapConfig, EpAlgorithm, FleetParams
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    params = FleetParams(
        num_experts=2 * world,
        max_tokens_per_rank=max_tokens,
        token_hidden_size=hidden,
        dtype_bytes=2,
        algorithm=algorithm if algorithm is not None else EpAlgorithm.LOW_LATENCY,
    )
    return NcclEpFleet(BootstrapConfig(world_size=world, rank=0), params)


def _make_handle(fleet, *, num_tokens=16, top_k=2, with_weights=True, extra_knobs=()):
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.config import HandleParams

    knobs = list(extra_knobs)
    if with_weights:
        knobs.append(
            HandleAlgoKnobTopKWeights(
                weights=torch.ones(num_tokens, top_k, dtype=torch.float32)
            )
        )
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int64)
    return fleet.create_handle(HandleParams(topk_ids=topk_ids), algo_knobs=knobs)


def _dispatch(handle, x):
    from flashinfer.moe_ep.config import DispatchInputParams

    return handle.dispatch(DispatchInputParams(x=[x]))


# -------------------------------------------------------------------- _wrap


def test_wrap_memoizes_small_tensors_by_address(fake_nccl_ep, bypass_build_checks):
    import torch

    handle = _make_handle(_make_fleet(fake_nccl_ep))
    t = torch.zeros(8, 8)

    w1 = handle._wrap(t)
    w2 = handle._wrap(t)
    assert w1 is w2


def test_wrap_misses_on_shape_change_at_same_address(fake_nccl_ep, bypass_build_checks):
    """A recycled address with a different shape must build a fresh wrapper —
    this is the property that makes address-keyed memoization alias-safe."""
    import torch

    handle = _make_handle(_make_fleet(fake_nccl_ep))
    t = torch.zeros(4, 4)
    v = t.view(16)
    assert v.data_ptr() == t.data_ptr()

    assert handle._wrap(t) is not handle._wrap(v)
    assert handle._wrap(t).shape != handle._wrap(v).shape


def test_wrap_never_caches_large_tensors(fake_nccl_ep, bypass_build_checks):
    import torch

    handle = _make_handle(_make_fleet(fake_nccl_ep))
    big = torch.empty(handle._WRAP_MEMO_MAX_BYTES + 8, dtype=torch.uint8)

    w1 = handle._wrap(big)
    w2 = handle._wrap(big)
    assert w1 is not w2
    key = (big.data_ptr(), big.dtype, tuple(big.shape))
    assert key not in handle._hot


def test_wrap_bounds_cache_size_by_clearing(fake_nccl_ep, bypass_build_checks):
    import torch

    handle = _make_handle(_make_fleet(fake_nccl_ep))
    keepalive = [torch.zeros(1) for _ in range(2 * handle._WRAP_MEMO_MAX_ENTRIES)]
    for t in keepalive:
        handle._wrap(t)

    # The bound is enforced by clearing, so the cache never grows unbounded
    # (a small overshoot past MAX_ENTRIES before the clear triggers is fine).
    assert len(handle._hot) <= handle._WRAP_MEMO_MAX_ENTRIES + 2


# ----------------------------------------------------- LL dispatch hot cache


def test_ll_dispatch_reuses_fleet_cached_buffers_across_handles(
    fake_nccl_ep, bypass_build_checks
):
    """vLLM creates a fresh Handle every forward; the recv buffer and the FFI
    descriptor objects must come from the fleet-level cache, not per-handle
    state — NCCL-EP caches dispatch by buffer address, so a silently changed
    address would deadlock the next collective."""
    import torch

    fleet = _make_fleet(fake_nccl_ep)
    x = torch.zeros(16, 64, dtype=torch.bfloat16)

    h1 = _make_handle(fleet)
    out1 = _dispatch(h1, x)
    h2 = _make_handle(fleet)
    out2 = _dispatch(h2, x)

    assert out1.expert_tensors is out2.expert_tensors
    assert fleet._hot_cache["ll_recv_buf"] is out1.expert_tensors
    # The cached DispatchOutputs FFI object is reused verbatim.
    d1 = next(c for c in fake_nccl_ep._log["handles"][0].calls if c[0] == "dispatch")
    d2 = next(c for c in fake_nccl_ep._log["handles"][1].calls if c[0] == "dispatch")
    assert d1[2] is d2[2]
    # Both handles share the fleet-cached recv-count tensor.
    assert out1.expert_counts is out2.expert_counts
    assert out1.num_tokens == 128 * 4


def test_ll_dispatch_rebuilds_cache_on_dtype_change(fake_nccl_ep, bypass_build_checks):
    import torch

    fleet = _make_fleet(fake_nccl_ep)
    out_bf16 = _dispatch(_make_handle(fleet), torch.zeros(16, 64, dtype=torch.bfloat16))
    out_fp32 = _dispatch(_make_handle(fleet), torch.zeros(16, 64, dtype=torch.float32))

    assert out_bf16.expert_tensors is not out_fp32.expert_tensors
    assert out_fp32.expert_tensors.dtype == torch.float32


def test_ll_dispatch_completes_and_marks_send_only_when_staged(
    fake_nccl_ep, bypass_build_checks
):
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobSplitOperation

    fleet = _make_fleet(fake_nccl_ep)
    x = torch.zeros(16, 64, dtype=torch.bfloat16)

    _dispatch(_make_handle(fleet), x)
    plain = fake_nccl_ep._log["handles"][0]
    disp = next(c for c in plain.calls if c[0] == "dispatch")
    assert disp[3]["config"].send_only == 0
    assert any(c[0] == "complete" for c in plain.calls)

    _dispatch(_make_handle(fleet, extra_knobs=[HandleAlgoKnobSplitOperation()]), x)
    staged = fake_nccl_ep._log["handles"][1]
    disp = next(c for c in staged.calls if c[0] == "dispatch")
    assert disp[3]["config"].send_only == 1


def test_update_topology_clears_hot_cache(fake_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep.config import BootstrapConfig

    fleet = _make_fleet(fake_nccl_ep, world=4)
    _dispatch(_make_handle(fleet), torch.zeros(16, 64, dtype=torch.bfloat16))
    assert fleet._hot_cache  # populated by the dispatch

    fleet.update_topology(BootstrapConfig(world_size=2, rank=0))

    assert fleet._hot_cache == {}
    assert len(fake_nccl_ep._log["groups"]) == 2  # group re-created


# ------------------------------------------------------------------ combine


def test_ll_combine_requires_topk_weights(fake_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep.config import CombineInputParams

    fleet = _make_fleet(fake_nccl_ep)
    handle = _make_handle(fleet, with_weights=False)
    out = _dispatch(handle, torch.zeros(16, 64, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="HandleAlgoKnobTopKWeights"):
        handle.combine(CombineInputParams(x=[out.expert_tensors]))


def test_ll_combine_reuses_cached_config(fake_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep.config import CombineInputParams

    fleet = _make_fleet(fake_nccl_ep)
    x = torch.zeros(16, 64, dtype=torch.bfloat16)

    h1 = _make_handle(fleet)
    h1.combine(CombineInputParams(x=[_dispatch(h1, x).expert_tensors]))
    h2 = _make_handle(fleet)
    h2.combine(CombineInputParams(x=[_dispatch(h2, x).expert_tensors]))

    c1 = next(c for c in fake_nccl_ep._log["handles"][0].calls if c[0] == "combine")
    c2 = next(c for c in fake_nccl_ep._log["handles"][1].calls if c[0] == "combine")
    assert c1[3]["config"] is c2[3]["config"]  # ("ll_comb_cfg", staged) hit


# ------------------------------------------------------------------ HT paths


def test_ht_dispatch_rejects_token_overflow(fake_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.config import EpAlgorithm

    fleet = _make_fleet(
        fake_nccl_ep, algorithm=EpAlgorithm.HIGH_THROUGHPUT, max_tokens=128
    )
    handle = _make_handle(fleet, num_tokens=129)

    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        _dispatch(handle, torch.zeros(129, 64, dtype=torch.bfloat16))


def test_ht_dispatch_within_cap_caches_recv_bufs(fake_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep.config import EpAlgorithm

    fleet = _make_fleet(
        fake_nccl_ep, algorithm=EpAlgorithm.HIGH_THROUGHPUT, max_tokens=128
    )
    x = torch.zeros(128, 64, dtype=torch.bfloat16)

    out1 = _dispatch(_make_handle(fleet, num_tokens=128), x)
    out2 = _dispatch(_make_handle(fleet, num_tokens=128), x)

    assert out1.recv_total_counter is None  # knob not set
    assert out1.num_tokens == 128 * 4
    cached = fleet._hot_cache["ht_recv_bufs"]
    assert out1.expert_tensors.data_ptr() == cached[0].data_ptr()
    assert out2.expert_tensors.data_ptr() == cached[0].data_ptr()


def test_ht_recv_count_knob_binds_layout_info_and_output(
    fake_nccl_ep, bypass_build_checks
):
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobNumReceivedTokens
    from flashinfer.moe_ep.config import EpAlgorithm

    fleet = _make_fleet(fake_nccl_ep, algorithm=EpAlgorithm.HIGH_THROUGHPUT)
    target = torch.zeros(1, dtype=torch.int32)
    handle = _make_handle(
        fleet, extra_knobs=[HandleAlgoKnobNumReceivedTokens(target=target)]
    )

    layout_info = fake_nccl_ep._log["handles"][-1].create_kwargs["layout_info"]
    assert layout_info is not None
    assert layout_info.recv_total_counter.buffer is target

    out = _dispatch(handle, torch.zeros(16, 64, dtype=torch.bfloat16))
    assert out.recv_total_counter is target


def test_ht_recv_count_knob_rejects_bad_dtype(fake_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobNumReceivedTokens
    from flashinfer.moe_ep.config import EpAlgorithm

    fleet = _make_fleet(fake_nccl_ep, algorithm=EpAlgorithm.HIGH_THROUGHPUT)
    bad = torch.zeros(1, dtype=torch.float32)

    with pytest.raises(ValueError, match="int32 or int64"):
        _make_handle(fleet, extra_knobs=[HandleAlgoKnobNumReceivedTokens(target=bad)])


def test_ll_ignores_recv_count_knob(fake_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobNumReceivedTokens

    fleet = _make_fleet(fake_nccl_ep)
    target = torch.zeros(1, dtype=torch.int32)
    _make_handle(fleet, extra_knobs=[HandleAlgoKnobNumReceivedTokens(target=target)])

    # LL rejects handle-time layout_info in the C library — the knob must not
    # leak into create_handle there.
    assert fake_nccl_ep._log["handles"][-1].create_kwargs["layout_info"] is None


# ---------------------------------------------------------------- Handle.update
#
# update() exists so ONE handle can serve many forwards. That is what makes CUDA
# graph capture possible: a graph records the device pointers it sees, so a
# handle created and destroyed inside the captured forward leaves the replay
# pointing at freed memory (observed on 4xB200 as an illegal memory access at
# replay, with capture itself succeeding). Creating the handle outside the
# capture and calling update() inside mirrors NCCL-EP's own graph recipe.


def test_update_rebinds_routing_without_recreating_the_handle(
    fake_nccl_ep, bypass_build_checks
):
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    n_handles = len(fake_nccl_ep._log["handles"])

    second = torch.ones(16, 2, dtype=torch.int64, device="cuda")
    h.update(HandleParams(topk_ids=second))

    # No new native handle: the point is that the old one stays valid, so a
    # captured graph's pointers remain live.
    assert len(fake_nccl_ep._log["handles"]) == n_handles
    native = fake_nccl_ep._log["handles"][-1]
    assert [c[0] for c in native.calls].count("update") == 1


def test_update_rejects_a_different_top_k(fake_nccl_ep, bypass_build_checks):
    """top_k is bound at InitHandle; changing it would need new buffers."""
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    with pytest.raises(ValueError, match="cannot change top_k"):
        h.update(
            HandleParams(topk_ids=torch.zeros(16, 4, dtype=torch.int64, device="cuda"))
        )


@pytest.mark.parametrize("tokens", [8, 32])
def test_update_rejects_a_different_token_count(
    fake_nccl_ep, bypass_build_checks, tokens
):
    """Neither growing nor shrinking is allowed.

    Growing would overrun buffers sized at creation. Shrinking is subtler and
    was allowed at first: the per-token weights bound via
    HandleAlgoKnobTopKWeights stay at the creating shape, so a shorter
    topk_ids leaves combine reading weights for rows that no longer exist.
    """
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    with pytest.raises(ValueError, match="cannot change the token count"):
        h.update(
            HandleParams(
                topk_ids=torch.zeros(tokens, 2, dtype=torch.int64, device="cuda")
            )
        )


def test_update_rejects_non_contiguous_routing(fake_nccl_ep, bypass_build_checks):
    """A strided view would be read as if it were packed."""
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    strided = torch.zeros(16, 4, dtype=torch.int64, device="cuda")[:, ::2]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        h.update(HandleParams(topk_ids=strided))


def test_dispatch_rejects_activation_count_mismatch(fake_nccl_ep, bypass_build_checks):
    """Activations must agree with the routing the handle currently holds."""
    import torch

    from flashinfer.moe_ep.config import DispatchInputParams
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    wrong = torch.zeros(8, 2048, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(MoEEpConfigError, match="activation rows"):
        h.dispatch(DispatchInputParams(x=[wrong]))


def test_update_is_optional_on_the_base_handle():
    """Backends that cannot rebind must say so, not silently no-op."""
    from flashinfer.moe_ep.core.comm.handle import Handle

    class _Bare(Handle):
        def dispatch(self, params):
            raise NotImplementedError

        def combine(self, params):
            raise NotImplementedError

        def complete(self):
            pass

    with pytest.raises(NotImplementedError, match="update"):
        _Bare().update(None)


def test_update_is_capturable(fake_nccl_ep, bypass_build_checks):
    """update() must contain no host sync -- capture fails outright on one.

    This is the property that separates update() from the HT prepare path,
    which is uncapturable precisely because it does int(recv_total.item()).
    A .item()/.cpu()/.tolist() added to update() would raise here rather than
    surviving to become an illegal memory access at replay on real hardware.

    Scope: the fake handle issues no device work, so this pins the Python
    path only. End-to-end capture over real NCCL-EP kernels is covered by
    tests/moe_ep/test_moe_ep_cudagraph_multirank.py.
    """
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    topk = torch.zeros(16, 2, dtype=torch.int64, device="cuda")

    # Warm up on a side stream first, as torch requires before capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        h.update(HandleParams(topk_ids=topk))
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        h.update(HandleParams(topk_ids=topk))
    g.replay()
    torch.cuda.synchronize()


def test_update_binds_the_caller_buffer_not_a_copy(fake_nccl_ep, bypass_build_checks):
    """The handle must reference the caller's tensor, not a snapshot of it.

    Under CUDA graphs the routing buffer is written in place between replays,
    so update() has to hand the transport that same allocation. If it copied,
    every replay would re-run against stale routing and the graph would look
    like it worked while silently ignoring new tokens.
    """
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    topk = torch.zeros(16, 2, dtype=torch.int64, device="cuda")
    h.update(HandleParams(topk_ids=topk))

    native = fake_nccl_ep._log["handles"][-1]
    bound = [c for c in native.calls if c[0] == "update"][-1][1]
    assert bound.buffer.data_ptr() == topk.data_ptr()


def test_ops_use_the_knob_stream_outside_capture(fake_nccl_ep, bypass_build_checks):
    """Outside capture, transport work stays on the handle's own stream.

    Pins that the capture-aware _op_stream() did not change non-graph
    behaviour: an explicit HandleAlgoKnobUserStream must still be honoured.
    """
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobUserStream
    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    pinned = torch.cuda.Stream()
    fleet = _make_fleet(fake_nccl_ep)
    h = _make_handle(
        fleet,
        num_tokens=16,
        top_k=2,
        extra_knobs=(HandleAlgoKnobUserStream(stream=pinned.cuda_stream),),
    )
    topk = torch.zeros(16, 2, dtype=torch.int64, device="cuda")
    h.update(HandleParams(topk_ids=topk))

    native = fake_nccl_ep._log["handles"][-1]
    stream = [c for c in native.calls if c[0] == "update"][-1][2]["stream"]
    assert stream == pinned.cuda_stream


def test_ops_move_to_the_capture_stream_under_capture(
    fake_nccl_ep, bypass_build_checks
):
    """Under capture, work must be recorded on the stream being captured.

    A handle that outlives a capture is created before it starts, so its own
    stream is NOT the capture stream. Issuing there records nothing into the
    graph, and the replay silently does no transport work -- the failure this
    assertion exists to catch.
    """
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobUserStream
    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    pinned = torch.cuda.Stream()
    fleet = _make_fleet(fake_nccl_ep)
    h = _make_handle(
        fleet,
        num_tokens=16,
        top_k=2,
        extra_knobs=(HandleAlgoKnobUserStream(stream=pinned.cuda_stream),),
    )
    topk = torch.zeros(16, 2, dtype=torch.int64, device="cuda")

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        h.update(HandleParams(topk_ids=topk))
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        h.update(HandleParams(topk_ids=topk))
        captured_stream = torch.cuda.current_stream().cuda_stream

    native = fake_nccl_ep._log["handles"][-1]
    stream = [c for c in native.calls if c[0] == "update"][-1][2]["stream"]
    assert stream == captured_stream != pinned.cuda_stream


def test_a_first_update_that_is_already_captured_is_rejected(
    fake_nccl_ep, bypass_build_checks
):
    """Capture-first has no way to be ordered after InitHandle, so refuse it.

    InitHandle runs on the handle's own stream. Work recorded into a capture
    runs on the capture stream, and the dependency between them cannot be
    built from inside: the capture stream may not wait on an event recorded
    before capture began, and cudaEventSynchronize during capture invalidates
    the capture (cudaErrorStreamCaptureInvalidated -- verified on device).
    The ordering has to come from a device sync before cudaStreamBeginCapture,
    which torch.cuda.graph() does in __enter__.

    This guard cannot see that sync, so it checks the one thing it can: that
    the documented recipe -- one update outside the capture -- was followed.
    The alternative is a race that stays silent until replay.
    """
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    topk = torch.zeros(16, 2, dtype=torch.int64, device="cuda")

    g = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="cannot be the captured one"),
        torch.cuda.graph(g),
    ):
        h.update(HandleParams(topk_ids=topk))


def test_an_update_outside_the_capture_unlocks_the_captured_one(
    fake_nccl_ep, bypass_build_checks
):
    """The warmup is what satisfies the guard above -- nothing else.

    Pins the pair: the same capture that raises on a fresh handle succeeds
    once one update has run outside it. Without this, the guard could be
    tightened into rejecting the working recipe and only the negative test
    would still pass.
    """
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    topk = torch.zeros(16, 2, dtype=torch.int64, device="cuda")

    h.update(HandleParams(topk_ids=topk))  # the warmup

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        h.update(HandleParams(topk_ids=topk))
    g.replay()
    torch.cuda.synchronize()


def test_the_capture_guard_never_fires_outside_capture(
    fake_nccl_ep, bypass_build_checks
):
    """Non-graph callers must not be able to trip the guard.

    _op_stream() returns the handle's own stream whenever we are not
    capturing, so an eager caller is ordered after InitHandle by the stream
    itself and the guard has nothing to say -- including on a caller whose
    current stream is not the handle's, which is the shape that would look
    cross-stream if the guard tested the *current* stream rather than the one
    ops are actually issued on.
    """
    import torch

    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    h = _make_handle(_make_fleet(fake_nccl_ep), num_tokens=16, top_k=2)
    topk = torch.zeros(16, 2, dtype=torch.int64, device="cuda")

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        h.update(HandleParams(topk_ids=topk))
    torch.cuda.synchronize()


@pytest.mark.parametrize("hidden", [256, 512, 1024, 3072])
def test_ll_rejects_unsupported_hidden_size(fake_nccl_ep, bypass_build_checks, hidden):
    """Unsupported hidden must raise, not abort the process.

    nccl_ep instantiates LL kernels only for the SWITCH_HIDDEN set; anything
    else hits EP_HOST_ASSERT(false and "Unsupported hidden") in
    device/low_latency.cu, which kills the worker with no Python traceback.
    Measured on 2xB200: hidden 256/512/1024 abort, 2048/4096 round-trip
    cleanly. 3072 is included here because DeepEP-LL supports it and nccl_ep
    does not -- copying DeepEP's list would silently reintroduce the abort.
    """
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    with pytest.raises(MoEEpConfigError, match="does not support token_hidden_size"):
        _make_fleet(fake_nccl_ep, hidden=hidden)


@pytest.mark.parametrize("hidden", [2048, 2560, 4096, 8192])
def test_ll_accepts_supported_hidden_size(fake_nccl_ep, bypass_build_checks, hidden):
    """The SWITCH_HIDDEN set itself must keep working."""
    assert _make_fleet(fake_nccl_ep, hidden=hidden) is not None


def test_ll_dispatch_rejects_more_tokens_than_the_fleet_was_sized_for(
    fake_nccl_ep, bypass_build_checks
):
    """Over-dispatching must raise, not overrun the LL staging buffer.

    LL sizes staging to max_tokens_per_rank. Sending more silently corrupted
    memory and the kernel died with a bare SIGSEGV; HT already refused this.
    Measured on 2xB200 at hidden=2048: (M=256, T=222) and (M=222, T=222) round
    trip cleanly, (M=128, T=222) kills the worker.
    """
    import torch

    from flashinfer.moe_ep.config import DispatchInputParams
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    # Create the handle at 222 rows on a fleet sized for 128, so the
    # activation count matches the routing and the capacity guard -- not the
    # equality guard -- is what rejects it.
    fleet = _make_fleet(fake_nccl_ep, max_tokens=128)
    h = _make_handle(fleet, num_tokens=222, top_k=2)
    too_many = torch.zeros(222, 2048, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(MoEEpConfigError, match="exceeding max_tokens_per_rank"):
        h.dispatch(DispatchInputParams(x=[too_many]))
