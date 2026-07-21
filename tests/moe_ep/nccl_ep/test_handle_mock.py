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


def _make_fleet(fake_nccl_ep, *, algorithm=None, world=4, max_tokens=128, hidden=64):
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
