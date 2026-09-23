"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""KDAPrefillPlanCache: rebinding a prepared BF16 KDA launch to new addresses.

A cache hit must reproduce a freshly prepared launch bit for bit (output, FP32
state pool rows and checkpoint rows), across changing tensor addresses,
interleaved shapes and CUDA-graph capture/replay of the hit path.
"""

import pytest
import torch

from flashinfer import (
    KDAPrefillPlanCache,
    kda_prefill_supports_fp32_checkpoints,
    prepare_bf16_kda_prefill,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="KDA export requires SM100a or SM103a",
)

HEAD_DIM = 128


def _inputs(lengths, heads, *, seed, storage_extra=0):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    tokens = sum(lengths)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    pool_slots = len(lengths) + 3
    offsets, cp_offsets = [0], [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
        cp_offsets.append(cp_offsets[-1] + (n + 63) // 64)
    return dict(
        q=randn(1, tokens, heads, HEAD_DIM) * 0.25,
        k=randn(1, tokens, heads, HEAD_DIM) * 0.25,
        v=randn(1, tokens, heads, HEAD_DIM) * 0.25,
        # g/beta storage may be longer than the logical token count and beta
        # may be a strided slice, as serving allocators produce.
        g=randn(1, tokens + storage_extra, heads, HEAD_DIM) * 0.25,
        beta=randn(1, tokens + storage_extra, heads + 24)[:, :, 8 : 8 + heads],
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=torch.full((heads, HEAD_DIM), -2.0, device="cuda"),
        out=torch.empty(1, tokens, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        pool=randn(pool_slots, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32) * 0.1,
        state_indices=torch.arange(len(lengths), device="cuda", dtype=torch.int32) + 1,
        cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int64),
        checkpoint_cu_starts=torch.tensor(cp_offsets, device="cuda", dtype=torch.int64),
        # Zero-filled so an unrequested checkpoint buffer stays comparable and
        # any stray write into it (stale pointer, out-of-bounds row) shows up.
        state_checkpoints=torch.zeros(
            (cp_offsets[-1], heads, HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.bfloat16
        ),
        lengths=tuple(lengths),
    )


def _prepare(d, cache=None, *, checkpoints=True, lower_bound=None):
    return prepare_bf16_kda_prefill(
        d["q"],
        d["k"],
        d["v"],
        d["g"],
        d["beta"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        out=d["out"],
        initial_state=d["pool"],
        final_state=d["pool"],
        lower_bound=lower_bound,
        cu_seqlens=d["cu_seqlens"],
        sequence_lengths=d["lengths"],
        state_indices=d["state_indices"],
        state_checkpoints=d["state_checkpoints"] if checkpoints else None,
        checkpoint_cu_starts=d["checkpoint_cu_starts"] if checkpoints else None,
        checkpoint_every_n_tokens=64 if checkpoints else 0,
        plan_cache=cache,
    )


def _run(d, cache=None, *, checkpoints=True, lower_bound=None):
    call = _prepare(d, cache, checkpoints=checkpoints, lower_bound=lower_bound)
    call.launch()
    if cache is None:
        call.close()
    return call


def _snapshot(d):
    return (
        d["out"].clone(),
        d["pool"][d["state_indices"].long()].clone(),
        d["state_checkpoints"].clone(),
    )


def _assert_same(got, want):
    # Bit patterns, not values: NaN payloads must compare equal too.
    for name, a, b in zip(("out", "state", "checkpoints"), got, want, strict=True):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), f"{name} differs after rebind"


@pytest.mark.parametrize(
    "lengths,heads,storage_extra",
    [
        ([64] * 4, 12, 0),
        ([64] * 16, 12, 16),
        ([64] * 64, 12, 0),
        ([17, 64, 65, 127, 128, 200, 255, 96], 12, 0),
        ([1024] * 4, 16, 0),
    ],
)
def test_cache_hit_matches_fresh_preparation_bitwise(lengths, heads, storage_extra):
    cache = KDAPrefillPlanCache(8)
    first = _inputs(lengths, heads, seed=1, storage_extra=storage_extra)
    second = _inputs(lengths, heads, seed=2, storage_extra=storage_extra)
    pool_before = second["pool"].clone()

    miss = _run(first, cache)
    hit = _run(second, cache)
    assert hit is miss, "same signature must reuse the prepared launch"
    assert (cache.misses, cache.hits) == (1, 1)
    got = _snapshot(second)

    second["pool"].copy_(pool_before)
    _run(second)
    _assert_same(got, _snapshot(second))

    # Rebinding back to the first address set is exact as well.
    pool_first = first["pool"].clone()
    _run(first, cache)
    got_first = _snapshot(first)
    first["pool"].copy_(pool_first)
    _run(first)
    _assert_same(got_first, _snapshot(first))


def test_cache_keys_on_sequence_lengths_and_interleaves_entries():
    cache = KDAPrefillPlanCache(8)
    short = _inputs([64] * 4, 12, seed=3)
    long = _inputs([64] * 16, 12, seed=4)
    pools = (short["pool"].clone(), long["pool"].clone())
    _run(short, cache)
    _run(long, cache)
    assert len(cache) == 2 and cache.misses == 2
    _run(short, cache)
    _run(long, cache)
    assert cache.hits == 2
    got = (_snapshot(short), _snapshot(long))
    for d, pool, want in zip((short, long), pools, got, strict=True):
        # The cached second run started from the pool the first run left
        # behind; replay the same two-step sequence without the cache.
        d["pool"].copy_(pool)
        _run(d)
        _run(d)
        _assert_same(want, _snapshot(d))


def test_split_sequence_affine_route_caches_and_rebinds():
    # A long bounded-gate sequence without a checkpoint request takes the
    # affine split route.  Its main/map/correction part launches and the
    # composite's own caller-aliasing attributes are covered by one rebind
    # plan, so a hit on a new address set reproduces a fresh preparation.
    cache = KDAPrefillPlanCache(8)
    first = _inputs([8192], 12, seed=5)
    second = _inputs([8192], 12, seed=6)
    pool_second = second["pool"].clone()
    miss = _run(first, cache, checkpoints=False, lower_bound=-5.0)
    assert "affine" in str(miss.schedule)
    assert cache.uncacheable == 0 and len(cache) == 1
    hit = _run(second, cache, checkpoints=False, lower_bound=-5.0)
    assert hit is miss and cache.hits == 1
    got = _snapshot(second)
    second["pool"].copy_(pool_second)
    _run(second, checkpoints=False, lower_bound=-5.0)
    _assert_same(got, _snapshot(second))


@pytest.mark.parametrize("checkpoints", [True, False])
def test_long_unbounded_sequence_on_fp32_state_takes_affine_split_and_caches(checkpoints):
    # The FP32 external state pool is the serving contract: with or without a
    # checkpoint request an unbounded long sequence takes the affine split
    # (FP32 state carrier in every part), and the composite plan is cacheable
    # and rebinds bitwise onto new addresses.
    cache = KDAPrefillPlanCache(8)
    d = _inputs([8192], 12, seed=5)
    pool = d["pool"].clone()
    call = _run(d, cache, checkpoints=checkpoints)
    schedule = str(call.schedule)
    assert "affine" in schedule
    assert cache.uncacheable == 0 and len(cache) == 1
    got = _snapshot(d)
    d["pool"].copy_(pool)
    _run(d, cache, checkpoints=checkpoints)
    assert cache.hits == 1
    _assert_same(got, _snapshot(d))


def test_cache_hit_is_cuda_graph_capturable():
    cache = KDAPrefillPlanCache(8)
    d = _inputs([64] * 16, 12, seed=6)
    pool = d["pool"].clone()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        _run(d, cache)
        d["pool"].copy_(pool)
        _run(d, cache)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    d["pool"].copy_(pool)
    with torch.cuda.graph(graph, stream=side):
        _prepare(d, cache).launch()
    for name in ("q", "k", "v", "g"):
        d[name].copy_(torch.randn_like(d[name]) * 0.25)
    d["beta"].copy_(torch.randn_like(d["beta"]))
    d["pool"].copy_(pool * 2.0)
    replay_pool = d["pool"].clone()
    graph.replay()
    torch.cuda.synchronize()
    got = _snapshot(d)
    d["pool"].copy_(replay_pool)
    _run(d)
    _assert_same(got, _snapshot(d))


def test_fp32_checkpoint_probe_is_per_gate_kind():
    # The FP32 intermediate-state contract is exported for the unbounded
    # softplus gate (Kimi-Linear / Kimi-K3); bounded-gate callers keep the
    # validated BF16 checkpoint rows and the probe must say so.
    device = torch.device("cuda", torch.cuda.current_device())
    assert kda_prefill_supports_fp32_checkpoints(device) is True
    assert kda_prefill_supports_fp32_checkpoints(device, lower_bound=None) is True
    assert kda_prefill_supports_fp32_checkpoints(device, lower_bound=-5.0) is False
    d = _inputs([200, 130], 16, seed=7)
    d["state_checkpoints"] = torch.empty_like(d["state_checkpoints"], dtype=torch.float32)
    call = _run(d)
    assert "fp32_checkpoint" in str(call.schedule)
    assert torch.isfinite(d["state_checkpoints"]).all()


def test_bf16_checkpoint_rows_keep_the_bf16_carrier_on_fp32_state():
    # BF16 checkpoint rows are typed by the same specialization as the chunk
    # carrier, so an unbounded FP32-state call with BF16 rows must not select
    # the FP32-carrier body (whose FP32 row stores would overflow the rows).
    d = _inputs([64] * 4, 12, seed=8)
    call = _run(d)
    assert "fp32" not in str(call.schedule)
    assert torch.isfinite(d["out"]).all() and torch.isfinite(d["state_checkpoints"]).all()


def _run_affine_epilogue(d, fused, monkeypatch, *, checkpoints, lower_bound):
    import flashinfer.cake_kda_tf32_runtime as runtime

    monkeypatch.setattr(
        runtime, "_fused_affine_epilogue_enabled", lambda: fused, raising=True
    )
    call = _run(d, checkpoints=checkpoints, lower_bound=lower_bound)
    assert "affine" in str(call.schedule)
    assert ("_fused_epilogue" in str(call.schedule)) == fused
    return _snapshot(d)


@pytest.mark.parametrize(
    "lengths,checkpoints,lower_bound",
    [
        ([8192], True, None),
        ([8192], False, None),
        ([8192, 8192], True, None),
        ([11000, 5384], True, None),
        ([8192], False, -5.0),
        ([2048, 8192], True, -5.0),
    ],
)
def test_affine_fused_epilogue_matches_torch_epilogue_bitwise(
    monkeypatch, lengths, checkpoints, lower_bound
):
    # The fused epilogue kernel (checkpoint-row merge + scatter, output tail
    # add, final-state select/add and pool scatter) replaces ~10 torch launches
    # and must reproduce them bit for bit on every row it writes.
    d = _inputs(lengths, 12, seed=11)
    pool = d["pool"].clone()
    want = _run_affine_epilogue(
        d, False, monkeypatch, checkpoints=checkpoints, lower_bound=lower_bound
    )
    d["pool"].copy_(pool)
    d["out"].zero_()
    d["state_checkpoints"].zero_()
    got = _run_affine_epilogue(
        d, True, monkeypatch, checkpoints=checkpoints, lower_bound=lower_bound
    )
    _assert_same(got, want)

