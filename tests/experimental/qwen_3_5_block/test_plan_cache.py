# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the ownership contract of the plan LRU behind the functional entry points -- an eager-only plan is freed when the LRU
(32 buckets) evicts it; a plan a CUDA graph was captured over is retained past its eviction and revived on the bucket's next
call; a cold bucket under capture is refused typed; ``clear_plan_cache()`` is the one release.  The real LRU, lookup and plan
dataclass run; compilation (``prepare``), the CUDA-pool validation (``kv_views``) and the capture query are replaced, so this
runs on any host without a GPU or the DSL."""

import gc
import weakref

import pytest
import torch

WORKSPACE_BYTES = 4096  # the stand-in workspace a fake plan owns


def _geometry():
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    return Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)


def _fake_prepare(
    form,
    *,
    geometry,
    device,
    weights,
    w_o,
    kv_cache,
    page_size,
    batch_size,
    dtype=torch.bfloat16,
    kv_layout="NHD",
    **kw,
):
    """A plan of the real dataclass that owns a (host) workspace, built without compiling anything."""
    from flashinfer.experimental.qwen_3_5_block.api import Qwen35BlockPlan

    pages = int(kv_cache.shape[0])
    return Qwen35BlockPlan(
        form=form,
        geometry=geometry,
        dtype=dtype,
        device=torch.device(device),
        kv_layout=kv_layout,
        page_size=page_size,
        num_pages=pages,
        batch_size=batch_size,
        q_len=kw.get("q_len_per_req", 1),
        max_pages=4,
        table_row_stride=4,
        max_seq_len=kw.get("max_seq_len"),
        decode_split_kv=kw.get("decode_split_kv"),
        workspace_size=WORKSPACE_BYTES,
        declaration={"form": form},
        _block=object(),
        _pool_shape=(pages, geometry.num_kv_heads, page_size, geometry.head_dim),
        _k_strides=(1, 1, 1, 1),
        _v_strides=(1, 1, 1, 1),
        _workspace=torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8),
    )


@pytest.fixture
def lru(monkeypatch):
    """The API module with compilation, the pool validation and the capture query replaced; the cache empty before and after."""
    from flashinfer.experimental.qwen_3_5_block import api

    api.clear_plan_cache()
    state = {"capturing": False, "prepared": 0}

    def prepare(form, **kw):
        state["prepared"] += 1
        return _fake_prepare(form, **kw)

    monkeypatch.setattr(api, "prepare", prepare)
    monkeypatch.setattr(
        api,
        "kv_views",
        lambda kv, layout, **kw: (
            kv[:, 0].permute(0, 2, 1, 3),
            kv[:, 1].permute(0, 2, 1, 3),
        ),
    )
    monkeypatch.setattr(api, "_capturing", lambda: state["capturing"])
    yield api, state
    state["capturing"] = False
    api.clear_plan_cache()


def _bucket(api, batch_size, **kw):
    """The functional entry points' lookup for one decode bucket (``batch_size`` tells the buckets apart)."""
    kv = torch.zeros(
        3, 2, 16, 2, 256, dtype=torch.bfloat16
    )  # a tiny NHD stand-in: shapes and strides are key material only
    return api.cached_plan(
        "decode",
        geometry=_geometry(),
        device=torch.device("cuda", 0),
        weights=None,
        w_o=None,
        kv_cache=kv,
        kv_layout="NHD",
        page_size=16,
        batch_size=batch_size,
        q_len_per_req=1,
        **kw,
    )


def test_an_eager_plan_is_freed_when_the_lru_evicts_it(lru):
    """What no graph holds the LRU still frees: its bound (32 buckets) stays the memory bound of an eager-only process."""
    api, state = lru
    first = _bucket(api, 1)
    refs = weakref.ref(first), weakref.ref(first._workspace)
    assert _bucket(api, 1) is first and state["prepared"] == 1, (
        "a second call of the bucket is a hit"
    )
    del first
    for b in range(2, api.PLAN_CACHE_SIZE + 2):  # 33 distinct buckets in all
        _bucket(api, b)
    assert len(api._PLANS) == api.PLAN_CACHE_SIZE and not api._RETAINED
    gc.collect()
    assert refs[0]() is None and refs[1]() is None, (
        "an eager-only plan and its workspace must be freed by the eviction"
    )


def test_a_captured_plan_outlives_its_eviction_and_is_revived(lru):
    """The ownership probe of the review, on the fix: bucket A warmed, captured (a hit under capture), evicted by 33 more buckets --
    the plan and its workspace stay alive (retained); the bucket's next call revives that very plan (no second compile); another
    churn retains it again; ``clear_plan_cache()`` releases it."""
    api, state = lru
    a = _bucket(api, 1)
    assert not a._captured
    state["capturing"] = True
    assert _bucket(api, 1) is a and a._captured, (
        "the capture goes through the cached plan and marks it"
    )
    state["capturing"] = False
    refs = weakref.ref(a), weakref.ref(a._workspace)
    key_a = next(iter(api._PLANS))
    del a
    for b in range(2, api.PLAN_CACHE_SIZE + 2):
        _bucket(api, b)
    assert len(api._PLANS) == api.PLAN_CACHE_SIZE and key_a not in api._PLANS
    gc.collect()
    assert refs[0]() is not None and refs[1]() is not None, (
        "the captured plan / workspace were freed by the eviction"
    )
    assert [k for k, _ in api._RETAINED] == [key_a] and api._RETAINED[0][1] is refs[0]()
    for b in range(
        100, 200
    ):  # a long churn: still exactly one retained plan, still alive, the LRU at its bound
        _bucket(api, b)
    gc.collect()
    assert (
        refs[0]() is not None
        and len(api._RETAINED) == 1
        and len(api._PLANS) == api.PLAN_CACHE_SIZE
    )
    prepared = state["prepared"]
    revived = _bucket(api, 1)
    assert revived is refs[0]() and state["prepared"] == prepared, (
        "the bucket's next call revives the retained plan, no compile"
    )
    assert (
        not api._RETAINED
        and api._PLANS.get(key_a) is revived
        and next(reversed(api._PLANS)) == key_a
    )
    del revived
    for b in range(200, 200 + api.PLAN_CACHE_SIZE):
        _bucket(api, b)
    gc.collect()
    assert refs[0]() is not None and [k for k, _ in api._RETAINED] == [key_a], (
        "a revived captured plan is retained again"
    )
    api.clear_plan_cache()
    gc.collect()
    assert refs[0]() is None and refs[1]() is None, (
        "clear_plan_cache() is the release of a retained plan"
    )
    assert not api._PLANS and not api._RETAINED


def test_a_cold_bucket_under_capture_is_refused_typed(lru):
    """Under capture a bucket without a cached plan cannot be built (a plan compiles and allocates): a typed error naming the
    remedy, nothing compiled or cached; ``clear_plan_cache()`` under capture is refused the same way."""
    api, state = lru
    state["capturing"] = True
    with pytest.raises(RuntimeError, match="capture"):
        _bucket(api, 7)
    assert state["prepared"] == 0 and not api._PLANS, (
        "nothing was compiled or cached under capture"
    )
    with pytest.raises(RuntimeError, match="capture"):
        api.clear_plan_cache()
    state["capturing"] = False
    _bucket(api, 7)
    assert state["prepared"] == 1 and len(api._PLANS) == 1
