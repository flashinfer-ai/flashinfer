# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the plan LRU behind the functional decode entry -- its KEY (buckets differing only in the list form, the prepare fusion, the
ring-slots arm, the table arm, the ring's capacity or the indexer cache dtype are DIFFERENT plans; the same bucket with another
side-cache STRIDE is the SAME plan: the kernels bind those strides symbolically) and the Qwen 3.5 ownership contract (an eager-only plan
freed at eviction, a captured one retained and revived, a cold bucket under capture refused typed).  The real LRU, lookup and plan
dataclass run; compilation (``prepare``) and the capture query are replaced, so this runs on any host without a GPU or the DSL."""

import gc
import weakref

import pytest
import torch

WORKSPACE_BYTES = 4096


def _geometry(**over):
    from flashinfer.qwen_3_8_block import Qwen38BlockGeometry

    kw = dict(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    kw.update(over)
    return Qwen38BlockGeometry(**kw)


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
    from flashinfer.experimental.qwen_3_8_block.api import Qwen38BlockPlan

    pages = int(kv_cache.shape[0])
    ring = kw["index_state_cache"]
    return Qwen38BlockPlan(
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
        ring_cap=int(ring.shape[1]),
        ring_slots=bool(kw.get("ring_slots", False)),
        rope_position_tables_only=bool(kw.get("rope_position_tables_only", True)),
        fuse_prepare=bool(kw.get("fuse_prepare", True)),
        _position_tables=kw.get("position_tables"),
    )


@pytest.fixture
def lru(monkeypatch):
    from flashinfer.experimental.qwen_3_8_block import api

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


def _side(cap=8, pages=3, strided=False):
    icache = torch.zeros(pages, 4, 1, 128, dtype=torch.bfloat16)
    ring = torch.zeros(4, cap, 1, 128, dtype=torch.bfloat16)
    if strided:  # another layout of the same FORM (a stack's planner interleaves the side caches): a transposed view's strides
        icache = torch.zeros(pages, 1, 4, 128, dtype=torch.bfloat16).transpose(1, 2)
        ring = torch.zeros(4, 1, cap, 128, dtype=torch.bfloat16).transpose(1, 2)
    return icache, ring


def _bucket(api, batch_size, *, geometry=None, cap=8, strided=False, **kw):
    kv = torch.zeros(3, 2, 16, 2, 256, dtype=torch.bfloat16)
    icache, ring = _side(cap=cap, strided=strided)
    return api.cached_plan(
        "decode",
        geometry=geometry or _geometry(),
        device=torch.device("cuda", 0),
        weights=None,
        w_o=None,
        kv_cache=kv,
        kv_layout="NHD",
        page_size=16,
        batch_size=batch_size,
        q_len_per_req=kw.pop("q_len_per_req", 1),
        index_kv_cache=icache,
        index_state_cache=ring,
        index_state_block_table=torch.zeros(batch_size, 1, dtype=torch.int32),
        position_tables=(
            torch.zeros(1, 48, 64, dtype=torch.bfloat16),
            torch.zeros(1, 48, 64, dtype=torch.bfloat16),
        ),
        **kw,
    )


def test_the_key_separates_the_forms_knobs_and_cache_forms_but_not_the_side_cache_strides(
    lru,
):
    api, state = lru
    base = _bucket(api, 1)
    assert _bucket(api, 1) is base and state["prepared"] == 1, (
        "the same bucket is a hit"
    )
    assert _bucket(api, 1, strided=True) is base and state["prepared"] == 1, (
        "another side-cache stride is the SAME plan (symbolic strides)"
    )
    distinct = [
        _bucket(
            api, 1, geometry=_geometry(index_list_form="per_row")
        ),  # the list form (explicit vs None) is in the geometry
        _bucket(
            api, 1, geometry=_geometry(index_cache_dtype=torch.bfloat16)
        ),  # the indexer cache dtype
        _bucket(api, 1, fuse_prepare=False),
        _bucket(api, 1, fuse_prepare=True),
        _bucket(api, 1, ring_slots=True),
        _bucket(api, 1, rope_position_tables_only=False),
        _bucket(api, 1, cap=12),  # the ring's capacity (its FORM)
        _bucket(api, 1, q_len_per_req=4, cap=8),
        _bucket(api, 1, use_pdl=False),
    ]
    assert all(p is not base for p in distinct) and len(
        {id(p) for p in distinct}
    ) == len(distinct)
    assert state["prepared"] == 1 + len(distinct)
    assert len(api._PLANS) == 1 + len(distinct)
    assert _bucket(api, 1, kv_cache_dtype=torch.bfloat16) is base, (
        "kv_cache_dtype=bf16 names the activation dtype: the same pools, the same plan"
    )


def test_an_eager_plan_is_freed_when_the_lru_evicts_it(lru):
    api, state = lru
    first = _bucket(api, 1)
    refs = weakref.ref(first), weakref.ref(first._workspace)
    del first
    for b in range(2, api.PLAN_CACHE_SIZE + 2):
        _bucket(api, b)
    assert len(api._PLANS) == api.PLAN_CACHE_SIZE and not api._RETAINED
    gc.collect()
    assert refs[0]() is None and refs[1]() is None


def test_a_captured_plan_outlives_its_eviction_and_is_revived(lru):
    api, state = lru
    a = _bucket(api, 1)
    state["capturing"] = True
    assert _bucket(api, 1) is a and a._captured
    state["capturing"] = False
    refs = weakref.ref(a), weakref.ref(a._workspace)
    key_a = next(iter(api._PLANS))
    del a
    for b in range(2, api.PLAN_CACHE_SIZE + 2):
        _bucket(api, b)
    assert key_a not in api._PLANS
    gc.collect()
    assert refs[0]() is not None and refs[1]() is not None, (
        "the captured plan / workspace were freed by the eviction"
    )
    assert [k for k, _ in api._RETAINED] == [key_a]
    prepared = state["prepared"]
    revived = _bucket(api, 1)
    assert revived is refs[0]() and state["prepared"] == prepared and not api._RETAINED
    api.clear_plan_cache()
    del revived
    gc.collect()
    assert refs[0]() is None and refs[1]() is None


def test_a_cold_bucket_under_capture_is_refused_typed(lru):
    api, state = lru
    state["capturing"] = True
    with pytest.raises(RuntimeError, match="capture"):
        _bucket(api, 7)
    assert state["prepared"] == 0 and not api._PLANS
    with pytest.raises(RuntimeError, match="capture"):
        api.clear_plan_cache()
    state["capturing"] = False
    _bucket(api, 7)
    assert state["prepared"] == 1 and len(api._PLANS) == 1
