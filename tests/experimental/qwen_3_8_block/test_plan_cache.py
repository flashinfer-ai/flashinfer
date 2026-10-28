# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the plan LRU behind the functional decode entry -- its KEY (buckets differing only in the list form, the prepare fusion, the sparse core's body knob, the
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
        _bucket(
            api, 1, sparse_tile_form="tcgen05"
        ),  # the sparse core's body knob: two bodies are two compiled plans
        _bucket(api, 1, sparse_tile_form="lean"),
    ]
    assert all(p is not base for p in distinct) and len(
        {id(p) for p in distinct}
    ) == len(distinct)
    assert state["prepared"] == 1 + len(distinct)
    assert len(api._PLANS) == 1 + len(distinct)
    assert _bucket(api, 1, kv_cache_dtype=torch.bfloat16) is base, (
        "kv_cache_dtype=bf16 names the activation dtype: the same pools, the same plan"
    )
    assert _bucket(api, 1, sparse_tile_form="auto") is base, (
        "sparse_tile_form='auto' is the default spelled out: the same plan"
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


class _RecordingBlock:
    """A stand-in for the vendored block: records the position tables every execute binds."""

    def __init__(self):
        self.bound = []

    def execute(self, *args, **kw):
        self.bound.append((kw["cos_pos"], kw["sin_pos"]))


@pytest.fixture
def functional(lru, monkeypatch):
    """The functional decode entry on CPU through the real LRU and the real ``Qwen38BlockPlan.run``: the vendored block is the
    recorder above and the per-call host checks that need device tensors (the weights, the operand binding, the workspace) are
    stubbed, so what reaches ``execute`` is exactly what the entry bound."""
    api, state = lru
    block = _RecordingBlock()
    fake_prepare = api.prepare

    def prepare(form, **kw):
        plan = fake_prepare(form, **kw)
        plan._block = block
        return plan

    monkeypatch.setattr(api, "prepare", prepare)
    monkeypatch.setattr(
        api, "_device_of", lambda d: torch.device(d)
    )  # the host's device passes the CUDA-only gate
    monkeypatch.setattr(api, "_check_weights", lambda *a, **k: None)
    monkeypatch.setattr(api, "_page_size_of", lambda kv, layout, geometry: 16)
    monkeypatch.setattr(
        api.Qwen38BlockPlan,
        "_bind",
        lambda self, **call: dict(
            call,
            h=call["hidden_states"],
            cos=call["rope_cos"],
            sin=call["rope_sin"],
            k_cache=None,
            v_cache=None,
            kv_lens=call["seq_lens"],
            rank=2,
            rows=self.num_tokens,
        ),
    )
    monkeypatch.setattr(
        api.Qwen38BlockPlan, "_workspace_for", lambda self, ws: self._workspace
    )
    return api, state, block


def _tables(fill, n_pos=64):
    return (
        torch.full((1, n_pos, 64), fill, dtype=torch.bfloat16),
        torch.full((1, n_pos, 64), -fill, dtype=torch.bfloat16),
    )


def test_a_functional_cache_hit_binds_this_calls_position_tables(functional):
    """The four-call probe on ONE bucket: the first call compiles the plan with its tables; a hit with OTHER valid tables binds
    THOSE (never the pair the first call compiled in); a call without tables is a typed error (never silently the plan's pair);
    a too-short pair is the form's typed error -- and neither refused call launches."""
    import types

    api, state, block = functional
    icache, ring = _side()
    weights = types.SimpleNamespace(
        w_qkvg=None, w_q_norm=None, w_k_norm=None, w_iq_norm=None, w_ik_norm=None
    )
    hidden = torch.zeros(2, 2560, dtype=torch.bfloat16)
    kw = dict(
        geometry=_geometry(),
        kv_cache=torch.zeros(3, 2, 16, 2, 256, dtype=torch.bfloat16),
        slot_mapping=torch.zeros(2, dtype=torch.int32),
        block_table=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32),
        index_kv_cache=icache,
        index_state_cache=ring,
        index_state_block_table=torch.zeros(2, 1, dtype=torch.int32),
        out=torch.zeros(2, 2560, dtype=torch.bfloat16),
    )
    first, second = _tables(0.0), _tables(1.0)
    api.decode(hidden, None, None, weights, None, position_tables=first, **kw)
    api.decode(hidden, None, None, weights, None, position_tables=second, **kw)
    assert state["prepared"] == 1, "one bucket: the second call is a cache hit"
    assert len(block.bound) == 2
    assert block.bound[0][0] is first[0] and block.bound[0][1] is first[1]
    assert block.bound[1][0] is second[0] and block.bound[1][1] is second[1], (
        "a cache hit bound the pair the FIRST call compiled in, not this call's"
    )
    with pytest.raises(ValueError, match="position_tables"):
        api.decode(hidden, None, None, weights, None, position_tables=None, **kw)
    with pytest.raises(ValueError, match="holds 32 positions"):
        api.decode(
            hidden, None, None, weights, None, position_tables=_tables(2.0, 32), **kw
        )
    assert len(block.bound) == 2, "a refused call must not launch"
    assert state["prepared"] == 1
