"""Tests for the functional MLA decode dispatch cache.

``trtllm_batch_decode_with_kv_cache_mla`` keeps ``(runner_name, tactic)``
decisions in ``flashinfer.mla._core._MLA_DECODE_DISPATCH_CACHE`` so that
calls outside tuning skip rebuilding every candidate runner and the tuning
config. These tests check that cached calls are bitwise identical to the
uncached path, that the cache holds no tensors, and that it is bypassed or
invalidated whenever the autotuner state can change the decision.
"""

import gc
import weakref

import pytest
import torch

import flashinfer
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.mla import _core as mla_core
from flashinfer.utils import get_compute_capability

_NUM_HEADS = 128
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM
_PAGE_SIZE = 64
_PAGES_PER_SEQ = 64  # 4096-token page tables, as in eager SGLang decode
_WORKSPACE_SIZE = 128 * 1024 * 1024


@pytest.fixture(autouse=True)
def _blackwell_and_clean_cache():
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        pytest.skip(f"Requires SM100 (Blackwell), got SM{cc[0]}{cc[1]}")
    AutoTuner.get().clear_cache()
    mla_core.clear_mla_decode_dispatch_cache()
    yield
    mla_core.clear_mla_decode_dispatch_cache()


def _make_inputs(batch_size: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    num_pages = batch_size * _PAGES_PER_SEQ
    query = torch.randn(
        batch_size, 1, _NUM_HEADS, _HEAD_DIM, device="cuda", generator=gen
    ).to(torch.bfloat16)
    kv_cache = torch.randn(
        num_pages, _PAGE_SIZE, _HEAD_DIM, device="cuda", generator=gen
    ).to(torch.bfloat16)
    block_tables = (
        torch.randperm(num_pages, device="cuda", generator=gen)
        .to(torch.int32)
        .view(batch_size, _PAGES_PER_SEQ)
    )
    workspace_buffer = torch.zeros(_WORKSPACE_SIZE, dtype=torch.int8, device="cuda")
    return query, kv_cache, block_tables, workspace_buffer


def _seq_lens(batch_size: int, max_seq_len: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    seq_lens = torch.randint(
        max(1, max_seq_len // 2),
        max_seq_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device="cuda",
        generator=gen,
    )
    seq_lens[0] = max_seq_len
    return seq_lens


def _decode(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    workspace_buffer,
    max_seq_len,
    backend="auto",
    return_lse=False,
):
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM) ** 0.5,
        bmm2_scale=1.0,
        backend=backend,
        return_lse=return_lse,
    )


def _uncached(*args, **kwargs):
    # Any autotune() context makes AutoTuner.selection_cache_token() return
    # None, so this call takes the full runner-construction + choose_one path.
    with autotune(False):
        return _decode(*args, **kwargs)


@pytest.mark.parametrize("backend", ["auto", "cute-dsl"])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("max_seq_len", [1500, 4095])
def test_cached_decode_is_bitwise_identical(backend, batch_size, max_seq_len):
    query, kv_cache, block_tables, workspace = _make_inputs(batch_size)
    seq_lens = _seq_lens(batch_size, max_seq_len)
    args = (query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)

    ref = _uncached(*args, backend=backend)
    assert len(mla_core._MLA_DECODE_DISPATCH_CACHE) == 0

    miss = _decode(*args, backend=backend)
    assert len(mla_core._MLA_DECODE_DISPATCH_CACHE) == 1
    hit = _decode(*args, backend=backend)
    assert mla_core._MLA_DECODE_DISPATCH_CACHE.hits == 1
    torch.cuda.synchronize()
    assert torch.equal(miss, ref)
    assert torch.equal(hit, ref)


def test_cached_decode_with_lse_is_bitwise_identical():
    batch_size, max_seq_len = 32, 1500
    query, kv_cache, block_tables, workspace = _make_inputs(batch_size)
    seq_lens = _seq_lens(batch_size, max_seq_len)
    args = (query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)

    ref_out, ref_lse = _uncached(*args, return_lse=True)
    _decode(*args, return_lse=True)
    out, lse = _decode(*args, return_lse=True)
    assert mla_core._MLA_DECODE_DISPATCH_CACHE.hits == 1
    assert torch.equal(out, ref_out)
    assert torch.equal(lse, ref_lse)


def test_growing_max_seq_len_reuses_entries_and_stays_exact():
    batch_size = 32
    query, kv_cache, block_tables, workspace = _make_inputs(batch_size)
    for step, max_seq_len in enumerate(range(3840, 4096)):
        seq_lens = _seq_lens(batch_size, max_seq_len, seed=step)
        args = (query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)
        out = _decode(*args)
        if step % 37 == 0:
            assert torch.equal(out, _uncached(*args))
    cache = mla_core._MLA_DECODE_DISPATCH_CACHE
    # One entry per 128-token K tile while cute-dsl is a candidate (30, 31, 32
    # tiles here); a single entry when only trtllm-gen is eligible.
    assert 1 <= len(cache) <= 3
    assert cache.hits >= 256 - len(cache)


def test_cache_holds_no_tensors():
    batch_size, max_seq_len = 8, 2048
    query, kv_cache, block_tables, workspace = _make_inputs(batch_size)
    seq_lens = _seq_lens(batch_size, max_seq_len)
    _decode(query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)
    _decode(query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)
    torch.cuda.synchronize()
    assert mla_core._MLA_DECODE_DISPATCH_CACHE.hits == 1

    kv_ref, ws_ref = weakref.ref(kv_cache), weakref.ref(workspace)
    del kv_cache, workspace
    gc.collect()
    assert kv_ref() is None
    assert ws_ref() is None
    assert len(mla_core._MLA_DECODE_DISPATCH_CACHE) == 1


def test_cache_bypassed_while_tuning():
    batch_size, max_seq_len = 4, 2048
    query, kv_cache, block_tables, workspace = _make_inputs(batch_size)
    seq_lens = _seq_lens(batch_size, max_seq_len)
    with autotune(True):
        _decode(query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)
    assert len(mla_core._MLA_DECODE_DISPATCH_CACHE) == 0


def test_cache_invalidated_when_tuner_state_changes():
    batch_size, max_seq_len = 4, 2048
    query, kv_cache, block_tables, workspace = _make_inputs(batch_size)
    seq_lens = _seq_lens(batch_size, max_seq_len)
    args = (query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)
    cache = mla_core._MLA_DECODE_DISPATCH_CACHE

    _decode(*args)
    _decode(*args)
    assert (cache.hits, cache.misses) == (1, 1)

    AutoTuner.get().clear_cache()
    _decode(*args)
    assert (cache.hits, cache.misses) == (1, 2)

    # Tuning adds a profiling result; the next call must not reuse the
    # pre-tuning decision.
    with autotune(True):
        _decode(*args)
    tuned = _decode(*args)
    assert (cache.hits, cache.misses) == (1, 3)
    assert torch.equal(tuned, _uncached(*args))
    _decode(*args)
    assert cache.hits == 2


def test_cache_is_lru_bounded(monkeypatch):
    monkeypatch.setattr(mla_core._MLA_DECODE_DISPATCH_CACHE, "max_entries", 2)
    max_seq_len = 2048
    for batch_size in (1, 2, 4):
        query, kv_cache, block_tables, workspace = _make_inputs(batch_size)
        seq_lens = _seq_lens(batch_size, max_seq_len)
        _decode(query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)
    assert len(mla_core._MLA_DECODE_DISPATCH_CACHE) == 2


def test_explicit_trtllm_gen_does_not_use_cache():
    batch_size, max_seq_len = 4, 2048
    query, kv_cache, block_tables, workspace = _make_inputs(batch_size)
    seq_lens = _seq_lens(batch_size, max_seq_len)
    out = _decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace,
        max_seq_len,
        backend="trtllm-gen",
    )
    assert len(mla_core._MLA_DECODE_DISPATCH_CACHE) == 0
    auto = _decode(query, kv_cache, block_tables, seq_lens, workspace, max_seq_len)
    torch.cuda.synchronize()
    assert torch.equal(out, auto)
