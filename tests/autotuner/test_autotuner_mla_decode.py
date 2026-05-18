"""Autotune tests for ``trtllm_batch_decode_with_kv_cache_mla``.

Modeled on:
- ``tests/autotuner/test_autotuner_bmm_fp8.py`` for the cache-populate-then-hit
  flow.
- ``tests/moe/test_cute_dsl_fused_moe.py::test_with_autotune`` for the
  ``with autotune(True):`` smoke-test style.

These tests cover:
1. End-to-end smoke under ``backend="auto"`` with autotune enabled.
2. Cache population: a tuned call followed by a non-tuned call must not grow
   the autotuner's profiling cache.
3. ``_can_use_cute_dsl_for_mla_decode`` filter behavior for the three
   trtllm-gen-only conditions (``sinks``, ``sparse_mla_top_k``, tensor scale).
4. Preserved legacy error behavior for explicit ``backend="cute-dsl"`` with
   incompatible parameters.
"""

import pytest
import torch

import flashinfer
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.mla._core import _can_use_cute_dsl_for_mla_decode
from flashinfer.utils import get_compute_capability


# DeepSeek-V3 MLA layer dimensions. The cute-dsl backend's
# `_check_can_implement` constrains to a small set of canonical shapes; these
# are the most common.
_NUM_HEADS = 128
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM  # query/kv_cache trailing dim
_PAGE_SIZE = 64
_MAX_SEQ_LEN = 2048
_WORKSPACE_SIZE = 128 * 1024 * 1024


def _skip_if_not_blackwell() -> None:
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        pytest.skip(f"Requires SM100 (Blackwell), got SM{cc[0]}{cc[1]}")


def _make_inputs(batch_size: int = 4, dtype: torch.dtype = torch.bfloat16):
    """Allocate the tensors needed by trtllm_batch_decode_with_kv_cache_mla."""
    device = "cuda"
    pages_per_seq = (_MAX_SEQ_LEN + _PAGE_SIZE - 1) // _PAGE_SIZE
    # 2x headroom so synthetic block_tables generated during autotune
    # profiling can sweep up to the batch ceiling without aliasing.
    num_pages = batch_size * pages_per_seq * 2

    query = torch.randn(batch_size, 1, _NUM_HEADS, _HEAD_DIM, device=device).to(dtype)

    kv_cache = torch.randn(num_pages, _PAGE_SIZE, _HEAD_DIM, device=device).to(dtype)

    block_tables = torch.zeros(
        batch_size, pages_per_seq, dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        block_tables[i] = torch.arange(
            i * pages_per_seq,
            (i + 1) * pages_per_seq,
            dtype=torch.int32,
            device=device,
        )

    seq_lens = torch.full(
        (batch_size,), _MAX_SEQ_LEN // 2, dtype=torch.int32, device=device
    )

    # trtllm-gen requires the workspace's counter region (8 MB) to be zero;
    # cute-dsl wants int8 dtype.
    workspace_buffer = torch.zeros(_WORKSPACE_SIZE, dtype=torch.int8, device=device)

    return query, kv_cache, block_tables, seq_lens, workspace_buffer


def _call_decode(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    workspace_buffer,
    backend: str = "auto",
    sinks=None,
    sparse_mla_top_k: int = 0,
    bmm1_scale=None,
):
    if bmm1_scale is None:
        bmm1_scale = 1.0 / (_HEAD_DIM**0.5)
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=_MAX_SEQ_LEN,
        sparse_mla_top_k=sparse_mla_top_k,
        bmm1_scale=bmm1_scale,
        sinks=sinks,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# End-to-end smoke + cache-population tests
# ---------------------------------------------------------------------------


def test_autotune_dispatcher_runs_with_auto_backend():
    """Smoke test: ``backend='auto'`` runs cleanly under ``autotune(True)``."""
    _skip_if_not_blackwell()

    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_inputs()
    AutoTuner.get().clear_cache()

    with autotune(True):
        out = _call_decode(query, kv_cache, block_tables, seq_lens, workspace_buffer)

    assert out.shape == (query.shape[0], 1, _NUM_HEADS, _KV_LORA_RANK)
    assert out.isfinite().all()


def test_autotune_populates_cache_and_subsequent_calls_hit():
    """A tune-mode call must add cache entries; a follow-up non-tune call must not."""
    _skip_if_not_blackwell()

    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_inputs()
    autotuner = AutoTuner.get()
    autotuner.clear_cache()
    assert len(autotuner.profiling_cache) == 0

    with autotune(True):
        _call_decode(query, kv_cache, block_tables, seq_lens, workspace_buffer)
    populated_size = len(autotuner.profiling_cache)
    assert populated_size > 0, (
        "Tune-mode call should have added at least one entry to the profiling cache"
    )

    with autotune(False):
        out = _call_decode(query, kv_cache, block_tables, seq_lens, workspace_buffer)

    assert len(autotuner.profiling_cache) == populated_size, (
        "Non-tune call should not have grown the cache (cache hit expected)"
    )
    assert out.isfinite().all()


# ---------------------------------------------------------------------------
# `_can_use_cute_dsl_for_mla_decode` filter unit tests
# ---------------------------------------------------------------------------


def _filter_kwargs_for(query):
    """Default kwargs for the filter representing a cute-dsl-viable call."""
    return dict(
        query=query,
        out_dtype=torch.bfloat16,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        sinks=None,
        sparse_mla_top_k=0,
        skip_softmax_threshold_scale_factor=None,
        uses_shared_paged_kv_idx=True,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        page_size=_PAGE_SIZE,
        is_var_seq=True,
    )


def test_filter_accepts_viable_call():
    """Baseline: all defaults are cute-dsl-compatible -> filter returns True."""
    _skip_if_not_blackwell()
    query, *_ = _make_inputs()
    assert _can_use_cute_dsl_for_mla_decode(**_filter_kwargs_for(query))


def test_filter_excludes_when_sinks_set():
    _skip_if_not_blackwell()
    query, *_ = _make_inputs()
    kwargs = _filter_kwargs_for(query)
    kwargs["sinks"] = [torch.zeros(_NUM_HEADS, device="cuda", dtype=torch.float32)]
    assert not _can_use_cute_dsl_for_mla_decode(**kwargs)


def test_filter_excludes_when_sparse_mla_top_k_set():
    _skip_if_not_blackwell()
    query, *_ = _make_inputs()
    kwargs = _filter_kwargs_for(query)
    kwargs["sparse_mla_top_k"] = 64
    assert not _can_use_cute_dsl_for_mla_decode(**kwargs)


def test_filter_excludes_when_bmm1_scale_is_tensor():
    _skip_if_not_blackwell()
    query, *_ = _make_inputs()
    kwargs = _filter_kwargs_for(query)
    kwargs["bmm1_scale"] = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    assert not _can_use_cute_dsl_for_mla_decode(**kwargs)


def test_filter_excludes_when_skip_softmax_set():
    _skip_if_not_blackwell()
    query, *_ = _make_inputs()
    kwargs = _filter_kwargs_for(query)
    kwargs["skip_softmax_threshold_scale_factor"] = 1.5
    assert not _can_use_cute_dsl_for_mla_decode(**kwargs)


def test_filter_excludes_when_separate_kv_page_indices():
    _skip_if_not_blackwell()
    query, *_ = _make_inputs()
    kwargs = _filter_kwargs_for(query)
    kwargs["uses_shared_paged_kv_idx"] = False
    assert not _can_use_cute_dsl_for_mla_decode(**kwargs)


# ---------------------------------------------------------------------------
# Preserved legacy errors for explicit `backend="cute-dsl"`
# ---------------------------------------------------------------------------


def test_explicit_cute_dsl_raises_for_sinks():
    """``backend='cute-dsl'`` with sinks must still raise (auto silently filters)."""
    _skip_if_not_blackwell()
    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_inputs()
    sinks = [torch.zeros(_NUM_HEADS, device="cuda", dtype=torch.float32)]

    with pytest.raises(ValueError, match="does not support sinks"):
        _call_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            workspace_buffer,
            backend="cute-dsl",
            sinks=sinks,
        )


def test_explicit_cute_dsl_raises_for_sparse_mla_top_k():
    _skip_if_not_blackwell()
    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_inputs()

    with pytest.raises(ValueError, match="does not support sparse_mla_top_k"):
        _call_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            workspace_buffer,
            backend="cute-dsl",
            sparse_mla_top_k=64,
        )


def test_explicit_cute_dsl_raises_for_tensor_bmm1_scale():
    _skip_if_not_blackwell()
    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_inputs()
    bmm1_scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="does not support tensor bmm1_scale"):
        _call_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            workspace_buffer,
            backend="cute-dsl",
            bmm1_scale=bmm1_scale,
        )
