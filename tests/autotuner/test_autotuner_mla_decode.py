"""Autotune smoke test for ``trtllm_batch_decode_with_kv_cache_mla``.

End-to-end check that ``backend="auto"`` runs cleanly under ``autotune(True)``.
"""

import pytest
import torch

import flashinfer
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
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

    # cute-dsl wants int8 dtype; trtllm-gen counters use a separate internal buffer.
    workspace_buffer = torch.empty(_WORKSPACE_SIZE, dtype=torch.int8, device=device)

    return query, kv_cache, block_tables, seq_lens, workspace_buffer


def _call_decode(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    workspace_buffer,
    backend: str = "auto",
):
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
        bmm1_scale=1.0 / (_HEAD_DIM**0.5),
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
