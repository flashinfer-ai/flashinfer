"""End-to-end autotune smoke coverage for the public MLA functional API."""

import pytest
import torch

import flashinfer
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import (
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)


# DeepSeek-V3 MLA layer dimensions. The CuTe DSL backend supports these
# canonical shapes, so they exercise the production auto-tuning path.
_NUM_HEADS = 128
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM
_PAGE_SIZE = 64
_MAX_SEQ_LEN = 2048
_WORKSPACE_SIZE = 128 * 1024 * 1024


def _skip_if_not_blackwell() -> None:
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        pytest.skip(f"Requires SM100 (Blackwell), got SM{cc[0]}{cc[1]}")


def _make_inputs(batch_size: int = 4, dtype: torch.dtype = torch.bfloat16):
    """Allocate inputs with headroom for autotune profiling candidates."""
    device = "cuda"
    pages_per_seq = (_MAX_SEQ_LEN + _PAGE_SIZE - 1) // _PAGE_SIZE
    num_pages = batch_size * pages_per_seq * 2

    query = torch.randn(batch_size, 1, _NUM_HEADS, _HEAD_DIM, device=device).to(dtype)
    kv_cache = torch.randn(num_pages, _PAGE_SIZE, _HEAD_DIM, device=device).to(dtype)
    block_tables = torch.zeros(
        batch_size, pages_per_seq, dtype=torch.int32, device=device
    )
    for batch_idx in range(batch_size):
        block_tables[batch_idx] = torch.arange(
            batch_idx * pages_per_seq,
            (batch_idx + 1) * pages_per_seq,
            dtype=torch.int32,
            device=device,
        )
    seq_lens = torch.full(
        (batch_size,), _MAX_SEQ_LEN // 2, dtype=torch.int32, device=device
    )
    workspace_buffer = torch.empty(_WORKSPACE_SIZE, dtype=torch.int8, device=device)
    return query, kv_cache, block_tables, seq_lens, workspace_buffer


def _call_decode(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    workspace_buffer,
    multi_ctas_kv_counter_buffer=None,
):
    return flashinfer.mla.batch_mla_paged_attention(
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
        backend="auto",
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
    )


def test_autotune_dispatcher_runs_with_auto_backend_and_caller_counter():
    """Autotune profiles use internal counters, not the final-call buffer."""
    _skip_if_not_blackwell()
    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_inputs()
    counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        query.size(0), query.size(2), get_device_sm_count(query.device)
    )
    caller_counter_buffer = torch.zeros(
        counter_bytes, dtype=torch.uint8, device=query.device
    )
    AutoTuner.get().clear_cache()

    with autotune(True):
        out = _call_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            workspace_buffer,
            multi_ctas_kv_counter_buffer=caller_counter_buffer,
        )

    assert out.shape == (query.shape[0], 1, _NUM_HEADS, _KV_LORA_RANK)
    assert out.isfinite().all()
    assert torch.count_nonzero(caller_counter_buffer).item() == 0
