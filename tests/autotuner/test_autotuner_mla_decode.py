"""Autotune smoke test for ``trtllm_batch_decode_with_kv_cache_mla``.

End-to-end check that ``backend="auto"`` runs cleanly under ``autotune(True)``.
"""

import pytest
import torch

import flashinfer
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.mla._batch_mla import _functional as batch_mla_core
from flashinfer.utils import (
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)


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
    multi_ctas_kv_counter_buffer=None,
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
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
    )


# ---------------------------------------------------------------------------
# End-to-end smoke + cache-population tests
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    ("backend", "expected_candidates"),
    (("trtllm-gen", ("trtllm-gen",)), ("auto", ("trtllm-gen", "cute-dsl"))),
)
def test_explicit_and_auto_candidate_names_and_order_remain_stable(
    monkeypatch, backend, expected_candidates
):
    seen = {}

    class FakeRunner:
        def __init__(self, name, **kwargs):
            self.name = name

        def __call__(self, *, inputs, tactic):
            seen["final_call"] = (self.name, inputs, tactic)

    class FakeAutoTuner:
        @classmethod
        def get(cls):
            return cls()

        def choose_one(self, op_name, runners, tuning_config, inputs):
            seen["choose"] = (
                op_name,
                tuple(runner.name for runner in runners),
                tuning_config,
            )
            return runners[0], -1

    monkeypatch.setattr(
        batch_mla_core,
        "TrtllmGenMlaDecodeRunner",
        lambda **kwargs: FakeRunner("trtllm-gen", **kwargs),
    )
    monkeypatch.setattr(
        batch_mla_core,
        "CuteDslMlaDecodeRunner",
        lambda **kwargs: FakeRunner("cute-dsl", **kwargs),
    )
    monkeypatch.setattr(batch_mla_core, "AutoTuner", FakeAutoTuner)
    monkeypatch.setattr(batch_mla_core, "device_support_pdl", lambda device: False)
    monkeypatch.setattr(batch_mla_core, "get_device_sm_count", lambda device: 120)
    monkeypatch.setattr(
        batch_mla_core,
        "_check_trtllm_gen_mla_shape",
        lambda query, kv_cache, *args, **kwargs: kv_cache,
    )
    monkeypatch.setattr(
        batch_mla_core,
        "_cute_dsl_incompatibility_reason",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        batch_mla_core,
        "_build_mla_decode_tuning_config",
        lambda **kwargs: seen.setdefault(
            "tuning_runner_names", tuple(kwargs["runner_names"])
        ),
    )

    query = torch.empty((2, 1, 128, 576), dtype=torch.bfloat16)
    out = torch.empty((2, 1, 128, 512), dtype=torch.bfloat16)
    result = batch_mla_core._run_mla_decode_trtllm_gen_or_cute_dsl_impl(
        query=query,
        kv_cache=torch.empty((4, 1, 64, 576), dtype=torch.bfloat16),
        workspace_buffer=torch.empty(16, dtype=torch.uint8),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=torch.zeros((2, 1), dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32),
        max_seq_len=64,
        out=out,
        backend=backend,
    )

    assert result is out
    assert seen["tuning_runner_names"] == expected_candidates
    assert seen["choose"][:2] == (
        "trtllm_batch_decode_mla",
        expected_candidates,
    )
    assert seen["final_call"][0] == expected_candidates[0]
