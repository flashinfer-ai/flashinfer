from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch


_SGLANG_PYTHON_ROOT = Path("/home/luke/projects/sglang/python")


def _import_sglang_nsa_backend():
    if not _SGLANG_PYTHON_ROOT.exists():
        pytest.skip(f"sglang sources not found at {_SGLANG_PYTHON_ROOT}")
    root = str(_SGLANG_PYTHON_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module("sglang.srt.layers.attention.nsa_backend")
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        pytest.skip(f"unable to import sglang NSA backend: {exc}")
    return module


class _FakeDecodeMode:
    def is_decode_or_idle(self) -> bool:
        return True

    def is_target_verify(self) -> bool:
        return False

    def is_draft_extend(self, include_v2: bool = False) -> bool:
        del include_v2
        return False


class _FakeBackend:
    def __init__(self, *, metadata, req_to_token: torch.Tensor, device: torch.device):
        self.decode_cuda_graph_metadata = {1: metadata}
        self.req_to_token = req_to_token
        self.device = device
        self.sm_count = (
            torch.cuda.get_device_properties(device).multi_processor_count
            if device.type == "cuda"
            else 1
        )
        self.nsa_index_topk = req_to_token.shape[1]
        self.nsa_decode_impl = "b12x_mla"
        self.nsa_prefill_impl = "flashmla_sparse"
        self.speculative_num_draft_tokens = 0
        self.real_page_size = 1

    def set_nsa_prefill_impl(self, forward_batch=None):
        del forward_batch

    def _transform_table_1_to_real(self, page_indices: torch.Tensor) -> torch.Tensor:
        return page_indices


def test_sglang_nsa_replay_cuda_graph_handles_frozen_metadata_for_b12x_decode() -> None:
    module = _import_sglang_nsa_backend()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    req_to_token = torch.tensor([[3, 4, 5, 6]], dtype=torch.int32, device=device)
    page_table_1 = torch.full((1, 4), -1, dtype=torch.int32, device=device)
    metadata = module.NSAMetadata(
        page_size=1,
        cache_seqlens_int32=torch.zeros((1,), dtype=torch.int32, device=device),
        max_seq_len_q=1,
        max_seq_len_k=4,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.zeros((2,), dtype=torch.int32, device=device),
        page_table_1=page_table_1,
        real_page_table=page_table_1,
        nsa_cache_seqlens_int32=torch.zeros((1,), dtype=torch.int32, device=device),
        nsa_cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
        nsa_cu_seqlens_k=torch.zeros((2,), dtype=torch.int32, device=device),
        nsa_extend_seq_lens_list=[1],
        nsa_seqlens_expanded=torch.zeros((1,), dtype=torch.int32, device=device),
    )
    backend = _FakeBackend(metadata=metadata, req_to_token=req_to_token, device=device)

    module.NativeSparseAttnBackend.init_forward_metadata_replay_cuda_graph(
        backend,
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([4], dtype=torch.int32, device=device),
        seq_lens_sum=4,
        encoder_lens=None,
        forward_mode=_FakeDecodeMode(),
        spec_info=None,
        seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
        out_cache_loc=None,
    )

    assert torch.equal(metadata.cache_seqlens_int32, torch.tensor([4], dtype=torch.int32, device=device))
    assert torch.equal(metadata.page_table_1[0, :4], req_to_token[0])
    if device.type == "cuda":
        assert metadata.paged_mqa_schedule_metadata is not None
        assert metadata.paged_mqa_schedule_metadata.shape == (backend.sm_count + 1, 2)
        assert metadata.paged_mqa_schedule_metadata.dtype == torch.int32
    else:
        assert metadata.paged_mqa_schedule_metadata is None
