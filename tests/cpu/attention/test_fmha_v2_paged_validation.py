# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""trtllm_fmha_v2_prefill rejects paged KV inputs the kernel would misread."""

import pytest
import torch

from flashinfer.prefill import trtllm_fmha_v2_prefill

_SEQ_LEN, _HEADS, _HEAD_DIM = 16, 2, 64


def _call(layout: str, page_size: int, kv_dtype: torch.dtype, separate_kv: bool):
    query = torch.zeros(_SEQ_LEN, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)
    if layout == "Q_PAGED_KV_NHD":
        page = (page_size, _HEADS, _HEAD_DIM)
    else:
        page = (_HEADS, page_size, _HEAD_DIM)
    if separate_kv:
        # One allocation, V one block after K, with [B, 2, M] block tables.
        kv = torch.zeros(2, 1, *page, dtype=kv_dtype)
        paged_kv = (kv[0], kv[1])
        block_tables = torch.tensor([[[0], [1]]], dtype=torch.int32)
    else:
        paged_kv = torch.zeros(1, 2, *page, dtype=kv_dtype)
        block_tables = torch.zeros(1, 1, dtype=torch.int32)
    trtllm_fmha_v2_prefill(
        (query, paged_kv),
        layout,
        workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
        seq_lens=torch.tensor([_SEQ_LEN], dtype=torch.int32),
        max_q_len=_SEQ_LEN,
        max_kv_len=_SEQ_LEN,
        bmm1_scale=_HEAD_DIM**-0.5,
        bmm2_scale=1.0,
        batch_size=1,
        cum_seq_lens_q=torch.tensor([0, _SEQ_LEN], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, _SEQ_LEN], dtype=torch.int32),
        block_tables=block_tables,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("layout", "separate_kv", "page_size", "kv_dtype", "match"),
    [
        # 832 tokens/page: vLLM's page size for hybrid attention/Mamba models.
        ("Q_PAGED_KV_NHD", False, 832, torch.bfloat16, "power-of-two page_size"),
        ("Q_PAGED_KV_NHD", False, 0, torch.bfloat16, "power-of-two page_size"),
        ("Q_PAGED_KV_HND", True, 1024, torch.float8_e4m3fn, "KV cache dtype"),
    ],
)
def test_rejects_paged_kv_the_kernel_misreads(
    layout: str, separate_kv: bool, page_size: int, kv_dtype: torch.dtype, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _call(layout, page_size, kv_dtype, separate_kv)
