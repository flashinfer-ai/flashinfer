# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cuTile MLA paged decode backend of BatchMLAPagedAttentionWrapper."""

import math

import pytest
import torch

import flashinfer
from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.utils import get_compute_capability

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def _require_blackwell():
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 10:
        pytest.skip("cuTile MLA decode requires SM100+ (Blackwell)")


def _torch_mla_decode_ref(
    q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table, page_size, sm_scale
):
    """Naive per-request paged MLA decode reference (fp32 math).

    scores = (q_nope . ckv + q_pe . kpe) * sm_scale over valid kv positions;
    out = softmax(scores) @ ckv  (V shares the compressed latent, head_dim_vo=512).
    """
    batch_size, num_heads, head_dim_ckv = q_nope.shape
    out = torch.empty(
        batch_size, num_heads, head_dim_ckv, dtype=torch.float32, device=q_nope.device
    )
    for b in range(batch_size):
        seq_len = int(kv_lens[b].item())
        n_pages = math.ceil(seq_len / page_size)
        pages = page_table[b, :n_pages]
        # gather [seq_len, dim] from the paged cache
        ckv = ckv_cache[pages].reshape(-1, head_dim_ckv)[:seq_len].float()
        kpe = kpe_cache[pages].reshape(-1, kpe_cache.shape[-1])[:seq_len].float()
        qn = q_nope[b].float()  # [H, 512]
        qp = q_pe[b].float()  # [H, 64]
        # [H, seq_len]
        scores = (qn @ ckv.t() + qp @ kpe.t()) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = probs @ ckv  # [H, 512]
    return out


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [256, 1024])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_heads", [16, 32])
def test_mla_decode_cutile_vs_torch(batch_size, max_seq_len, page_size, num_heads):
    device = torch.device("cuda")
    torch.manual_seed(42)
    dtype = torch.bfloat16
    head_dim_ckv = 512
    head_dim_kpe = 64
    total_page_num = 512
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    q_nope = torch.randn(
        batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(batch_size, num_heads, head_dim_kpe, dtype=dtype, device=device)
    ckv_cache = torch.randn(
        total_page_num, page_size, head_dim_ckv, dtype=dtype, device=device
    )
    kpe_cache = torch.randn(
        total_page_num, page_size, head_dim_kpe, dtype=dtype, device=device
    )
    # random but valid seq lengths (at least 1 token)
    kv_lens = torch.randint(
        1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )
    kv_lens[0] = max_seq_len  # ensure the long case is covered
    pages_per_batch = math.ceil(max_seq_len / page_size)
    page_table = torch.randint(
        0,
        total_page_num,
        (batch_size, pages_per_batch),
        dtype=torch.int32,
        device=device,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutile")

    # plan() only needs to stash sm_scale/page_size/dtypes for cutile; the
    # indptr args are unused by the cutile path but required by the signature.
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(1, dtype=torch.int32, device=device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,  # causal (ignored for single-token decode)
        sm_scale,
        dtype,
        dtype,
    )

    out = wrapper.run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_len=kv_lens,
        page_table=page_table,
    )

    ref = _torch_mla_decode_ref(
        q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table, page_size, sm_scale
    )

    assert out.shape == (batch_size, num_heads, head_dim_ckv)
    assert not out.isnan().any()
    torch.testing.assert_close(out.float(), ref, rtol=2e-1, atol=1e-2)


def test_mla_decode_cutile_preallocated_out():
    """Passing a preallocated out tensor must match the auto-allocated path."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    dtype = torch.bfloat16
    batch_size, num_heads, page_size, max_seq_len = 2, 32, 64, 512
    head_dim_ckv, head_dim_kpe = 512, 64
    total_page_num = 256
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    q_nope = torch.randn(
        batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(batch_size, num_heads, head_dim_kpe, dtype=dtype, device=device)
    ckv_cache = torch.randn(
        total_page_num, page_size, head_dim_ckv, dtype=dtype, device=device
    )
    kpe_cache = torch.randn(
        total_page_num, page_size, head_dim_kpe, dtype=dtype, device=device
    )
    kv_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    pages_per_batch = math.ceil(max_seq_len / page_size)
    page_table = torch.randint(
        0,
        total_page_num,
        (batch_size, pages_per_batch),
        dtype=torch.int32,
        device=device,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutile")
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(1, dtype=torch.int32, device=device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,
        sm_scale,
        dtype,
        dtype,
    )

    o_auto = wrapper.run(
        q_nope, q_pe, ckv_cache, kpe_cache, kv_len=kv_lens, page_table=page_table
    )
    o_pre = torch.empty_like(o_auto)
    wrapper.run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        out=o_pre,
        kv_len=kv_lens,
        page_table=page_table,
    )
    torch.testing.assert_close(o_auto, o_pre)


def test_mla_decode_cutile_rejects_unsupported():
    """cutile MLA backend must reject kv_len/page_table omission and fp8 scales."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutile")
    wrapper._backend = "cutile"
    wrapper._sm_scale = 1.0 / math.sqrt(576)

    q_nope = torch.randn(1, 32, 512, dtype=dtype, device=device)
    q_pe = torch.randn(1, 32, 64, dtype=dtype, device=device)
    ckv = torch.randn(8, 64, 512, dtype=dtype, device=device)
    kpe = torch.randn(8, 64, 64, dtype=dtype, device=device)

    with pytest.raises(ValueError, match="requires kv_len and page_table"):
        wrapper.run(q_nope, q_pe, ckv, kpe)

    kv_lens = torch.full((1,), 128, dtype=torch.int32, device=device)
    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="not supported"):
        wrapper.run(
            q_nope, q_pe, ckv, kpe, kv_len=kv_lens, page_table=page_table, o_scale=0.1
        )


if __name__ == "__main__":
    test_mla_decode_cutile_vs_torch(4, 1024, 64, 32)
    test_mla_decode_cutile_preallocated_out()
    test_mla_decode_cutile_rejects_unsupported()
