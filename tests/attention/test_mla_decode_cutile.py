# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cuTile MLA paged decode backend of BatchMLAPagedAttentionWrapper."""

import math
import warnings

import pytest
import torch

import flashinfer
from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.utils import get_compute_capability

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def _require_blackwell():
    """Skip unless this is a cuTile MLA-validated Blackwell target."""
    capability = get_compute_capability(torch.device("cuda"))
    if capability not in {(10, 0), (10, 3), (12, 0), (12, 1)}:
        pytest.skip("cuTile MLA decode requires SM100, SM103, SM120, or SM121")


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


def _run_mla_decode_case(
    *,
    batch_size,
    max_seq_len,
    page_size,
    num_heads,
    dtype=torch.bfloat16,
    packed=False,
    legacy_api=False,
):
    device = torch.device("cuda")
    torch.manual_seed(42 + page_size + num_heads)
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
    if batch_size > 1:
        kv_lens[1] = max_seq_len - 1  # guarantee a non-page-aligned tail
    pages_per_batch = math.ceil(max_seq_len / page_size)
    page_table = torch.randperm(total_page_num, dtype=torch.int32, device=device)[
        : batch_size * pages_per_batch
    ].reshape(batch_size, pages_per_batch)
    identity_pages = torch.arange(
        batch_size * pages_per_batch, dtype=torch.int32, device=device
    ).reshape_as(page_table)
    assert not torch.equal(page_table, identity_pages)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutile")

    if legacy_api:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            wrapper.plan(
                torch.arange(batch_size + 1, dtype=torch.int32, device=device),
                torch.arange(batch_size + 1, dtype=torch.int32, device=device)
                * pages_per_batch,
                page_table.reshape(-1),
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
            out = wrapper.run(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                kv_len=kv_lens,
                page_table=page_table,
            )
    else:
        metadata = flashinfer.mla.MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            block_tables=page_table,
            seq_lens=kv_lens,
        )
        layout = "packed" if packed else "split"
        wrapper.plan(
            metadata=metadata,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            page_size=page_size,
            causal=False,
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
            query_layout=layout,
            kv_cache_layout=layout,
        )
        query = torch.cat((q_nope, q_pe), dim=-1) if packed else (q_nope, q_pe)
        kv_cache = (
            torch.cat((ckv_cache, kpe_cache), dim=-1)
            if packed
            else (ckv_cache, kpe_cache)
        )
        out = wrapper.run(query=query, kv_cache=kv_cache)

    ref = _torch_mla_decode_ref(
        q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table, page_size, sm_scale
    )

    assert out.shape == (batch_size, num_heads, head_dim_ckv)
    assert not out.isnan().any()
    torch.testing.assert_close(out.float(), ref, rtol=2e-1, atol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [256, 1024])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_heads", [16, 32])
def test_mla_decode_cutile_vs_torch(batch_size, max_seq_len, page_size, num_heads):
    """cuTile paged MLA decode must match the torch reference across the shape sweep."""
    _run_mla_decode_case(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
        num_heads=num_heads,
    )


@pytest.mark.parametrize(
    ("dtype", "num_heads"),
    [
        (torch.float16, 32),
        (torch.bfloat16, 8),
        (torch.bfloat16, 48),
        (torch.bfloat16, 64),
        (torch.bfloat16, 96),
        (torch.bfloat16, 128),
    ],
)
def test_mla_decode_cutile_supported_dtype_and_head_contract(dtype, num_heads):
    """Persistent coverage for the restored dtype and head-count support."""
    _run_mla_decode_case(
        batch_size=1,
        max_seq_len=127,
        page_size=64,
        num_heads=num_heads,
        dtype=dtype,
    )


def test_mla_decode_cutile_packed_inputs():
    """Packed canonical inputs must lower to the unchanged split kernel."""
    _run_mla_decode_case(
        batch_size=2,
        max_seq_len=255,
        page_size=16,
        num_heads=32,
        packed=True,
    )


def test_mla_decode_cutile_legacy_flat_api():
    """The original flat-plan and positional-run compatibility path still works."""
    _run_mla_decode_case(
        batch_size=2,
        max_seq_len=127,
        page_size=64,
        num_heads=32,
        legacy_api=True,
    )


@pytest.mark.parametrize(
    "pages_per_batch",
    [2, 8],
    ids=["no_split", "split_kv"],
)
def test_mla_decode_cutile_zero_length_rows(pages_per_batch):
    """Empty KV rows must be zero for auto- and caller-allocated outputs."""
    device = torch.device("cuda")
    torch.manual_seed(73 + pages_per_batch)
    dtype = torch.bfloat16
    batch_size, num_heads, page_size = 2, 16, 64
    head_dim_ckv, head_dim_kpe = 512, 64
    max_seq_len = pages_per_batch * page_size
    total_page_num = batch_size * pages_per_batch
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
    kv_lens = torch.tensor([0, max_seq_len - 1], dtype=torch.int32, device=device)
    page_table = torch.arange(total_page_num, dtype=torch.int32, device=device).reshape(
        batch_size, pages_per_batch
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutile")
    wrapper.plan(
        metadata=flashinfer.mla.MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            block_tables=page_table,
            seq_lens=kv_lens,
        ),
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        query_layout="split",
        kv_cache_layout="split",
    )

    query = (q_nope, q_pe)
    kv_cache = (ckv_cache, kpe_cache)
    auto_out = wrapper.run(query=query, kv_cache=kv_cache)
    caller_out = torch.full_like(auto_out, 7.0)
    actual = wrapper.run(query=query, kv_cache=kv_cache, out=caller_out)

    assert actual is caller_out
    expected_empty = torch.zeros_like(auto_out[0])
    torch.testing.assert_close(auto_out[0], expected_empty, rtol=0, atol=0)
    torch.testing.assert_close(caller_out[0], expected_empty, rtol=0, atol=0)
    ref = _torch_mla_decode_ref(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_lens,
        page_table,
        page_size,
        sm_scale,
    )
    torch.testing.assert_close(auto_out.float(), ref, rtol=2e-1, atol=1e-2)
    torch.testing.assert_close(caller_out.float(), ref, rtol=2e-1, atol=1e-2)


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
    metadata = flashinfer.mla.MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        block_tables=page_table,
        seq_lens=kv_lens,
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        query_layout="split",
        kv_cache_layout="split",
    )

    o_auto = wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv_cache, kpe_cache))
    o_pre = torch.empty_like(o_auto)
    actual = wrapper.run(
        query=(q_nope, q_pe),
        kv_cache=(ckv_cache, kpe_cache),
        out=o_pre,
    )
    assert actual is o_pre
    torch.testing.assert_close(o_auto, o_pre)


def test_mla_decode_cutile_cuda_graph_replays_mutated_plan_metadata():
    """Captured runs must read updated values through stable metadata pointers."""
    device = torch.device("cuda")
    torch.manual_seed(11)
    dtype = torch.bfloat16
    batch_size, num_heads, page_size = 2, 16, 64
    head_dim_ckv, head_dim_kpe = 512, 64
    total_page_num = 8
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
    kv_lens = torch.tensor([96, 64], dtype=torch.int32, device=device)
    page_table = torch.tensor([[3, 1], [6, 2]], dtype=torch.int32, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    out = torch.empty(batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace,
        use_cuda_graph=True,
        backend="cutile",
    )
    wrapper.plan(
        metadata=flashinfer.mla.MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            block_tables=page_table,
            seq_lens=kv_lens,
        ),
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        query_layout="split",
        kv_cache_layout="split",
    )

    # Compile and warm the exact launch before capture.
    assert (
        wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv_cache, kpe_cache), out=out)
        is out
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = wrapper.run(
            query=(q_nope, q_pe), kv_cache=(ckv_cache, kpe_cache), out=out
        )
    assert captured_out is out
    graph.replay()
    torch.cuda.synchronize()
    initial = out.clone()

    updated_kv_lens = torch.tensor([0, 95], dtype=torch.int32, device=device)
    updated_page_table = torch.tensor(
        [[0, 5], [4, 7]], dtype=torch.int32, device=device
    )
    kv_lens.copy_(updated_kv_lens)
    page_table.copy_(updated_page_table)
    graph.replay()
    torch.cuda.synchronize()

    ref = _torch_mla_decode_ref(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        updated_kv_lens,
        updated_page_table,
        page_size,
        sm_scale,
    )
    assert not torch.equal(initial, out)
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]), rtol=0, atol=0)
    torch.testing.assert_close(out.float(), ref, rtol=2e-1, atol=1e-2)


if __name__ == "__main__":
    test_mla_decode_cutile_vs_torch(4, 1024, 64, 32)
    test_mla_decode_cutile_preallocated_out()
