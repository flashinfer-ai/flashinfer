# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DeepSeek V4 CuTe DSL HCA decode integration."""

import math

import pytest
import torch

import flashinfer.mla._core as mla_core
from flashinfer.mla import (
    trtllm_batch_decode_sparse_mla_dsv4,
)
from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported
from flashinfer.utils import get_compute_capability
from tests.test_helpers.hca_dsv4 import _cpu_compressed_page_aligned_conversion_inputs


@pytest.mark.arch_blackwell
def test_dsv4_hca_compressed_page_aligned_conversion_rejects_mixed_devices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    inputs = _cpu_compressed_page_aligned_conversion_inputs(32, 128)
    with pytest.raises(ValueError, match="swa_kv_cache must be on cpu"):
        mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
            inputs["sparse_indices"],
            inputs["sparse_topk_lens"],
            inputs["seq_lens"],
            inputs["swa_kv_cache"].cuda(),
            inputs["compressed_kv_cache"],
            q_len=inputs["q_len"],
        )


def _reference_hca(
    query,
    window_cache,
    compressed_cache,
    window_indices,
    compressed_tables,
    sparse_topk_lens,
    window_valid_lens,
    sinks,
    softmax_scale,
    output_scale,
):
    batch_size, q_len, _, _ = query.shape
    flat_window_cache = window_cache.reshape(-1, query.shape[-1])
    output = torch.empty_like(query, dtype=torch.float32)
    for batch_idx in range(batch_size):
        for query_idx in range(q_len):
            row = batch_idx * q_len + query_idx
            window = flat_window_cache.index_select(0, window_indices[row])
            window = window[: int(window_valid_lens[row].item())]
            compressed = compressed_cache.index_select(
                0, compressed_tables[row]
            ).reshape(-1, query.shape[-1])
            compressed_len = int(sparse_topk_lens[row].item()) - 128
            compressed = compressed[:compressed_len]
            kv = torch.cat((window, compressed), dim=0).float()
            scores = torch.einsum("hd,kd->hk", query[batch_idx, query_idx].float(), kv)
            scores *= softmax_scale
            log_norm = torch.logaddexp(torch.logsumexp(scores, dim=-1), sinks.float())
            probabilities = torch.exp(scores - log_norm[:, None])
            output[batch_idx, query_idx] = (
                torch.einsum("hk,kd->hd", probabilities, kv) * output_scale
            )
    return output


@pytest.mark.parametrize(
    "compare_trtllm", (False, True), ids=("pytorch-reference", "trtllm-parity")
)
def test_cute_dsl_hca_fp8_to_bf16_correctness(monkeypatch, compare_trtllm):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(torch.device("cuda")) not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("CuTe DSL HCA requires SM100/SM103/SM107")
    if not is_cute_dsl_arch_supported(*get_compute_capability(torch.device("cuda"))):
        pytest.skip("the installed CuTe DSL cannot target this device")
    pytest.importorskip("cutlass")

    torch.manual_seed(17)
    device = torch.device("cuda")
    batch_size, q_len, num_heads, head_dim = 2, 2, 128, 512
    query = (
        torch.randn((batch_size, q_len, num_heads, head_dim), device=device).clamp_(
            -2, 2
        )
    ).to(torch.float8_e4m3fn)
    window_cache = torch.randn((16, 32, head_dim), device=device).clamp_(-2, 2)
    window_cache = window_cache.to(torch.float8_e4m3fn)
    compressed_cache = torch.randn((5, 128, head_dim), device=device).clamp_(-2, 2)
    compressed_cache = compressed_cache.to(torch.float8_e4m3fn)
    window_tables = torch.tensor(
        [
            [7, 0, 13, 2],
            [3, 15, 1, 9],
            [12, 5, 8, 4],
            [10, 14, 6, 11],
        ],
        dtype=torch.int32,
        device=device,
    )
    window_offsets = torch.arange(32, dtype=torch.int32, device=device)
    window_indices = (window_tables.unsqueeze(-1) * 32 + window_offsets).flatten(1)
    window_shifts = torch.tensor(
        [7, 31, 63, 95], dtype=torch.int64, device=device
    ).unsqueeze(1)
    window_indices = window_indices.gather(
        1,
        torch.remainder(
            torch.arange(128, dtype=torch.int64, device=device).unsqueeze(0)
            + window_shifts,
            128,
        ),
    ).contiguous()
    compressed_tables = torch.tensor(
        [[3, 4], [3, 4], [1, 0], [1, 0]], dtype=torch.int32, device=device
    )
    # Keep three backing tiles for every request while the effective lengths
    # remain partial. This catches any causal mask optimization that assumes
    # the scheduler drops backing-cache tiles beyond sparse_topk_lens.
    hca_seq_lens = torch.tensor([384, 384], dtype=torch.int32, device=device)
    sparse_topk_lens = torch.tensor(
        [128, 129, 256, 257], dtype=torch.int32, device=device
    )
    window_valid_lens = torch.tensor(
        [127, 128, 128, 128], dtype=torch.int32, device=device
    )
    workspace = torch.empty(8 << 20, dtype=torch.uint8, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    output_scale = 1.0
    compressed_offsets = torch.arange(128, dtype=torch.int32, device=device)
    compressed_page_aligned_sparse_indices = torch.cat(
        (
            window_indices,
            (compressed_tables.unsqueeze(-1) * 128 + compressed_offsets).flatten(1),
        ),
        dim=1,
    )
    window_positions = torch.arange(128, device=device)
    compressed_positions = torch.arange(256, device=device)
    compressed_page_aligned_sparse_indices[:, :128].masked_fill_(
        window_positions.unsqueeze(0) >= window_valid_lens.unsqueeze(1), -1
    )
    compressed_page_aligned_sparse_indices[:, 128:].masked_fill_(
        compressed_positions.unsqueeze(0) >= (sparse_topk_lens - 128).unsqueeze(1),
        0,
    )
    raw_seq_lens = torch.tensor([128, 16512], dtype=torch.int32, device=device)
    precomputed_metadata = (
        mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
            compressed_page_aligned_sparse_indices,
            sparse_topk_lens,
            raw_seq_lens,
            window_cache,
            compressed_cache,
            q_len=q_len,
        )
    )

    for sinks in (
        torch.zeros(num_heads, dtype=torch.float32, device=device),
        torch.linspace(4.0, 6.0, num_heads, dtype=torch.float32, device=device),
    ):
        output = trtllm_batch_decode_sparse_mla_dsv4(
            query=query,
            swa_kv_cache=window_cache,
            workspace_buffer=workspace,
            sparse_indices=None,
            compressed_kv_cache=compressed_cache,
            sparse_topk_lens=sparse_topk_lens,
            out=None,
            bmm1_scale=softmax_scale,
            bmm2_scale=output_scale,
            sinks=sinks,
            swa_topk_lens=window_valid_lens,
            backend="cute-dsl",
            hca_swa_indices=window_indices,
            hca_compressed_block_tables=compressed_tables,
            hca_seq_lens=hca_seq_lens,
            hca_is_causal=True,
            hca_use_persistent=True,
        )
        tagged_output = trtllm_batch_decode_sparse_mla_dsv4(
            query=query,
            swa_kv_cache=window_cache,
            workspace_buffer=workspace,
            sparse_indices=compressed_page_aligned_sparse_indices,
            compressed_kv_cache=compressed_cache,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=raw_seq_lens,
            out=None,
            bmm1_scale=softmax_scale,
            bmm2_scale=output_scale,
            sinks=sinks,
            backend="cute-dsl",
            hca_sparse_indices_format="compressed-page-aligned",
            hca_is_causal=True,
            hca_use_persistent=True,
        )
        reference = _reference_hca(
            query,
            window_cache,
            compressed_cache,
            window_indices,
            compressed_tables,
            sparse_topk_lens,
            window_valid_lens,
            sinks,
            softmax_scale,
            output_scale,
        )
        trtllm_output = None
        if compare_trtllm:
            try:
                trtllm_output = trtllm_batch_decode_sparse_mla_dsv4(
                    query=query,
                    swa_kv_cache=window_cache,
                    workspace_buffer=workspace,
                    sparse_indices=compressed_page_aligned_sparse_indices,
                    compressed_kv_cache=compressed_cache,
                    sparse_topk_lens=sparse_topk_lens,
                    seq_lens=raw_seq_lens,
                    bmm1_scale=softmax_scale,
                    bmm2_scale=output_scale,
                    sinks=sinks,
                    backend="trtllm-gen",
                    enable_pdl=False,
                )
            except RuntimeError as error:
                message = str(error)
                if (
                    "trtllm_paged_attention_decode_sparse_mla_dsv4 is not available"
                    in message
                    or "Missing TRTLLM-GEN kernel (decode)" in message
                    or (
                        "Ninja build failed" in message
                        and "missing and no known rule" in message
                    )
                ):
                    pytest.skip("TRTLLM-GEN sparse MLA artifacts are unavailable")
                raise
        assert output.dtype == torch.bfloat16
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, atol=0.13, rtol=1e-5)
        if trtllm_output is not None:
            assert trtllm_output.dtype == torch.bfloat16
            assert trtllm_output.shape == query.shape
            assert torch.isfinite(trtllm_output).all()
            torch.testing.assert_close(
                trtllm_output.float(), reference, atol=0.13, rtol=1e-5
            )
            torch.testing.assert_close(output, trtllm_output, atol=0.13, rtol=1e-5)
        torch.testing.assert_close(tagged_output, output, atol=0.00390625, rtol=0)

    explicit_output = torch.empty_like(query, dtype=torch.bfloat16)
    precomputed_output = torch.empty_like(explicit_output)
    trtllm_batch_decode_sparse_mla_dsv4(
        query=query,
        swa_kv_cache=window_cache,
        workspace_buffer=workspace,
        sparse_indices=None,
        compressed_kv_cache=compressed_cache,
        sparse_topk_lens=sparse_topk_lens,
        out=explicit_output,
        bmm1_scale=softmax_scale,
        swa_topk_lens=window_valid_lens,
        backend="cute-dsl",
        hca_swa_indices=window_indices,
        hca_compressed_block_tables=compressed_tables,
        hca_seq_lens=hca_seq_lens,
        hca_use_persistent=True,
    )

    def run_precomputed_metadata():
        return trtllm_batch_decode_sparse_mla_dsv4(
            query=query,
            swa_kv_cache=window_cache,
            workspace_buffer=workspace,
            sparse_indices=None,
            compressed_kv_cache=compressed_cache,
            sparse_topk_lens=sparse_topk_lens,
            out=precomputed_output,
            bmm1_scale=softmax_scale,
            swa_topk_lens=precomputed_metadata.swa_topk_lens,
            backend="cute-dsl",
            hca_swa_indices=precomputed_metadata.hca_swa_indices,
            hca_compressed_block_tables=(
                precomputed_metadata.hca_compressed_block_tables
            ),
            hca_seq_lens=precomputed_metadata.hca_seq_lens,
            hca_is_causal=precomputed_metadata.hca_is_causal,
            hca_use_persistent=True,
        )

    run_precomputed_metadata()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_precomputed_metadata()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        precomputed_output, explicit_output, atol=0.00390625, rtol=0
    )

    with monkeypatch.context() as capture_monkeypatch:
        capture_monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda: True
        )
        with pytest.raises(RuntimeError, match="not CUDA Graph capture safe"):
            mla_core.convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
                compressed_page_aligned_sparse_indices,
                sparse_topk_lens,
                raw_seq_lens,
                window_cache,
                compressed_cache,
                q_len=q_len,
            )
        with pytest.raises(RuntimeError, match="not CUDA Graph capture safe"):
            trtllm_batch_decode_sparse_mla_dsv4(
                query=query,
                swa_kv_cache=window_cache,
                workspace_buffer=workspace,
                sparse_indices=compressed_page_aligned_sparse_indices,
                compressed_kv_cache=compressed_cache,
                sparse_topk_lens=sparse_topk_lens,
                seq_lens=raw_seq_lens,
                bmm1_scale=softmax_scale,
                backend="cute-dsl",
                hca_sparse_indices_format="compressed-page-aligned",
                hca_use_persistent=True,
            )

    # Capture the private LSE scratch/result to verify the all-empty split-K
    # reduction. The public DSV4 API intentionally returns output only.
    from flashinfer.cute_dsl.attention.wrappers import batch_hca

    real_compile_hca_kernel = batch_hca._compile_hca_kernel
    captured = {}

    def compile_and_capture_lse(*args, **kwargs):
        compiled_kernel = real_compile_hca_kernel(*args, **kwargs)

        def run_and_capture_lse(*kernel_args, **kernel_kwargs):
            captured["lse"] = kernel_args[6]
            return compiled_kernel(*kernel_args, **kernel_kwargs)

        return run_and_capture_lse

    monkeypatch.setattr(batch_hca, "_compile_hca_kernel", compile_and_capture_lse)
    sparse_topk_lens.fill_(128)
    window_valid_lens.zero_()
    output = trtllm_batch_decode_sparse_mla_dsv4(
        query=query,
        swa_kv_cache=window_cache,
        workspace_buffer=workspace,
        sparse_indices=None,
        compressed_kv_cache=compressed_cache,
        sparse_topk_lens=sparse_topk_lens,
        bmm1_scale=softmax_scale,
        bmm2_scale=output_scale,
        sinks=None,
        swa_topk_lens=window_valid_lens,
        backend="cute-dsl",
        hca_swa_indices=window_indices,
        hca_compressed_block_tables=compressed_tables,
        hca_seq_lens=hca_seq_lens,
        hca_is_causal=True,
        hca_use_persistent=True,
    )
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    assert torch.isneginf(captured["lse"]).all()


@pytest.mark.parametrize(
    ("num_heads", "window_page_size", "compressed_page_size"),
    ((64, 32, 128), (128, 32, 16), (128, 16, 32)),
)
def test_cute_dsl_hca_partial_head_tile_and_empty_attention(
    num_heads, window_page_size, compressed_page_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(torch.device("cuda")) not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("CuTe DSL HCA requires SM100/SM103/SM107")
    if not is_cute_dsl_arch_supported(*get_compute_capability(torch.device("cuda"))):
        pytest.skip("the installed CuTe DSL cannot target this device")
    pytest.importorskip("cutlass")

    torch.manual_seed(23)
    device = torch.device("cuda")
    batch_size, q_len, head_dim = 1, 1, 512
    query = torch.randn((batch_size, q_len, num_heads, head_dim), device=device).clamp_(
        -2, 2
    )
    query = query.to(torch.float8_e4m3fn)
    active_window_pages = 128 // window_page_size
    active_compressed_pages = 256 // compressed_page_size
    window_table_pages = 2 * active_window_pages
    compressed_table_pages = 2 * active_compressed_pages
    window_cache = torch.randn(
        (window_table_pages, window_page_size, head_dim), device=device
    ).clamp_(-2, 2)
    window_cache = window_cache.to(torch.float8_e4m3fn)
    compressed_cache = torch.randn(
        (compressed_table_pages, compressed_page_size, head_dim), device=device
    ).clamp_(-2, 2)
    compressed_cache = compressed_cache.to(torch.float8_e4m3fn)
    window_indices = torch.remainder(
        torch.arange(128, dtype=torch.int32, device=device) + 17, 128
    ).unsqueeze(0)
    compressed_tables = torch.arange(
        compressed_table_pages, dtype=torch.int32, device=device
    ).unsqueeze(0)
    hca_seq_lens = torch.tensor([384], dtype=torch.int32, device=device)
    workspace = torch.empty(4 << 20, dtype=torch.uint8, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    sparse_topk_lens = torch.tensor([384], dtype=torch.int32, device=device)
    window_valid_lens = torch.tensor([128], dtype=torch.int32, device=device)
    sinks = torch.linspace(2.0, 3.0, num_heads, dtype=torch.float32, device=device)
    output = trtllm_batch_decode_sparse_mla_dsv4(
        query=query,
        swa_kv_cache=window_cache,
        workspace_buffer=workspace,
        compressed_kv_cache=compressed_cache,
        sparse_topk_lens=sparse_topk_lens,
        bmm1_scale=softmax_scale,
        sinks=sinks,
        swa_topk_lens=window_valid_lens,
        backend="cute-dsl",
        hca_swa_indices=window_indices,
        hca_compressed_block_tables=compressed_tables,
        hca_seq_lens=hca_seq_lens,
    )
    reference = _reference_hca(
        query,
        window_cache,
        compressed_cache,
        window_indices,
        compressed_tables,
        sparse_topk_lens,
        window_valid_lens,
        sinks,
        softmax_scale,
        1.0,
    )
    torch.testing.assert_close(output.float(), reference, atol=0.13, rtol=1e-5)

    # With no visible KV entry and no attention sink, the defined contribution
    # is O=0 rather than NaN. This also exercises the split_kv=1 epilogue.
    sparse_topk_lens.zero_().add_(128)
    window_valid_lens.zero_()
    output = trtllm_batch_decode_sparse_mla_dsv4(
        query=query,
        swa_kv_cache=window_cache,
        workspace_buffer=workspace,
        compressed_kv_cache=compressed_cache,
        sparse_topk_lens=sparse_topk_lens,
        bmm1_scale=softmax_scale,
        sinks=None,
        swa_topk_lens=window_valid_lens,
        backend="cute-dsl",
        hca_swa_indices=window_indices,
        hca_compressed_block_tables=compressed_tables,
        hca_seq_lens=hca_seq_lens,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
