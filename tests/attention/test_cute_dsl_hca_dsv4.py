# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DeepSeek V4 CuTe DSL HCA decode integration."""

import math

import pytest
import torch

import flashinfer.mla._core as mla_core
from flashinfer.mla import (
    batch_decode_sparse_mla_dsv4,
    trtllm_batch_decode_sparse_mla_dsv4,
)
from flashinfer.utils import get_compute_capability


def _cpu_hca_inputs():
    batch_size, q_len, num_heads, head_dim = 2, 2, 128, 512
    rows = batch_size * q_len
    query = torch.empty(
        (batch_size, q_len, num_heads, head_dim), dtype=torch.float8_e4m3fn
    )
    window_cache = torch.empty((16, 32, head_dim), dtype=torch.float8_e4m3fn)
    compressed_cache = torch.empty((5, 128, head_dim), dtype=torch.float8_e4m3fn)
    return {
        "query": query,
        "swa_kv_cache": window_cache,
        "workspace_buffer": torch.empty(4 << 20, dtype=torch.uint8),
        "sparse_indices": None,
        "compressed_kv_cache": compressed_cache,
        "sparse_topk_lens": torch.tensor([128, 129, 256, 257], dtype=torch.int32),
        "swa_topk_lens": torch.tensor([127, 128, 128, 128], dtype=torch.int32),
        "hca_swa_block_tables": torch.zeros((rows, 4), dtype=torch.int32),
        "hca_compressed_block_tables": torch.zeros((rows, 2), dtype=torch.int32),
        "hca_seq_lens": torch.tensor([129, 257], dtype=torch.int32),
        "sinks": torch.zeros(num_heads, dtype=torch.float32),
        "out": torch.empty(
            (batch_size, q_len, num_heads, head_dim), dtype=torch.bfloat16
        ),
        "bmm1_scale": 1.0 / math.sqrt(head_dim),
        "bmm2_scale": 1.0,
        "backend": "cute-dsl",
        "hca_is_causal": True,
    }


def test_dsv4_backend_resolution_is_explicit(monkeypatch):
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    assert (
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "auto")
        == "trtllm-gen"
    )
    assert (
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cute-dsl")
        == "cute-dsl"
    )

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (12, 0))
    assert (
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "auto")
        == "sparse"
    )
    with pytest.raises(ValueError, match="requires SM100/SM103"):
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cute-dsl")

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 1))
    with pytest.raises(ValueError, match="supports SM100/SM103"):
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "auto")
    with pytest.raises(ValueError, match="requires SM100/SM103"):
        mla_core._resolve_dsv4_sparse_mla_backend(torch.device("cpu"), "cute-dsl")


def test_dsv4_backend_neutral_alias_is_backward_compatible():
    assert batch_decode_sparse_mla_dsv4 is trtllm_batch_decode_sparse_mla_dsv4
    assert batch_decode_sparse_mla_dsv4.__name__ == "batch_decode_sparse_mla_dsv4"


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_dsv4_hca_public_api_forwards_block_tables(monkeypatch, kv_layout):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.wrappers import batch_hca

    args = _cpu_hca_inputs()
    args["kv_layout"] = kv_layout
    args["hca_use_persistent"] = True
    if kv_layout == "NHD":
        args["swa_kv_cache"] = args["swa_kv_cache"].unsqueeze(2)
        args["compressed_kv_cache"] = args["compressed_kv_cache"].unsqueeze(2)
    captured = {}

    def fake_hca_decode(**kwargs):
        captured.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(batch_hca, "cute_dsl_hca_decode", fake_hca_decode)

    result = batch_decode_sparse_mla_dsv4(**args)
    assert result is args["out"]
    assert captured["window_block_tables"] is args["hca_swa_block_tables"]
    assert captured["compressed_block_tables"] is args["hca_compressed_block_tables"]
    assert captured["hca_seq_lens"] is args["hca_seq_lens"]
    assert captured["window_valid_lens"] is args["swa_topk_lens"]
    assert captured["is_persistent"] is True
    assert captured["window_kv_cache"].shape == (16, 32, 512)
    assert captured["compressed_kv_cache"].shape == (5, 128, 512)
    assert captured["window_kv_cache"].data_ptr() == args["swa_kv_cache"].data_ptr()
    assert (
        captured["compressed_kv_cache"].data_ptr()
        == args["compressed_kv_cache"].data_ptr()
    )


def test_dsv4_hca_rejects_legacy_token_indices(monkeypatch):
    args = _cpu_hca_inputs()
    args["sparse_indices"] = torch.zeros((4, 256), dtype=torch.int32)
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    with pytest.raises(ValueError, match="sparse_indices must be None"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)


def test_dsv4_hca_rejects_backend_specific_metadata(monkeypatch):
    args = _cpu_hca_inputs()
    args["hca_is_causal"] = False
    monkeypatch.setattr(mla_core, "get_compute_capability", lambda _device: (10, 0))
    with pytest.raises(ValueError, match="supports causal HCA only"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)

    args = _cpu_hca_inputs()
    args["seq_lens"] = torch.tensor([129, 257], dtype=torch.int32)
    with pytest.raises(ValueError, match="uses hca_seq_lens"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)

    args = _cpu_hca_inputs()
    args["backend"] = "trtllm-gen"
    args["sparse_indices"] = torch.zeros((4, 256), dtype=torch.int32)
    with pytest.raises(ValueError, match="does not accept hca_swa_block_tables"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)

    args = _cpu_hca_inputs()
    args["backend"] = "trtllm-gen"
    args["sparse_indices"] = torch.zeros((4, 256), dtype=torch.int32)
    args["hca_swa_block_tables"] = None
    args["hca_compressed_block_tables"] = None
    args["hca_seq_lens"] = None
    args["hca_is_causal"] = False
    with pytest.raises(ValueError, match="does not accept hca_is_causal"):
        trtllm_batch_decode_sparse_mla_dsv4(**args)


def test_hca_rejects_nonpersistent_grid_overflow():
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.wrappers.batch_hca import (
        _check_nonpersistent_grid,
    )

    _check_nonpersistent_grid(batch_size=1, q_len=8192)
    with pytest.raises(ValueError, match=r"batch_size \* q_len <= 65535"):
        _check_nonpersistent_grid(batch_size=8, q_len=8192)


def test_hca_opt_in_metadata_value_validation(monkeypatch):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.wrappers.batch_hca import (
        _validate_hca_values,
    )

    args = _cpu_hca_inputs()
    validation_args = {
        "window_kv_cache": args["swa_kv_cache"],
        "compressed_kv_cache": args["compressed_kv_cache"],
        "window_block_tables": args["hca_swa_block_tables"],
        "compressed_block_tables": args["hca_compressed_block_tables"],
        "hca_seq_lens": args["hca_seq_lens"],
        "sparse_topk_lens": args["sparse_topk_lens"],
        "window_valid_lens": args["swa_topk_lens"],
        "q_len": 2,
        "is_causal": True,
    }
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")

    # Only the fixed 128-slot window footprint and the hca_seq_lens-driven,
    # tile-rounded compressed footprint are loaded. Later columns may use
    # conventional -1 padding.
    validation_args["window_block_tables"] = torch.cat(
        (
            args["hca_swa_block_tables"],
            torch.full((4, 2), -1, dtype=torch.int32),
        ),
        dim=1,
    )
    validation_args["compressed_block_tables"] = torch.tensor(
        [[0, -1], [0, -1], [0, 1], [0, 1]], dtype=torch.int32
    )
    _validate_hca_values(**validation_args)

    validation_args["hca_seq_lens"] = torch.tensor([129, 385], dtype=torch.int32)
    with pytest.raises(ValueError, match="block-table capacity 384"):
        _validate_hca_values(**validation_args)

    validation_args["hca_seq_lens"] = args["hca_seq_lens"]
    validation_args["compressed_block_tables"][2, 1] = -1
    with pytest.raises(ValueError, match="compressed_block_tables active values"):
        _validate_hca_values(**validation_args)

    validation_args["compressed_block_tables"][2, 1] = 1
    validation_args["window_valid_lens"][0] = 0
    validation_args["window_block_tables"][0, 3] = -1
    with pytest.raises(ValueError, match="window_block_tables active values"):
        _validate_hca_values(**validation_args)


def test_hca_static_contract_accepts_fp8_to_bf16():
    pytest.importorskip("cutlass")
    import cutlass

    from flashinfer.cute_dsl.attention.dsa.hca_fp8 import (
        MAX_SPLITS,
        BlackwellHeavilyCompressedAttentionForwardFP8,
    )

    can_implement = BlackwellHeavilyCompressedAttentionForwardFP8.can_implement
    config = {
        "B": 2,
        "S": 2,
        "K": 384,
        "H": 128,
        "L": 512,
        "in_dtype": cutlass.Float8E4M3FN,
        "out_dtype": cutlass.BFloat16,
        "acc_dtype": cutlass.Float32,
        "lse_dtype": cutlass.Float32,
        "mma_qk_tiler_mn": (128, 128),
        "mma_pv_tiler_mn": (128, 256),
        "split_kv": 1,
        "is_persistent": False,
        "is_var_seq": True,
        "is_var_split_kv": False,
        "page_size_cmp": 128,
        "page_size_win": 32,
    }

    def supports(**overrides):
        return can_implement(**{**config, **overrides})

    assert supports()
    assert not supports(page_size_win=1)
    assert not supports(page_size_cmp=0)
    assert not supports(page_size_cmp=48)
    assert not supports(split_kv=0)
    assert not supports(split_kv=MAX_SPLITS + 1)
    assert not supports(B=8, S=8192)
    assert supports(B=8, S=8192, is_persistent=True)


def _reference_hca(
    query,
    window_cache,
    compressed_cache,
    window_tables,
    compressed_tables,
    sparse_topk_lens,
    window_valid_lens,
    sinks,
    softmax_scale,
    output_scale,
):
    batch_size, q_len, _, _ = query.shape
    output = torch.empty_like(query, dtype=torch.float32)
    for batch_idx in range(batch_size):
        for query_idx in range(q_len):
            row = batch_idx * q_len + query_idx
            window = window_cache.index_select(0, window_tables[row]).reshape(
                -1, query.shape[-1]
            )
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


@pytest.mark.arch_blackwell
def test_cute_dsl_hca_fp8_to_bf16_correctness(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("CuTe DSL HCA requires SM100/SM103")
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
    compressed_tables = torch.tensor(
        [[3, 4], [3, 4], [1, 0], [1, 0]], dtype=torch.int32, device=device
    )
    hca_seq_lens = torch.tensor([129, 257], dtype=torch.int32, device=device)
    sparse_topk_lens = torch.tensor(
        [128, 129, 256, 257], dtype=torch.int32, device=device
    )
    window_valid_lens = torch.tensor(
        [127, 128, 128, 128], dtype=torch.int32, device=device
    )
    workspace = torch.empty(8 << 20, dtype=torch.uint8, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    output_scale = 1.0

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
            hca_swa_block_tables=window_tables,
            hca_compressed_block_tables=compressed_tables,
            hca_seq_lens=hca_seq_lens,
            hca_is_causal=True,
            hca_use_persistent=True,
        )
        reference = _reference_hca(
            query,
            window_cache,
            compressed_cache,
            window_tables,
            compressed_tables,
            sparse_topk_lens,
            window_valid_lens,
            sinks,
            softmax_scale,
            output_scale,
        )
        assert output.dtype == torch.bfloat16
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, atol=0.13, rtol=1e-5)

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
        hca_swa_block_tables=window_tables,
        hca_compressed_block_tables=compressed_tables,
        hca_seq_lens=hca_seq_lens,
        hca_is_causal=True,
        hca_use_persistent=True,
    )
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    assert torch.isneginf(captured["lse"]).all()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("num_heads", "window_page_size", "compressed_page_size"),
    ((64, 32, 128), (128, 32, 16), (128, 16, 32)),
)
def test_cute_dsl_hca_partial_head_tile_and_empty_attention(
    num_heads, window_page_size, compressed_page_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("CuTe DSL HCA requires SM100/SM103")
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
    window_tables = torch.arange(
        window_table_pages, dtype=torch.int32, device=device
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
        hca_swa_block_tables=window_tables,
        hca_compressed_block_tables=compressed_tables,
        hca_seq_lens=hca_seq_lens,
    )
    reference = _reference_hca(
        query,
        window_cache,
        compressed_cache,
        window_tables,
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
        hca_swa_block_tables=window_tables,
        hca_compressed_block_tables=compressed_tables,
        hca_seq_lens=hca_seq_lens,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
