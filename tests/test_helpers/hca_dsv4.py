# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU fixtures shared by HCA contract and mixed-device tests."""

import math

import torch


def _cpu_hca_inputs():
    batch_size, q_len, num_heads, head_dim = 2, 2, 128, 512
    rows = batch_size * q_len
    query = torch.empty(
        (batch_size, q_len, num_heads, head_dim), dtype=torch.float8_e4m3fn
    )
    window_cache = torch.empty((16, 32, head_dim), dtype=torch.float8_e4m3fn)
    compressed_cache = torch.empty((5, 128, head_dim), dtype=torch.float8_e4m3fn)
    window_positions = torch.arange(128, dtype=torch.int32)
    window_bases = torch.arange(rows, dtype=torch.int32).unsqueeze(1) * 128
    window_shifts = torch.tensor([7, 31, 63, 95], dtype=torch.int32).unsqueeze(1)
    window_indices = window_bases + torch.remainder(
        window_positions.unsqueeze(0) + window_shifts, 128
    )
    return {
        "query": query,
        "swa_kv_cache": window_cache,
        "workspace_buffer": torch.empty(4 << 20, dtype=torch.uint8),
        "sparse_indices": None,
        "compressed_kv_cache": compressed_cache,
        "sparse_topk_lens": torch.tensor([128, 129, 256, 257], dtype=torch.int32),
        "swa_topk_lens": torch.tensor([127, 128, 128, 128], dtype=torch.int32),
        "hca_swa_indices": window_indices.contiguous(),
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


def _cpu_compressed_page_aligned_sparse_hca_inputs():
    args = _cpu_hca_inputs()
    rows = args["query"].shape[0] * args["query"].shape[1]
    compressed_page_size = args["compressed_kv_cache"].shape[1]
    window_indices = args["hca_swa_indices"].clone()
    compressed_tables = torch.tensor([[0, 1]] * rows, dtype=torch.int32)

    def expand(tables, page_size):
        offsets = torch.arange(page_size, dtype=torch.int32)
        return (tables.unsqueeze(-1) * page_size + offsets).flatten(1)

    sparse_indices = torch.cat(
        (
            window_indices,
            expand(compressed_tables, compressed_page_size),
        ),
        dim=1,
    )
    window_valid_lens = args["swa_topk_lens"]
    sparse_topk_lens = args["sparse_topk_lens"]
    window_columns = torch.arange(128)
    sparse_indices[:, :128].masked_fill_(
        window_columns.unsqueeze(0) >= window_valid_lens.unsqueeze(1), -1
    )
    compressed_columns = torch.arange(sparse_indices.shape[1] - 128)
    sparse_indices[:, 128:].masked_fill_(
        compressed_columns.unsqueeze(0) >= (sparse_topk_lens - 128).unsqueeze(1),
        0,
    )

    args["sparse_indices"] = sparse_indices
    args["seq_lens"] = torch.tensor([128, 16512], dtype=torch.int32)
    args["hca_sparse_indices_format"] = "compressed-page-aligned"
    args["swa_topk_lens"] = None
    args["hca_swa_indices"] = None
    args["hca_compressed_block_tables"] = None
    args["hca_seq_lens"] = None
    expected = {
        "window_indices": torch.where(
            torch.arange(128).unsqueeze(0) < window_valid_lens.unsqueeze(1),
            window_indices,
            torch.zeros_like(window_indices),
        ),
        "compressed_block_tables": torch.tensor(
            [[0, 0], [0, 0], [0, 0], [0, 1]], dtype=torch.int32
        ),
        "hca_seq_lens": torch.tensor([129, 257], dtype=torch.int32),
        "window_valid_lens": window_valid_lens,
    }
    return args, expected


def _cpu_compressed_page_aligned_conversion_inputs(
    window_page_size, compressed_page_size, extra_padding=0
):
    rows, head_dim = 4, 512
    seq_lens = torch.tensor([128, 16512], dtype=torch.int32)
    sparse_topk_lens = torch.tensor([128, 129, 256, 257], dtype=torch.int32)
    window_valid_lens = torch.tensor([127, 128, 128, 128], dtype=torch.int32)
    window_page_count = 128 // window_page_size
    compressed_table_capacity = 256
    compressed_input_capacity = 132
    compressed_page_count = compressed_table_capacity // compressed_page_size
    window_tables = (
        torch.arange(rows * window_page_count, dtype=torch.int32).reshape(rows, -1) + 3
    ) % 32
    compressed_tables = (
        torch.arange(rows * compressed_page_count, dtype=torch.int32).reshape(rows, -1)
        + 5
    ) % 32

    def expand(tables, page_size):
        offsets = torch.arange(page_size, dtype=torch.int32)
        return (tables.unsqueeze(-1) * page_size + offsets).flatten(1)

    window_indices = expand(window_tables, window_page_size)
    shifts = torch.tensor([7, 31, 63, 95], dtype=torch.int64).unsqueeze(1)
    gather_columns = torch.remainder(
        torch.arange(128, dtype=torch.int64).unsqueeze(0) + shifts, 128
    )
    window_indices = window_indices.gather(1, gather_columns)
    compressed_indices = expand(compressed_tables, compressed_page_size)[
        :, :compressed_input_capacity
    ]
    window_positions = torch.arange(128)
    compressed_positions = torch.arange(compressed_input_capacity)
    window_indices.masked_fill_(
        window_positions.unsqueeze(0) >= window_valid_lens.unsqueeze(1),
        torch.iinfo(torch.int32).max,
    )
    active_compressed_lens = sparse_topk_lens - 128
    compressed_indices.masked_fill_(
        compressed_positions.unsqueeze(0) >= active_compressed_lens.unsqueeze(1),
        torch.iinfo(torch.int32).max,
    )
    if extra_padding:
        compressed_indices = torch.cat(
            (
                compressed_indices,
                torch.full(
                    (rows, extra_padding),
                    torch.iinfo(torch.int32).max,
                    dtype=torch.int32,
                ),
            ),
            dim=1,
        )

    sparse_indices = torch.cat((window_indices, compressed_indices), dim=1)
    expected_window = torch.where(
        torch.arange(128).unsqueeze(0) < window_valid_lens.unsqueeze(1),
        window_indices,
        torch.zeros_like(window_indices),
    )
    expected_compressed = torch.where(
        (torch.arange(compressed_page_count) * compressed_page_size).unsqueeze(0)
        < active_compressed_lens.unsqueeze(1),
        compressed_tables,
        0,
    )
    return {
        "sparse_indices": sparse_indices,
        "sparse_topk_lens": sparse_topk_lens,
        "seq_lens": seq_lens,
        "swa_kv_cache": torch.empty(
            (32, window_page_size, head_dim), dtype=torch.float8_e4m3fn
        ),
        "compressed_kv_cache": torch.empty(
            (32, compressed_page_size, head_dim), dtype=torch.float8_e4m3fn
        ),
        "q_len": 2,
        "expected_window": expected_window,
        "expected_compressed": expected_compressed,
        "expected_window_lens": window_valid_lens,
    }
