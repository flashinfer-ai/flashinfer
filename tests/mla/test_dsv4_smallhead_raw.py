# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw logical metadata must match packed attention after changing-input replay."""

import pytest
import torch

from flashinfer.mla import _smallhead, _smallhead_raw
from flashinfer.mla.dsv4_raw import (
    get_dsv4_raw_decode_workspace_size,
    try_batch_decode_sparse_mla_dsv4_raw,
)
from tests.mla.test_dsv4_smallhead import _check_fp64, _fixture, _reference


@pytest.fixture(autouse=True)
def require_b300():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("raw-index small-head decode is qualified on SM103")
    reason = _smallhead._dsl_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)


def _raw_fixture(rows, heads, page):
    packed, pools = _fixture(rows, heads, "HND")
    # Retain nonzero origins and padded physical pages from the shared fixture.
    pools[1] = pools[1][:, :, :page]
    packed["compressed_kv_cache"] = pools[1]
    names = (
        "query",
        "swa_kv_cache",
        "compressed_kv_cache",
        "seq_lens",
        "sinks",
        "bmm1_scale",
        "bmm2_scale",
        "enable_pdl",
    )
    raw = {name: packed[name] for name in names}
    raw["query"] = packed["query"].view(rows, heads, 512)
    raw.update(
        swa_indices=torch.empty((rows + 1, 128), device="cuda", dtype=torch.int32)[1:],
        compressed_indices=torch.empty(
            (rows + 1, 512), device="cuda", dtype=torch.int32
        )[1:],
        request_ids=torch.empty(rows + 4, device="cuda", dtype=torch.int32)[4:],
        compressed_block_table=torch.empty(
            (rows + 1, 16), device="cuda", dtype=torch.int32
        )[1:],
        token_valid=torch.empty(rows + 16, device="cuda", dtype=torch.bool)[16:],
        out=torch.empty((rows + 1, heads, 512), device="cuda", dtype=torch.bfloat16)[
            1:
        ],
        workspace_buffer=torch.empty(
            get_dsv4_raw_decode_workspace_size(rows, heads),
            device="cuda",
            dtype=torch.uint8,
        ),
    )
    return raw, packed, pools


def _change(raw, packed, pools, generation):
    torch.manual_seed(831 + generation * 31)
    rows, heads, _ = raw["query"].shape
    page = pools[1].shape[2]
    raw["query"].copy_(
        (torch.randn_like(raw["query"], dtype=torch.float32) * 32).to(
            raw["query"].dtype
        )
    )
    for pool in pools:
        pool.copy_((torch.randn_like(pool, dtype=torch.float32) * 8).to(pool.dtype))
    for name, limit in (("swa_indices", 16 * 32), ("compressed_indices", 16 * page)):
        raw[name].copy_(
            torch.randint(limit, raw[name].shape, device="cuda", dtype=torch.int32)
        )
    raw["request_ids"].copy_(
        (torch.arange(rows, device="cuda", dtype=torch.int32) + generation) % rows
    )
    raw["compressed_block_table"].copy_(
        torch.randint(16, (rows, 16), device="cuda", dtype=torch.int32)
    )
    raw["token_valid"].fill_(True)
    raw["swa_indices"][:, 3::37] = -1
    raw["seq_lens"].fill_(7 if generation == 1 else 8192)
    raw["sinks"].copy_(torch.linspace(-2, 2, heads, device="cuda"))
    if generation == 1:
        raw["compressed_indices"][:, 479:] = -1
    elif generation == 2:
        raw["swa_indices"][0].fill_(-1)
        raw["compressed_indices"][0].fill_(-1)
    elif generation == 3:
        raw["token_valid"].fill_(False)
        pools[1].fill_(float("nan"))
    elif generation == 4:
        raw["compressed_indices"][:, 3::37] = -1
        raw["compressed_block_table"][:, 3] = -1
    elif generation == 5:
        raw["sinks"].fill_(10000)

    # Independent explicit mapping for the existing packed attention contract.
    # Count defines a prefix; do not compact holes or drop invalid page entries.
    swa = raw["swa_indices"].long()
    local = raw["compressed_indices"].long()
    requests = raw["request_ids"].long()
    physical = raw["compressed_block_table"][
        requests[:, None], local.clamp_min(0) // page
    ]
    packed["sparse_indices"][:, :128] = torch.where(
        swa >= 0, swa // 32 * 364 + swa % 32, -1
    )
    packed["sparse_indices"][:, 128:] = torch.where(
        (local >= 0) & (physical >= 0), physical.long() * 364 + local % page, -1
    )
    packed["sparse_topk_lens"].copy_(
        128 + ((local >= 0) & raw["token_valid"][:, None]).sum(-1)
    )


def _packed_run(packed):
    names = (
        "query",
        "swa_kv_cache",
        "compressed_kv_cache",
        "sparse_indices",
        "sparse_topk_lens",
        "seq_lens",
        "sinks",
        "workspace_buffer",
        "out",
        "bmm1_scale",
        "bmm2_scale",
        "enable_pdl",
    )
    kwargs = {name: packed[name] for name in names}
    kwargs["query"] = kwargs["query"].squeeze(1)
    kwargs["out"] = kwargs["out"].squeeze(1)
    return _smallhead.run(**kwargs)


@pytest.mark.parametrize("heads,rows", [(16, 1), (16, 2), (16, 4), (32, 1), (32, 2)])
@pytest.mark.parametrize("page", [64, 128])
@torch.inference_mode()
def test_raw_metadata_replay_matches_packed_and_fp64(heads, rows, page):
    raw, packed, pools = _raw_fixture(rows, heads, page)
    _change(raw, packed, pools, 0)
    assert try_batch_decode_sparse_mla_dsv4_raw(**raw)
    _packed_run(packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert try_batch_decode_sparse_mla_dsv4_raw(**raw)
        _packed_run(packed)
    for generation in range(6):
        _change(raw, packed, pools, generation)
        expected = _reference(packed, pools)
        raw["out"].fill_(float("nan"))
        packed["out"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _check_fp64(raw["out"], expected)
        assert torch.equal(
            raw["out"].view(torch.uint8), packed["out"].squeeze(1).view(torch.uint8)
        )


@torch.inference_mode()
def test_raw_unprepared_capture_declines_without_writes():
    raw, packed, pools = _raw_fixture(1, 16, 64)
    _change(raw, packed, pools, 0)
    raw["out"].fill_(42)
    raw["workspace_buffer"].fill_(17)
    _smallhead_raw._get_kernel.cache_clear()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert not try_batch_decode_sparse_mla_dsv4_raw(**raw)
    graph.replay()
    torch.cuda.synchronize()
    assert bool((raw["out"] == 42).all())
    assert bool((raw["workspace_buffer"] == 17).all())
    assert try_batch_decode_sparse_mla_dsv4_raw(**raw)
    torch.cuda.synchronize()
    _check_fp64(raw["out"], _reference(packed, pools))


@pytest.mark.parametrize(
    "case",
    [
        "pdl_off",
        "pdl_not_bool",
        "nonfinite_scale",
        "wrong_query_dtype",
        "short_workspace",
        "unaligned_workspace",
        "output_alias",
        "workspace_alias",
        "strided_table",
        "wrong_table_dtype",
        "strided_cache_tokens",
        "wrong_validity_dtype",
        "wrong_output_shape",
        "neighbor_h16_r8",
        "neighbor_h32_r4",
        "cpu_metadata",
    ],
)
@torch.inference_mode()
def test_raw_unsupported_metadata_declines_without_writes(case, monkeypatch):
    raw, packed, pools = _raw_fixture(1, 16, 64)
    _change(raw, packed, pools, 0)
    if case == "pdl_off":
        raw["enable_pdl"] = False
    elif case == "pdl_not_bool":
        raw["enable_pdl"] = 1
    elif case == "nonfinite_scale":
        raw["bmm1_scale"] = float("nan")
    elif case == "wrong_query_dtype":
        raw["query"] = raw["query"].float()
    elif case == "short_workspace":
        raw["workspace_buffer"] = raw["workspace_buffer"][:-1]
    elif case == "unaligned_workspace":
        raw["workspace_buffer"] = torch.empty(
            raw["workspace_buffer"].numel() + 1, dtype=torch.uint8, device="cuda"
        )[1:]
    elif case == "output_alias":
        raw["out"] = pools[0][0, 0, :32].view(torch.bfloat16).view(1, 16, 512)
    elif case == "workspace_alias":
        raw["out"] = (
            raw["workspace_buffer"][: 16 * 512 * 2]
            .view(torch.bfloat16)
            .view(1, 16, 512)
        )
    elif case == "strided_table":
        raw["compressed_block_table"] = raw["compressed_block_table"][:, ::2]
    elif case == "wrong_table_dtype":
        raw["compressed_block_table"] = raw["compressed_block_table"].long()
    elif case == "strided_cache_tokens":
        raw["swa_kv_cache"] = pools[0][:, :, ::2]
    elif case == "wrong_validity_dtype":
        raw["token_valid"] = raw["token_valid"].to(torch.uint8)
    elif case == "wrong_output_shape":
        raw["out"] = raw["out"].unsqueeze(1)
    elif case in ("neighbor_h16_r8", "neighbor_h32_r4"):
        shape = (8, 16, 512) if case == "neighbor_h16_r8" else (4, 32, 512)
        raw["query"] = torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
    elif case == "cpu_metadata":
        raw["request_ids"] = raw["request_ids"].cpu()
    else:
        raise AssertionError(case)
    raw["workspace_buffer"].fill_(17)
    raw["out"].fill_(42)
    before = {
        name: raw[name].view(torch.uint8).clone()
        for name in ("out", "workspace_buffer")
    }

    def unexpected_execution(**kwargs):
        pytest.fail("Unsupported metadata reached the execution path")

    monkeypatch.setattr(_smallhead_raw, "run", unexpected_execution)
    assert not try_batch_decode_sparse_mla_dsv4_raw(**raw)
    for name, value in before.items():
        assert torch.equal(raw[name].view(torch.uint8), value)
