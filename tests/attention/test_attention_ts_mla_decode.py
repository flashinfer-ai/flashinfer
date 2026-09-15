# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public-interface acceptance coverage for PrimTS MLA decode."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, dataclass, replace
import inspect
import math
import textwrap
from types import MappingProxyType, SimpleNamespace
from typing import Sequence

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl==4.7.0",
)

import cutlass.pipeline as pipeline
from cutlass.experimental.task_scheduling.enums import (
    SignalingThreads,
    TileSchedulerType,
)
from cutlass.experimental.task_scheduling.resources import (
    PipelineConfig,
    TileSchedulerConfig,
)

from flashinfer.attention.prims_ts import (
    BatchMLADecodePagedTSWrapper,
    batch_mla_decode_with_paged_kv_cache,
)
from flashinfer.attention.prims_ts._balanced_scheduler import (
    B200_BALANCED_COST_MODEL_ID,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
    make_mla_decode_config,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
    build_mla_decode_task_manager,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.resources import (
    MlaWorkQueue,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.work_partition import (
    equal_split_row_prefix_active_split_count,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.kernel_policy import (
    select_balanced_decode_mla_kernel_policy,
    select_mla_ts_kernel,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.helpers.query import (
    FlatQueryTileLayout,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
    make_throughput_latency_mla_config,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
    ThroughputLatencyMlaDecodeTs,
)
from flashinfer.mla import (
    get_prims_ts_batch_mla_decode_workspace_size,
    prims_ts_batch_mla_decode_with_kv_cache,
)
import flashinfer.attention.prims_ts.mla_decode as mla_decode_module


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS MLA decode requires SM100 or SM103",
)


_FP8 = torch.float8_e4m3fn
_LATENT_DIM = 512
_ROPE_DIM = 64
_QK_DIM = _LATENT_DIM + _ROPE_DIM
_DEFAULT_PAGE_SIZE = 32
_FP8_PROBABILITY_SCALE = 448.0
# CUTLASS DSL CUDA graphs retain raw pointers into their wrapper and input
# allocations. Keep only graph-parity anchor owners alive until module teardown
# so later parametrized cases cannot reuse those allocations prematurely.
_CUDA_GRAPH_ANCHOR_OWNERS: list[tuple[object, ...]] = []
_MLA_INTERNAL_TUNING_PARAMETERS = frozenset(
    {
        "autotuner",
        "config",
        "head_dim_per_cta_v",
        "kernel",
        "num_ctas_per_head_dim",
        "num_insts_kv",
        "num_stages",
        "num_warps",
        "separate_reducer_impl",
        "split_kv",
        "tile_size_kv",
        "tile_size_q",
        "use_cluster_reduction",
        "use_clc_dynamic_persistent_scheduler",
        "use_persistent_scheduler",
    }
)
_INTERNAL_TUNING_TOKEN_PREFIXES = frozenset(
    {
        "autotun",
        "clc",
        "config",
        "cta",
        "impl",
        "inst",
        "kernel",
        "mma",
        "pdl",
        "persist",
        "profil",
        "reduc",
        "schedul",
        "split",
        "stag",
        "tile",
        "warp",
    }
)
_INTERNAL_TUNING_TOKEN_SEQUENCES = (
    ("groups", "tokens", "heads"),
    ("single", "kv"),
    ("tensor", "cores"),
)


@pytest.mark.parametrize(
    "num_heads,seq_len_q,expected",
    (
        (8, 1, "throughput_latency_1cta"),
        (32, 1, "throughput_latency_1cta"),
        (33, 1, "throughput_2cta"),
        (64, 1, "throughput_2cta"),
        (128, 1, "throughput_2cta"),
        (32, 2, None),
        (64, 2, None),
    ),
)
def test_attention_ts_mla_balanced_decode_family_policy(num_heads, seq_len_q, expected):
    """Keep balanced decode selection independent of ordinary split topology."""

    assert select_balanced_decode_mla_kernel_policy(num_heads, seq_len_q) == expected


@pytest.mark.parametrize(
    "num_heads,seq_len_q,tile_size_q,expected",
    (
        (128, 3, 128, (384, 3, 128)),
        (96, 3, 128, (288, 3, 32)),
        (12, 11, 128, (132, 2, 4)),
        (12, 2, 8, (24, 3, 8)),
        (96, 2, 64, (192, 3, 64)),
        (12, 4, 16, (48, 3, 16)),
        (48, 4, 64, (192, 3, 64)),
    ),
)
def test_attention_ts_mla_flat_query_tile_layout(
    num_heads, seq_len_q, tile_size_q, expected
):
    layout = FlatQueryTileLayout.for_tile(num_heads, seq_len_q, tile_size_q)
    assert (layout.total_rows, layout.num_tiles, layout.tail_rows) == expected
    assert layout.logical_num_heads_q == num_heads
    assert layout.logical_seq_len_q == seq_len_q
    assert layout.tile_size_q == tile_size_q


@pytest.mark.parametrize(
    "num_heads,seq_len_q,tile_size_q",
    ((0, 1, 128), (64, 0, 128), (64, 1, 0), (-1, 1, 8)),
)
def test_attention_ts_mla_flat_query_tile_layout_rejects_invalid_extent(
    num_heads, seq_len_q, tile_size_q
):
    with pytest.raises(ValueError):
        FlatQueryTileLayout.for_tile(num_heads, seq_len_q, tile_size_q)


@pytest.mark.parametrize(
    "row_tiles,expected_pieces",
    (
        (0, 0),
        (1, 1),
        (248, 62),
        (249, 63),
        (251, 63),
        (252, 64),
        (254, 64),
        (255, 65),
        (256, 65),
        (257, 65),
    ),
)
def test_attention_ts_mla_equal_split_causal_prefix_boundaries(
    row_tiles: int,
    expected_pieces: int,
):
    """Match 257/65 quotient-remainder descriptor boundaries exactly."""

    assert (
        equal_split_row_prefix_active_split_count(row_tiles, 257, 65) == expected_pieces
    )


@dataclass(frozen=True)
class _MLACase:
    query: torch.Tensor
    kv_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    page_size: int
    output_dtype: torch.dtype
    mask_type: str
    bmm1_scale: float
    bmm2_scale: float


def _variable_seq_lens(
    batch_size: int, max_seq_len: int, page_size: int
) -> tuple[int, ...]:
    """Return stable, varied runtime KV lengths with non-aligned tails."""

    if batch_size == 1 or max_seq_len <= 1:
        return (max_seq_len,) * batch_size
    lower = max(1, min(max_seq_len - 1, max(page_size, max_seq_len // 2)))
    candidates = tuple(
        length for length in range(lower, max_seq_len) if length % page_size != 0
    )
    if not candidates:
        candidates = tuple(
            length for length in range(1, max_seq_len) if length % page_size != 0
        )
    if not candidates:
        candidates = (max_seq_len - 1,)
    lengths = [max_seq_len]
    for batch_idx in range(1, batch_size):
        candidate_idx = (batch_idx * 104729 + batch_size * 37) % len(candidates)
        lengths.append(candidates[candidate_idx])
    return tuple(lengths)


def _stored(real: torch.Tensor, dtype: torch.dtype, scale: float) -> torch.Tensor:
    return (real / scale).to(dtype) if dtype == _FP8 else real.to(dtype)


def _make_mla_case(
    *,
    batch_size: int,
    num_qo_heads: int,
    max_seq_len: int,
    qkv_dtype: torch.dtype,
    seq_len_q: int = 1,
    mask_type: str = "causal",
    page_size: int = _DEFAULT_PAGE_SIZE,
    kv_seq_lens: Sequence[int] | None = None,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> _MLACase:
    """Create a deterministic dense-page MLA problem with shuffled page IDs."""

    seq_lens = (
        _variable_seq_lens(batch_size, max_seq_len, page_size)
        if kv_seq_lens is None
        else tuple(int(length) for length in kv_seq_lens)
    )
    if len(seq_lens) != batch_size:
        raise ValueError("kv_seq_lens must provide one length per request")
    if min(seq_lens) <= 0 or max(seq_lens) != max_seq_len:
        raise ValueError("KV lengths must be positive and include max_seq_len")
    if mask_type == "causal" and min(seq_lens) < seq_len_q:
        raise ValueError("causal KV sequences must be at least as long as Q")
    pages_per_request = tuple(
        (length + page_size - 1) // page_size for length in seq_lens
    )
    num_referenced_pages = sum(pages_per_request)
    num_physical_pages = num_referenced_pages + 7
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    page_ids = torch.randperm(num_physical_pages, generator=cpu_generator)[
        :num_referenced_pages
    ]
    if torch.equal(page_ids, torch.arange(num_referenced_pages)):
        page_ids = torch.roll(page_ids, 1)
    block_tables_cpu = torch.zeros(
        (batch_size, max(pages_per_request)), dtype=torch.int32
    )
    page_offset = 0
    for batch_idx, page_count in enumerate(pages_per_request):
        block_tables_cpu[batch_idx, :page_count] = page_ids[
            page_offset : page_offset + page_count
        ]
        page_offset += page_count

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    q_real = 0.2 * torch.randn(
        batch_size,
        seq_len_q,
        num_qo_heads,
        _QK_DIM,
        generator=generator,
        device=device,
    )
    kv_real = 0.2 * torch.randn(
        num_physical_pages,
        1,
        page_size,
        _QK_DIM,
        generator=generator,
        device=device,
    )
    q_scale, kv_scale = (0.0625, 0.125) if qkv_dtype == _FP8 else (1.0, 1.0)
    return _MLACase(
        query=_stored(q_real, qkv_dtype, q_scale),
        kv_cache=_stored(kv_real, qkv_dtype, kv_scale),
        block_tables=block_tables_cpu.to(device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        max_seq_len=max_seq_len,
        page_size=page_size,
        output_dtype=torch.bfloat16,
        mask_type=mask_type,
        bmm1_scale=q_scale * kv_scale / math.sqrt(128 + _ROPE_DIM),
        bmm2_scale=kv_scale,
    )


def _pack_mla_case(
    case: _MLACase,
    q_lens: Sequence[int],
) -> tuple[_MLACase, torch.Tensor]:
    """Pack fixed-Q MLA storage and return cumulative runtime Q offsets."""

    if case.query.ndim != 4 or len(q_lens) != case.query.shape[0]:
        raise ValueError("packed-Q source must be [B, SQ, H, 576] with B lengths")
    if min(q_lens) < 0 or max(q_lens) > case.query.shape[1]:
        raise ValueError("packed Q lengths must be nonnegative and within source SQ")
    offsets = [0]
    for q_len in q_lens:
        offsets.append(offsets[-1] + q_len)
    qo_indptr = torch.tensor(offsets, dtype=torch.int32, device=case.query.device)
    packed_query = torch.cat(
        [case.query[batch_idx, :q_len] for batch_idx, q_len in enumerate(q_lens)]
    ).contiguous()
    return replace(case, query=packed_query), qo_indptr


def _with_trt_plane0_block_tables(
    case: _MLACase,
) -> tuple[_MLACase, torch.Tensor, int]:
    """Expose K-plane metadata with TRT's two-plane row stride and an offset."""

    batch_size, columns = case.block_tables.shape
    storage = torch.empty(
        1 + batch_size * 2 * columns,
        dtype=torch.int32,
        device=case.block_tables.device,
    )
    trt_block_offsets = storage[1:].view(batch_size, 2, columns)
    trt_block_offsets[:, 0].copy_(case.block_tables)
    used_pages = set(int(page) for row in case.block_tables.tolist() for page in row)
    poison_page = next(
        page for page in range(int(case.kv_cache.shape[0])) if page not in used_pages
    )
    trt_block_offsets[:, 1].fill_(poison_page)
    block_tables = trt_block_offsets[:, 0]
    assert block_tables.shape == case.block_tables.shape
    assert block_tables.stride() == (2 * columns, 1)
    assert block_tables.storage_offset() == 1
    assert block_tables.data_ptr() % 4 == 0
    assert block_tables.data_ptr() % 16 != 0
    return replace(case, block_tables=block_tables), trt_block_offsets, poison_page


def _gather_request_cache(case: _MLACase, batch_idx: int) -> torch.Tensor:
    seq_len = int(case.seq_lens[batch_idx].item())
    page_count = (seq_len + case.page_size - 1) // case.page_size
    page_ids = case.block_tables[batch_idx, :page_count].long()
    cache_pages = case.kv_cache[:, 0] if case.kv_cache.ndim == 4 else case.kv_cache
    return cache_pages[page_ids].reshape(-1, _QK_DIM)[:seq_len].float()


def _fill_logical_cache_tokens(
    case: _MLACase,
    *,
    token_begin: int,
    token_end: int,
    value: float,
) -> None:
    """Fill a request's latent values through its shuffled logical page map."""

    if case.block_tables.shape[0] != 1:
        raise ValueError("marker helper requires exactly one request")
    stored_value = value / case.bmm2_scale
    for token_idx in range(token_begin, token_end):
        page_slot, page_offset = divmod(token_idx, case.page_size)
        physical_page = int(case.block_tables[0, page_slot].item())
        case.kv_cache[physical_page, 0, page_offset, :_LATENT_DIM] = stored_value


def _visible_kv_len(
    *, kv_len: int, seq_len_q: int, query_idx: int, mask_type: str
) -> int:
    if mask_type == "dense":
        return kv_len
    if mask_type == "causal":
        return kv_len - seq_len_q + query_idx + 1
    raise ValueError("mask_type must be 'dense' or 'causal'")


def _fp8_request_reference(
    q_stored: torch.Tensor,
    cache_stored: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float,
    num_insts_kv: int,
    tile_size_kv: int,
    splits_kv: int,
) -> torch.Tensor:
    """Model P448 probabilities across split-local KV instruction streams."""

    if num_insts_kv <= 0 or tile_size_kv <= 0 or splits_kv <= 0:
        raise ValueError("KV instruction, tile, and split counts must be positive")

    scores = (
        q_stored[:, :_LATENT_DIM] @ cache_stored[:, :_LATENT_DIM].T
        + q_stored[:, _LATENT_DIM:] @ cache_stored[:, _LATENT_DIM:].T
    )
    num_heads, seq_len = scores.shape
    num_tiles = (seq_len + tile_size_kv - 1) // tile_size_kv
    tiles_per_group = splits_kv * num_insts_kv
    groups_per_split = (num_tiles + tiles_per_group - 1) // tiles_per_group
    local_tiles = max(groups_per_split * num_insts_kv, num_insts_kv)
    active_splits = (num_tiles + local_tiles - 1) // local_tiles
    stream_tiles = []
    for split_idx in range(active_splits):
        split_begin = split_idx * local_tiles
        split_end = min(split_begin + local_tiles, num_tiles)
        for instance_idx in range(num_insts_kv):
            tile_indices = range(split_begin + instance_idx, split_end, num_insts_kv)
            if tile_indices.start < tile_indices.stop:
                stream_tiles.append(tile_indices)
    stream_max = [
        torch.full((num_heads,), -torch.inf, device=scores.device) for _ in stream_tiles
    ]
    stream_sum = [torch.zeros(num_heads, device=scores.device) for _ in stream_tiles]
    stream_acc = [
        torch.zeros((num_heads, _LATENT_DIM), device=scores.device)
        for _ in stream_tiles
    ]
    stream_valid = [False] * len(stream_tiles)
    for stream_idx, tile_indices in enumerate(stream_tiles):
        for tile_idx in tile_indices:
            begin = tile_idx * tile_size_kv
            end = min(begin + tile_size_kv, seq_len)
            tile_scores = scores[:, begin:end]
            local_max = tile_scores.max(dim=-1).values
            new_max = (
                torch.maximum(stream_max[stream_idx], local_max)
                if stream_valid[stream_idx]
                else local_max
            )
            probabilities = (
                torch.exp((tile_scores - new_max[:, None]) * bmm1_scale)
                * _FP8_PROBABILITY_SCALE
            )
            quantized_probabilities = probabilities.to(_FP8).float()
            tile_sum = probabilities.sum(dim=-1)
            tile_acc = quantized_probabilities @ cache_stored[begin:end, :_LATENT_DIM]
            if stream_valid[stream_idx]:
                correction = torch.exp((stream_max[stream_idx] - new_max) * bmm1_scale)
                stream_sum[stream_idx] = stream_sum[stream_idx] * correction + tile_sum
                stream_acc[stream_idx] = (
                    stream_acc[stream_idx] * correction[:, None] + tile_acc
                )
            else:
                stream_sum[stream_idx] = tile_sum
                stream_acc[stream_idx] = tile_acc
                stream_valid[stream_idx] = True
            stream_max[stream_idx] = new_max

    final_max = (
        torch.stack(
            [
                value
                for value, valid in zip(stream_max, stream_valid, strict=True)
                if valid
            ]
        )
        .max(dim=0)
        .values
    )
    final_sum = torch.zeros_like(final_max)
    final_acc = torch.zeros_like(stream_acc[0])
    for maximum, denominator, accumulator, valid in zip(
        stream_max, stream_sum, stream_acc, stream_valid, strict=True
    ):
        if valid:
            correction = torch.exp((maximum - final_max) * bmm1_scale)
            final_sum += denominator * correction
            final_acc += accumulator * correction[:, None]
    return final_acc / final_sum[:, None] * bmm2_scale


@torch.no_grad()
def _mla_reference(
    case: _MLACase,
    *,
    num_insts_kv: int,
    tile_size_kv: int,
    splits_kv: int = 1,
    batch_indices: Sequence[int] | None = None,
    qo_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the selected policy's independent FP32 MLA output oracle."""

    if batch_indices is None:
        batch_indices = range(case.query.shape[0])
    outputs = []
    for batch_idx in batch_indices:
        cache = _gather_request_cache(case, int(batch_idx))
        request_outputs = []
        if qo_indptr is None:
            request_queries = case.query[batch_idx]
        else:
            q_begin = int(qo_indptr[batch_idx].item())
            q_end = int(qo_indptr[batch_idx + 1].item())
            request_queries = case.query[q_begin:q_end]
        for query_idx in range(request_queries.shape[0]):
            visible = _visible_kv_len(
                kv_len=cache.shape[0],
                seq_len_q=request_queries.shape[0],
                query_idx=query_idx,
                mask_type=case.mask_type,
            )
            q_stored = request_queries[query_idx].float()
            visible_cache = cache[:visible]
            if case.query.dtype == _FP8:
                output = _fp8_request_reference(
                    q_stored,
                    visible_cache,
                    bmm1_scale=case.bmm1_scale,
                    bmm2_scale=case.bmm2_scale,
                    num_insts_kv=num_insts_kv,
                    tile_size_kv=tile_size_kv,
                    splits_kv=splits_kv,
                )
            else:
                scores = (
                    q_stored[:, :_LATENT_DIM] @ visible_cache[:, :_LATENT_DIM].T
                    + q_stored[:, _LATENT_DIM:] @ visible_cache[:, _LATENT_DIM:].T
                )
                probabilities = torch.softmax(scores * case.bmm1_scale, dim=-1)
                output = (
                    probabilities @ visible_cache[:, :_LATENT_DIM] * case.bmm2_scale
                )
            request_outputs.append(output)
        if request_outputs:
            outputs.append(torch.stack(request_outputs))
        else:
            outputs.append(
                torch.empty(
                    (0, request_queries.shape[1], _LATENT_DIM),
                    dtype=torch.float32,
                    device=request_queries.device,
                )
            )
    return torch.cat(outputs) if qo_indptr is not None else torch.stack(outputs)


def _mla_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return (5e-2, 1.5e-3) if dtype == _FP8 else (1e-2, 5e-4)


def _case(
    batch_size: int,
    num_qo_heads: int,
    max_seq_len: int,
    qkv_dtype: torch.dtype,
    seed: int,
    *,
    seq_len_q: int = 1,
    mask_type: str = "causal",
    page_size: int = _DEFAULT_PAGE_SIZE,
) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "num_qo_heads": num_qo_heads,
        "seq_len_q": seq_len_q,
        "max_seq_len": max_seq_len,
        "qkv_dtype": qkv_dtype,
        "mask_type": mask_type,
        "page_size": page_size,
        "seed": seed,
    }


def _param(
    case_kwargs: dict[str, object],
    expected_policy: dict[str, object],
    correction_pattern: str | None,
    overprovision: bool,
    *,
    exercise_all_paths: bool = False,
    id: str,
):
    return pytest.param(
        case_kwargs,
        expected_policy,
        correction_pattern,
        overprovision,
        exercise_all_paths,
        id=id,
    )


_MLA_CASES = (
    _param(
        _case(2, 8, 64, torch.bfloat16, 32000),
        {"kernel": "throughput_2cta"},
        None,
        False,
        id="bf16-short-k-2cta-fallback",
    ),
    _param(
        _case(2, 8, 2048, torch.bfloat16, 32001),
        {
            "kernel": "throughput_latency_1cta",
            "use_cluster_reduction": True,
        },
        "identity",
        False,
        exercise_all_paths=True,
        id="bf16-1cta-cluster-reduction",
    ),
    _param(
        _case(4, 16, 4097, torch.float8_e4m3fn, 32002),
        {
            "kernel": "throughput_latency_1cta",
            "separate_reducer_impl": "parallel",
        },
        "mixed",
        True,
        id="fp8-1cta-parallel-reduction",
    ),
    _param(
        _case(128, 32, 2048, torch.bfloat16, 32003),
        {"kernel": "throughput_latency_1cta", "split_kv": 1},
        None,
        False,
        id="bf16-1cta-direct",
    ),
    _param(
        _case(128, 64, 2048, torch.float8_e4m3fn, 32004),
        {"kernel": "throughput_latency_1cta", "split_kv": 1},
        "identity",
        False,
        id="fp8-1cta-direct",
    ),
    _param(
        _case(8, 128, 2048, torch.float8_e4m3fn, 32005),
        {
            "kernel": "throughput_2cta",
            "separate_reducer_impl": "reference",
        },
        "mixed",
        False,
        id="fp8-2cta-reference-reduction",
    ),
    _param(
        _case(
            4,
            16,
            4097,
            torch.float8_e4m3fn,
            32006,
            seq_len_q=4,
        ),
        {
            "kernel": "throughput_latency_1cta",
            "separate_reducer_impl": "parallel",
        },
        "mixed",
        False,
        id="fp8-multi-q-1cta-parallel-reduction",
    ),
    _param(
        _case(4, 16, 4097, torch.bfloat16, 32007, seq_len_q=8),
        {
            "kernel": "throughput_2cta",
            "separate_reducer_impl": "reference",
        },
        "identity",
        False,
        exercise_all_paths=True,
        id="bf16-multi-q-2cta-reference-reduction",
    ),
    _param(
        _case(
            2,
            16,
            2049,
            torch.bfloat16,
            32008,
            seq_len_q=4,
            mask_type="dense",
        ),
        {
            "kernel": "throughput_latency_1cta",
            "separate_reducer_impl": "parallel",
        },
        "tail",
        False,
        id="speculative-dense-tail",
    ),
    _param(
        _case(2, 16, 2049, torch.bfloat16, 32008, seq_len_q=4),
        {
            "kernel": "throughput_latency_1cta",
            "separate_reducer_impl": "parallel",
        },
        "tail",
        False,
        id="speculative-causal-tail",
    ),
    _param(
        _case(128, 128, 2048, torch.bfloat16, 32009),
        {
            "kernel": "throughput_2cta",
            "split_kv": 1,
            "use_persistent_scheduler": True,
            "use_clc_dynamic_persistent_scheduler": True,
        },
        None,
        False,
        id="bf16-2cta-persistent-direct",
    ),
    _param(
        _case(5, 65, 256, torch.bfloat16, 32010, seq_len_q=2),
        {
            "kernel": "throughput_2cta",
            "separate_reducer_impl": "reference",
        },
        None,
        False,
        id="bf16-2cta-reference-reducer-tail",
    ),
    _param(
        _case(
            128,
            128,
            4097,
            torch.bfloat16,
            32011,
            page_size=128,
        ),
        {
            "kernel": "throughput_2cta",
            "split_kv": 1,
            "use_persistent_scheduler": True,
            "use_clc_dynamic_persistent_scheduler": True,
        },
        None,
        False,
        id="bf16-page128-2cta-persistent-direct",
    ),
)


def _plan_case(
    case,
    *,
    qo_indptr: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
    balanced: bool = False,
):
    wrapper = BatchMLADecodePagedTSWrapper()
    num_heads = int(
        case.query.shape[1] if qo_indptr is not None else case.query.shape[2]
    )
    resolved_max_seq_len_q = (
        int(case.query.shape[1])
        if qo_indptr is None and max_seq_len_q is None
        else max_seq_len_q
    )
    if resolved_max_seq_len_q is None:
        raise ValueError("packed plan coverage requires max_seq_len_q")
    plan = wrapper.plan_balanced if balanced else wrapper.plan
    plan_kwargs = {}
    if balanced:
        plan_kwargs["seq_lens"] = tuple(int(value) for value in case.seq_lens.cpu())
    plan(
        case.query.device,
        int(case.block_tables.shape[0]),
        num_heads,
        _LATENT_DIM,
        _ROPE_DIM,
        case.page_size,
        case.max_seq_len,
        max_seq_len_q=resolved_max_seq_len_q,
        packed_query=qo_indptr is not None,
        q_data_type=case.query.dtype,
        kv_data_type=case.kv_cache.dtype,
        o_data_type=case.output_dtype,
        mask_type=case.mask_type,
        **plan_kwargs,
    )
    return wrapper


def _run_case(wrapper, case, *, qo_indptr=None, out=None, validate=True):
    return wrapper.run(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens,
        qo_indptr=qo_indptr,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out=out,
        validate=validate,
    )


@pytest.mark.parametrize(
    "batch_size,num_qo_heads,balanced_kernel,ordinary_kernel",
    (
        (64, 32, "throughput_latency_1cta", "throughput_2cta"),
        (32, 64, "throughput_2cta", "throughput_latency_1cta"),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_decode_family_overrides_only_balanced_policy(
    batch_size: int,
    num_qo_heads: int,
    balanced_kernel: str,
    ordinary_kernel: str,
):
    """Apply the calibrated family crossover without changing ordinary MLA."""

    common = (
        torch.cuda.current_device(),
        batch_size,
        num_qo_heads,
        _LATENT_DIM,
        _ROPE_DIM,
        _DEFAULT_PAGE_SIZE,
        131072,
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "causal",
        1,
    )
    balanced = mla_decode_module._resolve_mla_decode_launch_spec(*common, True)
    ordinary = mla_decode_module._resolve_mla_decode_launch_spec(*common, False)

    assert dict(balanced.policy)["kernel"] == balanced_kernel
    assert dict(ordinary.policy)["kernel"] == ordinary_kernel


@pytest.mark.parametrize(
    ("num_qo_heads", "expected_kernel"),
    (
        pytest.param(16, "throughput_latency_1cta", id="1cta"),
        pytest.param(64, "throughput_2cta", id="2cta-h64"),
        pytest.param(128, "throughput_2cta", id="2cta"),
    ),
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_kernel_dtype_product(
    num_qo_heads: int,
    expected_kernel: str,
    qkv_dtype: torch.dtype,
):
    """Exercise balanced 1CTA/2CTA producers with BF16 and E4M3 inputs."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=num_qo_heads,
        max_seq_len=32768,
        kv_seq_lens=(32768, 257),
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=33100 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    wrapper = _plan_case(case, balanced=True)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == expected_kernel
    assert policy["balanced_scheduler"] is True
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.balanced_plan is not None

    output = _run_case(wrapper, case)
    _assert_case_correct(output, case, policy)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_b64_h32_uses_calibrated_1cta_family():
    """Cover the bucket where ordinary direct-output selection chose 2CTA."""

    batch_size = 64
    case = _make_mla_case(
        batch_size=batch_size,
        num_qo_heads=32,
        max_seq_len=32768,
        kv_seq_lens=(32768,) + (257,) * (batch_size - 1),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=33164,
    )
    wrapper = _plan_case(case, balanced=True)
    policy = _policy_dict(wrapper)

    assert policy["kernel"] == "throughput_latency_1cta"
    output = _run_case(wrapper, case)
    _assert_case_correct(output, case, policy)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("num_qo_heads", "expected_tile_size_q"),
    (
        pytest.param(4, 8, id="m8"),
        pytest.param(8, 16, id="m16"),
        pytest.param(16, 32, id="m32"),
    ),
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.parametrize("packed", (False, True), ids=("fixed", "packed"))
def test_attention_ts_mla_balanced_1cta_causal_mask_respects_descriptor_end(
    num_qo_heads: int,
    expected_tile_size_q: int,
    qkv_dtype: torch.dtype,
    packed: bool,
):
    """Do not expose a paired MMA tile beyond an odd balanced descriptor."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=num_qo_heads,
        seq_len_q=2,
        max_seq_len=257,
        kv_seq_lens=(257,),
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=33400
        + expected_tile_size_q
        + (1 if qkv_dtype == _FP8 else 0)
        + (2 if packed else 0),
    )
    case.query.zero_()
    case.kv_cache.zero_()
    # The final page in descriptor [0, 1) is deliberately nonzero. The
    # swaps-MMA-AB producer pads that descriptor to a second 128-token tile;
    # clamped page lookup must not make four copies of this page visible.
    _fill_logical_cache_tokens(case, token_begin=96, token_end=128, value=1.0)

    qo_indptr = None
    if packed:
        case, qo_indptr = _pack_mla_case(case, (2,))
    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=2 if packed else None,
        balanced=True,
    )
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_latency_1cta"
    assert policy["balanced_scheduler"] is True
    assert policy["tile_size_q"] == expected_tile_size_q
    plan = wrapper._plan_state.balanced_plan
    assert plan is not None
    descriptors = plan.work_descriptors[: plan.last_descriptor_count].cpu()
    assert torch.any(descriptors[:, 2] - descriptors[:, 1] == 1)

    output_shape = (
        (2, num_qo_heads, _LATENT_DIM) if packed else (1, 2, num_qo_heads, _LATENT_DIM)
    )
    graph_out = torch.empty(output_shape, dtype=torch.bfloat16, device="cuda")
    _run_case(wrapper, case, qo_indptr=qo_indptr, out=graph_out)
    _assert_case_correct(graph_out, case, policy, qo_indptr=qo_indptr)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(
            wrapper,
            case,
            qo_indptr=qo_indptr,
            out=graph_out,
            validate=False,
        )
    assert captured is graph_out

    for runtime_seq_len in (129, 257):
        wrapper.replan((runtime_seq_len,))
        case.seq_lens.fill_(runtime_seq_len)
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_case_correct(graph_out, case, policy, qo_indptr=qo_indptr)

    _CUDA_GRAPH_ANCHOR_OWNERS.append((graph, graph_out, wrapper, case, qo_indptr))


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_forced_m32_reducer_uses_logical_query_shape(
    monkeypatch,
):
    """Keep both K129 pieces visible for logical H128/SQ1 on physical M32."""

    from flashinfer.attention.prims_ts.kernels.mla_decode import kernel_policy
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta import (
        config as one_cta_config,
    )

    def force_one_cta_policy(*args, **kwargs):
        del args, kwargs
        return "throughput_latency_1cta", "forced-test"

    def force_m32_launch_shape(*, num_heads_q, seq_len_q):
        return one_cta_config.FlatQueryLaunchShape.for_tile(
            num_heads_q,
            seq_len_q,
            32,
        )

    monkeypatch.setattr(
        kernel_policy,
        "resolve_mla_kernel_policy",
        force_one_cta_policy,
    )
    monkeypatch.setattr(
        one_cta_config,
        "resolve_auto_flat_query_launch_shape",
        force_m32_launch_shape,
    )
    mla_decode_module._resolve_mla_decode_launch_spec.cache_clear()
    try:
        case = _make_mla_case(
            batch_size=1,
            num_qo_heads=128,
            seq_len_q=1,
            max_seq_len=129,
            kv_seq_lens=(129,),
            qkv_dtype=torch.bfloat16,
            device="cuda",
            seed=33529,
        )
        wrapper = _plan_case(case, balanced=True)
        assert wrapper._plan_state is not None
        policy = dict(wrapper._plan_state.policy)
        assert policy["kernel"] == "throughput_latency_1cta"
        assert policy["source"] == "forced-test"
        assert policy["tile_size_q"] == 32
        assert int(policy["split_kv"]) > 1

        output = _run_case(wrapper, case)

        _assert_case_correct(output, case, policy)
    finally:
        # Do not retain the test-only forced policy after monkeypatch teardown.
        mla_decode_module._resolve_mla_decode_launch_spec.cache_clear()


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_2cta_keeps_flat_query_tile_layout():
    """Keep H96/SQ2 producer and reducer rows on the shared M128 layout."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=96,
        seq_len_q=2,
        max_seq_len=32768,
        kv_seq_lens=(32768, 257),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=33196,
    )
    wrapper = _plan_case(case, balanced=True)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_2cta"
    assert policy["balanced_scheduler"] is True
    assert int(policy["split_kv"]) > 1

    output = _run_case(wrapper, case)
    _assert_case_correct(output, case, policy)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_2cta_keeps_flat_packed_query_tile_layout():
    """Map uneven packed H96 queries through the same flat M128 producer tiles."""

    q_lens = (2, 1, 0)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=96,
        seq_len_q=max(q_lens),
        max_seq_len=32768,
        kv_seq_lens=(32768, 2049, 257),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=33296,
    )
    case, qo_indptr = _pack_mla_case(case, q_lens)
    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max(q_lens),
        balanced=True,
    )
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_2cta"
    assert policy["balanced_scheduler"] is True
    assert int(policy["split_kv"]) > 1

    output = _run_case(wrapper, case, qo_indptr=qo_indptr)
    _assert_case_correct(output, case, policy, qo_indptr=qo_indptr)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_causal_reducer_uses_equal_split_boundaries():
    """Include the final equal-split piece intersecting a causal row prefix."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=128,
        seq_len_q=2,
        max_seq_len=32769,
        kv_seq_lens=(32769,),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=33328,
    )
    wrapper = _plan_case(case, balanced=True)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_2cta"
    assert policy["balanced_scheduler"] is True
    assert wrapper._plan_state is not None
    plan = wrapper._plan_state.balanced_plan
    assert plan is not None
    # This regression targets the four-tile equal-split geometry, independent of
    # whichever target the device's calibrated cost model currently selects.
    plan._replan_forced_target((32769,), 4)
    assert plan.last_target_piece_tiles == 4
    assert plan.last_combine_request_count == 1
    combine = plan.combine_descriptors[0].cpu().tolist()
    assert combine[:3] == [0, 65, 0]

    graph_out = torch.empty(
        (1, 2, 128, _LATENT_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    eager = _run_case(wrapper, case, out=graph_out)
    _assert_case_correct(eager, case, policy)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(wrapper, case, out=graph_out, validate=False)
    assert captured is graph_out

    wrapper.replan((32769,))
    kernel_workspace = wrapper._plan_state.workspace_views.kernel_workspace
    assert kernel_workspace is not None
    kernel_workspace.fill_(-1)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(graph_out, case, policy)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
def test_attention_ts_mla_balanced_1cta_m64_uses_persistent_work_queue(
    qkv_dtype: torch.dtype,
):
    """Keep descriptor persistence enabled for balanced keeps-MMA-AB."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=16,
        seq_len_q=3,
        max_seq_len=4097,
        kv_seq_lens=(4097,),
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=33164 + (1 if qkv_dtype == _FP8 else 0),
    )
    wrapper = _plan_case(case, balanced=True)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_latency_1cta"
    assert policy["balanced_scheduler"] is True
    assert policy["tile_size_q"] == 64
    assert policy["use_persistent_scheduler"] is True
    assert policy["use_clc_dynamic_persistent_scheduler"] is False
    assert int(policy["split_kv"]) > 1

    output = _run_case(wrapper, case)
    _assert_case_correct(output, case, policy)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("num_qo_heads", "seq_len_q", "max_seq_len"),
    (
        pytest.param(6, 1, 32768, id="m8-padded-heads"),
        pytest.param(16, 3, 4097, id="m64-padded-flat-q"),
    ),
)
def test_attention_ts_mla_balanced_1cta_reducer_uses_logical_output_stride(
    num_qo_heads: int,
    seq_len_q: int,
    max_seq_len: int,
):
    """Publish padded 1CTA replays within the logical public batch extent."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=num_qo_heads,
        seq_len_q=seq_len_q,
        max_seq_len=max_seq_len,
        kv_seq_lens=(max_seq_len, max_seq_len),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=33106 + seq_len_q,
    )
    wrapper = _plan_case(case, balanced=True)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_latency_1cta"
    assert policy["balanced_scheduler"] is True
    logical_rows = seq_len_q * num_qo_heads
    tile_size_q = int(policy["tile_size_q"])
    physical_rows = tile_size_q * math.ceil(logical_rows / tile_size_q)
    assert physical_rows > logical_rows
    assert int(policy["split_kv"]) > 1

    output_elements = 2 * logical_rows * _LATENT_DIM
    guard_elements = 128 * _LATENT_DIM
    sentinel = 123.0
    storage = torch.full(
        (guard_elements + output_elements + guard_elements,),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = storage[guard_elements : guard_elements + output_elements].view(
        2, seq_len_q, num_qo_heads, _LATENT_DIM
    )

    def assert_output_and_guards():
        torch.testing.assert_close(
            storage[:guard_elements],
            torch.full_like(storage[:guard_elements], sentinel),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            storage[-guard_elements:],
            torch.full_like(storage[-guard_elements:], sentinel),
            rtol=0,
            atol=0,
        )
        _assert_case_correct(output, case, policy)

    _run_case(wrapper, case, out=output)
    torch.cuda.synchronize()
    assert_output_and_guards()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(wrapper, case, out=output, validate=False)
    assert captured is output

    # The captured topology first runs split work, switches entirely to direct
    # output, then returns to split reduction without changing public strides.
    for runtime_seq_lens in (
        (max_seq_len, max_seq_len),
        (max(seq_len_q, 127), max(seq_len_q, 65)),
        (max_seq_len, max_seq_len),
    ):
        wrapper.replan(runtime_seq_lens)
        case.seq_lens.copy_(
            torch.tensor(runtime_seq_lens, dtype=torch.int32, device="cuda")
        )
        storage.fill_(sentinel)
        graph.replay()
        torch.cuda.synchronize()
        assert_output_and_guards()


def test_attention_ts_mla_balanced_1cta_reducer_waits_for_pdl_producer():
    """Read split partials only after accepting the producer's PDL signal."""

    source = textwrap.dedent(
        inspect.getsource(ThroughputLatencyMlaDecodeTs.balanced_gmem_reduction_kernel)
    )
    function = ast.parse(source).body[0]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    waits = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "griddepcontrol"
        and any(
            keyword.arg == "kind"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "WAIT"
            for keyword in call.keywords
        )
    ]
    reduction_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "run_balanced_reduction_kernel"
    ]

    assert len(waits) == 1
    assert len(reduction_calls) == 1
    assert waits[0].lineno < reduction_calls[0].lineno


@pytest.mark.parametrize(
    ("num_qo_heads", "expected_kernel", "qkv_dtype"),
    (
        pytest.param(
            16,
            "throughput_latency_1cta",
            torch.float8_e4m3fn,
            id="1cta-fp8",
        ),
        pytest.param(128, "throughput_2cta", torch.bfloat16, id="2cta-bf16"),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_replan_across_graph_replay(
    num_qo_heads: int,
    expected_kernel: str,
    qkv_dtype: torch.dtype,
):
    """Refill balanced descriptors without changing captured addresses."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=num_qo_heads,
        max_seq_len=32768,
        kv_seq_lens=(32768, 257),
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=33200 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    wrapper = _plan_case(case, balanced=True)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == expected_kernel
    assert wrapper._plan_state is not None
    plan = wrapper._plan_state.balanced_plan
    assert plan is not None
    plan_addresses = (
        plan.work_descriptors.data_ptr(),
        plan.partition_offsets.data_ptr(),
        plan.combine_descriptors.data_ptr(),
        plan.num_combine_descriptors.data_ptr(),
    )

    graph_out = torch.empty(
        (2, 1, num_qo_heads, _LATENT_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    _run_case(wrapper, case, out=graph_out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(wrapper, case, out=graph_out, validate=False)
    assert captured is graph_out

    for runtime_seq_lens in ((127, 65), (16385, 129), (32768, 257)):
        wrapper.replan(runtime_seq_lens)
        case.seq_lens.copy_(
            torch.tensor(runtime_seq_lens, dtype=torch.int32, device="cuda")
        )
        kernel_workspace = wrapper._plan_state.workspace_views.kernel_workspace
        if kernel_workspace is not None:
            # Make an early reducer read deterministic and visibly invalid.
            # Every active split producer must replace these bytes before the
            # PDL-dependent reducer consumes its partials.
            kernel_workspace.fill_(-1)
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        replayed = graph_out.clone()

        eager = _run_case(wrapper, case)
        torch.testing.assert_close(replayed, eager, rtol=0, atol=0)
        _assert_case_correct(replayed, case, policy)

    assert plan_addresses == (
        plan.work_descriptors.data_ptr(),
        plan.partition_offsets.data_ptr(),
        plan.combine_descriptors.data_ptr(),
        plan.num_combine_descriptors.data_ptr(),
    )


@pytest.mark.parametrize(
    ("num_qo_heads", "expected_kernel"),
    (
        pytest.param(16, "throughput_latency_1cta", id="1cta"),
        pytest.param(128, "throughput_2cta", id="2cta"),
    ),
)
@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, torch.float8_e4m3fn))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_inactive_slots_across_graph_replay(
    num_qo_heads: int,
    expected_kernel: str,
    qkv_dtype: torch.dtype,
):
    """A compact reducer grid-strides zero-KV slots across graph replays."""

    batch_size = 160
    storage_seq_lens = (4096,) + (257,) * (batch_size - 1)
    initial_seq_lens = (0,) * batch_size
    case = _make_mla_case(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_seq_len=4096,
        kv_seq_lens=storage_seq_lens,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=33600 + num_qo_heads + qkv_dtype.itemsize,
    )
    case.seq_lens.zero_()
    wrapper = _plan_case(case, balanced=True)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == expected_kernel
    assert policy["balanced_reducer_capacity"] < batch_size
    assert wrapper._plan_state is not None

    graph_out = torch.empty(
        (*case.query.shape[:-1], _LATENT_DIM),
        dtype=case.output_dtype,
        device=case.query.device,
    )
    _run_case(wrapper, case, out=graph_out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_case(wrapper, case, out=graph_out, validate=False)

    for runtime_seq_lens in (
        storage_seq_lens,
        (0,) + storage_seq_lens[1:],
        initial_seq_lens,
        (4096,) + (0,) * (batch_size - 1),
        storage_seq_lens,
    ):
        wrapper.replan(runtime_seq_lens)
        case.seq_lens.copy_(
            torch.tensor(runtime_seq_lens, dtype=torch.int32, device="cuda")
        )
        graph_out.fill_(float("nan"))
        wrapper._plan_state.workspace_views.lse.fill_(float("nan"))
        kernel_workspace = wrapper._plan_state.workspace_views.kernel_workspace
        if kernel_workspace is not None:
            kernel_workspace.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()

        inactive = tuple(
            request_idx
            for request_idx, seq_len in enumerate(runtime_seq_lens)
            if seq_len == 0
        )
        active = tuple(
            request_idx
            for request_idx, seq_len in enumerate(runtime_seq_lens)
            if seq_len > 0
        )
        if inactive:
            torch.testing.assert_close(
                graph_out[list(inactive)],
                torch.zeros_like(graph_out[list(inactive)]),
                rtol=0,
                atol=0,
            )
            assert torch.isneginf(
                wrapper._plan_state.workspace_views.lse[list(inactive)]
            ).all()
        if active:
            expected = _mla_reference(
                case,
                num_insts_kv=int(policy["num_insts_kv"]),
                tile_size_kv=int(policy["tile_size_kv"]),
                splits_kv=int(policy["split_kv"]),
                batch_indices=active,
            )
            actual = graph_out[list(active)].float()
            rtol, atol = _mla_tolerances(case.query.dtype)
            if case.query.dtype == torch.float8_e4m3fn:
                # This large-capacity test checks more than ten million FP8
                # outputs per full transition; retain the normal relative
                # tolerance while allowing the observed sub-2e-3 tail.
                atol = max(atol, 2e-3)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_validated_run_requires_current_schedule():
    """Validated runs reject live lengths that were not installed by replan."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=16,
        max_seq_len=512,
        kv_seq_lens=(512, 129),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=33701,
    )
    wrapper = _plan_case(case, balanced=True)

    updated_seq_lens = (385, 65)
    case.seq_lens.copy_(
        torch.tensor(updated_seq_lens, dtype=torch.int32, device="cuda")
    )
    with pytest.raises(ValueError, match=r"live seq_lens.*replan"):
        _run_case(wrapper, case)

    wrapper.replan(updated_seq_lens)
    output = _run_case(wrapper, case)
    _assert_case_correct(output, case, _policy_dict(wrapper))


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_workspace_size_is_publicly_queryable():
    """The public size query covers balanced scratch but not plan descriptors."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=16,
        max_seq_len=32768,
        kv_seq_lens=(32768, 257),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=33702,
    )
    workspace_size = get_prims_ts_batch_mla_decode_workspace_size(
        2,
        16,
        _LATENT_DIM,
        _ROPE_DIM,
        case.page_size,
        case.max_seq_len,
        max_seq_len_q=1,
        q_dtype=case.query.dtype,
        kv_dtype=case.kv_cache.dtype,
        out_dtype=case.output_dtype,
        mask_type=case.mask_type,
        balanced=True,
        device=case.query.device,
    )
    workspace = torch.empty(workspace_size, dtype=torch.int8, device="cuda")
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan_balanced(
        case.query.device,
        2,
        16,
        _LATENT_DIM,
        _ROPE_DIM,
        case.page_size,
        case.max_seq_len,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=case.query.dtype,
        kv_data_type=case.kv_cache.dtype,
        o_data_type=case.output_dtype,
        seq_lens=(32768, 257),
        mask_type=case.mask_type,
        workspace_buffer=workspace,
    )

    assert wrapper._plan_state is not None
    state = wrapper._plan_state
    assert workspace_size == state.workspace_layout.total_bytes
    assert state.workspace_layout.lse.byte_size == 2 * 16 * 4
    assert state.balanced_plan is not None
    plan_info = wrapper.plan_info
    assert dict(plan_info["policy"]) == dict(state.policy)
    assert plan_info["balanced"] is True
    assert plan_info["cost_model"]["model_id"] == B200_BALANCED_COST_MODEL_ID
    assert plan_info["cost_model"]["bucket"] == (state.balanced_plan.last_cost_bucket)
    assert plan_info["cost_model"]["parameters"] == state.balanced_plan.cost
    workspace_end = workspace.data_ptr() + workspace.numel()
    for descriptor in (
        state.balanced_plan.work_descriptors,
        state.balanced_plan.partition_offsets,
        state.balanced_plan.combine_descriptors,
        state.balanced_plan.num_combine_descriptors,
    ):
        descriptor_begin = descriptor.data_ptr()
        descriptor_end = (
            descriptor_begin + descriptor.numel() * descriptor.element_size()
        )
        assert (
            descriptor_end <= workspace.data_ptr() or descriptor_begin >= workspace_end
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_auto_gate_runs_balanced_and_standard_plans():
    """Exercise both production-gate decisions through the public wrapper."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=128,
        max_seq_len=8192,
        kv_seq_lens=(8192, 512),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=33703,
    )
    wrapper = BatchMLADecodePagedTSWrapper()
    plan_kwargs = {
        "device": case.query.device,
        "batch_size": 2,
        "num_heads": 128,
        "kv_lora_rank": _LATENT_DIM,
        "qk_rope_head_dim": _ROPE_DIM,
        "page_size": case.page_size,
        "max_kv_len": case.max_seq_len,
        "max_seq_len_q": 1,
        "packed_query": False,
        "q_data_type": case.query.dtype,
        "kv_data_type": case.kv_cache.dtype,
        "o_data_type": case.output_dtype,
        "seq_lens": tuple(int(value) for value in case.seq_lens.cpu()),
        "mask_type": case.mask_type,
    }

    assert wrapper.plan_auto(
        **plan_kwargs,
        expected_mean_seq_len=4096,
        expected_max_seq_len=8192,
    )
    assert wrapper.plan_info["balanced"] is True
    balanced_output = _run_case(wrapper, case)
    _assert_case_correct(balanced_output, case, _policy_dict(wrapper))

    assert not wrapper.plan_auto(
        **plan_kwargs,
        expected_mean_seq_len=8192,
        expected_max_seq_len=8192,
    )
    assert wrapper.plan_info["balanced"] is False
    standard_output = _run_case(wrapper, case)
    _assert_case_correct(standard_output, case, _policy_dict(wrapper))


def _run_standalone(
    case,
    *,
    qo_indptr: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
    out: torch.Tensor | None = None,
):
    """Run the caller-workspace public entry point for wrapper parity."""

    packed_query = qo_indptr is not None
    resolved_max_seq_len_q = max_seq_len_q if packed_query else int(case.query.shape[1])
    if resolved_max_seq_len_q is None:
        raise ValueError("packed standalone coverage requires max_seq_len_q")
    num_heads = int(case.query.shape[1] if packed_query else case.query.shape[2])
    workspace_size = get_prims_ts_batch_mla_decode_workspace_size(
        case.block_tables.shape[0],
        num_heads,
        _LATENT_DIM,
        _ROPE_DIM,
        case.page_size,
        case.max_seq_len,
        max_seq_len_q=resolved_max_seq_len_q,
        q_dtype=case.query.dtype,
        kv_dtype=case.kv_cache.dtype,
        out_dtype=case.output_dtype,
        mask_type=case.mask_type,
        device=case.query.device,
    )
    workspace = torch.empty(
        workspace_size,
        dtype=torch.int8,
        device=case.query.device,
    )
    output_shape = (
        (case.query.shape[0], num_heads, _LATENT_DIM)
        if packed_query
        else (
            case.query.shape[0],
            resolved_max_seq_len_q,
            num_heads,
            _LATENT_DIM,
        )
    )
    output = (
        torch.empty(
            output_shape,
            dtype=case.output_dtype,
            device=case.query.device,
        )
        if out is None
        else out
    )
    result = prims_ts_batch_mla_decode_with_kv_cache(
        case.query,
        case.kv_cache,
        workspace,
        _LATENT_DIM,
        _ROPE_DIM,
        case.block_tables,
        case.seq_lens,
        case.max_seq_len,
        qo_indptr=qo_indptr,
        max_seq_len_q=resolved_max_seq_len_q,
        out=output,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        mask_type=case.mask_type,
        out_dtype=case.output_dtype,
    )
    assert result is output
    return output


def _policy_dict(wrapper) -> dict[str, object]:
    assert wrapper._plan_state is not None
    policy = dict(wrapper._plan_state.policy)
    assert policy["source"] == "auto"
    return policy


def _assert_auto_policy(
    policy: dict[str, object],
    expected_b200: dict[str, object],
    *,
    device: torch.device,
) -> None:
    """Contract requested feature coverage and portable Blackwell legality."""

    assert policy["source"] == "auto"
    assert policy["kernel"] in ("throughput_latency_1cta", "throughput_2cta")
    assert policy["tile_size_q"] in (8, 16, 32, 64, 128)
    assert policy["tile_size_kv"] == 128
    assert int(policy["num_insts_kv"]) in (1, 2)
    split_kv = int(policy["split_kv"])
    assert split_kv >= 1
    head_dim_per_cta_v = int(policy["head_dim_per_cta_v"])
    num_ctas_per_head_dim = int(policy["num_ctas_per_head_dim"])
    assert head_dim_per_cta_v in (128, 256, 512)
    assert num_ctas_per_head_dim in (1, 2, 4)
    assert head_dim_per_cta_v * num_ctas_per_head_dim == _LATENT_DIM
    use_cluster = bool(policy["use_cluster_reduction"])
    persistent = bool(policy["use_persistent_scheduler"])
    use_clc = bool(policy["use_clc_dynamic_persistent_scheduler"])
    separate_reducer = policy["separate_reducer_impl"]
    assert separate_reducer in ("none", "reference", "parallel")
    if use_cluster:
        assert split_kv > 1
        assert policy["kernel"] == "throughput_latency_1cta"
        assert separate_reducer == "none"
    if separate_reducer != "none":
        assert split_kv > 1
    if split_kv == 1:
        assert separate_reducer == "none"
    if use_clc:
        assert persistent
    if policy["kernel"] == "throughput_2cta":
        assert policy["tile_size_q"] == 128
        assert policy["head_dim_per_cta_v"] == 256
        assert policy["num_ctas_per_head_dim"] == 2
        assert not use_cluster

    if torch.cuda.get_device_capability(device) == (10, 0):
        for key, expected in expected_b200.items():
            assert policy[key] == expected, (key, policy, expected_b200)


def _assert_case_correct(output, case, policy, *, qo_indptr=None):
    expected_shape = (
        (case.query.shape[0], case.query.shape[1], _LATENT_DIM)
        if qo_indptr is not None
        else (
            case.query.shape[0],
            case.query.shape[1],
            case.query.shape[2],
            _LATENT_DIM,
        )
    )
    assert output.shape == expected_shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()

    batch_size = case.block_tables.shape[0]
    if qo_indptr is not None or batch_size <= 8:
        batch_indices = tuple(range(batch_size))
    else:
        batch_indices = (0, 1, batch_size // 2, batch_size - 1)
    expected = _mla_reference(
        case,
        num_insts_kv=int(policy["num_insts_kv"]),
        tile_size_kv=int(policy["tile_size_kv"]),
        splits_kv=int(policy["split_kv"]),
        batch_indices=batch_indices,
        qo_indptr=qo_indptr,
    )
    actual = (
        output.float() if qo_indptr is not None else output[list(batch_indices)].float()
    )
    rtol, atol = _mla_tolerances(case.query.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    if expected.numel() == 0:
        return
    relative_l2 = torch.linalg.vector_norm(
        actual - expected
    ) / torch.linalg.vector_norm(expected)
    assert float(relative_l2) <= (
        0.1 if case.query.dtype == torch.float8_e4m3fn else 0.02
    )


def _exercise_public_paths(
    wrapper,
    case,
    policy,
    *,
    exercise_all_paths: bool,
    qo_indptr: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
):
    """Always check eager; reserve standalone/graph parity for anchor rows."""

    eager = _run_case(wrapper, case, qo_indptr=qo_indptr)
    _assert_case_correct(eager, case, policy, qo_indptr=qo_indptr)
    if not exercise_all_paths:
        return eager

    standalone = _run_standalone(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    _assert_case_correct(standalone, case, policy, qo_indptr=qo_indptr)
    torch.testing.assert_close(standalone, eager, rtol=0, atol=0)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(
            wrapper,
            case,
            qo_indptr=qo_indptr,
            out=graph_out,
            validate=False,
        )
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(graph_out, case, policy, qo_indptr=qo_indptr)
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)
    _CUDA_GRAPH_ANCHOR_OWNERS.append((graph, graph_out, wrapper, case))
    return eager


def _exercise_auto_mla_case(
    case: _MLACase,
    *,
    expected_b200: dict[str, object] | None = None,
    qo_indptr: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
) -> dict[str, object]:
    """Plan automatically and validate one eager public-interface launch."""

    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {} if expected_b200 is None else expected_b200,
        device=case.query.device,
    )
    _exercise_public_paths(
        wrapper,
        case,
        policy,
        exercise_all_paths=False,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    return policy


@torch.no_grad()
def _apply_mla_correction_pattern(case, pattern: str):
    query = case.query
    query.zero_()
    logical_rows = torch.arange(
        query.shape[1] * query.shape[2], device=query.device
    ).view(query.shape[1], query.shape[2])
    if pattern == "identity":
        signs = torch.ones_like(logical_rows)
    elif pattern == "mixed":
        signs = torch.where(logical_rows.remainder(2) == 0, 1, -1)
    else:
        raise ValueError(f"unsupported correction pattern {pattern!r}")
    magnitude = 128 if query.dtype == torch.float8_e4m3fn else 1
    query[..., 0] = (magnitude * signs).to(query.dtype).unsqueeze(0)

    page_size = case.page_size
    cache_pages = case.kv_cache[:, 0] if case.kv_cache.ndim == 4 else case.kv_cache
    for batch_idx, seq_len in enumerate(case.seq_lens.tolist()):
        page_count = (int(seq_len) + page_size - 1) // page_size
        page_ids = case.block_tables[batch_idx, :page_count].to(torch.long)
        logical_tokens = torch.arange(page_count * page_size, device=query.device)
        stored_k = (32 - logical_tokens // 128).clamp_min(0).to(case.kv_cache.dtype)
        cache_pages[page_ids, :, 0] = stored_k.view(page_count, page_size)
    return case


@pytest.mark.parametrize("compact_cache", (False, True), ids=("rank4", "rank3"))
def test_attention_ts_mla_correction_pattern_uses_planned_page_size(compact_cache):
    """Correction stress data addresses both accepted paged-cache layouts."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=8,
        max_seq_len=257,
        qkv_dtype=torch.bfloat16,
        page_size=16,
        device="cpu",
        seed=20260806,
    )
    if compact_cache:
        case = replace(case, kv_cache=case.kv_cache[:, 0])
    _apply_mla_correction_pattern(case, "identity")

    page_count = (case.max_seq_len + case.page_size - 1) // case.page_size
    page_ids = case.block_tables[0, :page_count].long()
    cache_pages = case.kv_cache[:, 0] if case.kv_cache.ndim == 4 else case.kv_cache
    actual = cache_pages[page_ids, :, 0].reshape(-1)
    logical_tokens = torch.arange(page_count * case.page_size)
    expected = (32 - logical_tokens // 128).clamp_min(0).to(actual.dtype)
    torch.testing.assert_close(actual, expected)


@torch.no_grad()
def _apply_mla_tail_markers(case):
    """Force dense SQ>1 to see tail tokens hidden from early causal rows."""

    if case.query.shape[1] <= 1:
        raise ValueError("tail-marker coverage requires SQ>1 input")
    case.query.zero_()
    case.query[..., 0] = 128 if case.query.dtype == _FP8 else 1
    case.kv_cache.zero_()
    for batch_idx, seq_len in enumerate(case.seq_lens.tolist()):
        page_count = (int(seq_len) + case.page_size - 1) // case.page_size
        page_ids = case.block_tables[batch_idx, :page_count].long()
        for tail_idx in range(case.query.shape[1] - 1):
            logical_token = int(seq_len) - case.query.shape[1] + 1 + tail_idx
            page_id = page_ids[logical_token // case.page_size]
            page_offset = logical_token % case.page_size
            # MLA's compressed latent acts as both K and V. The large positive
            # value makes visibility of these tail tokens numerically decisive.
            case.kv_cache[page_id, 0, page_offset, 0] = 80
    return case


def _make_clc_work_queue(cfg) -> MlaWorkQueue:
    """Construct the 2CTA CLC queue used to inspect skip-path scheduling."""

    return MlaWorkQueue(
        tile_sched_params=None,
        cfg=cfg,
        static_split_kv=1,
        static_seq_len_k=128,
        logical_num_heads_q=128,
        logical_seq_len_q=1,
        static_problem_shape_b=1,
        static_problem_shape_s=1,
        use_clc_dynamic=True,
        name="mla_work_queue",
        tile_scheduler_config=TileSchedulerConfig(
            TileSchedulerType.ClcDynamicPersistent,
            None,
            None,
        ),
        pipeline_config=PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=2,
            num_bytes=16,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                cfg.threads_per_cta * cfg.num_mma_ctas - 2 * cfg.threads_per_warp,
            ),
            cta_layout_vmnk=(cfg.num_mma_ctas, 1, 1, 1),
            producer_signaling_threads=SignalingThreads.CtaLeader,
            consumer_signaling_threads=SignalingThreads.All,
        ),
    )


def _schedule_slot(entry):
    resource, schedule_stage, call_id, _ = entry
    return id(resource), schedule_stage, call_id


def _empty_mla_runtime() -> mla_decode_module._MLARuntime:
    return mla_decode_module._MLARuntime(
        query=torch.empty(8),
        normalized_cache=torch.empty(8),
        out=torch.empty(8),
        num_physical_pages=1,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


def test_attention_ts_mla_public_surfaces_hide_internal_tuning_policy():
    """Keep kernel policy automatic and the public signatures explicit."""

    surfaces = (
        BatchMLADecodePagedTSWrapper.__init__,
        BatchMLADecodePagedTSWrapper.plan,
        BatchMLADecodePagedTSWrapper.plan_auto,
        BatchMLADecodePagedTSWrapper.run,
        batch_mla_decode_with_paged_kv_cache,
        get_prims_ts_batch_mla_decode_workspace_size,
        prims_ts_batch_mla_decode_with_kv_cache,
    )
    violations = []
    for surface in surfaces:
        for parameter in inspect.signature(surface).parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                violations.append(f"{surface.__qualname__}.**{parameter.name}")
                continue
            tokens = tuple(parameter.name.split("_"))
            has_forbidden_token = any(
                token.startswith(prefix)
                for token in tokens
                for prefix in _INTERNAL_TUNING_TOKEN_PREFIXES
            )
            has_forbidden_sequence = any(
                tokens[index : index + len(sequence)] == sequence
                for sequence in _INTERNAL_TUNING_TOKEN_SEQUENCES
                for index in range(len(tokens) - len(sequence) + 1)
            )
            if (
                parameter.name in _MLA_INTERNAL_TUNING_PARAMETERS
                or has_forbidden_token
                or has_forbidden_sequence
            ):
                violations.append(f"{surface.__qualname__}.{parameter.name}")

    assert violations == []


def test_attention_ts_mla_plan_info_is_read_only_and_requires_plan():
    wrapper = BatchMLADecodePagedTSWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        _ = wrapper.plan_info

    model_info = MappingProxyType(
        {
            "model_id": "test-model",
            "source": "calibrated-registry",
            "bucket": "sparse",
        }
    )
    wrapper._plan_state = SimpleNamespace(
        policy=(("source", "auto"), ("kernel", "throughput_2cta")),
        balanced_plan=SimpleNamespace(cost_model_info=model_info),
    )
    info = wrapper.plan_info

    assert info["balanced"] is True
    assert info["policy"]["kernel"] == "throughput_2cta"
    assert info["cost_model"] == model_info
    with pytest.raises(TypeError):
        info["balanced"] = False
    with pytest.raises(TypeError):
        info["policy"]["kernel"] = "throughput_latency_1cta"
    with pytest.raises(TypeError):
        info["cost_model"]["bucket"] = "dense_large"


def test_attention_ts_mla_standard_plan_info_has_no_cost_model():
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper._plan_state = SimpleNamespace(
        policy=(("source", "auto"), ("kernel", "throughput_2cta")),
        balanced_plan=None,
    )

    info = wrapper.plan_info

    assert info["balanced"] is False
    assert info["cost_model"] is None


@pytest.mark.parametrize("use_balanced", (False, True))
def test_attention_ts_mla_auto_plan_installs_gate_selection(monkeypatch, use_balanced):
    wrapper = BatchMLADecodePagedTSWrapper()
    calls = []

    monkeypatch.setattr(
        mla_decode_module,
        "should_use_prims_ts_balanced_mla",
        lambda **_kwargs: use_balanced,
    )

    def record_plan(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(wrapper, "_plan", record_plan)
    seq_lens = (8192, 4096)

    selected = wrapper.plan_auto(
        "cuda:0",
        2,
        128,
        _LATENT_DIM,
        _ROPE_DIM,
        _DEFAULT_PAGE_SIZE,
        8192,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        seq_lens=seq_lens,
        expected_mean_seq_len=4096,
        expected_max_seq_len=8192,
    )

    assert selected is use_balanced
    assert len(calls) == 1
    assert calls[0][1]["balanced"] is use_balanced
    assert calls[0][1]["balanced_seq_lens"] == (seq_lens if use_balanced else None)


def test_attention_ts_mla_balanced_plan_fails_calibration_before_policy_or_allocation(
    monkeypatch,
):
    import flashinfer.attention.prims_ts._balanced_plan as balanced_plan_module

    events = []

    def reject_calibration(_device):
        events.append("calibration")
        raise NotImplementedError("run calibration benchmark")

    def unexpected_policy(*_args, **_kwargs):
        events.append("policy")
        raise AssertionError("policy resolution must not run")

    monkeypatch.setattr(
        mla_decode_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cuda:7"), 7),
    )
    monkeypatch.setattr(
        balanced_plan_module,
        "require_balanced_mla_calibration",
        reject_calibration,
    )
    monkeypatch.setattr(
        mla_decode_module, "_resolve_mla_decode_launch_spec", unexpected_policy
    )

    wrapper = BatchMLADecodePagedTSWrapper()
    with pytest.raises(NotImplementedError, match="run calibration benchmark"):
        wrapper.plan_balanced(
            "cuda:7",
            1,
            8,
            _LATENT_DIM,
            _ROPE_DIM,
            _DEFAULT_PAGE_SIZE,
            128,
            max_seq_len_q=1,
            packed_query=False,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            seq_lens=(128,),
        )

    assert events == ["calibration"]


def test_attention_ts_mla_wrapper_uses_compile_oriented_contract():
    """Keep request metadata out of construction and static planning."""

    init_parameters = inspect.signature(
        BatchMLADecodePagedTSWrapper.__init__
    ).parameters
    assert tuple(init_parameters) == ("self",)

    plan_parameters = inspect.signature(BatchMLADecodePagedTSWrapper.plan).parameters
    assert tuple(plan_parameters) == (
        "self",
        "device",
        "batch_size",
        "num_heads",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "page_size",
        "max_kv_len",
        "max_seq_len_q",
        "packed_query",
        "q_data_type",
        "kv_data_type",
        "o_data_type",
        "mask_type",
        "workspace_buffer",
    )
    for name in (
        "device",
        "batch_size",
        "num_heads",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "page_size",
        "max_kv_len",
        "max_seq_len_q",
        "packed_query",
        "q_data_type",
        "kv_data_type",
        "o_data_type",
    ):
        assert plan_parameters[name].default is inspect.Parameter.empty
    for request_metadata in (
        "block_tables",
        "seq_lens",
        "qo_indptr",
    ):
        assert request_metadata not in plan_parameters

    run_parameters = inspect.signature(BatchMLADecodePagedTSWrapper.run).parameters
    assert tuple(run_parameters) == (
        "self",
        "query",
        "kv_cache",
        "block_tables",
        "seq_lens",
        "qo_indptr",
        "bmm1_scale",
        "bmm2_scale",
        "out",
        "validate",
    )
    assert run_parameters["block_tables"].default is inspect.Parameter.empty
    assert run_parameters["seq_lens"].default is inspect.Parameter.empty
    assert run_parameters["qo_indptr"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["validate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["validate"].default is True


def test_attention_ts_mla_plan_publishes_frozen_state_after_workspace_binding(
    monkeypatch,
):
    """Keep plan publication atomic and compilation after workspace checks."""

    events = []
    policy = (("source", "auto"), ("split_kv", 1))
    spec = mla_decode_module._MLADecodeLaunchSpec(
        kernel=object(),
        policy=policy,
        kernel_workspace_bytes=0,
        split_kv=1,
    )
    original_bind = mla_decode_module._bind_mla_workspace

    def resolve_device(device):
        assert device == "cuda:0"
        return torch.device("cpu"), 0

    def resolve_spec(*args):
        events.append("spec")
        return spec

    def validate_workspace(workspace_buffer, *, device, required_bytes):
        assert workspace_buffer.device == device
        assert workspace_buffer.numel() >= required_bytes
        events.append("validate_workspace")

    def bind_workspace(workspace_buffer, layout):
        events.append("bind_workspace")
        return original_bind(workspace_buffer, layout)

    def compile_plan(*args):
        assert events == ["spec", "validate_workspace", "bind_workspace"]
        events.append("compile")
        return lambda *launch_args: None

    monkeypatch.setattr(mla_decode_module, "_resolve_cuda_device", resolve_device)
    monkeypatch.setattr(
        mla_decode_module, "_resolve_mla_decode_launch_spec", resolve_spec
    )
    monkeypatch.setattr(
        mla_decode_module, "_validate_workspace_buffer", validate_workspace
    )
    monkeypatch.setattr(mla_decode_module, "_bind_mla_workspace", bind_workspace)
    monkeypatch.setattr(
        mla_decode_module,
        "_make_mla_decode_compile_spec",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(mla_decode_module, "_get_compiled_mla_decode", compile_plan)

    workspace = torch.empty(256, dtype=torch.int8)
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        "cuda:0",
        1,
        8,
        _LATENT_DIM,
        _ROPE_DIM,
        _DEFAULT_PAGE_SIZE,
        32,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        workspace_buffer=workspace,
    )

    assert events == ["spec", "validate_workspace", "bind_workspace", "compile"]
    assert tuple(vars(wrapper)) == ("_plan_state",)
    state = wrapper._plan_state
    assert state is not None
    with pytest.raises(FrozenInstanceError):
        state.batch_size = 2

    def fail_replan(*args):
        raise RuntimeError("replan failed")

    monkeypatch.setattr(
        mla_decode_module, "_resolve_mla_decode_launch_spec", fail_replan
    )
    with pytest.raises(RuntimeError, match="replan failed"):
        wrapper.plan(
            "cuda:0",
            1,
            8,
            _LATENT_DIM,
            _ROPE_DIM,
            _DEFAULT_PAGE_SIZE,
            32,
            max_seq_len_q=1,
            packed_query=False,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            workspace_buffer=workspace,
        )
    assert wrapper._plan_state is state


def test_attention_ts_mla_decode_bound_wrapper_trace_uses_plan_state():
    """Trace packed-Q shapes from the immutable MLA wrapper plan state."""
    from flashinfer.fi_trace import fi_trace

    wrapper = BatchMLADecodePagedTSWrapper()
    query = torch.empty((5, 8, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((9, 32, 576), dtype=torch.bfloat16)
    block_tables = torch.empty((2, 1), dtype=torch.int32)
    seq_lens = torch.empty((2,), dtype=torch.int32)
    qo_indptr = torch.tensor((0, 2, 5), dtype=torch.int32)
    kwargs = {
        "query": query,
        "kv_cache": kv_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "qo_indptr": qo_indptr,
    }

    with pytest.raises(
        ValueError,
        match=r"requires the live wrapper's plan state.*flashinfer\.fi_trace",
    ):
        wrapper.run.fi_trace(**kwargs)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        fi_trace(wrapper.run, **kwargs)

    wrapper._plan_state = SimpleNamespace(
        packed_query=True,
        mask_type="causal",
        max_seq_len_q=3,
        max_kv_len=64,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
    )
    for required_name in ("block_tables", "seq_lens", "qo_indptr"):
        incomplete_kwargs = dict(kwargs)
        incomplete_kwargs.pop(required_name)
        with pytest.raises(ValueError, match=required_name):
            fi_trace(wrapper.run, **incomplete_kwargs)

    defn = fi_trace(wrapper.run, **kwargs)
    assert defn["name"].startswith("prims_ts_decode_mla_wrapper_packed_q")
    assert defn["inputs"]["query"]["shape"] == [
        "total_q",
        "num_heads",
        "head_dim_qk",
    ]
    assert defn["outputs"]["output"]["shape"] == [
        "total_q",
        "num_heads",
        "kv_lora_rank",
    ]
    assert defn["axes"]["kv_lora_rank"] == {
        "type": "const",
        "value": _LATENT_DIM,
        "description": "Latent K/V rank frozen by plan().",
    }
    assert defn["axes"]["max_seq_len_q"]["value"] == 3
    assert defn["axes"]["max_kv_len"]["value"] == 64
    assert "mask:causal" in defn["tags"]


def test_attention_ts_mla_output_guard_covers_every_live_allocation():
    """Reject output overlap with inputs retained through an MLA launch."""

    for aliased_name in (
        "kv_cache",
        "block_tables",
        "seq_lens",
        "qo_indptr",
        "workspace_buffer",
    ):
        runtime = _empty_mla_runtime()
        inputs = {
            "block_tables": torch.empty(8),
            "seq_lens": torch.empty(8),
            "qo_indptr": torch.empty(8),
            "workspace_buffer": torch.empty(8),
        }
        if aliased_name == "kv_cache":
            runtime = replace(runtime, normalized_cache=runtime.out)
        else:
            inputs[aliased_name] = runtime.out

        with pytest.raises(
            ValueError,
            match=rf"out must not overlap {aliased_name} storage",
        ):
            mla_decode_module._validate_mla_output_aliasing(runtime, **inputs)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_attention_ts_mla_bf16_clc_skipped_tiles_preserve_progress():
    """Skip data work symmetrically while preserving CLC queue progress."""

    cfg = make_mla_decode_config(
        qkv_dtype="bf16",
        o_dtype="bf16",
        is_persistent=True,
    )
    work_queue = _make_clc_work_queue(cfg)
    task_manager, _, _ = build_mla_decode_task_manager(
        cfg,
        domain=1,
        work_queue=work_queue,
        exhaustive_deadlock_race_check=False,
    )

    queue_entries = 0
    throttle_stages = set()
    for task in task_manager.tasks:
        assert task.skip_if is not None
        for entries, skippable_slots in (
            (task.head_schedule_list, task.skippable_head_slots),
            (task.tail_schedule_list, task.skippable_tail_slots),
        ):
            for entry in entries:
                resource, stage, _, _ = entry
                is_skippable = _schedule_slot(entry) in skippable_slots
                if resource is work_queue:
                    # A zero-K tile still has to fetch and retire its CLC work.
                    assert not is_skippable
                    queue_entries += 1
                elif resource.name == "work_throttle":
                    # Both sides of the cross-CTA throttle disappear together.
                    assert is_skippable
                    throttle_stages.add(stage.name)
                elif not is_skippable:
                    # Register initializers must dominate the loop and tail.
                    assert stage.name in {"ProducerAuxWork", "ConsumerAuxWork"}

    assert queue_entries
    assert {
        "ProducerTryAcquire",
        "ProducerAcquire",
        "ProducerCommit",
        "ConsumerWait",
        "ConsumerRelease",
    }.issubset(throttle_stages)


def test_attention_ts_mla_run_requires_plan():
    wrapper = BatchMLADecodePagedTSWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        wrapper.run(None, None, None, None)


def test_attention_ts_mla_run_validate_false_bypasses_explicit_validators(
    monkeypatch,
):
    """Leave validation outside compiled and captured MLA run regions."""

    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper._plan_state = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        num_heads=8,
        max_seq_len_q=1,
        packed_query=False,
        page_size=32,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        workspace_buffer=torch.empty(8),
        kv_lora_rank=_LATENT_DIM,
        split_kv=1,
        workspace_views=object(),
        compiled=object(),
    )
    runtime = _empty_mla_runtime()
    sentinel = torch.empty(1)

    def fail_validation(*args, **kwargs):
        pytest.fail("validate=False must bypass explicit runtime validators")

    def prepare_runtime(*args, **kwargs):
        assert kwargs["validate"] is False
        return runtime

    def launch(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(mla_decode_module, "_prepare_mla_runtime", prepare_runtime)
    monkeypatch.setattr(
        mla_decode_module, "_validate_mla_run_metadata", fail_validation
    )
    monkeypatch.setattr(
        mla_decode_module,
        "_validate_tensor_does_not_overlap_inputs",
        fail_validation,
    )
    monkeypatch.setattr(
        mla_decode_module, "_validate_mla_output_aliasing", fail_validation
    )
    monkeypatch.setattr(mla_decode_module, "_launch_mla_decode", launch)

    tensor = torch.empty(1)
    assert wrapper.run(tensor, tensor, tensor, tensor, validate=False) is sentinel
    with pytest.raises(TypeError, match="validate must be a bool"):
        wrapper.run(tensor, tensor, tensor, tensor, validate=0)


def test_attention_ts_mla_explicit_split_preserves_explicit_profile():
    """Resolve compatible explicit profile/split pairs and reject mismatches."""

    select_kwargs = {
        "requested_policy": "throughput_latency_1cta",
        "batch_size": 1,
        "num_heads": 8,
        "seq_len_q": 1,
        "seq_len_k": 4096,
        "latent_dim": _LATENT_DIM,
        "rope_dim": _ROPE_DIM,
        "page_size": _DEFAULT_PAGE_SIZE,
        "dtype": "bf16",
        "out_dtype": "bf16",
        "throughput_latency_tile_size_q": 8,
        "max_active_clusters": 148,
        "throughput_latency_split_kv": 4,
    }
    profile_name = "h8_splitkv4_hdim128"
    decision = select_mla_ts_kernel(
        **select_kwargs,
        throughput_latency_profile=profile_name,
    )
    assert decision.profile_name == profile_name
    assert decision.config is not None
    assert decision.config.num_ctas_per_seq_kv == 4

    with pytest.raises(ValueError, match=r"profile 'h8_static' is not valid"):
        select_mla_ts_kernel(
            **select_kwargs,
            throughput_latency_profile="h8_static",
        )


def test_attention_ts_mla_balanced_keeps_fp8_serializes_q_work_tiles():
    """Keep the load task behind the previous balanced q64 MMA tail."""

    config_kwargs = {
        "batch_size": 2,
        "num_heads_q": 64,
        "seq_len_q": 1,
        "seq_len_kv": 32768,
        "qkv_dtype": "e4m3",
        "profile": "h64_keeps_mma_ab",
        "max_active_clusters": 148,
    }
    direct = make_throughput_latency_mla_config(**config_kwargs)
    balanced = make_throughput_latency_mla_config(
        **config_kwargs,
        use_balanced_scheduler=True,
        balanced_descriptor_capacity=2,
        balanced_partial_capacity=2,
    )

    assert direct.q_stages == 2
    assert balanced.q_stages == 1
    # Preserve the long-descriptor K/V pipeline; Q is the per-work-tile fence.
    assert balanced.kv_stages == direct.kv_stages == 8


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_balanced_q64_fp8_mixed_deep_queue_is_stable(
    monkeypatch,
):
    """Exercise the mixed B256 schedule that exposed cross-work-tile state."""

    from flashinfer.attention.prims_ts.kernels.mla_decode import kernel_policy

    device = torch.device("cuda")
    batch_size = 256
    num_heads = 64
    long_kv_len = 50048
    short_kv_len = 128
    seq_lens_cpu = torch.tensor(
        (long_kv_len,) * (batch_size // 2) + (short_kv_len,) * (batch_size // 2),
        dtype=torch.int32,
    )
    pages_per_row = math.ceil(long_kv_len / _DEFAULT_PAGE_SIZE)
    # Reuse a deterministic physical-page pool to keep this exact scheduling
    # reproducer small enough for routine GPU acceptance runs.
    num_physical_pages = 8192
    block_tables = (
        torch.arange(
            batch_size * pages_per_row,
            dtype=torch.int32,
            device=device,
        ).reshape(batch_size, pages_per_row)
        % num_physical_pages
    )
    generator = torch.Generator(device=device).manual_seed(33864)
    q_scale, kv_scale = 0.0625, 0.125
    query = _stored(
        0.2
        * torch.randn(
            batch_size,
            1,
            num_heads,
            _QK_DIM,
            dtype=torch.bfloat16,
            generator=generator,
            device=device,
        ),
        _FP8,
        q_scale,
    )
    kv_cache = _stored(
        0.2
        * torch.randn(
            num_physical_pages,
            1,
            _DEFAULT_PAGE_SIZE,
            _QK_DIM,
            dtype=torch.bfloat16,
            generator=generator,
            device=device,
        ),
        _FP8,
        kv_scale,
    )
    case = _MLACase(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_cpu.to(device),
        max_seq_len=long_kv_len,
        page_size=_DEFAULT_PAGE_SIZE,
        output_dtype=torch.bfloat16,
        mask_type="causal",
        bmm1_scale=q_scale * kv_scale / math.sqrt(_QK_DIM),
        bmm2_scale=kv_scale,
    )

    selected_family = "throughput_latency_1cta"

    def force_family(*args, **kwargs):
        del args, kwargs
        return selected_family, "forced-deep-queue-test"

    monkeypatch.setattr(kernel_policy, "resolve_mla_kernel_policy", force_family)
    mla_decode_module._resolve_mla_decode_launch_spec.cache_clear()
    try:
        one_cta = _plan_case(case, balanced=True)
        assert one_cta._plan_state is not None
        one_plan = one_cta._plan_state.balanced_plan
        assert one_plan is not None
        partition_offsets = one_plan.partition_offsets.cpu()
        partition_depths = partition_offsets[1:] - partition_offsets[:-1]
        assert one_plan.last_descriptor_count == batch_size
        assert partition_depths.numel() == 148
        assert int(partition_depths.max()) >= 6
        assert int((partition_depths > 1).sum()) >= 20

        selected_family = "throughput_2cta"
        mla_decode_module._resolve_mla_decode_launch_spec.cache_clear()
        two_cta = _plan_case(case, balanced=True)

        one_cta_output = _run_case(one_cta, case)
        two_cta_output = _run_case(two_cta, case)
        torch.cuda.synchronize()

        assert torch.isfinite(one_cta_output).all()
        assert torch.isfinite(two_cta_output).all()
        assert (
            float((one_cta_output.float() - two_cta_output.float()).abs().max()) < 2e-3
        )
        one_lse = one_cta._plan_state.workspace_views.lse
        two_lse = two_cta._plan_state.workspace_views.lse
        assert float((one_lse - two_lse).abs().max()) < 1e-4

        for _ in range(4):
            repeated = _run_case(one_cta, case)
            torch.cuda.synchronize()
            assert torch.equal(repeated, one_cta_output)
    finally:
        mla_decode_module._resolve_mla_decode_launch_spec.cache_clear()


def test_attention_ts_mla_partial_tail_rejects_cluster_reduction():
    """Keep partial flat-row tails on the predicated standalone reducer."""

    config_kwargs = {
        "batch_size": 1,
        "num_heads_q": 8,
        "seq_len_q": 1,
        "seq_len_kv": 4096,
        "logical_num_heads_q": 5,
        "logical_seq_len_q": 1,
        "tile_size_q": 8,
        "explicit_split_kv": 2,
        "max_active_clusters": 148,
    }
    automatic = make_throughput_latency_mla_config(**config_kwargs)
    assert automatic.use_cluster_reduction == 0

    with pytest.raises(
        ValueError,
        match=r"cluster reduction requires every launched Q tile.*tail_rows=5",
    ):
        make_throughput_latency_mla_config(
            **config_kwargs,
            reduction_mode="cluster",
        )


def test_attention_ts_mla_int32_kv_coordinate_bound():
    """The public K/V bound reserves the largest padded split-KV span."""

    safe_max = 2**31 - 32768
    assert safe_max == mla_decode_module._MLA_MAX_KV_LEN
    assert (
        mla_decode_module._validate_mla_max_kv_len(safe_max, "max_seq_len") == safe_max
    )
    assert safe_max + mla_decode_module._MLA_MAX_KV_COORDINATE_SPAN == 2**31

    with pytest.raises(
        NotImplementedError,
        match=rf"max_seq_len must be <= {safe_max}.*signed int32",
    ):
        mla_decode_module._validate_mla_max_kv_len(safe_max + 1, "max_seq_len")

    assert (
        mla_decode_module._validate_mla_int32_extent(2**31 - 1, "block_tables elements")
        == 2**31 - 1
    )
    with pytest.raises(
        NotImplementedError,
        match=r"kv_cache physical pages must fit in a signed int32",
    ):
        mla_decode_module._validate_mla_int32_extent(2**31, "kv_cache physical pages")

    mla_decode_module._validate_mla_query_head_extent(
        batch_size=1,
        num_heads=1,
        max_seq_len_q=2**31 - 1,
    )
    mla_decode_module._validate_mla_query_head_extent(
        batch_size=1,
        num_heads=1,
        max_seq_len_q=1,
        total_q=0,
    )
    with pytest.raises(
        NotImplementedError,
        match=r"batch_size \* max_seq_len_q \* num_heads must fit",
    ):
        mla_decode_module._validate_mla_query_head_extent(
            batch_size=1,
            num_heads=2,
            max_seq_len_q=2**30,
        )


def test_attention_ts_mla_balanced_1cta_packed_coordinate_bound():
    """Reject only balanced queue coordinates above signed Int32 maximum."""

    mla_decode_module._validate_balanced_1cta_packed_coordinate(
        batch_size=32768,
        num_head_tiles=1,
        descriptor_capacity=65536,
    )
    with pytest.raises(
        NotImplementedError,
        match=r"balanced 1CTA packed work coordinate must fit in a signed int32",
    ):
        mla_decode_module._validate_balanced_1cta_packed_coordinate(
            batch_size=32769,
            num_head_tiles=1,
            descriptor_capacity=65536,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_public_balanced_plan_rejects_packed_coordinate_overflow():
    """Apply the packed-coordinate guard during public policy resolution."""

    with pytest.raises(
        NotImplementedError,
        match=r"balanced 1CTA packed work coordinate must fit in a signed int32",
    ):
        get_prims_ts_batch_mla_decode_workspace_size(
            50_000,
            8,
            _LATENT_DIM,
            _ROPE_DIM,
            _DEFAULT_PAGE_SIZE,
            257,
            max_seq_len_q=1,
            balanced=True,
            device="cuda",
        )
    with pytest.raises(
        NotImplementedError,
        match=r"total_q \* num_heads must fit",
    ):
        mla_decode_module._validate_mla_query_head_extent(
            batch_size=1,
            num_heads=2,
            max_seq_len_q=1,
            total_q=2**30,
        )

    maximal_policy = (
        ("tile_size_kv", 128),
        ("num_insts_kv", 2),
        ("split_kv", 128),
    )
    mla_decode_module._validate_mla_policy_coordinate_span(maximal_policy)
    with pytest.raises(RuntimeError, match=r"span no larger than 32768.*got 33024"):
        mla_decode_module._validate_mla_policy_coordinate_span(
            (*maximal_policy[:-1], ("split_kv", 129))
        )


@pytest.mark.parametrize(
    ("offsets", "expected"),
    (
        pytest.param((0, 8, 8, 9, 12), (8, 12, (8, 0, 1, 3)), id="mixed"),
        pytest.param((0, 0, 0), (0, 0, (0, 0)), id="all-empty"),
    ),
)
def test_attention_ts_mla_packed_q_offsets_allow_zero_lengths(offsets, expected):
    qo_indptr = torch.tensor(offsets, dtype=torch.int32)
    assert (
        mla_decode_module._derive_max_seq_len_q(qo_indptr, batch_size=len(offsets) - 1)
        == expected
    )


def test_attention_ts_mla_packed_q_offsets_reject_decrease():
    qo_indptr = torch.tensor((0, 2, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="qo_indptr must be nondecreasing"):
        mla_decode_module._derive_max_seq_len_q(qo_indptr, batch_size=2)


def test_attention_ts_mla_workspace_rejects_unsafe_int32_kv_bound():
    """Workspace policy resolution rejects unsafe bounds before CUDA work."""

    with pytest.raises(NotImplementedError, match=r"padded MLA K/V coordinates"):
        get_prims_ts_batch_mla_decode_workspace_size(
            1,
            8,
            _LATENT_DIM,
            _ROPE_DIM,
            _DEFAULT_PAGE_SIZE,
            mla_decode_module._MLA_MAX_KV_LEN + 1,
        )

    with pytest.raises(
        NotImplementedError,
        match=r"batch_size \* max_seq_len_q \* num_heads must fit",
    ):
        get_prims_ts_batch_mla_decode_workspace_size(
            1,
            2,
            _LATENT_DIM,
            _ROPE_DIM,
            _DEFAULT_PAGE_SIZE,
            1,
            max_seq_len_q=2**30,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_public_interfaces_reject_output_alias():
    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=8,
        max_seq_len=128,
        qkv_dtype=torch.bfloat16,
        seq_len_q=1,
        device="cuda",
        seed=20260718,
    )
    # O is a compact view over the leading bytes of the 576-element query.
    output_shape = (*case.query.shape[:-1], _LATENT_DIM)
    output_elements = math.prod(output_shape)
    aliased_out = case.query.view(-1)[:output_elements].view(output_shape)
    wrapper = _plan_case(case)

    with pytest.raises(ValueError, match="out must not overlap query storage"):
        _run_case(wrapper, case, out=aliased_out)
    with pytest.raises(ValueError, match="out must not overlap query storage"):
        _run_standalone(case, out=aliased_out)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_block_table_stride_contract():
    """Accept padded rows and reject non-unit inner or overlapping strides."""

    device = torch.device("cuda")
    batch_size, columns = 2, 3
    seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    compact = torch.zeros((batch_size, columns), dtype=torch.int32, device=device)
    padded_storage = torch.zeros(
        1 + batch_size * 2 * columns, dtype=torch.int32, device=device
    )
    padded = padded_storage[1:].view(batch_size, 2, columns)[:, 0]

    assert mla_decode_module._validate_mla_metadata(compact, seq_lens)[1:] == (
        batch_size,
        columns,
    )
    assert mla_decode_module._validate_mla_metadata(padded, seq_lens)[1:] == (
        batch_size,
        columns,
    )

    non_unit_inner = torch.zeros(
        (columns, batch_size), dtype=torch.int32, device=device
    ).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous within each row"):
        mla_decode_module._validate_mla_metadata(non_unit_inner, seq_lens)

    overlapping = torch.zeros(
        columns + batch_size, dtype=torch.int32, device=device
    ).as_strided((batch_size, columns), (columns - 1, 1))
    with pytest.raises(ValueError, match="rows must not overlap"):
        mla_decode_module._validate_mla_metadata(overlapping, seq_lens)


@pytest.mark.parametrize(
    ("seq_lens", "max_kv_len", "message"),
    (
        ((0, 1), 64, "at least one KV token"),
        ((-1, 1), 64, "at least one KV token"),
        ((65, 32), 64, r"longer than max_kv_len \(64\): got 65"),
    ),
    ids=("zero-length", "negative-length", "exceeds-explicit-bound"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_run_rejects_invalid_kv_lengths(
    seq_lens,
    max_kv_len,
    message,
):
    """Validate every runtime MLA K/V length against the static plan bound."""

    device = torch.device("cuda")
    block_tables = torch.zeros((2, 3), dtype=torch.int32, device=device)
    runtime_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    query = torch.empty((2, 1, 8, _QK_DIM), dtype=torch.bfloat16, device=device)
    kv_cache = torch.empty(
        (1, _DEFAULT_PAGE_SIZE, _QK_DIM), dtype=torch.bfloat16, device=device
    )
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        device,
        2,
        8,
        _LATENT_DIM,
        _ROPE_DIM,
        _DEFAULT_PAGE_SIZE,
        max_kv_len,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )
    with pytest.raises(ValueError, match=message):
        wrapper.run(
            query,
            kv_cache,
            block_tables,
            runtime_seq_lens,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_run_rejects_invalid_active_page_id():
    """Validate every active page reference against runtime cache storage."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=8,
        max_seq_len=32,
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=20260901,
    )
    invalid_block_tables = case.block_tables.clone()
    invalid_block_tables[0, 0] = int(case.kv_cache.shape[0])
    wrapper = _plan_case(case)
    with pytest.raises(
        ValueError,
        match=r"block_tables values.*physical K/V cache.*invalid page ID",
    ):
        wrapper.run(
            case.query,
            case.kv_cache,
            invalid_block_tables,
            case.seq_lens,
        )


@pytest.mark.parametrize("packed_q", (False, True), ids=("fixed", "packed"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_rejects_per_request_causal_q_longer_than_kv(
    packed_q: bool,
):
    """Reject a short KV row even when another row satisfies the global bound."""

    device = torch.device("cuda")
    page_size = 16
    block_tables = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([4, 16], dtype=torch.int32, device=device)
    qo_indptr = (
        torch.tensor([0, 5, 6], dtype=torch.int32, device=device) if packed_q else None
    )
    max_seq_len_q = 5 if packed_q else 8
    query = torch.empty(
        (6, 8, _QK_DIM) if packed_q else (2, 8, 8, _QK_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.empty((2, page_size, _QK_DIM), dtype=torch.bfloat16, device=device)
    match = r"request 0 has Q=(5|8) and K/V=4"

    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        device,
        2,
        8,
        _LATENT_DIM,
        _ROPE_DIM,
        page_size,
        16,
        max_seq_len_q=max_seq_len_q,
        packed_query=packed_q,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        mask_type="causal",
    )
    with pytest.raises(ValueError, match=match):
        wrapper.run(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            qo_indptr=qo_indptr,
        )
    with pytest.raises(ValueError, match=match):
        batch_mla_decode_with_paged_kv_cache(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            mask_type="causal",
            max_kv_len=16,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_packed_query_requires_standalone_static_bound():
    """The standalone ABI cannot derive a packed-Q JIT bound on its hot path."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=8,
        max_seq_len=32,
        kv_seq_lens=(32, 31),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=32098,
    )
    case, qo_indptr = _pack_mla_case(case, (1, 1))
    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="max_seq_len_q is required"):
        prims_ts_batch_mla_decode_with_kv_cache(
            case.query,
            case.kv_cache,
            workspace,
            _LATENT_DIM,
            _ROPE_DIM,
            case.block_tables,
            case.seq_lens,
            case.max_seq_len,
            qo_indptr=qo_indptr,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_fp8_reference_uses_p448():
    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=8,
        max_seq_len=128,
        qkv_dtype=torch.float8_e4m3fn,
        device="cuda",
        seed=20260706,
    )
    output = _mla_reference(
        case,
        num_insts_kv=2,
        tile_size_kv=128,
    )
    assert _FP8_PROBABILITY_SCALE == 448.0
    assert torch.isfinite(output).all()
    assert bool((output != 0).any())


@pytest.mark.parametrize(
    ("case_kwargs", "expected_policy", "expected_kernel_workspace_bytes"),
    (
        pytest.param(
            {
                "batch_size": 2,
                "num_qo_heads": 16,
                "max_seq_len": 1025,
                "kv_seq_lens": (1025, 1),
                "qkv_dtype": torch.float8_e4m3fn,
                "seed": 20260808,
            },
            {
                "kernel": "throughput_latency_1cta",
                "use_cluster_reduction": True,
            },
            0,
            id="smem-p-cluster",
        ),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_fp8_fully_masked_split_partials(
    case_kwargs,
    expected_policy,
    expected_kernel_workspace_bytes,
):
    """Fully masked FP8 split tiles publish zero P and neutral partials."""

    case = _make_mla_case(device="cuda", **case_kwargs)
    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    _assert_auto_policy(policy, expected_policy, device=case.query.device)
    assert wrapper._plan_state is not None
    assert (
        wrapper._plan_state.workspace_layout.kernel_workspace.byte_size
        == expected_kernel_workspace_bytes
    )
    _exercise_public_paths(wrapper, case, policy, exercise_all_paths=False)


def test_attention_ts_mla_speculative_mask_oracle_distinguishes_tail_visibility():
    common = dict(
        batch_size=1,
        num_qo_heads=8,
        max_seq_len=128,
        seq_len_q=4,
        qkv_dtype=torch.bfloat16,
        device="cpu",
        seed=32099,
    )
    dense = _apply_mla_tail_markers(_make_mla_case(mask_type="dense", **common))
    causal = _apply_mla_tail_markers(_make_mla_case(mask_type="causal", **common))
    dense_reference = _mla_reference(dense, num_insts_kv=1, tile_size_kv=128)
    causal_reference = _mla_reference(causal, num_insts_kv=1, tile_size_kv=128)
    assert float(dense_reference[0, 0, 0, 0]) > 20
    assert float(causal_reference[0, 0, 0, 0]) == 0
    torch.testing.assert_close(causal_reference[:, -1], dense_reference[:, -1])


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_q_public_parity():
    """Define variable and empty runtime Q lengths with cumulative offsets."""

    q_lens = (1, 0, 8)
    max_seq_len_q = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=16,
        max_seq_len=4097,
        seq_len_q=max_seq_len_q,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        device="cuda",
        seed=32100,
    )
    case, qo_indptr = _pack_mla_case(case, q_lens)
    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.packed_query is True
    assert wrapper._plan_state.max_seq_len_q == max_seq_len_q
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {
            "kernel": "throughput_2cta",
            "split_kv": 17,
            "separate_reducer_impl": "reference",
        },
        device=case.query.device,
    )

    eager = _exercise_public_paths(
        wrapper,
        case,
        policy,
        exercise_all_paths=True,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    one_shot = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        mask_type=case.mask_type,
        max_kv_len=case.max_seq_len,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out_dtype=case.output_dtype,
    )
    _assert_case_correct(one_shot, case, policy, qo_indptr=qo_indptr)
    torch.testing.assert_close(one_shot, eager, rtol=0, atol=0)

    derived_bound_one_shot = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens,
        qo_indptr=qo_indptr,
        mask_type=case.mask_type,
        max_kv_len=case.max_seq_len,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out_dtype=case.output_dtype,
    )
    torch.testing.assert_close(derived_bound_one_shot, eager, rtol=0, atol=0)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_q_clc_empty_tile_progress():
    """Advance CLC bookkeeping when a packed request has no row in a Q tile."""

    q_lens = tuple(0 if batch_idx % 2 == 0 else 2 for batch_idx in range(64))
    max_seq_len_q = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=128,
        max_seq_len=129,
        seq_len_q=max_seq_len_q,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        device="cuda",
        seed=32101,
    )
    case, qo_indptr = _pack_mla_case(case, q_lens)
    policy = _exercise_auto_mla_case(
        case,
        expected_b200={
            "kernel": "throughput_2cta",
            "split_kv": 1,
            "use_persistent_scheduler": True,
            "use_clc_dynamic_persistent_scheduler": True,
        },
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    assert policy["source"] == "auto"


@pytest.mark.parametrize("table_layout", ("compact", "trt-plane0"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_row_strided_page_table_metadata_mutation(
    table_layout: str,
):
    """Reload compact or raw TRT page rows after eager and graph mutations."""

    case = _make_mla_case(
        batch_size=3,
        num_qo_heads=8,
        max_seq_len=129,
        qkv_dtype=torch.bfloat16,
        page_size=32,
        device="cuda",
        seed=32103,
    )
    assert case.block_tables.shape == (3, 5)
    seq_lens_storage = torch.empty(
        case.seq_lens.numel() + 1,
        dtype=torch.int32,
        device=case.seq_lens.device,
    )
    seq_lens = seq_lens_storage[1:]
    seq_lens.copy_(case.seq_lens)
    assert seq_lens.is_contiguous()
    assert seq_lens.data_ptr() % 4 == 0
    assert seq_lens.data_ptr() % 16 != 0
    case = replace(case, seq_lens=seq_lens)
    trt_block_offsets = None
    poison_page = None
    if table_layout == "trt-plane0":
        case, trt_block_offsets, poison_page = _with_trt_plane0_block_tables(case)
        case.kv_cache[poison_page].fill_(float("nan"))
    else:
        assert case.block_tables.is_contiguous()
        assert case.block_tables.stride() == (5, 1)

    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    initial = _run_case(wrapper, case).clone()
    _assert_case_correct(initial, case, policy)
    standalone = _run_standalone(case)
    _assert_case_correct(standalone, case, policy)
    torch.testing.assert_close(standalone, initial, rtol=0, atol=0)

    case.block_tables.copy_(torch.roll(case.block_tables.clone(), 1, dims=0))
    case.seq_lens.copy_(torch.roll(case.seq_lens.clone(), 1))
    eager_mutated = _run_case(wrapper, case).clone()
    _assert_case_correct(eager_mutated, case, policy)
    assert not torch.allclose(
        initial.float(), eager_mutated.float(), rtol=1e-3, atol=1e-3
    )

    graph_out = torch.full_like(eager_mutated, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(wrapper, case, out=graph_out, validate=False)
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, eager_mutated, rtol=0, atol=0)

    case.block_tables.copy_(torch.roll(case.block_tables.clone(), 1, dims=0))
    case.seq_lens.copy_(torch.roll(case.seq_lens.clone(), 1))
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(graph_out, case, policy)
    assert not torch.allclose(
        graph_out.float(), eager_mutated.float(), rtol=1e-3, atol=1e-3
    )

    if trt_block_offsets is not None:
        assert poison_page is not None
        assert bool((trt_block_offsets[:, 1] == poison_page).all().item())


@pytest.mark.parametrize(
    (
        "case_kwargs",
        "expected_policy",
        "correction_pattern",
        "overprovision",
        "exercise_all_paths",
    ),
    _MLA_CASES,
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_compact_variable_k_acceptance(
    case_kwargs,
    expected_policy,
    correction_pattern,
    overprovision,
    exercise_all_paths,
):
    """Exercise pairwise dtype/shape/mask policies with runtime-valued K."""

    case = _make_mla_case(device="cuda", **case_kwargs)
    if overprovision:
        case = replace(
            case,
            block_tables=torch.cat(
                (
                    case.block_tables,
                    torch.zeros(
                        (case.block_tables.shape[0], 17),
                        dtype=torch.int32,
                        device=case.block_tables.device,
                    ),
                ),
                dim=1,
            ),
        )
    if correction_pattern == "tail":
        case = _apply_mla_tail_markers(case)
    elif correction_pattern is not None:
        case = _apply_mla_correction_pattern(case, correction_pattern)

    assert int(case.seq_lens.max().item()) == case.max_seq_len
    assert torch.unique(case.seq_lens).numel() == case.seq_lens.numel()
    assert bool((case.seq_lens[1:] % case.page_size != 0).all().item())

    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    _assert_auto_policy(policy, expected_policy, device=case.query.device)

    _exercise_public_paths(
        wrapper,
        case,
        policy,
        exercise_all_paths=exercise_all_paths,
    )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_2cta_graph_reloads_remapped_page_window():
    """Graph replay reloads a 33-page table through the 2CTA page window."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=128,
        max_seq_len=4097,
        qkv_dtype=torch.bfloat16,
        page_size=128,
        device="cuda",
        seed=32012,
    )
    assert case.block_tables.shape == (1, 33)

    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {"kernel": "throughput_2cta"},
        device=case.query.device,
    )
    assert policy["kernel"] == "throughput_2cta"

    eager = _run_case(wrapper, case).clone()
    _assert_case_correct(eager, case, policy)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(wrapper, case, out=graph_out, validate=False)
    assert captured is graph_out

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)

    table_ptr = case.block_tables.data_ptr()
    table_shape = case.block_tables.shape
    table_stride = case.block_tables.stride()
    original_page_ids = case.block_tables.clone()
    num_physical_pages = case.kv_cache.shape[0]
    remapped_page_ids = (original_page_ids + 1) % num_physical_pages
    case.block_tables.copy_(remapped_page_ids)
    assert case.block_tables.data_ptr() == table_ptr
    assert case.block_tables.shape == table_shape
    assert case.block_tables.stride() == table_stride
    assert not torch.equal(case.block_tables, original_page_ids)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(graph_out, case, policy)
    assert not torch.allclose(graph_out.float(), eager.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_graph_reloads_all_live_metadata():
    """One replay reloads packed Q offsets, K lengths, and every page-table row."""

    q_lens = tuple(0 if batch_idx % 2 == 0 else 2 for batch_idx in range(64))
    replay_q_lens = tuple(reversed(q_lens))
    max_seq_len_q = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=128,
        max_seq_len=129,
        seq_len_q=max_seq_len_q,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        device="cuda",
        seed=32102,
    )
    case, qo_indptr = _pack_mla_case(case, q_lens)
    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {
            "kernel": "throughput_2cta",
            "split_kv": 1,
            "use_persistent_scheduler": True,
            "use_clc_dynamic_persistent_scheduler": True,
        },
        device=case.query.device,
    )

    eager = _run_case(wrapper, case, qo_indptr=qo_indptr).clone()
    _assert_case_correct(eager, case, policy, qo_indptr=qo_indptr)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(
            wrapper,
            case,
            qo_indptr=qo_indptr,
            out=graph_out,
            validate=False,
        )
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)

    replay_offsets = [0]
    for q_len in replay_q_lens:
        replay_offsets.append(replay_offsets[-1] + q_len)
    qo_indptr.copy_(
        torch.tensor(replay_offsets, dtype=torch.int32, device=case.query.device)
    )
    original_seq_lens = case.seq_lens.clone()
    case.seq_lens.copy_(torch.roll(original_seq_lens, 1))
    original_page_ids = case.block_tables.clone()
    case.block_tables.copy_((original_page_ids + 1) % int(case.kv_cache.shape[0]))
    assert not torch.equal(case.seq_lens, original_seq_lens)
    assert not torch.equal(case.block_tables, original_page_ids)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_case_correct(graph_out, case, policy, qo_indptr=qo_indptr)
    assert not torch.allclose(graph_out.float(), eager.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "page_size", (16, 32, 64, 128), ids=lambda value: f"page{value}"
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_page_size_dtype_product(
    page_size: int,
    qkv_dtype: torch.dtype,
):
    """Cross native page sizes with both supported MLA input dtypes."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=16,
        max_seq_len=2049,
        qkv_dtype=qkv_dtype,
        page_size=page_size,
        device="cuda",
        seed=34000 + page_size + (1 if qkv_dtype == _FP8 else 0),
    )
    if page_size in (16, 64):
        case = replace(case, kv_cache=case.kv_cache[:, 0])
    assert case.kv_cache.ndim == (3 if page_size in (16, 64) else 4)
    assert case.kv_cache.shape[-2] == page_size
    assert bool((case.seq_lens % page_size != 0).all().item())

    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    _assert_auto_policy(policy, {}, device=case.query.device)
    _exercise_public_paths(
        wrapper,
        case,
        policy,
        exercise_all_paths=False,
    )


@pytest.mark.parametrize(
    "page_size", (16, 32, 64, 128), ids=lambda value: f"page{value}"
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_2cta_page_size_dtype_product(
    page_size: int,
    qkv_dtype: torch.dtype,
):
    """Cross every native page size and dtype at a 2CTA geometry."""

    case = _make_mla_case(
        batch_size=8,
        num_qo_heads=128,
        max_seq_len=2049,
        qkv_dtype=qkv_dtype,
        page_size=page_size,
        device="cuda",
        seed=35000 + page_size + (1 if qkv_dtype == _FP8 else 0),
    )
    assert case.kv_cache.shape[2] == page_size
    assert bool((case.seq_lens % page_size != 0).all().item())

    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {"kernel": "throughput_2cta"},
        device=case.query.device,
    )
    _exercise_public_paths(
        wrapper,
        case,
        policy,
        exercise_all_paths=False,
    )


@pytest.mark.parametrize(
    "num_qo_heads", (8, 16, 32, 64, 128), ids=lambda value: f"h{value}"
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_head_dtype_product(
    num_qo_heads: int,
    qkv_dtype: torch.dtype,
):
    """Cross every MLA head/tile family with both public input dtypes."""

    case = _make_mla_case(
        batch_size=4,
        num_qo_heads=num_qo_heads,
        max_seq_len=2049,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=36000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    _exercise_auto_mla_case(case)


@pytest.mark.parametrize("num_qo_heads", (12, 96), ids=("h12", "h96"))
@pytest.mark.parametrize("packed_query", (False, True), ids=("fixed", "packed"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_reuses_compiled_topology_across_batch_sizes(
    num_qo_heads: int,
    packed_query: bool,
):
    """One resolved topology accepts different batch extents."""

    wrappers = []
    for batch_size in (3, 4):
        case = _make_mla_case(
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_seq_len=1024,
            qkv_dtype=torch.bfloat16,
            device="cuda",
            seed=39000 + num_qo_heads + batch_size,
        )
        qo_indptr = None
        if packed_query:
            case, qo_indptr = _pack_mla_case(case, (1,) * batch_size)
        wrapper = _plan_case(
            case,
            qo_indptr=qo_indptr,
            max_seq_len_q=1 if packed_query else None,
        )
        policy = _policy_dict(wrapper)
        output = _run_case(wrapper, case, qo_indptr=qo_indptr)
        _assert_case_correct(output, case, policy, qo_indptr=qo_indptr)
        wrappers.append(wrapper)

    first_state = wrappers[0]._plan_state
    second_state = wrappers[1]._plan_state
    assert first_state is not None
    assert second_state is not None
    assert first_state.compiled is second_state.compiled


@pytest.mark.parametrize(
    "num_qo_heads,seq_len_q",
    (
        pytest.param(6, 8, id="h6-sq8"),
        pytest.param(12, 4, id="h12-sq4"),
        pytest.param(24, 2, id="h24-sq2"),
        pytest.param(48, 1, id="h48-sq1"),
    ),
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_non_power_of_two_heads_1cta_auto(
    num_qo_heads: int,
    seq_len_q: int,
    qkv_dtype: torch.dtype,
):
    """Decode equivalent 48-row non-power head shapes with public auto dispatch."""

    case = _make_mla_case(
        batch_size=4,
        num_qo_heads=num_qo_heads,
        max_seq_len=257,
        seq_len_q=seq_len_q,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=40000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    _exercise_auto_mla_case(
        case,
        expected_b200={"kernel": "throughput_latency_1cta"},
    )


@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_non_power_heads_1cta_split_reduction(
    qkv_dtype: torch.dtype,
):
    """Validate split reduction for non-power query heads in the 1CTA family."""

    case = _make_mla_case(
        batch_size=4,
        num_qo_heads=12,
        max_seq_len=32769,
        seq_len_q=1,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=40112 + (1 if qkv_dtype == _FP8 else 0),
    )
    policy = _exercise_auto_mla_case(
        case,
        expected_b200={"kernel": "throughput_latency_1cta"},
    )
    assert int(policy["split_kv"]) > 1


@pytest.mark.parametrize(
    "batch_size,num_qo_heads,seq_len_q,seed",
    (
        pytest.param(160, 6, 8, 40564, id="direct-grid"),
        pytest.param(320, 8, 1, 40508, id="persistent-grid"),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_one_cta_multiwave_progress(
    batch_size: int,
    num_qo_heads: int,
    seq_len_q: int,
    seed: int,
):
    """Execute every request when 1CTA work exceeds one resident wave."""

    case = _make_mla_case(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_seq_len=129,
        seq_len_q=seq_len_q,
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=seed,
    )
    _exercise_auto_mla_case(
        case,
        expected_b200={"kernel": "throughput_latency_1cta"},
    )


@pytest.mark.parametrize(
    "num_qo_heads,seq_len_q",
    (
        pytest.param(96, 4, id="h96-sq4"),
        pytest.param(48, 8, id="h48-sq8"),
        pytest.param(24, 16, id="h24-sq16"),
        pytest.param(12, 32, id="h12-sq32"),
    ),
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_non_power_of_two_heads_2cta_matched_rows(
    num_qo_heads: int,
    seq_len_q: int,
    qkv_dtype: torch.dtype,
):
    """Decode equal-row non-power head shapes through public 2CTA dispatch."""

    case = _make_mla_case(
        batch_size=16,
        num_qo_heads=num_qo_heads,
        max_seq_len=1024,
        seq_len_q=seq_len_q,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=41000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {"kernel": "throughput_2cta"},
        device=case.query.device,
    )
    _exercise_public_paths(wrapper, case, policy, exercise_all_paths=False)


@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_non_power_heads_2cta_split_reduction(
    qkv_dtype: torch.dtype,
):
    """Validate split reduction for a partial M128 query tile."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=96,
        max_seq_len=32769,
        seq_len_q=1,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=41596 + (1 if qkv_dtype == _FP8 else 0),
    )
    policy = _exercise_auto_mla_case(
        case,
        expected_b200={"kernel": "throughput_2cta"},
    )
    assert int(policy["split_kv"]) > 1


@pytest.mark.parametrize("num_qo_heads", (8, 16, 32, 64), ids=lambda value: f"h{value}")
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_sq2_head_dtype_mask_product(
    num_qo_heads: int,
    qkv_dtype: torch.dtype,
    mask_type: str,
):
    """Cross SQ2 flat query rows with dtype and speculative mask semantics."""

    case = _make_mla_case(
        batch_size=3,
        num_qo_heads=num_qo_heads,
        max_seq_len=2049,
        seq_len_q=2,
        qkv_dtype=qkv_dtype,
        mask_type=mask_type,
        device="cuda",
        seed=37000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    case = _apply_mla_tail_markers(case)
    _exercise_auto_mla_case(case)


@pytest.mark.parametrize("page_size", (16, 64), ids=lambda value: f"page{value}")
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_dtype_mask_page_product(
    page_size: int,
    qkv_dtype: torch.dtype,
    mask_type: str,
):
    """Cross uneven packed Q with dtype, mask, page, and cache rank."""

    q_lens = (1, 3, 5)
    max_seq_len_q = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=8,
        max_seq_len=4097,
        seq_len_q=max_seq_len_q,
        qkv_dtype=qkv_dtype,
        mask_type=mask_type,
        page_size=page_size,
        device="cuda",
        seed=38000 + page_size + (1 if qkv_dtype == _FP8 else 0),
    )
    case = _apply_mla_tail_markers(case)
    if page_size == 16:
        case = replace(case, kv_cache=case.kv_cache[:, 0])
    case, qo_indptr = _pack_mla_case(case, q_lens)
    policy = _exercise_auto_mla_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    assert policy["source"] == "auto"


@pytest.mark.parametrize(
    "num_qo_heads,expected_kernel",
    (
        pytest.param(6, "throughput_latency_1cta", id="h6-1cta"),
        pytest.param(96, "throughput_2cta", id="h96-2cta"),
    ),
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_non_power_of_two_heads(
    num_qo_heads: int,
    expected_kernel: str | None,
    qkv_dtype: torch.dtype,
):
    """Mixed empty packed requests predicate flat tiles in both families."""

    q_lens = (8, 1, 0, 3)
    max_seq_len_q = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=num_qo_heads,
        max_seq_len=1024,
        seq_len_q=max_seq_len_q,
        qkv_dtype=qkv_dtype,
        mask_type="causal",
        page_size=64,
        device="cuda",
        seed=43000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    case = _apply_mla_tail_markers(case)
    case, qo_indptr = _pack_mla_case(case, q_lens)
    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {"kernel": expected_kernel} if expected_kernel is not None else {},
        device=case.query.device,
    )
    assert policy["kernel"] in ("throughput_latency_1cta", "throughput_2cta")
    eager = _exercise_public_paths(
        wrapper,
        case,
        policy,
        exercise_all_paths=True,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    one_shot = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        mask_type=case.mask_type,
        max_kv_len=case.max_seq_len,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out_dtype=case.output_dtype,
    )
    _assert_case_correct(one_shot, case, policy, qo_indptr=qo_indptr)
    torch.testing.assert_close(one_shot, eager, rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_qo_heads,expected_kernel",
    (
        pytest.param(6, "throughput_latency_1cta", id="h6-1cta"),
        pytest.param(96, "throughput_2cta", id="h96-2cta"),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_all_empty_packed_query_noop(
    num_qo_heads: int,
    expected_kernel: str,
):
    """An explicitly bounded all-empty batch returns empty public outputs."""

    q_lens = (0, 0, 0, 0)
    max_seq_len_q = 8
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=num_qo_heads,
        max_seq_len=257,
        seq_len_q=max_seq_len_q,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        page_size=64,
        device="cuda",
        seed=44000 + num_qo_heads,
    )
    case, qo_indptr = _pack_mla_case(case, q_lens)

    with pytest.raises(
        ValueError, match="max_seq_len_q is required for an all-empty packed query"
    ):
        batch_mla_decode_with_paged_kv_cache(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens,
            qo_indptr=qo_indptr,
            max_kv_len=case.max_seq_len,
        )

    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {"kernel": expected_kernel},
        device=case.query.device,
    )

    eager_out = torch.empty(
        (0, num_qo_heads, _LATENT_DIM),
        dtype=case.output_dtype,
        device=case.query.device,
    )
    eager = _run_case(wrapper, case, qo_indptr=qo_indptr, out=eager_out)
    assert eager is eager_out
    _assert_case_correct(eager, case, policy, qo_indptr=qo_indptr)

    standalone = _run_standalone(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    _assert_case_correct(standalone, case, policy, qo_indptr=qo_indptr)

    one_shot_out = torch.empty_like(eager)
    one_shot = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        mask_type=case.mask_type,
        max_kv_len=case.max_seq_len,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out=one_shot_out,
        out_dtype=case.output_dtype,
    )
    assert one_shot is one_shot_out
    _assert_case_correct(one_shot, case, policy, qo_indptr=qo_indptr)


@pytest.mark.parametrize(
    ("num_qo_heads", "expected_kernel"),
    (
        pytest.param(16, "throughput_latency_1cta", id="1cta"),
        pytest.param(128, "throughput_2cta", id="2cta"),
    ),
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_runtime_k_pruning_product(
    num_qo_heads: int,
    expected_kernel: str,
    qkv_dtype: torch.dtype,
):
    """Exercise padded split-KV CTAs from one token through the static K bound."""

    runtime_k = (1, 129, 2049, 4097)
    case = _make_mla_case(
        batch_size=len(runtime_k),
        num_qo_heads=num_qo_heads,
        max_seq_len=max(runtime_k),
        kv_seq_lens=runtime_k,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=39000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    assert case.seq_lens.tolist() == list(runtime_k)
    _exercise_auto_mla_case(
        case,
        expected_b200={"kernel": expected_kernel},
    )
