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

from dataclasses import dataclass, replace
import inspect
import functools
import math
from types import SimpleNamespace
from typing import Sequence

import pytest
from tests.test_helpers.parametrize import parametrize_product
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
    MLAAttentionTuning,
    batch_mla_decode_with_paged_kv_cache,
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
from flashinfer.attention.prims_ts.kernels.mla_decode.kernel_policy import (
    select_mla_ts_kernel,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.helpers.query import (
    FlatQueryTileLayout,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
    make_throughput_latency_mla_config,
)
from flashinfer.mla import (
    get_prims_ts_batch_mla_decode_workspace_size,
)
import flashinfer.attention.prims_ts.mla_decode as mla_decode_module


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="PrimTS MLA decode requires SM100, SM103 or SM107",
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
    "num_qo_heads,seq_len_q,tile_size_q,expected",
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
    num_qo_heads, seq_len_q, tile_size_q, expected
):
    layout = FlatQueryTileLayout.for_tile(num_qo_heads, seq_len_q, tile_size_q)
    assert (layout.total_rows, layout.num_tiles, layout.tail_rows) == expected
    assert layout.logical_num_heads_q == num_qo_heads
    assert layout.logical_seq_len_q == seq_len_q
    assert layout.tile_size_q == tile_size_q


@pytest.mark.parametrize(
    "num_qo_heads,seq_len_q,tile_size_q",
    ((0, 1, 128), (64, 0, 128), (64, 1, 0), (-1, 1, 8)),
)
def test_attention_ts_mla_flat_query_tile_layout_rejects_invalid_extent(
    num_qo_heads, seq_len_q, tile_size_q
):
    with pytest.raises(ValueError):
        FlatQueryTileLayout.for_tile(num_qo_heads, seq_len_q, tile_size_q)


@dataclass(frozen=True)
class _MLACase:
    query: torch.Tensor
    kv_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens_kv: torch.Tensor
    max_kv_len: int
    page_size: int
    out_dtype: torch.dtype
    mask_type: str
    softmax_scale: float
    output_scale: float


def _variable_seq_lens(
    batch_size: int, max_kv_len: int, page_size: int
) -> tuple[int, ...]:
    """Return stable, varied runtime KV lengths with non-aligned tails."""

    if batch_size == 1 or max_kv_len <= 1:
        return (max_kv_len,) * batch_size
    lower = max(1, min(max_kv_len - 1, max(page_size, max_kv_len // 2)))
    candidates = tuple(
        length for length in range(lower, max_kv_len) if length % page_size != 0
    )
    if not candidates:
        candidates = tuple(
            length for length in range(1, max_kv_len) if length % page_size != 0
        )
    if not candidates:
        candidates = (max_kv_len - 1,)
    lengths = [max_kv_len]
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
    max_kv_len: int,
    qkv_dtype: torch.dtype,
    seq_len_q: int = 1,
    mask_type: str = "causal",
    page_size: int = _DEFAULT_PAGE_SIZE,
    kv_seq_lens: Sequence[int] | None = None,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> _MLACase:
    """Create a deterministic dense-page MLA problem with shuffled page IDs."""

    seq_lens_kv = (
        _variable_seq_lens(batch_size, max_kv_len, page_size)
        if kv_seq_lens is None
        else tuple(int(length) for length in kv_seq_lens)
    )
    if len(seq_lens_kv) != batch_size:
        raise ValueError("kv_seq_lens must provide one length per request")
    if min(seq_lens_kv) <= 0 or max(seq_lens_kv) != max_kv_len:
        raise ValueError("KV lengths must be positive and include max_kv_len")
    if mask_type == "causal" and min(seq_lens_kv) < seq_len_q:
        raise ValueError("causal KV sequences must be at least as long as Q")
    pages_per_request = tuple(
        (length + page_size - 1) // page_size for length in seq_lens_kv
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
        seq_lens_kv=torch.tensor(seq_lens_kv, dtype=torch.int32, device=device),
        max_kv_len=max_kv_len,
        page_size=page_size,
        out_dtype=torch.bfloat16,
        mask_type=mask_type,
        softmax_scale=q_scale * kv_scale / math.sqrt(128 + _ROPE_DIM),
        output_scale=kv_scale,
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
    seq_len = int(case.seq_lens_kv[batch_idx].item())
    page_count = (seq_len + case.page_size - 1) // case.page_size
    page_ids = case.block_tables[batch_idx, :page_count].long()
    cache_pages = case.kv_cache[:, 0] if case.kv_cache.ndim == 4 else case.kv_cache
    return cache_pages[page_ids].reshape(-1, _QK_DIM)[:seq_len].float()


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
    softmax_scale: float,
    output_scale: float,
    num_insts_kv: int,
    tile_size_kv: int,
    splits_kv: int,
    softmax_rescale_threshold_log2: float = 0.0,
    freeze: bool = True,
) -> torch.Tensor:
    """Model E4M3 P across split-local KV instruction streams.

    A positive ``softmax_rescale_threshold_log2`` T uses the P scale
    ``448 * 2**-T`` and freezes each stream's running max while a tile raises
    it by at most T. ``freeze=False`` keeps that scale but re-anchors on every
    new maximum, which lets tests tell a working freeze from an ignored
    threshold.
    """
    p_scale = _FP8_PROBABILITY_SCALE * 2.0**-softmax_rescale_threshold_log2
    raw_freeze_threshold = (
        softmax_rescale_threshold_log2 / (softmax_scale * math.log2(math.e))
        if softmax_rescale_threshold_log2 > 0.0
        else 0.0
    )

    if num_insts_kv <= 0 or tile_size_kv <= 0 or splits_kv <= 0:
        raise ValueError("KV instruction, tile, and split counts must be positive")

    scores = (
        q_stored[:, :_LATENT_DIM] @ cache_stored[:, :_LATENT_DIM].T
        + q_stored[:, _LATENT_DIM:] @ cache_stored[:, _LATENT_DIM:].T
    )
    num_qo_heads, seq_len = scores.shape
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
        torch.full((num_qo_heads,), -torch.inf, device=scores.device)
        for _ in stream_tiles
    ]
    stream_sum = [torch.zeros(num_qo_heads, device=scores.device) for _ in stream_tiles]
    stream_acc = [
        torch.zeros((num_qo_heads, _LATENT_DIM), device=scores.device)
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
            if (
                stream_valid[stream_idx]
                and softmax_rescale_threshold_log2 > 0.0
                and freeze
            ):
                frozen = (new_max - stream_max[stream_idx]) <= raw_freeze_threshold
                new_max = torch.where(frozen, stream_max[stream_idx], new_max)
            probabilities = (
                torch.exp((tile_scores - new_max[:, None]) * softmax_scale) * p_scale
            )
            quantized_probabilities = probabilities.to(_FP8).float()
            tile_sum = probabilities.sum(dim=-1)
            tile_acc = quantized_probabilities @ cache_stored[begin:end, :_LATENT_DIM]
            if stream_valid[stream_idx]:
                correction = torch.exp(
                    (stream_max[stream_idx] - new_max) * softmax_scale
                )
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
            correction = torch.exp((maximum - final_max) * softmax_scale)
            final_sum += denominator * correction
            final_acc += accumulator * correction[:, None]
    return final_acc / final_sum[:, None] * output_scale


@torch.no_grad()
def _mla_reference(
    case: _MLACase,
    *,
    num_insts_kv: int,
    tile_size_kv: int,
    splits_kv: int = 1,
    batch_indices: Sequence[int] | None = None,
    qo_indptr: torch.Tensor | None = None,
    softmax_rescale_threshold_log2: float = 0.0,
    fp8_freeze: bool = True,
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
                    softmax_scale=case.softmax_scale,
                    output_scale=case.output_scale,
                    num_insts_kv=num_insts_kv,
                    tile_size_kv=tile_size_kv,
                    splits_kv=splits_kv,
                    softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
                    freeze=fp8_freeze,
                )
            else:
                scores = (
                    q_stored[:, :_LATENT_DIM] @ visible_cache[:, :_LATENT_DIM].T
                    + q_stored[:, _LATENT_DIM:] @ visible_cache[:, _LATENT_DIM:].T
                )
                probabilities = torch.softmax(scores * case.softmax_scale, dim=-1)
                output = (
                    probabilities @ visible_cache[:, :_LATENT_DIM] * case.output_scale
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
    max_kv_len: int,
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
        "max_kv_len": max_kv_len,
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
        exercise_all_paths=True,
        id="fp8-1cta-parallel-reduction",
    ),
    _param(
        _case(128, 32, 2048, torch.bfloat16, 32003),
        {"kernel": "throughput_latency_1cta", "split_kv": 1},
        None,
        False,
        exercise_all_paths=True,
        id="bf16-1cta-direct",
    ),
    _param(
        _case(128, 64, 2048, torch.float8_e4m3fn, 32004),
        {"kernel": "throughput_latency_1cta", "split_kv": 1},
        "identity",
        False,
        exercise_all_paths=True,
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
        exercise_all_paths=True,
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
    max_q_len: int | None = None,
    softmax_rescale_threshold_log2: float = 0.0,
    **plan_options,
):
    wrapper = BatchMLADecodePagedTSWrapper()
    num_qo_heads = int(
        case.query.shape[1] if qo_indptr is not None else case.query.shape[2]
    )
    resolved_max_seq_len_q = (
        int(case.query.shape[1])
        if qo_indptr is None and max_q_len is None
        else max_q_len
    )
    if resolved_max_seq_len_q is None:
        raise ValueError("packed plan coverage requires max_q_len")
    options = dict(
        packed_query=qo_indptr is not None,
        q_dtype=case.query.dtype,
        kv_dtype=case.kv_cache.dtype,
        out_dtype=case.out_dtype,
        mask_type=case.mask_type,
        max_q_len=resolved_max_seq_len_q,
        tuning=MLAAttentionTuning(
            softmax_rescale_threshold_log2=softmax_rescale_threshold_log2
        ),
    )
    options.update(plan_options)
    wrapper.plan(
        case.query.device,
        int(case.block_tables.shape[0]),
        num_qo_heads,
        _LATENT_DIM,
        _ROPE_DIM,
        case.page_size,
        case.max_kv_len,
        **options,
    )

    return wrapper


def _run_case(wrapper, case, *, qo_indptr=None, out=None, validate=True, **kwargs):
    return wrapper.run(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        qo_indptr=qo_indptr,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
        out=out,
        validate=validate,
        **kwargs,
    )


def _run_standalone(
    case,
    *,
    qo_indptr: torch.Tensor | None = None,
    max_q_len: int | None = None,
    out: torch.Tensor | None = None,
):
    """Run the caller-workspace public entry point for wrapper parity."""

    packed_query = qo_indptr is not None
    resolved_max_seq_len_q = max_q_len if packed_query else int(case.query.shape[1])
    if resolved_max_seq_len_q is None:
        raise ValueError("packed standalone coverage requires max_q_len")
    num_qo_heads = int(case.query.shape[1] if packed_query else case.query.shape[2])
    workspace_size = get_prims_ts_batch_mla_decode_workspace_size(
        case.block_tables.shape[0],
        num_qo_heads,
        _LATENT_DIM,
        _ROPE_DIM,
        case.page_size,
        case.max_kv_len,
        max_q_len=resolved_max_seq_len_q,
        q_dtype=case.query.dtype,
        kv_dtype=case.kv_cache.dtype,
        out_dtype=case.out_dtype,
        mask_type=case.mask_type,
        device=case.query.device,
    )
    workspace = torch.empty(
        workspace_size,
        dtype=torch.int8,
        device=case.query.device,
    )
    output_shape = (
        (case.query.shape[0], num_qo_heads, _LATENT_DIM)
        if packed_query
        else (
            case.query.shape[0],
            resolved_max_seq_len_q,
            num_qo_heads,
            _LATENT_DIM,
        )
    )
    output = (
        torch.empty(
            output_shape,
            dtype=case.out_dtype,
            device=case.query.device,
        )
        if out is None
        else out
    )
    result = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        workspace_buffer=workspace,
        max_kv_len=case.max_kv_len,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        qo_indptr=qo_indptr,
        max_q_len=resolved_max_seq_len_q,
        out=output,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
        mask_type=case.mask_type,
        out_dtype=case.out_dtype,
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


def _assert_case_correct(
    output, case, policy, *, qo_indptr=None, softmax_rescale_threshold_log2=0.0
):
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
        softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
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
    max_q_len: int | None = None,
):
    """Always check eager; reserve standalone/graph parity for anchor rows."""

    eager = _run_case(wrapper, case, qo_indptr=qo_indptr)
    _assert_case_correct(eager, case, policy, qo_indptr=qo_indptr)
    if not exercise_all_paths:
        return eager

    standalone = _run_standalone(
        case,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
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
    if qo_indptr is None:
        saved_length = case.seq_lens_kv[0].clone()
        saved_pages = case.block_tables[0].clone()
        case.seq_lens_kv[0] = 0
        case.block_tables[0].fill_(-1)
        graph_out.fill_(8)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            graph_out[0].float(),
            torch.full_like(graph_out[0].float(), 8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            graph_out[1:].float(), eager[1:].float(), rtol=0, atol=0
        )
        case.seq_lens_kv[0].copy_(saved_length)
        case.block_tables[0].copy_(saved_pages)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out.float(), eager.float(), rtol=0, atol=0)
    _CUDA_GRAPH_ANCHOR_OWNERS.append((graph, graph_out, wrapper, case))
    return eager


def _exercise_auto_mla_case(
    case: _MLACase,
    *,
    expected_b200: dict[str, object] | None = None,
    qo_indptr: torch.Tensor | None = None,
    max_q_len: int | None = None,
) -> dict[str, object]:
    """Plan automatically and validate one eager public-interface launch."""

    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
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
        max_q_len=max_q_len,
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
    for batch_idx, seq_len in enumerate(case.seq_lens_kv.tolist()):
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
        max_kv_len=257,
        qkv_dtype=torch.bfloat16,
        page_size=16,
        device="cpu",
        seed=20260806,
    )
    if compact_cache:
        case = replace(case, kv_cache=case.kv_cache[:, 0])
    _apply_mla_correction_pattern(case, "identity")

    page_count = (case.max_kv_len + case.page_size - 1) // case.page_size
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
    for batch_idx, seq_len in enumerate(case.seq_lens_kv.tolist()):
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
        softmax_scale=1.0,
        output_scale=1.0,
    )


def test_attention_ts_mla_public_surfaces_hide_internal_tuning_policy():
    """Keep kernel policy automatic and the public signatures explicit."""

    surfaces = (
        BatchMLADecodePagedTSWrapper.__init__,
        BatchMLADecodePagedTSWrapper.plan,
        BatchMLADecodePagedTSWrapper.run,
        batch_mla_decode_with_paged_kv_cache,
        get_prims_ts_batch_mla_decode_workspace_size,
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
    # Unknown constructor options are rejected.
    with pytest.raises(TypeError, match="kernel"):
        BatchMLADecodePagedTSWrapper(kernel="unsupported")


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
        "num_qo_heads",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "page_size",
        "max_kv_len",
        "max_q_len",
        "packed_query",
        "q_dtype",
        "kv_dtype",
        "out_dtype",
        "mask_type",
        "lse_mode",
        "store_softmax_stats",
        "workspace_buffer",
        "validate",
        "tuning",
    )
    for name in (
        "device",
        "packed_query",
        "q_dtype",
        "kv_dtype",
        "out_dtype",
    ):
        assert plan_parameters[name].default is inspect.Parameter.empty
    for name in ("num_qo_heads", "kv_lora_rank", "qk_rope_head_dim", "page_size"):
        assert plan_parameters[name].default is inspect.Parameter.empty
    assert {"batch_size", "max_kv_len", "max_q_len"}.issubset(plan_parameters)
    for request_metadata in (
        "block_tables",
        "seq_lens_kv",
        "qo_indptr",
    ):
        assert request_metadata not in plan_parameters

    run_parameters = inspect.signature(BatchMLADecodePagedTSWrapper.run).parameters
    assert tuple(run_parameters) == (
        "self",
        "query",
        "kv_cache",
        "block_tables",
        "seq_lens_kv",
        "qo_indptr",
        "max_q_len",
        "softmax_scale",
        "output_scale",
        "out",
        "lse",
        "return_lse",
        "softmax_stats",
        "validate",
    )
    assert run_parameters["block_tables"].default is inspect.Parameter.empty
    assert run_parameters["seq_lens_kv"].default is inspect.Parameter.empty
    assert run_parameters["qo_indptr"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["validate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["validate"].default is True
    assert plan_parameters["store_softmax_stats"].default is False
    assert run_parameters["softmax_stats"].default is None


@pytest.mark.parametrize("split_kv", (1, 4, 64))
def test_attention_ts_mla_softmax_stats_workspace_is_optional(split_kv):
    import cutlass
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
        compute_workspace_size as size_2cta,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        compute_workspace_size as size_1cta,
    )

    cfg = SimpleNamespace(
        batch_size=2,
        seq_len_q=3,
        num_heads_q=16,
        head_dim_v=512,
        num_ctas_per_seq_kv=split_kv,
        use_cluster_reduction=0,
    )
    dtype_args = dict(partial_o_dtype=cutlass.BFloat16, lse_dtype=cutlass.Float32)
    baseline = size_1cta(cfg=cfg, **dtype_args)
    enabled = size_1cta(cfg=cfg, store_softmax_stats=True, **dtype_args)
    assert enabled - baseline == (2 * 3 * 16 * split_kv * 8 if split_kv > 1 else 0)
    cfg.use_cluster_reduction = 1
    assert size_1cta(cfg=cfg, store_softmax_stats=True, **dtype_args) == 0
    shape = dict(
        tile_size_q=128, num_q_tiles=3, latent_dim=512, batch_size=2, split_kv=split_kv
    )
    baseline = size_2cta(**shape, **dtype_args)
    enabled = size_2cta(**shape, store_softmax_stats=True, **dtype_args)
    assert enabled - baseline == (2 * 3 * 128 * split_kv * 8 if split_kv > 1 else 0)


def test_attention_ts_mla_decode_wrapper_trace_names_softmax_rescale_threshold_log2():
    """A nonzero plan-time rescale threshold gets its own trace definition."""
    from flashinfer.fi_trace import fi_trace

    wrapper = BatchMLADecodePagedTSWrapper()
    kwargs = {
        "query": torch.empty((2, 1, 8, 576), dtype=torch.bfloat16),
        "kv_cache": torch.empty((9, 32, 576), dtype=torch.bfloat16),
        "block_tables": torch.empty((2, 1), dtype=torch.int32),
        "seq_lens_kv": torch.empty((2,), dtype=torch.int32),
    }
    state = dict(
        packed_query=False,
        mask_type="causal",
        max_q_len=1,
        max_kv_len=64,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
    )
    wrapper._plan_state = SimpleNamespace(**state, softmax_rescale_threshold_log2=0.0)
    exact = fi_trace(wrapper.run, **kwargs)
    wrapper._plan_state = SimpleNamespace(**state, softmax_rescale_threshold_log2=2.5)
    skipped = fi_trace(wrapper.run, **kwargs)
    assert "_rescale_thr" not in exact["name"]
    assert "_rescale_thr2p5" in skipped["name"]
    assert "softmax_rescale_threshold_log2:2.5" in skipped["tags"]
    wrapper._plan_state = SimpleNamespace(
        **state, softmax_rescale_threshold_log2=2.5001
    )
    nearby = fi_trace(wrapper.run, **kwargs)
    assert nearby["name"] != skipped["name"]
    assert not any(
        tag.startswith("softmax_rescale_threshold_log2") for tag in exact["tags"]
    )


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("validate", (False, True))
def test_mla_stats_buffer_presence_is_a_plan_contract(enabled, validate):
    query = torch.empty((1, 2, 8, 576), dtype=torch.bfloat16)
    stats = torch.empty((*query.shape[:-1], 2))
    validator = mla_decode_module._validate_mla_softmax_stats
    validator(query, stats if enabled else None, enabled=enabled, validate=validate)
    with pytest.raises(ValueError, match="exactly when"):
        validator(query, None if enabled else stats, enabled=enabled, validate=validate)


def test_mla_stats_buffer_layout_validation():
    query = torch.empty((1, 2, 8, 576), dtype=torch.bfloat16)
    validator = mla_decode_module._validate_mla_softmax_stats
    for invalid in (
        torch.empty((1, 2, 8, 2), dtype=torch.bfloat16),
        torch.empty((1, 2, 8, 3)),
        torch.empty((1, 2, 8, 4))[..., ::2],
    ):
        with pytest.raises(ValueError):
            validator(query, invalid, enabled=True, validate=True)
    guard = torch.empty((35,), dtype=torch.float32)
    validator(query, guard[1:33].view(1, 2, 8, 2), enabled=True, validate=True)


def test_mla_stats_rejects_deferred_max_before_device_resolution(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported statistics tuning reached device/allocation work")

    monkeypatch.setattr(mla_decode_module, "_resolve_cuda_device", unexpected)
    with pytest.raises(ValueError, match="zero-threshold"):
        BatchMLADecodePagedTSWrapper().plan(
            "cuda",
            1,
            8,
            512,
            64,
            32,
            256,
            max_q_len=1,
            packed_query=False,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            store_softmax_stats=True,
            tuning=MLAAttentionTuning(softmax_rescale_threshold_log2=1.0),
        )


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("with_lse", (False, True))
def test_mla_stats_preserves_persisted_launch_optional_argument_slots(packed, with_lse):
    shape = (6, 8, 576) if packed else (2, 3, 8, 576)
    query = torch.empty(shape, dtype=torch.bfloat16)
    output = torch.empty((*shape[:-1], 512), dtype=query.dtype)
    stats = torch.empty((*shape[:-1], 2))
    lse = torch.empty(shape[:-1]) if with_lse else None
    runtime = SimpleNamespace(
        query=query,
        out=output,
        normalized_cache=torch.empty((8, 32, 576)),
        extra_cache=None,
        sparse_inputs=None,
        softmax_scale=0.25,
        output_scale=0.75,
        output_quant_args=None,
    )
    calls = []
    result = mla_decode_module._launch_mla_decode(
        runtime,
        block_tables=torch.empty((2, 4), dtype=torch.int32),
        seq_lens_kv=torch.empty((2,), dtype=torch.int32),
        qo_indptr=torch.tensor([0, 3, 6], dtype=torch.int32) if packed else None,
        packed_query=packed,
        kv_lora_rank=512,
        split_kv=4,
        workspace=SimpleNamespace(kernel_workspace=None, lse=None),
        compiled=lambda *args: calls.append(args),
        lse=lse,
        max_q_len=3,
        softmax_stats=stats,
    )
    assert result is output and len(calls) == 1 and len(calls[0]) == 17
    args = calls[0]
    assert args[12:16] == (0.25, 0.75, None, 3)
    assert args[16].data_ptr() == stats.data_ptr()
    assert (args[6] is not None) == with_lse
    if with_lse:
        assert args[6].data_ptr() == lse.data_ptr()


def test_mla_stats_participates_in_kernel_and_compile_identity():
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
        MlaDecodeTs,
    )

    kernel = MlaDecodeTs(is_persistent=False)
    launch = mla_decode_module._MLADecodeLaunchSpec(kernel, (), 0, 1)
    kwargs = dict(
        device_index=0,
        num_qo_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=32,
        q_dtype_key="bfloat16",
        out_dtype_key="bfloat16",
        max_q_len=1,
        packed_query=False,
        write_final_lse=False,
    )
    disabled = mla_decode_module._make_mla_decode_compile_spec(launch, **kwargs)
    enabled = mla_decode_module._make_mla_decode_compile_spec(
        launch, **kwargs, store_softmax_stats=True
    )
    assert disabled != enabled
    assert disabled.kernel_signature != enabled.kernel_signature
    assert not enabled.write_final_lse
    assert not kernel.store_softmax_stats


def test_mla_stats_compile_and_reload_use_the_same_optional_argument_order():
    import ast
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
        MlaDecodeTs,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )

    for kernel in (MlaDecodeTs, ThroughputLatencyMlaDecodeTs):
        assert tuple(inspect.signature(kernel.__call__).parameters)[-3:] == (
            "output_quant_args",
            "max_q_len",
            "softmax_stats",
        )
    tree = ast.parse(inspect.getsource(mla_decode_module._get_compiled_mla_decode))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "compile"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert ast.unparse(call.args[-3]) == "output_quant_args"
    assert ast.unparse(call.args[-2]) == "cutlass.Int32(0)"
    assert ast.unparse(call.args[-1]) == "stats_fake"
    assert {kw.arg for kw in call.keywords} == {"options"}


@pytest.mark.parametrize(
    "family,reduction",
    [("1cta", mode) for mode in ("direct", "cluster", "serial", "parallel")]
    + [("2cta", mode) for mode in ("direct", "serial", "parallel")],
)
def test_mla_stats_forced_topologies_retain_main_policy_and_cache_contract(
    monkeypatch, family, reduction
):
    from contextlib import nullcontext
    from flashinfer.attention.prims_ts import _hardware
    from tests.attention.test_attention_ts_softmax_stats import _force_mla_topology

    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(mla_decode_module, "get_max_active_clusters", lambda *_: 64)
    monkeypatch.setattr(_hardware, "get_max_active_clusters", lambda *_: 64)
    monkeypatch.setattr(mla_decode_module, "get_device_sm_count", lambda *_: 128)
    builds = []
    monkeypatch.setattr(
        mla_decode_module,
        "build_and_load_attention_kernel",
        lambda *args, **kwargs: builds.append(kwargs),
    )
    caches = _force_mla_topology(monkeypatch, family, reduction)
    for cache in caches:
        cache.cache_clear()
    try:
        spec = mla_decode_module._resolve_mla_decode_launch_spec(
            0,
            1,
            8 if family == "1cta" else 128,
            512,
            64,
            32,
            8193,
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "causal",
            1,
            0.0,
            True,
        )
        policy = dict(spec.policy)
        assert (policy["split_kv"] == 1) == (reduction == "direct")
        assert policy["use_cluster_reduction"] == (reduction == "cluster")
        assert (
            policy["separate_reducer_impl"]
            == {
                "direct": "none",
                "cluster": "none",
                "serial": "reference",
                "parallel": "parallel",
            }[reduction]
        )
        assert spec.kernel.store_softmax_stats
        mla_decode_module.build_and_load_attention_kernel(
            "mla_decode", "main", None, (), None, extra=("original",)
        )
        assert builds == [
            {"extra": ("original", "stats-topology-test", family, reduction)}
        ]
    finally:
        for cache in caches:
            cache.cache_clear()


def test_attention_ts_mla_decode_bound_wrapper_trace_uses_plan_state():
    """Trace packed-Q shapes from the immutable MLA wrapper plan state."""
    from flashinfer.fi_trace import fi_trace

    wrapper = BatchMLADecodePagedTSWrapper()
    query = torch.empty((5, 8, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((9, 32, 576), dtype=torch.bfloat16)
    block_tables = torch.empty((2, 1), dtype=torch.int32)
    seq_lens_kv = torch.empty((2,), dtype=torch.int32)
    qo_indptr = torch.tensor((0, 2, 5), dtype=torch.int32)
    kwargs = {
        "query": query,
        "kv_cache": kv_cache,
        "block_tables": block_tables,
        "seq_lens_kv": seq_lens_kv,
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
        max_q_len=3,
        max_kv_len=64,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
    )
    for required_name in ("block_tables", "seq_lens_kv", "qo_indptr"):
        incomplete_kwargs = dict(kwargs)
        incomplete_kwargs.pop(required_name)
        with pytest.raises(ValueError, match=required_name):
            fi_trace(wrapper.run, **incomplete_kwargs)

    defn = fi_trace(wrapper.run, **kwargs)
    assert defn["name"].startswith("prims_ts_decode_mla_wrapper_packed_q")
    assert defn["inputs"]["query"]["shape"] == [
        "total_q",
        "num_qo_heads",
        "head_dim_qk",
    ]
    assert defn["outputs"]["output"]["shape"] == [
        "total_q",
        "num_qo_heads",
        "kv_lora_rank",
    ]
    assert defn["axes"]["kv_lora_rank"] == {
        "type": "const",
        "value": _LATENT_DIM,
        "description": "Latent K/V rank frozen by plan().",
    }
    assert defn["axes"]["max_q_len"]["value"] == 3
    assert defn["axes"]["max_kv_len"]["value"] == 64
    assert "mask:causal" in defn["tags"]


@pytest.mark.parametrize(
    "qk_tile,pv_tile",
    [
        ((128, 256), (128, 256)),
        ((128, 64), (128, 256)),
        ((128, 128), (128, 128)),
        ((64, 128), (64, 256)),
    ],
)
def test_attention_ts_mla_nvfp4_rejects_unsupported_tiles(qk_tile, pv_tile):
    """Reject tile overrides incompatible with the fixed NVFP4 DSMEM layout."""
    with pytest.raises(ValueError, match="NVFP4 2CTA requires QK"):
        make_mla_decode_config(
            mma_qk_tiler_mn=qk_tile,
            mma_pv_tiler_mn=pv_tile,
            rope_dim=0,
            page_size=1,
            qkv_dtype="e4m3",
            kv_cache_format="nvfp4",
            extra_kv_cache_format="nvfp4",
        )


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


@pytest.mark.parametrize("store_softmax_stats", (False, True))
def test_attention_ts_mla_run_validate_false_bypasses_explicit_validators(
    monkeypatch,
    store_softmax_stats,
):
    """Leave validation outside compiled and captured MLA run regions."""

    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper._plan_state = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        num_qo_heads=8,
        max_q_len=1,
        packed_query=False,
        page_size=32,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        workspace_buffer=torch.empty(8),
        kv_lora_rank=_LATENT_DIM,
        split_kv=1,
        workspace_views=object(),
        compiled=object(),
        lse_mode="none",
        store_softmax_stats=store_softmax_stats,
    )
    runtime = _empty_mla_runtime()
    sentinel = torch.empty(1)
    stats = torch.empty((1, 1, 8, 2)) if store_softmax_stats else None

    def fail_validation(*args, **kwargs):
        pytest.fail("validate=False must bypass explicit runtime validators")

    def prepare_runtime(*args, **kwargs):
        assert kwargs["validate"] is False
        return runtime

    def launch(*args, **kwargs):
        assert kwargs["softmax_stats"] is stats
        return sentinel

    monkeypatch.setattr(mla_decode_module, "_prepare_mla_runtime", prepare_runtime)
    monkeypatch.setattr(
        mla_decode_module, "_validate_mla_run_metadata", fail_validation
    )
    monkeypatch.setattr(mla_decode_module, "_launch_mla_decode", launch)

    tensor = torch.empty((1, 1, 8, _LATENT_DIM + _ROPE_DIM))
    assert (
        wrapper.run(tensor, tensor, tensor, tensor, validate=False, softmax_stats=stats)
        is sentinel
    )
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


def test_attention_ts_mla_build_advisories_are_opt_in(monkeypatch, capsys):
    """Quiet default builds retain errors and expose advisories in debug mode."""
    import warnings

    from flashinfer.attention.prims_ts.kernels.mla_decode.helpers.diagnostics import (
        mla_build_diagnostics,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
        make_mla_decode_config,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
        build_mla_decode_task_manager,
    )

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with mla_build_diagnostics():
            manager, _, _ = build_mla_decode_task_manager(
                make_mla_decode_config(),
                domain=2,
                exhaustive_deadlock_race_check=False,
            )
            warnings.warn("unrelated build warning", UserWarning, stacklevel=2)
    assert "[head-loop-overlap]" not in capsys.readouterr().out
    assert [str(item.message) for item in emitted] == ["unrelated build warning"]

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "1")
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with mla_build_diagnostics():
            build_mla_decode_task_manager(
                make_mla_decode_config(),
                domain=2,
                exhaustive_deadlock_race_check=False,
            )
    assert "[head-loop-overlap]" in capsys.readouterr().out
    assert any("never routed" in str(item.message) for item in emitted)

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    manager.tasks[0].num_registers = 512
    with pytest.raises(ValueError), mla_build_diagnostics():
        manager.print_and_verify()


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
        mla_decode_module._validate_mla_max_kv_len(safe_max, "max_kv_len") == safe_max
    )
    assert safe_max + mla_decode_module._MLA_LEGACY_KV_COORDINATE_RESERVE == 2**31

    with pytest.raises(
        NotImplementedError,
        match=rf"max_kv_len must be <= {safe_max}.*signed int32",
    ):
        mla_decode_module._validate_mla_max_kv_len(safe_max + 1, "max_kv_len")

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
        num_qo_heads=1,
        max_q_len=2**31 - 1,
    )
    mla_decode_module._validate_mla_query_head_extent(
        batch_size=1,
        num_qo_heads=1,
        max_q_len=1,
        total_q=0,
    )
    with pytest.raises(
        NotImplementedError,
        match=r"batch_size \* max_q_len \* num_qo_heads must fit",
    ):
        mla_decode_module._validate_mla_query_head_extent(
            batch_size=1,
            num_qo_heads=2,
            max_q_len=2**30,
        )
    with pytest.raises(
        NotImplementedError,
        match=r"total_q \* num_qo_heads must fit",
    ):
        mla_decode_module._validate_mla_query_head_extent(
            batch_size=1,
            num_qo_heads=2,
            max_q_len=1,
            total_q=2**30,
        )

    maximal_policy = (
        ("tile_size_kv", 128),
        ("num_insts_kv", 2),
        ("split_kv", 128),
    )
    mla_decode_module._validate_mla_policy_coordinate_span(maximal_policy)
    mla_decode_module._validate_mla_policy_coordinate_span(
        (*maximal_policy[:-1], ("split_kv", 152)), max_kv_len=32768
    )
    with pytest.raises(NotImplementedError, match="padded K/V extent"):
        mla_decode_module._validate_mla_policy_coordinate_span(
            (*maximal_policy[:-1], ("split_kv", 152)), max_kv_len=2**31 - 32768
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


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_block_table_stride_contract():
    """Accept padded rows and reject non-unit inner or overlapping strides."""

    device = torch.device("cuda")
    batch_size, columns = 2, 3
    seq_lens_kv = torch.ones(batch_size, dtype=torch.int32, device=device)
    compact = torch.zeros((batch_size, columns), dtype=torch.int32, device=device)
    padded_storage = torch.zeros(
        1 + batch_size * 2 * columns, dtype=torch.int32, device=device
    )
    padded = padded_storage[1:].view(batch_size, 2, columns)[:, 0]

    assert mla_decode_module._validate_mla_metadata(compact, seq_lens_kv)[1:] == (
        batch_size,
        columns,
    )
    assert mla_decode_module._validate_mla_metadata(padded, seq_lens_kv)[1:] == (
        batch_size,
        columns,
    )

    non_unit_inner = torch.zeros(
        (columns, batch_size), dtype=torch.int32, device=device
    ).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous within each row"):
        mla_decode_module._validate_mla_metadata(non_unit_inner, seq_lens_kv)

    overlapping = torch.zeros(
        columns + batch_size, dtype=torch.int32, device=device
    ).as_strided((batch_size, columns), (columns - 1, 1))
    with pytest.raises(ValueError, match="rows must not overlap"):
        mla_decode_module._validate_mla_metadata(overlapping, seq_lens_kv)


@pytest.mark.parametrize(
    ("seq_lens_kv", "max_kv_len", "message"),
    (
        ((0, 1), 64, None),
        ((-1, 1), 64, None),
        ((65, 32), 64, r"longer than max_kv_len \(64\): got 65"),
    ),
    ids=("zero-length", "negative-length", "exceeds-explicit-bound"),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_run_validates_kv_lengths(
    seq_lens_kv,
    max_kv_len,
    message,
):
    """Inactive rows preserve output; active lengths must fit the plan bound."""

    device = torch.device("cuda")
    block_tables = torch.zeros((2, 3), dtype=torch.int32, device=device)
    runtime_seq_lens_kv = torch.tensor(seq_lens_kv, dtype=torch.int32, device=device)
    query = torch.zeros((2, 1, 8, _QK_DIM), dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(
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
        packed_query=False,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        max_q_len=1,
    )
    if message is not None:
        with pytest.raises(ValueError, match=message):
            wrapper.run(query, kv_cache, block_tables, runtime_seq_lens_kv)
    else:
        block_tables[0].fill_(-1)
        output = torch.full(
            (2, 1, 8, _LATENT_DIM), 8, dtype=torch.bfloat16, device=device
        )
        wrapper.run(query, kv_cache, block_tables, runtime_seq_lens_kv, out=output)
        torch.testing.assert_close(output[0], torch.full_like(output[0], 8))
        torch.testing.assert_close(output[1], torch.zeros_like(output[1]))


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_run_rejects_invalid_active_page_id():
    """Validate every active page reference against runtime cache storage."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=8,
        max_kv_len=32,
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
            case.seq_lens_kv,
        )


@pytest.mark.parametrize("packed_q", (False, True), ids=("fixed", "packed"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_rejects_per_request_causal_q_longer_than_kv(
    packed_q: bool,
):
    """Reject a short KV row even when another row satisfies the global bound."""

    device = torch.device("cuda")
    page_size = 16
    block_tables = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    seq_lens_kv = torch.tensor([4, 16], dtype=torch.int32, device=device)
    qo_indptr = (
        torch.tensor([0, 5, 6], dtype=torch.int32, device=device) if packed_q else None
    )
    max_q_len = 5 if packed_q else 8
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
        packed_query=packed_q,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        mask_type="causal",
        max_q_len=max_q_len,
    )
    with pytest.raises(ValueError, match=match):
        wrapper.run(query, kv_cache, block_tables, seq_lens_kv, qo_indptr=qo_indptr)
    with pytest.raises(ValueError, match=match):
        batch_mla_decode_with_paged_kv_cache(
            query,
            kv_cache,
            block_tables,
            seq_lens_kv,
            qo_indptr=qo_indptr,
            max_q_len=max_q_len,
            mask_type="causal",
            max_kv_len=16,
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("validate", (False, True))
def test_attention_ts_mla_packed_query_requires_explicit_bound(validate):
    """Both validation modes require a caller-provided packed-Q bound."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=8,
        max_kv_len=32,
        kv_seq_lens=(32, 31),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=32098,
    )
    case, qo_indptr = _pack_mla_case(case, (1, 1))
    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="requires max_q_len"):
        batch_mla_decode_with_paged_kv_cache(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            workspace_buffer=workspace,
            max_kv_len=case.max_kv_len,
            kv_lora_rank=_LATENT_DIM,
            qk_rope_head_dim=_ROPE_DIM,
            qo_indptr=qo_indptr,
            validate=validate,
        )


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_fp8_reference_uses_p448():
    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=8,
        max_kv_len=128,
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
                "max_kv_len": 1025,
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
        max_kv_len=128,
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


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_launch_bounds():
    """Inferred and bucketed Q bounds preserve packed MLA numerical output."""
    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=16,
        max_kv_len=257,
        seq_len_q=4,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        device="cuda",
        seed=710646,
    )
    case, offsets = _pack_mla_case(case, (2, 3))
    wrapper = _plan_case(case, qo_indptr=offsets, max_q_len=4)
    policy = _policy_dict(wrapper)
    out = torch.empty_like(case.query[..., :_LATENT_DIM])

    def run(**bounds):
        return wrapper.run(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            qo_indptr=offsets,
            softmax_scale=case.softmax_scale,
            output_scale=case.output_scale,
            out=out,
            **bounds,
        )

    for bounds in ({}, dict(max_q_len=3), dict(max_q_len=4)):
        _assert_case_correct(run(**bounds), case, policy, qo_indptr=offsets)
    for bound, message in ((2, "cover"), (5, "capacity")):
        out.fill_(float("nan"))
        with pytest.raises(ValueError, match=f"max_q_len.*{message}"):
            run(max_q_len=bound)
        assert out.isnan().all()


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_q_public_parity():
    """Define variable and empty runtime Q lengths with cumulative offsets."""

    q_lens = (1, 0, 8)
    max_q_len = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=16,
        max_kv_len=4097,
        seq_len_q=max_q_len,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        device="cuda",
        seed=32100,
    )
    case, qo_indptr = _pack_mla_case(case, q_lens)
    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
    )
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.packed_query is True
    assert wrapper._plan_state.max_q_len == max_q_len
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
        max_q_len=max_q_len,
    )
    one_shot = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
        mask_type=case.mask_type,
        max_kv_len=case.max_kv_len,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
        out_dtype=case.out_dtype,
    )
    _assert_case_correct(one_shot, case, policy, qo_indptr=qo_indptr)
    torch.testing.assert_close(one_shot, eager, rtol=0, atol=0)

    with pytest.raises(ValueError, match="requires max_q_len"):
        batch_mla_decode_with_paged_kv_cache(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            qo_indptr=qo_indptr,
            mask_type=case.mask_type,
            max_kv_len=case.max_kv_len,
            softmax_scale=case.softmax_scale,
            output_scale=case.output_scale,
            out_dtype=case.out_dtype,
        )


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_q_clc_empty_tile_progress():
    """Advance CLC bookkeeping when a packed request has no row in a Q tile."""

    q_lens = tuple(0 if batch_idx % 2 == 0 else 2 for batch_idx in range(64))
    max_q_len = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=128,
        max_kv_len=129,
        seq_len_q=max_q_len,
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
        max_q_len=max_q_len,
    )
    assert policy["source"] == "auto"


@pytest.mark.parametrize("table_layout", ("compact", "trt-plane0"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_row_strided_page_table_metadata_mutation(
    table_layout: str,
):
    """Reload compact or raw TRT page rows after eager and graph mutations."""

    case = _make_mla_case(
        batch_size=3,
        num_qo_heads=8,
        max_kv_len=129,
        qkv_dtype=torch.bfloat16,
        page_size=32,
        device="cuda",
        seed=32103,
    )
    assert case.block_tables.shape == (3, 5)
    seq_lens_storage = torch.empty(
        case.seq_lens_kv.numel() + 1,
        dtype=torch.int32,
        device=case.seq_lens_kv.device,
    )
    seq_lens_kv = seq_lens_storage[1:]
    seq_lens_kv.copy_(case.seq_lens_kv)
    assert seq_lens_kv.is_contiguous()
    assert seq_lens_kv.data_ptr() % 4 == 0
    assert seq_lens_kv.data_ptr() % 16 != 0
    case = replace(case, seq_lens_kv=seq_lens_kv)
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
    case.seq_lens_kv.copy_(torch.roll(case.seq_lens_kv.clone(), 1))
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
    case.seq_lens_kv.copy_(torch.roll(case.seq_lens_kv.clone(), 1))
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


# Regular runs retain public eager/standalone/graph anchors and FP8 2CTA.
# The broader policy/shape sweep remains available through --full.
@parametrize_product(
    {
        "case_kwargs,expected_policy,correction_pattern,overprovision,exercise_all_paths": tuple(
            case.values for case in _MLA_CASES
        )
    },
    regular=tuple(
        (case.values,)
        for case in _MLA_CASES
        if case.values[-1] or case.id == "fp8-2cta-reference-reduction"
    ),
)
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

    assert int(case.seq_lens_kv.max().item()) == case.max_kv_len
    assert torch.unique(case.seq_lens_kv).numel() == case.seq_lens_kv.numel()
    assert bool((case.seq_lens_kv[1:] % case.page_size != 0).all().item())

    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    _assert_auto_policy(policy, expected_policy, device=case.query.device)

    _exercise_public_paths(
        wrapper,
        case,
        policy,
        exercise_all_paths=exercise_all_paths,
    )


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_2cta_graph_reloads_remapped_page_window():
    """Graph replay reloads a 33-page table through the 2CTA page window."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=128,
        max_kv_len=4097,
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


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_graph_reloads_all_live_metadata():
    """One replay reloads packed Q offsets, K lengths, and every page-table row."""

    q_lens = tuple(0 if batch_idx % 2 == 0 else 2 for batch_idx in range(64))
    replay_q_lens = tuple(reversed(q_lens))
    max_q_len = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=128,
        max_kv_len=129,
        seq_len_q=max_q_len,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        device="cuda",
        seed=32102,
    )
    case, qo_indptr = _pack_mla_case(case, q_lens)
    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
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
    original_seq_lens = case.seq_lens_kv.clone()
    case.seq_lens_kv.copy_(torch.roll(original_seq_lens, 1))
    original_page_ids = case.block_tables.clone()
    case.block_tables.copy_((original_page_ids + 1) % int(case.kv_cache.shape[0]))
    assert not torch.equal(case.seq_lens_kv, original_seq_lens)
    assert not torch.equal(case.block_tables, original_page_ids)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_case_correct(graph_out, case, policy, qo_indptr=qo_indptr)
    assert not torch.allclose(graph_out.float(), eager.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "page_size", (4, 8, 16, 32, 64, 128), ids=lambda value: f"page{value}"
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_page_size_dtype_product(
    page_size: int,
    qkv_dtype: torch.dtype,
):
    """Cross native page sizes with both supported MLA input dtypes."""

    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=16,
        max_kv_len=2049,
        qkv_dtype=qkv_dtype,
        page_size=page_size,
        device="cuda",
        seed=34000 + page_size + (1 if qkv_dtype == _FP8 else 0),
    )
    if page_size in (16, 64):
        case = replace(case, kv_cache=case.kv_cache[:, 0])
    assert case.kv_cache.ndim == (3 if page_size in (16, 64) else 4)
    assert case.kv_cache.shape[-2] == page_size
    assert bool((case.seq_lens_kv % page_size != 0).all().item())

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
    ("batch_size", "page_size", "max_kv_len"),
    (
        (2, 16, 32769),
        (128, 16, 2049),
        (128, 128, 4225),
    ),
    ids=("split-page16", "page16", "page128"),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_fp8_2cta_page_window(
    batch_size: int, page_size: int, max_kv_len: int
):
    """FP8 2CTA K/V loaders read page IDs through a prefetched 32-entry window.

    One window covers ``32 / pages_per_tile`` K tiles. Every case gives the
    longest request's work tiles more K tiles than one window, so the loaders
    swap in the prefetched window. The two-request case is long enough to
    split KV, so windows start at each split's first K tile. Page 128 covers
    the two CTAs sharing one page.
    """

    case = _make_mla_case(
        batch_size=batch_size,
        num_qo_heads=128,
        max_kv_len=max_kv_len,
        qkv_dtype=_FP8,
        page_size=page_size,
        device="cuda",
        seed=37000 + page_size + batch_size,
    )
    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_2cta"
    k_tiles = -(-max_kv_len // 128)
    tiles_per_work_tile = -(-k_tiles // int(policy["split_kv"]))
    assert tiles_per_work_tile > 32 // (128 // page_size), policy
    _assert_case_correct(_run_case(wrapper, case), case, policy)


@pytest.mark.parametrize(
    "page_size", (4, 8, 16, 32, 64, 128), ids=lambda value: f"page{value}"
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("bf16", "fp8"),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_2cta_page_size_dtype_product(
    page_size: int,
    qkv_dtype: torch.dtype,
):
    """Cross every native page size and dtype at a 2CTA geometry."""

    case = _make_mla_case(
        batch_size=8,
        num_qo_heads=128,
        max_kv_len=2049,
        qkv_dtype=qkv_dtype,
        page_size=page_size,
        device="cuda",
        seed=35000 + page_size + (1 if qkv_dtype == _FP8 else 0),
    )
    assert case.kv_cache.shape[2] == page_size
    assert bool((case.seq_lens_kv % page_size != 0).all().item())

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
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_head_dtype_product(
    num_qo_heads: int,
    qkv_dtype: torch.dtype,
):
    """Cross every MLA head/tile family with both public input dtypes."""

    case = _make_mla_case(
        batch_size=4,
        num_qo_heads=num_qo_heads,
        max_kv_len=2049,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=36000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    _exercise_auto_mla_case(case)


@parametrize_product(
    {"num_qo_heads": (12, 96), "packed_query": (False, True)},
    regular=((12, False), (96, True)),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_reuses_compiled_topology_across_batch_sizes(
    num_qo_heads: int,
    packed_query: bool,
):
    """One resolved topology accepts different batch, Q and KV extents."""

    wrappers = []
    policies = []
    for batch_size, q_length, kv_length in ((3, 3, 1025), (4, 4, 1151)):
        case = _make_mla_case(
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_kv_len=kv_length,
            seq_len_q=q_length,
            qkv_dtype=torch.bfloat16,
            device="cuda",
            seed=39000 + num_qo_heads + batch_size,
        )
        qo_indptr = None
        if packed_query:
            case, qo_indptr = _pack_mla_case(case, (q_length,) * batch_size)
        wrapper = _plan_case(
            case,
            qo_indptr=qo_indptr,
            max_q_len=q_length if packed_query else None,
        )
        policy = _policy_dict(wrapper)
        output = _run_case(wrapper, case, qo_indptr=qo_indptr)
        _assert_case_correct(output, case, policy, qo_indptr=qo_indptr)
        wrappers.append(wrapper)
        policies.append(policy)

    first_state = wrappers[0]._plan_state
    second_state = wrappers[1]._plan_state
    assert first_state is not None
    assert second_state is not None
    # Split-KV, tile and CTA choices depend on the GPU's SM count, so a device
    # can resolve the two plans to different topologies. Reuse is only promised
    # between plans with the same topology.
    if policies[0] == policies[1] and first_state.split_kv == second_state.split_kv:
        assert first_state.compiled is second_state.compiled


# Keep the multi-query and single-query endpoints; all four equivalent
# flat-row factorizations with both dtypes are exercised under --full.
@parametrize_product(
    {
        "num_qo_heads,seq_len_q": ((6, 8), (12, 4), (24, 2), (48, 1)),
        "qkv_dtype": (torch.bfloat16, _FP8),
    },
    regular=(((6, 8), torch.bfloat16), ((48, 1), _FP8)),
)
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
        max_kv_len=257,
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
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_non_power_heads_1cta_split_reduction(
    qkv_dtype: torch.dtype,
):
    """Validate split reduction for non-power query heads in the 1CTA family."""

    case = _make_mla_case(
        batch_size=4,
        num_qo_heads=12,
        max_kv_len=32769,
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
        max_kv_len=129,
        seq_len_q=seq_len_q,
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=seed,
    )
    _exercise_auto_mla_case(
        case,
        expected_b200={"kernel": "throughput_latency_1cta"},
    )


@parametrize_product(
    {
        "num_qo_heads,seq_len_q": ((96, 4), (48, 8), (24, 16), (12, 32)),
        "qkv_dtype": (torch.bfloat16, _FP8),
    },
    regular=(((96, 4), torch.bfloat16), ((12, 32), _FP8)),
)
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
        max_kv_len=1024,
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
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_non_power_heads_2cta_split_reduction(
    qkv_dtype: torch.dtype,
):
    """Validate split reduction for a partial M128 query tile."""

    case = _make_mla_case(
        batch_size=1,
        num_qo_heads=96,
        max_kv_len=32769,
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


# Cover every head count and each dtype/mask pair without compiling the
# full three-axis product. SQ1 dtype coverage is kept separately above.
@parametrize_product(
    {
        "num_qo_heads": (8, 16, 32, 64),
        "qkv_dtype": (torch.bfloat16, _FP8),
        "mask_type": ("dense", "causal"),
    },
    regular=(
        (8, torch.bfloat16, "dense"),
        (16, _FP8, "causal"),
        (32, torch.bfloat16, "causal"),
        (64, _FP8, "dense"),
    ),
)
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
        max_kv_len=2049,
        seq_len_q=2,
        qkv_dtype=qkv_dtype,
        mask_type=mask_type,
        device="cuda",
        seed=37000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    case = _apply_mla_tail_markers(case)
    _exercise_auto_mla_case(case)


@parametrize_product(
    {
        "page_size": (16, 64),
        "qkv_dtype": (torch.bfloat16, _FP8),
        "mask_type": ("dense", "causal"),
    },
    regular=(
        (16, torch.bfloat16, "dense"),
        (16, _FP8, "causal"),
        (64, torch.bfloat16, "causal"),
        (64, _FP8, "dense"),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_dtype_mask_page_product(
    page_size: int,
    qkv_dtype: torch.dtype,
    mask_type: str,
):
    """Cross uneven packed Q with dtype, mask, page, and cache rank."""

    q_lens = (1, 3, 5)
    max_q_len = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=8,
        max_kv_len=4097,
        seq_len_q=max_q_len,
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
        max_q_len=max_q_len,
    )
    assert policy["source"] == "auto"


@parametrize_product(
    {
        "num_qo_heads,expected_kernel": (
            (6, "throughput_latency_1cta"),
            (96, "throughput_2cta"),
        ),
        "qkv_dtype": (torch.bfloat16, _FP8),
    },
    regular=(
        ((6, "throughput_latency_1cta"), torch.bfloat16),
        ((96, "throughput_2cta"), _FP8),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_packed_non_power_of_two_heads(
    num_qo_heads: int,
    expected_kernel: str | None,
    qkv_dtype: torch.dtype,
):
    """Mixed empty packed requests predicate flat tiles in both families."""

    q_lens = (8, 1, 0, 3)
    max_q_len = max(q_lens)
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=num_qo_heads,
        max_kv_len=1024,
        seq_len_q=max_q_len,
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
        max_q_len=max_q_len,
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
        max_q_len=max_q_len,
    )
    one_shot = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
        mask_type=case.mask_type,
        max_kv_len=case.max_kv_len,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
        out_dtype=case.out_dtype,
    )
    _assert_case_correct(one_shot, case, policy, qo_indptr=qo_indptr)
    torch.testing.assert_close(one_shot, eager, rtol=0, atol=0)


# Empty queries bypass kernel execution, so one family covers the no-op
# contract in regular runs; retain both planned families under --full.
@parametrize_product(
    {
        "num_qo_heads,expected_kernel": (
            (6, "throughput_latency_1cta"),
            (96, "throughput_2cta"),
        )
    },
    regular=(((6, "throughput_latency_1cta"),),),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_all_empty_packed_query_noop(
    num_qo_heads: int,
    expected_kernel: str,
):
    """An explicitly bounded all-empty batch returns empty public outputs."""

    q_lens = (0, 0, 0, 0)
    max_q_len = 8
    case = _make_mla_case(
        batch_size=len(q_lens),
        num_qo_heads=num_qo_heads,
        max_kv_len=257,
        seq_len_q=max_q_len,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        page_size=64,
        device="cuda",
        seed=44000 + num_qo_heads,
    )
    case, qo_indptr = _pack_mla_case(case, q_lens)

    with pytest.raises(ValueError, match="requires max_q_len"):
        batch_mla_decode_with_paged_kv_cache(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            qo_indptr=qo_indptr,
            max_kv_len=case.max_kv_len,
        )

    wrapper = _plan_case(
        case,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
    )
    policy = _policy_dict(wrapper)
    _assert_auto_policy(
        policy,
        {"kernel": expected_kernel},
        device=case.query.device,
    )

    eager_out = torch.empty(
        (0, num_qo_heads, _LATENT_DIM),
        dtype=case.out_dtype,
        device=case.query.device,
    )
    eager = _run_case(wrapper, case, qo_indptr=qo_indptr, out=eager_out)
    assert eager is eager_out
    _assert_case_correct(eager, case, policy, qo_indptr=qo_indptr)

    standalone = _run_standalone(
        case,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
    )
    _assert_case_correct(standalone, case, policy, qo_indptr=qo_indptr)

    one_shot_out = torch.empty_like(eager)
    one_shot = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
        mask_type=case.mask_type,
        max_kv_len=case.max_kv_len,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
        out=one_shot_out,
        out_dtype=case.out_dtype,
    )
    assert one_shot is one_shot_out
    _assert_case_correct(one_shot, case, policy, qo_indptr=qo_indptr)


@parametrize_product(
    {
        "num_qo_heads,expected_kernel": (
            (16, "throughput_latency_1cta"),
            (128, "throughput_2cta"),
        ),
        "qkv_dtype": (torch.bfloat16, _FP8),
    },
    regular=(
        ((16, "throughput_latency_1cta"), torch.bfloat16),
        ((128, "throughput_2cta"), _FP8),
    ),
)
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
        max_kv_len=max(runtime_k),
        kv_seq_lens=runtime_k,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=39000 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
    )
    assert case.seq_lens_kv.tolist() == list(runtime_k)
    _exercise_auto_mla_case(
        case,
        expected_b200={"kernel": expected_kernel},
    )


@pytest.mark.parametrize(
    ("shape", "qkv_dtype", "softmax_rescale_threshold_log2", "expected_kernel"),
    (
        ((8, 128, 2049), torch.bfloat16, 8.0, "throughput_2cta"),
        ((8, 128, 2049), torch.float8_e4m3fn, 8.0, "throughput_2cta"),
        ((2, 8, 2048), torch.bfloat16, 8.0, "throughput_latency_1cta"),
        ((4, 16, 4097), torch.float8_e4m3fn, 8.0, "throughput_latency_1cta"),
        ((128, 16, 4096), torch.float8_e4m3fn, 8.0, "throughput_latency_1cta"),
    ),
    ids=(
        "2cta-bf16-t8",
        "2cta-fp8-t8",
        "1cta-bf16-t8",
        "1cta-fp8-t8",
        "1cta-fp8-t8-unsplit",
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_softmax_rescale_threshold(
    shape: tuple[int, int, int],
    qkv_dtype: torch.dtype,
    softmax_rescale_threshold_log2: float,
    expected_kernel: str,
) -> None:
    """A rescale threshold matches the protocol oracle on the 1CTA and 2CTA kernels.

    BF16 P is unscaled, so the exact softmax oracle applies; E4M3 P uses the
    scale ``448 * 2**-T``, which the FP8 oracle models per KV stream.
    """

    batch_size, num_qo_heads, max_kv_len = shape
    case = _make_mla_case(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_kv_len=max_kv_len,
        qkv_dtype=qkv_dtype,
        device="cuda",
        seed=36000
        + int(softmax_rescale_threshold_log2)
        + (1 if qkv_dtype == _FP8 else 0)
        + 7 * batch_size,
    )
    wrapper = _plan_case(
        case, softmax_rescale_threshold_log2=softmax_rescale_threshold_log2
    )
    policy = _policy_dict(wrapper)
    expected_policy: dict[str, object] = {"kernel": expected_kernel}
    if shape == (128, 16, 4096):
        # The unsplit case is what makes the freeze check mandatory on 1CTA.
        expected_policy["split_kv"] = 1
    _assert_auto_policy(policy, expected_policy, device=case.query.device)
    assert wrapper._plan_state is not None
    assert (
        wrapper._plan_state.softmax_rescale_threshold_log2
        == softmax_rescale_threshold_log2
    )

    output = _run_case(wrapper, case)
    _assert_case_correct(
        output,
        case,
        policy,
        softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
    )

    # The exact-rescale plan on identical inputs agrees to output rounding.
    baseline = _run_case(_plan_case(case), case)
    rtol, atol = _mla_tolerances(qkv_dtype)
    torch.testing.assert_close(output.float(), baseline.float(), rtol=rtol, atol=atol)
    if qkv_dtype == _FP8:
        # Positive control: the threshold kernel quantizes P at 448 * 2**-T, so
        # reusing the T = 0 kernel by mistake would match the baseline bitwise.
        assert not torch.equal(output, baseline)
        # The freeze itself: with P quantized against the held anchor, the
        # output sits closer to the freezing oracle than to one that keeps the
        # same scale but re-anchors on every new maximum.
        checked = tuple(range(batch_size)) if batch_size <= 8 else (0, batch_size - 1)
        oracle_args = dict(
            num_insts_kv=int(policy["num_insts_kv"]),
            tile_size_kv=int(policy["tile_size_kv"]),
            splits_kv=int(policy["split_kv"]),
            batch_indices=checked,
            softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
        )
        frozen = _mla_reference(case, **oracle_args)
        reanchored = _mla_reference(case, fp8_freeze=False, **oracle_args)
        # Heavy split-KV can leave one tile per KV stream, where nothing can
        # freeze. The 2CTA case and unsplit plans keep multi-tile streams and
        # must discriminate.
        if expected_kernel == "throughput_2cta" or int(policy["split_kv"]) == 1:
            assert not torch.equal(frozen, reanchored)
        if not torch.equal(frozen, reanchored):
            actual = output[list(checked)].float()
            assert torch.linalg.vector_norm(
                actual - frozen
            ) < 0.5 * torch.linalg.vector_norm(actual - reanchored)

    one_shot = batch_mla_decode_with_paged_kv_cache(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
        mask_type=case.mask_type,
        tuning=MLAAttentionTuning(
            softmax_rescale_threshold_log2=softmax_rescale_threshold_log2
        ),
    )
    torch.testing.assert_close(one_shot, output, rtol=0, atol=0)


def _spike_late_keys(case: _MLACase, factor: float) -> _MLACase:
    """Scale the RoPE part of one key in each of two consecutive late K tiles.

    Only the RoPE channels change, so the logits jump while V (the latent
    channels) and the output magnitude stay as generated.
    """

    kv_cache = case.kv_cache.clone()
    pages = kv_cache.view(kv_cache.shape[0], case.page_size, kv_cache.shape[-1])
    for batch_idx, seq_len in enumerate(case.seq_lens_kv.tolist()):
        for token in (seq_len - 2, seq_len - 2 - 128):
            page = int(case.block_tables[batch_idx, token // case.page_size])
            rope = pages[page, token % case.page_size, _LATENT_DIM:]
            pages[page, token % case.page_size, _LATENT_DIM:] = (
                rope.float() * factor
            ).to(rope.dtype)
    return replace(case, kv_cache=kv_cache)


def _max_tile_increase_log2(case: _MLACase) -> float:
    """Return the largest per-tile row-max increase (log2 units) in the case."""

    pages = case.kv_cache.view(
        case.kv_cache.shape[0], case.page_size, case.kv_cache.shape[-1]
    )
    largest = 0.0
    for batch_idx, seq_len in enumerate(case.seq_lens_kv.tolist()):
        page_ids = case.block_tables[batch_idx, : -(-seq_len // case.page_size)]
        keys = pages[page_ids.long()].reshape(-1, pages.shape[-1])[:seq_len].float()
        logits = case.query[batch_idx, 0].float() @ keys.T * case.softmax_scale
        tile_max = torch.stack(
            [tile.amax(dim=-1) for tile in logits.split(128, dim=-1)], dim=-1
        )
        running = torch.cummax(tile_max, dim=-1).values
        increase = (tile_max[:, 1:] - running[:, :-1]) * math.log2(math.e)
        largest = max(largest, float(increase.max()))
    return largest


@pytest.mark.parametrize(
    ("shape", "qkv_dtype", "softmax_rescale_threshold_log2", "expected_policy"),
    (
        (
            (64, 128, 1025),
            torch.bfloat16,
            0.5,
            {"kernel": "throughput_2cta", "split_kv": 1},
        ),
        (
            (64, 128, 1025),
            torch.float8_e4m3fn,
            1.0,
            {"kernel": "throughput_2cta", "split_kv": 1},
        ),
        (
            (128, 16, 1025),
            torch.bfloat16,
            0.5,
            {"kernel": "throughput_latency_1cta", "split_kv": 1},
        ),
        (
            (128, 16, 1025),
            torch.float8_e4m3fn,
            1.0,
            {"kernel": "throughput_latency_1cta"},
        ),
        # A 64-row Q tile is only legal for keeps-MMA-AB 1CTA profiles, which
        # keep the anchor and row sum in a separate scratch path.
        (
            (16, 64, 1025),
            torch.bfloat16,
            0.5,
            {"kernel": "throughput_latency_1cta", "tile_size_q": 64},
        ),
        (
            (16, 64, 1025),
            torch.float8_e4m3fn,
            1.0,
            {"kernel": "throughput_latency_1cta", "tile_size_q": 64},
        ),
    ),
    ids=(
        "2cta-bf16-t0.5",
        "2cta-fp8-t1",
        "1cta-bf16-t0.5",
        "1cta-fp8-t1",
        "1cta-keeps-ab-bf16-t0.5",
        "1cta-keeps-ab-fp8-t1",
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_softmax_rescale_threshold_reanchors_when_max_jumps(
    shape: tuple[int, int, int],
    qkv_dtype: torch.dtype,
    softmax_rescale_threshold_log2: float,
    expected_policy: dict[str, object],
) -> None:
    """A late tile raising the max by more than T re-anchors and rescales O.

    Random inputs almost never raise a row max by more than T after the first
    tiles, so the other threshold test only takes the frozen-anchor path. This
    test forces the correction path (new anchor, O and the row sum rescaled)
    after earlier tiles froze, and compares against the exact T=0 kernel.

    Spiked RoPE keys in two consecutive late K tiles (so both FP8 2CTA softmax
    groups) raise row maxima by more than the small threshold after earlier
    tiles froze the anchor. The 2CTA and swap cases run unsplit, so every spike
    lands behind frozen tiles.
    """

    batch_size, num_qo_heads, max_kv_len = shape
    case = _spike_late_keys(
        _make_mla_case(
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_kv_len=max_kv_len,
            qkv_dtype=qkv_dtype,
            device="cuda",
            seed=36500 + num_qo_heads + (1 if qkv_dtype == _FP8 else 0),
        ),
        factor=40.0,
    )
    assert _max_tile_increase_log2(case) > softmax_rescale_threshold_log2

    wrapper = _plan_case(
        case, softmax_rescale_threshold_log2=softmax_rescale_threshold_log2
    )
    policy = _policy_dict(wrapper)
    _assert_auto_policy(policy, expected_policy, device=case.query.device)

    output = _run_case(wrapper, case)
    _assert_case_correct(
        output,
        case,
        policy,
        softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
    )
    baseline = _run_case(_plan_case(case), case)
    rtol, atol = _mla_tolerances(qkv_dtype)
    torch.testing.assert_close(output.float(), baseline.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "heads,query_length,kv_length,batch",
    (
        (129, 1, 129, 2),
        (129, 3, 2049, 1),
        (191, 2, 513, 2),
        (192, 4, 2049, 1),
        (255, 3, 257, 2),
        (256, 1, 129, 2),
        (3, 5, 31, 2),
        (5, 9, 127, 2),
    ),
)
@pytest.mark.parametrize(
    "dtype", (torch.bfloat16, torch.float8_e4m3fn), ids=("bf16", "fp8")
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_generic_logical_head_rows(
    heads, query_length, kv_length, batch, dtype
):
    """Check O, split reduction, and graphs without externally padding heads."""
    case = _make_mla_case(
        batch_size=batch,
        num_qo_heads=heads,
        max_kv_len=kv_length,
        seq_len_q=query_length,
        qkv_dtype=dtype,
        device="cuda",
        seed=42000 + heads,
    )
    wrapper = _plan_case(case)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_2cta"
    _exercise_public_paths(wrapper, case, policy, exercise_all_paths=True)


@pytest.mark.parametrize("heads", (129, 256))
@pytest.mark.parametrize(
    "dtype", (torch.bfloat16, torch.float8_e4m3fn), ids=("bf16", "fp8")
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_decode_generic_packed_logical_rows(heads, dtype):
    """Packed partial tiles retain logical indexing across an empty request."""
    case = _make_mla_case(
        batch_size=3,
        num_qo_heads=heads,
        max_kv_len=257,
        seq_len_q=5,
        qkv_dtype=dtype,
        device="cuda",
        seed=43000 + heads,
    )
    case, offsets = _pack_mla_case(case, (5, 0, 2))
    wrapper = _plan_case(case, qo_indptr=offsets, max_q_len=5)
    policy = _policy_dict(wrapper)
    assert policy["kernel"] == "throughput_2cta"
    _exercise_public_paths(
        wrapper,
        case,
        policy,
        qo_indptr=offsets,
        max_q_len=5,
        exercise_all_paths=True,
    )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("heads", (16, 32, 64, 128, 129))
@pytest.mark.parametrize("dtype", (torch.bfloat16, _FP8), ids=("bf16", "fp8"))
@pytest.mark.parametrize("packed_query", (False, True), ids=("fixed", "packed"))
def test_attention_ts_mla_replanned_sizes(heads, dtype, packed_query):
    """Replanned problem sizes retain numerical accuracy and graph replay."""
    wrapper = BatchMLADecodePagedTSWrapper()
    workspace = torch.empty(
        max(
            get_prims_ts_batch_mla_decode_workspace_size(
                batch_size=batch,
                num_qo_heads=heads,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                page_size=32,
                max_kv_len=sk,
                max_q_len=sq,
                q_dtype=dtype,
                out_dtype=torch.bfloat16,
                device="cuda",
            )
            for batch, sq, sk in ((2, 1, 513), (3, 3, 641), (1, 5, 769), (2, 1, 514))
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    for batch, sq, sk in ((2, 1, 513), (3, 3, 641), (1, 5, 769), (2, 1, 514)):
        case = _make_mla_case(
            batch_size=batch,
            num_qo_heads=heads,
            max_kv_len=sk,
            seq_len_q=sq,
            qkv_dtype=dtype,
            seed=90400 + batch + sk,
        )
        qo_indptr = None
        if packed_query:
            case, qo_indptr = _pack_mla_case(case, (sq,) * batch)

        wrapper.plan(
            device="cuda",
            num_qo_heads=heads,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            page_size=32,
            packed_query=packed_query,
            q_dtype=dtype,
            kv_dtype=dtype,
            out_dtype=torch.bfloat16,
            batch_size=batch,
            max_q_len=sq,
            max_kv_len=sk,
            workspace_buffer=workspace,
        )

        def run(out=None, validate=True):
            return wrapper.run(
                case.query,
                case.kv_cache,
                case.block_tables,
                case.seq_lens_kv,
                qo_indptr=qo_indptr,
                softmax_scale=case.softmax_scale,
                output_scale=case.output_scale,
                out=out,
                validate=validate,
            )

        output = run()
        _assert_case_correct(
            output, case, dict(wrapper._plan_state.policy), qo_indptr=qo_indptr
        )
        assert wrapper._plan_state.workspace_buffer.data_ptr() == workspace.data_ptr()
    out = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out=out, validate=False)
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(
        out, case, dict(wrapper._plan_state.policy), qo_indptr=qo_indptr
    )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("batch,sq,kv", ((2, 3, 641), (1, 5, 769)))
def test_attention_ts_mla_packed_query_allocation_boundary(batch, sq, kv):
    """Packed query TMA must stay within the caller's physical allocation."""
    from tests.attention.test_attention_ts_q_token_kv_block_sparse_metadata import (
        _query_at_unmapped_boundary,
    )

    case = _make_mla_case(
        batch_size=batch,
        num_qo_heads=16,
        max_kv_len=kv,
        seq_len_q=sq,
        qkv_dtype=_FP8,
        seed=90400 + batch + kv,
    )
    case, offsets = _pack_mla_case(case, (sq,) * batch)
    with _query_at_unmapped_boundary(case.query.shape, case.query.dtype) as query:
        query.copy_(case.query)
        case = replace(case, query=query)
        wrapper = _plan_case(case, qo_indptr=offsets, max_q_len=sq)
        output = _run_case(wrapper, case, qo_indptr=offsets)
        _assert_case_correct(
            output, case, dict(wrapper._plan_state.policy), qo_indptr=offsets
        )


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_standalone_reuses_caller_workspace():
    """Standalone MLA reuses caller scratch and omits unused final LSE."""
    workspace = torch.empty(
        max(
            get_prims_ts_batch_mla_decode_workspace_size(
                batch_size=batch,
                num_qo_heads=32,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                page_size=32,
                max_kv_len=sk,
                max_q_len=sq,
                q_dtype=torch.bfloat16,
                device="cuda",
            )
            for batch, sq, sk in ((2, 1, 513), (3, 3, 641), (1, 5, 769))
        ),
        dtype=torch.int8,
        device="cuda",
    )
    for batch, sq, sk in ((2, 1, 513), (3, 3, 641), (1, 5, 769)):
        case = _make_mla_case(
            batch_size=batch,
            num_qo_heads=32,
            max_kv_len=sk,
            seq_len_q=sq,
            qkv_dtype=torch.bfloat16,
            seed=90700 + sk,
        )
        output = batch_mla_decode_with_paged_kv_cache(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            max_kv_len=sk,
            workspace_buffer=workspace,
            softmax_scale=case.softmax_scale,
            output_scale=case.output_scale,
        )
        expected = _mla_reference(case, num_insts_kv=1, tile_size_kv=128, splits_kv=1)
        torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.02)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("dtype", (torch.bfloat16, _FP8), ids=("bf16", "fp8"))
def test_attention_ts_mla_runtime_sm_split_ceiling(dtype):
    """Canonical KV geometry retains live counts above the historical 128 cap."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms <= 128:
        pytest.skip("requires more than 128 physical SMs")
    wrapper = BatchMLADecodePagedTSWrapper()
    workspace = torch.empty(
        get_prims_ts_batch_mla_decode_workspace_size(
            1, 16, 512, 64, 32, sms * 256, q_dtype=dtype, device="cuda"
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    compiled = []
    for sk in (sms * 256, (sms - 1) * 256):
        case = _make_mla_case(
            batch_size=1,
            num_qo_heads=16,
            max_kv_len=sk,
            qkv_dtype=dtype,
            seed=90900 + sk,
        )
        wrapper.plan(
            device="cuda",
            num_qo_heads=16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            page_size=32,
            packed_query=False,
            q_dtype=dtype,
            kv_dtype=dtype,
            out_dtype=torch.bfloat16,
            batch_size=1,
            max_q_len=1,
            max_kv_len=sk,
            workspace_buffer=workspace,
        )
        wrapper._plan_state.workspace_buffer.view(torch.uint8).fill_(255)
        output = wrapper.run(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            softmax_scale=case.softmax_scale,
            output_scale=case.output_scale,
        )
        policy = dict(wrapper._plan_state.policy)
        assert 128 < policy["split_kv"] <= sms
        _assert_case_correct(output, case, policy)
        assert wrapper._plan_state.workspace_views.lse is None
        compiled.append(wrapper._plan_state.compiled)
    assert compiled[0] is compiled[1]


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    "dtype,heads,nq,nk,packed,mode",
    [
        (torch.bfloat16, 32, 1, 129, False, "base2"),
        (torch.bfloat16, 128, 1, 8192, False, "base2"),
        (torch.float8_e4m3fn, 64, 4, 2049, False, "base2"),
        (torch.bfloat16, 129, 3, 257, True, "base2"),
        (torch.float8_e4m3fn, 256, 3, 513, True, "base2"),
    ],
)
@pytest.mark.parametrize("store_stats", (False, True))
def test_mla_final_lse(dtype, heads, nq, nk, packed, mode, store_stats):
    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=heads,
        seq_len_q=nq,
        max_kv_len=nk,
        page_size=32,
        qkv_dtype=dtype,
        mask_type="causal",
    )
    offsets = None
    if packed:
        case, offsets = _pack_mla_case(case, (nq, 0))
    query = case.query
    wrapper = _plan_case(
        case,
        qo_indptr=offsets,
        max_q_len=nq,
        lse_mode=mode,
        store_softmax_stats=store_stats,
    )
    refs = []
    stats_refs = []
    for b, n in enumerate(case.seq_lens_kv.tolist()):
        qb = query[int(offsets[b]) : int(offsets[b + 1])] if packed else query[b]
        pages = case.block_tables[b, : (n + 31) // 32].long()
        kv = case.kv_cache[pages, 0].reshape(-1, 576)[:n].float()
        scores = torch.einsum("qhd,kd->qhk", qb.float(), kv) * case.softmax_scale
        keys = torch.arange(n, device="cuda")
        end = torch.arange(qb.shape[0], device="cuda") + n - qb.shape[0]
        scores.masked_fill_(keys[None, None] > end[:, None, None], -torch.inf)
        refs.append(torch.logsumexp(scores, -1))
        maximum = scores.max(dim=-1).values
        denominator = torch.exp(scores - maximum[..., None]).sum(-1)
        stats_refs.append(torch.stack((maximum, denominator), -1))
    expected = torch.cat(refs) if packed else torch.stack(refs)
    if mode == "base2":
        expected = expected / math.log(2)
    # Scalar FP32 LSE stores must accept a 4-byte-aligned offset view.
    guard = torch.full((expected.numel() + 2,), float("nan"), device="cuda")
    lse = guard[1:-1].view(expected.shape)
    softmax_stats = (
        torch.full((*query.shape[:-1], 2), torch.nan, device="cuda")
        if store_stats
        else None
    )
    expected_stats = torch.cat(stats_refs) if packed else torch.stack(stats_refs)
    run = functools.partial(
        _run_case,
        wrapper,
        case,
        qo_indptr=offsets,
        lse=lse,
        softmax_stats=softmax_stats,
    )
    out, stats = run(return_lse=True)
    _assert_case_correct(out, case, dict(wrapper._plan_state.policy), qo_indptr=offsets)
    assert stats is lse
    torch.testing.assert_close(lse, expected, atol=0.02, rtol=0.002)
    if store_stats:
        torch.testing.assert_close(
            softmax_stats, expected_stats, atol=0.002, rtol=0.002
        )
    assert torch.isnan(guard[0]) and torch.isnan(guard[-1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out=out, validate=False)
    out.fill_(torch.nan)
    lse.fill_(float("nan"))
    if store_stats:
        softmax_stats.fill_(torch.nan)
    graph.replay()
    _assert_case_correct(out, case, dict(wrapper._plan_state.policy), qo_indptr=offsets)
    torch.testing.assert_close(lse, expected, atol=0.02, rtol=0.002)
    if store_stats:
        torch.testing.assert_close(
            softmax_stats, expected_stats, atol=0.002, rtol=0.002
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
@pytest.mark.parametrize(
    "heads,batch,kv",
    [
        (16, 3, 2304),  # Rounded capacity changes cluster residency.
        (64, 1, 8192),
        (64, 1, 19456),
        (128, 1, 8192),  # Balance a live prefix across the full Q128 cluster.
        (128, 2, 8192),
    ],
)
def test_mla_runtime_count_changes(dtype, heads, batch, kv):
    case = _make_mla_case(
        batch_size=batch, num_qo_heads=heads, max_kv_len=kv, qkv_dtype=dtype
    )
    wrapper = _plan_case(case, lse_mode="base2")
    state = wrapper._plan_state
    policy = dict(state.policy)
    capacity = policy["split_capacity"]
    assert capacity <= torch.cuda.get_device_properties(0).multi_processor_count
    compiled = state.compiled
    misses = mla_decode_module._get_compiled_mla_decode.cache_info().misses
    out = torch.empty(
        (*case.query.shape[:-1], 512), device="cuda", dtype=torch.bfloat16
    )
    lse = torch.empty(case.query.shape[:-1], device="cuda")
    # Reuse the same compiled callable/scratch for increasing and decreasing counts.
    for count in (
        *[n for n in (1, 2, 3, 4, 8, 9, 16, 17, 32) if n <= capacity],
        capacity,
        2,
    ):
        wrapper._plan_state = replace(state, split_kv=count)
        scratch = state.workspace_views.kernel_workspace
        if scratch is not None:
            scratch.fill_(-1)
        result, stats = wrapper.run(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            softmax_scale=case.softmax_scale,
            output_scale=case.output_scale,
            out=out,
            lse=lse,
            return_lse=True,
        )
        expected = _mla_reference(
            case, num_insts_kv=policy["num_insts_kv"], tile_size_kv=128, splits_kv=count
        )
        rtol, atol = _mla_tolerances(dtype)
        torch.testing.assert_close(result.float(), expected, rtol=rtol, atol=atol)
        assert torch.isfinite(stats).all()
        assert wrapper._plan_state.compiled is compiled
    assert mla_decode_module._get_compiled_mla_decode.cache_info().misses == misses
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            softmax_scale=case.softmax_scale,
            output_scale=case.output_scale,
            out=out,
            lse=lse,
            validate=False,
        )
    out.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(out.float(), expected, rtol=rtol, atol=atol)

    # Graph-live shorter lengths must contract the useful prefix and ignore
    # previously published partials beyond it without host policy/JIT work.
    case.seq_lens_kv.clamp_(max=129)
    graph.replay()
    shortened = _mla_reference(
        case, num_insts_kv=policy["num_insts_kv"], tile_size_kv=128, splits_kv=2
    )
    torch.testing.assert_close(out.float(), shortened, rtol=rtol, atol=atol)
    assert mla_decode_module._get_compiled_mla_decode.cache_info().misses == misses

    live_lengths = case.seq_lens_kv.clone()
    live_pages = case.block_tables.clone()
    case.block_tables.fill_(-123)
    for inactive_length in (0, -1):
        case.seq_lens_kv.fill_(inactive_length)
        out.fill_(8)
        lse.fill_(9)
        graph.replay()
        torch.testing.assert_close(out, torch.full_like(out, 8), rtol=0, atol=0)
        torch.testing.assert_close(lse, torch.full_like(lse, 9), rtol=0, atol=0)
    case.seq_lens_kv.copy_(live_lengths)
    case.block_tables.copy_(live_pages)
    out.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(out.float(), shortened, rtol=rtol, atol=atol)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(("seq_len_q", "kv_len"), ((8, 8192), (4, 2048)))
def test_attention_ts_mla_decode_2cta_runtime_small_split_reduction(seq_len_q, kv_len):
    """Runtime small splits preserve 2CTA numerical and graph behavior."""
    case = _make_mla_case(
        batch_size=4,
        num_qo_heads=128,
        max_kv_len=kv_len,
        seq_len_q=seq_len_q,
        kv_seq_lens=(kv_len, kv_len - 191, kv_len - 193, kv_len - 292),
        qkv_dtype=torch.bfloat16,
        device="cuda",
        seed=710629,
    )
    wrapper = _plan_case(case)
    _exercise_public_paths(
        wrapper,
        case,
        _policy_dict(wrapper),
        exercise_all_paths=True,
    )
