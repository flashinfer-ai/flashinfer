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

"""Compact public-interface acceptance coverage for PrimTS MLA decode."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
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
    batch_decode_mla_with_paged_kv_cache,
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
    get_prims_ts_batch_decode_mla_workspace_size,
    prims_ts_batch_decode_with_kv_cache_mla,
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


def _gather_request_cache(case: _MLACase, batch_idx: int) -> torch.Tensor:
    seq_len = int(case.seq_lens[batch_idx].item())
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
):
    wrapper = BatchMLADecodePagedTSWrapper()
    num_heads = int(
        case.query.shape[1] if qo_indptr is not None else case.query.shape[2]
    )
    wrapper.plan(
        case.block_tables,
        case.seq_lens,
        num_heads,
        _LATENT_DIM,
        _ROPE_DIM,
        case.page_size,
        qo_indptr=qo_indptr,
        max_seq_len_q=(
            int(case.query.shape[1])
            if qo_indptr is None and max_seq_len_q is None
            else max_seq_len_q
        ),
        q_data_type=case.query.dtype,
        kv_data_type=case.kv_cache.dtype,
        o_data_type=case.output_dtype,
        mask_type=case.mask_type,
        max_kv_len=case.max_seq_len,
    )
    return wrapper


def _run_case(wrapper, case, *, out=None):
    return wrapper.run(
        case.query,
        case.kv_cache,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out=out,
    )


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
    workspace_size = get_prims_ts_batch_decode_mla_workspace_size(
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
    result = prims_ts_batch_decode_with_kv_cache_mla(
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
    policy = dict(wrapper._policy)
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

    eager = _run_case(wrapper, case)
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
        captured = _run_case(wrapper, case, out=graph_out)
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
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


def test_attention_ts_mla_public_surfaces_hide_internal_tuning_policy():
    """Keep kernel policy automatic and the public signatures explicit."""

    surfaces = (
        BatchMLADecodePagedTSWrapper.__init__,
        BatchMLADecodePagedTSWrapper.plan,
        BatchMLADecodePagedTSWrapper.run,
        batch_decode_mla_with_paged_kv_cache,
        get_prims_ts_batch_decode_mla_workspace_size,
        prims_ts_batch_decode_with_kv_cache_mla,
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


def test_attention_ts_mla_decode_bound_wrapper_trace_uses_plan_state():
    """Trace packed-Q shapes from the live MLA wrapper plan state."""
    from flashinfer.fi_trace import fi_trace

    wrapper = BatchMLADecodePagedTSWrapper()
    query = torch.empty((5, 8, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((9, 32, 576), dtype=torch.bfloat16)
    kwargs = {"query": query, "kv_cache": kv_cache}

    with pytest.raises(
        ValueError,
        match=r"requires the live wrapper's plan state.*flashinfer\.fi_trace",
    ):
        wrapper.run.fi_trace(**kwargs)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        fi_trace(wrapper.run, **kwargs)

    wrapper._planned = True
    wrapper._packed_query = True
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
    assert defn["axes"]["kv_lora_rank"]["type"] == "var"


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
        wrapper.run(None, None)


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
        get_prims_ts_batch_decode_mla_workspace_size(
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
        get_prims_ts_batch_decode_mla_workspace_size(
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
def test_attention_ts_mla_plan_rejects_invalid_kv_lengths(
    seq_lens,
    max_kv_len,
    message,
):
    """Validate every planned MLA K/V length and its explicit static bound."""

    device = torch.device("cuda")
    block_tables = torch.zeros((2, 3), dtype=torch.int32, device=device)
    planned_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match=message):
        BatchMLADecodePagedTSWrapper().plan(
            block_tables,
            planned_seq_lens,
            8,
            _LATENT_DIM,
            _ROPE_DIM,
            _DEFAULT_PAGE_SIZE,
            max_kv_len=max_kv_len,
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
    seq_len_q = None if packed_q else 8
    max_seq_len_q = 5 if packed_q else None
    query = torch.empty(
        (6, 8, _QK_DIM) if packed_q else (2, 8, 8, _QK_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.empty((2, page_size, _QK_DIM), dtype=torch.bfloat16, device=device)
    match = r"request 0 has Q=(5|8) and K/V=4"

    with pytest.raises(ValueError, match=match):
        BatchMLADecodePagedTSWrapper().plan(
            block_tables,
            seq_lens,
            8,
            _LATENT_DIM,
            _ROPE_DIM,
            page_size,
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            mask_type="causal",
            max_kv_len=16,
        )
    with pytest.raises(ValueError, match=match):
        batch_decode_mla_with_paged_kv_cache(
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
        prims_ts_batch_decode_with_kv_cache_mla(
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
    assert (
        wrapper._workspace_layout.kernel_workspace.byte_size
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
    assert wrapper._packed_query is True
    assert wrapper._max_seq_len_q == max_seq_len_q
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
    one_shot = batch_decode_mla_with_paged_kv_cache(
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

    derived_bound_one_shot = batch_decode_mla_with_paged_kv_cache(
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
        captured = _run_case(wrapper, case, out=graph_out)
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

    eager = _run_case(wrapper, case).clone()
    _assert_case_correct(eager, case, policy, qo_indptr=qo_indptr)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(wrapper, case, out=graph_out)
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
        output = _run_case(wrapper, case)
        _assert_case_correct(output, case, policy, qo_indptr=qo_indptr)
        wrappers.append(wrapper)

    assert wrappers[0]._compiled is wrappers[1]._compiled


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
    one_shot = batch_decode_mla_with_paged_kv_cache(
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
        _plan_case(case, qo_indptr=qo_indptr)
    with pytest.raises(
        ValueError, match="max_seq_len_q is required for an all-empty packed query"
    ):
        batch_decode_mla_with_paged_kv_cache(
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
    eager = _run_case(wrapper, case, out=eager_out)
    assert eager is eager_out
    _assert_case_correct(eager, case, policy, qo_indptr=qo_indptr)

    standalone = _run_standalone(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    _assert_case_correct(standalone, case, policy, qo_indptr=qo_indptr)

    one_shot_out = torch.empty_like(eager)
    one_shot = batch_decode_mla_with_paged_kv_cache(
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
