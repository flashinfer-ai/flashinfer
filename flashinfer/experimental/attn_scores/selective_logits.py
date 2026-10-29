# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepared API for selective FP8 MQA logits publication.

This module provides the selective-logits producer: a request-tiled engine
consumes FlashInfer's fused paged FP8 KV representation and publishes only
bounded samples, candidates, radix histograms, or exact-completion indices.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import importlib.util
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Tuple

import torch

from ...api_logging import flashinfer_api
from ...utils import get_compute_capability
from ...attn_scores.selective_logits import FP8PagedMQATopKWrapper as _PublicWrapper
from ...attn_scores._fp8_mqa_constants import (
    _COMPUTE_BLOCK_KV,
    _EPI_UNROLL,
    _FP8_MMA_INST_K,
    _MAX_BLOCKS_PER_MMA,
)
from ...attn_scores._math_policy import _MathMode, _validate_math_mode
from ._selective_contract import (
    _CANDIDATE_CAPACITY,
    _COUNT_SEGMENTS,
    _MAX_STRIPED_SPLIT_KV,
    _REPAIR_CAPACITY,
    _SAMPLE_HISTOGRAM_BINS,
    _SPLIT_KV_CHOICES,
    _SelectiveLogitsMode,
    _h64_single_pool_topology,
    _q1_owner_pool_topology,
)
from .trace import (
    fp8_paged_mqa_topk_trace_dispatch,
)

try:  # pragma: no cover - exercised by whichever branch this torch provides
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover
    _raw_stream = None

_CUTE_DSL_AVAILABLE = (
    importlib.util.find_spec("cutlass") is not None
    and importlib.util.find_spec("cutlass.cute") is not None
)

_MAX_QUERY_TILE = 16
_MIN_SHORT_QUERY_LOGICAL_N = 32
_METADATA_HEADER_COLUMNS = 4
_DEFAULT_TOP_K = 512
_MAX_TOP_K = 2048
_SAMPLE_CAPACITY = 3840
_MAX_TARGET_CANDIDATES = 4096


def _select_sparse_sample_group_rows(query_tile: int) -> int:
    """Keep 128-row sharing where legal; otherwise share one complete tile."""
    return 128 if 128 % query_tile == 0 else query_tile


def _sparse_sample_gather4_serves(
    num_pages: int, page_stride: int, page_size: int, head_dim: int
) -> bool:
    """Admit gather4 only when its padded byte pool has int32 row coordinates."""
    return (
        head_dim == 128
        and page_size in (32, 64, 128)
        and page_stride % 128 == 0
        and (num_pages - 1) * (page_stride // 128) + page_size * (head_dim + 4) // 128
        <= 2**31 - 1
    )


def _validate_selective_device(device: torch.device) -> None:
    """Reject devices outside the selective kernels' supported architectures."""
    if get_compute_capability(device) not in ((10, 0), (10, 3)):
        raise ValueError("selective FP8 logits requires SM100 or SM103")


def _candidate_target(top_k: int) -> int:
    """Target four times K candidates, clipped to 512..4096.

    Publication capacity is selected separately for each ownership layout.
    Targets above K1024 remain capped at 4096; bank overflow and underfill use
    exact repair. Sample ranks remain clipped to each row's valid samples.
    The target/K margin decreases to 2 at K2048. Population and bank margins
    are throughput heuristics; bounded publication and exact repair establish
    correctness independently of the sample's population estimate.
    """
    return min(_MAX_TARGET_CANDIDATES, max(512, 4 * top_k))


def _validate_selective_top_k(top_k: int) -> None:
    """Require an integer TopK in 1..2048."""
    if type(top_k) is not int or not 0 < top_k <= _MAX_TOP_K:
        raise ValueError("top_k must be an integer in [1, 2048]")


_CandidatePublication = Literal["warp_striped", "warp_pooled", "lane_local"]
_WARP_STRIPED_PUBLICATION: Literal["warp_striped"] = "warp_striped"
_WARP_POOLED_PUBLICATION: Literal["warp_pooled"] = "warp_pooled"


class _CaptureStatusError(RuntimeError):
    """A host boundary observed an invalidated CUDA capture."""


@functools.cache
def _capture_runtime():
    # Resolve bindings before capture, without compiling a native helper.
    try:
        from cuda.bindings import runtime
    except ImportError:  # CUDA Python before the bindings namespace split
        from cuda import cudart as runtime

    return runtime


def _check_capture_status(stream, step):
    runtime = _capture_runtime()
    result = runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
    error, status = result[:2]
    if error != runtime.cudaError_t.cudaSuccess or (
        status == runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusInvalidated
    ):
        raise _CaptureStatusError(
            f"CUDA graph capture invalid after {step}: "
            f"cudaStreamGetCaptureInfo returned {error}, capture status {status}"
        )


@dataclass(frozen=True)
class _CandidateSchedule:
    """Shape-selected candidate kernel topology."""

    query_tile: int
    logical_n: int
    num_epi_subtiles: int
    num_umma_stages: int
    split_kv: int
    publication: _CandidatePublication
    count_segments: int
    sample_count_segments: int
    # Publication capacity and bank count also bound the sampled ladder policy.
    candidate_capacity: int = _CANDIDATE_CAPACITY
    # Single-pool H64 prefill published by the specialized H64 candidate
    # variant, one query row per math warpgroup in four warp segments.
    h64_candidate: bool = False


def _candidate_count_segments(split_kv: int) -> int:
    """Return striped low-split or pooled high-split publisher segments."""
    return 4 * split_kv if split_kv <= _MAX_STRIPED_SPLIT_KV else split_kv


def _select_candidate_publication(
    split_kv: int,
    *,
    lane_local: bool = False,
) -> tuple[_CandidatePublication, int]:
    """Choose the production publisher layout for a split count.

    Prefill uses lane-local publication through four splits to preserve
    independent candidate cursors in the long-running score kernel. Decode uses
    compact warp-ranked stripes, then pooled segments above eight splits.
    """
    if lane_local:
        if split_kv > 4:
            raise ValueError("lane-local publication supports at most four splits")
        return "lane_local", 32 * split_kv
    if split_kv <= _MAX_STRIPED_SPLIT_KV:
        return _WARP_STRIPED_PUBLICATION, _candidate_count_segments(split_kv)
    return _WARP_POOLED_PUBLICATION, _candidate_count_segments(split_kv)


if _CUTE_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    # Load DSL definitions before a caller owns plan or graph storage. Supported
    # DSL decorators can retain their importing frames until cyclic collection;
    # a first import inside a plan would keep that plan's CUDA owners alive.
    # Kernel compilation remains lazy.
    for _definition_module in (
        "flashinfer.attn_scores.kernels.schedule_kernel",
        ".kernels._fp8_mqa_selective",
        ".kernels.selective_logits_metadata",
        "flashinfer.topk_varlen.kernels.gvr2_topk_decode",
    ):
        importlib.import_module(_definition_module, __package__)
    del _definition_module


@dataclass(frozen=True)
class _PreparedFP8MQASelectiveLogits:
    """Address-stable metadata and specialization for selective FP8 logits.

    Instances are created by :func:`_prepare_fp8_mqa_selective_logits`. Their
    tensor addresses are retained by compiled CuTe launch arguments; use
    :func:`_refresh_fp8_mqa_selective_logits` to update values in place rather
    than replacing these tensors during CUDA graph replay.
    """

    context_lens: torch.Tensor
    tile_meta: torch.Tensor
    schedule_meta: torch.Tensor
    schedule_prefix: torch.Tensor
    rows: int
    num_phys_blocks: int
    page_size: int
    query_tile: int
    num_epi_subtiles: int
    tiles: int
    batches: int
    num_sms: int
    kv_chunk_size: int
    split_kv: int
    causal_bounds: bool
    row_end_base: int
    num_heads: int
    head_dim: int
    math_mode: _MathMode = "relu_strict"
    num_umma_stages: int = 2
    block_table_columns: int = 0
    capacity_bounds: bool = False
    _launches: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _metadata_modes: set[str] = field(
        default_factory=set, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_math_mode(self.math_mode)


@dataclass
class _CachedSelectiveLogitsLaunch:
    identity: tuple[Any, ...]
    tensors: tuple[torch.Tensor, ...]
    compiled: Any
    args: tuple[Any, ...]
    count_view: torch.Tensor | None


@dataclass(frozen=True)
class _CachedFinalizeLaunch:
    identity: tuple[Any, ...]
    tensors: tuple[torch.Tensor, ...]
    compiled: Any
    args: tuple[Any, ...]


def _selective_logits_launch_identity(
    tensors: tuple[torch.Tensor, ...],
    capacity: int,
) -> tuple[Any, ...]:
    """Identify an address- and layout-stable compiled launch binding."""
    return (
        capacity,
        tuple(
            (
                id(tensor),
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
            )
            for tensor in tensors
        ),
    )


def _selective_logits_tensor_specs(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[Any, ...]:
    """Return the non-address tensor contract used for launch rebinding."""
    return tuple(
        (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )
        for tensor in tensors
    )


_METADATA_COMPILED: dict[Tuple[int, ...], Any] = {}
_COMPACT_METADATA_COMPILED: dict[Tuple[int, ...], Any] = {}
_SELECTIVE_LOGITS_COMPILED: dict[Tuple[Any, ...], Any] = {}
_FINALIZE_COMPILED: dict[Tuple[int, ...], Any] = {}
_RADIX_SELECT_COMPILED: dict[Tuple[int, ...], Any] = {}
_COMBINE_EXACT_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_THRESHOLD_COMPILED: dict[Tuple[int, ...], Any] = {}
_PACKED_TOPK_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_RESOLVE_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_ROW_COMPACT_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_ROW_PACK_COMPILED: dict[Tuple[int, ...], Any] = {}
_COMPACT_REPAIR_RESOLVE_COMPILED: dict[Tuple[int, ...], Any] = {}
_COMPACT_COMBINE_EXACT_COMPILED: dict[Tuple[int, ...], Any] = {}
_STREAM_CACHE: dict[Tuple[int, int], Any] = {}


@functools.cache
def _shared_engine_source_token() -> str:
    """Bind compiled executors to this process's shared-engine source snapshot.

    Read once before compilation, never per replay. A fresh process is required
    after source edits; this is not a Python module hot-reload contract. Missing
    source files fail closed rather than producing a shape-only cache identity.
    """
    kernel_dir = Path(__file__).parent / "kernels"
    core_dir = Path(__file__).parents[2] / "attn_scores"
    digest = hashlib.sha256()
    for name, path in (
        ("fp8_paged_mqa_logits.py", core_dir / "kernels/fp8_paged_mqa_logits.py"),
        ("_fp8_mqa_selective.py", kernel_dir / "_fp8_mqa_selective.py"),
        ("../_math_policy.py", core_dir / "_math_policy.py"),
        ("../_selective_contract.py", kernel_dir.parent / "_selective_contract.py"),
        ("../_fp8_mqa_constants.py", core_dir / "_fp8_mqa_constants.py"),
    ):
        contents = path.read_bytes()
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def _byte_range(tensor: torch.Tensor) -> Tuple[int, int]:
    """Return the occupied positive-stride byte interval for alias checks."""
    elements = 1
    if tensor.numel() != 0:
        elements += sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    begin = tensor.data_ptr()
    return begin, begin + elements * tensor.element_size()


def _overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _select_candidate_split_kv(
    q_lengths: list[int],
    kv_lengths: list[int],
    *,
    query_tile: int,
    num_sms: int,
) -> int:
    """Choose the smallest production split with the best SM-wave utilization."""
    best_split = 1
    best_work_items = 0
    best_wave_slots = 1
    base_work_items = sum(
        math.ceil(q_length / query_tile)
        for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True)
        if q_length > 0 and kv_length > 0
    )
    # Higher split counts are useful only while request tiles underfill one
    # device wave. Once the unsplit grid fills a wave, keep the established
    # compact-layout choice and avoid multiplying finalization work.
    decode_shaped = max(q_lengths, default=0) <= query_tile
    split_candidates: tuple[int, ...]
    if decode_shaped and base_work_items == 1:
        split_candidates = _SPLIT_KV_CHOICES
    elif decode_shaped and base_work_items < num_sms:
        split_candidates = _SPLIT_KV_CHOICES[:4]
    else:
        split_candidates = _SPLIT_KV_CHOICES[:2]
    for split_kv in split_candidates:
        work_items = sum(
            math.ceil(q_length / query_tile)
            * min(split_kv, math.ceil(kv_length / (2 * _COMPUTE_BLOCK_KV)))
            for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True)
            if q_length > 0 and kv_length > 0
        )
        wave_slots = math.ceil(work_items / num_sms) * num_sms
        if work_items * best_wave_slots > best_work_items * wave_slots:
            best_split = split_kv
            best_work_items = work_items
            best_wave_slots = wave_slots
    return best_split


def _require_aligned(tensors: tuple[torch.Tensor, ...]) -> None:
    if any(tensor.data_ptr() % 16 != 0 for tensor in tensors):
        raise ValueError("selective-logits tensor bases must be 16-byte aligned")


def _validate_specialization(
    device: torch.device,
    num_heads: int,
    head_dim: int,
    *,
    max_query_tile: int = _MAX_QUERY_TILE,
    num_umma_stages: int | None = None,
) -> Any:
    """Choose the largest legal request tile for the paged FP8 kernel shape."""
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    if head_dim % _FP8_MMA_INST_K:
        raise ValueError(
            f"head_dim must be a multiple of {_FP8_MMA_INST_K}; got {head_dim}"
        )
    if not isinstance(max_query_tile, int) or not 0 < max_query_tile <= _MAX_QUERY_TILE:
        raise ValueError(f"max_query_tile must be in [1, {_MAX_QUERY_TILE}]")
    from .kernels._fp8_mqa_selective import SELECTIVE_LOGITS_TILING

    properties = torch.cuda.get_device_properties(device)
    smem_limit = max(
        int(getattr(properties, "shared_memory_per_block_optin", 0)),
        int(getattr(properties, "sharedMemPerBlockOptin", 0)),
        int(getattr(properties, "shared_memory_per_block", 0)),
    )
    # Preserve the tuned two-stage policy wherever it fits. Fall back to one
    # UMMA stage only for shapes in the full paged kernel's N=256 envelope.
    if num_umma_stages is not None and num_umma_stages not in (1, 2):
        raise ValueError("num_umma_stages must be 1 or 2")
    stage_counts = (
        (SELECTIVE_LOGITS_TILING.num_umma_stages, 1)
        if num_umma_stages is None
        else (num_umma_stages,)
    )
    for num_umma_stages in stage_counts:
        tmem_limit = 512 // (2 * num_umma_stages)
        for query_tile in range(max_query_tile, 0, -1):
            logical_n = query_tile * num_heads
            if logical_n < 8 or logical_n > tmem_limit or logical_n % 8:
                continue
            num_epi_subtiles = next(
                (
                    subtiles
                    for subtiles in (8, 4, 2, 1)
                    if logical_n % subtiles == 0
                    and (logical_n // subtiles) % _EPI_UNROLL == 0
                    and (subtiles % query_tile == 0 or query_tile % subtiles == 0)
                ),
                None,
            )
            if num_epi_subtiles is None:
                continue
            w_stage_bytes = ((logical_n * 4 + 127) // 128) * 128
            kv_scale_bytes = (
                2
                * (
                    SELECTIVE_LOGITS_TILING.block_kv * head_dim
                    + SELECTIVE_LOGITS_TILING.block_kv * 4
                )
                * SELECTIVE_LOGITS_TILING.num_kv_stages
            )
            q_weight_bytes = (
                logical_n * head_dim + w_stage_bytes
            ) * SELECTIVE_LOGITS_TILING.num_q_stages
            store_score_bytes = (
                query_tile * SELECTIVE_LOGITS_TILING.block_kv * 2 * 2 * 4
            )
            required_smem = kv_scale_bytes + q_weight_bytes + store_score_bytes + 1024
            if required_smem <= smem_limit:
                return replace(
                    SELECTIVE_LOGITS_TILING,
                    next_n=query_tile,
                    num_epi_subtiles=num_epi_subtiles,
                    num_umma_stages=num_umma_stages,
                )
    raise ValueError(
        f"no legal selective FP8 query tile for num_heads={num_heads}, "
        f"head_dim={head_dim} within N<={tmem_limit} and {smem_limit} bytes SMEM"
    )


def _select_request_tiling(
    device: torch.device,
    num_heads: int,
    head_dim: int,
    q_lengths: list[int],
) -> Any:
    """Select a request tile without padding short-query work to Q16."""
    min_query_tile = math.ceil(_MIN_SHORT_QUERY_LOGICAL_N / num_heads)
    # Scope N8 to H8/D128 decode; preserve other short-query tiling policies.
    if (
        num_heads == 8
        and head_dim == 128
        and q_lengths
        and all(q_length == 1 for q_length in q_lengths)
    ):
        min_query_tile = 1
    max_query_tile = min(
        _MAX_QUERY_TILE,
        max(max(q_lengths, default=1), min_query_tile),
    )
    return _validate_specialization(
        device,
        num_heads,
        head_dim,
        max_query_tile=max_query_tile,
    )


def _h64_candidate_variant_serves(page_size: int) -> bool:
    """Whether the H64 candidate variant publishes a single-pool schedule.

    The variant reads one page per compute tile, so smaller pages keep the
    pooled layout.
    """
    return page_size == _COMPUTE_BLOCK_KV


def _h4_pooled_candidate_serves(
    num_heads: int,
    head_dim: int,
    query_tile: int,
    page_size: int,
    top_k: int,
    long_query: bool,
) -> bool:
    """Use warp-ranked banks for the qualified H4 long-query shape."""
    return (
        long_query
        and num_heads == 4
        and head_dim == 128
        and query_tile == 16
        and page_size == _COMPUTE_BLOCK_KV
        and _candidate_target(top_k) == _MAX_TARGET_CANDIDATES
    )


def _select_candidate_schedule(
    device: torch.device,
    num_heads: int,
    head_dim: int,
    q_lengths: list[int],
    kv_lengths: list[int],
    *,
    num_sms: int,
    top_k: int,
    q1_owner_pool: bool | None = None,
    page_size: int = _COMPUTE_BLOCK_KV,
) -> _CandidateSchedule:
    """Choose scheduling, pooling eligible Q1 owners unless explicitly disabled.

    Single-pool H64 prefill on compute-tile pages publishes through the
    specialized H64 candidate variant.
    """
    tiling = _select_request_tiling(device, num_heads, head_dim, q_lengths)
    if num_heads == 64 and head_dim == 128 and max(q_lengths, default=0) > 4:
        # Retain both KV streams, trading UMMA double buffering for Q4 reuse.
        # SAMPLE and both repair plans keep their original request tiling.
        tiling = _validate_specialization(
            device, num_heads, head_dim, max_query_tile=4, num_umma_stages=1
        )
        if tiling.next_n != 4:
            raise ValueError("H64 Q4 candidate exceeds the device SMEM budget")
    split_kv = _select_candidate_split_kv(
        q_lengths,
        kv_lengths,
        query_tile=tiling.next_n,
        num_sms=num_sms,
    )
    long_query = max(q_lengths, default=0) > tiling.next_n
    _, sample_count_segments = _select_candidate_publication(
        split_kv, lane_local=long_query
    )
    # H8/H4 start with lane-local publication; eligible H4 prefill is pooled
    # below. H64 split>1 keeps vector banks.
    warp_ranked_prefill = (
        long_query and num_heads == 64 and head_dim == 128 and split_kv <= 4
    )
    publication, count_segments = _select_candidate_publication(
        split_kv,
        lane_local=long_query and not warp_ranked_prefill,
    )
    candidate_capacity = _CANDIDATE_CAPACITY
    if (
        publication == "lane_local"
        and _candidate_target(top_k) == _MAX_TARGET_CANDIDATES
    ):
        # A lane-bank overflow sets the row sentinel before total capacity is
        # reached. Large targets need wider banks; a later H4 pooling override
        # retains this repair-wide arena. Smaller targets use narrower TopK.
        candidate_capacity = _REPAIR_CAPACITY
    if _h4_pooled_candidate_serves(
        num_heads, head_dim, tiling.next_n, page_size, top_k, long_query
    ):
        # A warp prefix compacts eligible values into four shared stripes,
        # rather than overflowing one of the 32 * split lane-local quotas.
        # Preserve the independently selected SAMPLE topology above.
        split_kv = 1
        publication, count_segments = _WARP_STRIPED_PUBLICATION, 4
    h64_candidate = False
    if warp_ranked_prefill and _h64_single_pool_topology(
        num_heads,
        head_dim,
        tiling.next_n,
        tiling.num_umma_stages,
        split_kv,
        1,
        _CANDIDATE_CAPACITY,
    ):
        # A single warp owns the row, including both streams and all vectors.
        # Preserve the original sample layout selected before this override.
        publication, count_segments = _WARP_POOLED_PUBLICATION, 1
        # One warp owns the entire row, so it can use the existing repair-wide
        # backing without changing sampling or any multi-bank stride/ABI.
        candidate_capacity = _REPAIR_CAPACITY
        h64_candidate = _h64_candidate_variant_serves(page_size)
        if h64_candidate:
            # Each warp of a row's warpgroup owns one of four equal segments of
            # the same repair-wide backing.
            publication, count_segments = _WARP_STRIPED_PUBLICATION, 4
    q1_pool_eligible = all(
        length in (0, 1) for length in q_lengths
    ) and _q1_owner_pool_topology(
        num_heads,
        head_dim,
        tiling.next_n,
        tiling.num_umma_stages,
        split_kv,
        split_kv,
        candidate_capacity,
    )
    if q1_owner_pool is True and not q1_pool_eligible:
        raise ValueError(
            "Q1 owner pooling requires H8/H64 D128 Q1 UMMA2 split<=8 capacity 6144"
        )
    if q1_owner_pool is not False and q1_pool_eligible:
        # SAMPLE retains the layout chosen above. Only CANDIDATE publication
        # uses one cursor per real owner instead of four vector stripes.
        publication, count_segments = _WARP_POOLED_PUBLICATION, split_kv
    return _CandidateSchedule(
        query_tile=tiling.next_n,
        logical_n=tiling.next_n * num_heads,
        num_epi_subtiles=tiling.num_epi_subtiles,
        num_umma_stages=tiling.num_umma_stages,
        split_kv=split_kv,
        publication=publication,
        count_segments=count_segments,
        sample_count_segments=sample_count_segments,
        candidate_capacity=candidate_capacity,
        h64_candidate=h64_candidate,
    )


def _candidate_schedule_matches_prepared(
    schedule: _CandidateSchedule,
    prepared: _PreparedFP8MQASelectiveLogits,
) -> bool:
    """Check that preparation preserved every compiled topology knob."""
    return (
        schedule.query_tile == prepared.query_tile
        and schedule.logical_n == prepared.query_tile * prepared.num_heads
        and schedule.num_epi_subtiles == prepared.num_epi_subtiles
        and schedule.num_umma_stages == prepared.num_umma_stages
        and schedule.split_kv == prepared.split_kv
    )


def _validate_prepared(
    prepared: _PreparedFP8MQASelectiveLogits,
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Validate the address-stable metadata ABI before entering CuTe."""
    _validate_math_mode(prepared.math_mode)
    max_tiles = prepared.tiles
    metadata = (
        prepared.context_lens,
        prepared.tile_meta,
        prepared.schedule_meta,
        prepared.schedule_prefix,
    )
    if (
        prepared.rows != q.shape[0]
        or prepared.num_phys_blocks != kv_fused.shape[0]
        or prepared.page_size != kv_fused.shape[1]
        or prepared.batches != block_table.shape[0]
    ):
        raise ValueError("prepared metadata is stale for the supplied q/KV layout")
    _validate_selective_device(q.device)
    if (
        prepared.num_sms
        > torch.cuda.get_device_properties(q.device).multi_processor_count
    ):
        raise ValueError("prepared metadata is stale for the current CUDA device")
    if any(
        tensor.device != q.device or not tensor.is_contiguous() for tensor in metadata
    ):
        raise ValueError("prepared metadata must be contiguous on the input device")
    if (
        prepared.context_lens.dtype != torch.int32
        or prepared.context_lens.shape != (max_tiles,)
        or prepared.tile_meta.dtype != torch.int32
        or prepared.tile_meta.shape
        != (max_tiles, _METADATA_HEADER_COLUMNS + 2 * prepared.query_tile)
        or prepared.schedule_meta.dtype != torch.int32
        or prepared.schedule_meta.shape != (prepared.num_sms + 1, 2)
        or prepared.schedule_prefix.dtype != torch.int32
        or prepared.schedule_prefix.shape != (max_tiles,)
    ):
        raise ValueError("prepared metadata has an invalid selective-logits ABI")
    _require_aligned(metadata)
    return metadata


def _device_ordinal(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _current_stream(device: torch.device):
    device_index = _device_ordinal(device)
    raw_stream = (
        _raw_stream(device_index)
        if _raw_stream is not None
        else torch.cuda.current_stream(device_index).cuda_stream
    )
    key = (device_index, raw_stream)
    stream = _STREAM_CACHE.get(key)
    if stream is None:
        stream = cuda.CUstream(raw_stream)
        _STREAM_CACHE[key] = stream
    return stream


def _compile_for_device(device: torch.device, kernel, *args):
    """Keep build context and generated architecture aligned with the tensor key."""
    with torch.cuda.device(device):
        major, minor = torch.cuda.get_device_capability(device)
        options = f"--gpu-arch sm_{major}{minor}a"
        max_registers = getattr(kernel, "ptxas_max_registers", None)
        if max_registers is not None:
            options += f" --ptxas-options=--maxrregcount={max_registers}"
        compiled = cute.compile(kernel, *args, options=options)
        # Compilation alone leaves CuTe's CUDA library/function resolution lazy.
        # Resolve it before publishing the launch into a graph-capture cache.
        compiled.to(_device_ordinal(device))
        return compiled


def _require_cute() -> None:
    if not _CUTE_DSL_AVAILABLE:
        raise RuntimeError("selective FP8 logits requires nvidia-cutlass-dsl")


def _validate_prefixes(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> None:
    tensors = (q, kv_fused, weights, cu_q, cu_kv, block_table)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("selective-logits inputs must be torch tensors")
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError("q must be float8_e4m3fn")
    if kv_fused.dtype != torch.uint8:
        raise ValueError("kv_fused must be uint8")
    if weights.dtype != torch.float32:
        raise ValueError("weights must be float32")
    if tuple(q.shape[1:]) != (num_heads, head_dim) or q.dim() != 3:
        raise ValueError(f"q must have shape [Q, {num_heads}, {head_dim}]")
    if (
        kv_fused.dim() != 4
        or kv_fused.shape[2] != 1
        or kv_fused.shape[3] != head_dim + 4
    ):
        raise ValueError(
            f"kv_fused must have shape [num_blocks, page_size, 1, {head_dim + 4}]"
        )
    page_size = kv_fused.shape[1]
    if (
        page_size <= 0
        or _COMPUTE_BLOCK_KV % page_size
        or _COMPUTE_BLOCK_KV // page_size > _MAX_BLOCKS_PER_MMA
    ):
        raise ValueError("page_size must be one of 32, 64, or 128")
    if tuple(weights.shape) != (q.shape[0], num_heads):
        raise ValueError(f"weights must have shape [Q, {num_heads}]")
    if cu_q.dtype != torch.int32 or cu_kv.dtype != torch.int32:
        raise ValueError("cu_q and cu_kv must be int32")
    if cu_q.dim() != 1 or cu_q.shape != cu_kv.shape or cu_q.shape[0] < 2:
        raise ValueError("cu_q and cu_kv must have matching shape [B+1]")
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != cu_q.shape[0] - 1
    ):
        raise ValueError("block_table must be int32 [B, max_pages_per_request]")
    device = q.device
    if not q.is_cuda or any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError(
            "all selective-logits inputs must be CUDA tensors on one device"
        )
    _validate_selective_device(device)
    kv_flat = kv_fused.flatten(1)
    if not (
        q.is_contiguous()
        and weights.is_contiguous()
        and cu_q.is_contiguous()
        and cu_kv.is_contiguous()
        and block_table.is_contiguous()
        and kv_flat.data_ptr() == kv_fused.data_ptr()
        and kv_flat.stride(1) == 1
    ):
        raise ValueError(
            "selective-logits inputs require contiguous q/weights/prefixes/page table "
            "and contiguous bytes within each KV page"
        )
    _require_aligned(tensors)


def _validate_device_metadata_values(
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    *,
    rows: int,
    page_size: int,
    num_phys_blocks: int,
    block_table: torch.Tensor | None = None,
    block_table_columns: int = 0,
    causal_bounds: bool,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    row_end_base: int = 0,
) -> None:
    """Perform opt-in synchronized checks for trusted device metadata."""
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") in ("0", ""):
        return
    if torch.cuda.is_current_stream_capturing():
        return

    q_prefix = cu_q.tolist()
    kv_prefix = cu_kv.tolist()
    if q_prefix[0] != 0 or kv_prefix[0] != 0:
        raise ValueError("cu_q and cu_kv must start at zero")
    if q_prefix[-1] != rows:
        raise ValueError(f"cu_q[-1] must equal q.shape[0] ({rows})")
    if any(left > right for left, right in zip(q_prefix, q_prefix[1:], strict=False)):
        raise ValueError("cu_q must be monotonically nondecreasing")
    if any(left > right for left, right in zip(kv_prefix, kv_prefix[1:], strict=False)):
        raise ValueError("cu_kv must be monotonically nondecreasing")

    required_columns = 0
    for batch, (q_begin, q_end, kv_begin, kv_end) in enumerate(
        zip(q_prefix, q_prefix[1:], kv_prefix, kv_prefix[1:], strict=False)
    ):
        q_len = q_end - q_begin
        kv_len = kv_end - kv_begin
        if causal_bounds and q_len > kv_len:
            raise ValueError(
                f"causal request {batch} has q length {q_len} greater than "
                f"kv length {kv_len}"
            )
        need = math.ceil(kv_len / page_size)
        required_columns = max(required_columns, need)

    available_columns = (
        block_table.shape[1] if block_table is not None else block_table_columns
    )
    if available_columns < required_columns:
        raise ValueError(
            f"block_table has {available_columns} columns but selective logits "
            f"requires at least {required_columns} for these KV prefixes"
        )

    if block_table is not None and required_columns:
        table = block_table[:, :required_columns].tolist()
        for batch, (kv_begin, kv_end) in enumerate(
            zip(kv_prefix, kv_prefix[1:], strict=False)
        ):
            need = math.ceil((kv_end - kv_begin) / page_size)
            if any(not 0 <= index < num_phys_blocks for index in table[batch][:need]):
                raise ValueError(
                    f"block_table row {batch} contains a physical block index "
                    f"outside [0, {num_phys_blocks})"
                )

    if row_starts is None and row_ends is None:
        return
    assert row_starts is not None and row_ends is not None
    starts = row_starts.tolist()
    ends = row_ends.tolist()
    for batch, (q_begin, q_end, kv_begin, kv_end) in enumerate(
        zip(q_prefix, q_prefix[1:], kv_prefix, kv_prefix[1:], strict=False)
    ):
        kv_len = kv_end - kv_begin
        for row in range(q_begin, q_end):
            start = starts[row]
            end = ends[row]
            if end == 0:
                if start != 0:
                    raise ValueError(
                        f"inactive explicit row {row} must use start=end=0"
                    )
                continue
            effective_end = end + row_end_base
            if start < 0 or start > effective_end or effective_end > kv_len:
                raise ValueError(
                    f"explicit row {row} must satisfy 0 <= start <= "
                    f"end + row_end_base <= request {batch} KV length ({kv_len})"
                )


def _launch_metadata(
    prepared: _PreparedFP8MQASelectiveLogits,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    *,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    _compile_only: bool = False,
) -> None:
    """Refresh caller-owned metadata with no allocation or host readback."""
    device = prepared.context_lens.device
    if (
        cu_q.dtype != torch.int32
        or cu_kv.dtype != torch.int32
        or cu_q.shape != (prepared.batches + 1,)
        or cu_kv.shape != cu_q.shape
        or cu_q.device != device
        or cu_kv.device != device
        or not cu_q.is_contiguous()
        or not cu_kv.is_contiguous()
    ):
        raise ValueError(
            "cu_q and cu_kv must be contiguous int32 [prepared.batches+1] "
            "on the prepared device"
        )
    explicit_bounds = row_starts is not None or row_ends is not None
    if active_rows is not None:
        if (
            not explicit_bounds
            or prepared.batches != 1
            or active_rows.shape != (1,)
            or active_rows.dtype != torch.int32
            or active_rows.device != device
            or not active_rows.is_contiguous()
        ):
            raise ValueError(
                "active_rows requires single-request explicit bounds and int32 [1]"
            )
        _require_aligned((active_rows,))
    if explicit_bounds == prepared.causal_bounds:
        expected = (
            "explicit row bounds" if not prepared.causal_bounds else "causal bounds"
        )
        raise ValueError(f"prepared metadata requires {expected}")
    if explicit_bounds:
        if row_starts is None or row_ends is None:
            raise ValueError("row_starts and row_ends must be provided together")
        if (
            row_starts.dtype != torch.int32
            or row_ends.dtype != torch.int32
            or row_starts.shape != (prepared.rows,)
            or row_ends.shape != (prepared.rows,)
            or row_starts.device != device
            or row_ends.device != device
            or not row_starts.is_contiguous()
            or not row_ends.is_contiguous()
        ):
            raise ValueError(
                "row_starts and row_ends must be contiguous int32 [Q] on q.device"
            )
        _require_aligned((cu_q, cu_kv, row_starts, row_ends))
    else:
        _require_aligned((cu_q, cu_kv))

    if not prepared.capacity_bounds and not _compile_only:
        _validate_device_metadata_values(
            cu_q,
            cu_kv,
            rows=prepared.rows,
            page_size=prepared.page_size,
            num_phys_blocks=prepared.num_phys_blocks,
            block_table_columns=prepared.block_table_columns,
            causal_bounds=prepared.causal_bounds,
            row_starts=row_starts,
            row_ends=row_ends,
            row_end_base=prepared.row_end_base,
        )

    base_args = (
        from_dlpack(cu_q, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(cu_kv, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
    )
    output_args = (
        from_dlpack(prepared.context_lens, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(prepared.tile_meta, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(prepared.schedule_meta, assumed_align=16),
        from_dlpack(
            prepared.schedule_prefix, assumed_align=16
        ).mark_compact_shape_dynamic(mode=0, divisibility=1),
        cutlass.Int32(prepared.batches),
    )
    args: tuple[Any, ...]
    if explicit_bounds:
        args = (
            *base_args,
            from_dlpack(row_starts, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(prepared.row_end_base),
            from_dlpack(active_rows, assumed_align=16)
            if active_rows is not None
            else None,
            *output_args,
        )
    else:
        args = (*base_args, *output_args)
    stream = _current_stream(device)
    capability = torch.cuda.get_device_capability(device)
    key = (
        *capability,
        _device_ordinal(device),
        prepared.num_sms,
        prepared.query_tile,
        prepared.kv_chunk_size,
        prepared.split_kv,
        int(explicit_bounds),
        active_rows is not None,
    )
    compiled = _METADATA_COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "metadata kernels must be compiled before CUDA graph capture"
            )
        from .kernels.selective_logits_metadata import (
            SelectiveLogitsExplicitMetadataScheduleKernel,
            SelectiveLogitsMetadataScheduleKernel,
        )

        metadata_kernel = (
            SelectiveLogitsExplicitMetadataScheduleKernel(
                prepared.num_sms,
                query_tile=prepared.query_tile,
                kv_chunk_size=prepared.kv_chunk_size,
                split_kv=prepared.split_kv,
                paged=True,
            )
            if explicit_bounds
            else SelectiveLogitsMetadataScheduleKernel(
                prepared.num_sms,
                query_tile=prepared.query_tile,
                kv_chunk_size=prepared.kv_chunk_size,
                split_kv=prepared.split_kv,
                paged=True,
            )
        )
        compiled = _compile_for_device(device, metadata_kernel, *args, stream)
        _METADATA_COMPILED[key] = compiled
    if not _compile_only:
        if active_rows is not None:
            # Keep this sticky: a previously captured refresh may replay later.
            prepared._metadata_modes.add("active_rows")
        compiled(*args, stream)


def _launch_compact_metadata(
    prepared: _PreparedFP8MQASelectiveLogits,
    cu_q: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    row_flags: torch.Tensor,
) -> None:
    """Build a fixed-capacity schedule containing only flagged query tiles."""
    if prepared.causal_bounds:
        raise ValueError("compact repair metadata requires explicit preparation")
    args = (
        from_dlpack(cu_q, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(row_starts, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(row_ends, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(row_flags, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(prepared.context_lens, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(prepared.tile_meta, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(prepared.schedule_meta, assumed_align=16),
        from_dlpack(
            prepared.schedule_prefix, assumed_align=16
        ).mark_compact_shape_dynamic(mode=0, divisibility=1),
        cutlass.Int32(prepared.batches),
        cutlass.Int32(prepared.rows),
    )
    stream = _current_stream(prepared.context_lens.device)
    capability = torch.cuda.get_device_capability(prepared.context_lens.device)
    key = (
        *capability,
        _device_ordinal(prepared.context_lens.device),
        prepared.num_sms,
        prepared.kv_chunk_size,
        prepared.split_kv,
    )
    key += (prepared.query_tile,)
    compiled = _COMPACT_METADATA_COMPILED.get(key)
    if compiled is None:
        from .kernels.selective_logits_metadata import (
            SelectiveLogitsCompactMetadataScheduleKernel,
        )

        compiled = _compile_for_device(
            cu_q.device,
            SelectiveLogitsCompactMetadataScheduleKernel(
                prepared.num_sms,
                query_tile=prepared.query_tile,
                kv_chunk_size=prepared.kv_chunk_size,
                split_kv=prepared.split_kv,
                paged=True,
            ),
            *args,
            stream,
        )
        _COMPACT_METADATA_COMPILED[key] = compiled
    compiled(*args, stream)


def _prepare_fp8_mqa_selective_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    block_table: torch.Tensor,
    *,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    row_end_base: int = 0,
    num_sms: int | None = None,
    kv_chunk_size: int = 0,
    split_kv: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    query_tile: int | None = None,
    num_umma_stages: int | None = None,
    math_mode: str = "relu_strict",
    _capacity_bounds: bool = False,
) -> _PreparedFP8MQASelectiveLogits:
    """Allocate and build request-owned selective scheduling metadata.

    ``kv_fused`` and ``block_table`` use the same paged FP8 representation and
    natural block-table extent as :func:`fp8_paged_mqa_logits`; table entries may
    map logical pages to arbitrary valid physical blocks. ``num_heads`` and
    ``head_dim`` define the prepared problem shape and become compile-time
    kernel configuration. The request tile is selected automatically from those
    constraints. Optional bounds are trusted device metadata in request-local
    coordinates; callers maintain
    ``0 <= start <= end <= request_kv_length``. For compact rectangular repair
    rows, ``row_end_base`` is added to every nonzero end; zero remains the
    inactive-row sentinel. Supplying explicit bounds is required for compact
    sampled KV and arbitrary compact repair rows.
    ``num_sms`` may cap an exceptional-path launch, and
    ``kv_chunk_size`` partitions each nonempty query tile into fixed-size work
    items for modes whose global atomics make that ownership safe. Candidate
    publication instead uses split-K-like ``split_kv`` ownership. Each active
    split receives a pair-aligned KV range. Splits through eight use four
    warp-ranked publisher stripes per owner; higher splits pool those stripes
    into one publisher segment per owner. Short ranges may activate fewer
    splits than requested.
    Use :func:`_refresh_fp8_mqa_selective_logits` inside replay when these device-side
    prefixes or bounds change.

    Args:
        q: Contiguous FP8 E4M3 query tensor ``[Q, num_heads, head_dim]``.
        kv_fused: Contiguous paged byte tensor
            ``[num_blocks, page_size, 1, head_dim + 4]``. Each page stores FP8
            keys followed by one FP32 scale per token.
        weights: Contiguous FP32 per-head weights ``[Q, num_heads]``.
        cu_q: Contiguous int32 CUDA query prefix sums ``[B + 1]``.
        cu_kv: Contiguous int32 CUDA logical KV prefix sums ``[B + 1]``.
        block_table: Contiguous int32 CUDA physical page indices
            ``[B, max_pages]``.
        row_starts: Optional contiguous int32 request-local inclusive starts
            ``[Q]``. Must be provided together with ``row_ends``.
        row_ends: Optional contiguous int32 request-local exclusive ends
            ``[Q]``. Zero denotes an inactive row.
        row_end_base: Value added to every nonzero explicit row end.
        num_sms: Optional persistent CTA count. Defaults to the device SM count.
        kv_chunk_size: Fixed token extent for atomic output modes. Zero disables
            chunking; nonzero values must be positive multiples of 256.
        split_kv: Candidate-mode split count. Supported values are 1, 2, 4,
            8, 16, 32, and 64.
        num_heads: Query head count, specialized at compile time.
        head_dim: Head dimension, specialized at compile time and divisible by
            32.
        query_tile: Optional internal request-tile specialization. Public plans
            select this from their per-request query lengths.
        num_umma_stages: Optional exact internal UMMA-stage specialization.

    Returns:
        Prepared metadata whose storage must remain alive while launches or
        captured graphs use it.

    Note:
        Prefix values and page indices live on the device and are trusted by
        default. ``cu_q`` and ``cu_kv`` must start at zero, be monotonic, and
        end at ``Q`` and the logical KV total respectively. Causal requests
        require ``q_len <= kv_len``. The page table must contain one entry per
        logical page, and every entry the kernel
        may read must be in ``[0, num_blocks)``. Explicit rows require
        ``0 <= start <= end + row_end_base <= request_kv_len``; inactive rows
        use ``start == end == 0``. Set ``FLASHINFER_VALIDATE_INPUTS=1`` for
        synchronized development-time checks outside CUDA graph capture.
    """
    math_mode = _validate_math_mode(math_mode)
    _require_cute()
    if not isinstance(row_end_base, int) or row_end_base < 0:
        raise ValueError("row_end_base must be a nonnegative integer")
    if row_end_base and row_starts is None and row_ends is None:
        raise ValueError("row_end_base requires explicit row bounds")
    _validate_prefixes(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    if not _capacity_bounds:
        _validate_device_metadata_values(
            cu_q,
            cu_kv,
            rows=q.shape[0],
            page_size=kv_fused.shape[1],
            num_phys_blocks=kv_fused.shape[0],
            block_table=block_table,
            causal_bounds=row_starts is None and row_ends is None,
            row_starts=row_starts,
            row_ends=row_ends,
            row_end_base=row_end_base,
        )
    tiling = _validate_specialization(
        q.device,
        num_heads,
        head_dim,
        max_query_tile=query_tile if query_tile is not None else _MAX_QUERY_TILE,
        num_umma_stages=num_umma_stages,
    )
    if query_tile is not None and tiling.next_n != query_tile:
        raise ValueError(
            f"query_tile {query_tile} is not legal for H={num_heads}, D={head_dim}"
        )
    rows = q.shape[0]
    batches = cu_q.shape[0] - 1
    if rows <= 0 or kv_fused.shape[0] <= 0:
        raise ValueError("selective-logits q and kv lengths must be positive")
    query_tile = tiling.next_n
    max_tiles = (rows + query_tile - 1) // query_tile + batches - 1
    device_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    if num_sms is None:
        num_sms = device_sms
    if not isinstance(num_sms, int) or not 0 < num_sms <= device_sms:
        raise ValueError(f"num_sms must be in [1, {device_sms}]")
    # Atomic scorer cursors count pairs of 128-token math blocks.
    if type(kv_chunk_size) is not int or kv_chunk_size not in (0, 256):
        raise ValueError("kv_chunk_size must be zero or the 256-token scorer work unit")
    if split_kv not in _SPLIT_KV_CHOICES:
        raise ValueError("split_kv must be 1, 2, 4, 8, 16, 32, or 64")
    if kv_chunk_size and split_kv != 1:
        raise ValueError("kv_chunk_size and split_kv > 1 are mutually exclusive")
    prepared = _PreparedFP8MQASelectiveLogits(
        context_lens=torch.empty(max_tiles, dtype=torch.int32, device=q.device),
        tile_meta=torch.zeros(
            (max_tiles, _METADATA_HEADER_COLUMNS + 2 * query_tile),
            dtype=torch.int32,
            device=q.device,
        ),
        schedule_meta=torch.empty((num_sms + 1, 2), dtype=torch.int32, device=q.device),
        schedule_prefix=torch.empty(max_tiles, dtype=torch.int32, device=q.device),
        rows=rows,
        num_phys_blocks=kv_fused.shape[0],
        page_size=kv_fused.shape[1],
        query_tile=query_tile,
        num_epi_subtiles=tiling.num_epi_subtiles,
        tiles=max_tiles,
        batches=batches,
        num_sms=num_sms,
        kv_chunk_size=kv_chunk_size,
        split_kv=split_kv,
        causal_bounds=row_starts is None and row_ends is None,
        row_end_base=row_end_base,
        num_heads=num_heads,
        head_dim=head_dim,
        num_umma_stages=tiling.num_umma_stages,
        math_mode=math_mode,
        block_table_columns=block_table.shape[1],
        capacity_bounds=_capacity_bounds,
    )
    _launch_metadata(
        prepared,
        cu_q,
        cu_kv,
        row_starts=row_starts,
        row_ends=row_ends,
    )
    return prepared


def _refresh_fp8_mqa_selective_logits(
    prepared: _PreparedFP8MQASelectiveLogits,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    *,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
) -> None:
    """Refresh address-stable request metadata on the current CUDA stream.

    Args:
        prepared: Object returned by :func:`_prepare_fp8_mqa_selective_logits`.
        cu_q: Updated contiguous int32 CUDA query prefixes ``[B + 1]``.
        cu_kv: Updated contiguous int32 CUDA logical KV prefixes ``[B + 1]``.
        row_starts: Updated explicit inclusive starts ``[Q]``, when the
            prepared object uses explicit bounds.
        row_ends: Updated explicit exclusive ends ``[Q]``, when the prepared
            object uses explicit bounds.
        active_rows: Optional device count of the initialized compact prefix
            for a single-request explicit plan. The queue producer must have
            completed before this launch.

    The prepared object fixes tensor shapes, page-table width, bounds mode,
    and specialization. Refresh changes values only and is safe to capture in
    a CUDA graph; synchronized opt-in input checks are skipped during capture.
    """
    _require_cute()
    _launch_metadata(
        prepared,
        cu_q,
        cu_kv,
        row_starts=row_starts,
        row_ends=row_ends,
        active_rows=active_rows,
    )


def _runtime_args(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    values: torch.Tensor,
    index_counts: torch.Tensor,
    prepared: _PreparedFP8MQASelectiveLogits,
    policy_words: torch.Tensor,
    sample_page_indices: torch.Tensor | None,
    sparse_sample_metadata: tuple[torch.Tensor, ...] | None = None,
) -> Tuple[Any, ...]:
    q_view = q.reshape(q.shape[0] * prepared.num_heads, prepared.head_dim).unsqueeze(-1)
    kv_flat = kv_fused.flatten(1)
    return (
        from_dlpack(kv_flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(
            q_view.view(torch.uint8), assumed_align=16
        ).mark_compact_shape_dynamic(
            mode=0,
            stride_order=q_view.dim_order(),
            divisibility=prepared.num_heads,
        ),
        from_dlpack(weights, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(values, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(block_table, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(index_counts, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(prepared.context_lens, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(prepared.schedule_meta, assumed_align=16),
        from_dlpack(prepared.tile_meta, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(policy_words, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        tuple(
            from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            )
            for tensor in sparse_sample_metadata
        )
        if sparse_sample_metadata is not None
        else (
            from_dlpack(sample_page_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            if sample_page_indices is not None
            else None
        ),
    )


def _fp8_mqa_selective_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    prepared: _PreparedFP8MQASelectiveLogits,
    *,
    mode: _SelectiveLogitsMode,
    capacity: int,
    values: torch.Tensor,
    index_counts: torch.Tensor,
    policy_values: torch.Tensor | None = None,
    sample_page_indices: torch.Tensor | None = None,
    radix_shift: int = 0,
    _compile_only: bool = False,
    _clear_counts: bool = True,
    _candidate_counts_precleared: bool = False,
    _h64_candidate: bool = False,
    _sparse_sample_load_warps: int = 0,
    _sparse_sample_group_rows: int = 128,
    _sparse_sample_metadata: tuple[torch.Tensor, ...] | None = None,
) -> None:
    """Run one explicitly selected logits phase into caller-owned storage.

    ``kv_fused`` and ``block_table`` follow :func:`fp8_paged_mqa_logits`, and
    this function never dispatches to that full-logits API based on row length.

    Candidate/repair use packed int64 ``values[Q, capacity]``. A one-column
    count tensor publishes directly. The cooperative path uses four striped
    segments per split-KV owner through split 8, then one pooled segment per
    owner for split 16/32/64. Lane-local publication is also supported
    through the split-1 Qx33, split-2 Qx65, and split-4 Qx129 layouts. All
    segmented forms require
    :func:`_finalize_fp8_mqa_packed_candidates` to compact values and publish
    either the exact count or the packed ``capacity + 1`` overflow sentinel in
    column zero. Radix uses ``index_counts[Q, 256]``. Sample uses float32
    ``values[Q, capacity]``.
    Clearing outputs is stream ordered and therefore participates in replay.
    ``EXACT`` writes int32 indices into two equal row banks, ``> T`` first and
    ``== T`` second, with one logical count per bank row; each bank must
    contain at least the padded query rows. The caller
    may disable histogram clearing only after a preceding selector has reset
    every active histogram bin on the same stream.

    Args:
        q: Contiguous FP8 E4M3 queries matching ``prepared``.
        kv_fused: Fused paged FP8 KV storage matching ``prepared``.
        weights: Contiguous FP32 per-head weights ``[Q, num_heads]``.
        block_table: Contiguous int32 physical-page table compatible with the
            prefixes used to refresh ``prepared``.
        prepared: Address-stable metadata and compile-time specialization.
        mode: Explicit output phase; this API performs no length-based routing.
        capacity: Number of elements available in each output row.
        values: Caller-owned output storage with dtype/layout determined by
            ``mode``.
        index_counts: Caller-owned count or histogram storage with the layout
            described above.
        policy_values: Required threshold or radix-prefix words for every mode
            except ``SAMPLE``.
        sample_page_indices: Logical page positions for internal SAMPLE lookup.
            Only valid in ``SAMPLE`` mode; block_table then holds original pages.
        radix_shift: Byte shift for ``RADIX_HISTOGRAM``; one of 0, 8, 16, 24.
        _clear_counts: Clear counts before scoring. Only radix histograms may
            opt out when their reset is owned by the preceding selector.
        _candidate_counts_precleared: CANDIDATE-only declaration that the caller
            cleared all count words earlier on the same stream.

    This stateful prepare/run interface intentionally owns its compile cache
    and retained DLPack bindings instead of using the stateless
    ``flashinfer_api`` wrapper.
    """
    _require_cute()
    if _sparse_sample_load_warps and not _sparse_sample_gather4_serves(
        kv_fused.shape[0],
        kv_fused.stride(0),
        prepared.page_size,
        prepared.head_dim,
    ):
        raise ValueError(
            "gather4 SAMPLE requires 32/64/128-key pages, "
            "a 128-byte-aligned page pitch and int32 row coordinates"
        )
    if _sparse_sample_load_warps and (
        mode != _SelectiveLogitsMode.SAMPLE
        or _sparse_sample_load_warps not in (4, 8)
        or _sparse_sample_group_rows <= 0
        or _sparse_sample_group_rows % prepared.query_tile
        or (sample_page_indices is None and _sparse_sample_metadata is None)
    ):
        raise ValueError("sparse SAMPLE requires 4/8 loaders and complete index groups")
    if _h64_candidate and "active_rows" in prepared._metadata_modes:
        raise ValueError(
            "the H64 candidate variant does not support active_rows metadata"
        )
    if not _clear_counts and mode != _SelectiveLogitsMode.RADIX_HISTOGRAM:
        raise ValueError("only radix histograms support caller-owned count reset")
    if _candidate_counts_precleared and mode != _SelectiveLogitsMode.CANDIDATE:
        raise ValueError("precleared candidate counts require CANDIDATE mode")
    launch_tensors: tuple[torch.Tensor, ...] = (
        q,
        kv_fused,
        weights,
        block_table,
        values,
        index_counts,
    )
    if _sparse_sample_metadata is not None:
        if (
            not _sparse_sample_load_warps
            or sample_page_indices is not None
            or len(_sparse_sample_metadata) != 4
            or any(
                tensor.dtype != torch.int32
                or tensor.device != q.device
                or tensor.ndim != 1
                or not tensor.is_contiguous()
                for tensor in _sparse_sample_metadata
            )
            or any(t.numel() < prepared.rows for t in _sparse_sample_metadata[:2])
            or any(
                t.numel() != prepared.batches + 1 for t in _sparse_sample_metadata[2:]
            )
        ):
            raise ValueError(
                "on-the-fly SAMPLE requires row ends/counts and request prefixes"
            )
        _require_aligned(_sparse_sample_metadata)
        launch_tensors = (*launch_tensors, *_sparse_sample_metadata)
    if policy_values is not None:
        launch_tensors = (*launch_tensors, policy_values)
    if sample_page_indices is not None:
        if (
            mode != _SelectiveLogitsMode.SAMPLE
            or sample_page_indices.dtype != torch.int64
            or sample_page_indices.device != q.device
            or sample_page_indices.ndim != 2
            or sample_page_indices.shape[0]
            != (
                (prepared.rows + _sparse_sample_group_rows - 1)
                // _sparse_sample_group_rows
                if _sparse_sample_load_warps
                else prepared.batches
            )
            or sample_page_indices.shape[1] == 0
            or (_sparse_sample_load_warps and sample_page_indices.shape[1] < capacity)
            or not sample_page_indices.is_contiguous()
        ):
            raise ValueError(
                "internal SAMPLE requires contiguous int64 logical pages [B, P]"
            )
        _require_aligned((sample_page_indices,))
        launch_tensors = (*launch_tensors, sample_page_indices)
    if isinstance(prepared, _PreparedFP8MQASelectiveLogits):
        # Cached CuTe arguments retain these metadata addresses alongside the
        # ordinary inputs and outputs. Include their structural identity so a
        # caller-visible storage rebind rebuilds the DLPack launch binding.
        launch_tensors = (
            *launch_tensors,
            prepared.context_lens,
            prepared.tile_meta,
            prepared.schedule_meta,
        )
    cached_required_words = 0
    if isinstance(index_counts, torch.Tensor) and index_counts.dim() == 2:
        if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM:
            cached_required_words = 256
        elif mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR):
            cached_required_words = index_counts.shape[1]
        else:
            cached_required_words = 1
    cache_key: Tuple[Any, ...] = (
        mode,
        capacity,
        radix_shift,
        cached_required_words,
        sample_page_indices is not None,
        _h64_candidate,
        _sparse_sample_load_warps,
        _sparse_sample_group_rows,
        _sparse_sample_metadata is not None,
        prepared.math_mode
        if isinstance(prepared, _PreparedFP8MQASelectiveLogits)
        else None,
    )
    if mode in (
        _SelectiveLogitsMode.SAMPLE,
        _SelectiveLogitsMode.CANDIDATE,
        _SelectiveLogitsMode.REPAIR,
        _SelectiveLogitsMode.RADIX_HISTOGRAM,
        _SelectiveLogitsMode.EXACT,
    ):
        cache_key = (*cache_key, _shared_engine_source_token())
    if (
        isinstance(mode, _SelectiveLogitsMode)
        and isinstance(prepared, _PreparedFP8MQASelectiveLogits)
        and all(isinstance(tensor, torch.Tensor) for tensor in launch_tensors)
    ):
        identity = _selective_logits_launch_identity(
            launch_tensors,
            capacity,
        )
        cached = prepared._launches.get(cache_key)
        if cached is not None and cached.identity == identity:
            validation_enabled = os.environ.get(
                "FLASHINFER_VALIDATE_INPUTS", "0"
            ) not in ("0", "")
            if not validation_enabled or torch.cuda.is_current_stream_capturing():
                if _compile_only:
                    return
                if (
                    cached.count_view is not None
                    and _clear_counts
                    and not _candidate_counts_precleared
                ):
                    cached.count_view.zero_()
                cached.compiled(*cached.args, _current_stream(q.device))
                return
    if q.dtype != torch.float8_e4m3fn or kv_fused.dtype != torch.uint8:
        raise ValueError("q must be float8_e4m3fn and kv_fused must be uint8")
    if weights.dtype != torch.float32:
        raise ValueError("weights must be float32")
    if q.dim() != 3 or tuple(q.shape[1:]) != (
        prepared.num_heads,
        prepared.head_dim,
    ):
        raise ValueError(
            f"q must have shape [Q, {prepared.num_heads}, {prepared.head_dim}]"
        )
    if (
        kv_fused.dim() != 4
        or kv_fused.shape[0] != prepared.num_phys_blocks
        or kv_fused.shape[1] != prepared.page_size
        or kv_fused.shape[2] != 1
        or kv_fused.shape[3] != prepared.head_dim + 4
    ):
        raise ValueError("kv_fused shape does not match prepared paged KV")
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != prepared.batches
        or block_table.shape[1] < prepared.block_table_columns
    ):
        raise ValueError("block_table must be int32 [B, max_pages_per_request]")
    if tuple(weights.shape) != (q.shape[0], prepared.num_heads):
        raise ValueError(f"weights must have shape [Q, {prepared.num_heads}]")
    if not q.is_cuda or any(
        tensor.device != q.device for tensor in (kv_fused, weights, block_table)
    ):
        raise ValueError(
            "q, kv_fused, weights, and block_table must share one CUDA device"
        )
    kv_flat = kv_fused.flatten(1)
    if not (
        q.is_contiguous()
        and weights.is_contiguous()
        and block_table.is_contiguous()
        and kv_flat.data_ptr() == kv_fused.data_ptr()
        and kv_flat.stride(1) == 1
    ):
        raise ValueError(
            "q, weights, block_table, and bytes within each KV page must be contiguous"
        )
    metadata = _validate_prepared(prepared, q, kv_fused, block_table)
    if (
        os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("0", "")
        and not prepared.capacity_bounds
        and not torch.cuda.is_current_stream_capturing()
        and block_table.numel()
    ):
        # Capacity chains validate active prefixes/pages on device. Their
        # planning prefixes describe storage bounds, not live map contents.
        # Only live logical pages are addressable; request padding is unspecified.
        page_counts: dict[int, int] = {}
        for length, request in zip(
            prepared.context_lens.tolist(),
            prepared.tile_meta[:, 1].tolist(),
            strict=True,
        ):
            if length <= 0:
                continue
            if not 0 <= request < prepared.batches:
                raise ValueError("prepared metadata contains an invalid request ID")
            page_counts[request] = max(
                page_counts.get(request, 0), math.ceil(length / prepared.page_size)
            )
        for request, count in page_counts.items():
            if sample_page_indices is None:
                columns = list(range(count))
            else:
                if count > sample_page_indices.shape[1]:
                    raise ValueError("sample_page_indices is smaller than live pages")
                columns = sample_page_indices[request, :count].tolist()
            if any(not 0 <= column < block_table.shape[1] for column in columns):
                raise ValueError("live page index is outside block_table columns")
            block_min, block_max = torch.aminmax(block_table[request, columns])
            if int(block_min) < 0 or int(block_max) >= prepared.num_phys_blocks:
                raise ValueError(
                    "block_table contains a physical block index outside the prepared "
                    f"range [0, {prepared.num_phys_blocks})"
                )
    if not isinstance(mode, _SelectiveLogitsMode):
        raise ValueError("mode must be a _SelectiveLogitsMode")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM and radix_shift not in (
        0,
        8,
        16,
        24,
    ):
        raise ValueError("radix_shift must be one of {0, 8, 16, 24}")
    padded_rows = (
        (prepared.rows + prepared.query_tile - 1) // prepared.query_tile
    ) * prepared.query_tile
    if mode == _SelectiveLogitsMode.SAMPLE:
        values_dtype = torch.float32
    elif mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR):
        values_dtype = torch.int64
    else:
        values_dtype = torch.int32
    if values.dtype != values_dtype or values.device != q.device:
        raise ValueError(
            f"values must be caller-owned {values_dtype} storage on q.device"
        )
    if values.dim() != 2 or values.shape[0] < padded_rows or values.shape[1] < capacity:
        raise ValueError("values storage is smaller than [padded_Q, capacity]")
    if index_counts.dtype != torch.int32 or index_counts.device != q.device:
        raise ValueError("index_counts must be caller-owned int32 storage on q.device")
    if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM:
        required_words = 256
    elif mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR):
        single_pool = (
            mode == _SelectiveLogitsMode.CANDIDATE
            and index_counts.dim() == 2
            and _h64_single_pool_topology(
                prepared.num_heads,
                prepared.head_dim,
                prepared.query_tile,
                prepared.num_umma_stages,
                prepared.split_kv,
                index_counts.shape[1] - 1,
                capacity,
            )
        )
        q1_pool = (
            mode == _SelectiveLogitsMode.CANDIDATE
            and index_counts.dim() == 2
            and _q1_owner_pool_topology(
                prepared.num_heads,
                prepared.head_dim,
                prepared.query_tile,
                prepared.num_umma_stages,
                prepared.split_kv,
                index_counts.shape[1] - 1,
                capacity,
            )
        )
        if index_counts.dim() != 2 or (
            index_counts.shape[1]
            not in (1, *tuple(segments + 1 for segments in _COUNT_SEGMENTS))
            and not single_pool
            and not q1_pool
        ):
            raise ValueError(
                "candidate counts must use one global word or "
                "4/8/12/16/32/64/128 segments, or a supported pooled-owner ABI"
            )
        required_words = index_counts.shape[1]
    else:
        required_words = 1
    if index_counts.dim() != 2 or index_counts.shape[0] < padded_rows:
        raise ValueError("index_counts has too few rows")
    if index_counts.shape[1] < required_words:
        raise ValueError("index_counts storage is too narrow for the selected mode")
    if mode == _SelectiveLogitsMode.EXACT and (
        values.shape[0] != index_counts.shape[0]
        or values.shape[0] % 2 != 0
        or values.shape[0] // 2 < padded_rows
    ):
        raise ValueError("exact emission requires two equally sized padded row banks")
    num_count_segments = (
        required_words - 1
        if mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR)
        else 0
    )
    if (
        prepared.kv_chunk_size
        and mode == _SelectiveLogitsMode.CANDIDATE
        and num_count_segments > 0
    ):
        raise ValueError(
            "segmented candidate store publication requires exclusive query tiles"
        )
    striped_split_layout = (
        prepared.split_kv <= _MAX_STRIPED_SPLIT_KV
        and num_count_segments == 4 * prepared.split_kv
    )
    single_pool = mode == _SelectiveLogitsMode.CANDIDATE and _h64_single_pool_topology(
        prepared.num_heads,
        prepared.head_dim,
        prepared.query_tile,
        prepared.num_umma_stages,
        prepared.split_kv,
        num_count_segments,
        capacity,
    )
    q1_pool = mode == _SelectiveLogitsMode.CANDIDATE and _q1_owner_pool_topology(
        prepared.num_heads,
        prepared.head_dim,
        prepared.query_tile,
        prepared.num_umma_stages,
        prepared.split_kv,
        num_count_segments,
        capacity,
    )
    pooled_split_layout = (
        single_pool
        or q1_pool
        or (
            prepared.split_kv > _MAX_STRIPED_SPLIT_KV
            and num_count_segments == prepared.split_kv
        )
    )
    lane_local_layout = (
        prepared.split_kv <= 4 and num_count_segments == 32 * prepared.split_kv
    )
    valid_split_layout = (
        striped_split_layout or pooled_split_layout or lane_local_layout
    )
    if num_count_segments > 0 and (
        mode != _SelectiveLogitsMode.CANDIDATE or not valid_split_layout
    ):
        required_layout = f"Qx{_candidate_count_segments(prepared.split_kv) + 1}"
        if prepared.split_kv <= 4:
            required_layout += f" or Qx{32 * prepared.split_kv + 1}"
        raise ValueError(
            f"split_kv={prepared.split_kv} candidate publication requires "
            f"{required_layout} candidate counts"
        )
    if num_count_segments and capacity % num_count_segments:
        raise ValueError(
            "segmented candidate capacity must be divisible by its segment count: "
            f"mode={mode}, capacity={capacity}, segments={num_count_segments}"
        )
    lane_local_layout = mode == _SelectiveLogitsMode.CANDIDATE and lane_local_layout
    if lane_local_layout and (capacity // num_count_segments) % (
        256 // num_count_segments
    ):
        raise ValueError(
            "lane-local candidates require per-slice capacity divisible by "
            "their cooperating-thread count"
        )
    count_segment_span = 0
    if (
        mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR)
        and required_words > 1
    ):
        # Each store warp consumes one fixed 32-token slice from each math
        # stream and appends both into the same warp-ranked segment. The
        # lane-local layouts instead assign one fixed slice to every store-lane
        # vector or adjacent vector pair.
        count_segment_span = (
            1 if lane_local_layout else 128 if pooled_split_layout else 32
        )
    if (
        mode == _SelectiveLogitsMode.CANDIDATE
        and required_words > 1
        and not single_pool
        and not q1_pool
        and num_count_segments not in (4, 8, 12, 16, 32, 64, 128)
    ):
        raise ValueError(
            "candidate publication requires 4, 8, 12, 16, 32, 64, or 128 segments"
        )
    if not values.is_contiguous() or not index_counts.is_contiguous():
        raise ValueError("selective-logits outputs must be contiguous")
    output_ranges = (_byte_range(values), _byte_range(index_counts))
    input_ranges = tuple(
        _byte_range(tensor) for tensor in (q, kv_fused, weights, block_table)
    )
    metadata_ranges = tuple(_byte_range(tensor) for tensor in metadata)
    if _overlap(*output_ranges) or any(
        _overlap(output_range, protected_range)
        for output_range in output_ranges
        for protected_range in (*input_ranges, *metadata_ranges)
    ):
        raise ValueError(
            "selective-logits outputs must not alias each other, inputs, or prepared metadata"
        )

    if mode != _SelectiveLogitsMode.SAMPLE:
        if policy_values is None or policy_values.shape != (prepared.rows,):
            raise ValueError("this mode requires one policy value per Q row")
        if policy_values.device != q.device or not policy_values.is_contiguous():
            raise ValueError("policy_values must be contiguous on q.device")
        if any(_overlap(_byte_range(policy_values), item) for item in output_ranges):
            raise ValueError("policy_values must not alias selective-logits outputs")
        if any(_overlap(_byte_range(policy_values), item) for item in metadata_ranges):
            raise ValueError("policy_values must not alias prepared metadata")
        if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM:
            if policy_values.dtype not in (torch.int32, torch.uint32):
                raise ValueError("radix policy_values must contain uint32 prefix bits")
            policy_words = policy_values.view(torch.int32)
        else:
            if policy_values.dtype != torch.float32:
                raise ValueError("threshold policy_values must be float32")
            policy_words = policy_values.view(torch.int32)
    else:
        # Sample never reads policy words; retain a valid address without
        # introducing a dummy allocation into the captureable launch path.
        policy_words = prepared.context_lens

    _require_aligned(
        (
            q,
            kv_fused,
            weights,
            block_table,
            values,
            index_counts,
            policy_words,
            *metadata,
        )
    )
    if (
        mode != _SelectiveLogitsMode.SAMPLE
        and _clear_counts
        and not _candidate_counts_precleared
    ):
        index_counts[:, :required_words].zero_()

    args = (
        *_runtime_args(
            q,
            kv_fused,
            weights,
            block_table,
            values,
            index_counts,
            prepared,
            policy_words,
            sample_page_indices,
            _sparse_sample_metadata,
        ),
        cutlass.Int32(kv_fused.shape[0]),
        cutlass.Int32(prepared.context_lens.shape[0]),
    )
    stream = _current_stream(q.device)
    capability = torch.cuda.get_device_capability(q.device)
    if _h64_candidate and not (
        mode == _SelectiveLogitsMode.CANDIDATE
        and (prepared.num_heads, prepared.head_dim) == (64, 128)
        and prepared.query_tile == 4
        and prepared.page_size == _COMPUTE_BLOCK_KV
        and prepared.split_kv == 1
        and required_words == 1 + _candidate_count_segments(prepared.split_kv)
        and sample_page_indices is None
    ):
        raise ValueError(
            "the H64 candidate variant requires an H64 D128 Q4 split-1 CANDIDATE "
            "launch with 128-token pages and four count segments"
        )
    key: Tuple[Any, ...] = (
        capability[0] * 10 + capability[1],
        _device_ordinal(q.device),
        prepared.num_sms,
        mode,
        capacity,
        required_words,
        count_segment_span,
        radix_shift,
        prepared.causal_bounds,
        prepared.split_kv,
        prepared.num_heads,
        prepared.head_dim,
        prepared.page_size,
        prepared.query_tile,
        prepared.num_epi_subtiles,
        prepared.num_umma_stages,
        sample_page_indices is not None,
        prepared.math_mode,
        _h64_candidate,
        _sparse_sample_load_warps,
        _sparse_sample_group_rows,
        _sparse_sample_metadata is not None,
    )
    if mode in (
        _SelectiveLogitsMode.SAMPLE,
        _SelectiveLogitsMode.CANDIDATE,
        _SelectiveLogitsMode.REPAIR,
        _SelectiveLogitsMode.RADIX_HISTOGRAM,
        _SelectiveLogitsMode.EXACT,
    ):
        key = (*key, _shared_engine_source_token())
    compiled = _SELECTIVE_LOGITS_COMPILED.get(key)
    if compiled is None and torch.cuda.is_current_stream_capturing():
        # A launch-binding miss under capture may only build DLPack arguments;
        # compiling would run host work the captured graph cannot contain.
        raise RuntimeError(
            "selective logits kernel is not compiled; prepare a pool before capture"
        )
    if compiled is None and _h64_candidate:
        from .kernels._fp8_mqa_selective import FP8MQASelectiveH64CandidateKernel

        compiled = _compile_for_device(
            q.device,
            FP8MQASelectiveH64CandidateKernel(
                num_sms=prepared.num_sms,
                capacity=capacity,
                math_mode=prepared.math_mode,
            ),
            *args,
            stream,
        )
        _SELECTIVE_LOGITS_COMPILED[key] = compiled
    if compiled is None:
        from .kernels._fp8_mqa_selective import (
            FP8MQASelectiveLogitsKernel,
            FP8MQASparseGather4SampleKernel,
            SELECTIVE_LOGITS_TILING,
        )

        kernel_class: Any = FP8MQASelectiveLogitsKernel
        sparse_options = {}
        if _sparse_sample_load_warps:
            kernel_class = FP8MQASparseGather4SampleKernel
            sparse_options = dict(
                load_warps=_sparse_sample_load_warps,
                group_rows=_sparse_sample_group_rows,
                on_the_fly=_sparse_sample_metadata is not None,
            )
        compiled = _compile_for_device(
            q.device,
            kernel_class(
                num_heads=prepared.num_heads,
                head_dim=prepared.head_dim,
                phys_block_kv=prepared.page_size,
                num_sms=prepared.num_sms,
                math_mode=prepared.math_mode,
                mode=mode,
                capacity=capacity,
                radix_shift=radix_shift,
                candidate_uses_causal_bounds=prepared.causal_bounds,
                split_kv=prepared.split_kv,
                num_count_segments=(
                    required_words - 1
                    if mode
                    in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR)
                    else 0
                ),
                count_segment_span=count_segment_span,
                sample_internal=sample_page_indices is not None,
                tiling=replace(
                    SELECTIVE_LOGITS_TILING,
                    next_n=prepared.query_tile,
                    num_epi_subtiles=prepared.num_epi_subtiles,
                    num_umma_stages=prepared.num_umma_stages,
                ),
                **sparse_options,
            ),
            *args,
            stream,
        )
        _SELECTIVE_LOGITS_COMPILED[key] = compiled
    if not _compile_only:
        compiled(*args, stream)
    prepared._launches[cache_key] = _CachedSelectiveLogitsLaunch(
        identity=_selective_logits_launch_identity(
            launch_tensors,
            capacity,
        ),
        tensors=launch_tensors,
        compiled=compiled,
        args=args,
        count_view=(
            index_counts[:, :required_words]
            if mode != _SelectiveLogitsMode.SAMPLE
            else None
        ),
    )


def _finalize_fp8_mqa_packed_candidates(
    index_counts: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    repair_flags: torch.Tensor,
    *,
    rows: int,
    capacity: int,
    top_k: int = _DEFAULT_TOP_K,
    _compile_only: bool = False,
    _candidate_prepared: _PreparedFP8MQASelectiveLogits | None = None,
    row_ends: torch.Tensor | None = None,
) -> None:
    """Compact segmented candidates and publish exact count/repair metadata.

    Args:
        index_counts: Contiguous int32 CUDA counts with one logical count plus
            4, 8, 12, 16, 32, 64, or 128 segment counters. The exact prepared
            H64 split1 pooled candidate also supports one segment (Qx2).
            Eligible Q1 per-owner pooling uses one segment per split with
            capacity 6144 and the matching prepared topology.
        values: Contiguous int64 packed candidate rows ``[padded_Q, capacity]``.
        lengths: Contiguous int32 CUDA output lengths with at least ``rows``
            elements.
        repair_flags: Contiguous int32 CUDA output flags with at least ``rows``
            elements.
        rows: Number of logical rows to finalize.
        capacity: Candidate capacity per row; divisible by the segment count.
        top_k: Minimum retained candidates required before exact repair.
        row_ends: Optional contiguous int32 CUDA per-row exclusive ends. When
            given, a complete row shorter than ``top_k`` is not flagged; only a
            total below ``min(top_k, row_end)`` (or an overflow) is.

    The function compacts retained segment prefixes in place. Column zero is
    set to the exact logical count, or ``capacity + 1`` when any segment
    overflows; ``repair_flags`` also marks underfilled rows.
    """
    _require_cute()
    tensors = (index_counts, values, lengths, repair_flags)
    identity_tensors = tensors if row_ends is None else (*tensors, row_ends)
    single_pool = (
        _candidate_prepared is not None
        and index_counts.dim() == 2
        and index_counts.shape[1] == 2
        and _h64_single_pool_topology(
            _candidate_prepared.num_heads,
            _candidate_prepared.head_dim,
            _candidate_prepared.query_tile,
            _candidate_prepared.num_umma_stages,
            _candidate_prepared.split_kv,
            index_counts.shape[1] - 1,
            capacity,
        )
    )
    q1_pool = (
        _candidate_prepared is not None
        and index_counts.dim() == 2
        and _q1_owner_pool_topology(
            _candidate_prepared.num_heads,
            _candidate_prepared.head_dim,
            _candidate_prepared.query_tile,
            _candidate_prepared.num_umma_stages,
            _candidate_prepared.split_kv,
            index_counts.shape[1] - 1,
            capacity,
        )
    )
    layout_identity = (
        ("q1-owner-pool-v1", capacity, index_counts.shape[1] - 1)
        if q1_pool
        else ("h64-single-pool-v2", capacity)
        if single_pool
        else ()
    )
    if all(isinstance(tensor, torch.Tensor) for tensor in identity_tensors):
        identity = (
            rows,
            top_k,
            *layout_identity,
            *_selective_logits_launch_identity(identity_tensors, capacity),
        )
        cached = getattr(values, "_flashinfer_packed_finalize_launch", None)
        if cached is not None and cached.identity == identity:
            if _compile_only:
                return
            cached.compiled(*cached.args, _current_stream(index_counts.device))
            return
    if rows <= 0 or capacity <= 0:
        raise ValueError("rows and capacity must be positive")
    _validate_selective_top_k(top_k)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(
            "candidate values, counts, lengths, and flags must be contiguous CUDA"
        )
    if (
        index_counts.dtype != torch.int32
        or values.dtype != torch.int64
        or lengths.dtype != torch.int32
        or repair_flags.dtype != torch.int32
    ):
        raise ValueError("candidate finalizer requires int64 values and int32 metadata")
    if not all(tensor.device == index_counts.device for tensor in tensors[1:]):
        raise ValueError("candidate finalizer tensors must share one device")
    if (
        index_counts.dim() != 2
        or index_counts.shape[0] < rows
        or (
            index_counts.shape[1]
            not in (1, *tuple(segments + 1 for segments in _COUNT_SEGMENTS))
            and not single_pool
            and not q1_pool
        )
        or values.dim() != 2
        or values.shape[0] < rows
        or values.shape[1] < capacity
        or lengths.dim() != 1
        or lengths.numel() < rows
        or repair_flags.dim() != 1
        or repair_flags.numel() < rows
    ):
        raise ValueError(
            "candidate finalizer storage is smaller than the requested rows"
        )
    num_segments = index_counts.shape[1] - 1
    if num_segments and capacity % num_segments:
        raise ValueError(
            "segmented candidate capacity must be divisible by its segment count"
        )
    if num_segments in (32, 64, 128) and (capacity // num_segments) % (
        256 // num_segments
    ):
        raise ValueError(
            "lane-local candidates require per-slice capacity divisible by "
            "their cooperating-thread count"
        )
    ranges = tuple(_byte_range(tensor) for tensor in tensors)
    if any(
        _overlap(ranges[left], ranges[right])
        for left in range(len(ranges))
        for right in range(left + 1, len(ranges))
    ):
        raise ValueError("candidate finalizer tensors must not alias")
    if row_ends is not None and (
        not isinstance(row_ends, torch.Tensor)
        or row_ends.dtype != torch.int32
        or row_ends.device != index_counts.device
        or row_ends.dim() != 1
        or row_ends.numel() < rows
        or not row_ends.is_contiguous()
    ):
        raise ValueError("row_ends must be contiguous int32 CUDA storage per row")
    _require_aligned(tensors)

    args = (
        from_dlpack(index_counts, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(values, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(lengths, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(repair_flags, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        cutlass.Int32(rows),
        cutlass.Int32(capacity),
        (
            from_dlpack(row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            )
            if row_ends is not None
            else None
        ),
    )
    stream = _current_stream(index_counts.device)
    capability = torch.cuda.get_device_capability(index_counts.device)
    key: Tuple[int, ...] = (
        *capability,
        capacity,
        num_segments,
        top_k,
        row_ends is not None,
    )
    compiled = _FINALIZE_COMPILED.get(key)
    if compiled is None:
        from .kernels.selective_logits_metadata import SelectiveLogitsFinalizeKernel

        compiled = _compile_for_device(
            index_counts.device,
            SelectiveLogitsFinalizeKernel(capacity, num_segments, top_k),
            *args,
            stream,
        )
        _FINALIZE_COMPILED[key] = compiled
    if not _compile_only:
        compiled(*args, stream)
    values._flashinfer_packed_finalize_launch = _CachedFinalizeLaunch(
        identity=(
            rows,
            top_k,
            *layout_identity,
            *_selective_logits_launch_identity(identity_tensors, capacity),
        ),
        tensors=identity_tensors,
        compiled=compiled,
        args=args,
    )


class _FP8PagedMQASelectiveTopKEngine:
    """Private scoring, candidate selection and exact-repair launch owners.

    The capacity plan supplies address-stable device metadata and shares this
    engine's scratch across serialized layer calls. Packed int32 output has
    shape ``[rows, top_k]`` with request-local indices and -1 suffix padding.
    Engine instances must not run concurrently on different streams.
    """

    def __init__(self, *, math_mode: str = "relu_strict") -> None:
        self._math_mode = _validate_math_mode(math_mode)
        self._math_mode_at_construction = self._math_mode
        self._sample_gather_views: (
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None
        self._sample_ladder: torch.Tensor | None = None
        self._sample_populations: torch.Tensor | None = None
        self._sample_page_indices: torch.Tensor | None = None
        self._sample_sparse_metadata: tuple[torch.Tensor, ...] | None = None
        self._candidate_schedule: _CandidateSchedule | None = None
        self._prepared: _PreparedFP8MQASelectiveLogits | None = None
        self._repair_prepared: _PreparedFP8MQASelectiveLogits | None = None
        self._candidate_prepared: _PreparedFP8MQASelectiveLogits | None = None
        self._sample_prepared: _PreparedFP8MQASelectiveLogits | None = None
        self._rows = 0
        self._batches = 0
        self._top_k = _DEFAULT_TOP_K
        self._padded_rows = 0
        self._kv_rows = 0
        self._cu_q: torch.Tensor | None = None
        self._cu_kv: torch.Tensor | None = None
        self._physical_cu_kv: torch.Tensor | None = None
        self._planned_compression_ratio = 1
        self._empty_physical = False
        self._needs_sampling = False
        self._compact_repair = False
        self._repair_q: torch.Tensor | None = None
        self._repair_weights: torch.Tensor | None = None
        self._repair_cu_q: torch.Tensor | None = None
        self._repair_row_ids: torch.Tensor | None = None
        self._repair_active_count: torch.Tensor | None = None
        self._first_repair_reset: torch.Tensor | None = None
        self._terminal_repair_reset: torch.Tensor | None = None
        self._second_row_ids: torch.Tensor | None = None
        self._second_active_count: torch.Tensor | None = None
        self._compact_repair_flags: torch.Tensor | None = None
        self._sample_count = 0
        self._sample_indices: torch.Tensor | None = None
        self._sample_request_ids: torch.Tensor | None = None
        self._sample_request_starts: torch.Tensor | None = None
        self._sample_ends: torch.Tensor | None = None
        self._sample_fused: torch.Tensor | None = None
        self._sample_block_table: torch.Tensor | None = None
        self._sample_scores: torch.Tensor | None = None
        self._sample_scratch: torch.Tensor | None = None
        self._sample_rank_capacity = 0
        self._sample_rank_indices: torch.Tensor | None = None
        self._sample_ranks: torch.Tensor | None = None
        self._thresholds: torch.Tensor | None = None
        self._selected_bins: torch.Tensor | None = None
        self._candidate_values: torch.Tensor | None = None
        self._candidate_counts: torch.Tensor | None = None
        self._candidate_lengths: torch.Tensor | None = None
        self._repair_flags: torch.Tensor | None = None
        self._row_ends: torch.Tensor | None = None
        self._repair_starts: torch.Tensor | None = None
        self._repair_ends: torch.Tensor | None = None
        self._repair_thresholds: torch.Tensor | None = None
        self._packed_repair_starts: torch.Tensor | None = None
        self._packed_repair_ends: torch.Tensor | None = None
        self._packed_repair_thresholds: torch.Tensor | None = None
        self._repair_counts: torch.Tensor | None = None
        self._repair_lengths: torch.Tensor | None = None
        self._effective_lengths: torch.Tensor | None = None
        self._second_repair_flags: torch.Tensor | None = None
        self._topk_row_states: torch.Tensor | None = None
        self._repair_scores: torch.Tensor | None = None
        self._repair_topk_positions_i32: torch.Tensor | None = None
        self._repair_topk_compiled: Any = None
        self._histograms: torch.Tensor | None = None
        self._prefixes: torch.Tensor | None = None
        self._ranks: torch.Tensor | None = None
        self._threshold_bits: torch.Tensor | None = None
        self._radix_values: torch.Tensor | None = None
        self._exact_indices: torch.Tensor | None = None
        self._exact_counts: torch.Tensor | None = None
        self._greater_indices: torch.Tensor | None = None
        self._greater_counts: torch.Tensor | None = None
        self._equal_indices: torch.Tensor | None = None
        self._equal_counts: torch.Tensor | None = None
        self._selected: torch.Tensor | None = None
        self._exact_error: torch.Tensor | None = None
        self._repair_if_module: Any = None
        self._repair_body_stream: torch.cuda.Stream | None = None
        self._repair_body_stream_owner: Any = None
        self._radix_select_launch: Any = None
        self._combine_exact_launch: Any = None
        self._repair_threshold_launch: Any = None
        self._packed_topk_launch: Any = None
        self._repair_resolve_launch: Any = None
        self._repair_row_compact_launch: Any = None
        self._repair_row_pack_launch: Any = None
        self._second_repair_row_pack_launch: Any = None
        self._compact_repair_resolve_launch: Any = None
        self._compact_combine_exact_launch: Any = None
        self._sample_threshold_launch: Any = None

    def _request_sample_indices(
        self, kv_length: int, page_size: int
    ) -> tuple[list[int], list[int]]:
        """Return sorted logical sample coordinates and optional page indices."""
        pages = math.ceil(kv_length / page_size)
        selected_pages = min(pages, _SAMPLE_CAPACITY // page_size)
        page_indices = [
            i * (pages - 1) // max(selected_pages - 1, 1) for i in range(selected_pages)
        ]
        return [
            page * page_size + offset
            for page in page_indices
            for offset in range(min(page_size, kv_length - page * page_size))
        ], page_indices

    def _use_compact_repair(self, batches, num_heads, head_dim, schedule):
        return batches == 1 and num_heads == 8 and head_dim == 128

    @property
    def math_mode(self) -> _MathMode:
        """Immutable scoring policy shared by sampling, candidates and repair."""
        return self._math_mode

    def _check_math_mode(self) -> None:
        if self._math_mode != self._math_mode_at_construction:
            raise RuntimeError("math_mode is immutable; construct a new wrapper")
        for prepared in (
            self._prepared,
            self._candidate_prepared,
            self._repair_prepared,
            self._sample_prepared,
        ):
            if prepared is not None and prepared.math_mode != self.math_mode:
                raise RuntimeError("prepared math_mode does not match its wrapper")

    def _allocate_selective_plan(
        self,
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        *,
        q_prefix,
        kv_prefix,
        logical_ends_host,
        top_k,
        num_heads,
        head_dim,
        num_sms,
        compression_ratio,
        capacity_bounds=False,
    ) -> None:
        """Allocate a chain from validated host geometry or declared capacities.

        Capacity callers supply reserved geometry and run a device validator
        before every score/metadata launch; no device prefix values are read
        here. Fixed kernel tests supply their case geometry directly.
        """
        _explicit_row_ends = logical_ends_host if compression_ratio != 1 else None
        self._candidate_schedule = None
        self._sample_gather_views = None
        self._sample_rank_capacity = 0
        self._rows = q.shape[0]
        self._batches = len(q_prefix) - 1
        self._top_k = top_k
        self._kv_rows = kv_prefix[-1]
        self._cu_q = cu_q
        self._cu_kv = cu_kv
        self._physical_cu_kv = (
            cu_kv
            if compression_ratio == 1
            else torch.tensor(kv_prefix, dtype=torch.int32, device=q.device)
        )
        cu_kv = self._physical_cu_kv
        self._planned_compression_ratio = compression_ratio
        self._empty_physical = kv_prefix[-1] == 0
        if self._empty_physical:
            # No scorer or dummy key is needed when every physical extent is zero.
            self._prepared = self._candidate_prepared = self._repair_prepared = None
            self._sample_prepared = None
            self._row_ends = torch.tensor(
                logical_ends_host, device=q.device, dtype=torch.int32
            )
            self._selected = torch.empty(
                (self._rows, top_k), device=q.device, dtype=torch.int32
            )
            return
        logical_ends_host = []
        q_lengths: list[int] = []
        kv_lengths: list[int] = []
        if _explicit_row_ends is not None and len(_explicit_row_ends) != self._rows:
            raise ValueError("explicit row ends must cover every query row")
        for q_begin, q_end, kv_begin, kv_end in zip(
            q_prefix[:-1],
            q_prefix[1:],
            kv_prefix[:-1],
            kv_prefix[1:],
            strict=True,
        ):
            q_len = q_end - q_begin
            kv_len = kv_end - kv_begin
            if _explicit_row_ends is None and kv_len < q_len:
                raise ValueError(
                    "causal selective TopK requires every KV length >= its Q length"
                )
            q_lengths.append(q_len)
            kv_lengths.append(kv_len)
            if _explicit_row_ends is None:
                logical_ends_host.extend(range(kv_len - q_len + 1, kv_len + 1))
            else:
                ends = _explicit_row_ends[q_begin:q_end]
                if any(end < 0 or end > kv_len for end in ends):
                    raise ValueError("explicit row ends must fit physical KV lengths")
                logical_ends_host.extend(ends)
        if not logical_ends_host or (
            _explicit_row_ends is None and min(logical_ends_host) <= 0
        ):
            raise ValueError("every query row must have at least one causal KV row")
        self._padded_rows = (
            (self._rows + _MAX_QUERY_TILE - 1) // _MAX_QUERY_TILE * _MAX_QUERY_TILE
        )
        device = q.device
        requested_num_sms = num_sms
        device_sms = torch.cuda.get_device_properties(device).multi_processor_count
        if requested_num_sms is None:
            requested_num_sms = device_sms
        if (
            not isinstance(requested_num_sms, int)
            or requested_num_sms <= 0
            or requested_num_sms > device_sms
        ):
            raise ValueError(f"num_sms must be in [1, {device_sms}]")
        candidate_schedule = _select_candidate_schedule(
            device,
            num_heads,
            head_dim,
            q_lengths,
            kv_lengths,
            num_sms=requested_num_sms,
            top_k=top_k,
            q1_owner_pool=None,
            page_size=kv_fused.shape[1],
        )
        query_tile = candidate_schedule.query_tile
        reference_query_tile = _select_request_tiling(
            device, num_heads, head_dim, q_lengths
        ).next_n
        split_kv = candidate_schedule.split_kv
        self._candidate_schedule = candidate_schedule
        self._row_ends = torch.tensor(
            logical_ends_host, device=device, dtype=torch.int32
        )
        self._repair_starts = torch.zeros(self._rows, dtype=torch.int32, device=device)
        self._repair_ends = torch.zeros_like(self._repair_starts)
        # H64 packs flagged rows into reference Q2 work items; candidate
        # scheduling, sampling and both exact-repair stages stay unchanged.
        self._compact_repair = self._use_compact_repair(
            self._batches, num_heads, head_dim, candidate_schedule
        )
        repair_q = q
        repair_weights = weights
        repair_cu_q = cu_q
        repair_starts = self._repair_starts
        repair_ends = self._repair_ends
        if self._compact_repair:
            self._repair_q = torch.empty_like(q)
            self._repair_weights = torch.empty_like(weights)
            self._repair_cu_q = torch.tensor(
                [0, self._rows], dtype=torch.int32, device=device
            )
            self._repair_row_ids = torch.empty(
                self._rows, dtype=torch.int32, device=device
            )
            # Reset ranges share packed ends and alignment padding. Keep the first
            # count outside the terminal range: resolution still consumes it.
            aligned_rows = (self._rows + 3) // 4 * 4
            repair_state = torch.zeros(
                3 * aligned_rows + 12, dtype=torch.int32, device=device
            )
            self._first_repair_reset = repair_state[: 4 + aligned_rows]
            self._terminal_repair_reset = repair_state[4:]
            self._repair_active_count = repair_state[:1]
            self._second_row_ids = torch.empty_like(self._repair_row_ids)
            self._second_active_count = repair_state[
                4 + aligned_rows : 5 + aligned_rows
            ]
            self._packed_repair_starts = torch.zeros_like(self._repair_starts)
            self._packed_repair_ends = repair_state[4 : 4 + self._rows]
            self._packed_repair_thresholds = torch.empty(
                self._rows, dtype=torch.float32, device=device
            )
            self._compact_repair_flags = torch.empty(
                self._rows, dtype=torch.int32, device=device
            )
            repair_q = self._repair_q
            repair_weights = self._repair_weights
            repair_cu_q = self._repair_cu_q
            repair_starts = self._packed_repair_starts
            repair_ends = self._packed_repair_ends

        self._candidate_prepared = _prepare_fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            row_starts=self._repair_starts if _explicit_row_ends is not None else None,
            row_ends=self._row_ends if _explicit_row_ends is not None else None,
            num_sms=num_sms,
            split_kv=split_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            query_tile=query_tile,
            num_umma_stages=candidate_schedule.num_umma_stages,
            math_mode=self.math_mode,
            _capacity_bounds=capacity_bounds,
        )
        if not _candidate_schedule_matches_prepared(
            candidate_schedule, self._candidate_prepared
        ):
            raise RuntimeError("candidate preparation changed the selected topology")
        self._repair_prepared = _prepare_fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            repair_cu_q,
            cu_kv,
            block_table,
            row_starts=repair_starts,
            row_ends=repair_ends,
            num_sms=num_sms,
            kv_chunk_size=256,
            num_heads=num_heads,
            head_dim=head_dim,
            query_tile=reference_query_tile,
            math_mode=self.math_mode,
            _capacity_bounds=capacity_bounds,
        )
        self._prepared = _prepare_fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            repair_cu_q,
            cu_kv,
            block_table,
            row_starts=repair_starts,
            row_ends=repair_ends,
            num_sms=requested_num_sms,
            kv_chunk_size=256,
            num_heads=num_heads,
            head_dim=head_dim,
            query_tile=reference_query_tile,
            math_mode=self.math_mode,
            _capacity_bounds=capacity_bounds,
        )
        self._padded_rows = (
            (self._rows + self._prepared.query_tile - 1)
            // self._prepared.query_tile
            * self._prepared.query_tile
        )
        self._needs_sampling = any(
            kv_length > _CANDIDATE_CAPACITY for kv_length in kv_lengths
        )
        sample_indices: list[int] = []
        sample_request_ids: list[int] = []
        sample_request_starts = [0]
        sample_ends_host: list[int] = []
        sample_pages_per_request: list[int] = []
        logical_row = 0
        target_candidates = _candidate_target(top_k)
        sample_ranks: list[int] = []
        sample_page_indices: list[list[int]] = []
        for request, kv_length in enumerate(kv_lengths):
            request_indices, page_indices = self._request_sample_indices(
                kv_length, kv_fused.shape[1]
            )
            sample_page_indices.append(page_indices)
            sample_count = len(request_indices)
            sample_indices.extend(request_indices)
            sample_request_ids.extend([request] * sample_count)
            sample_request_starts.append(len(sample_indices))
            sample_pages_per_request.append(math.ceil(sample_count / _COMPUTE_BLOCK_KV))
            q_len = q_prefix[request + 1] - q_prefix[request]
            for row_end in logical_ends_host[logical_row : logical_row + q_len]:
                valid = bisect.bisect_left(request_indices, row_end)
                sample_ends_host.append(valid)
                sample_ranks.append(
                    0
                    if row_end == 0
                    else max(
                        1,
                        min(
                            valid,
                            math.ceil(target_candidates * valid / row_end),
                        ),
                    )
                )
            logical_row += q_len
        self._sample_count = len(sample_indices)
        # Token-pack metadata is not consumed by direct-page sampling. Its
        # variable tail length must not constrain a page-capacity replan.
        self._sample_indices = None
        self._sample_request_ids = None
        self._sample_request_starts = None
        self._sample_fused = kv_fused
        max_sample_pages = max(sample_pages_per_request, default=0)
        sample_table_host = [[0] * max_sample_pages for _ in range(self._batches)]
        page_begin = 0
        for request, page_count in enumerate(sample_pages_per_request):
            sample_table_host[request][:page_count] = range(
                page_begin, page_begin + page_count
            )
            page_begin += page_count
        self._sample_block_table = torch.tensor(
            sample_table_host, dtype=torch.int32, device=device
        )
        max_pages = max(map(len, sample_page_indices))
        self._sample_page_indices = torch.tensor(
            [pages + [0] * (max_pages - len(pages)) for pages in sample_page_indices],
            dtype=torch.int64,
            device=device,
        )
        self._sample_block_table = torch.empty(
            (self._batches, max_pages), dtype=torch.int32, device=device
        )
        if _explicit_row_ends is not None and 0 in kv_lengths:
            # Never even read a zero-physical request's padded table row.
            # Static nonempty views are retained for every future replay.
            self._sample_gather_views = [
                (
                    block_table[request : request + 1],
                    self._sample_page_indices[request : request + 1],
                    self._sample_block_table[request : request + 1],
                )
                for request, length in enumerate(kv_lengths)
                if length > 0
            ]
            for source, indices, destination in self._sample_gather_views:
                torch.gather(source, 1, indices, out=destination)
        else:
            torch.gather(
                block_table,
                1,
                self._sample_page_indices,
                out=self._sample_block_table,
            )
        self._sample_fused = kv_fused
        sample_ends = torch.tensor(sample_ends_host, dtype=torch.int32, device=device)
        self._sample_ends = sample_ends
        sample_starts = torch.zeros_like(sample_ends)
        sample_cu_kv = torch.tensor(
            sample_request_starts, dtype=torch.int32, device=device
        )
        self._sample_prepared = _prepare_fp8_mqa_selective_logits(
            q,
            self._sample_fused,
            weights,
            cu_q,
            sample_cu_kv,
            self._sample_block_table,
            row_starts=sample_starts,
            row_ends=sample_ends,
            num_sms=num_sms,
            num_heads=num_heads,
            head_dim=head_dim,
            query_tile=reference_query_tile,
            math_mode=self.math_mode,
            _capacity_bounds=capacity_bounds,
        )
        self._sample_scores = torch.full(
            (self._padded_rows, _SAMPLE_CAPACITY),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        self._sample_scratch = torch.empty(
            (self._padded_rows, 1), dtype=torch.int32, device=device
        )
        # Rank selection reads the fixed score arena, not a rank-sized buffer.
        self._sample_rank_capacity = _SAMPLE_CAPACITY
        self._sample_rank_indices = (
            torch.tensor(sample_ranks, dtype=torch.int64, device=device)
            .sub_(1)
            .unsqueeze(1)
        )
        self._sample_ranks = torch.tensor(
            sample_ranks, dtype=torch.int32, device=device
        )
        self._thresholds = torch.empty(self._rows, dtype=torch.float32, device=device)
        self._selected_bins = torch.empty(self._rows, dtype=torch.int32, device=device)
        self._sample_ladder = torch.empty(
            (self._rows, 3), dtype=torch.float32, device=device
        )
        self._sample_populations = torch.empty(
            (self._rows, 3), dtype=torch.int32, device=device
        )

        count_words = 1 + candidate_schedule.count_segments
        candidate_padded_rows = max(
            self._padded_rows,
            math.ceil(self._rows / query_tile) * query_tile,
        )
        self._candidate_values = torch.empty(
            (candidate_padded_rows, _REPAIR_CAPACITY),
            dtype=torch.int64,
            device=device,
        )
        self._candidate_counts = torch.empty(
            (candidate_padded_rows, count_words), dtype=torch.int32, device=device
        )
        self._candidate_lengths = torch.empty(
            self._rows, dtype=torch.int32, device=device
        )
        self._repair_flags = torch.zeros_like(self._candidate_lengths)
        self._repair_thresholds = torch.empty_like(self._thresholds)
        self._repair_counts = torch.empty(
            (self._padded_rows, 1), dtype=torch.int32, device=device
        )
        self._repair_lengths = torch.empty_like(self._candidate_lengths)
        self._effective_lengths = torch.empty_like(self._candidate_lengths)
        self._second_repair_flags = (
            repair_state[8 + aligned_rows : 8 + aligned_rows + self._rows]
            if self._compact_repair
            else torch.zeros_like(self._candidate_lengths)
        )
        self._repair_scores = (
            self._candidate_values[: self._rows]
            .view(torch.float32)
            .as_strided(
                (self._rows, _REPAIR_CAPACITY),
                (_REPAIR_CAPACITY * 2, 2),
                storage_offset=1,
            )
        )
        from ...topk_varlen.topk_varlen import (
            _RADIX_STATE_SIZE,
            _compile_radix,
            _radix_get_chunk_config,
            get_device_sm_count,
            get_shared_bytes_per_block_optin,
            torch_to_cutlass_dtype,
        )

        topk_num_sms = get_device_sm_count(device)
        repair_topk_ctas, repair_topk_chunk = _radix_get_chunk_config(
            _REPAIR_CAPACITY,
            torch.float32,
            self._rows,
            topk_num_sms,
            get_shared_bytes_per_block_optin(device),
        )
        topk_groups = max(1, topk_num_sms // repair_topk_ctas)
        self._topk_row_states = torch.zeros(
            (topk_groups, _RADIX_STATE_SIZE), dtype=torch.int32, device=device
        )
        self._repair_topk_compiled = _compile_radix(
            torch_to_cutlass_dtype(torch.float32),
            top_k,
            1,
            1,
            _REPAIR_CAPACITY,
            False,
            repair_topk_ctas,
            repair_topk_chunk,
            topk_num_sms,
            2,
            _REPAIR_CAPACITY * 2,
            has_active_rows=self._compact_repair,
            has_active_mask=not self._compact_repair,
        )
        self._histograms = torch.empty(
            (self._padded_rows, 256), dtype=torch.int32, device=device
        )
        self._prefixes = torch.empty(self._rows, dtype=torch.int32, device=device)
        self._ranks = (
            repair_state[8 + 2 * aligned_rows : 8 + 2 * aligned_rows + self._rows]
            if self._compact_repair
            else torch.empty(self._rows, dtype=torch.int32, device=device)
        )
        self._threshold_bits = torch.empty(self._rows, dtype=torch.int32, device=device)
        self._radix_values = torch.empty(
            (self._padded_rows, 1), dtype=torch.int32, device=device
        )
        # Independent aligned banks share one scorer launch and count reset.
        exact_bank_rows = ((self._padded_rows + 3) // 4) * 4
        self._exact_indices = torch.empty(
            (2 * exact_bank_rows, top_k), dtype=torch.int32, device=device
        )
        self._exact_counts = torch.empty(
            (2 * exact_bank_rows, 1), dtype=torch.int32, device=device
        )
        self._greater_indices = self._exact_indices[:exact_bank_rows]
        self._equal_indices = self._exact_indices[exact_bank_rows:]
        # First-repair resolution retires positions before terminal EXACT
        # owns this bank.
        if num_heads == 64 and head_dim == 128:
            self._repair_topk_positions_i32 = self._greater_indices[: self._rows]
        else:
            self._repair_topk_positions_i32 = torch.empty(
                (self._rows, top_k), dtype=torch.int32, device=device
            )
        self._greater_counts = self._exact_counts[:exact_bank_rows]
        self._equal_counts = self._exact_counts[exact_bank_rows:]
        self._selected = torch.empty(
            (self._rows, top_k), dtype=torch.int32, device=device
        )
        self._exact_error = (
            repair_state[8 + 3 * aligned_rows : 9 + 3 * aligned_rows]
            if self._compact_repair
            else torch.zeros(1, dtype=torch.int32, device=device)
        )
        self._compile_auxiliary_launches()
        if self._compact_repair:
            self._compile_compact_repair_launches(q, weights)
        self._precompile_score_launches(q, kv_fused, weights, block_table)
        if not self._compact_repair:
            _launch_compact_metadata(
                self._repair_prepared,
                self._cu_q,
                self._repair_starts,
                self._repair_ends,
                self._repair_flags,
            )
            _launch_compact_metadata(
                self._prepared,
                self._cu_q,
                self._repair_starts,
                self._repair_ends,
                self._second_repair_flags,
            )

    def _compile_compact_repair_launches(
        self, q: torch.Tensor, weights: torch.Tensor
    ) -> None:
        """Compile and bind the batch-one dense repair-row path."""
        assert self._prepared is not None
        assert self._repair_flags is not None
        assert self._repair_thresholds is not None
        assert self._row_ends is not None
        assert self._repair_row_ids is not None
        assert self._repair_active_count is not None
        assert self._second_row_ids is not None
        assert self._second_active_count is not None
        assert self._repair_q is not None
        assert self._repair_weights is not None
        assert self._packed_repair_ends is not None
        assert self._packed_repair_thresholds is not None
        assert self._compact_repair_flags is not None
        assert self._candidate_values is not None
        assert self._repair_lengths is not None
        assert self._repair_topk_positions_i32 is not None
        assert self._selected is not None
        assert self._ranks is not None
        assert self._second_repair_flags is not None
        assert self._exact_error is not None
        assert self._greater_indices is not None
        assert self._greater_counts is not None
        assert self._equal_indices is not None
        assert self._equal_counts is not None
        from .kernels.selective_logits_metadata import (
            SelectiveCompactRepairResolveKernel,
            SelectiveCompactTopKCombineExactKernel,
            SelectiveRepairRowCompactKernel,
            SelectiveRepairRowPackKernel,
        )

        stream = _current_stream(q.device)
        capability = torch.cuda.get_device_capability(q.device)

        fused_first_pack = self._rows > 1
        fused_repair_queue = self._rows > 1
        fused_second_pack = self._rows > 1
        compact_args: tuple[Any, ...] = (
            from_dlpack(
                self._repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_thresholds, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._repair_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._packed_repair_thresholds, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._packed_repair_ends, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            cutlass.Int32(self._rows),
        )
        compact_kernel = SelectiveRepairRowCompactKernel(
            self._top_k, self._prepared.num_heads, self._prepared.head_dim
        )
        compact_entry = compact_kernel
        if fused_first_pack:
            compact_entry = (
                compact_kernel.with_policy_pack
                if fused_repair_queue
                else compact_kernel.with_pack
            )
            packed_inputs = tuple(
                from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
                    mode=0, divisibility=1
                )
                for tensor in (
                    q.view(torch.int32).reshape(-1),
                    weights.reshape(-1),
                    self._repair_q.view(torch.int32).reshape(-1),
                    self._repair_weights.reshape(-1),
                )
            )
            compact_args = (compact_args[:-1], packed_inputs, compact_args[-1])
        if fused_repair_queue:
            if not fused_first_pack:
                compact_entry = compact_kernel.with_policy
            compact_args += (
                (
                    from_dlpack(
                        self._candidate_counts, assumed_align=16
                    ).mark_layout_dynamic(leading_dim=1),
                    *(
                        from_dlpack(
                            tensor, assumed_align=16
                        ).mark_compact_shape_dynamic(mode=0, divisibility=1)
                        for tensor in (
                            self._thresholds,
                            self._selected_bins,
                            self._repair_ends,
                        )
                    ),
                ),
            )
        compact_key = (
            *capability,
            256,
            self._top_k,
            fused_repair_queue,
            fused_first_pack,
            self._prepared.num_heads,
            self._prepared.head_dim,
        )
        compact_compiled = _REPAIR_ROW_COMPACT_COMPILED.get(compact_key)
        if compact_compiled is None:
            compact_compiled = _compile_for_device(
                q.device, compact_entry, *compact_args, stream
            )
            _REPAIR_ROW_COMPACT_COMPILED[compact_key] = compact_compiled
        self._repair_row_compact_launch = (compact_compiled, compact_args)

        q_bytes = q.view(torch.uint8).reshape(-1)
        packed_q_bytes = self._repair_q.view(torch.uint8).reshape(-1)
        pack_args = (
            from_dlpack(q_bytes, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                weights.reshape(-1), assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(packed_q_bytes, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._repair_weights.reshape(-1), assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            cutlass.Int32(self._rows),
        )
        pack_key = (
            *capability,
            self._prepared.num_heads,
            self._prepared.head_dim,
            self._prepared.num_sms,
        )
        pack_compiled = _REPAIR_ROW_PACK_COMPILED.get(pack_key)
        if pack_compiled is None:
            pack_compiled = _compile_for_device(
                q.device,
                SelectiveRepairRowPackKernel(
                    self._prepared.num_heads,
                    self._prepared.head_dim,
                    self._prepared.num_sms,
                ),
                *pack_args,
                stream,
            )
            _REPAIR_ROW_PACK_COMPILED[pack_key] = pack_compiled
        self._repair_row_pack_launch = (pack_compiled, pack_args)
        second_pack_args = (
            *pack_args[:2],
            from_dlpack(
                self._second_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            *pack_args[4:],
        )
        self._second_repair_row_pack_launch = (pack_compiled, second_pack_args)

        resolve_args: tuple[Any, ...] = (
            from_dlpack(self._candidate_values, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._repair_lengths, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_topk_positions_i32, assumed_align=16
            ).mark_layout_dynamic(leading_dim=1),
            from_dlpack(
                self._repair_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._compact_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._second_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._packed_repair_ends, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._second_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._exact_error, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        resolve_kernel = SelectiveCompactRepairResolveKernel(
            self._top_k,
            _REPAIR_CAPACITY,
            num_heads=self._prepared.num_heads,
            head_dim=self._prepared.head_dim,
        )
        resolve_entry = resolve_kernel
        if fused_second_pack:
            resolve_entry = resolve_kernel.with_pack
            packed_inputs = tuple(
                from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
                    mode=0, divisibility=1
                )
                for tensor in (
                    q.view(torch.int32).reshape(-1),
                    weights.reshape(-1),
                    self._repair_q.view(torch.int32).reshape(-1),
                    self._repair_weights.reshape(-1),
                )
            )
            resolve_args = (resolve_args[:-1], packed_inputs, resolve_args[-1])
        resolve_key = (
            *capability,
            self._top_k,
            _REPAIR_CAPACITY,
            256,
            self._prepared.num_heads,
            self._prepared.head_dim,
            fused_second_pack,
        )
        resolve_compiled = _COMPACT_REPAIR_RESOLVE_COMPILED.get(resolve_key)
        if resolve_compiled is None:
            resolve_compiled = _compile_for_device(
                q.device,
                resolve_entry,
                *resolve_args,
                stream,
            )
            _COMPACT_REPAIR_RESOLVE_COMPILED[resolve_key] = resolve_compiled
        self._compact_repair_resolve_launch = (resolve_compiled, resolve_args)

        combine_args = (
            from_dlpack(self._greater_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._greater_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._equal_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._equal_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._packed_repair_ends, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._exact_error, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        combine_key = (*capability, self._top_k, 256)
        combine_compiled = _COMPACT_COMBINE_EXACT_COMPILED.get(combine_key)
        if combine_compiled is None:
            combine_compiled = _compile_for_device(
                q.device,
                SelectiveCompactTopKCombineExactKernel(self._top_k),
                *combine_args,
                stream,
            )
            _COMPACT_COMBINE_EXACT_COMPILED[combine_key] = combine_compiled
        self._compact_combine_exact_launch = (combine_compiled, combine_args)

    def _precompile_score_launches(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        """Compile score phases and repair metadata before graph capture."""
        self._check_math_mode()
        assert self._prepared is not None
        assert self._repair_prepared is not None
        assert self._candidate_prepared is not None
        assert self._sample_prepared is not None
        assert self._histograms is not None
        assert self._prefixes is not None
        assert self._radix_values is not None
        assert self._greater_indices is not None
        assert self._greater_counts is not None
        assert self._equal_indices is not None
        assert self._equal_counts is not None
        assert self._repair_flags is not None
        repair_q = self._repair_q if self._compact_repair else q
        repair_weights = self._repair_weights if self._compact_repair else weights
        assert repair_q is not None
        assert repair_weights is not None
        assert self._repair_thresholds is not None
        assert self._repair_counts is not None
        assert self._repair_lengths is not None
        assert self._effective_lengths is not None
        assert self._second_repair_flags is not None
        assert self._row_ends is not None
        assert self._threshold_bits is not None
        assert self._thresholds is not None
        assert self._selected_bins is not None
        assert self._candidate_counts is not None
        assert self._repair_flags is not None
        assert self._row_ends is not None
        assert self._repair_thresholds is not None
        assert self._repair_ends is not None
        assert self._second_repair_flags is not None
        assert self._sample_scores is not None
        assert self._sample_scratch is not None
        assert self._sample_fused is not None
        assert self._sample_block_table is not None
        assert self._thresholds is not None
        assert self._candidate_values is not None
        assert self._candidate_counts is not None
        assert self._candidate_lengths is not None
        assert self._repair_flags is not None
        if self._compact_repair:
            assert self._physical_cu_kv is not None
            assert self._repair_cu_q is not None
            assert self._packed_repair_starts is not None
            assert self._packed_repair_ends is not None
            assert self._repair_active_count is not None
            assert self._second_active_count is not None
            for prepared, active_rows in (
                (self._repair_prepared, self._repair_active_count),
                (self._prepared, self._second_active_count),
            ):
                _launch_metadata(
                    prepared,
                    self._repair_cu_q,
                    self._physical_cu_kv,
                    row_starts=self._packed_repair_starts,
                    row_ends=self._packed_repair_ends,
                    active_rows=active_rows,
                    _compile_only=True,
                )
        _fp8_mqa_selective_logits(
            q,
            self._sample_fused,
            weights,
            block_table,
            self._sample_prepared,
            mode=_SelectiveLogitsMode.SAMPLE,
            capacity=_SAMPLE_CAPACITY,
            values=self._sample_scores,
            index_counts=self._sample_scratch,
            sample_page_indices=(
                self._sample_page_indices
                if self._sample_sparse_metadata is None
                else None
            ),
            _sparse_sample_metadata=self._sample_sparse_metadata,
            _sparse_sample_load_warps=4
            if self._sample_sparse_metadata is not None
            else 0,
            _sparse_sample_group_rows=_select_sparse_sample_group_rows(
                self._sample_prepared.query_tile
            ),
            _compile_only=True,
        )
        _fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            block_table,
            self._candidate_prepared,
            mode=_SelectiveLogitsMode.CANDIDATE,
            capacity=self._candidate_schedule.candidate_capacity,
            values=self._candidate_values,
            index_counts=self._candidate_counts,
            policy_values=self._thresholds,
            _compile_only=True,
            _h64_candidate=self._candidate_schedule.h64_candidate,
        )
        _finalize_fp8_mqa_packed_candidates(
            self._candidate_counts,
            self._candidate_values,
            self._candidate_lengths,
            self._repair_flags,
            rows=self._rows,
            capacity=self._candidate_schedule.candidate_capacity,
            top_k=self._top_k,
            _compile_only=True,
            _candidate_prepared=self._candidate_prepared,
            row_ends=self._row_ends,
        )
        _fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            block_table,
            self._repair_prepared,
            mode=_SelectiveLogitsMode.REPAIR,
            capacity=_REPAIR_CAPACITY,
            values=self._candidate_values,
            index_counts=self._repair_counts,
            policy_values=(
                self._packed_repair_thresholds
                if self._compact_repair
                else self._repair_thresholds
            ),
            _compile_only=True,
        )
        _finalize_fp8_mqa_packed_candidates(
            self._repair_counts,
            self._candidate_values,
            self._repair_lengths,
            (
                self._compact_repair_flags
                if self._compact_repair
                else self._second_repair_flags
            ),
            rows=self._rows,
            capacity=_REPAIR_CAPACITY,
            top_k=self._top_k,
            _compile_only=True,
        )
        for shift in (24, 16, 8, 0):
            _fp8_mqa_selective_logits(
                repair_q,
                kv_fused,
                repair_weights,
                block_table,
                self._prepared,
                mode=_SelectiveLogitsMode.RADIX_HISTOGRAM,
                capacity=1,
                values=self._radix_values,
                index_counts=self._histograms,
                policy_values=self._prefixes,
                radix_shift=shift,
                _compile_only=True,
            )
        thresholds = self._threshold_bits.view(torch.float32)
        for mode, values, counts in (
            (
                _SelectiveLogitsMode.EXACT,
                self._exact_indices,
                self._exact_counts,
            ),
        ):
            _fp8_mqa_selective_logits(
                repair_q,
                kv_fused,
                repair_weights,
                block_table,
                self._prepared,
                mode=mode,
                capacity=self._top_k,
                values=values,
                index_counts=counts,
                policy_values=thresholds,
                _compile_only=True,
            )

    def _compile_auxiliary_launches(self) -> None:
        assert self._histograms is not None
        assert self._prefixes is not None
        assert self._ranks is not None
        assert self._threshold_bits is not None
        assert self._thresholds is not None
        assert self._selected_bins is not None
        assert self._greater_indices is not None
        assert self._greater_counts is not None
        assert self._equal_indices is not None
        assert self._equal_counts is not None
        assert self._selected is not None
        assert self._exact_error is not None
        assert self._sample_ranks is not None
        assert self._sample_ends is not None
        assert self._sample_prepared is not None
        from .kernels.selective_logits_metadata import (
            SelectivePackedTopKKernel,
            SelectiveRepairThresholdKernel,
            SelectiveRepairResolveKernel,
            SelectiveTopKCombineExactKernel,
            SelectiveTopKRadixSelectKernel,
        )

        stream = _current_stream(self._histograms.device)
        capability = torch.cuda.get_device_capability(self._histograms.device)
        from .jit.selective_sample_ladder import (
            get_selective_sample_ladder_module,
            launch_selective_sample_ladder,
        )

        sample_threshold_module = get_selective_sample_ladder_module()
        sample_threshold_module.prepare(_device_ordinal(self._thresholds.device))
        sample_threshold_compiled = launch_selective_sample_ladder
        sample_threshold_args = (
            sample_threshold_module,
            self._sample_scores,
            self._sample_ends,
            self._sample_ranks,
            self._row_ends,
            self._thresholds,
            self._selected_bins,
            self._sample_ladder,
            self._sample_populations,
            self._top_k,
            self._candidate_schedule.candidate_capacity,
            self._candidate_schedule.count_segments,
        )
        self._sample_threshold_launch = (
            sample_threshold_compiled,
            sample_threshold_args,
        )

        packed_topk_args: tuple[Any, ...] = (
            from_dlpack(self._candidate_values, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._candidate_lengths, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            cutlass.Int32(self._rows),
        )
        num_segments = 0
        if self._rows > 1:
            num_segments = self._candidate_counts.shape[1] - 1
        packed_topk_key = (
            *capability,
            self._top_k,
            self._candidate_schedule.candidate_capacity,
            512,
            num_segments,
        )
        packed_topk_kernel = SelectivePackedTopKKernel(
            self._top_k, self._candidate_schedule.candidate_capacity, num_segments
        )
        packed_topk_entry = packed_topk_kernel
        if num_segments:
            packed_topk_entry = packed_topk_kernel.segmented
            packed_topk_args += (
                from_dlpack(
                    self._candidate_counts, assumed_align=16
                ).mark_layout_dynamic(leading_dim=1),
                from_dlpack(
                    self._repair_flags, assumed_align=16
                ).mark_compact_shape_dynamic(mode=0, divisibility=1),
                from_dlpack(
                    self._row_ends, assumed_align=16
                ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            )
        packed_topk_compiled = _PACKED_TOPK_COMPILED.get(packed_topk_key)
        if packed_topk_compiled is None:
            packed_topk_compiled = _compile_for_device(
                self._thresholds.device,
                packed_topk_entry,
                *packed_topk_args,
                stream,
            )
            _PACKED_TOPK_COMPILED[packed_topk_key] = packed_topk_compiled
        self._packed_topk_launch = (packed_topk_compiled, packed_topk_args)
        repair_resolve_args = (
            from_dlpack(self._candidate_values, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._repair_lengths, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_topk_positions_i32, assumed_align=16
            ).mark_layout_dynamic(leading_dim=1),
            from_dlpack(
                self._repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._repair_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._prefixes, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._exact_error, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        repair_resolve_key = (
            *capability,
            self._top_k,
            _REPAIR_CAPACITY,
            256,
        )
        repair_resolve_compiled = _REPAIR_RESOLVE_COMPILED.get(repair_resolve_key)
        if repair_resolve_compiled is None:
            repair_resolve_compiled = _compile_for_device(
                self._thresholds.device,
                SelectiveRepairResolveKernel(self._top_k, _REPAIR_CAPACITY),
                *repair_resolve_args,
                stream,
            )
            _REPAIR_RESOLVE_COMPILED[repair_resolve_key] = repair_resolve_compiled
        self._repair_resolve_launch = (
            repair_resolve_compiled,
            repair_resolve_args,
        )

        radix_args: tuple[Any, ...] = (
            from_dlpack(self._histograms, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._prefixes, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._threshold_bits, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            cutlass.Int32(self._rows),
            cutlass.Int32(24),
        )
        radix_kernel = SelectiveTopKRadixSelectKernel(clear_histogram=True)
        radix_entry = radix_kernel
        if self._compact_repair:
            assert self._second_active_count is not None
            radix_args = (
                radix_args[:-2]
                + (from_dlpack(self._second_active_count, assumed_align=16),)
                + radix_args[-2:]
            )
            radix_entry = radix_kernel.compact
        radix_key = (*capability, 256, True, self._compact_repair)
        radix_compiled = _RADIX_SELECT_COMPILED.get(radix_key)
        if radix_compiled is None:
            radix_compiled = _compile_for_device(
                self._thresholds.device, radix_entry, *radix_args, stream
            )
            _RADIX_SELECT_COMPILED[radix_key] = radix_compiled
        self._radix_select_launch = (radix_compiled, radix_args)

        repair_threshold_args = (
            from_dlpack(self._candidate_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._thresholds, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._selected_bins, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._repair_thresholds, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._repair_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        repair_threshold_key = (*capability, self._top_k, 8, 256)
        repair_threshold_compiled = _REPAIR_THRESHOLD_COMPILED.get(repair_threshold_key)
        if repair_threshold_compiled is None:
            repair_threshold_compiled = _compile_for_device(
                self._thresholds.device,
                SelectiveRepairThresholdKernel(self._top_k),
                *repair_threshold_args,
                stream,
            )
            _REPAIR_THRESHOLD_COMPILED[repair_threshold_key] = repair_threshold_compiled
        self._repair_threshold_launch = (
            repair_threshold_compiled,
            repair_threshold_args,
        )

        combine_args = (
            from_dlpack(self._greater_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._greater_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._equal_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._equal_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._second_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._exact_error, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        combine_key = (*capability, self._top_k, 256)
        combine_compiled = _COMBINE_EXACT_COMPILED.get(combine_key)
        if combine_compiled is None:
            combine_compiled = _compile_for_device(
                self._thresholds.device,
                SelectiveTopKCombineExactKernel(self._top_k),
                *combine_args,
                stream,
            )
            _COMBINE_EXACT_COMPILED[combine_key] = combine_compiled
        self._combine_exact_launch = (combine_compiled, combine_args)

    def _reset_compact_repair(self, *, terminal: bool) -> None:
        """Clear dependency-compatible queue state in one launch."""
        reset = self._terminal_repair_reset if terminal else self._first_repair_reset
        assert reset is not None
        reset.zero_()

    def _prepare_repair_if(self, device: torch.device) -> None:
        """Load conditional repair support before caller capture."""
        from .jit.selective_repair_if import get_selective_repair_if_module

        self._repair_if_module = get_selective_repair_if_module()
        if (
            self._repair_body_stream is None
            or self._repair_body_stream.device != device
        ):
            # Torch pooled streams are non-exclusive and can alias the parent
            # after pool wrap. The native owner outlives this borrowed view.
            owner = self._repair_if_module.create_body_stream(device.index)
            body = torch.cuda.ExternalStream(owner(), device=device)
            self._repair_body_stream, self._repair_body_stream_owner = body, owner

    def _enqueue_repair(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Capture both repair stages under one device-controlled IF."""
        module = self._repair_if_module
        if module is None or not torch.cuda.is_current_stream_capturing():
            return self._run_repair(q, kv_fused, weights, block_table)
        assert self._repair_body_stream is not None
        assert self._repair_flags is not None
        assert self._second_repair_flags is not None
        assert self._exact_error is not None
        owner = self._repair_body_stream_owner
        assert owner is not None
        body_stream = self._repair_body_stream
        module.preflight_body(q, body_stream.cuda_stream)
        state = module.begin_if(
            self._repair_flags,
            self._second_repair_flags,
            self._exact_error,
            self._repair_active_count if self._compact_repair else None,
            self._second_active_count if self._compact_repair else None,
        )
        body_started = False
        try:
            module.begin_body(q, body_stream.cuda_stream, state[2])
            body_started = True
            with torch.cuda.stream(body_stream):
                self._run_repair(q, kv_fused, weights, block_table)
            module.end_body(q, body_stream.cuda_stream, state[2])
        except BaseException as error:
            try:
                # Also clean up a begin failure that started capture before
                # reporting an error. Never end the caller's parent capture.
                if body_started or module.body_capture_status(
                    q, body_stream.cuda_stream
                ):
                    module.abort_body(q, body_stream.cuda_stream)
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"repair body cleanup failed: {cleanup_error}")
            raise
        # A failed body never publishes a partial IF dependency as success.
        module.finish_if(q, state)
        assert self._selected is not None
        return self._selected

    def _run_impl(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        *,
        _candidate_counts_precleared: bool = False,
        _sampling_policy_initialized: bool = False,
    ) -> torch.Tensor:
        """Enqueue the complete selective path and return stable TopK indices.

        Sampling, thresholding, candidate publication/finalization, TopK, and
        device-flagged exact repair all execute on the caller's current CUDA
        stream. No public phase selection or host data-dependent branch is
        involved.

        Args:
            q: Query tensor matching the storage and layout supplied to
                :meth:`plan`.
            kv_fused: Fused paged KV tensor matching :meth:`plan`.
            weights: Head weights matching :meth:`plan`.
            block_table: Physical-page map matching :meth:`plan`.
            _candidate_counts_precleared: The capacity caller cleared candidate
                counts on the current stream before this layer.
            _sampling_policy_initialized: A nonsampling capacity plan initialized
                its immutable all-key threshold policy before capture.

        Returns:
            Stable plan-owned int32 storage with shape ``[total_q, top_k]``.
            Each row contains request-local KV indices, and ``cu_q`` maps rows
            to requests. A row with fewer than ``top_k`` causal keys retains
            every available key and pads the suffix with -1; rows with no
            visible key are entirely -1. A later call overwrites the same
            storage.

        ``run`` performs no CuTe compilation or transient CUDA allocation after
        a successful plan and is safe to capture and replay in a CUDA graph.
        Calls on this wrapper must not overlap.
        """
        self._check_math_mode()
        if self._empty_physical:
            assert self._selected is not None
            return self._selected.fill_(-1)
        if (
            self._prepared is None
            or self._repair_prepared is None
            or self._candidate_prepared is None
        ):
            raise RuntimeError("plan must be called before run")
        assert self._sample_prepared is not None
        assert self._cu_q is not None
        assert self._cu_kv is not None
        assert self._sample_fused is not None
        assert self._sample_block_table is not None
        assert self._sample_scores is not None
        assert self._sample_scratch is not None
        assert self._sample_rank_capacity > 0
        assert self._sample_rank_indices is not None
        assert self._sample_ranks is not None
        assert self._thresholds is not None
        assert self._candidate_values is not None
        assert self._candidate_counts is not None
        assert self._candidate_lengths is not None
        assert self._repair_flags is not None
        assert self._repair_starts is not None
        assert self._repair_ends is not None
        assert self._repair_thresholds is not None
        assert self._repair_counts is not None
        assert self._repair_lengths is not None
        assert self._effective_lengths is not None
        assert self._second_repair_flags is not None
        assert self._row_ends is not None
        assert self._topk_row_states is not None
        assert self._repair_scores is not None
        assert self._repair_topk_positions_i32 is not None
        assert self._repair_topk_compiled is not None
        assert self._packed_topk_launch is not None
        assert self._repair_resolve_launch is not None
        assert self._sample_threshold_launch is not None
        assert self._histograms is not None
        assert self._prefixes is not None
        assert self._ranks is not None
        assert self._threshold_bits is not None
        assert self._radix_values is not None
        assert self._greater_indices is not None
        assert self._greater_counts is not None
        assert self._equal_indices is not None
        assert self._equal_counts is not None
        assert self._selected is not None
        assert self._exact_error is not None

        if not _candidate_counts_precleared:
            # Direct engine users do not pass through the capacity layer's
            # scratch reset. No conditional graph predicate resets these now.
            torch._foreach_zero_([self._second_repair_flags, self._exact_error])

        _validate_prepared(self._candidate_prepared, q, kv_fused, block_table)
        if (
            tuple(weights.shape) != (self._rows, self._candidate_prepared.num_heads)
            or weights.dtype != torch.float32
            or weights.device != q.device
            or not weights.is_contiguous()
        ):
            raise ValueError(
                "weights must be contiguous FP32 storage matching the prepared problem"
            )

        stream = _current_stream(q.device)
        capturing = torch.cuda.is_current_stream_capturing()
        capture_stream = torch.cuda.current_stream(q.device) if capturing else None
        if self._needs_sampling:
            # The direct sampler writes every eligible score. Threshold kernels
            # read only _sample_ends, so the unused suffix needs no per-run clear.
            _fp8_mqa_selective_logits(
                q,
                self._sample_fused,
                weights,
                block_table,
                self._sample_prepared,
                mode=_SelectiveLogitsMode.SAMPLE,
                capacity=_SAMPLE_CAPACITY,
                values=self._sample_scores,
                index_counts=self._sample_scratch,
                sample_page_indices=(
                    self._sample_page_indices
                    if self._sample_sparse_metadata is None
                    else None
                ),
                _sparse_sample_metadata=self._sample_sparse_metadata,
                _sparse_sample_load_warps=4
                if self._sample_sparse_metadata is not None
                else 0,
                _sparse_sample_group_rows=_select_sparse_sample_group_rows(
                    self._sample_prepared.query_tile
                ),
            )
            if capturing:
                _check_capture_status(capture_stream, "sample scorer")
            sample_threshold_compiled, sample_threshold_args = (
                self._sample_threshold_launch
            )
            sample_threshold_compiled(*sample_threshold_args, stream)
            if capturing:
                _check_capture_status(capture_stream, "sample threshold ladder")
        elif not _sampling_policy_initialized:
            self._thresholds.fill_(float("-inf"))
            self._sample_ladder.fill_(float("-inf"))
            self._selected_bins.fill_(_SAMPLE_HISTOGRAM_BINS - 1)

        _fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            block_table,
            self._candidate_prepared,
            mode=_SelectiveLogitsMode.CANDIDATE,
            capacity=self._candidate_schedule.candidate_capacity,
            values=self._candidate_values,
            index_counts=self._candidate_counts,
            policy_values=self._thresholds,
            _candidate_counts_precleared=_candidate_counts_precleared,
            _h64_candidate=self._candidate_schedule.h64_candidate,
        )
        if capturing:
            _check_capture_status(capture_stream, "candidate scorer")
        if not ((self._rows > 1) and self._candidate_counts.shape[1] > 1):
            _finalize_fp8_mqa_packed_candidates(
                self._candidate_counts,
                self._candidate_values,
                self._candidate_lengths,
                self._repair_flags,
                rows=self._rows,
                capacity=self._candidate_schedule.candidate_capacity,
                top_k=self._top_k,
                _candidate_prepared=self._candidate_prepared,
                row_ends=self._row_ends,
            )
            if capturing:
                _check_capture_status(capture_stream, "candidate finalizer")
        packed_topk_compiled, packed_topk_args = self._packed_topk_launch
        packed_topk_compiled(*packed_topk_args, stream)
        if capturing:
            _check_capture_status(capture_stream, "packed TopK")

        # Every final-output writer pads short rows with -1 for every ratio.
        selected = self._enqueue_repair(q, kv_fused, weights, block_table)
        if capturing:
            _check_capture_status(capture_stream, "repair enqueue")
        return selected

    def _run_repair(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Enqueue the unchanged first and terminal repair stages."""
        assert self._repair_prepared is not None
        assert self._prepared is not None
        assert self._cu_q is not None
        assert self._physical_cu_kv is not None
        assert self._repair_starts is not None
        assert self._repair_ends is not None
        assert self._repair_flags is not None
        assert self._repair_thresholds is not None
        assert self._candidate_values is not None
        assert self._repair_counts is not None
        assert self._repair_lengths is not None
        assert self._repair_topk_positions_i32 is not None
        assert self._second_repair_flags is not None
        assert self._prefixes is not None
        assert self._ranks is not None
        assert self._threshold_bits is not None
        assert self._histograms is not None
        assert self._exact_indices is not None
        assert self._exact_counts is not None
        assert self._selected is not None
        stream = _current_stream(q.device)
        repair_threshold_compiled, repair_threshold_args = self._repair_threshold_launch
        if not ((self._rows > 1) and self._compact_repair):
            repair_threshold_compiled(*repair_threshold_args, stream)
        if self._compact_repair:
            assert self._repair_active_count is not None
            assert self._packed_repair_ends is not None
            assert self._repair_row_compact_launch is not None
            assert self._repair_row_pack_launch is not None
            assert self._repair_cu_q is not None
            assert self._repair_q is not None
            assert self._repair_weights is not None
            assert self._packed_repair_starts is not None
            assert self._packed_repair_thresholds is not None
            self._reset_compact_repair(terminal=False)
            compact_compiled, compact_args = self._repair_row_compact_launch
            compact_compiled(*compact_args, stream)
            pack_compiled, pack_args = self._repair_row_pack_launch
            if not (self._rows > 1):
                pack_compiled(*pack_args, stream)
            _refresh_fp8_mqa_selective_logits(
                self._repair_prepared,
                self._repair_cu_q,
                self._physical_cu_kv,
                row_starts=self._packed_repair_starts,
                row_ends=self._packed_repair_ends,
                active_rows=self._repair_active_count,
            )
            repair_q = self._repair_q
            repair_weights = self._repair_weights
            repair_thresholds = self._packed_repair_thresholds
        else:
            _refresh_fp8_mqa_selective_logits(
                self._repair_prepared,
                self._cu_q,
                self._physical_cu_kv,
                row_starts=self._repair_starts,
                row_ends=self._repair_ends,
            )
            repair_q = q
            repair_weights = weights
            repair_thresholds = self._repair_thresholds
        _fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            block_table,
            self._repair_prepared,
            mode=_SelectiveLogitsMode.REPAIR,
            capacity=_REPAIR_CAPACITY,
            values=self._candidate_values,
            index_counts=self._repair_counts,
            policy_values=repair_thresholds,
        )
        _finalize_fp8_mqa_packed_candidates(
            self._repair_counts,
            self._candidate_values,
            self._repair_lengths,
            (
                self._compact_repair_flags
                if self._compact_repair
                else self._second_repair_flags
            ),
            rows=self._rows,
            capacity=_REPAIR_CAPACITY,
            top_k=self._top_k,
        )
        self._repair_topk_compiled(
            self._repair_scores,
            self._topk_row_states,
            self._repair_lengths,
            self._repair_topk_positions_i32,
            None,
            self._repair_active_count if self._compact_repair else self._repair_flags,
        )
        if self._compact_repair:
            assert self._second_active_count is not None
            assert self._second_repair_row_pack_launch is not None
            assert self._compact_repair_resolve_launch is not None
            # Compact resolution only writes active slots. Clear stale ranks
            # before building the next repair queue.
            self._reset_compact_repair(terminal=True)
            resolve_compiled, resolve_args = self._compact_repair_resolve_launch
            resolve_compiled(*resolve_args, stream)
            second_pack_compiled, second_pack_args = self._second_repair_row_pack_launch
            if not (self._rows > 1):
                second_pack_compiled(*second_pack_args, stream)
            _refresh_fp8_mqa_selective_logits(
                self._prepared,
                self._repair_cu_q,
                self._physical_cu_kv,
                row_starts=self._packed_repair_starts,
                row_ends=self._packed_repair_ends,
                active_rows=self._second_active_count,
            )
        else:
            repair_resolve_compiled, repair_resolve_args = self._repair_resolve_launch
            repair_resolve_compiled(*repair_resolve_args, stream)
            _refresh_fp8_mqa_selective_logits(
                self._prepared,
                self._cu_q,
                self._physical_cu_kv,
                row_starts=self._repair_starts,
                row_ends=self._repair_ends,
            )
        radix_compiled, radix_args = self._radix_select_launch
        for shift in (24, 16, 8, 0):
            _fp8_mqa_selective_logits(
                repair_q,
                kv_fused,
                repair_weights,
                block_table,
                self._prepared,
                mode=_SelectiveLogitsMode.RADIX_HISTOGRAM,
                capacity=1,
                values=self._radix_values,
                index_counts=self._histograms,
                policy_values=self._prefixes,
                radix_shift=shift,
                _clear_counts=shift == 24,
            )
            radix_compiled(*radix_args[:-1], cutlass.Int32(shift), stream)

        thresholds = self._threshold_bits.view(torch.float32)
        _fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            block_table,
            self._prepared,
            mode=_SelectiveLogitsMode.EXACT,
            capacity=self._top_k,
            values=self._exact_indices,
            index_counts=self._exact_counts,
            policy_values=thresholds,
        )
        if self._compact_repair:
            assert self._compact_combine_exact_launch is not None
            combine_compiled, combine_args = self._compact_combine_exact_launch
        else:
            combine_compiled, combine_args = self._combine_exact_launch
        combine_compiled(*combine_args, stream)

        return self._selected


def _public_wrapper_method(method):
    """Register logging and traces under the unchanged public API identity."""
    method.__module__ = "flashinfer.attn_scores.selective_logits"
    method.is_experimental = True
    return method


class FP8PagedMQATopKWrapper(_PublicWrapper):
    """Experimental exact paged FP8 MQA TopK with an explicit scoring strategy.

    ``strategy="full"`` materializes the production paged logits and applies
    public GVR-v2 TopK512, TopK1024 or TopK2048. ``strategy="selective"`` uses bounded
    candidate publication and exact repair. The strategy is explicit and
    remains fixed for the wrapper lifetime; no sequence-length routing occurs.

    Selective specializes heads, dimension, page size and TopK at
    construction, which also reserves its workspace for ``max_context_length``,
    ``max_chunk_size`` and ``max_batch_size`` and one KV pool layout, taken from
    an unretained ``kv_layout`` exemplar or from ``num_pages`` and ``device``.
    ``plan(qo_indptr, kv_indptr, block_table)`` binds step metadata;
    ``run(q, weights, kv_fused, *, out=None)`` shares the workspace across layers.
    Full uses the same constructor, step-plan and layer-run signatures;
    its host-planned groups require plan outside capture. Both return stable
    request-local int32 indices with shape ``[supplied_q_rows, top_k]``.

    Full always uses GVR-v2 and owns independent per-plan scratch. Selective
    supports TopK1..2048 with fixed candidate and repair selectors.
    Neither strategy selects or falls back to another full-score backend.

    Under caller capture, exact repair is recorded as a CUDA conditional node
    in the caller's graph so repair-free replays skip it. The wrapper never
    creates, instantiates or replays a graph of its own. Eager selective runs
    enqueue the device-flagged repair sequence directly.

    ``compression_ratio=4`` consumes already-compressed KV for the selective
    strategy. Public KV prefixes remain in original-token units; each row's
    compressed end is floor((L-Q+i+1)/4). Returned IDs index compressed slots,
    with -1 suffix padding including entirely empty rows. Default ratio1
    uses uncompressed physical keys. The ratio is immutable; no
    compressor, query reduction or partial block is implied.

    ``math_mode="relu_strict"`` (default) uses range-safe NaN-aware max and
    full scale across full scoring, sampling and every repair phase.
    ``"relu_fast"`` opts into packed FP32 x+abs(x) with one half-scale compensation.
    Fast doubled intermediates can overflow, tiny half-scales can underflow,
    and -inf becomes NaN instead of zero. Both local ReLUs propagate NaN
    without a payload guarantee and map signed zero to +0. Selective ranks
    final NaN scores as -inf in every phase; NaNs tie with genuine -inf and
    may supply valid indices when fewer than K higher scores exist. Full
    keeps the public scorer's NaN behavior. Whole results need not be
    bit-identical. The selected recipe is immutable for
    this wrapper's lifetime, including replan and graph capture. Construct
    another wrapper to change it. TopK is exact for the selected score recipe.
    """

    def _initialize(
        self,
        *,
        strategy: Literal["full", "selective"],
        num_heads: int = 8,
        head_dim: int = 128,
        page_size: int = 128,
        top_k: int = 512,
        math_mode: str = "relu_strict",
        compression_ratio: int = 1,
        max_context_length: int | None = None,
        max_chunk_size: int | None = None,
        max_batch_size: int | None = None,
        kv_layout: torch.Tensor | None = None,
        num_pages: int | None = None,
        device: torch.device | str | int | None = None,
    ) -> None:
        """Specialize either strategy and reserve its fixed input capacities.

        Both strategies require ``max_context_length``, ``max_chunk_size`` and
        ``max_batch_size`` plus the KV pool layout: either ``kv_layout``, a
        contiguous CUDA uint8 ``[num_pages, page_size, 1, head_dim + 4]``
        exemplar whose shape, strides, dtype and device are read without
        retaining it, or ``num_pages`` and ``device`` for a contiguous pool
        with standard strides. Context bounds count original tokens; chunk size
        bounds total supplied Q rows, not rows per request. The owned table has
        ``max_batch_size`` rows and enough columns for the maximum per-request
        physical extent. Construction allocates the step workspace outside
        CUDA graph capture without compiling kernels. The first ``prepare_pool``
        or eager ``run`` compiles scoring kernels and reserves score scratch.
        Full requires ``plan`` before preparation and reads prefixes on the host;
        each new full plan invalidates the previous pool bindings. Selective can
        prepare the layout before its first step and refresh metadata in capture.
        Unsupported
        scoring specializations are rejected when that chain is first built;
        an empty physical capacity returns -1 results without building it.
        Construction validates capacities, layout and device, but does not
        validate the scoring kernel specialization. Construction requires CUDA.
        Capacities and layout are immutable:
        construct another wrapper to change them. Both strategies use these bounds.

        In a serving engine, construct the selective wrapper after allocating
        the indexer KV cache and before graph capture. Recreate it when the
        engine reallocates its cache. For same-layout layer-pool swaps within
        an allocation, discard caller graphs, call ``clear_pools()`` and
        recapture. Pools must retain the constructed shape, strides, dtype and
        device.

        Args:
            strategy: ``"full"`` for public logits plus GVR-v2, or
                ``"selective"`` for bounded candidates with exact repair.
            num_heads: Number of query heads for scoring.
            head_dim: Query/key dimension, divisible by 32.
            page_size: KV tokens per physical page: 32, 64 or 128.
            top_k: Output width: selective 1..2048; full 512, 1024 or 2048.
            math_mode: Immutable ``"relu_strict"`` or ``"relu_fast"`` policy.
            compression_ratio: Original tokens per stored key, 1 or 4.
                Full supports only 1.
            max_context_length: Maximum original-token context per request.
            max_chunk_size: Maximum total supplied query rows.
            max_batch_size: Maximum number of requests per step.
            kv_layout: Optional contiguous CUDA uint8 KV layout exemplar of
                shape ``[num_pages, page_size, 1, head_dim + 4]``. Not retained.
            num_pages: Physical page capacity, instead of an exemplar.
            device: CUDA device for ``num_pages``; not used with ``kv_layout``.

        Returns:
            None.

        Raises:
            ValueError: A strategy, argument value, capacity, layout or device
                is unsupported. Selective requires SM100 or SM103. Unsupported
                scoring kernel specializations are rejected when the first
                ``prepare_pool`` or eager ``run`` builds a nonempty chain.
            RuntimeError: Construction occurs during CUDA graph capture.
        """
        math_mode = _validate_math_mode(math_mode)
        if strategy not in ("full", "selective"):
            raise ValueError("strategy must be 'full' or 'selective'")
        if type(compression_ratio) is not int or compression_ratio not in (1, 4):
            raise ValueError("compression_ratio must be 1 or 4")
        if strategy == "full" and compression_ratio != 1:
            raise ValueError("compression_ratio is only supported by selective")
        capacities = (max_context_length, max_chunk_size, max_batch_size)
        if any(value is None for value in capacities):
            raise ValueError(
                "both strategies require max_context_length, max_chunk_size and max_batch_size"
            )
        if strategy == "full" and top_k not in (512, 1024, 2048):
            raise ValueError("full strategy requires top_k in (512, 1024, 2048)")
        self._math_mode = math_mode
        self._math_mode_at_construction = math_mode
        self._strategy = strategy
        self._compression_ratio = compression_ratio
        self._compression_ratio_at_construction = compression_ratio
        self._capacity_plan = None
        if strategy == "selective":
            from .selective_capacity import _CapacitySelectiveTopKPlan

            self._capacity_plan = _CapacitySelectiveTopKPlan(
                max_context_length=max_context_length,
                max_chunk_size=max_chunk_size,
                max_batch_size=max_batch_size,
                kv_layout=kv_layout,
                num_pages=num_pages,
                device=device,
                num_heads=num_heads,
                head_dim=head_dim,
                page_size=page_size,
                top_k=top_k,
                compression_ratio=compression_ratio,
                math_mode=math_mode,
            )
        else:
            from .selective_capacity import (
                _CapacitySelectiveTopKPlan,
                _CapacityStepInputs,
                _SelectiveCapacity,
            )

            capacity = _SelectiveCapacity(
                num_heads,
                head_dim,
                page_size,
                compression_ratio,
                max_context_length,
                max_chunk_size,
                max_batch_size,
            )
            layout, device, pages = _CapacitySelectiveTopKPlan._resolve_layout(
                capacity, kv_layout, num_pages, device
            )
            self._full_state = _CapacityStepInputs(
                capacity, device=device, num_pages=pages
            )
            self._full_state.q.zero_()
            self._full_state.weights.zero_()
            self._full_pool_layout = layout
            self._full_pool_plans: dict[tuple[Any, ...], dict[str, Any]] = {}
            self._full_step_is_prepared = False
            self._full_num_heads, self._full_head_dim, self._full_top_k = (
                num_heads,
                head_dim,
                top_k,
            )
            self._full_supplied_rows = max_chunk_size
            self._full_batches = max_batch_size
            self._full_q_payload = self._full_state.q
            self._full_weight_payload = self._full_state.weights
            self._full_block_table = self._full_state.block_table
            self._full_repair_error = torch.zeros(1, dtype=torch.int32, device=device)
        self._full_groups: list[
            tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._full_scores: torch.Tensor | None = None
        self._full_row_ends: torch.Tensor | None = None
        self._full_binding_identity: tuple[Any, ...] | None = None
        self._full_max_context_len = 0
        self._full_impl: Any = None
        self._full_gvr2_workspace: torch.Tensor | None = None
        self._selected: torch.Tensor | None = None

    @property
    def math_mode(self) -> _MathMode:
        """Immutable scoring policy used by the selected strategy."""
        return self._math_mode

    def _check_math_mode(self) -> None:
        if self._math_mode != self._math_mode_at_construction:
            raise RuntimeError("math_mode is immutable; construct a new wrapper")

    @property
    def strategy(self) -> Literal["full", "selective"]:
        """The caller-selected scoring strategy."""
        return self._strategy

    @property
    def compression_ratio(self) -> int:
        """Immutable number of original tokens represented by each stored key."""
        return self._compression_ratio

    def _check_compression_ratio(self) -> None:
        if self._compression_ratio != self._compression_ratio_at_construction:
            raise RuntimeError(
                "compression_ratio is immutable; construct a new wrapper"
            )

    def _allocate_full_plan(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        block_table: torch.Tensor,
        *,
        top_k: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Prepare full paged logits and exact per-row TopK."""
        from ...topk_varlen.topk_varlen import top_k_varlen
        from ...attn_scores.attn_scores import (
            compute_paged_mqa_logits_schedule,
            fp8_paged_mqa_logits,
            padded_seq_len,
        )

        _validate_selective_top_k(top_k)
        if top_k not in (512, 1024, 2048):
            raise ValueError(
                "strategy='full' uses GVR-v2 and requires top_k in (512, 1024, 2048)"
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(head_dim, int) or head_dim <= 0 or head_dim % _FP8_MMA_INST_K:
            raise ValueError(
                f"head_dim must be a positive multiple of {_FP8_MMA_INST_K}"
            )
        if q.dim() != 3 or tuple(q.shape[1:]) != (num_heads, head_dim):
            raise ValueError(
                "q must have shape [total_q, num_heads, head_dim] matching "
                "num_heads and head_dim"
            )
        if weights.shape != (q.shape[0], num_heads):
            raise ValueError("weights must have shape [total_q, num_heads]")
        if cu_q.numel() < 2 or cu_kv.numel() != cu_q.numel():
            raise ValueError("cu_q and cu_kv must have matching [batch+1] extents")
        if block_table.dim() != 2 or block_table.shape[0] != cu_q.numel() - 1:
            raise ValueError("block_table must have one row per request")

        q_prefix = cu_q.tolist()
        kv_prefix = cu_kv.tolist()
        if q_prefix[0] != 0 or kv_prefix[0] != 0:
            raise ValueError("cu_q and cu_kv must start at zero")
        if q_prefix[-1] > q.shape[0]:
            raise ValueError("qo_indptr[-1] must not exceed supplied Q rows")
        if any(a > b for a, b in zip(q_prefix, q_prefix[1:], strict=False)):
            raise ValueError("cu_q must be monotonically nondecreasing")
        if any(a > b for a, b in zip(kv_prefix, kv_prefix[1:], strict=False)):
            raise ValueError("cu_kv must be monotonically nondecreasing")

        row_ends: list[int] = []
        chunks: list[tuple[int, int, int, int]] = []
        max_next_n = _COMPUTE_BLOCK_KV // num_heads
        if max_next_n < 1:
            raise ValueError("num_heads exceeds the full paged-logits N envelope")
        for request, (q_begin, q_end, kv_begin, kv_end) in enumerate(
            zip(
                q_prefix[:-1],
                q_prefix[1:],
                kv_prefix[:-1],
                kv_prefix[1:],
                strict=True,
            )
        ):
            q_len = q_end - q_begin
            kv_len = kv_end - kv_begin
            if q_len == 0:
                continue
            if kv_len < q_len:
                raise ValueError(
                    "full causal TopK requires nonempty Q and every KV length >= Q"
                )
            first_end = kv_len - q_len + 1
            row_ends.extend(range(first_end, kv_len + 1))
            chunk_begin = q_begin
            while chunk_begin < q_end:
                next_n = min(max_next_n, q_end - chunk_begin)
                chunk_end = chunk_begin + next_n
                context_end = first_end + (chunk_end - q_begin) - 1
                chunks.append((chunk_begin, chunk_end, request, context_end))
                chunk_begin = chunk_end

        # Invalidate before replacing storage; a failed rebuild must not
        # execute bindings from the previous plan generation.
        self._full_binding_identity = None
        self._full_impl = None
        self._full_gvr2_workspace = None
        self._rows = q.shape[0]
        self._batches = len(q_prefix) - 1
        self._top_k = top_k
        self._kv_rows = kv_prefix[-1]
        self._cu_q = cu_q
        self._cu_kv = cu_kv
        row_ends.extend([0] * (q.shape[0] - len(row_ends)))
        self._full_max_context_len = max(row_ends, default=0)
        self._full_row_ends = torch.tensor(row_ends, dtype=torch.int32, device=q.device)
        score_columns = max(padded_seq_len(self._full_max_context_len), self._top_k)
        # Keep GVR2 on its streaming, spill-free path. Logical
        # row lengths still exclude padding from score selection.
        score_columns = max(score_columns, 8192)
        self._full_scores = torch.empty(
            (self._rows, score_columns),
            dtype=torch.float32,
            device=q.device,
        )
        self._selected = torch.empty(
            (self._rows, top_k), dtype=torch.int32, device=q.device
        )

        self._full_groups = []
        group_begin = 0
        while group_begin < len(chunks):
            q_begin, _, _, _ = chunks[group_begin]
            next_n = chunks[group_begin][1] - q_begin
            group_end = group_begin + 1
            while group_end < len(chunks):
                prev = chunks[group_end - 1]
                current = chunks[group_end]
                if current[0] != prev[1] or current[1] - current[0] != next_n:
                    break
                group_end += 1
            selected_chunks = chunks[group_begin:group_end]
            q_end = selected_chunks[-1][1]
            context_lens = torch.tensor(
                [chunk[3] for chunk in selected_chunks],
                dtype=torch.int32,
                device=q.device,
            )
            request_ids = torch.tensor(
                [chunk[2] for chunk in selected_chunks],
                dtype=torch.int64,
                device=block_table.device,
            )
            group_block_table = block_table.index_select(0, request_ids).contiguous()
            schedule = compute_paged_mqa_logits_schedule(
                context_lens, device=q.device, next_n=next_n, variant="fp8"
            )
            self._full_groups.append(
                (
                    q_begin,
                    q_end,
                    next_n,
                    group_block_table,
                    context_lens,
                    schedule,
                    request_ids,
                )
            )
            group_begin = group_end

        full_scores = self._full_scores
        full_row_ends = self._full_row_ends
        selected = self._selected
        full_groups = tuple(self._full_groups)
        full_max_context_len = self._full_max_context_len
        selected_top_k = self._top_k
        full_math_mode = self.math_mode
        from ...topk_varlen.kernels.gvr2_topk_host import workspace_bytes

        # GVR2's default scratch is stream-local, but warmup and capture
        # use different streams. Own one zeroed slab across both and any
        # enclosing graph; concurrent wrappers must not share this scratch.
        full_gvr2_workspace = torch.zeros(
            workspace_bytes(), dtype=torch.uint8, device=q.device
        ).view(torch.int32)
        full_topk_workspace = {"gvr2_workspace": full_gvr2_workspace}
        # In-range, evenly spaced hints enable the native self-sampling
        # selector. They steer sampling, never determine the exact result.
        # Use int64 during construction to avoid overflowing long row ends.
        hint_columns = torch.arange(selected_top_k, dtype=torch.int64, device=q.device)
        full_hints = (
            full_row_ends[:, None] * hint_columns[None, :] // selected_top_k
        ).to(torch.int32)

        def full_impl() -> torch.Tensor:
            if not full_groups:
                selected.fill_(-1)
                return selected
            for (
                q_begin,
                q_end,
                next_n,
                group_block_table,
                context_lens,
                schedule,
                request_ids,
            ) in full_groups:
                # Page mappings are per-run payload, including graph replay.
                torch.index_select(block_table, 0, request_ids, out=group_block_table)
                batch = (q_end - q_begin) // next_n
                fp8_paged_mqa_logits(
                    q[q_begin:q_end].view(batch, next_n, num_heads, head_dim),
                    kv_fused,
                    weights[q_begin:q_end],
                    group_block_table,
                    context_lens,
                    full_max_context_len,
                    math_mode=full_math_mode,
                    schedule_meta=schedule,
                    out=full_scores[q_begin:q_end],
                )
            # Consume causal lengths directly and write the stable int32 output;
            # a dense torch.topk would scan padding and require an index copy.
            top_k_varlen(
                full_scores,
                full_row_ends,
                selected_top_k,
                pre_idx=full_hints,
                out_indices=selected,
                workspace=full_topk_workspace,
                backend="gvr_2",
            )
            # GVR-v2 keeps every key of a short row and pads its suffix with -1.
            selected[q_prefix[-1] :].fill_(-1)
            return selected

        new_identity = _selective_logits_launch_identity(
            (q, kv_fused, weights, block_table), 0
        )
        full_impl()
        torch.cuda.synchronize(q.device)
        self._full_gvr2_workspace = full_gvr2_workspace
        self._full_impl = full_impl
        self._full_binding_identity = new_identity

    def _plan_full_payload(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        block_table: torch.Tensor,
        *,
        top_k: int = _DEFAULT_TOP_K,
        num_heads: int = 8,
        head_dim: int = 128,
    ) -> None:
        """Prepare exact causal TopK on Blackwell SM100/SM103 outside graph capture.

        Args:
            q: Contiguous CUDA FP8 E4M3 queries ``[total_q, num_heads, head_dim]``.
            kv_fused: Contiguous CUDA uint8 storage
                ``[num_pages, page_size, 1, head_dim + 4]``. Each physical page
                contains all FP8 keys followed by one FP32 scale per token;
                the bytes are not interleaved per token. Page size is 32, 64,
                or 128. Tables need one entry per logical page.
            weights: Contiguous CUDA FP32 weights ``[total_q, num_heads]``.
            cu_q: Contiguous CUDA int32 query prefix sums ``[batch + 1]``.
                Requires at least one query in every request.
            cu_kv: Contiguous CUDA int32 KV prefix sums ``[batch + 1]``.
                Prefixes start at zero and are nondecreasing; each request
                must have at least as many original KV tokens as query rows.
            block_table: Contiguous CUDA int32 physical page indices
                ``[batch, max_pages]``, with valid entries for logical pages.
            top_k: Integer output width 512, 1024 or 2048, default 512.
            num_heads: Scoring head count, default 8; must admit a legal
                paged-logits compute tile for the selected strategy.
            head_dim: Head dimension, a positive multiple of 32, default 128;
                the selected tile must fit device shared memory.

        All tensors must share one device. Planning reads prefixes on the host
        and allocates/precompiles run resources. Replans replace storage.
        Full owns 20,973,568 bytes of GVR-v2 scratch. Each successful
        plan owns separate zero-initialized storage held across eager calls
        and caller-owned graph captures. Successful selector launches reset their
        consumed state; this is not a lock or CUDA-context-fault recovery.
        Replan after changing prefixes, tensor objects, storage/layout, or
        specialization, and recapture any enclosing graph after replanning.
        A failed rebuild invalidates run until a successful plan; early
        validation failures can leave the prior plan usable. Synchronize prior
        work before replanning.
        """
        self._check_math_mode()
        self._check_compression_ratio()
        device = q.device if isinstance(q, torch.Tensor) and q.is_cuda else -1
        # Retain retired bindings until replacement preparation finishes,
        # including on a failed rebuild.
        previous_plan = tuple(self.__dict__.values())
        try:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "full strategy plan must be called outside CUDA graph capture"
                    )
                self._allocate_full_plan(
                    q,
                    kv_fused,
                    weights,
                    cu_q,
                    cu_kv,
                    block_table,
                    top_k=top_k,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
        finally:
            del previous_plan

    def _run_full_payload(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Return plan-owned int32 request-local indices ``[total_q, top_k]``.

        Args:
            q: The same query tensor object passed to plan, with unchanged
                storage, shape and layout.
            kv_fused: The same fused KV tensor object passed to plan, with
                unchanged storage and layout.
            weights: The same FP32 weights tensor object passed to plan, with
                unchanged storage and layout.
            block_table: The same page-map tensor object passed to plan, with
                unchanged storage and layout.

        Input contents, including page mappings, may change in place between
        calls. Prefix changes require replanning. Output is overwritten on the
        next run; clone it to retain a result. Ordering is unspecified, ties may
        select equivalent keys, and underfilled rows retain all causal keys
        with remaining slots filled with -1.

        A successful plan is required. Eager and caller-captured calls enqueue
        the prepared operations directly, except selective caller capture
        records exact repair in a conditional body. Calls
        on one wrapper must not overlap; synchronize before switching streams
        or replanning. Replacing a tensor object, even with a view sharing the
        same storage, or changing storage/layout raises ValueError. Recapture
        enclosing graphs after replanning; they keep the previous bindings.
        If outer capture fails, discard that graph before retrying. A failed
        plan rebuild remains invalid until a successful plan.
        Enclosing capture must use a stream on the input tensors' device.
        """
        self._check_math_mode()
        self._check_compression_ratio()
        device = q.device if isinstance(q, torch.Tensor) and q.is_cuda else -1
        with torch.cuda.device(device):
            if self._full_binding_identity is None or self._full_impl is None:
                raise RuntimeError("plan must be called before run")
            identity = _selective_logits_launch_identity(
                (q, kv_fused, weights, block_table), 0
            )
            if identity != self._full_binding_identity:
                raise ValueError("run tensors must match the address-stable plan")
            return self._full_impl()

    @flashinfer_api
    @_public_wrapper_method
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        block_table: torch.Tensor,
        *,
        num_query_rows: int | None = None,
    ) -> None:
        """Bind the current step's prefixes and page map for every layer.

        Args:
            qo_indptr: Contiguous CUDA int32 query prefix sums ``[B + 1]``.
            kv_indptr: Contiguous CUDA int32 original-token KV prefix sums ``[B + 1]``.
            block_table: Contiguous CUDA int32 page map, with at least
                ``ceil((max_context_length // compression_ratio) / page_size)`` columns.
            num_query_rows: Supplied Q extent, default ``max_chunk_size``.
                Active query prefixes may end below this extent.

        Returns:
            None. The wrapper owns staged metadata and reset status buffers.

        Raises:
            ValueError: A prefix, map layout or supplied extent is invalid.
            RuntimeError: Full planning occurs during capture, or selective
                capture requires a scoring layout that has not been prepared.

        Selective refreshes device metadata without host value reads and can
        be captured after the layout is prepared. Full uses host prefix values
        to build its scorer groups, so call its plan outside capture and warm
        each pool before capturing run. Full planning synchronizes previous
        work before replacing its per-pool score buffers and bindings. Discard
        caller graphs from the previous full plan before replanning; they must
        be recaptured. Both share step metadata across layers.
        """
        self._check_math_mode()
        self._check_compression_ratio()
        if self._capacity_plan is not None:
            return self._capacity_plan.plan(
                qo_indptr, kv_indptr, block_table, num_query_rows=num_query_rows
            )
        state = self._full_state
        with torch.cuda.device(state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "full strategy plan must be called outside CUDA graph capture"
                )
            if self._full_pool_plans:
                torch.cuda.synchronize(state.device)
            rows = (
                state.capacity.max_chunk_size
                if num_query_rows is None
                else num_query_rows
            )
            state.stage_step(qo_indptr, kv_indptr, block_table, num_query_rows=rows)
            if not state.is_prepared:
                state.prepare()
            state.refresh()
            self._full_repair_error.zero_()
            self._full_pool_plans.clear()
            self._full_supplied_rows = rows
            self._full_batches = qo_indptr.numel() - 1
            self._full_block_table = state.block_table[: self._full_batches]
            self._full_q_payload = state.q[:rows]
            self._full_weight_payload = state.weights[:rows]
            self._full_step_is_prepared = True

    def _full_pool_key(self, kv_fused):
        from .selective_capacity import _CapacitySelectiveTopKPlan

        layout = _CapacitySelectiveTopKPlan._layout(kv_fused)
        if layout != self._full_pool_layout or not kv_fused.is_contiguous():
            raise ValueError("KV pool layout must match the constructed layout")
        return kv_fused.data_ptr(), layout

    def _restore_full_pool(self, binding):
        for name, value in binding.items():
            setattr(self, name, value)

    @flashinfer_api(trace=fp8_paged_mqa_topk_trace_dispatch)
    @_public_wrapper_method
    def run(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        kv_fused: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Enqueue the complete selected route and return stable int32 TopK.

        Selective: ``run(q, weights, kv_fused, *, out=None)`` accepts
        contiguous CUDA FP8 E4M3 Q[R,H,D], FP32 weights[R,H], and a contiguous
        uint8 pool with the planned shape, strides, dtype and device, including
        page count. R must equal plan's supplied row extent. The final
        device query prefix may be below R; inactive/short-row suffixes are -1.
        There is no host prefix-value read or synchronization on prepared pools.
        Inputs stage into stable buffers; matching ``input_buffers()`` views
        skip copying. Optional out is contiguous int32[R,K], on the same device,
        and cannot overlap inputs or other plan storage. The returned tensor is
        overwritten by later layers and steps; clone results that must
        survive. Ties may choose equivalent indices.

        ``error`` and ``repair_error`` are device status buffers, reset only by
        plan. Repair failures accumulate across layers. Invalid prefixes/maps/capacities produce -1 rows; exact-repair
        errors also fail closed. Inspect status after an appropriate dependency
        or synchronization; run does not synchronously raise device-value errors.

        Full uses the same layer-call signature and stages Q/weights into
        owned buffers. Bind the page map through plan for both strategies.
        Both strategies enqueue operations on the caller's stream. Selective
        caller capture records exact repair in a conditional node's body;
        neither strategy owns, instantiates or replays a parent CUDA graph.
        Captured copy addresses and supplied extents are fixed in that graph;
        change device prefix contents in place or recapture for new bindings.
        The graph addresses this wrapper's workspace: keep the wrapper (and
        its bound pools) alive while that graph may replay.
        Release wrappers and their plans outside all CUDA graph capture,
        including captures owned by other wrappers: retirement unloads private
        CUDA libraries immediately.
        Selective caller capture needs one earlier ``prepare_pool`` or eager run
        to compile the layout; it then binds any equal-layout pool in place.
        Use one wrapper per concurrent stream; calls must not overlap.

        Args:
            q: Contiguous CUDA FP8 E4M3 queries ``[R, H, D]``.
            weights: Contiguous CUDA FP32 per-head weights ``[R, H]``.
            kv_fused: Contiguous CUDA uint8 paged KV pool
                ``[num_pages, page_size, 1, D + 4]`` with the constructed layout.
            out: Optional CUDA int32 destination ``[R, top_k]``;
                must not overlap inputs or workspace.

        Returns:
            CUDA int32 request-local indices ``[R, top_k]`` with ``-1`` suffix
            padding. Without ``out``, subsequent calls overwrite the storage.

        Raises:
            ValueError: A tensor layout, supplied extent or output is invalid.
            RuntimeError: No step is planned, nothing was prepared before
                capture, or a planned full-strategy binding has changed.
        """
        self._check_math_mode()
        self._check_compression_ratio()
        if self._capacity_plan is not None:
            return self._capacity_plan.run(q, weights, kv_fused, out=out)
        if not self._full_step_is_prepared:
            raise RuntimeError("plan must be called before run")
        state = self._full_state
        state._payload_copies(q, weights)
        if q.shape[0] != self._full_supplied_rows:
            raise ValueError("Q rows must match plan num_query_rows")
        key = self._full_pool_key(kv_fused)
        if out is not None:
            if (
                not isinstance(out, torch.Tensor)
                or out.device != state.device
                or out.dtype != torch.int32
                or out.shape != (q.shape[0], self._full_top_k)
                or not out.is_contiguous()
            ):
                raise ValueError("out must be contiguous CUDA int32 [Q, top_k]")
            for tensor in (q, weights, kv_fused, *state.__dict__.values()):
                if isinstance(tensor, torch.Tensor) and _overlap(
                    _byte_range(out), _byte_range(tensor)
                ):
                    raise ValueError(
                        "out must not overlap inputs or plan-owned storage"
                    )
        with torch.cuda.device(state.device):
            if key not in self._full_pool_plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "prepare_pool or one eager run must precede caller capture for this full pool"
                    )
                self.prepare_pool(kv_fused)
            self._restore_full_pool(self._full_pool_plans[key])
            if (
                out is not None
                and self._full_scores is not None
                and _overlap(_byte_range(out), _byte_range(self._full_scores))
            ):
                raise ValueError("out must not overlap plan-owned scores")
            state.stage_payload(q, weights)
            selected = self._run_full_payload(
                self._full_q_payload,
                kv_fused,
                self._full_weight_payload,
                self._full_block_table,
            )
            if out is not None and out.data_ptr() != selected.data_ptr():
                out.copy_(selected)
            return selected if out is None else out

    def prepare_pool(self, kv_fused):
        """Warm a layer pool outside capture for the selected strategy.

        Selective shares tensor scratch across pools. The first preparation,
        or the first eager run, compiles metadata and
        scoring kernels for the layout and validates any step staged by an
        earlier plan call; one of them is required before the first caller
        capture. Later pools need no preparation: caller capture binds them in
        place. Bindings key on address and full layout and keep pool owners
        alive; the constructor's ``kv_layout``
        exemplar is not bound.

        Full requires plan first and owns a separate score buffer, GVR-v2
        workspace per prepared pool. Each full plan replaces these
        bindings, so prepare every pool again before caller capture in that
        step; eager run may prepare it on demand. Full pools share staged
        inputs and step metadata, not scoring scratch.
        """
        if self._capacity_plan is not None:
            return self._capacity_plan.prepare_pool(kv_fused)
        key = self._full_pool_key(kv_fused)
        state = self._full_state
        with torch.cuda.device(state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare_pool must run outside CUDA graph capture")
            if not self._full_step_is_prepared:
                raise RuntimeError("full plan must precede prepare_pool")
            if key in self._full_pool_plans:
                return
            try:
                self._plan_full_payload(
                    self._full_q_payload,
                    kv_fused,
                    self._full_weight_payload,
                    state.safe_qo_indptr[: self._full_batches + 1],
                    state.physical_kv_indptr[: self._full_batches + 1],
                    self._full_block_table,
                    top_k=self._full_top_k,
                    num_heads=self._full_num_heads,
                    head_dim=self._full_head_dim,
                )
            except BaseException:
                self._full_step_is_prepared = False
                raise
            fields = (
                "_full_groups",
                "_full_scores",
                "_full_row_ends",
                "_full_binding_identity",
                "_full_max_context_len",
                "_full_impl",
                "_full_gvr2_workspace",
                "_selected",
            )
            self._full_pool_plans[key] = {name: getattr(self, name) for name in fields}

    def clear_pools(self):
        """Synchronize and release cached pool bindings.

        Call outside capture after discarding caller graphs using these pools.
        Their captures must be rebuilt before reuse. Pool tensors and launch
        owners are released; capacities, compiled kernels and step status
        remain. Selective retains its shared tensor scratch and can bind later
        pools during capture after layout warmup. Full retains staging but
        requires fresh pool preparation outside capture.
        """
        if self._capacity_plan is not None:
            return self._capacity_plan.clear_pools()
        with torch.cuda.device(self._full_state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("clear_pools must run outside CUDA graph capture")
            torch.cuda.synchronize(self._full_state.device)
            self._full_pool_plans.clear()
            self._full_impl = self._full_binding_identity = None

    def input_buffers(self) -> dict[str, torch.Tensor]:
        """Return Q/weight/prefix staging tensors owned by either strategy.

        Pass prefix/map views to plan to publish row/batch extents;
        pass Q/weight views to run to skip their staging copies.
        Contents change with subsequent calls; the views stay valid for the
        wrapper's lifetime.
        """
        return (
            self._capacity_plan.input_buffers()
            if self._capacity_plan is not None
            else self._full_state.input_buffers()
        )

    @property
    def error(self) -> torch.Tensor:
        """Int32[1] device validation status for either strategy: bits 1/2/4/8.

        Bits mean extents/prefixes/capacity/page IDs. Extent errors suppress
        other checks; prefix/capacity errors suppress page-map reads.
        """
        return (
            self._capacity_plan.error
            if self._capacity_plan is not None
            else self._full_state.error
        )

    @property
    def repair_error(self) -> torch.Tensor:
        """Int32[1] OR-accumulated repair flag; full always returns zero.

        This is not a row count. It resets only at the next per-step plan.
        """
        return (
            self._capacity_plan.repair_error
            if self._capacity_plan is not None
            else self._full_repair_error
        )

    @property
    def active_shape(self) -> torch.Tensor:
        """Int32[2] active row/request counts; zeros on validation error."""
        return (
            self._capacity_plan.active_shape
            if self._capacity_plan is not None
            else self._full_state.active_shape
        )


__all__ = ["FP8PagedMQATopKWrapper"]
