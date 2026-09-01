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

"""Compact public-interface acceptance coverage for PrimTS FMHA decode."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
from typing import Optional, Sequence
import warnings

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl==4.7.0",
)

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float8E4M3FN, Int32
from cutlass.cute.runtime import make_ptr
from cutlass import utils as cutlass_utils
from cutlass.experimental.task_scheduling.memory import SmemAllocation

from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    batch_decode_with_paged_kv_cache,
)
from flashinfer.attention.prims_ts.decode import (
    _DECODE_MAX_KV_LEN,
    _DECODE_MAX_KV_TILE_SIZE,
    _DecodeRuntime,
    _make_decode_workspace_layout,
    _planned_kv_domain_has_unpaired_tail,
    _validate_decode_query_head_extent,
    _validate_decode_output_aliasing,
    _validate_decode_policy_kv_tile_size,
    _validate_max_kv_len,
)
from flashinfer.attention.prims_ts._tensor_aliasing import (
    _validate_tensor_does_not_overlap_inputs,
)
from flashinfer.attention.prims_ts.split_kv_mode_policy import select_split_kv_modes
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
    FmhaDecodeConfig,
    make_decode_config,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
    _build_decode_gen_schedule,
    _decode_min_blocks_per_mp,
    build_decode_task_manager,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.helpers_kv_tile_idx import (
    _runtime_last_valid_page_idx,
    _runtime_total_kv_tiles,
)
from flashinfer.decode import (
    get_prims_ts_batch_decode_workspace_size,
    prims_ts_batch_decode_with_kv_cache,
)
from flashinfer.utils import is_sm100a_supported


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0)
    or not is_sm100a_supported(torch.device("cuda")),
    reason=(
        "PrimTS FMHA decode is signoff-qualified on SM100; "
        "SM103/B300 and GB300 qualification is pending"
    ),
)


_FP8 = torch.float8_e4m3fn
_FP8_PROBABILITY_SCALE = 448.0
_FP8_KV_TILE_SIZE = 128
_FP8_NUM_KV_INSTANCES = 2


def _single_cta_wave_capacity(device: torch.device | str = "cuda") -> int:
    """Return the runtime SM count used by direct-versus-persistent policy."""

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


@cute.kernel
def _runtime_int32_kv_ceil_div_kernel(
    values: cute.Pointer,
    output: cute.Pointer,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Evaluate the production runtime K-tile/page helpers on device."""

    index, _, _ = cute.arch.block_idx()
    seq_len_kv = Int32(values[index])
    output[index * Int32(2)] = _runtime_total_kv_tiles(
        cfg,
        seq_len_kv,
        Int32(1),
        Int32(0),
    )
    output[index * Int32(2) + Int32(1)] = _runtime_last_valid_page_idx(
        cfg,
        seq_len_kv,
    )


@cute.jit
def _launch_runtime_int32_kv_ceil_div(
    values: cute.Pointer,
    output: cute.Pointer,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    count: cutlass.Constexpr[int],
) -> None:
    _runtime_int32_kv_ceil_div_kernel(values, output, cfg).launch(
        grid=(count, 1, 1),
        block=(1, 1, 1),
    )


@dataclass(frozen=True)
class _DecodeCase:
    """One deterministic native-CSR problem used only by this test module."""

    q: torch.Tensor
    paged_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    reference_real: torch.Tensor
    output_dtype: torch.dtype
    mask_type: str
    bmm1_scale: float
    bmm2_scale: float
    q_scale: float
    k_scale: float
    v_scale: float
    o_scale: float
    window_left: int = -1


def _stored(real: torch.Tensor, dtype: torch.dtype, scale: float) -> torch.Tensor:
    return (real / scale).to(dtype) if dtype == _FP8 else real.to(dtype)


def _seq_lens_from_csr(
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    page_counts = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
    return (page_counts - 1) * page_size + paged_kv_last_page_len


def _visible_kv_bounds(
    *,
    kv_len: int,
    seq_len_q: int,
    query_idx: int,
    mask_type: str,
    window_left: int = -1,
) -> tuple[int, int]:
    """Return FlashInfer's bottom-right decode mask interval ``[begin, end)``."""

    if mask_type == "dense":
        end = kv_len
    elif mask_type == "causal":
        end = kv_len - seq_len_q + query_idx + 1
    else:
        raise ValueError("mask_type must be 'dense' or 'causal'")
    if end <= 0:
        raise ValueError("every causal KV sequence must be at least as long as Q")
    begin = 0 if window_left < 0 else max(0, end - window_left - 1)
    return begin, end


@torch.no_grad()
def _decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    *,
    sm_scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    mask_type: str,
    window_left: int = -1,
    qo_indptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Small independent FP32 oracle for HND paged GQA decode."""

    packed_q = qo_indptr is not None
    if packed_q:
        if q.ndim != 3:
            raise ValueError("packed Q must be [total_q, Hq, D]")
        batch_size = qo_indptr.numel() - 1
        num_qo_heads, head_dim = q.shape[-2:]
        q_tokens = q
        preserve_sq1_shape = False
    elif q.ndim == 3:
        q_tokens = q.unsqueeze(1)
        preserve_sq1_shape = True
    elif q.ndim == 4:
        q_tokens = q
        preserve_sq1_shape = False
    else:
        raise ValueError("Q must be [B, Hq, D] or [B, SQ, Hq, D]")
    if not packed_q:
        batch_size, _, num_qo_heads, head_dim = q_tokens.shape
    _, num_kv_heads, page_size, cache_head_dim = k_cache.shape
    if cache_head_dim != head_dim or num_qo_heads % num_kv_heads:
        raise ValueError("invalid GQA head geometry")

    group_size = num_qo_heads // num_kv_heads
    q_real = q_tokens.float() * q_scale
    k_real = k_cache.float() * k_scale
    v_real = v_cache.float() * v_scale
    output = torch.empty_like(q_real)
    for batch_idx in range(batch_size):
        page_begin = int(paged_kv_indptr[batch_idx].item())
        page_end = int(paged_kv_indptr[batch_idx + 1].item())
        page_ids = paged_kv_indices[page_begin:page_end].long()
        kv_len = (page_end - page_begin - 1) * page_size + int(
            paged_kv_last_page_len[batch_idx].item()
        )
        keys = (
            k_real[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        values = (
            v_real[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        if packed_q:
            q_begin = int(qo_indptr[batch_idx].item())
            q_end = int(qo_indptr[batch_idx + 1].item())
            request_queries = q_real[q_begin:q_end]
        else:
            request_queries = q_real[batch_idx]
            q_begin = 0
        seq_len_q = request_queries.shape[0]
        for query_idx in range(seq_len_q):
            begin, end = _visible_kv_bounds(
                kv_len=kv_len,
                seq_len_q=seq_len_q,
                query_idx=query_idx,
                mask_type=mask_type,
                window_left=window_left,
            )
            logits = torch.einsum(
                "hd,thd->ht",
                request_queries[query_idx],
                keys[begin:end],
            )
            probabilities = torch.softmax(logits * sm_scale, dim=-1)
            result = torch.einsum("ht,thd->hd", probabilities, values[begin:end])
            if packed_q:
                output[q_begin + query_idx] = result
            else:
                output[batch_idx, query_idx] = result
    return output[:, 0] if preserve_sq1_shape else output


@torch.no_grad()
def _fp8_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float,
    output_dtype: torch.dtype,
    o_scale: float,
    mask_type: str,
    window_left: int = -1,
    qo_indptr: Optional[torch.Tensor] = None,
    splits_kv: int = 1,
    num_insts_kv: int = _FP8_NUM_KV_INSTANCES,
) -> torch.Tensor:
    """Model the kernel's P448 operand and split/instance stream merge."""

    if q.dtype != _FP8 or k_cache.dtype != _FP8 or v_cache.dtype != _FP8:
        raise ValueError("the FP8 reference requires E4M3 Q, K, and V")
    if output_dtype not in (torch.float16, _FP8):
        raise ValueError("FP8 FMHA decode supports float16 or E4M3 output")
    if splits_kv <= 0 or num_insts_kv <= 0:
        raise ValueError("splits_kv and num_insts_kv must be positive")

    packed_q = qo_indptr is not None
    if packed_q:
        if q.ndim != 3:
            raise ValueError("packed Q must be [total_q, Hq, D]")
        batch_size = qo_indptr.numel() - 1
        num_qo_heads, head_dim = q.shape[-2:]
        q_tokens = q
        preserve_sq1_shape = False
    elif q.ndim == 3:
        batch_size, num_qo_heads, head_dim = q.shape
        q_tokens = q.unsqueeze(1)
        preserve_sq1_shape = True
    elif q.ndim == 4:
        batch_size, _, num_qo_heads, head_dim = q.shape
        q_tokens = q
        preserve_sq1_shape = False
    else:
        raise ValueError("Q must be [B, Hq, D] or [B, SQ, Hq, D]")

    _, num_kv_heads, page_size, cache_head_dim = k_cache.shape
    if cache_head_dim != head_dim or num_qo_heads % num_kv_heads:
        raise ValueError("invalid GQA head geometry")
    group_size = num_qo_heads // num_kv_heads
    k_stored = k_cache.float()
    v_stored = v_cache.float()
    output_shape = (
        q.shape if packed_q else (batch_size, q_tokens.shape[1], num_qo_heads, head_dim)
    )
    output = torch.empty(output_shape, dtype=torch.float32, device=q.device)

    for batch_idx in range(batch_size):
        page_begin = int(paged_kv_indptr[batch_idx].item())
        page_end = int(paged_kv_indptr[batch_idx + 1].item())
        page_ids = paged_kv_indices[page_begin:page_end].long()
        kv_len = (page_end - page_begin - 1) * page_size + int(
            paged_kv_last_page_len[batch_idx].item()
        )
        keys = (
            k_stored[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        values = (
            v_stored[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        if packed_q:
            q_begin = int(qo_indptr[batch_idx].item())
            q_end = int(qo_indptr[batch_idx + 1].item())
            request_queries = q_tokens[q_begin:q_end].float()
        else:
            request_queries = q_tokens[batch_idx].float()
            q_begin = 0

        seq_len_q = request_queries.shape[0]
        for query_idx in range(seq_len_q):
            visible_begin, visible_end = _visible_kv_bounds(
                kv_len=kv_len,
                seq_len_q=seq_len_q,
                query_idx=query_idx,
                mask_type=mask_type,
                window_left=window_left,
            )
            visible_keys = keys[visible_begin:visible_end]
            visible_values = values[visible_begin:visible_end]
            scores = torch.einsum(
                "hd,thd->ht", request_queries[query_idx], visible_keys
            )
            num_tiles = (
                visible_keys.shape[0] + _FP8_KV_TILE_SIZE - 1
            ) // _FP8_KV_TILE_SIZE
            tiles_per_group = splits_kv * num_insts_kv
            groups_per_split = (num_tiles + tiles_per_group - 1) // tiles_per_group
            local_tiles = max(
                groups_per_split * num_insts_kv,
                num_insts_kv,
            )
            active_splits = (num_tiles + local_tiles - 1) // local_tiles
            stream_tiles = []
            for split_idx in range(active_splits):
                split_begin = split_idx * local_tiles
                split_end = min(split_begin + local_tiles, num_tiles)
                for instance_idx in range(num_insts_kv):
                    tile_indices = range(
                        split_begin + instance_idx,
                        split_end,
                        num_insts_kv,
                    )
                    if tile_indices.start < tile_indices.stop:
                        stream_tiles.append(tile_indices)

            stream_max = [
                torch.full(
                    (num_qo_heads,),
                    -torch.inf,
                    dtype=torch.float32,
                    device=q.device,
                )
                for _ in stream_tiles
            ]
            stream_sum = [
                torch.zeros(num_qo_heads, dtype=torch.float32, device=q.device)
                for _ in stream_tiles
            ]
            stream_acc = [
                torch.zeros(
                    (num_qo_heads, head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                for _ in stream_tiles
            ]
            stream_valid = [False] * len(stream_tiles)

            for stream_idx, tile_indices in enumerate(stream_tiles):
                for tile_idx in tile_indices:
                    tile_begin = tile_idx * _FP8_KV_TILE_SIZE
                    tile_end = min(
                        tile_begin + _FP8_KV_TILE_SIZE, visible_keys.shape[0]
                    )
                    tile_scores = scores[:, tile_begin:tile_end]
                    local_max = tile_scores.max(dim=-1).values
                    new_max = (
                        torch.maximum(stream_max[stream_idx], local_max)
                        if stream_valid[stream_idx]
                        else local_max
                    )
                    probabilities = (
                        torch.exp((tile_scores - new_max.unsqueeze(-1)) * bmm1_scale)
                        * _FP8_PROBABILITY_SCALE
                    )
                    quantized_probabilities = probabilities.to(_FP8).float()
                    local_sum = probabilities.sum(dim=-1)
                    tile_acc = torch.einsum(
                        "ht,thd->hd",
                        quantized_probabilities,
                        visible_values[tile_begin:tile_end],
                    )
                    if stream_valid[stream_idx]:
                        correction = torch.exp(
                            (stream_max[stream_idx] - new_max) * bmm1_scale
                        )
                        stream_sum[stream_idx] = (
                            stream_sum[stream_idx] * correction + local_sum
                        )
                        stream_acc[stream_idx] = (
                            stream_acc[stream_idx] * correction.unsqueeze(-1) + tile_acc
                        )
                    else:
                        stream_sum[stream_idx] = local_sum
                        stream_acc[stream_idx] = tile_acc
                        stream_valid[stream_idx] = True
                    stream_max[stream_idx] = new_max

            final_max = (
                torch.stack(
                    [
                        maximum
                        for maximum, valid in zip(stream_max, stream_valid, strict=True)
                        if valid
                    ]
                )
                .max(dim=0)
                .values
            )
            final_sum = torch.zeros_like(final_max)
            final_acc = torch.zeros_like(stream_acc[0])
            for maximum, denominator, accumulator, valid in zip(
                stream_max,
                stream_sum,
                stream_acc,
                stream_valid,
                strict=True,
            ):
                if valid:
                    correction = torch.exp((maximum - final_max) * bmm1_scale)
                    final_sum += denominator * correction
                    final_acc += accumulator * correction.unsqueeze(-1)
            output_real = (final_acc / final_sum.unsqueeze(-1) * bmm2_scale).to(
                output_dtype
            ).float() * o_scale
            if packed_q:
                output[q_begin + query_idx] = output_real
            else:
                output[batch_idx, query_idx] = output_real
    return output[:, 0] if preserve_sq1_shape else output


def _make_decode_case(
    *,
    kv_lens: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len_q: int,
    page_size: int,
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    cache_form: str,
    mask_type: str,
    window_left: int = -1,
    device: str | torch.device,
    seed: int,
) -> _DecodeCase:
    """Create random data with deterministic nonidentity physical page IDs."""

    if mask_type == "causal" and min(kv_lens) < seq_len_q:
        raise ValueError("causal KV sequences must be at least as long as Q")
    batch_size = len(kv_lens)
    pages_per_request = tuple(
        (length + page_size - 1) // page_size for length in kv_lens
    )
    num_referenced_pages = sum(pages_per_request)
    num_physical_pages = num_referenced_pages + 3
    generator = torch.Generator(device="cpu").manual_seed(seed)
    page_ids = torch.randperm(num_physical_pages, generator=generator)[
        :num_referenced_pages
    ]
    if torch.equal(page_ids, torch.arange(num_referenced_pages)):
        page_ids = torch.roll(page_ids, 1)
    indptr = [0]
    for page_count in pages_per_request:
        indptr.append(indptr[-1] + page_count)

    q_shape = (
        (batch_size, num_qo_heads, head_dim)
        if seq_len_q == 1
        else (batch_size, seq_len_q, num_qo_heads, head_dim)
    )
    q_real = 0.25 * torch.randn(q_shape, generator=generator)
    k_real = 0.25 * torch.randn(
        num_physical_pages, num_kv_heads, page_size, head_dim, generator=generator
    )
    v_real = 0.25 * torch.randn(
        num_physical_pages, num_kv_heads, page_size, head_dim, generator=generator
    )
    q_scale, k_scale, v_scale = (
        (0.5, 0.25, 0.75) if qkv_dtype == _FP8 else (1.0, 1.0, 1.0)
    )
    o_scale = 0.625 if output_dtype == _FP8 else 1.0
    q = _stored(q_real, qkv_dtype, q_scale).to(device)
    k = _stored(k_real, qkv_dtype, k_scale).to(device)
    v = _stored(v_real, qkv_dtype, v_scale).to(device)
    indptr_tensor = torch.tensor(indptr, dtype=torch.int32, device=device)
    indices_tensor = page_ids.to(dtype=torch.int32, device=device)
    last_page_lens = torch.tensor(
        [(length - 1) % page_size + 1 for length in kv_lens],
        dtype=torch.int32,
        device=device,
    )
    if cache_form == "combined":
        combined = torch.stack((k, v), dim=1)
        paged_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = combined
        k, v = combined[:, 0], combined[:, 1]
    elif cache_form == "tuple":
        paged_kv_cache = (k, v)
    else:
        raise ValueError("cache_form must be 'combined' or 'tuple'")

    sm_scale = 1.0 / math.sqrt(head_dim)
    bmm1_scale = sm_scale * q_scale * k_scale
    bmm2_scale = v_scale / o_scale
    if qkv_dtype == _FP8:
        reference = _fp8_decode_reference(
            q,
            k,
            v,
            indptr_tensor,
            indices_tensor,
            last_page_lens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            output_dtype=output_dtype,
            o_scale=o_scale,
            mask_type=mask_type,
            window_left=window_left,
        )
    else:
        reference = _decode_reference(
            q,
            k,
            v,
            indptr_tensor,
            indices_tensor,
            last_page_lens,
            sm_scale=sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            mask_type=mask_type,
            window_left=window_left,
        )
    return _DecodeCase(
        q=q,
        paged_kv_cache=paged_kv_cache,
        k_cache=k,
        v_cache=v,
        paged_kv_indptr=indptr_tensor,
        paged_kv_indices=indices_tensor,
        paged_kv_last_page_len=last_page_lens,
        reference_real=reference,
        output_dtype=output_dtype,
        mask_type=mask_type,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_scale=o_scale,
        window_left=window_left,
    )


def _ragged_lengths(
    batch_size: int, max_seq_len: int, page_size: int
) -> tuple[int, ...]:
    """Return deterministic positive lengths spanning half to full maximum K."""

    if batch_size <= 0 or max_seq_len <= 0 or page_size <= 0:
        raise ValueError("batch size, maximum length, and page size must be positive")
    if batch_size == 1:
        return (max_seq_len,)
    lower = max(page_size, max_seq_len // 2)
    span = max_seq_len - lower
    lengths = [max_seq_len]
    used = {max_seq_len}
    for batch_idx in range(1, batch_size):
        candidate = lower + ((batch_idx * 104729 + batch_size * 37) % span)
        candidate = min(candidate, max_seq_len - 1)
        if candidate % page_size == 0:
            candidate -= 1
        while candidate in used and candidate > lower:
            candidate -= 1
            if candidate % page_size == 0:
                candidate -= 1
        if candidate in used:
            raise AssertionError("ragged test geometry does not have enough lengths")
        used.add(candidate)
        lengths.append(candidate)
    return tuple(lengths)


def _case(
    batch_size: int,
    max_seq_len: int,
    num_qo_heads: int,
    head_dim: int,
    qkv_dtype: torch.dtype,
    seed: int,
    *,
    num_kv_heads: int = 1,
    seq_len_q: int = 1,
    page_size: int = 32,
    output_dtype: torch.dtype | None = None,
    cache_form: str = "combined",
    mask_type: str = "dense",
    window_left: int = -1,
) -> dict[str, object]:
    return {
        "kv_lens": _ragged_lengths(batch_size, max_seq_len, page_size),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "seq_len_q": seq_len_q,
        "page_size": page_size,
        "qkv_dtype": qkv_dtype,
        "output_dtype": qkv_dtype if output_dtype is None else output_dtype,
        "cache_form": cache_form,
        "mask_type": mask_type,
        "window_left": window_left,
        "seed": seed,
    }


_UNSET = object()


def _policy(
    mma_variant: str,
    tile_size_q: int,
    *,
    grouped: bool = True,
    tile_size_kv: int | object = _UNSET,
    splits: int | object = _UNSET,
    split: bool | object = _UNSET,
    cluster: bool | object = _UNSET,
    separate: bool | object = _UNSET,
    persistent: bool | object = _UNSET,
) -> dict[str, object]:
    """Build only the policy fields that a row intentionally contracts."""

    expected = {
        "mma_variant": mma_variant,
        "tile_size_q": tile_size_q,
        "groups_tokens_heads_q": grouped,
    }
    optional = {
        "tile_size_kv": tile_size_kv,
        "splits_kv": splits,
        "use_split_kv": split,
        "use_cluster_smem_reduction": cluster,
        "use_separate_reduction_kernel": separate,
        "use_persistent_scheduler": persistent,
    }
    expected.update(
        {key: value for key, value in optional.items() if value is not _UNSET}
    )
    return expected


def _param(
    case_kwargs: dict[str, object],
    expected_policy: dict[str, object],
    correction_pattern: str | None,
    *,
    exercise_all_paths: bool = False,
    id: str,
):
    return pytest.param(
        case_kwargs,
        expected_policy,
        correction_pattern,
        exercise_all_paths,
        id=id,
    )


_FMHA_CASES = (
    _param(
        _case(4, 4096, 8, 128, torch.bfloat16, 31001),
        _policy(
            "swaps_mma_ab",
            8,
            tile_size_kv=128,
            split=True,
            cluster=True,
            separate=False,
            persistent=False,
        ),
        "identity",
        exercise_all_paths=True,
        id="F1-bf16-q8-cluster-ragged",
    ),
    _param(
        _case(5, 4097, 64, 256, torch.float8_e4m3fn, 31002, num_kv_heads=4),
        _policy(
            "swaps_mma_ab",
            16,
            splits=6,
            cluster=True,
            separate=False,
            persistent=False,
        ),
        "mixed",
        id="F2-fp8-d256-q16-s6-cluster",
    ),
    _param(
        _case(
            38,
            2177,
            128,
            64,
            torch.bfloat16,
            31003,
            num_kv_heads=4,
            page_size=64,
            cache_form="tuple",
        ),
        _policy(
            "swaps_mma_ab",
            32,
            split=False,
            cluster=False,
            separate=False,
            persistent=True,
        ),
        None,
        id="F3-bf16-d64-q32-direct-page64",
    ),
    _param(
        _case(
            38,
            2048,
            32,
            128,
            torch.float16,
            31004,
            num_kv_heads=4,
            page_size=128,
        ),
        _policy(
            "swaps_mma_ab",
            8,
            tile_size_kv=128,
            split=False,
            cluster=False,
            separate=False,
            persistent=True,
        ),
        None,
        id="F4-fp16-q8-clc-page128",
    ),
    _param(
        _case(
            8,
            4097,
            24,
            128,
            torch.float8_e4m3fn,
            31005,
            seq_len_q=4,
            output_dtype=torch.float16,
            mask_type="causal",
        ),
        _policy(
            "swaps_mma_ab",
            32,
            splits=4,
            split=True,
            cluster=True,
            separate=False,
            persistent=False,
        ),
        "mixed",
        id="F5-fp8-fp16-sq4-swaps-q32-cluster",
    ),
    _param(
        _case(
            4,
            8192,
            64,
            64,
            torch.float8_e4m3fn,
            31006,
            num_kv_heads=4,
            seq_len_q=4,
            mask_type="causal",
        ),
        _policy(
            "keeps_mma_ab",
            64,
            splits=8,
            separate=False,
        ),
        "identity",
        id="F6-fp8-d64-sq4-q64-s8",
    ),
    _param(
        _case(
            4,
            8192,
            128,
            128,
            torch.float8_e4m3fn,
            31007,
            num_kv_heads=4,
            seq_len_q=4,
            cache_form="tuple",
            mask_type="causal",
        ),
        _policy(
            "keeps_mma_ab",
            128,
            splits=8,
            separate=True,
        ),
        "mixed",
        id="F7-fp8-sq4-q128-separate",
    ),
    _param(
        _case(
            2,
            2049,
            16,
            128,
            torch.bfloat16,
            31008,
            seq_len_q=4,
            page_size=16,
            mask_type="dense",
        ),
        _policy(
            "swaps_mma_ab",
            16,
            splits=5,
            cluster=True,
        ),
        "tail",
        id="F8-spec-dense-tail-visible",
    ),
    _param(
        _case(
            2,
            2049,
            16,
            128,
            torch.bfloat16,
            31008,
            seq_len_q=4,
            page_size=16,
            mask_type="causal",
        ),
        _policy(
            "swaps_mma_ab",
            16,
            splits=5,
            cluster=True,
        ),
        "tail",
        id="F9-spec-causal-tail-progressive",
    ),
    _param(
        _case(
            3,
            2051,
            32,
            256,
            torch.bfloat16,
            31009,
            num_kv_heads=4,
            page_size=16,
        ),
        _policy("swaps_mma_ab", 8),
        None,
        id="F10-bf16-d256-page16-staged",
    ),
    _param(
        _case(
            16,
            2051,
            96,
            128,
            torch.float8_e4m3fn,
            31010,
            num_kv_heads=8,
            seq_len_q=8,
            page_size=32,
            mask_type="causal",
        ),
        _policy("keeps_mma_ab", 128, splits=1, split=False),
        "tail",
        id="F11-fp8-sq8-q128-runtime-causal",
    ),
    _param(
        _case(
            1,
            2048,
            64,
            64,
            torch.bfloat16,
            31011,
            num_kv_heads=4,
        ),
        _policy(
            "swaps_mma_ab",
            8,
            grouped=False,
            splits=4,
            split=True,
            cluster=True,
            separate=False,
            persistent=False,
        ),
        None,
        exercise_all_paths=True,
        id="F12-bf16-q16-underfilled-head-bands",
    ),
)


def _plan_case(
    case,
    *,
    max_kv_len: int,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
):
    wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
    seq_len_q = 1 if case.q.ndim == 3 else int(case.q.shape[1])
    wrapper.plan(
        case.paged_kv_indptr,
        case.paged_kv_indices,
        case.paged_kv_last_page_len,
        case.q.shape[-2],
        case.k_cache.shape[1],
        case.q.shape[-1],
        case.k_cache.shape[2],
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        q_data_type=case.q.dtype,
        kv_data_type=case.k_cache.dtype,
        o_data_type=case.output_dtype,
        mask_type=case.mask_type,
        window_left=case.window_left,
        max_kv_len=max_kv_len,
    )
    return wrapper


def _assert_auto_policy(
    policy: dict[str, object],
    expected_b200: dict[str, object],
    *,
    device: torch.device,
) -> None:
    """Contract exact B200 coverage and portable Blackwell legality."""

    assert policy.get("source", "auto") == "auto"
    assert policy["mma_variant"] in ("swaps_mma_ab", "keeps_mma_ab")
    assert policy["tile_size_q"] in (8, 16, 32, 64, 128)
    assert policy["tile_size_kv"] in (128, 256)
    assert isinstance(policy["groups_tokens_heads_q"], bool)
    assert policy["query_layout"] == policy["output_layout"]
    splits_kv = int(policy["splits_kv"])
    assert 1 <= splits_kv <= int(policy["max_splits_kv"])
    use_split_kv = bool(policy["use_split_kv"])
    use_cluster = bool(policy["use_cluster_smem_reduction"])
    use_separate = bool(policy["use_separate_reduction_kernel"])
    assert not (use_cluster and use_separate)
    if not use_split_kv:
        assert splits_kv == 1
        assert not use_cluster
        assert not use_separate
    if use_cluster or use_separate:
        assert use_split_kv
    if "tile_size_kv" in expected_b200:
        assert policy["tile_size_kv"] == expected_b200["tile_size_kv"]

    if (
        torch.cuda.get_device_capability(device) == (10, 0)
        and policy["tile_size_kv"] == 128
        and _single_cta_wave_capacity(device) == 148
    ):
        for key, expected in expected_b200.items():
            assert policy[key] == expected, (key, policy, expected_b200)


def _run_case(wrapper, case, *, out=None):
    return wrapper.run(
        case.q,
        case.paged_kv_cache,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out=out,
    )


def _run_standalone(
    case,
    seq_lens,
    *,
    max_kv_len: int,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
):
    """Run the caller-workspace public entry point for wrapper parity."""

    seq_len_q = 1 if case.q.ndim == 3 else int(case.q.shape[1])
    # The standalone ABI owns persistent reduction counters in this buffer.
    if workspace_buffer is None:
        workspace_size = get_prims_ts_batch_decode_workspace_size(
            case.paged_kv_indptr.numel() - 1,
            case.q.shape[-2],
            case.k_cache.shape[1],
            case.q.shape[-1],
            case.k_cache.shape[2],
            max_kv_len,
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            q_dtype=case.q.dtype,
            kv_dtype=case.k_cache.dtype,
            out_dtype=case.output_dtype,
            mask_type=case.mask_type,
            window_left=case.window_left,
            kv_layout="HND",
            device=case.q.device,
        )
        workspace = torch.zeros(workspace_size, dtype=torch.int8, device=case.q.device)
    else:
        workspace = workspace_buffer
    output = torch.empty_like(case.q, dtype=case.output_dtype) if out is None else out
    result = prims_ts_batch_decode_with_kv_cache(
        case.q,
        case.paged_kv_cache,
        workspace,
        case.paged_kv_indptr,
        case.paged_kv_indices,
        seq_lens,
        max_kv_len,
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out=output,
        out_dtype=case.output_dtype,
        mask_type=case.mask_type,
        window_left=case.window_left,
        kv_layout="HND",
    )
    assert result is output
    return output


def _assert_case_correct(output, case):
    actual = output.float() * case.o_scale
    expected = case.reference_real
    assert output.shape == case.q.shape
    assert output.dtype == case.output_dtype
    assert torch.isfinite(actual).all()
    if case.output_dtype == _FP8:
        rtol, atol = 5e-2, 2e-3
    elif case.q.dtype == _FP8:
        rtol, atol = 1e-3, 2.5e-4
    else:
        rtol, atol = 1e-2, 1e-2
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _exercise_public_paths(
    wrapper,
    case,
    seq_lens,
    *,
    max_kv_len: int,
    exercise_all_paths: bool,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
):
    """Always check eager; reserve standalone/graph parity for anchor rows."""

    if case.q.dtype == _FP8:
        case = _with_reference(
            case,
            qo_indptr=qo_indptr,
            splits_kv=int(dict(wrapper._policy)["splits_kv"]),
        )
    eager = _run_case(wrapper, case)
    _assert_case_correct(eager, case)
    if not exercise_all_paths:
        return eager

    standalone = _run_standalone(
        case,
        seq_lens,
        max_kv_len=max_kv_len,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    _assert_case_correct(standalone, case)
    torch.testing.assert_close(standalone, eager, rtol=0, atol=0)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(wrapper, case, out=graph_out)
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(graph_out, case)
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)
    return eager


def _with_reference(
    case,
    *,
    q: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    splits_kv: int = 1,
):
    """Return a case whose oracle reflects its current mutable Q/K/V data."""

    query = case.q if q is None else q
    if query.dtype == _FP8:
        reference = _fp8_decode_reference(
            query,
            case.k_cache,
            case.v_cache,
            case.paged_kv_indptr,
            case.paged_kv_indices,
            case.paged_kv_last_page_len,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
            output_dtype=case.output_dtype,
            o_scale=case.o_scale,
            mask_type=case.mask_type,
            window_left=case.window_left,
            qo_indptr=qo_indptr,
            splits_kv=splits_kv,
        )
    else:
        reference = _decode_reference(
            query,
            case.k_cache,
            case.v_cache,
            case.paged_kv_indptr,
            case.paged_kv_indices,
            case.paged_kv_last_page_len,
            sm_scale=1.0 / math.sqrt(query.shape[-1]),
            q_scale=case.q_scale,
            k_scale=case.k_scale,
            v_scale=case.v_scale,
            mask_type=case.mask_type,
            window_left=case.window_left,
            qo_indptr=qo_indptr,
        )
    return replace(case, q=query, reference_real=reference)


def _pack_decode_case(
    case: _DecodeCase,
    q_lens: Sequence[int],
) -> tuple[_DecodeCase, torch.Tensor]:
    """Pack fixed-Q storage and return its cumulative runtime Q offsets."""

    if case.q.ndim != 4 or len(q_lens) != case.q.shape[0]:
        raise ValueError("packed-Q source must be [B, SQ, H, D] with B lengths")
    if min(q_lens) <= 0 or max(q_lens) > case.q.shape[1]:
        raise ValueError("packed Q lengths must be positive and within source SQ")
    offsets = [0]
    for q_len in q_lens:
        offsets.append(offsets[-1] + q_len)
    qo_indptr = torch.tensor(offsets, dtype=torch.int32, device=case.q.device)
    packed_q = torch.cat(
        [case.q[batch_idx, :q_len] for batch_idx, q_len in enumerate(q_lens)]
    ).contiguous()
    return _with_reference(case, q=packed_q, qo_indptr=qo_indptr), qo_indptr


def _exercise_auto_case(
    case: _DecodeCase,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    exercise_all_paths: bool = False,
) -> dict[str, object]:
    """Plan automatically and validate one eager public-interface launch."""

    page_size = int(case.k_cache.shape[2])
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        page_size,
    )
    max_kv_len = int(seq_lens.max().item())
    wrapper = _plan_case(
        case,
        max_kv_len=max_kv_len,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    policy = dict(wrapper._policy)
    _assert_auto_policy(policy, {}, device=case.q.device)
    _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=max_kv_len,
        exercise_all_paths=exercise_all_paths,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    return policy


def _exercise_explicit_kv256_case(
    monkeypatch: pytest.MonkeyPatch,
    case: _DecodeCase,
    *,
    exercise_all_paths: bool = False,
) -> dict[str, object]:
    """Run a public path with the qualified logical KV256 profile pinned."""

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    original_make_decode_config = fmha_decode_config.make_decode_config
    explicit_profile = {
        "use_keeps_mma_ab": True,
        "tile_size_q": 64,
        "tile_size_kv": 256,
        "groups_tokens_heads_q": True,
    }

    def _make_explicit_kv256_config(*args, **kwargs):
        source = kwargs.get("args")
        kwargs["args"] = (
            explicit_profile if source is None else (source, explicit_profile)
        )
        return original_make_decode_config(*args, **kwargs)

    monkeypatch.setattr(
        fmha_decode_config,
        "make_decode_config",
        _make_explicit_kv256_config,
    )
    decode_module._resolve_decode_launch_spec.cache_clear()
    decode_module._get_compiled_decode.cache_clear()
    try:
        return _exercise_auto_case(
            case,
            exercise_all_paths=exercise_all_paths,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()
        decode_module._get_compiled_decode.cache_clear()


@torch.no_grad()
def _apply_decode_correction_pattern(case, pattern: str):
    query = case.q.unsqueeze(1) if case.q.ndim == 3 else case.q
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
    magnitude = 8 if query.dtype == torch.float8_e4m3fn else 1
    query[..., 0] = (magnitude * signs).to(query.dtype).unsqueeze(0)

    page_size = case.k_cache.shape[2]
    for batch_idx in range(query.shape[0]):
        page_begin = int(case.paged_kv_indptr[batch_idx].item())
        page_end = int(case.paged_kv_indptr[batch_idx + 1].item())
        page_ids = case.paged_kv_indices[page_begin:page_end].to(torch.long)
        logical_tokens = torch.arange(page_ids.numel() * page_size, device=query.device)
        stored_k = (32 - logical_tokens // 128).clamp_min(0).to(case.k_cache.dtype)
        case.k_cache[page_ids, :, :, 0] = stored_k.view(-1, 1, page_size).expand(
            -1, case.k_cache.shape[1], -1
        )

    return _with_reference(case)


@torch.no_grad()
def _apply_speculative_tail_markers(case):
    """Make dense and bottom-right causal SQ>1 results observably different."""

    query = case.q if case.q.ndim == 4 else case.q.unsqueeze(1)
    if query.shape[1] <= 1:
        raise ValueError("tail-marker coverage requires SQ>1 input")
    query.zero_()
    query[..., 0] = 8 if query.dtype == _FP8 else 1
    case.k_cache.zero_()
    case.v_cache.zero_()
    page_size = int(case.k_cache.shape[2])
    for batch_idx in range(query.shape[0]):
        page_begin = int(case.paged_kv_indptr[batch_idx].item())
        page_end = int(case.paged_kv_indptr[batch_idx + 1].item())
        page_ids = case.paged_kv_indices[page_begin:page_end].long()
        kv_len = (page_end - page_begin - 1) * page_size + int(
            case.paged_kv_last_page_len[batch_idx].item()
        )
        for tail_idx in range(query.shape[1] - 1):
            logical_token = kv_len - query.shape[1] + 1 + tail_idx
            page_id = page_ids[logical_token // page_size]
            page_offset = logical_token % page_size
            case.k_cache[page_id, :, page_offset, 0] = 80
            value_marker = (
                tail_idx + 1 if case.v_cache.dtype == _FP8 else 8 * (tail_idx + 1)
            )
            case.v_cache[page_id, :, page_offset, 0] = value_marker
    return _with_reference(case)


def _decode_runtime_for_aliasing() -> _DecodeRuntime:
    """Build the smallest runtime object accepted by the alias validator."""

    return _DecodeRuntime(
        q=torch.empty(8),
        k_cache=torch.empty(8),
        v_cache=torch.empty(8),
        out=torch.empty(8),
        num_physical_pages=1,
        k_page_stride=8,
        v_page_stride=8,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


def test_attention_ts_decode_alias_guard_covers_every_live_allocation() -> None:
    """The output may not reuse storage that remains live during a launch."""

    for aliased_name in (
        "k_cache",
        "v_cache",
        "seq_lens",
        "qo_indptr",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "workspace_buffer",
    ):
        runtime = _decode_runtime_for_aliasing()
        metadata = {
            "seq_lens": torch.empty(8),
            "qo_indptr": torch.empty(8),
            "paged_kv_indptr": torch.empty(8),
            "paged_kv_indices": torch.empty(8),
            "paged_kv_last_page_len": torch.empty(8),
            "workspace_buffer": torch.empty(8),
        }
        if aliased_name in ("k_cache", "v_cache"):
            runtime = replace(runtime, **{aliased_name: runtime.out})
        else:
            metadata[aliased_name] = runtime.out

        with pytest.raises(
            ValueError,
            match=rf"out must not overlap {aliased_name} storage",
        ):
            _validate_decode_output_aliasing(runtime, **metadata)


def test_attention_ts_decode_public_query_geometry_guards() -> None:
    """Reject unsupported public Q/head geometry before device probing."""

    int32_max = 2**31 - 1
    _validate_decode_query_head_extent(
        batch_size=int32_max,
        num_qo_heads=1,
        max_seq_len_q=1,
    )
    with pytest.raises(
        NotImplementedError,
        match=r"batch_size \* max_seq_len_q \* num_qo_heads.*signed int32",
    ):
        _validate_decode_query_head_extent(
            batch_size=int32_max + 1,
            num_qo_heads=1,
            max_seq_len_q=1,
        )

    # The public sizing path must reject the unsafe semantic key before device
    # probing, policy resolution, or workspace arithmetic.
    with pytest.raises(
        NotImplementedError,
        match=r"batch_size \* max_seq_len_q \* num_qo_heads.*signed int32",
    ):
        get_prims_ts_batch_decode_workspace_size(
            batch_size=2,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            max_seq_len=128,
            seq_len_q=2**27,
            device="cuda:0",
        )

    with pytest.raises(ValueError, match=r"Hq/Hkv <= 32"):
        get_prims_ts_batch_decode_workspace_size(
            batch_size=1,
            num_qo_heads=33,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            max_seq_len=128,
            device="cuda:0",
        )

    from flashinfer.attention.prims_ts import decode as decode_module

    decode_module._resolve_decode_launch_spec.cache_clear()
    try:
        with pytest.raises(ValueError, match=r"Hq/Hkv <= 32"):
            decode_module._resolve_decode_launch_spec(
                0,
                1,
                33,
                1,
                128,
                32,
                128,
                1,
                "float16",
                "float16",
                "float16",
                "HND",
                "dense",
                False,
                -1,
            )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()


def test_attention_ts_decode_reserves_int32_kv_tile_padding() -> None:
    """The public K/V bound leaves room for the final exclusive tile endpoint."""

    safe_max = 2**31 - _DECODE_MAX_KV_TILE_SIZE
    assert safe_max == _DECODE_MAX_KV_LEN
    assert _validate_max_kv_len(safe_max, "max_seq_len") == safe_max
    assert safe_max + _DECODE_MAX_KV_TILE_SIZE - 1 == 2**31 - 1
    assert (
        safe_max + _DECODE_MAX_KV_TILE_SIZE - 1
    ) // _DECODE_MAX_KV_TILE_SIZE * _DECODE_MAX_KV_TILE_SIZE <= 2**31 - 1
    assert (
        safe_max + 1 + _DECODE_MAX_KV_TILE_SIZE - 1
    ) // _DECODE_MAX_KV_TILE_SIZE * _DECODE_MAX_KV_TILE_SIZE == 2**31

    with pytest.raises(
        NotImplementedError,
        match=rf"max_seq_len must be <= {safe_max}.*signed int32",
    ):
        _validate_max_kv_len(safe_max + 1, "max_seq_len")

    _validate_decode_policy_kv_tile_size(
        type("Config", (), {"tile_size_kv": _DECODE_MAX_KV_TILE_SIZE})()
    )
    unsupported_tile_size = _DECODE_MAX_KV_TILE_SIZE * 2
    with pytest.raises(
        RuntimeError,
        match=(
            rf"K/V tile no larger than {_DECODE_MAX_KV_TILE_SIZE}"
            rf".*got {unsupported_tile_size}"
        ),
    ):
        _validate_decode_policy_kv_tile_size(
            type("Config", (), {"tile_size_kv": unsupported_tile_size})()
        )


def test_attention_ts_decode_workspace_rejects_unsafe_int32_kv_bound() -> None:
    """Workspace policy lookup rejects the first unsafe bound before CUDA work."""

    with pytest.raises(
        NotImplementedError, match=r"padded FMHA decode K/V coordinates"
    ):
        get_prims_ts_batch_decode_workspace_size(
            batch_size=1,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            max_seq_len=_DECODE_MAX_KV_LEN + 1,
            device="cuda:0",
        )


def test_attention_ts_decode_workspace_layout_uses_explicit_reducer_mode() -> None:
    """FP8 partial-O storage follows the selected reducer, not shape inference."""

    scratch_shapes = (
        (2, 3, 4, 5, 128),
        (2, 3, 4, 5),
        (2, 3, 5),
    )
    separate = _make_decode_workspace_layout(
        scratch_shapes,
        torch.float8_e4m3fn,
        use_separate_reduction_kernel=True,
    )
    inline = _make_decode_workspace_layout(
        scratch_shapes,
        torch.float8_e4m3fn,
        use_separate_reduction_kernel=False,
    )

    assert separate.partial_o.dtype == torch.bfloat16
    assert inline.partial_o.dtype == torch.float16


@pytest.mark.parametrize(
    ("max_kv_len", "expected"),
    ((128, True), (256, False), (257, True), (384, True), (385, False)),
)
def test_attention_ts_decode_planned_kv_domain_detects_unpaired_tail(
    max_kv_len: int,
    expected: bool,
) -> None:
    """Classify paired K128 instructions from the immutable plan bound."""

    config = FmhaDecodeConfig(tile_size_kv=128, num_insts_kv=2)
    assert _planned_kv_domain_has_unpaired_tail(config, max_kv_len) is expected


@pytest.mark.parametrize(
    (
        "tile_size_q",
        "tile_size_kv",
        "use_keeps_mma_ab",
        "total_kv_tiles",
        "expected_budgets",
    ),
    (
        (64, 128, True, 32, (None, None, None)),
        (128, 128, False, 32, (None, None, None)),
        (128, 128, True, 1, (184, 88, 56)),
        (64, 256, True, 31, (None, None, None)),
        (64, 256, True, 32, (176, 104, 56)),
    ),
)
def test_attention_ts_decode_register_reallocation_follows_task_graph(
    tile_size_q: int,
    tile_size_kv: int,
    use_keeps_mma_ab: bool,
    total_kv_tiles: int,
    expected_budgets: tuple[int | None, int | None, int | None],
) -> None:
    """Register hand-off follows topology and amortized KV256 loop length."""

    config = FmhaDecodeConfig(
        tile_size_q=tile_size_q,
        tile_size_kv=tile_size_kv,
        use_keeps_mma_ab=use_keeps_mma_ab,
        total_kv_tiles=total_kv_tiles,
    )
    budgets = (
        config.softmax_task_num_registers,
        config.correction_task_num_registers,
        config.mma_load_task_num_registers,
    )
    assert budgets == expected_budgets
    assert config.uses_task_register_reallocation is (expected_budgets[0] is not None)
    if config.uses_task_register_reallocation:
        assert all(value is not None and value % 8 == 0 for value in budgets)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_launch_and_plan_reject_unsafe_int32_kv_bound() -> None:
    """Standalone launch and reusable planning share the padded-coordinate cap."""

    device = torch.device("cuda")
    page_size = 16
    assert (
        get_prims_ts_batch_decode_workspace_size(
            batch_size=1,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=64,
            page_size=page_size,
            max_seq_len=_DECODE_MAX_KV_LEN,
            device=device,
        )
        > 0
    )
    q = torch.empty((1, 8, 64), dtype=torch.float16, device=device)
    kv_cache = torch.empty((1, 2, 1, page_size, 64), dtype=torch.float16, device=device)
    paged_kv_indptr = torch.tensor((0, 1), dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor((0,), dtype=torch.int32, device=device)
    last_page_len = torch.tensor((page_size,), dtype=torch.int32, device=device)
    seq_lens = last_page_len.clone()
    unsafe_max = _DECODE_MAX_KV_LEN + 1

    with pytest.raises(
        NotImplementedError, match=r"padded FMHA decode K/V coordinates"
    ):
        prims_ts_batch_decode_with_kv_cache(
            q,
            kv_cache,
            torch.empty(1, dtype=torch.uint8, device=device),
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens,
            unsafe_max,
        )

    with pytest.raises(
        NotImplementedError, match=r"padded FMHA decode K/V coordinates"
    ):
        BatchDecodePagedTSWrapper().plan(
            paged_kv_indptr,
            paged_kv_indices,
            last_page_len,
            8,
            1,
            64,
            page_size,
            max_kv_len=unsafe_max,
        )


def test_attention_ts_workspace_alias_guard() -> None:
    """Caller-owned scratch must be disjoint from every live allocation."""

    storage = torch.empty(64, dtype=torch.uint8)
    workspace = storage[:32]
    overlapping_query = storage[16:48]
    with pytest.raises(
        ValueError,
        match="workspace_buffer must not overlap query storage",
    ):
        _validate_tensor_does_not_overlap_inputs(
            workspace,
            "workspace_buffer",
            ("query", overlapping_query),
        )


@pytest.mark.parametrize(
    ("tile_size_q", "head_dim", "split_kv"),
    ((8, 64, 2), (16, 128, 32), (128, 256, 128)),
)
def test_attention_ts_decode_reduction_mode_order_is_structural(
    tile_size_q: int,
    head_dim: int,
    split_kv: int,
) -> None:
    """FMHA reducer order is independent of measured shape crossovers."""

    assert select_split_kv_modes(
        family="fmha_decode",
        topology="1cta",
        tile_size_q=tile_size_q,
        head_dim=head_dim,
        head_dim_per_cta_v=None,
        split_kv=split_kv,
        available_modes=(
            "gmem_reduction",
            "gmem_reduction_with_separate_kernel",
            "cluster_smem_reduction",
        ),
    ) == (
        "cluster_smem_reduction",
        "gmem_reduction_with_separate_kernel",
        "gmem_reduction",
    )


_DECODE_PUBLIC_SURFACES = (
    BatchDecodePagedTSWrapper.__init__,
    BatchDecodePagedTSWrapper.plan,
    BatchDecodePagedTSWrapper.run,
    batch_decode_with_paged_kv_cache,
    get_prims_ts_batch_decode_workspace_size,
    prims_ts_batch_decode_with_kv_cache,
)
_FORBIDDEN_DECODE_TUNING_PARAMETERS = frozenset(
    {
        "args",
        "auto_tuner",
        "autotuner",
        "config",
        "enable_clc",
        "enable_pdl",
        "fixed_split_size",
        "groups_tokens_heads_q",
        "kv_stages",
        "mma_variant",
        "num_insts_kv",
        "num_stages",
        "num_warps",
        "o_stages",
        "profile",
        "q_stages",
        "reduction_mode",
        "schedule",
        "single_kv",
        "split_kv",
        "split_kv_mode",
        "splits_kv",
        "tile_size_kv",
        "tile_size_q",
        "use_cluster_smem_reduction",
        "use_keeps_mma_ab",
        "use_persistent_scheduler",
        "use_separate_reduction_kernel",
        "warp_specialization",
    }
)
_FORBIDDEN_TUNING_TOKEN_PREFIXES = frozenset(
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
_FORBIDDEN_TUNING_TOKEN_SEQUENCES = (
    ("groups", "tokens", "heads"),
    ("single", "kv"),
    ("tensor", "cores"),
)


def test_attention_ts_decode_public_surfaces_have_no_internal_tuning_knobs() -> None:
    """Keep launch-policy controls private without freezing whole signatures."""

    violations = []
    for surface in _DECODE_PUBLIC_SURFACES:
        parameters = inspect.signature(surface).parameters
        for parameter in parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                violations.append(f"{surface.__qualname__}.**{parameter.name}")
                continue
            tokens = tuple(parameter.name.split("_"))
            has_forbidden_token = any(
                token.startswith(prefix)
                for token in tokens
                for prefix in _FORBIDDEN_TUNING_TOKEN_PREFIXES
            )
            has_forbidden_sequence = any(
                tokens[index : index + len(sequence)] == sequence
                for sequence in _FORBIDDEN_TUNING_TOKEN_SEQUENCES
                for index in range(len(tokens) - len(sequence) + 1)
            )
            if (
                parameter.name in _FORBIDDEN_DECODE_TUNING_PARAMETERS
                or has_forbidden_token
                or has_forbidden_sequence
            ):
                violations.append(f"{surface.__qualname__}.{parameter.name}")

    assert violations == []


def test_attention_ts_decode_bound_wrapper_trace_uses_plan_state():
    """Trace packed-Q shape and planned output dtype from the live wrapper."""
    from flashinfer.fi_trace import fi_trace

    wrapper = BatchDecodePagedTSWrapper()
    q = torch.empty((5, 32, 128), dtype=_FP8)
    k_cache = torch.empty((8, 4, 32, 128), dtype=_FP8)
    v_cache = torch.empty_like(k_cache)
    kwargs = {"q": q, "paged_kv_cache": (k_cache, v_cache)}

    with pytest.raises(
        ValueError,
        match=r"requires the live wrapper's plan state.*flashinfer\.fi_trace",
    ):
        wrapper.run.fi_trace(**kwargs)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        fi_trace(wrapper.run, **kwargs)

    # A successful plan publishes these fields atomically. Set them directly
    # here so the trace dispatcher remains a CPU-only contract test.
    wrapper._planned = True
    wrapper._use_packed_q = True
    wrapper._output_dtype = torch.float16
    defn = fi_trace(wrapper.run, **kwargs)
    assert defn["name"].startswith("prims_ts_decode_wrapper_tuple_fp16_output_packed_q")
    assert defn["inputs"]["q"]["shape"] == [
        "total_q",
        "num_qo_heads",
        "head_dim",
    ]
    assert defn["outputs"]["output"] == {
        "shape": ["total_q", "num_qo_heads", "head_dim"],
        "dtype": "float16",
        "param": "out",
    }


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _decode_resources_by_name(resource_dependency_graph):
    resources_by_id = {}
    for resource, dependencies in resource_dependency_graph.items():
        resources_by_id[id(resource)] = resource
        for dependency in dependencies:
            resources_by_id[id(dependency)] = dependency
    return {resource.name: resource for resource in resources_by_id.values()}


def _make_contiguous_keeps_config(*, dtype, tile_size_q: int, headdim: int = 128):
    return make_decode_config(
        headdim=headdim,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": tile_size_q,
            "groups_tokens_heads_q": False,
        },
        seq_len_q=1,
        seq_len_kv=4096,
        batch_size=8,
        num_heads_q=4 * tile_size_q,
        num_heads_kv=4,
        qkv_dtype=dtype,
        o_dtype=Float16 if dtype == Float8E4M3FN else dtype,
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type="dense",
        auto_tuner=False,
    )


def _make_contiguous_kv256_config(
    *,
    dtype=BFloat16,
    persistent: bool | None = None,
    config_args: dict[str, object] | None = None,
    split_kv_mode: str = "disabled",
    splits_kv: int = 1,
):
    """Build the qualified Q64/KV256 profile for schedule-level tests."""

    args = {
        "use_keeps_mma_ab": True,
        "tile_size_q": 64,
        "tile_size_kv": 256,
        "groups_tokens_heads_q": True,
    }
    if persistent is not None:
        args["use_persistent_scheduler"] = persistent
    if config_args is not None:
        args.update(config_args)
    return make_decode_config(
        headdim=128,
        args=args,
        seq_len_q=64,
        seq_len_kv=4096,
        batch_size=1,
        num_heads_q=32,
        num_heads_kv=32,
        qkv_dtype=dtype,
        o_dtype=dtype,
        qkv_layout="contiguousKv",
        split_kv_mode=split_kv_mode,
        splits_kv=splits_kv,
        mask_type="dense",
        auto_tuner=False,
    )


def _make_paged_window_crossing_config(*, page_size: int):
    """Build two visible K tiles whose page IDs cross the 32-ID window."""

    pages_per_tile = 128 // page_size
    page_window_tiles = 32 // pages_per_tile
    return make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 64,
            "groups_tokens_heads_q": False,
        },
        seq_len_q=1,
        seq_len_kv=(page_window_tiles + 1) * 128,
        batch_size=1,
        num_heads_q=64,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=page_size,
        split_kv_mode="disabled",
        splits_kv=1,
        sliding_window_causal=True,
        attention_window_size=256,
        mask_type="causal",
        auto_tuner=False,
    )


def _build_decode_resources(cfg):
    cfg.total_kv_tiles = 32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (
            _tasks,
            resource_dependency_graph,
            _dma_consumer_release_labels,
            smem_allocator,
            tmem_allocator,
            _correction_resources,
        ) = _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=cfg.total_kv_tiles,
            # Resource construction stores but does not dereference this marker.
            tma_desc_q=object(),
        )
    return (
        _decode_resources_by_name(resource_dependency_graph),
        smem_allocator,
        tmem_allocator,
    )


def _assert_decode_smem_within_capacity(cfg, smem_allocator) -> None:
    unified_smem_bytes = (
        _align_up(smem_allocator.total_smem_bytes, 8)
        + smem_allocator.barrier_smem_bytes
    )
    launch_smem_bytes = _align_up(unified_smem_bytes, cfg.stensor_align)
    assert launch_smem_bytes <= cutlass_utils.get_smem_capacity_in_bytes("sm_100")


@pytest.mark.parametrize("page_size", (16, 32, 64, 128))
def test_attention_ts_decode_page_offsets_cross_window_schedule_is_safe(
    page_size: int,
) -> None:
    """Exhaustively check the schedule when page loading crosses ID 32."""

    cfg = _make_paged_window_crossing_config(page_size=page_size)
    pages_per_tile = cfg.tile_size_kv // page_size
    page_window_tiles = 32 // pages_per_tile
    seq_len_kv = (page_window_tiles + 1) * cfg.tile_size_kv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_decode_task_manager(
            cfg,
            seq_len_kv=seq_len_kv,
            batch_size=1,
            num_heads_kv=1,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )

    assert cfg.static_num_skipped_kv_tiles == page_window_tiles - 1
    assert cfg.total_kv_tiles == 2
    assert cfg.static_num_skipped_kv_tiles * pages_per_tile == 32 - pages_per_tile
    assert (cfg.static_num_skipped_kv_tiles + 1) * pages_per_tile == 32


@pytest.mark.parametrize("dtype", (BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("headdim", (64, 128))
def test_attention_ts_decode_q128_tmem_p_aliases_consumed_s_region(
    dtype,
    headdim: int,
) -> None:
    """Q128 P overlays only its own consumed S region within capacity."""

    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=128,
        headdim=headdim,
    )
    assert cfg.uses_two_inst_tmem_p
    resources, smem_allocator, tmem_allocator = _build_decode_resources(cfg)
    s0 = resources["tmemS0"]._alloc
    s1 = resources["tmemS1"]._alloc
    p0 = resources["smemP0"]._tmem_alloc
    p1 = resources["smemP1"]._tmem_alloc
    output = resources["tmemO"]._alloc

    for name in ("tmemSoftmaxLocal0", "tmemSoftmaxLocal1"):
        stats = resources[name]
        if cfg.keeps_stats_via_smem:
            assert stats._alloc is None
        else:
            assert stats._alloc is not None
    if cfg.keeps_stats_via_smem:
        assert p0.offset == s0.offset + cfg.tmem_stats_cols
        assert p1.offset == s1.offset + cfg.tmem_stats_cols
    else:
        stats0 = resources["tmemSoftmaxLocal0"]._alloc
        stats1 = resources["tmemSoftmaxLocal1"]._alloc
        assert p0.offset == s0.offset
        assert p1.offset == s1.offset
        assert stats0.offset == s1.offset + s1.num_columns
        assert stats1.offset == stats0.offset + stats0.num_columns
        assert output.offset == stats1.offset + stats1.num_columns

    expected_p_cols = cfg.tile_size_kv * cfg.q_dtype_bytes // 4
    assert p0.num_columns == p1.num_columns == expected_p_cols
    assert s0.offset <= p0.offset
    assert p0.offset + p0.num_columns <= s0.offset + s0.num_columns
    assert s1.offset <= p1.offset
    assert p1.offset + p1.num_columns <= s1.offset + s1.num_columns
    assert output.offset >= s1.offset + s1.num_columns
    assert resources["smemP0"]._alloc is None
    assert resources["smemP1"]._alloc is None
    assert {"smemK0", "smemK1", "smemV0", "smemV1"} <= resources.keys()
    assert tmem_allocator.total_tmem_columns == cfg.tmem_total_cols <= 512
    _assert_decode_smem_within_capacity(cfg, smem_allocator)


def test_attention_ts_decode_kv256_uses_fragment_ready_p_policy() -> None:
    """KV256 publishes four TMEM P fragments without a full/empty FIFO."""

    cfg = _make_contiguous_kv256_config()
    resources, smem_allocator, _tmem_allocator = _build_decode_resources(cfg)

    assert cfg.keeps_stats_via_smem
    assert resources["tmemSoftmaxLocal0"]._alloc is None
    assert resources["tmemSoftmaxLocal1"]._alloc is None
    assert resources["tmemO"]._alloc.offset == 0
    assert resources["smemP0"]._tmem_alloc.offset == resources["tmemS0"]._alloc.offset
    assert resources["smemP1"]._tmem_alloc.offset == resources["tmemS1"]._alloc.offset

    # One score generation advances both P-ready parity and the matching
    # two-stage O completion credit exactly once per instance. These structural
    # invariants keep the hand-managed fragment protocol aligned across
    # persistent work-tile boundaries.
    assert cfg.num_insts_kv == cfg.o_stages == 2
    assert resources["tmemS0"].pipeline_config.num_stages == 1
    assert resources["tmemS1"].pipeline_config.num_stages == 1
    assert resources["tmemO"].pipeline_config.num_stages == cfg.o_stages
    for name in ("smemP0", "smemP1"):
        p = resources[name]
        assert p.pipeline_config is None
        assert p.tmem_o_ref is resources["tmemO"]
        assert isinstance(p._fragment_ready_alloc, SmemAllocation)
        assert p._fragment_ready_alloc.size_bytes == (
            cfg.num_softmax_score_fragments * 8
        )
    _assert_decode_smem_within_capacity(cfg, smem_allocator)


def test_attention_ts_decode_kv256_static_skips_unmodeled_fragment_alias_check() -> (
    None
):
    """Keep structural validation enabled when TS cannot model P barriers."""

    cfg = _make_contiguous_kv256_config(persistent=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        task_manager = build_decode_task_manager(
            cfg,
            seq_len_kv=2 * cfg.tile_size_kv,
            batch_size=1,
            num_heads_kv=32,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )

    assert task_manager._exhaustive_deadlock_race_check is False


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type", "message"),
    (
        pytest.param(
            "mma_tile_n_bmm1",
            128,
            ValueError,
            "mma_tile_n_bmm1",
            id="mma-geometry",
        ),
        pytest.param(
            "kv_stages",
            4,
            ValueError,
            "KV256 shared-memory pipeline",
            id="pipeline-capacity",
        ),
        pytest.param(
            "load_warp_idx",
            14,
            ValueError,
            "load_warp_idx",
            id="static-load-role",
        ),
        pytest.param(
            "kv_stages",
            2.5,
            TypeError,
            "kv_stages must be a Python integer",
            id="fractional-kv-stages",
        ),
        pytest.param(
            "q_stages",
            True,
            TypeError,
            "q_stages must be a Python integer",
            id="boolean-q-stages",
        ),
        pytest.param(
            "load_warp_idx",
            13.0,
            TypeError,
            "load_warp_idx must be a Python integer",
            id="fractional-load-role",
        ),
    ),
)
def test_attention_ts_decode_kv256_rejects_incompatible_profile_overrides(
    field_name: str,
    invalid_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject unsupported KV256 geometry, capacity, types, and roles."""

    with pytest.raises(error_type, match=message):
        _make_contiguous_kv256_config(
            config_args={field_name: invalid_value},
        )


@pytest.mark.parametrize(
    ("persistent", "use_attention_sinks", "expected_error"),
    (
        pytest.param(False, False, None, id="static"),
        pytest.param(
            True,
            False,
            "persistent KV256 requires kv_stages=3",
            id="persistent-direct",
        ),
        pytest.param(True, True, None, id="persistent-sinks"),
    ),
)
def test_attention_ts_decode_kv256_explicit_pipeline_depth_contract(
    persistent: bool,
    use_attention_sinks: bool,
    expected_error: str | None,
) -> None:
    """Keep KV2 where work boundaries make its fixed exchange safe."""

    config_args = {
        "kv_stages": 2,
        "use_attention_sinks": use_attention_sinks,
    }
    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            _make_contiguous_kv256_config(
                persistent=persistent,
                config_args=config_args,
            )
        return

    cfg = _make_contiguous_kv256_config(
        persistent=persistent,
        config_args=config_args,
    )
    assert cfg.kv_stages == 2
    assert cfg.use_persistent_scheduler is persistent
    assert cfg.use_attention_sinks is use_attention_sinks
    assert not cfg.uses_rotating_kv256_exchange

    if not persistent:
        resources, smem_allocator, _tmem_allocator = _build_decode_resources(cfg)
        assert resources["smemKv"].pipeline_config.num_stages == 2
        _assert_decode_smem_within_capacity(cfg, smem_allocator)


def test_attention_ts_decode_kv256_rotates_compact_direct_exchange() -> None:
    """Direct persistent KV256 carries its drained-stage alias in one credit."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.tmem_corr import (
        TmemCorrResource,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_tasks import (
        SmemKvReuseCreditResource,
    )

    cfg = _make_contiguous_kv256_config(persistent=True)
    assert cfg.uses_rotating_kv256_exchange
    reuse_credit = SmemKvReuseCreditResource(
        cfg=cfg,
        pipeline_config=None,
        name="smem_kv_reuse_credit",
    )
    credit_alloc = reuse_credit.get_smem_requirements()[0]
    tmem_corr1 = TmemCorrResource(
        cfg=cfg,
        inst_id=1,
        pipeline_config=None,
        name="tmemCorr1",
    )
    exchange_alloc = tmem_corr1.get_smem_requirements()[-1]

    # Direct output exchanges 128 float4 stats plus 64 rows of 132 floats.
    # The live 35,840-byte view dynamically selects one stage inside a
    # full-ring allocation envelope that aliases, but does not enlarge, KV.
    assert tmem_corr1._kv_tile_256_exchange_entries() * 4 == 35_840
    assert exchange_alloc.size_bytes == 3 * 65_536
    assert credit_alloc.size_bytes == 4


def test_attention_ts_decode_kv256_reuse_credit_requires_three_stage_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reuse-credit cursor arithmetic is specialized to three stages."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_tasks

    cfg = _make_contiguous_kv256_config(persistent=True)
    assert cfg.uses_rotating_kv256_exchange
    monkeypatch.setattr(fmha_decode_tasks, "KV_TILE_256_SHARED_FIFO_STAGES", 4)

    with pytest.raises(AssertionError, match="exactly three shared FIFO stages"):
        fmha_decode_tasks.SmemKvReuseCreditResource(
            cfg=cfg,
            pipeline_config=None,
            name="smem_kv_reuse_credit",
        )


def test_attention_ts_decode_rejects_reuse_credit_before_schedule_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotating reuse credit is invalid without a persistent work queue."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_tasks

    def unexpected_schedule_capture(_fn):
        pytest.fail("invalid reuse-credit topology reached schedule capture")

    monkeypatch.setattr(fmha_decode_tasks, "schedule", unexpected_schedule_capture)

    with pytest.raises(ValueError, match="reuse credit requires a work queue"):
        fmha_decode_tasks.create_correction_task(
            object(),
            object(),
            object(),
            object(),
            object(),
            None,
            object(),
            object(),
            domain=0,
        )


@pytest.mark.parametrize(
    ("split_kv_mode", "uses_separate_reduction"),
    (
        pytest.param("gmem_reduction", False, id="fused"),
        pytest.param(
            "gmem_reduction_with_separate_kernel",
            True,
            id="separate",
        ),
    ),
)
def test_attention_ts_decode_kv256_split_uses_compact_exchange(
    split_kv_mode: str,
    uses_separate_reduction: bool,
) -> None:
    """Both KV256 split reducers exchange only 64 logical output rows."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.tmem_corr import (
        TmemCorrResource,
    )

    cfg = _make_contiguous_kv256_config(
        dtype=Float16,
        split_kv_mode=split_kv_mode,
        splits_kv=2,
    )
    assert cfg.use_split_kv
    assert cfg.splits_kv == 2
    assert cfg.use_separate_reduction_kernel is uses_separate_reduction

    tmem_corr1 = TmemCorrResource(
        cfg=cfg,
        inst_id=1,
        pipeline_config=None,
        name="tmemCorr1",
    )
    exchange_alloc = tmem_corr1.get_smem_requirements()[-1]

    assert tmem_corr1._kv_tile_256_exchange_entries() * 4 == 35_840
    assert exchange_alloc.size_bytes == 35_840


def test_attention_ts_decode_rotating_exchange_is_a_storage_agnostic_capability() -> (
    None
):
    """Select rotation by physical topology, not contiguous versus paged K/V."""

    contiguous = _make_contiguous_kv256_config(persistent=True)
    paged = make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 64,
            "tile_size_kv": 256,
            "groups_tokens_heads_q": True,
            "use_persistent_scheduler": True,
        },
        seq_len_q=64,
        seq_len_kv=4096,
        batch_size=1,
        num_heads_q=32,
        num_heads_kv=32,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=64,
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type="dense",
        auto_tuner=False,
    )

    assert contiguous.uses_rotating_kv256_exchange
    assert paged.uses_rotating_kv256_exchange
    assert not replace(
        contiguous, use_persistent_scheduler=False
    ).uses_rotating_kv256_exchange
    assert not replace(contiguous, use_split_kv=True).uses_rotating_kv256_exchange
    assert not replace(
        contiguous, use_attention_sinks=True
    ).uses_rotating_kv256_exchange


def test_attention_ts_decode_kv256_register_launch_bound_is_amortized() -> None:
    """Only long KV256 mainloops pay for dynamic register hand-off."""

    cfg = _make_contiguous_kv256_config()
    assert _decode_min_blocks_per_mp(cfg, 31 * cfg.tile_size_kv) == 0
    assert _decode_min_blocks_per_mp(cfg, 31 * cfg.tile_size_kv + 1) == 1


def test_attention_ts_decode_kv256_register_budget_matches_launch_bound() -> None:
    """KV256 enables its 176/104/56 hand-off only once it is amortized."""

    cfg = make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 64,
            "tile_size_kv": 256,
            "groups_tokens_heads_q": True,
        },
        seq_len_q=64,
        seq_len_kv=8192,
        batch_size=1,
        num_heads_q=32,
        num_heads_kv=32,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type="dense",
        auto_tuner=False,
    )

    short_cfg = replace(cfg, total_kv_tiles=31)
    long_cfg = replace(cfg, total_kv_tiles=32)
    assert (
        short_cfg.softmax_task_num_registers,
        short_cfg.correction_task_num_registers,
        short_cfg.mma_load_task_num_registers,
    ) == (None, None, None)
    assert (
        long_cfg.softmax_task_num_registers,
        long_cfg.correction_task_num_registers,
        long_cfg.mma_load_task_num_registers,
    ) == (176, 104, 56)
    assert cfg.tmem_s_cols == 128
    assert cfg.mma_tile_n_bmm1 == 256


@pytest.mark.parametrize("dtype", (BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("headdim", (64, 128))
def test_attention_ts_decode_q64_keeps_p_in_smem_within_capacity(
    dtype,
    headdim: int,
) -> None:
    """Q64 keeps P in SMEM so the next QK wave can reuse released S."""

    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=64,
        headdim=headdim,
    )
    assert not cfg.uses_tmem_p
    resources, smem_allocator, tmem_allocator = _build_decode_resources(cfg)
    s0 = resources["tmemS0"]._alloc
    s1 = resources["tmemS1"]._alloc
    output = resources["tmemO"]._alloc
    for name in ("tmemSoftmaxLocal0", "tmemSoftmaxLocal1"):
        stats = resources[name]
        if cfg.keeps_stats_via_smem:
            assert stats._alloc is None
        else:
            assert stats._alloc is not None
    if cfg.keeps_stats_via_smem:
        assert output.offset == 0
    else:
        stats0 = resources["tmemSoftmaxLocal0"]._alloc
        stats1 = resources["tmemSoftmaxLocal1"]._alloc
        assert stats0.offset == 0
        assert stats1.offset == stats0.offset + stats0.num_columns
        assert output.offset == stats1.offset + stats1.num_columns
    assert s0.offset == output.offset + output.num_columns
    assert s1.offset == s0.offset + s0.num_columns
    assert s1.offset + s1.num_columns == cfg.tmem_total_cols
    for name in ("smemP0", "smemP1"):
        p = resources[name]
        assert isinstance(p._alloc, SmemAllocation)
        assert p._alloc.size_bytes == cfg.smem_p_tile_bytes
        assert p._tmem_alloc is None
    assert "tmemStatsDone0" not in resources
    assert "tmemStatsDone1" not in resources
    assert tmem_allocator.total_tmem_columns == cfg.tmem_total_cols <= 512
    _assert_decode_smem_within_capacity(cfg, smem_allocator)


@pytest.mark.parametrize("dtype", (BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("tile_size_q", (64, 128))
def test_attention_ts_decode_d256_staged_tmem_p_has_overwrite_gate(
    dtype,
    tile_size_q: int,
) -> None:
    """D256 P remains inside S and retains its overwrite-credit gate."""

    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=tile_size_q,
        headdim=256,
    )
    assert cfg.uses_staged_one_inst_tmem_p
    resources, smem_allocator, tmem_allocator = _build_decode_resources(cfg)
    s = resources["tmemS0"]._alloc
    p = resources["smemP0"]._tmem_alloc

    assert cfg.keeps_stats_via_smem
    assert resources["tmemSoftmaxLocal0"]._alloc is None
    assert p.offset == s.offset + cfg.tmem_stats_cols

    assert "tmemStatsDone0" in resources
    assert "tmemStatsDone1" not in resources
    assert resources["smemP0"]._alloc is None
    assert s.offset <= p.offset
    assert p.offset + p.num_columns <= s.offset + s.num_columns
    assert cfg.tmem_total_cols <= tmem_allocator.total_tmem_columns <= 512
    _assert_decode_smem_within_capacity(cfg, smem_allocator)


@pytest.mark.parametrize(
    ("tile_size_q", "dtype", "headdim"),
    (
        (128, BFloat16, 128),
        (128, Float8E4M3FN, 128),
        (64, BFloat16, 64),
        (64, Float8E4M3FN, 128),
    ),
    ids=("q128-bf16", "q128-fp8", "q64-bf16-d64", "q64-fp8-d128"),
)
def test_attention_ts_decode_keeps_alias_schedule_is_race_free(
    tile_size_q: int,
    dtype,
    headdim: int,
) -> None:
    """Exhaust HEAD, LOOP, and odd TAIL reuse for representative profiles."""

    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=tile_size_q,
        headdim=headdim,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_decode_task_manager(
            cfg,
            seq_len_kv=3 * cfg.tile_size_kv,
            batch_size=1,
            num_heads_kv=4,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_runtime_kv_ceil_div_covers_int32_domain() -> None:
    """Keep runtime K-tile and page ceilings safe through signed Int32 max."""

    int32_max = 2**31 - 1
    for page_size in (16, 32, 64, 128):
        cfg = FmhaDecodeConfig(num_tokens_per_page=page_size)
        host_values = (
            0,
            1,
            page_size - 1,
            page_size,
            page_size + 1,
            int32_max - page_size + 1,
            int32_max - page_size + 2,
            int32_max,
        )
        values = torch.tensor(host_values, device="cuda", dtype=torch.int32)
        output = torch.empty(
            (2 * len(host_values),),
            device="cuda",
            dtype=torch.int32,
        )
        values_ptr = make_ptr(
            Int32,
            values.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        output_ptr = make_ptr(
            Int32,
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )

        _launch_runtime_int32_kv_ceil_div(
            values_ptr,
            output_ptr,
            cfg,
            len(host_values),
        )

        expected = torch.tensor(
            tuple(
                (
                    (seq_len_kv + cfg.tile_size_kv - 1) // cfg.tile_size_kv,
                    max((seq_len_kv + page_size - 1) // page_size - 1, 0),
                )
                for seq_len_kv in host_values
            ),
            dtype=torch.int32,
        )
        torch.testing.assert_close(
            output.cpu().view(-1, 2),
            expected,
            rtol=0,
            atol=0,
        )


def test_attention_ts_decode_run_requires_plan():
    wrapper = BatchDecodePagedTSWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        wrapper.run(None, None)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_public_interfaces_reject_output_alias():
    max_kv_len = 128
    case = _make_decode_case(
        kv_lens=(max_kv_len,),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=1,
        page_size=16,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=20260718,
    )
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    )
    wrapper = _plan_case(case, max_kv_len=max_kv_len)

    with pytest.raises(ValueError, match="out must not overlap query storage"):
        _run_case(wrapper, case, out=case.q)
    with pytest.raises(ValueError, match="out must not overlap query storage"):
        _run_standalone(
            case,
            seq_lens,
            max_kv_len=max_kv_len,
            out=case.q,
        )


@pytest.mark.parametrize(
    ("indptr", "indices_count", "last_page_lens", "message"),
    (
        ((1, 2, 3), 3, (1, 1), "must start at zero"),
        ((0, 1, 1), 1, (1, 1), "must be strictly increasing"),
        ((0, 2, 1), 1, (1, 1), "must be strictly increasing"),
        ((0, 1, 3), 2, (1, 1), "must equal paged_kv_indices.numel"),
        ((0, 1, 2), 2, (0, 1), r"must be in \[1, 32\]"),
        ((0, 1, 2), 2, (1, 33), r"must be in \[1, 32\]"),
    ),
    ids=(
        "indptr-start",
        "indptr-repeated",
        "indptr-decreasing",
        "indptr-terminal",
        "last-page-empty",
        "last-page-too-long",
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_plan_rejects_malformed_paged_metadata(
    indptr,
    indices_count,
    last_page_lens,
    message,
):
    """Reject malformed native CSR values before selecting or compiling a kernel."""

    device = torch.device("cuda")
    paged_kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(indices_count, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor(
        last_page_lens, dtype=torch.int32, device=device
    )
    with pytest.raises(ValueError, match=message):
        BatchDecodePagedTSWrapper().plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            8,
            1,
            128,
            32,
        )


@pytest.mark.parametrize("packed_q", (False, True), ids=("fixed", "packed"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_rejects_per_request_causal_q_longer_than_kv(
    packed_q: bool,
):
    """Reject a short KV row even when another row satisfies the global bound."""

    device = torch.device("cuda")
    page_size = 16
    paged_kv_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor([4, 16], dtype=torch.int32, device=device)
    qo_indptr = (
        torch.tensor([0, 5, 6], dtype=torch.int32, device=device) if packed_q else None
    )
    seq_len_q = 1 if packed_q else 8
    max_seq_len_q = 5 if packed_q else None
    q = torch.empty(
        (6, 8, 64) if packed_q else (2, 8, 8, 64),
        dtype=torch.float16,
        device=device,
    )
    kv_cache = torch.empty((2, 2, 1, page_size, 64), dtype=torch.float16, device=device)
    match = r"request 0 has Q=(5|8) and K/V=4"

    with pytest.raises(ValueError, match=match):
        BatchDecodePagedTSWrapper().plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            8,
            1,
            64,
            page_size,
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            mask_type="causal",
            max_kv_len=16,
        )
    with pytest.raises(ValueError, match=match):
        batch_decode_with_paged_kv_cache(
            q,
            kv_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            mask_type="causal",
        )


def test_attention_ts_decode_shared_arch_guard_rejects_unsupported_gpu(monkeypatch):
    """Both public decode workspace APIs enforce the shared architecture guard."""

    from contextlib import nullcontext

    from flashinfer.mla import get_prims_ts_batch_decode_mla_workspace_size

    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (9, 0)
    )
    workspace_queries = (
        lambda: get_prims_ts_batch_decode_workspace_size(
            batch_size=1,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            max_seq_len=128,
            device="cuda:0",
        ),
        lambda: get_prims_ts_batch_decode_mla_workspace_size(
            batch_size=1,
            num_heads=8,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            page_size=32,
            max_seq_len=128,
            device="cuda:0",
        ),
    )
    for query in workspace_queries:
        with pytest.raises(
            NotImplementedError,
            match=r"requires an SM100a/B200 or SM103a/B300 GPU.*\(9, 0\)",
        ):
            query()


def test_attention_ts_decode_auto_launch_persists_only_above_one_sm_wave(
    monkeypatch,
):
    """A partial second CTA wave is the general persistence boundary."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        _select_auto_launch_mode,
    )

    class _FourSmHardware:
        def get_device_multiprocessor_count(self) -> int:
            return 4

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _FourSmHardware)
    modes = tuple(
        _select_auto_launch_mode(
            batch_size=ctas,
            num_heads_kv=1,
            seq_len_kv=128,
            num_q_tiles=1,
            tile_size_kv=128,
        )
        for ctas in (3, 4, 5)
    )
    assert modes == ("static", "static", "persistent")


@pytest.mark.parametrize(
    ("tile_size_kv", "threshold_tiles"),
    (
        pytest.param(128, 16, id="kv128"),
        pytest.param(256, 8, id="kv256"),
    ),
)
def test_attention_ts_decode_auto_split_threshold_scales_with_kv_tile(
    monkeypatch,
    tile_size_kv: int,
    threshold_tiles: int,
) -> None:
    """Equivalent scheduled KV work reaches split at either tile width."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    class _FourSmHardware:
        def get_device_multiprocessor_count(self) -> int:
            return 4

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _FourSmHardware)

    modes = tuple(
        fmha_decode_config._select_auto_launch_mode(
            batch_size=1,
            num_heads_kv=1,
            seq_len_kv=seq_len_kv,
            tile_size_kv=tile_size_kv,
        )
        for seq_len_kv in (
            (threshold_tiles - 1) * tile_size_kv,
            (threshold_tiles - 1) * tile_size_kv + 1,
        )
    )
    assert modes == ("static", "gmem_reduction")


@pytest.mark.parametrize("seq_len_q", (3, 5, 17))
def test_attention_ts_decode_config_accepts_arbitrary_positive_q_length(
    seq_len_q: int,
):
    """The public fixed-Q geometry is not restricted to power-of-two lengths."""

    from cutlass import BFloat16
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        make_decode_config,
    )

    cfg = make_decode_config(
        headdim=128,
        seq_len_q=seq_len_q,
        seq_len_kv=257,
        batch_size=1,
        num_heads_q=8,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=32,
        mask_type="causal",
        auto_tuner=False,
    )
    assert cfg.max_seq_len_q == seq_len_q
    assert cfg.num_q_ctas == seq_len_q


@pytest.mark.parametrize(
    ("num_heads_q", "num_heads_kv"),
    (
        pytest.param(0, 1, id="zero-q-heads"),
        pytest.param(1, 0, id="zero-kv-heads"),
    ),
)
def test_attention_ts_decode_config_requires_positive_head_counts(
    num_heads_q: int,
    num_heads_kv: int,
) -> None:
    """Reject non-positive head counts before automatic profile selection."""

    with pytest.raises(ValueError, match="head counts must be positive"):
        make_decode_config(
            headdim=128,
            seq_len_q=1,
            seq_len_kv=2048,
            batch_size=1,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            qkv_dtype=BFloat16,
            o_dtype=BFloat16,
            qkv_layout="pagedKv",
            num_tokens_per_page=16,
            auto_tuner=True,
        )


def _make_auto_kv_tile_config(monkeypatch, **overrides):
    """Resolve one automatic KV-tile candidate without compiling a kernel."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: "static",
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda _cluster_size: 148,
    )
    kwargs = {
        "headdim": 128,
        "seq_len_q": 64,
        "seq_len_kv": 1024,
        "batch_size": 1,
        "num_heads_q": 32,
        "num_heads_kv": 32,
        "qkv_dtype": BFloat16,
        "o_dtype": BFloat16,
        "qkv_layout": "pagedKv",
        "num_tokens_per_page": 16,
        "mask_type": "dense",
        "auto_tuner": True,
    }
    return fmha_decode_config.make_decode_config(**{**kwargs, **overrides})


@pytest.mark.parametrize("dtype", (BFloat16, Float16))
def test_attention_ts_decode_auto_config_selects_kv256(monkeypatch, dtype):
    """A Q64 cost-model result promotes both qualified 16-bit dtypes."""

    dtype_args = {
        "seq_len_q": 64,
        "num_heads_q": 128,
        "num_heads_kv": 4,
        "qkv_dtype": dtype,
        "o_dtype": dtype,
        "num_tokens_per_page": 32,
        "mask_type": "causal",
    }
    cfg = _make_auto_kv_tile_config(monkeypatch, **dtype_args)
    explicit_reference = _make_auto_kv_tile_config(
        monkeypatch,
        **dtype_args,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 64,
            "tile_size_kv": 256,
            "groups_tokens_heads_q": True,
        },
    )

    assert cfg.use_keeps_mma_ab is True
    assert cfg.tile_size_q == 64
    assert cfg.tile_size_kv == 256
    assert cfg.groups_tokens_heads_q is True
    assert cfg.q_tokens_per_cta == 2
    register_cfg = replace(cfg, total_kv_tiles=32)
    assert register_cfg.softmax_task_num_registers == 176
    assert register_cfg.correction_task_num_registers == 104
    assert cfg == explicit_reference


@pytest.mark.parametrize(
    (
        "overrides",
        "expected_tile_size_q",
        "expected_tile_size_kv",
        "expected_keeps",
        "expected_splits",
        "expect_q_cost_model",
        "expected_events",
    ),
    (
        pytest.param(
            {
                "seq_len_q": 1,
                "num_heads_q": 32,
                "num_heads_kv": 4,
                "qkv_dtype": Float16,
                "o_dtype": Float16,
                "num_tokens_per_page": 32,
                "mask_type": "causal",
            },
            8,
            128,
            False,
            1,
            False,
            ("q", "kv", "launch"),
            id="sq1-bypasses-q-cost-and-keeps-kv128",
        ),
        pytest.param(
            {
                "seq_len_q": 64,
                "num_heads_q": 128,
                "num_heads_kv": 4,
                "num_tokens_per_page": 32,
                "mask_type": "causal",
            },
            64,
            256,
            True,
            1,
            True,
            ("q", "kv", "launch"),
            id="q-cost-selects-q64-before-kv256",
        ),
        pytest.param(
            {
                "seq_len_q": 8,
                "num_heads_q": 32,
                "num_heads_kv": 1,
                "args": {"use_variable_seqlens_q": True},
            },
            32,
            128,
            False,
            1,
            False,
            ("q", "kv", "launch"),
            id="variable-q",
        ),
        pytest.param(
            {
                "seq_len_q": 1,
                "num_heads_q": 64,
                "num_heads_kv": 4,
                "qkv_dtype": Float16,
                "o_dtype": Float16,
                "num_tokens_per_page": 32,
                "mask_type": "causal",
            },
            16,
            128,
            False,
            1,
            False,
            ("q", "kv", "launch"),
            id="sq1-ratio16-bypasses-q-cost-and-keeps-kv128",
        ),
        pytest.param(
            {
                "headdim": 256,
                "seq_len_q": 2,
                "seq_len_kv": 512,
                "num_heads_q": 16,
                "num_heads_kv": 1,
                "qkv_dtype": Float8E4M3FN,
                "o_dtype": Float16,
                "num_tokens_per_page": 32,
                "mask_type": "causal",
            },
            64,
            128,
            True,
            2,
            True,
            ("q", "kv"),
            id="d256-cost-uses-finalized-one-inst-profile",
        ),
        pytest.param(
            {
                "seq_len_q": 1,
                "num_heads_q": 32,
                "num_heads_kv": 4,
                "args": {
                    "use_keeps_mma_ab": True,
                    "tile_size_q": 64,
                    "tile_size_kv": 256,
                    "groups_tokens_heads_q": True,
                },
            },
            64,
            256,
            True,
            1,
            False,
            ("q", "kv", "launch"),
            id="explicit-kv256",
        ),
    ),
)
def test_attention_ts_decode_auto_kv_tile_policy(
    monkeypatch,
    overrides: dict[str, object],
    expected_tile_size_q: int,
    expected_tile_size_kv: int,
    expected_keeps: bool,
    expected_splits: int,
    expect_q_cost_model: bool,
    expected_events: tuple[str, ...],
) -> None:
    """Cover KV/Q composition at utilization and compatibility boundaries."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    events = []
    launch_kv_tiles = []
    launch_persistent_flags = []
    q_cost_kv_tiles = []
    grouped_q_selections = []
    original_q_selector = fmha_decode_config._apply_auto_grouped_q_mma_config
    original_kv_promoter = fmha_decode_config._try_apply_auto_kv256_profile
    original_launch_selector = fmha_decode_config._apply_auto_launch_mode
    original_make_q_recipe = fmha_decode_config.make_grouped_q_launch_candidate

    def _spy_q_selector(*args, **kwargs):
        events.append("q")
        selected = original_q_selector(*args, **kwargs)
        grouped_q_selections.append(selected)
        return selected

    def _spy_kv_promoter(*args, **kwargs):
        events.append("kv")
        return original_kv_promoter(*args, **kwargs)

    def _spy_make_q_recipe(*args, **kwargs):
        q_cost_kv_tiles.append(kwargs["tile_size_kv"])
        return original_make_q_recipe(*args, **kwargs)

    def _spy_launch_selector(*args, **kwargs):
        events.append("launch")
        launch_kv_tiles.append(args[0].tile_size_kv)
        launch_persistent_flags.append(args[0].use_persistent_scheduler)
        return original_launch_selector(*args, **kwargs)

    monkeypatch.setattr(
        fmha_decode_config,
        "_apply_auto_grouped_q_mma_config",
        _spy_q_selector,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "_try_apply_auto_kv256_profile",
        _spy_kv_promoter,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "make_grouped_q_launch_candidate",
        _spy_make_q_recipe,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "_apply_auto_launch_mode",
        _spy_launch_selector,
    )
    cfg = _make_auto_kv_tile_config(monkeypatch, **overrides)

    assert cfg.tile_size_q == expected_tile_size_q
    assert cfg.tile_size_kv == expected_tile_size_kv
    assert cfg.use_keeps_mma_ab is expected_keeps
    assert cfg.splits_kv == expected_splits
    assert len(grouped_q_selections) == 1
    assert (grouped_q_selections[0] is not None) is expect_q_cost_model
    assert bool(q_cost_kv_tiles) is expect_q_cost_model
    assert set(q_cost_kv_tiles) <= {128}
    assert tuple(events) == expected_events
    expected_launch_kv_tiles = (
        (expected_tile_size_kv,) if "launch" in expected_events else ()
    )
    assert tuple(launch_kv_tiles) == expected_launch_kv_tiles
    assert tuple(launch_persistent_flags) == (False,) * len(expected_launch_kv_tiles)


def test_attention_ts_decode_auto_config_selection_is_device_agnostic(monkeypatch):
    """Kernel config selection relies on the public wrapper's device guard."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _make_auto_kv_tile_config(
        monkeypatch,
        seq_len_q=64,
        num_heads_q=128,
        num_heads_kv=4,
        num_tokens_per_page=32,
        mask_type="causal",
    )

    assert cfg.tile_size_kv == 256


def test_attention_ts_decode_public_sq1_head_band_stays_kv128(monkeypatch) -> None:
    """Keep the legacy public Q8 override outside KV256 selection."""

    from contextlib import nullcontext

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: "gmem_reduction",
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "select_splits_kv",
        lambda **_kwargs: 4,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 148 // cluster_size,
    )
    make_config_calls = []
    kv_q_candidates = []
    original_make_decode_config = fmha_decode_config.make_decode_config
    original_kv_promoter = fmha_decode_config._try_apply_auto_kv256_profile

    def _record_make_decode_config(*args, **kwargs):
        cfg = original_make_decode_config(*args, **kwargs)
        make_config_calls.append(cfg)
        return cfg

    def _record_kv_promoter(*args, q_candidate, **kwargs):
        kv_q_candidates.append(q_candidate)
        return original_kv_promoter(*args, q_candidate=q_candidate, **kwargs)

    monkeypatch.setattr(
        fmha_decode_config,
        "make_decode_config",
        _record_make_decode_config,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "_try_apply_auto_kv256_profile",
        _record_kv_promoter,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    decode_module._resolve_decode_launch_spec.cache_clear()
    try:
        public_spec = decode_module._resolve_decode_launch_spec(
            0,
            1,
            16,
            1,
            128,
            32,
            2048,
            1,
            "float16",
            "float16",
            "float16",
            "HND",
            "causal",
            False,
            -1,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()

    assert len(make_config_calls) == 2
    grouped_cfg, head_band_cfg = make_config_calls
    assert grouped_cfg.tile_size_q == 16
    assert grouped_cfg.groups_tokens_heads_q is True
    assert grouped_cfg.tile_size_kv == 128
    assert head_band_cfg.tile_size_q == 8
    assert head_band_cfg.groups_tokens_heads_q is False
    assert head_band_cfg.tile_size_kv == 128
    assert kv_q_candidates == [None, None]
    assert public_spec.config.tile_size_q == 8
    assert public_spec.config.groups_tokens_heads_q is False
    assert public_spec.config.tile_size_kv == 128


def test_attention_ts_decode_public_head_band_does_not_reduce_kv_fanout(
    monkeypatch,
) -> None:
    """Keep grouped Q when extra head-band CTAs would reduce KV fanout."""

    from contextlib import nullcontext

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 148 // cluster_size,
    )

    class _B200Hardware:
        def get_device_multiprocessor_count(self) -> int:
            return 148

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _B200Hardware)
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    decode_module._resolve_decode_launch_spec.cache_clear()
    try:
        spec = decode_module._resolve_decode_launch_spec(
            0,
            3,
            32,
            1,
            128,
            32,
            8192,
            1,
            "float16",
            "float16",
            "float16",
            "HND",
            "causal",
            False,
            -1,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()

    cfg = spec.config

    assert cfg.tile_size_q == 32
    assert cfg.groups_tokens_heads_q is True
    assert cfg.tile_size_kv == 128
    assert cfg.use_split_kv is True
    assert cfg.splits_kv == 16
    assert cfg.max_splits_kv == 16
    assert cfg.use_cluster_smem_reduction is True
    assert cfg.use_separate_reduction_kernel is False
    assert cfg.use_persistent_scheduler is False


def test_attention_ts_decode_explicit_kv_does_not_change_sq1_q_policy(
    monkeypatch,
) -> None:
    """Keep explicit KV materialization independent of the SQ1 Q policy."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: "static",
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 148 // cluster_size,
    )
    common = dict(
        headdim=64,
        seq_len_q=1,
        seq_len_kv=1024,
        batch_size=1,
        num_heads_q=16,
        num_heads_kv=1,
        qkv_dtype=Float16,
        o_dtype=Float16,
        qkv_layout="pagedKv",
        num_tokens_per_page=16,
        mask_type="dense",
        auto_tuner=True,
    )
    implicit = fmha_decode_config.make_decode_config(**common)
    explicit = fmha_decode_config.make_decode_config(
        **common,
        args={"tile_size_kv": 128},
    )

    assert implicit.tile_size_q == explicit.tile_size_q == 16
    assert implicit.groups_tokens_heads_q is explicit.groups_tokens_heads_q is True
    assert implicit.tile_size_kv == explicit.tile_size_kv == 128


@pytest.mark.parametrize(
    "overrides",
    (
        pytest.param({"headdim": 64}, id="head-dim-64"),
        pytest.param({"args": {"tile_size_kv": 128}}, id="explicit-kv128"),
        pytest.param(
            {"qkv_dtype": Float16, "o_dtype": BFloat16},
            id="mixed-16-bit-io",
        ),
    ),
)
def test_attention_ts_decode_auto_config_falls_back_to_kv128(
    monkeypatch,
    overrides,
):
    """An unqualified or explicitly pinned request retains generic KV128."""

    qualified_auto_shape = {
        "seq_len_q": 64,
        "num_heads_q": 128,
        "num_heads_kv": 4,
        "qkv_dtype": Float16,
        "o_dtype": Float16,
        "num_tokens_per_page": 32,
        "mask_type": "causal",
    }
    cfg = _make_auto_kv_tile_config(
        monkeypatch,
        **{**qualified_auto_shape, **overrides},
    )

    assert cfg.tile_size_kv == 128


def test_attention_ts_decode_q_cost_uses_kv128_with_explicit_kv256(
    monkeypatch,
) -> None:
    """Keep explicit KV materialization downstream of the TileQ cost model."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    q_cost_kv_tiles = []
    selected_q_candidates = []
    original_make_q_recipe = fmha_decode_config.make_grouped_q_launch_candidate
    original_kv_promoter = fmha_decode_config._try_apply_auto_kv256_profile

    def _record_q_cost_kv_tile(*args, **kwargs):
        q_cost_kv_tiles.append(kwargs["tile_size_kv"])
        return original_make_q_recipe(*args, **kwargs)

    def _record_q_winner(*args, q_candidate, **kwargs):
        selected_q_candidates.append(q_candidate)
        return original_kv_promoter(*args, q_candidate=q_candidate, **kwargs)

    monkeypatch.setattr(
        fmha_decode_config,
        "make_grouped_q_launch_candidate",
        _record_q_cost_kv_tile,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "_try_apply_auto_kv256_profile",
        _record_q_winner,
    )

    with pytest.raises(ValueError, match="KV256"):
        _make_auto_kv_tile_config(
            monkeypatch,
            seq_len_q=8,
            seq_len_kv=4096,
            num_heads_q=32,
            num_heads_kv=4,
            qkv_dtype=Float16,
            o_dtype=Float16,
            num_tokens_per_page=32,
            mask_type="causal",
            args={"tile_size_kv": 256},
        )

    assert q_cost_kv_tiles
    assert set(q_cost_kv_tiles) == {128}
    assert [candidate.tile_size_q for candidate in selected_q_candidates] == [8]


@pytest.mark.parametrize(
    ("packed_q", "sliding_window"),
    (
        pytest.param(True, False, id="packed-q"),
        pytest.param(False, True, id="sliding-window"),
        pytest.param(True, True, id="packed-q-sliding-window"),
    ),
)
def test_attention_ts_decode_runtime_q_features_use_structural_persistence(
    monkeypatch,
    packed_q: bool,
    sliding_window: bool,
):
    """Packed-Q and sliding-window grids persist only above one CTA wave."""

    from cutlass import BFloat16
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    class _FourSmHardware:
        def get_device_multiprocessor_count(self) -> int:
            return 4

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _FourSmHardware)
    # Keep this launch-mode unit test on the generic KV128 profile. A selected
    # Q64 16-bit D128 auto candidate uses KV256 and changes the physical
    # Q CTAs used by the wave boundary below.
    config_args: dict[str, object] = {"tile_size_kv": 128}
    if packed_q:
        config_args["use_variable_seqlens_q"] = True
    common = dict(
        headdim=128,
        args=config_args,
        seq_len_q=3,
        seq_len_kv=257,
        num_heads_q=8,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=32,
        split_kv_mode="disabled",
        mask_type="causal",
        sliding_window_causal=sliding_window,
        attention_window_size=128 if sliding_window else 0,
        auto_tuner=True,
    )
    one_wave = fmha_decode_config.make_decode_config(batch_size=1, **common)
    multi_wave = fmha_decode_config.make_decode_config(batch_size=2, **common)

    assert one_wave.use_persistent_scheduler is False
    assert multi_wave.use_persistent_scheduler is True
    assert multi_wave.use_split_kv is False


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_persistent_d256_graph_reloads_page_ids():
    """Persistence must cover a partial second wave and reload page IDs."""

    num_kv_heads = 4
    service_capacity = _single_cta_wave_capacity()
    batch_size = service_capacity // num_kv_heads + 1
    logical_work = batch_size * num_kv_heads
    assert (batch_size - 1) * num_kv_heads <= service_capacity < logical_work

    case = _make_decode_case(
        # Use the smallest whole batch that occupies a partial second CTA wave
        # on the GPU running this test.
        kv_lens=(257,) * batch_size,
        num_qo_heads=32,
        num_kv_heads=num_kv_heads,
        head_dim=256,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=31100,
    )
    with pytest.raises(
        ValueError,
        match=r"planned KV metadata.*longer than max_kv_len \(256\)",
    ):
        _plan_case(case, max_kv_len=256)

    wrapper = _plan_case(case, max_kv_len=257)
    policy = dict(wrapper._policy)
    assert policy["use_persistent_scheduler"] is True
    assert policy["kv_lengths_mode"] == "dynamic"

    eager = _run_case(wrapper, case).clone()
    _assert_case_correct(eager, case)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(wrapper, case, out=graph_out)
    assert captured is graph_out

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)

    indices_ptr = case.paged_kv_indices.data_ptr()
    indices_shape = case.paged_kv_indices.shape
    indices_stride = case.paged_kv_indices.stride()
    original_page_ids = case.paged_kv_indices.clone()

    num_physical_pages = case.k_cache.shape[0]
    remapped_page_ids = (original_page_ids + 1) % num_physical_pages
    case.paged_kv_indices.copy_(remapped_page_ids)
    assert case.paged_kv_indices.data_ptr() == indices_ptr
    assert case.paged_kv_indices.shape == indices_shape
    assert case.paged_kv_indices.stride() == indices_stride
    assert not torch.equal(case.paged_kv_indices, original_page_ids)
    remapped_case = _with_reference(case)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_case_correct(graph_out, remapped_case)
    assert not torch.allclose(graph_out.float(), eager.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("kv_len", "expected_kv_lengths_mode"),
    ((256, "planned_uniform_max"), (257, "dynamic")),
)
def test_attention_ts_decode_static_fp8_d128_odd_kv_tail_is_finite(
    kv_len: int,
    expected_kv_lengths_mode: str,
):
    """Static grouped Q8 specializes paired tails and guards unpaired tails."""

    case = _make_decode_case(
        kv_lens=(kv_len,),
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=128,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=_FP8,
        output_dtype=_FP8,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=31103,
    )
    wrapper = _plan_case(case, max_kv_len=kv_len)
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == 8
    assert policy["groups_tokens_heads_q"] is True
    assert policy["use_persistent_scheduler"] is False
    assert policy["use_split_kv"] is False
    assert policy["kv_lengths_mode"] == expected_kv_lengths_mode

    first_output = None
    for _ in range(3):
        output = _run_case(wrapper, case).clone()
        _assert_case_correct(output, case)
        if first_output is None:
            first_output = output
        else:
            torch.testing.assert_close(output, first_output, rtol=0, atol=0)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_standalone_graph_reloads_all_live_metadata():
    """One replay reloads packed Q offsets, native CSR, K lengths, and page IDs."""

    max_seq_len_q = 8
    max_kv_len = 257
    case = _make_decode_case(
        kv_lens=(65, max_kv_len),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=128,
        seq_len_q=max_seq_len_q,
        page_size=32,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="causal",
        device="cuda",
        seed=31101,
    )
    case, qo_indptr = _pack_decode_case(case, (1, 7))
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    )

    eager = _run_standalone(
        case,
        seq_lens,
        max_kv_len=max_kv_len,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    _assert_case_correct(eager, case)

    workspace_size = get_prims_ts_batch_decode_workspace_size(
        2,
        case.q.shape[-2],
        case.k_cache.shape[1],
        case.q.shape[-1],
        case.k_cache.shape[2],
        max_kv_len,
        seq_len_q=1,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        q_dtype=case.q.dtype,
        kv_dtype=case.k_cache.dtype,
        out_dtype=case.output_dtype,
        mask_type=case.mask_type,
        kv_layout="HND",
        device=case.q.device,
    )
    workspace = torch.zeros(workspace_size, dtype=torch.int8, device=case.q.device)
    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_standalone(
            case,
            seq_lens,
            max_kv_len=max_kv_len,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            out=graph_out,
            workspace_buffer=workspace,
        )
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)

    qo_indptr.copy_(torch.tensor((0, 4, 8), dtype=torch.int32, device="cuda"))
    case.paged_kv_indptr.copy_(
        torch.tensor((0, 9, 12), dtype=torch.int32, device="cuda")
    )
    seq_lens.copy_(torch.tensor((257, 65), dtype=torch.int32, device="cuda"))
    original_page_ids = case.paged_kv_indices.clone()
    case.paged_kv_indices.copy_((original_page_ids + 1) % int(case.k_cache.shape[0]))
    assert not torch.equal(case.paged_kv_indices, original_page_ids)
    replay_case = _with_reference(case, qo_indptr=qo_indptr)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_case_correct(graph_out, replay_case)
    assert not torch.allclose(graph_out.float(), eager.float(), rtol=1e-3, atol=1e-3)


def test_attention_ts_speculative_mask_oracle_distinguishes_tail_visibility():
    common = dict(
        kv_lens=(65,),
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=4,
        page_size=16,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        device="cpu",
        seed=31099,
    )
    dense = _apply_speculative_tail_markers(
        _make_decode_case(mask_type="dense", **common)
    )
    causal = _apply_speculative_tail_markers(
        _make_decode_case(mask_type="causal", **common)
    )
    # MTP/Eagle bottom-right causal hides the last SQ-1 tokens from the first
    # row, then reveals one token per row. DFlash/DSpark dense sees all tails.
    assert float(dense.reference_real[0, 0, 0, 0]) > 4
    assert float(causal.reference_real[0, 0, 0, 0]) == 0
    torch.testing.assert_close(
        causal.reference_real[:, -1], dense.reference_real[:, -1]
    )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_packed_q_sliding_window_public_parity():
    """Use cumulative Q lengths as the sole runtime ragged-Q definition."""

    q_lens = (1, 3, 8)
    max_seq_len_q = max(q_lens)
    case = _make_decode_case(
        kv_lens=(4097, 3071, 2053),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=max_seq_len_q,
        page_size=64,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type="causal",
        window_left=127,
        device="cuda",
        seed=31100,
    )
    case, qo_indptr = _pack_decode_case(case, q_lens)
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    )
    wrapper = _plan_case(
        case,
        max_kv_len=int(seq_lens.max().item()),
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    policy = dict(wrapper._policy)
    assert policy["use_packed_q"] is True
    assert policy["max_seq_len_q"] == max_seq_len_q
    assert policy["window_left"] == 127

    eager = _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=int(seq_lens.max().item()),
        exercise_all_paths=True,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    one_shot = batch_decode_with_paged_kv_cache(
        case.q,
        case.paged_kv_cache,
        case.paged_kv_indptr,
        case.paged_kv_indices,
        case.paged_kv_last_page_len,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        mask_type=case.mask_type,
        window_left=case.window_left,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out_dtype=case.output_dtype,
    )
    _assert_case_correct(one_shot, case)
    torch.testing.assert_close(one_shot, eager, rtol=0, atol=0)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_packed_q_sliding_window_clc_persistent():
    """Run live packed offsets and sliding bounds through the CLC scheduler."""

    q_lens = tuple((3, 5, 7)[batch_idx % 3] for batch_idx in range(22))
    max_seq_len_q = max(q_lens)
    case = _make_decode_case(
        kv_lens=(257,) * len(q_lens),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=max_seq_len_q,
        page_size=32,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="causal",
        window_left=127,
        device="cuda",
        seed=31102,
    )
    case, qo_indptr = _pack_decode_case(case, q_lens)
    policy = _exercise_auto_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    assert policy["use_persistent_scheduler"] is True
    assert policy["use_split_kv"] is False


@pytest.mark.parametrize(
    (
        "qkv_dtype",
        "output_dtype",
        "head_dim",
        "seq_len_q",
        "work_tiles_per_batch",
    ),
    (
        pytest.param(_FP8, _FP8, 128, 1, 4, id="fp8-d128-fixed"),
        pytest.param(
            torch.bfloat16,
            torch.bfloat16,
            256,
            2,
            2,
            id="bf16-d256-sliding",
        ),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_clc_persistent_dtype_head_dim_product(
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    head_dim: int,
    seq_len_q: int,
    work_tiles_per_batch: int,
):
    """Cover fixed and sliding CLC across FP8/BF16 and D128/D256."""

    service_capacity = _single_cta_wave_capacity()
    batch_size = service_capacity // work_tiles_per_batch + 1
    logical_work = batch_size * work_tiles_per_batch
    assert (batch_size - 1) * work_tiles_per_batch <= service_capacity < logical_work
    num_kv_heads = 4 if seq_len_q == 1 else 1
    case = _make_decode_case(
        kv_lens=(257,) * batch_size,
        num_qo_heads=8 * num_kv_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=seq_len_q,
        page_size=32,
        qkv_dtype=qkv_dtype,
        output_dtype=output_dtype,
        cache_form="combined",
        mask_type="causal" if seq_len_q > 1 else "dense",
        window_left=127 if seq_len_q > 1 else -1,
        device="cuda",
        seed=31200 + head_dim,
    )
    policy = _exercise_auto_case(case)
    assert policy["use_persistent_scheduler"] is True
    assert policy["use_split_kv"] is False
    assert policy["kv_lengths_mode"] == "dynamic"


@pytest.mark.parametrize(
    (
        "qkv_dtype",
        "seq_len_q",
        "kv_len",
        "batch_work_tiles",
        "window_left",
    ),
    (
        pytest.param(_FP8, 1, 256, 4, 127, id="sq1-sliding-parity-inversion"),
        pytest.param(
            torch.bfloat16,
            145,
            512,
            40,
            -1,
            id="multi-q-causal-parity-inversion",
        ),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_persistent_effective_k_domain_parity(
    qkv_dtype: torch.dtype,
    seq_len_q: int,
    kv_len: int,
    batch_work_tiles: int,
    window_left: int,
):
    """Keep Q/window-dependent odd K tails on runtime-safe resources."""

    service_capacity = _single_cta_wave_capacity()
    batch_size = service_capacity // batch_work_tiles + 1
    case = _make_decode_case(
        kv_lens=(kv_len,) * batch_size,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=128,
        seq_len_q=seq_len_q,
        page_size=32,
        qkv_dtype=qkv_dtype,
        output_dtype=qkv_dtype,
        cache_form="combined",
        mask_type="causal",
        window_left=window_left,
        device="cuda",
        seed=31300 + seq_len_q,
    )
    policy = _exercise_auto_case(case, exercise_all_paths=True)
    assert policy["use_persistent_scheduler"] is True
    assert policy["use_split_kv"] is False
    assert policy["kv_lengths_mode"] == "dynamic"


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_static_sliding_domain_uses_runtime_lengths():
    """A direct sliding plan guards the effective domain after leading skips."""

    case = _make_decode_case(
        kv_lens=(256,),
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=128,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=_FP8,
        output_dtype=_FP8,
        cache_form="combined",
        mask_type="causal",
        window_left=127,
        device="cuda",
        seed=31301,
    )
    policy = _exercise_auto_case(case, exercise_all_paths=True)
    assert policy["use_persistent_scheduler"] is False
    assert policy["use_split_kv"] is False
    assert policy["kv_lengths_mode"] == "dynamic"


@pytest.mark.parametrize(
    (
        "case_kwargs",
        "expected_policy",
        "correction_pattern",
        "exercise_all_paths",
    ),
    _FMHA_CASES,
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_compact_variable_k_acceptance(
    case_kwargs,
    expected_policy,
    correction_pattern,
    exercise_all_paths,
):
    """Exercise pairwise dtype/shape/mask policies with runtime-valued K."""

    case = _make_decode_case(device="cuda", **case_kwargs)
    if correction_pattern == "tail":
        case = _apply_speculative_tail_markers(case)
    elif correction_pattern is not None:
        case = _apply_decode_correction_pattern(case, correction_pattern)

    page_size = int(case.k_cache.shape[2])
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        page_size,
    )
    assert seq_lens.tolist() == list(case_kwargs["kv_lens"])
    assert torch.unique(seq_lens).numel() == seq_lens.numel()
    assert int(seq_lens.max().item()) == max(case_kwargs["kv_lens"])
    assert bool((seq_lens[1:] % page_size != 0).all().item())

    wrapper = _plan_case(case, max_kv_len=max(case_kwargs["kv_lens"]))
    policy = dict(wrapper._policy)
    _assert_auto_policy(policy, expected_policy, device=case.q.device)
    expected_kv_lengths_mode = (
        "planned_uniform_max"
        if all(
            length == max(case_kwargs["kv_lens"]) for length in case_kwargs["kv_lens"]
        )
        else "dynamic"
    )
    assert policy["kv_lengths_mode"] == expected_kv_lengths_mode

    _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=max(case_kwargs["kv_lens"]),
        exercise_all_paths=exercise_all_paths,
    )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, torch.float16))
def test_attention_ts_decode_qfirst_auto_selects_q64_kv256(
    qkv_dtype: torch.dtype,
):
    """Exercise a cost-model Q64 result and derived KV256 public launch."""

    case = _make_decode_case(
        kv_lens=(1024,),
        num_qo_heads=128,
        num_kv_heads=4,
        head_dim=128,
        seq_len_q=64,
        page_size=32,
        qkv_dtype=qkv_dtype,
        output_dtype=qkv_dtype,
        cache_form="combined",
        mask_type="causal",
        device="cuda",
        seed=20260729,
    )

    policy = _exercise_auto_case(case)

    assert policy["tile_size_q"] == 64
    assert policy["tile_size_kv"] == 256
    assert policy["mma_variant"] == "keeps_mma_ab"
    assert policy["use_split_kv"] is False


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("qkv_dtype", "kv_len", "num_heads", "forced_mode", "expect_separate"),
    (
        pytest.param(
            torch.float16,
            8192,
            32,
            None,
            True,
            id="fp16-separate-multi-head",
        ),
        pytest.param(
            torch.float16,
            8192,
            16,
            "gmem_reduction",
            False,
            id="fp16-fused",
        ),
        pytest.param(
            torch.bfloat16,
            65536,
            1,
            None,
            True,
            id="bf16-separate-long-kv",
        ),
    ),
)
def test_attention_ts_decode_q64_kv256_split_launch(
    monkeypatch: pytest.MonkeyPatch,
    qkv_dtype: torch.dtype,
    kv_len: int,
    num_heads: int,
    forced_mode: str | None,
    expect_separate: bool,
) -> None:
    """Exercise KV256 split-KV across distinct reducers and fanout shapes."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    if forced_mode is not None:
        monkeypatch.setattr(
            fmha_decode_config,
            "select_split_kv_modes",
            lambda **_kwargs: (forced_mode,),
        )

    case = _make_decode_case(
        kv_lens=(kv_len,),
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=128,
        seq_len_q=64,
        page_size=128,
        qkv_dtype=qkv_dtype,
        output_dtype=qkv_dtype,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=20260805,
    )

    policy = _exercise_explicit_kv256_case(monkeypatch, case)

    assert policy["tile_size_kv"] == 256
    assert policy["use_split_kv"] is True
    assert policy["use_separate_reduction_kernel"] is expect_separate


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    (
        "seq_len_q",
        "kv_lens",
        "page_size",
        "cache_form",
        "mask_type",
        "window_left",
        "persistent",
    ),
    (
        pytest.param(
            64,
            (129,),
            16,
            "combined",
            "causal",
            -1,
            False,
            id="single-kv-tile-tail",
        ),
        pytest.param(
            64,
            (769,),
            128,
            "tuple",
            "causal",
            127,
            False,
            id="static-window-kv-tail",
        ),
        pytest.param(
            512,
            # Two, four, and six physical KV tiles produce runtime loop
            # domains 0/1/2, exercising every modulo-three cursor advance in
            # one persistent launch.
            (257, 769, 1281),
            64,
            "combined",
            "dense",
            -1,
            True,
            id="persistent-ragged-rotating-cursor",
        ),
    ),
)
def test_attention_ts_decode_q64_kv256_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    seq_len_q: int,
    kv_lens: tuple[int, ...],
    page_size: int,
    cache_form: str,
    mask_type: str,
    window_left: int,
    persistent: bool,
):
    """Cover KV256 tails under windowed static and ragged persistent work."""

    case = _make_decode_case(
        kv_lens=kv_lens,
        num_qo_heads=32,
        num_kv_heads=32,
        head_dim=128,
        seq_len_q=seq_len_q,
        page_size=page_size,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form=cache_form,
        mask_type=mask_type,
        window_left=window_left,
        device="cuda",
        seed=20260804 + max(kv_lens),
    )

    policy = _exercise_explicit_kv256_case(monkeypatch, case)

    assert policy["tile_size_q"] == 64
    assert policy["tile_size_kv"] == 256
    assert policy["window_left"] == window_left
    assert policy["use_persistent_scheduler"] is persistent
    if persistent:
        assert policy["use_split_kv"] is False


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("packed_query", (False, True), ids=("fixed", "packed"))
def test_attention_ts_decode_reuses_compiled_topology_across_batch_sizes(
    packed_query: bool,
):
    """One resolved paged-decode topology accepts different batch extents."""

    wrappers = []
    for batch_size in (2, 3):
        case = _make_decode_case(
            kv_lens=(2049,) * batch_size,
            num_qo_heads=16,
            num_kv_heads=2,
            head_dim=128,
            seq_len_q=2 if packed_query else 1,
            page_size=32,
            qkv_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cache_form="combined",
            mask_type="dense",
            device="cuda",
            seed=32900 + batch_size,
        )
        qo_indptr = None
        if packed_query:
            case, qo_indptr = _pack_decode_case(case, (1,) * batch_size)
        wrapper = _plan_case(
            case,
            max_kv_len=2049,
            qo_indptr=qo_indptr,
            max_seq_len_q=1 if packed_query else None,
        )
        output = _run_case(wrapper, case)
        _assert_case_correct(output, case)
        wrappers.append(wrapper)

    assert wrappers[0]._compiled_main is wrappers[1]._compiled_main
    assert wrappers[0]._compiled_reducer is wrappers[1]._compiled_reducer


@pytest.mark.parametrize("head_dim", (64, 128, 256), ids=lambda value: f"d{value}")
@pytest.mark.parametrize(
    "num_qo_heads_per_kv",
    (1, 8, 15, 16, 32),
    ids=lambda value: f"gqa{value}",
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_head_dim_gqa_product(
    head_dim: int,
    num_qo_heads_per_kv: int,
):
    """Cross head width with GQA ratio using automatic runtime-K planning."""

    num_kv_heads = 2
    max_kv_len = 2049
    page_size = 32
    kv_lens = _ragged_lengths(2, max_kv_len, page_size)
    case = _make_decode_case(
        kv_lens=kv_lens,
        num_qo_heads=num_kv_heads * num_qo_heads_per_kv,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=1,
        page_size=page_size,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=33000 + head_dim + num_qo_heads_per_kv,
    )
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        page_size,
    )
    assert seq_lens.tolist() == list(kv_lens)
    assert bool((seq_lens % page_size != 0).all().item())

    wrapper = _plan_case(case, max_kv_len=max_kv_len)
    policy = dict(wrapper._policy)
    _assert_auto_policy(policy, {}, device=case.q.device)
    if policy["groups_tokens_heads_q"]:
        assert int(policy["tile_size_q"]) >= num_qo_heads_per_kv
    else:
        assert int(policy["tile_size_q"]) == 8
        assert num_qo_heads_per_kv > int(policy["tile_size_q"])
    _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=max_kv_len,
        exercise_all_paths=False,
    )


@pytest.mark.parametrize(
    ("qkv_dtype", "output_dtype"),
    (
        pytest.param(torch.bfloat16, torch.bfloat16, id="bf16"),
        pytest.param(_FP8, _FP8, id="fp8"),
    ),
)
@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@pytest.mark.parametrize("seq_len_q", (2, 8), ids=lambda value: f"sq{value}")
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_fixed_q_dtype_mask_product(
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    mask_type: str,
    seq_len_q: int,
):
    """Cross the missing fixed speculative-Q bounds with dtype and mask."""

    case = _make_decode_case(
        device="cuda",
        **_case(
            2,
            2051,
            16,
            128,
            qkv_dtype,
            36000 + seq_len_q + (100 if qkv_dtype == _FP8 else 0),
            num_kv_heads=2,
            seq_len_q=seq_len_q,
            output_dtype=output_dtype,
            mask_type=mask_type,
        ),
    )
    case = _apply_speculative_tail_markers(case)
    _exercise_auto_case(case)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_non_power_of_two_q_above_16k_kv():
    """Exercise arbitrary fixed SQ and a KV bound beyond the former 16K cap."""

    case = _make_decode_case(
        device="cuda",
        **_case(
            1,
            32769,
            8,
            64,
            torch.bfloat16,
            36503,
            seq_len_q=3,
            mask_type="causal",
        ),
    )
    _exercise_auto_case(case)


@pytest.mark.parametrize(
    "q_lens",
    (
        pytest.param((1, 2, 1, 2), id="maxsq2"),
        pytest.param((1, 2, 4, 3), id="maxsq4"),
        pytest.param((1, 3, 8, 5), id="maxsq8"),
    ),
)
@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_packed_fp8_q_mask_product(
    q_lens: tuple[int, ...],
    mask_type: str,
):
    """Cross packed FP8 static Q bounds with dense and causal masks."""

    max_seq_len_q = max(q_lens)
    case = _make_decode_case(
        device="cuda",
        **_case(
            len(q_lens),
            2051,
            16,
            128,
            _FP8,
            37000 + max_seq_len_q,
            num_kv_heads=2,
            seq_len_q=max_seq_len_q,
            mask_type=mask_type,
        ),
    )
    case = _apply_speculative_tail_markers(case)
    case, qo_indptr = _pack_decode_case(case, q_lens)
    policy = _exercise_auto_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    assert policy["use_packed_q"] is True
    assert policy["max_seq_len_q"] == max_seq_len_q


@pytest.mark.parametrize(
    ("qkv_dtype", "output_dtype", "correction_pattern"),
    (
        pytest.param(torch.float16, torch.float16, "identity", id="fp16"),
        pytest.param(_FP8, torch.float16, None, id="fp8-fp16"),
    ),
)
@pytest.mark.parametrize("head_dim", (64, 256), ids=lambda value: f"d{value}")
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_missing_dtype_head_dim_product(
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    correction_pattern: str | None,
    head_dim: int,
):
    """Fill unsupported-by-existing-tests dtype and head-width cells."""

    case = _make_decode_case(
        device="cuda",
        **_case(
            2,
            2049,
            16,
            head_dim,
            qkv_dtype,
            38000 + head_dim + (1 if qkv_dtype == _FP8 else 0),
            num_kv_heads=2,
            output_dtype=output_dtype,
        ),
    )
    if correction_pattern is not None:
        case = _apply_decode_correction_pattern(case, correction_pattern)
    _exercise_auto_case(case)


@pytest.mark.parametrize("page_size", (16, 32, 64, 128))
@pytest.mark.parametrize("cache_form", ("combined", "tuple"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_page_cache_sliding_product(
    page_size: int,
    cache_form: str,
):
    """Cross native page/cache forms and one padded outer page stride."""

    case = _make_decode_case(
        device="cuda",
        **_case(
            2,
            2051,
            8,
            64,
            torch.bfloat16,
            39000 + page_size + (1 if cache_form == "tuple" else 0),
            seq_len_q=2,
            page_size=page_size,
            cache_form=cache_form,
            mask_type="causal",
            window_left=127,
        ),
    )
    if page_size == 32 and cache_form == "tuple":
        compact_page_elements = case.k_cache[0].numel()
        padded_page_stride = compact_page_elements + 8

        def with_padded_outer_stride(cache: torch.Tensor) -> torch.Tensor:
            storage = cache.new_empty((cache.shape[0] * padded_page_stride,))
            padded = storage.as_strided(
                cache.shape,
                (padded_page_stride, *cache.stride()[1:]),
            )
            padded.copy_(cache)
            return padded

        padded_k = with_padded_outer_stride(case.k_cache)
        padded_v = with_padded_outer_stride(case.v_cache)
        assert padded_k.stride(0) > compact_page_elements
        assert padded_k.stride(0) * padded_k.element_size() % 16 == 0
        case = replace(
            case,
            paged_kv_cache=(padded_k, padded_v),
            k_cache=padded_k,
            v_cache=padded_v,
        )
        case = _with_reference(case)

    policy = _exercise_auto_case(case)
    assert policy["window_left"] == 127
