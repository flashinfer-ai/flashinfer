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

import pytest
import torch

from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    batch_decode_with_paged_kv_cache,
)
from flashinfer.decode import (
    get_prims_ts_batch_decode_workspace_size,
    prims_ts_batch_decode_with_kv_cache,
)


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS FMHA decode requires SM100 or SM103",
)


_FP8 = torch.float8_e4m3fn
_FP8_PROBABILITY_SCALE = 448.0
_FP8_KV_TILE_SIZE = 128
_FP8_NUM_KV_INSTANCES = 2


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
    splits: int | object = _UNSET,
    split: bool | object = _UNSET,
    cga: bool | object = _UNSET,
    separate: bool | object = _UNSET,
    persistent: bool | object = _UNSET,
) -> dict[str, object]:
    """Build only the policy fields that a row intentionally contracts."""

    expected = {
        "mma_variant": mma_variant,
        "tile_size_q": tile_size_q,
        "groups_tokens_heads_q": True,
    }
    optional = {
        "splits_kv": splits,
        "use_split_kv": split,
        "use_cga_smem_reduction": cga,
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
            split=True,
            cga=True,
            separate=False,
            persistent=False,
        ),
        "identity",
        exercise_all_paths=True,
        id="F1-bf16-q8-cga-ragged",
    ),
    _param(
        _case(5, 4097, 64, 256, torch.float8_e4m3fn, 31002, num_kv_heads=4),
        _policy(
            "swaps_mma_ab",
            16,
            splits=6,
            cga=True,
            separate=False,
            persistent=False,
        ),
        "mixed",
        id="F2-fp8-d256-q16-s6-cga",
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
            cga=False,
            separate=False,
            persistent=False,
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
            split=False,
            cga=False,
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
            cga=True,
            separate=False,
            persistent=False,
        ),
        "mixed",
        id="F5-fp8-fp16-sq4-swaps-q32-cga",
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
            cga=True,
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
            cga=True,
        ),
        "tail",
        id="F9-spec-causal-tail-progressive",
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
    assert policy["tile_size_kv"] == 128
    assert isinstance(policy["groups_tokens_heads_q"], bool)
    assert policy["query_layout"] == policy["output_layout"]
    splits_kv = int(policy["splits_kv"])
    assert 1 <= splits_kv <= int(policy["max_splits_kv"])
    use_split_kv = bool(policy["use_split_kv"])
    use_cga = bool(policy["use_cga_smem_reduction"])
    use_separate = bool(policy["use_separate_reduction_kernel"])
    assert not (use_cga and use_separate)
    if not use_split_kv:
        assert splits_kv == 1
        assert not use_cga
        assert not use_separate
    if use_cga or use_separate:
        assert use_split_kv

    if torch.cuda.get_device_capability(device) == (10, 0):
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
):
    """Run the caller-workspace public entry point for wrapper parity."""

    seq_len_q = 1 if case.q.ndim == 3 else int(case.q.shape[1])
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
    # The standalone ABI owns persistent reduction counters in this buffer.
    workspace = torch.zeros(workspace_size, dtype=torch.int8, device=case.q.device)
    output = torch.empty_like(case.q, dtype=case.output_dtype)
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
        exercise_all_paths=False,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    return policy


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


def test_attention_ts_public_surface_is_semantic():
    surfaces = (
        BatchDecodePagedTSWrapper.plan,
        BatchDecodePagedTSWrapper.run,
        batch_decode_with_paged_kv_cache,
        get_prims_ts_batch_decode_workspace_size,
        prims_ts_batch_decode_with_kv_cache,
    )
    forbidden = (
        "autotuner",
        "config",
        "persistent",
        "profile",
        "reduction",
        "schedule",
        "split",
        "stage",
        "tile",
        "warp",
    )
    violations = [
        parameter
        for surface in surfaces
        for parameter in inspect.signature(surface).parameters
        if any(part in parameter for part in forbidden)
    ]
    assert violations == []
    assert (
        inspect.signature(BatchDecodePagedTSWrapper.plan)
        .parameters["seq_len_q"]
        .default
        == 1
    )
    assert (
        inspect.signature(prims_ts_batch_decode_with_kv_cache)
        .parameters["seq_len_q"]
        .default
        == 1
    )
    for surface in (
        BatchDecodePagedTSWrapper.plan,
        batch_decode_with_paged_kv_cache,
        get_prims_ts_batch_decode_workspace_size,
        prims_ts_batch_decode_with_kv_cache,
    ):
        parameters = inspect.signature(surface).parameters
        assert parameters["qo_indptr"].default is None
        assert parameters["max_seq_len_q"].default is None
        assert parameters["window_left"].default == -1


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

    _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=max(case_kwargs["kv_lens"]),
        exercise_all_paths=exercise_all_paths,
    )


@pytest.mark.parametrize("head_dim", (64, 128, 256), ids=lambda value: f"d{value}")
@pytest.mark.parametrize(
    "num_qo_heads_per_kv",
    (1, 8, 16, 32),
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
    assert policy["groups_tokens_heads_q"] is True
    assert int(policy["tile_size_q"]) >= num_qo_heads_per_kv
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
    """Cross native page and cache forms under fixed-Q sliding attention."""

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
    policy = _exercise_auto_case(case)
    assert policy["window_left"] == 127
