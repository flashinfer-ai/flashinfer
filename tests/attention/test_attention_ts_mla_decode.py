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

from flashinfer.attention.prims_ts import (
    BatchMLADecodePagedTSWrapper,
    batch_decode_mla_with_paged_kv_cache,
)
from flashinfer.mla import (
    get_prims_ts_batch_decode_mla_workspace_size,
    prims_ts_batch_decode_with_kv_cache_mla,
)


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
    """Return stable, distinct, non-page-aligned runtime KV lengths."""

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
        used.add(candidate)
        lengths.append(candidate)
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
    if min(q_lens) <= 0 or max(q_lens) > case.query.shape[1]:
        raise ValueError("packed Q lengths must be positive and within source SQ")
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
        outputs.append(torch.stack(request_outputs))
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


def _policy(
    kernel: str,
    tile_size_q: int,
    split_kv: int,
    head_dim_per_cta_v: int,
    *,
    cga: bool,
    persistent: bool = False,
    clc: bool = False,
    schedulers: bool = True,
) -> dict[str, object]:
    expected = {
        "kernel": kernel,
        "tile_size_q": tile_size_q,
        "split_kv": split_kv,
        "head_dim_per_cta_v": head_dim_per_cta_v,
        "use_cga_reduction": cga,
    }
    if schedulers:
        expected.update(
            use_persistent_scheduler=persistent,
            use_clc_dynamic_persistent_scheduler=clc,
        )
    return expected


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
        _policy("throughput_latency_1cta", 8, 8, 128, cga=True),
        "identity",
        False,
        exercise_all_paths=True,
        id="M1-bf16-q8-v128-s8-cga",
    ),
    _param(
        _case(4, 16, 4097, torch.float8_e4m3fn, 32002),
        _policy("throughput_latency_1cta", 16, 9, 128, cga=False),
        "mixed",
        True,
        id="M2-fp8-q16-v128-s9-separate",
    ),
    _param(
        _case(128, 32, 2048, torch.bfloat16, 32003),
        _policy(
            "throughput_latency_1cta",
            32,
            1,
            512,
            cga=False,
            persistent=True,
            clc=True,
        ),
        None,
        False,
        id="M3-bf16-q32-v512-direct-clc",
    ),
    _param(
        _case(128, 64, 2048, torch.float8_e4m3fn, 32004),
        _policy("throughput_latency_1cta", 64, 1, 512, cga=False),
        "identity",
        False,
        id="M4-fp8-q64-keeps-direct-static",
    ),
    _param(
        _case(8, 128, 2048, torch.float8_e4m3fn, 32005),
        _policy("throughput_2cta", 128, 8, 256, cga=False),
        "mixed",
        False,
        id="M5-fp8-2cta-q128-v256-s8-separate",
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
        _policy("throughput_latency_1cta", 64, 17, 256, cga=False),
        "mixed",
        False,
        id="M6-fp8-sq4-grouped-q64-v256-s17",
    ),
    _param(
        _case(4, 16, 4097, torch.bfloat16, 32007, seq_len_q=8),
        _policy("throughput_2cta", 128, 17, 256, cga=False),
        "identity",
        False,
        id="M7-bf16-sq8-grouped-2cta-q128-v256-s17",
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
        _policy("throughput_latency_1cta", 64, 9, 128, cga=False, schedulers=False),
        "tail",
        False,
        id="M8-spec-dense-tail-visible",
    ),
    _param(
        _case(2, 16, 2049, torch.bfloat16, 32008, seq_len_q=4),
        _policy("throughput_latency_1cta", 64, 9, 128, cga=False, schedulers=False),
        "tail",
        False,
        id="M9-spec-causal-tail-progressive",
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
    output = torch.empty(
        output_shape,
        dtype=case.output_dtype,
        device=case.query.device,
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
    """Contract exact B200 coverage and portable Blackwell legality."""

    assert policy["source"] == "auto"
    assert policy["kernel"] in ("throughput_latency_1cta", "throughput_2cta")
    assert policy["tile_size_q"] in (8, 16, 32, 64, 128)
    assert policy["tile_size_kv"] == 128
    assert int(policy["num_insts_kv"]) in (1, 2)
    assert int(policy["split_kv"]) >= 1
    head_dim_per_cta_v = int(policy["head_dim_per_cta_v"])
    num_ctas_per_head_dim = int(policy["num_ctas_per_head_dim"])
    assert head_dim_per_cta_v in (128, 256, 512)
    assert num_ctas_per_head_dim in (1, 2, 4)
    assert head_dim_per_cta_v * num_ctas_per_head_dim == _LATENT_DIM
    use_cga = bool(policy["use_cga_reduction"])
    persistent = bool(policy["use_persistent_scheduler"])
    use_clc = bool(policy["use_clc_dynamic_persistent_scheduler"])
    if use_cga:
        assert int(policy["split_kv"]) > 1
        assert policy["kernel"] == "throughput_latency_1cta"
    if use_clc:
        assert persistent
    if policy["kernel"] == "throughput_2cta":
        assert policy["tile_size_q"] == 128
        assert policy["head_dim_per_cta_v"] == 256
        assert policy["num_ctas_per_head_dim"] == 2
        assert not use_cga

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

    page_size = case.kv_cache.shape[2]
    for batch_idx, seq_len in enumerate(case.seq_lens.tolist()):
        page_count = (int(seq_len) + page_size - 1) // page_size
        page_ids = case.block_tables[batch_idx, :page_count].to(torch.long)
        logical_tokens = torch.arange(page_count * page_size, device=query.device)
        stored_k = (32 - logical_tokens // 128).clamp_min(0).to(case.kv_cache.dtype)
        case.kv_cache[page_ids, 0, :, 0] = stored_k.view(page_count, page_size)
    return case


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


def test_attention_ts_mla_public_surface_is_semantic():
    surfaces = (
        BatchMLADecodePagedTSWrapper.plan,
        BatchMLADecodePagedTSWrapper.run,
        get_prims_ts_batch_decode_mla_workspace_size,
        prims_ts_batch_decode_with_kv_cache_mla,
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
        inspect.signature(get_prims_ts_batch_decode_mla_workspace_size)
        .parameters["max_seq_len_q"]
        .default
        is None
    )
    assert (
        inspect.signature(get_prims_ts_batch_decode_mla_workspace_size)
        .parameters["seq_len_q"]
        .default
        is None
    )
    for surface in (
        BatchMLADecodePagedTSWrapper.plan,
        prims_ts_batch_decode_with_kv_cache_mla,
    ):
        parameters = inspect.signature(surface).parameters
        assert parameters["qo_indptr"].default is None
        assert parameters["max_seq_len_q"].default is None


def test_attention_ts_mla_fp8_reference_uses_p448():
    case = _make_mla_case(
        batch_size=2,
        num_qo_heads=8,
        max_seq_len=128,
        qkv_dtype=torch.float8_e4m3fn,
        device="cpu",
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
    """Define variable runtime Q lengths exclusively with cumulative offsets."""

    q_lens = (1, 3, 8)
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
    """Cross SQ2 grouped heads with dtype and speculative mask semantics."""

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
