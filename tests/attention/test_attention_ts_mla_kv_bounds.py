# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native Prims-TS MLA per-query local-KV prefix bounds and merge statistics."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="Prims-TS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper
import flashinfer.attention.prims_ts.mla_decode as mla_decode_module
from tests.attention.test_attention_ts_mla_decode import (
    _CUDA_GRAPH_ANCHOR_OWNERS,
    _FP8,
    _LATENT_DIM,
    _MLACase,
    _REQUIRES_PRIMTS_GPU,
    _ROPE_DIM,
    _fp8_request_reference,
    _gather_request_cache,
    _make_mla_case,
    _mla_tolerances,
    _pack_mla_case,
    _policy_dict,
)


def _make_bounded_case(
    *,
    num_heads: int = 96,
    seq_len_q: int = 6,
    kv_lengths: tuple[int, ...] = (0, 65, 257),
    qkv_dtype: torch.dtype = torch.bfloat16,
    mask_type: str = "causal",
    page_size: int = 64,
    device: str = "cuda",
) -> _MLACase:
    """Reuse shuffled pages while allowing empty rank-local KV sequences."""

    capacity = max(page_size, max(kv_lengths))
    case = _make_mla_case(
        batch_size=len(kv_lengths),
        num_qo_heads=num_heads,
        max_kv_len=capacity,
        seq_len_q=seq_len_q,
        kv_seq_lens=(capacity,) * len(kv_lengths),
        qkv_dtype=qkv_dtype,
        mask_type="dense",
        page_size=page_size,
        device=device,
        seed=62100 + num_heads + seq_len_q,
    )
    return replace(
        case,
        seq_lens_kv=torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        mask_type=mask_type,
    )


def _plan_bounds(
    case: _MLACase,
    *,
    max_q_len: int,
    packed_query: bool = False,
    use_kv_bounds: bool = True,
    store_softmax_stats: bool = True,
) -> BatchMLADecodePagedTSWrapper:
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        case.query.device,
        case.block_tables.shape[0],
        case.query.shape[-2],
        _LATENT_DIM,
        _ROPE_DIM,
        case.page_size,
        case.max_kv_len,
        max_q_len=max_q_len,
        packed_query=packed_query,
        q_dtype=case.query.dtype,
        kv_dtype=case.kv_cache.dtype,
        out_dtype=case.out_dtype,
        mask_type=case.mask_type,
        use_kv_bounds=use_kv_bounds,
        store_softmax_stats=store_softmax_stats,
    )
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.use_kv_bounds is use_kv_bounds
    assert callable(wrapper._plan_state.compiled)
    assert _policy_dict(wrapper)["kernel"] in (
        "throughput_latency_1cta",
        "throughput_2cta",
    )
    return wrapper


def _run_bounds(
    wrapper: BatchMLADecodePagedTSWrapper,
    case: _MLACase,
    bounds: torch.Tensor | None,
    *,
    qo_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    stats: torch.Tensor | None = None,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stats is None:
        stats = torch.full(
            (*case.query.shape[:-1], 2),
            float("nan"),
            dtype=torch.float32,
            device=case.query.device,
        )
    output = wrapper.run(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        qo_indptr=qo_indptr,
        kv_bounds=bounds,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
        out=out,
        softmax_stats=stats,
        validate=validate,
    )
    return output, stats


@torch.no_grad()
def _bounded_reference(
    case: _MLACase,
    bounds: torch.Tensor,
    *,
    qo_indptr: torch.Tensor | None = None,
    policy: dict[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent FP32 stats, optionally modeling the FP8 P schedule."""

    query = case.query.reshape(-1, case.query.shape[-2], case.query.shape[-1])
    output = torch.zeros(
        (*query.shape[:-1], _LATENT_DIM), dtype=torch.float32, device=query.device
    )
    stats = torch.zeros(
        (*query.shape[:-1], 2), dtype=torch.float32, device=query.device
    )
    stats[..., 0] = -torch.inf
    offsets = (
        qo_indptr.tolist()
        if qo_indptr is not None
        else list(range(0, query.shape[0] + 1, case.query.shape[1]))
    )
    bound_values = bounds.tolist()
    for batch_idx, (begin, end) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        cache = _gather_request_cache(case, batch_idx)
        for query_idx in range(begin, end):
            visible = int(bound_values[query_idx])
            if visible == 0:
                continue
            visible_cache = cache[:visible]
            logits = query[query_idx].float() @ visible_cache.T * case.softmax_scale
            maximum = logits.max(dim=-1).values
            weights = torch.exp(logits - maximum[:, None])
            denominator = weights.sum(dim=-1)
            if case.query.dtype == _FP8 and policy is not None:
                output[query_idx] = _fp8_request_reference(
                    query[query_idx].float(),
                    cache,
                    softmax_scale=case.softmax_scale,
                    output_scale=case.output_scale,
                    num_insts_kv=int(policy["num_insts_kv"]),
                    tile_size_kv=int(policy["tile_size_kv"]),
                    splits_kv=int(policy["split_kv"]),
                    visible_kv_len=visible,
                )
            else:
                output[query_idx] = (
                    weights
                    @ visible_cache[:, :_LATENT_DIM]
                    / denominator[:, None]
                    * case.output_scale
                )
            stats[query_idx, :, 0] = maximum
            stats[query_idx, :, 1] = denominator
    return (
        output.reshape(*case.query.shape[:-1], _LATENT_DIM),
        stats.reshape(*case.query.shape[:-1], 2),
    )


def _assert_output_reference(
    output: torch.Tensor,
    expected: torch.Tensor,
    fp32_expected: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> None:
    rtol, atol = _mla_tolerances(dtype)
    torch.testing.assert_close(output.float(), expected, rtol=rtol, atol=atol)
    if dtype == _FP8:
        expected_norm = torch.linalg.vector_norm(fp32_expected)
        error_norm = torch.linalg.vector_norm(output.float() - fp32_expected)
        # P448 rounding is checked elementwise above. Keep an independent
        # mathematical-attention quality gate, including exact all-zero output.
        assert float(error_norm) <= 0.05 * float(expected_norm)


def _merge_reference(
    output: torch.Tensor, stats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = stats[..., 0].max(dim=0).values
    mass = torch.exp(stats[..., 0] - maximum) * stats[..., 1]
    denominator = mass.sum(dim=0)
    merged = (output.float() * mass[..., None]).sum(dim=0) / denominator[..., None]
    return merged, torch.stack((maximum, denominator), dim=-1)


def _assert_bounded_result(
    case: _MLACase,
    bounds: torch.Tensor,
    output: torch.Tensor,
    stats: torch.Tensor,
    *,
    policy: dict[str, object],
    qo_indptr: torch.Tensor | None = None,
) -> None:
    expected_output, expected_stats = _bounded_reference(
        case, bounds, qo_indptr=qo_indptr, policy=policy
    )
    assert output.dtype == case.out_dtype
    assert stats.dtype == torch.float32
    assert torch.isfinite(output).all()
    fp32_output = (
        _bounded_reference(case, bounds, qo_indptr=qo_indptr)[0]
        if case.query.dtype == _FP8
        else expected_output
    )
    _assert_output_reference(
        output, expected_output, fp32_output, dtype=case.query.dtype
    )
    torch.testing.assert_close(stats, expected_stats, rtol=2e-3, atol=2e-4)
    empty = bounds == 0
    flat_output = output.reshape(-1, output.shape[-2], _LATENT_DIM)
    flat_stats = stats.reshape(-1, stats.shape[-2], 2)
    assert torch.count_nonzero(flat_output[empty]) == 0
    assert torch.isneginf(flat_stats[empty, :, 0]).all()
    assert torch.count_nonzero(flat_stats[empty, :, 1]) == 0


def _request_bounds(length: int, q_len: int) -> list[int]:
    if q_len == 0:
        return []
    if q_len == 1:
        return [min(length, 64)]
    candidates = [0, 1, 63, 64, 65, length, length, length]
    return sorted(min(length, value) for value in candidates[:q_len])


@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, _FP8), ids=("bf16", "fp8"))
@pytest.mark.parametrize("num_heads", (8, 64, 96), ids=("h8", "h64", "h96"))
@pytest.mark.parametrize("seq_len_q", (1, 6, 8), ids=("sq1", "sq6", "sq8"))
@pytest.mark.parametrize("packed_query", (False, True), ids=("fixed", "packed"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_native_parity(
    qkv_dtype: torch.dtype, num_heads: int, seq_len_q: int, packed_query: bool
) -> None:
    """Cover empty local KV, uneven packed requests, and partial query tiles."""

    kv_lengths = (0, 65, 257, 0)
    case = _make_bounded_case(
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        kv_lengths=kv_lengths,
        qkv_dtype=qkv_dtype,
    )
    q_lengths = (seq_len_q, 1, seq_len_q, 0) if packed_query else (seq_len_q,) * 4
    qo_indptr = None
    if packed_query:
        case, qo_indptr = _pack_mla_case(case, q_lengths)
    bounds = torch.tensor(
        [
            value
            for kv_len, q_len in zip(kv_lengths, q_lengths, strict=True)
            for value in _request_bounds(kv_len, q_len)
        ],
        dtype=torch.int32,
        device=case.query.device,
    )
    wrapper = _plan_bounds(case, max_q_len=seq_len_q, packed_query=packed_query)
    output, stats = _run_bounds(wrapper, case, bounds, qo_indptr=qo_indptr)
    _assert_bounded_result(
        case, bounds, output, stats, qo_indptr=qo_indptr, policy=_policy_dict(wrapper)
    )


@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, _FP8), ids=("bf16", "fp8"))
@pytest.mark.parametrize("num_heads", (8, 96), ids=("h8", "h96"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_split_reduction(
    qkv_dtype: torch.dtype, num_heads: int
) -> None:
    """Split reducers must preserve neutral rows and short partial prefixes."""

    case = _make_bounded_case(
        num_heads=num_heads,
        seq_len_q=1,
        kv_lengths=(32769, 32769, 32769, 32769),
        qkv_dtype=qkv_dtype,
    )
    bounds = torch.tensor([0, 1, 129, 32769], dtype=torch.int32, device="cuda")
    wrapper = _plan_bounds(case, max_q_len=1)
    assert int(_policy_dict(wrapper)["split_kv"]) > 1
    output, stats = _run_bounds(wrapper, case, bounds)
    _assert_bounded_result(case, bounds, output, stats, policy=_policy_dict(wrapper))


@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, _FP8), ids=("bf16", "fp8"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_helix_page_boundary(qkv_dtype: torch.dtype) -> None:
    """Global positions 63,64,65 expose rank prefixes [64,64,64] and [0,1,2]."""

    seq_len_q = 6
    case = _make_bounded_case(
        seq_len_q=seq_len_q, kv_lengths=(64, 5, 0), qkv_dtype=qkv_dtype
    )
    # Every CP rank sees the same global query rows but has independent local pages.
    case.query[1:].copy_(case.query[:1].expand(2, -1, -1, -1))
    bounds = torch.tensor(
        [64] * 6 + list(range(6)) + [0] * 6, dtype=torch.int32, device="cuda"
    )
    assert bounds.reshape(3, 6)[:2, :3].tolist() == [[64, 64, 64], [0, 1, 2]]
    wrapper = _plan_bounds(case, max_q_len=seq_len_q)
    output, stats = _run_bounds(wrapper, case, bounds)
    policy = _policy_dict(wrapper)
    _assert_bounded_result(case, bounds, output, stats, policy=policy)

    merged, merged_stats = _merge_reference(output, stats)
    local_expected, local_stats = _bounded_reference(case, bounds, policy=policy)
    quantized_merged, expected_merged_stats = _merge_reference(
        local_expected, local_stats
    )
    global_cache = torch.cat([_gather_request_cache(case, rank) for rank in range(3)])
    expected_rows = []
    for q_idx in range(seq_len_q):
        visible = global_cache[: 64 + q_idx]
        logits = case.query[0, q_idx].float() @ visible.T * case.softmax_scale
        expected_rows.append(
            torch.softmax(logits, dim=-1) @ visible[:, :_LATENT_DIM] * case.output_scale
        )
    _assert_output_reference(
        merged, quantized_merged, torch.stack(expected_rows), dtype=qkv_dtype
    )
    torch.testing.assert_close(
        merged_stats, expected_merged_stats, rtol=2e-3, atol=2e-4
    )


@pytest.mark.parametrize(
    "first_position,qkv_dtype",
    (
        pytest.param(127, _FP8, id="fp8-cross128"),
        pytest.param(1023, _FP8, id="fp8-cross1024"),
        pytest.param(1023, torch.bfloat16, id="bf16-cross1024"),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_cp8_page128(
    first_position: int, qkv_dtype: torch.dtype
) -> None:
    """Single-device CP8 page-interleaving and merge oracle for H96/Q8."""

    cp_size, page_size, seq_len_q = 8, 128, 8
    total_tokens = first_position + seq_len_q

    def local_prefix(global_prefix: int, rank: int) -> int:
        full_cycles, tail = divmod(global_prefix, cp_size * page_size)
        return full_cycles * page_size + min(max(tail - rank * page_size, 0), page_size)

    kv_lengths = tuple(local_prefix(total_tokens, rank) for rank in range(cp_size))
    case = _make_bounded_case(
        num_heads=96,
        seq_len_q=seq_len_q,
        kv_lengths=kv_lengths,
        qkv_dtype=qkv_dtype,
        page_size=page_size,
    )
    case.query[1:].copy_(case.query[:1].expand(cp_size - 1, -1, -1, -1))
    bounds = torch.tensor(
        [
            local_prefix(position + 1, rank)
            for rank in range(cp_size)
            for position in range(first_position, total_tokens)
        ],
        dtype=torch.int32,
        device=case.query.device,
    )
    per_rank_bounds = bounds.reshape(cp_size, seq_len_q)
    if first_position == 127:
        assert per_rank_bounds[:2, :3].tolist() == [[128, 128, 128], [0, 1, 2]]
        assert torch.count_nonzero(per_rank_bounds[2:]) == 0
    else:
        assert per_rank_bounds[0, :3].tolist() == [128, 129, 130]
        assert per_rank_bounds[7, :3].tolist() == [128, 128, 128]

    wrapper = _plan_bounds(case, max_q_len=seq_len_q)
    assert _policy_dict(wrapper)["kernel"] == "throughput_2cta"
    output, stats = _run_bounds(wrapper, case, bounds)
    policy = _policy_dict(wrapper)
    _assert_bounded_result(case, bounds, output, stats, policy=policy)

    merged, merged_stats = _merge_reference(output, stats)
    local_expected, local_stats = _bounded_reference(case, bounds, policy=policy)
    quantized_merged, _ = _merge_reference(local_expected, local_stats)

    # Reconstruct global token order by explicit page ownership rather than the
    # prefix arithmetic above. CP8 wraps to rank zero at the ninth global page.
    positions = torch.arange(total_tokens, device=case.query.device)
    owners = (positions // page_size) % cp_size
    global_cache = torch.empty(
        (total_tokens, case.query.shape[-1]), device=case.query.device
    )
    for rank in range(cp_size):
        global_cache[owners == rank] = _gather_request_cache(case, rank)
    expected_rows, expected_stats = [], []
    for q_idx in range(seq_len_q):
        visible = global_cache[: first_position + q_idx + 1]
        logits = case.query[0, q_idx].float() @ visible.T * case.softmax_scale
        row_maximum = logits.max(dim=-1).values
        weights = torch.exp(logits - row_maximum[:, None])
        row_sum = weights.sum(dim=-1)
        expected_rows.append(
            weights @ visible[:, :_LATENT_DIM] / row_sum[:, None] * case.output_scale
        )
        expected_stats.append(torch.stack((row_maximum, row_sum), dim=-1))
    _assert_output_reference(
        merged, quantized_merged, torch.stack(expected_rows), dtype=qkv_dtype
    )
    torch.testing.assert_close(
        merged_stats, torch.stack(expected_stats), rtol=2e-3, atol=2e-4
    )


@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_override_mask_and_preserve_default(
    mask_type: str,
) -> None:
    case = _make_bounded_case(kv_lengths=(65,), mask_type=mask_type)
    bounds = torch.tensor([0, 1, 63, 64, 65, 65], dtype=torch.int32, device="cuda")
    bounded = _plan_bounds(case, max_q_len=6)
    output, stats = _run_bounds(bounded, case, bounds)
    _assert_bounded_result(case, bounds, output, stats, policy=_policy_dict(bounded))

    ordinary_bounds = torch.tensor(
        [65] * 6 if mask_type == "dense" else list(range(60, 66)),
        dtype=torch.int32,
        device="cuda",
    )
    legacy = _plan_bounds(case, max_q_len=6, use_kv_bounds=False)
    legacy_output, legacy_stats = _run_bounds(legacy, case, None)
    _assert_bounded_result(
        case, ordinary_bounds, legacy_output, legacy_stats, policy=_policy_dict(legacy)
    )
    bounded_output, bounded_stats = _run_bounds(bounded, case, ordinary_bounds)
    torch.testing.assert_close(bounded_output, legacy_output, rtol=1e-2, atol=5e-4)
    torch.testing.assert_close(bounded_stats, legacy_stats, rtol=2e-3, atol=2e-4)
    assert not torch.equal(output, legacy_output)


@pytest.mark.parametrize("packed_query", (False, True), ids=("fixed", "packed"))
@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, _FP8), ids=("bf16", "fp8"))
@pytest.mark.parametrize("num_heads", (8, 96), ids=("h8", "h96"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_cuda_graph_reloads_device_bounds(
    packed_query: bool, qkv_dtype: torch.dtype, num_heads: int
) -> None:
    case = _make_bounded_case(
        kv_lengths=(65, 65), qkv_dtype=qkv_dtype, num_heads=num_heads
    )
    qo_indptr = None
    if packed_query:
        case, qo_indptr = _pack_mla_case(case, (6, 1))
    initial = [0, 1, 1, 63, 64, 65] + ([1] if packed_query else [0, 0, 1, 1, 64, 65])
    updated = [0, 0, 0, 1, 1, 2] + ([65] if packed_query else [63, 64, 64, 65, 65, 65])
    _exercise_bounds_graph(case, initial, updated, max_q_len=6, qo_indptr=qo_indptr)


def _exercise_bounds_graph(
    case: _MLACase,
    initial: list[int],
    updated: list[int],
    *,
    max_q_len: int,
    qo_indptr: torch.Tensor | None = None,
    expected_kernel: str | None = None,
) -> None:
    bounds = torch.tensor(initial, dtype=torch.int32, device=case.query.device)
    wrapper = _plan_bounds(
        case, max_q_len=max_q_len, packed_query=qo_indptr is not None
    )
    if expected_kernel is not None:
        assert _policy_dict(wrapper)["kernel"] == expected_kernel
    eager_output, eager_stats = _run_bounds(wrapper, case, bounds, qo_indptr=qo_indptr)
    _assert_bounded_result(
        case,
        bounds,
        eager_output,
        eager_stats,
        qo_indptr=qo_indptr,
        policy=_policy_dict(wrapper),
    )
    graph_out = torch.empty_like(eager_output)
    graph_stats = torch.empty_like(eager_stats)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_stats = _run_bounds(
            wrapper,
            case,
            bounds,
            qo_indptr=qo_indptr,
            out=graph_out,
            stats=graph_stats,
            validate=False,
        )
    assert captured is graph_out
    assert captured_stats is graph_stats
    bounds_ptr = bounds.data_ptr()
    original_pages = case.block_tables[0].clone()
    original_kv_len = int(case.seq_lens_kv[0].item())
    first_q_len = int(qo_indptr[1].item()) if qo_indptr is not None else max_q_len
    for values, empty_request in (
        (updated, False),
        ([0] * first_q_len + updated[first_q_len:], True),
        (initial, False),
    ):
        case.seq_lens_kv[0] = 0 if empty_request else original_kv_len
        if empty_request:
            case.block_tables[0].fill_(-1)
        else:
            case.block_tables[0].copy_(original_pages)
        bounds.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        assert bounds.data_ptr() == bounds_ptr
        graph_out.fill_(float("nan"))
        graph_stats.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_bounded_result(
            case,
            bounds,
            graph_out,
            graph_stats,
            qo_indptr=qo_indptr,
            policy=_policy_dict(wrapper),
        )
    _CUDA_GRAPH_ANCHOR_OWNERS.append(
        (graph, wrapper, case, bounds, qo_indptr, graph_out, graph_stats)
    )


@pytest.mark.parametrize(
    "num_heads,seq_len_q,qkv_dtype,expected_kernel",
    (
        pytest.param(
            8, 6, torch.bfloat16, "throughput_latency_1cta", id="h8-q6-bf16-1cta"
        ),
        pytest.param(64, 1, _FP8, "throughput_latency_1cta", id="h64-q1-fp8-1cta"),
        pytest.param(96, 8, _FP8, "throughput_2cta", id="h96-q8-fp8-2cta"),
        pytest.param(96, 6, _FP8, "throughput_2cta", id="h96-q6-fp8-dep"),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_page128_graph_smoke(
    num_heads: int,
    seq_len_q: int,
    qkv_dtype: torch.dtype,
    expected_kernel: str,
) -> None:
    """Small page128 family/recipe geometry anchors, not a deployment replay."""

    kv_len = 127 if expected_kernel == "throughput_latency_1cta" else 129
    case = _make_bounded_case(
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        kv_lengths=(kv_len, kv_len),
        qkv_dtype=qkv_dtype,
        page_size=128,
    )
    initial = (
        [kv_len]
        if seq_len_q == 1
        else [
            min(value, kv_len)
            for value in [0, 1, 127, 128, 129, 129, 129, 129][:seq_len_q]
        ]
    )
    updated = [min(2, value) for value in initial]
    _exercise_bounds_graph(
        case,
        initial + [kv_len] * seq_len_q,
        updated + [kv_len] * seq_len_q,
        max_q_len=seq_len_q,
        expected_kernel=expected_kernel,
    )


@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, _FP8), ids=("bf16", "fp8"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_all_empty_local_rank(
    qkv_dtype: torch.dtype,
) -> None:
    case = _make_bounded_case(kv_lengths=(0, 0), qkv_dtype=qkv_dtype)
    bounds = torch.zeros(12, dtype=torch.int32, device="cuda")
    wrapper = _plan_bounds(case, max_q_len=6)
    output, stats = _run_bounds(wrapper, case, bounds)
    _assert_bounded_result(case, bounds, output, stats, policy=_policy_dict(wrapper))


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_output_only() -> None:
    """Bounds remain usable without compiling or supplying merge statistics."""

    case = _make_bounded_case(kv_lengths=(65,), qkv_dtype=_FP8)
    bounds = torch.tensor([0, 1, 63, 64, 65, 65], dtype=torch.int32, device="cuda")
    wrapper = _plan_bounds(case, max_q_len=6, store_softmax_stats=False)
    output = wrapper.run(
        case.query,
        case.kv_cache,
        case.block_tables,
        case.seq_lens_kv,
        kv_bounds=bounds,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
    )
    expected, _ = _bounded_reference(case, bounds, policy=_policy_dict(wrapper))
    fp32_expected, _ = _bounded_reference(case, bounds)
    _assert_output_reference(output, expected, fp32_expected, dtype=case.query.dtype)
    assert torch.count_nonzero(output[:, 0]) == 0


@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, _FP8), ids=("bf16", "fp8"))
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_lse_and_stats_graph(qkv_dtype):
    """Bounds and neutral rows coexist with main's optional base-2 LSE ABI."""
    case = _make_bounded_case(kv_lengths=(0, 65), qkv_dtype=qkv_dtype)
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        case.query.device,
        2,
        96,
        512,
        64,
        case.page_size,
        case.max_kv_len,
        max_q_len=6,
        packed_query=False,
        q_dtype=qkv_dtype,
        kv_dtype=qkv_dtype,
        out_dtype=case.out_dtype,
        use_kv_bounds=True,
        store_softmax_stats=True,
        lse_mode="base2",
    )
    bounds = torch.tensor(
        [0] * 6 + [0, 1, 2, 63, 64, 65], device="cuda", dtype=torch.int32
    )
    out = torch.empty(
        (*case.query.shape[:-1], 512), device="cuda", dtype=case.out_dtype
    )
    stats = torch.empty((*case.query.shape[:-1], 2), device="cuda")
    lse = torch.empty(case.query.shape[:-1], device="cuda")

    def run(validate):
        return wrapper.run(
            case.query,
            case.kv_cache,
            case.block_tables,
            case.seq_lens_kv,
            kv_bounds=bounds,
            out=out,
            softmax_stats=stats,
            lse=lse,
            return_lse=True,
            softmax_scale=case.softmax_scale,
            output_scale=case.output_scale,
            validate=validate,
        )

    result, returned_lse = run(True)
    assert result is out and returned_lse is lse
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(False)
    for prefix in ([0, 1, 2, 63, 64, 65], [0] * 6, [0, 1, 2, 63, 64, 65]):
        bounds[6:].copy_(torch.tensor(prefix, device="cuda", dtype=torch.int32))
        out.fill_(torch.nan)
        stats.fill_(torch.nan)
        lse.fill_(torch.nan)
        graph.replay()
        _assert_bounded_result(case, bounds, out, stats, policy=_policy_dict(wrapper))
        _, expected_stats = _bounded_reference(case, bounds)
        expected_lse = (
            expected_stats[..., 0] + expected_stats[..., 1].log()
        ) / torch.log(torch.tensor(2.0, device="cuda"))
        torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)
    _CUDA_GRAPH_ANCHOR_OWNERS.append((graph, wrapper, case, bounds, out, stats, lse))


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_all_empty_packed_query() -> None:
    case = _make_bounded_case(kv_lengths=(0, 65))
    case, qo_indptr = _pack_mla_case(case, (0, 0))
    bounds = torch.empty(0, dtype=torch.int32, device="cuda")
    wrapper = _plan_bounds(case, max_q_len=6, packed_query=True)
    output, stats = _run_bounds(wrapper, case, bounds, qo_indptr=qo_indptr)
    assert output.shape == (0, 96, _LATENT_DIM)
    assert stats.shape == (0, 96, 2)


@pytest.mark.parametrize("packed_query", (False, True), ids=("fixed", "packed"))
@pytest.mark.parametrize(
    "invalid",
    (
        "dtype",
        "cpu",
        "rank",
        "short",
        "long",
        "stride",
        "negative",
        "overflow",
        "decreasing",
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_rejects_invalid_inputs(
    packed_query: bool, invalid: str
) -> None:
    case = _make_bounded_case(kv_lengths=(65, 65))
    qo_indptr = None
    if packed_query:
        case, qo_indptr = _pack_mla_case(case, (6, 1))
    size = 7 if packed_query else 12
    bounds = torch.ones(size, dtype=torch.int32, device="cuda")
    if invalid == "dtype":
        bounds = bounds.to(torch.int64)
    elif invalid == "cpu":
        bounds = bounds.cpu()
    elif invalid == "rank":
        bounds = bounds.reshape(1, -1)
    elif invalid == "short":
        bounds = bounds[:-1]
    elif invalid == "long":
        bounds = torch.ones(size + 1, dtype=torch.int32, device="cuda")
    elif invalid == "stride":
        bounds = torch.ones(size * 2, dtype=torch.int32, device="cuda")[::2]
    elif invalid == "negative":
        bounds[0] = -1
    elif invalid == "overflow":
        bounds[-1] = 66
    elif invalid == "decreasing":
        bounds[0] = 2
    wrapper = _plan_bounds(case, max_q_len=6, packed_query=packed_query)
    with pytest.raises((TypeError, ValueError), match="kv_bounds"):
        _run_bounds(wrapper, case, bounds, qo_indptr=qo_indptr)


def test_attention_ts_mla_kv_bounds_cpu_fp8_reference_preserves_raw_splits() -> None:
    """Prefix masking must not repartition the raw request's FP8 P streams."""

    query = torch.zeros((1, _LATENT_DIM + _ROPE_DIM))
    query[:, 0] = 1
    cache = torch.zeros((512, _LATENT_DIM + _ROPE_DIM))
    cache[:128, 0] = 1
    cache[128:256, 1] = 1
    cache[256:, 0] = 8
    kwargs = dict(
        softmax_scale=0.2,
        output_scale=1.0,
        num_insts_kv=1,
        tile_size_kv=128,
        splits_kv=2,
    )
    bounded = _fp8_request_reference(query, cache, visible_kv_len=256, **kwargs)
    scores = query @ cache[:256].T
    probabilities = (
        torch.exp((scores - scores.max(dim=-1).values[:, None]) * 0.2) * 448.0
    )
    expected = (
        probabilities.to(_FP8).float()
        @ cache[:256, :_LATENT_DIM]
        / probabilities.sum(dim=-1)[:, None]
    )
    torch.testing.assert_close(bounded, expected, rtol=1e-6, atol=1e-6)
    trimmed = _fp8_request_reference(query, cache[:256], **kwargs)
    assert float((bounded - trimmed).abs().max()) > 1e-4
    assert (
        torch.count_nonzero(
            _fp8_request_reference(query, cache, visible_kv_len=0, **kwargs)
        )
        == 0
    )
    torch.testing.assert_close(
        _fp8_request_reference(query, cache, visible_kv_len=512, **kwargs),
        _fp8_request_reference(query, cache, **kwargs),
        rtol=0,
        atol=0,
    )


def test_attention_ts_mla_kv_bounds_cpu_reference_empty_and_page_boundary() -> None:
    """The independent oracle distinguishes zero, one, and two visible tokens."""

    case = _make_bounded_case(num_heads=8, seq_len_q=3, kv_lengths=(2,), device="cpu")
    case.query.zero_()
    page = int(case.block_tables[0, 0])
    case.kv_cache[page, 0, 0, :_LATENT_DIM] = 2
    case.kv_cache[page, 0, 1, :_LATENT_DIM] = 6
    output, stats = _bounded_reference(case, torch.tensor([0, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(output[0, :, 0, 0], torch.tensor([0.0, 2.0, 4.0]))
    torch.testing.assert_close(stats[0, :, 0, 1], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.isneginf(stats[0, 0, :, 0]).all()
    assert torch.count_nonzero(stats[0, 1:, :, 0]) == 0


def test_attention_ts_mla_kv_bounds_cpu_metadata_limits_and_legacy_inactivity(
    monkeypatch,
):
    device = torch.device("cpu")
    runtime = SimpleNamespace(query=torch.empty((1, 3, 8, 576)), num_physical_pages=1)
    monkeypatch.setattr(
        mla_decode_module, "_validate_mla_metadata", lambda *_: (device, 1, 4)
    )
    table = torch.zeros((1, 4), dtype=torch.int32)
    kwargs = dict(
        device=device,
        batch_size=1,
        required_page_columns=4,
        packed_query=False,
        max_q_len=3,
        max_kv_len=128,
        page_size=32,
        mask_type="causal",
    )
    validate = mla_decode_module._validate_mla_run_metadata
    assert (
        validate(
            runtime,
            table,
            torch.tensor([2]),
            None,
            kv_bounds=torch.tensor([0, 1, 2]),
            **kwargs,
        )
        == 3
    )
    for lengths, bounds, message in (
        ([-1], [0, 0, 0], "nonnegative"),
        ([2], [-1, 0, 2], "values"),
        ([2], [0, 1, 3], "values"),
        ([2], [0, 2, 1], "nondecreasing"),
    ):
        with pytest.raises(ValueError, match=message):
            validate(
                runtime,
                table,
                torch.tensor(lengths),
                None,
                kv_bounds=torch.tensor(bounds),
                **kwargs,
            )
    table.fill_(-1)
    assert (
        validate(
            runtime,
            table,
            torch.tensor([0]),
            None,
            kv_bounds=torch.zeros(3, dtype=torch.int32),
            **kwargs,
        )
        == 3
    )
    # Unbounded zero/negative lengths retain main's inactive-request contract.
    for inactive in (0, -1):
        assert validate(runtime, table, torch.tensor([inactive]), None, **kwargs) == 3


@pytest.mark.parametrize("use_kv_bounds", (False, True))
@pytest.mark.parametrize("validate", (False, True))
def test_attention_ts_mla_kv_bounds_cpu_presence_contract(
    monkeypatch: pytest.MonkeyPatch, use_kv_bounds: bool, validate: bool
) -> None:
    """Presence mismatch must fail before preparing or launching any runtime."""

    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper._plan_state = SimpleNamespace(
        use_kv_bounds=use_kv_bounds, store_softmax_stats=False
    )

    def fail_prepare(*args: object, **kwargs: object) -> None:
        pytest.fail("kv_bounds presence must be checked before preparing runtime")

    monkeypatch.setattr(mla_decode_module, "_prepare_mla_runtime", fail_prepare)
    bounds = None if use_kv_bounds else torch.ones(1, dtype=torch.int32)
    tensor = torch.empty(1)
    with pytest.raises(ValueError, match="use_kv_bounds"):
        wrapper.run(tensor, tensor, tensor, tensor, kv_bounds=bounds, validate=validate)


@pytest.mark.parametrize("use_kv_bounds", (False, True))
@pytest.mark.parametrize("store_softmax_stats", (False, True))
def test_attention_ts_mla_kv_bounds_cpu_plan_wiring(
    monkeypatch: pytest.MonkeyPatch, use_kv_bounds: bool, store_softmax_stats: bool
) -> None:
    """Each bounds/stats specialization is part of immutable plan/compile state."""

    policy = (("source", "auto"), ("split_kv", 1))
    spec = mla_decode_module._MLADecodeLaunchSpec(
        kernel=object(), policy=policy, kernel_workspace_bytes=0, split_kv=1
    )
    compile_kwargs = {}

    def make_compile_spec(*args: object, **kwargs: object) -> object:
        compile_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(
        mla_decode_module, "_resolve_cuda_device", lambda _: (torch.device("cpu"), 0)
    )
    monkeypatch.setattr(
        mla_decode_module, "_resolve_mla_decode_launch_spec", lambda *args: spec
    )
    monkeypatch.setattr(
        mla_decode_module, "_cached_mla_decode_compile_spec", make_compile_spec
    )
    monkeypatch.setattr(
        mla_decode_module, "_get_compiled_mla_decode", lambda _: lambda *args: None
    )
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        "cuda:0",
        1,
        8,
        _LATENT_DIM,
        _ROPE_DIM,
        64,
        128,
        max_q_len=6,
        packed_query=False,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        use_kv_bounds=use_kv_bounds,
        store_softmax_stats=store_softmax_stats,
        validate=False,
    )
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.use_kv_bounds is use_kv_bounds
    assert wrapper._plan_state.store_softmax_stats is store_softmax_stats
    assert compile_kwargs["use_kv_bounds"] is use_kv_bounds
    assert compile_kwargs["store_softmax_stats"] is store_softmax_stats


@pytest.mark.parametrize("invalid", (0, 1, None, "true"))
def test_attention_ts_mla_kv_bounds_cpu_plan_requires_boolean(invalid: object) -> None:
    wrapper = BatchMLADecodePagedTSWrapper()
    with pytest.raises(TypeError, match="use_kv_bounds must be a bool"):
        wrapper.plan(
            "cuda:0",
            1,
            8,
            _LATENT_DIM,
            _ROPE_DIM,
            64,
            128,
            max_q_len=6,
            packed_query=False,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            use_kv_bounds=invalid,
            validate=False,
        )


@pytest.mark.parametrize("use_bounds", (False, True))
@pytest.mark.parametrize("mask", ("dense", "causal"))
def test_attention_ts_mla_kv_bounds_cpu_2cta_mask_branch_ast(use_bounds, mask):
    """Execute the real mask branches without TMEM or a CUDA context."""
    import ast
    import inspect
    import textwrap
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta import (
        resources,
    )

    method = ast.parse(
        textwrap.dedent(inspect.getsource(resources.TmemSResource._load_s_impl))
    ).body[0]
    begin = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "needs_row_causal_mask"
            for target in node.targets
        )
    )
    masked_store = next(
        node
        for node in method.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "group_needs_mask"
    )
    code = compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[*method.body[begin : begin + 3], masked_store], type_ignores=[]
            )
        ),
        "<actual-2cta-mask-branches>",
        "exec",
    )
    reads = []

    def query_bounds(*args):
        reads.append("causal")
        return 0, 6

    cfg = SimpleNamespace(
        use_kv_bounds=use_bounds,
        mask_type=mask,
        mma_qk_tiler=(128, 128),
        num_mma_ctas=2,
        page_size=32,
    )
    scope = dict(
        cfg=cfg,
        self=SimpleNamespace(
            cta_rank=0,
            kv_bounds=object(),
            cu_seqlens_q=None,
            logical_num_heads_q=8,
            logical_seq_len_q=6,
            page_offsets=None,
        ),
        cutlass=SimpleNamespace(const_expr=lambda x: x, range_constexpr=range),
        cute=SimpleNamespace(math=SimpleNamespace(max=max, min=min)),
        MaskType=resources.MaskType,
        Int32=int,
        local_tidx=0,
        EPILOGUE_ROW_MASK=63,
        EPILOGUE_COLUMN_GROUP_SHIFT=6,
        work_tile=SimpleNamespace(tile_idx=(0, 0, 0, 0)),
        tile_offset_k=0,
        K=65,
        query_kv_bound=lambda *args: 2,
        query_batch_bounds=query_bounds,
        kv_tile_needs_right_mask=lambda start, width, length: start + width > length,
        cached_mask_low=None,
        mask_routes=None,
        neg_inf=-float("inf"),
        qk_acc_regs=[1.0] * 64,
        invalid_sparse_token=lambda token, length, *args, **kwargs: token >= length,
    )
    # In bounded mode this name must remain uninitialized, not merely unused
    # after a coincidental initialization by the ordinary causal branch.
    assert "logical_seq_len_q" not in scope
    exec(code, scope)
    visible = 2 if use_bounds else (60 if mask == "causal" else 64)
    assert scope["qk_acc_regs"] == [1.0] * visible + [-float("inf")] * (64 - visible)
    assert reads == ([] if use_bounds or mask == "dense" else ["causal"])
    if use_bounds:
        assert "logical_seq_len_q" not in scope
