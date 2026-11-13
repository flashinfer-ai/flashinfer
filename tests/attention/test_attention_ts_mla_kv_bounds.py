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

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="Prims-TS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper
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


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_split_reduction() -> None:
    """Split reducers must preserve neutral rows and short partial prefixes."""

    case = _make_bounded_case(
        num_heads=96,
        seq_len_q=1,
        kv_lengths=(32769, 32769, 32769, 32769),
        qkv_dtype=_FP8,
        page_size=128,
    )
    bounds = torch.tensor([0, 1, 129, 32769], dtype=torch.int32, device="cuda")
    wrapper = _plan_bounds(case, max_q_len=1)
    assert int(_policy_dict(wrapper)["split_kv"]) > 1
    output, stats = _run_bounds(wrapper, case, bounds)
    _assert_bounded_result(case, bounds, output, stats, policy=_policy_dict(wrapper))


@pytest.mark.parametrize(
    "first_position",
    (
        pytest.param(127, id="cross128-empty-ranks"),
        pytest.param(1023, id="cross1024-wrap-rank0"),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_cp8_page128(first_position: int) -> None:
    """Single-device CP8 page-interleaving and merge oracle for H96/Q8."""

    cp_size, page_size, seq_len_q = 8, 128, 8
    qkv_dtype = _FP8
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
    "num_heads,seq_len_q,expected_kernel",
    (
        pytest.param(6, 1, "throughput_latency_1cta", id="tp16-q1-1cta"),
        pytest.param(96, 8, "throughput_2cta", id="helix8-q8-2cta"),
        pytest.param(96, 6, "throughput_2cta", id="dep16-q6-2cta"),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_kv_bounds_page128_graph_smoke(
    num_heads: int,
    seq_len_q: int,
    expected_kernel: str,
) -> None:
    """K3 TP16, Helix8/MTP7, and DEP16/MTP5 runtime bounds under one graph."""

    kv_len = 127 if expected_kernel == "throughput_latency_1cta" else 129
    case = _make_bounded_case(
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        kv_lengths=(kv_len, kv_len),
        qkv_dtype=_FP8,
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
