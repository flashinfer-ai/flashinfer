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

"""Global-length CP masking and merge-compatible LSE in native Prims-TS MLA."""

from __future__ import annotations

from dataclasses import replace
import math

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
    _pack_mla_case,
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


def _plan_cp(
    case: _MLACase,
    *,
    max_q_len: int,
    packed_query: bool = False,
    cp_world: int = 8,
    cp_interleave_size: int = 128,
    zero_empty_rows: bool = True,
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
        cp_world=cp_world,
        cp_interleave_size=cp_interleave_size,
        lse_mode="base2",
        zero_empty_rows=zero_empty_rows,
    )
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.cp_world == cp_world
    assert wrapper._plan_state.cp_interleave_size == cp_interleave_size
    assert callable(wrapper._plan_state.compiled)
    assert _policy_dict(wrapper)["kernel"] in (
        "throughput_latency_1cta",
        "throughput_2cta",
    )
    return wrapper


def _run_cp(
    wrapper: BatchMLADecodePagedTSWrapper,
    case: _MLACase,
    global_lengths: torch.Tensor,
    *,
    cp_rank: int = 0,
    qo_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    stats: torch.Tensor | None = None,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stats is None:
        stats = torch.full(
            case.query.shape[:-1],
            float("nan"),
            dtype=torch.float32,
            device=case.query.device,
        )
    output = wrapper.run(
        case.query,
        case.kv_cache,
        case.block_tables,
        global_lengths,
        qo_indptr=qo_indptr,
        cp_rank=cp_rank,
        softmax_scale=case.softmax_scale,
        output_scale=case.output_scale,
        out=out,
        lse=stats,
        validate=validate,
    )
    return output, stats


def _cp_reference_inputs(
    case: _MLACase,
    global_lengths: torch.Tensor,
    *,
    cp_rank: int,
    cp_world: int,
    cp_interleave_size: int,
    qo_indptr: torch.Tensor | None = None,
) -> tuple[_MLACase, torch.Tensor]:
    """Enumerate global ownership independently of the kernel prefix formula."""
    q_lengths = (
        (qo_indptr[1:] - qo_indptr[:-1]).tolist()
        if qo_indptr is not None
        else [case.query.shape[1]] * case.block_tables.shape[0]
    )
    local_lengths, bounds = [], []
    for length, q_len in zip(global_lengths.tolist(), q_lengths, strict=True):
        positions = torch.arange(length, device=case.query.device)
        owned = positions[(positions // cp_interleave_size) % cp_world == cp_rank]
        local_lengths.append(owned.numel())
        for q_idx in range(q_len):
            endpoint = (
                length if case.mask_type == "dense" else length - q_len + q_idx + 1
            )
            bounds.append(int((owned < endpoint).sum().item()))
    return (
        replace(
            case,
            seq_lens_kv=torch.tensor(
                local_lengths, dtype=torch.int32, device=case.query.device
            ),
        ),
        torch.tensor(bounds, dtype=torch.int32, device=case.query.device),
    )


@torch.no_grad()
def _bounded_reference(
    case: _MLACase,
    bounds: torch.Tensor,
    *,
    qo_indptr: torch.Tensor | None = None,
    policy: dict[str, object] | None = None,
    bf16_single_tile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute base-2 LSE and model the selected low-precision PV operands."""

    query = case.query.reshape(-1, case.query.shape[-2], case.query.shape[-1])
    output = torch.zeros(
        (*query.shape[:-1], _LATENT_DIM), dtype=torch.float32, device=query.device
    )
    stats = torch.full(
        query.shape[:-1], -torch.inf, dtype=torch.float32, device=query.device
    )
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
                probabilities = weights
                if bf16_single_tile:
                    # This BF16 oracle covers one KV tile, without online or
                    # split rescaling. PV rounds P; the denominator does not.
                    assert case.query.dtype == torch.bfloat16 and policy is not None
                    assert cache.shape[0] <= int(policy["tile_size_kv"])
                    assert int(policy["split_kv"]) == 1
                    assert int(policy["num_insts_kv"]) == 1
                    probabilities = weights.to(torch.bfloat16).float()
                output[query_idx] = (
                    probabilities
                    @ visible_cache[:, :_LATENT_DIM]
                    / denominator[:, None]
                    * case.output_scale
                )
            stats[query_idx] = logits.logsumexp(-1) / math.log(2)
    return (
        output.reshape(*case.query.shape[:-1], _LATENT_DIM),
        stats.reshape(case.query.shape[:-1]),
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
    maximum = stats.max(dim=0).values
    mass = torch.exp2(stats - maximum)
    denominator = mass.sum(dim=0)
    merged = (output.float() * mass[..., None]).sum(dim=0) / denominator[..., None]
    return merged, maximum + denominator.log2()


def _assert_bounded_result(
    case: _MLACase,
    bounds: torch.Tensor,
    output: torch.Tensor,
    stats: torch.Tensor,
    *,
    policy: dict[str, object],
    qo_indptr: torch.Tensor | None = None,
    bf16_single_tile: bool = False,
) -> None:
    expected_output, expected_stats = _bounded_reference(
        case,
        bounds,
        qo_indptr=qo_indptr,
        policy=policy,
        bf16_single_tile=bf16_single_tile,
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
    flat_stats = stats.reshape(-1, stats.shape[-1])
    assert torch.count_nonzero(flat_output[empty]) == 0
    assert torch.isneginf(flat_stats[empty]).all()


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_cp_split_reduction() -> None:
    """Split reducers must preserve neutral rows and short partial prefixes."""

    case = _make_bounded_case(
        num_heads=96,
        seq_len_q=1,
        kv_lengths=(32769, 32769, 32769, 32769),
        qkv_dtype=_FP8,
        page_size=128,
        mask_type="dense",
    )
    global_lengths = torch.tensor(
        [0, 9, 1033, 262153], dtype=torch.int32, device="cuda"
    )
    wrapper = _plan_cp(case, max_q_len=1, cp_interleave_size=1)
    assert int(_policy_dict(wrapper)["split_kv"]) > 1
    output, stats = _run_cp(wrapper, case, global_lengths, cp_rank=1)
    local_case, bounds = _cp_reference_inputs(
        case, global_lengths, cp_rank=1, cp_world=8, cp_interleave_size=1
    )
    assert bounds.tolist() == [0, 1, 129, 32769]
    _assert_bounded_result(
        local_case, bounds, output, stats, policy=_policy_dict(wrapper)
    )


def _run_cp_ranks(
    case: _MLACase, *, global_length: int, cp_world: int, cp_interleave_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Run every rank through the same B=1 plan; rank must not specialize JIT."""

    def rank_case(rank):
        return replace(
            case,
            query=case.query[rank : rank + 1],
            block_tables=case.block_tables[rank : rank + 1],
            seq_lens_kv=case.seq_lens_kv[rank : rank + 1],
        )

    wrapper = _plan_cp(
        rank_case(0),
        max_q_len=case.query.shape[1],
        cp_world=cp_world,
        cp_interleave_size=cp_interleave_size,
    )
    compiled = wrapper._plan_state.compiled
    global_lengths = torch.tensor(
        [global_length], dtype=torch.int32, device=case.query.device
    )
    outputs, stats, all_bounds = [], [], []
    for rank in range(cp_world):
        local_case = rank_case(rank)
        local_case, bounds = _cp_reference_inputs(
            local_case,
            global_lengths,
            cp_rank=rank,
            cp_world=cp_world,
            cp_interleave_size=cp_interleave_size,
        )
        output, lse = _run_cp(wrapper, local_case, global_lengths, cp_rank=rank)
        _assert_bounded_result(
            local_case,
            bounds,
            output,
            lse,
            policy=_policy_dict(wrapper),
            bf16_single_tile=case.query.dtype == torch.bfloat16,
        )
        assert wrapper._plan_state.compiled is compiled
        outputs.append(output)
        stats.append(lse)
        all_bounds.append(bounds)
    return (
        torch.cat(outputs),
        torch.cat(stats),
        torch.cat(all_bounds),
        _policy_dict(wrapper),
    )


@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_cp_helix_page_boundary() -> None:
    """BF16 PV rounding on short local prefixes must not corrupt LSE merging."""

    case = _make_bounded_case(
        seq_len_q=6, kv_lengths=(64, 5, 0), qkv_dtype=torch.bfloat16
    )
    case.query[1:].copy_(case.query[:1].expand(2, -1, -1, -1))
    output, stats, bounds, _ = _run_cp_ranks(
        case, global_length=69, cp_world=3, cp_interleave_size=64
    )
    assert bounds.tolist() == [64] * 6 + list(range(6)) + [0] * 6

    merged, merged_stats = _merge_reference(output, stats)
    global_cache = torch.cat([_gather_request_cache(case, rank) for rank in range(3)])
    expected_rows, expected_stats = [], []
    for row in range(6):
        visible = global_cache[: 64 + row]
        logits = case.query[0, row].float() @ visible.T * case.softmax_scale
        maximum = logits.max(dim=-1).values
        weights = torch.exp(logits - maximum[:, None])
        denominator = weights.sum(dim=-1)
        expected_rows.append(
            weights
            @ visible[:, :_LATENT_DIM]
            / denominator[:, None]
            * case.output_scale
        )
        expected_stats.append(logits.logsumexp(-1) / math.log(2))
    # Keep an independent mathematical-attention gate for the merged output.
    rtol, atol = _mla_tolerances(case.query.dtype)
    torch.testing.assert_close(merged, torch.stack(expected_rows), rtol=rtol, atol=atol)
    torch.testing.assert_close(
        merged_stats, torch.stack(expected_stats), rtol=2e-3, atol=2e-4
    )


@pytest.mark.parametrize(
    "first_position",
    (
        pytest.param(127, id="cross128-empty-ranks"),
        pytest.param(1023, id="cross1024-wrap-rank0"),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_cp8_page128(first_position: int) -> None:
    """Single-device CP8 page-interleaving and merge oracle for H96/Q8."""

    cp_size, page_size, seq_len_q = 8, 128, 8
    qkv_dtype = _FP8
    total_tokens = first_position + seq_len_q

    positions = torch.arange(total_tokens, device="cuda")
    owners = (positions // page_size) % cp_size
    kv_lengths = tuple(int((owners == rank).sum().item()) for rank in range(cp_size))
    case = _make_bounded_case(
        num_heads=96,
        seq_len_q=seq_len_q,
        kv_lengths=kv_lengths,
        qkv_dtype=qkv_dtype,
        page_size=page_size,
    )
    case.query[1:].copy_(case.query[:1].expand(cp_size - 1, -1, -1, -1))
    output, stats, bounds, policy = _run_cp_ranks(
        case, global_length=total_tokens, cp_world=cp_size, cp_interleave_size=page_size
    )
    assert policy["kernel"] == "throughput_2cta"
    per_rank_bounds = bounds.reshape(cp_size, seq_len_q)
    if first_position == 127:
        assert per_rank_bounds[:2, :3].tolist() == [[128, 128, 128], [0, 1, 2]]
        assert torch.count_nonzero(per_rank_bounds[2:]) == 0
    else:
        assert per_rank_bounds[0, :3].tolist() == [128, 129, 130]
        assert per_rank_bounds[7, :3].tolist() == [128, 128, 128]

    merged, merged_stats = _merge_reference(output, stats)
    local_expected, local_stats = _bounded_reference(case, bounds, policy=policy)
    quantized_merged, _ = _merge_reference(local_expected, local_stats)

    # CP8 wraps to rank zero at the ninth global page.
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
        expected_stats.append(logits.logsumexp(-1) / math.log(2))
    _assert_output_reference(
        merged, quantized_merged, torch.stack(expected_rows), dtype=qkv_dtype
    )
    torch.testing.assert_close(
        merged_stats, torch.stack(expected_stats), rtol=2e-3, atol=2e-4
    )


def _exercise_cp_graph(
    case: _MLACase,
    initial: list[int],
    updated: list[int],
    *,
    max_q_len: int,
    cp_rank: int,
    cp_interleave_size: int,
    qo_indptr: torch.Tensor | None = None,
    updated_offsets: list[int] | None = None,
    expected_kernel: str | None = None,
    check_opt_out: bool = False,
) -> None:
    global_lengths = torch.tensor(initial, dtype=torch.int32, device=case.query.device)
    wrapper = _plan_cp(
        case,
        max_q_len=max_q_len,
        packed_query=qo_indptr is not None,
        cp_interleave_size=cp_interleave_size,
    )
    if expected_kernel is not None:
        assert _policy_dict(wrapper)["kernel"] == expected_kernel
    eager_output, eager_stats = _run_cp(
        wrapper, case, global_lengths, cp_rank=cp_rank, qo_indptr=qo_indptr
    )
    local_case, bounds = _cp_reference_inputs(
        case,
        global_lengths,
        cp_rank=cp_rank,
        cp_world=8,
        cp_interleave_size=cp_interleave_size,
        qo_indptr=qo_indptr,
    )
    _assert_bounded_result(
        local_case,
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
        captured, captured_stats = _run_cp(
            wrapper,
            case,
            global_lengths,
            cp_rank=cp_rank,
            qo_indptr=qo_indptr,
            out=graph_out,
            stats=graph_stats,
            validate=False,
        )
    assert captured is graph_out
    assert captured_stats is graph_stats
    lengths_ptr = global_lengths.data_ptr()
    original_pages = case.block_tables.clone()
    initial_offsets = qo_indptr.tolist() if qo_indptr is not None else None
    for values, offsets in (
        (updated, updated_offsets),
        ([0] * len(initial), updated_offsets),
        (initial, initial_offsets),
    ):
        global_lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        if qo_indptr is not None:
            qo_indptr.copy_(torch.tensor(offsets, dtype=torch.int32, device="cuda"))
        local_case, bounds = _cp_reference_inputs(
            case,
            global_lengths,
            cp_rank=cp_rank,
            cp_world=8,
            cp_interleave_size=cp_interleave_size,
            qo_indptr=qo_indptr,
        )
        case.block_tables.copy_(original_pages)
        case.block_tables[local_case.seq_lens_kv == 0] = -1
        assert global_lengths.data_ptr() == lengths_ptr
        graph_out.fill_(float("nan"))
        graph_stats.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_bounded_result(
            local_case,
            bounds,
            graph_out,
            graph_stats,
            qo_indptr=qo_indptr,
            policy=_policy_dict(wrapper),
        )
    _CUDA_GRAPH_ANCHOR_OWNERS.append(
        (graph, wrapper, case, global_lengths, qo_indptr, graph_out, graph_stats)
    )
    if check_opt_out:
        no_init = _plan_cp(
            case,
            max_q_len=max_q_len,
            packed_query=qo_indptr is not None,
            zero_empty_rows=False,
            cp_interleave_size=cp_interleave_size,
        )
        assert no_init._plan_state.compiled is not wrapper._plan_state.compiled
        assert (
            no_init._plan_state.workspace_layout == wrapper._plan_state.workspace_layout
        )
        output, stats = _run_cp(
            no_init, case, global_lengths, cp_rank=cp_rank, qo_indptr=qo_indptr
        )
        no_init_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(no_init_graph):
            _run_cp(
                no_init,
                case,
                global_lengths,
                cp_rank=cp_rank,
                qo_indptr=qo_indptr,
                out=output,
                stats=stats,
                validate=False,
            )
        output.fill_(float("nan"))
        stats.fill_(float("nan"))
        no_init_graph.replay()
        torch.cuda.synchronize()
        active = bounds > 0
        # Opt-out leaves empty rows unspecified; nonempty rows retain exact parity.
        torch.testing.assert_close(
            output.flatten(0, -3)[active],
            graph_out.flatten(0, -3)[active],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            stats.reshape(-1, stats.shape[-1])[active],
            graph_stats.reshape(-1, graph_stats.shape[-1])[active],
            rtol=0,
            atol=0,
        )
        _CUDA_GRAPH_ANCHOR_OWNERS.append((no_init_graph, no_init, output, stats))


@pytest.mark.parametrize(
    "num_heads,seq_len_q,expected_kernel",
    (
        pytest.param(6, 1, "throughput_latency_1cta", id="tp16-q1-1cta"),
        pytest.param(96, 8, "throughput_2cta", id="helix8-q8-2cta"),
        pytest.param(96, 6, "throughput_2cta", id="dep16-q6-2cta"),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_cp_graph_smoke(
    num_heads: int,
    seq_len_q: int,
    expected_kernel: str,
) -> None:
    """K3 TP16, Helix8/MTP7, and packed DEP16/MTP5 global lengths under a graph."""

    kv_len = 127 if expected_kernel == "throughput_latency_1cta" else 129
    case = _make_bounded_case(
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        kv_lengths=(kv_len,) * (3 if seq_len_q == 6 else 2),
        qkv_dtype=_FP8,
        page_size=128,
    )
    qo_indptr = None
    updated_offsets = None
    if seq_len_q == 1:
        initial, updated = [1015, 1009], [16, 1]
        cp_rank, interleave = 7, 1
    elif seq_len_q == 8:
        initial, updated = [135, 136], [1025, 1026]
        cp_rank, interleave = 1, 128
    else:
        case, qo_indptr = _pack_mla_case(case, (3, 6, 0))
        initial, updated = [131, 133, 0], [1030, 0, 131]
        updated_offsets = [0, 6, 6, 9]
        cp_rank, interleave = 1, 128
    _exercise_cp_graph(
        case,
        initial,
        updated,
        max_q_len=seq_len_q,
        cp_rank=cp_rank,
        cp_interleave_size=interleave,
        qo_indptr=qo_indptr,
        updated_offsets=updated_offsets,
        expected_kernel=expected_kernel,
        check_opt_out=seq_len_q in (1, 8),
    )
