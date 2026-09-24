# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deferred finalize (``plan(..., do_finalize=False)``) for the planned MXFP4
SiTU MoE: GEMM2 rows in permuted order, route weights and the
assignment-to-row map, mirroring ``trtllm_fp4_block_scale_routed_moe``
with ``do_finalize=False``.

Checks: capability / workspace / row-capacity queries without a GPU, the
gathered (token, slot) rows against the FP64 row oracle and against
trtllm-gen's unfinalized rows gathered through its own map, consistency with
the finalized plan after the caller's route-weight reduction, the EP
rank-sum and TP shard-sum identities, and CUDA Graph replay for T=1..16 in
both parallel layouts.
"""

from dataclasses import replace
import json

import pytest
import torch

from .mxfp4_situ_reference import (
    error_metrics,
    finalize_deferred_rows,
    gather_deferred_rows,
    make_case,
    make_routing,
    make_trt_baseline,
    paired_accuracy,
    reference_moe,
    reference_moe_rows,
)
from .test_cute_dsl_mxfp4_situ import (
    _compiled_counts,
    _require_blackwell,
    prepare_candidate,
)
from .test_cute_dsl_mxfp4_situ_layouts import (
    SMALL,
    SMALL_WAYS,
    _layouts,
    _mxfp4,
    _norm,
    check_partial_sum,
    prepare_rank_candidate,
    shard_case,
)


def _norm_floor(reference):
    return _norm(reference.to(torch.bfloat16).double() - reference)


def _deferred_plan(case, **wrapper_kwargs):
    """Plan the deferred form for an unsharded (all-local) case."""
    from flashinfer.fused_moe.cute_dsl.mxfp4 import CuteDslMxfp4MoEWrapper

    from .mxfp4_situ_reference import prepare_cute_weights

    wrapper = CuteDslMxfp4MoEWrapper(
        case.num_experts,
        case.topk_ids.shape[1],
        case.hidden_size,
        case.intermediate_size,
        num_local_experts=case.local_num_experts,
        local_expert_offset=case.local_expert_offset,
        **wrapper_kwargs,
    )
    tokens = case.x.shape[0]
    rows = wrapper.get_deferred_output_rows(tokens)
    output = torch.full(
        (rows, case.hidden_size),
        float("nan"),
        device=case.x.device,
        dtype=torch.bfloat16,
    )
    workspace = torch.empty(
        wrapper.get_workspace_size(tokens, do_finalize=False),
        device=case.x.device,
        dtype=torch.uint8,
    )
    plan = wrapper.plan(
        case.x,
        case.x_scale,
        case.topk_ids,
        case.topk_weights,
        *prepare_cute_weights(case),
        beta=case.beta,
        linear_beta=case.linear_beta,
        workspace=workspace,
        output=output,
        do_finalize=False,
    )
    return wrapper, plan, output, workspace


def _gathered(plan, output, case):
    tokens, top_k = case.topk_ids.shape
    return gather_deferred_rows(
        output, plan.expanded_idx_to_permuted_idx, tokens, top_k
    )


# ---------------------------------------------------------------------------
# Metadata (no GPU)
# ---------------------------------------------------------------------------


def test_capability_reports_deferred_output():
    m = _mxfp4()
    kimi = dict(hidden_size=7168, intermediate_size=3072, num_experts=896, top_k=16)
    for mode, size, rank in (("expert_parallel", 8, 3), ("moe_tensor_parallel", 8, 0)):
        layout = m.Mxfp4MoEParallelLayout(mode=mode, size=size, rank=rank)
        finalized = m.mxfp4_moe_capability(gpu_arch=103, parallel_layout=layout, **kimi)
        deferred = m.mxfp4_moe_capability(
            gpu_arch=103, parallel_layout=layout, do_finalize=False, **kimi
        )
        assert finalized.supported and finalized.deferred_output
        assert deferred.supported and deferred.cuda_graph and deferred.deferred_output
        assert deferred.layout == finalized.layout
    swiglu = m.mxfp4_moe_capability(
        gpu_arch=103,
        activation_type=m.ActivationType.Swiglu,
        do_finalize=False,
        num_local_experts=112,
        local_expert_offset=0,
        **kimi,
    )
    assert not swiglu.supported and "deferred" in swiglu.reason
    assert not swiglu.deferred_output


@pytest.mark.parametrize("tokens", [1, 16, 128, 4096])
@pytest.mark.parametrize("mode", ["expert_parallel", "moe_tensor_parallel"])
def test_deferred_workspace_and_row_queries(tokens, mode):
    m = _mxfp4()
    layout = m.Mxfp4MoEParallelLayout(mode=mode, size=8, rank=3)
    wrapper = m.CuteDslMxfp4MoEWrapper(896, 16, 7168, 3072, parallel_layout=layout)
    rows = wrapper.get_deferred_output_rows(tokens)
    tile = wrapper._swap_tile(tokens)
    tiles = m.get_max_num_tiles(tokens, 16, wrapper.num_local_experts, tile)
    assert rows == tiles * tile
    assert rows >= min(tokens * 16, tiles * tile) and rows % tile == 0
    # The deferred workspace always holds the swap-AB regions, independent of
    # the finalized path's token threshold.
    deferred = wrapper.get_workspace_size(tokens, do_finalize=False)
    assert deferred > 0 and deferred % 256 == 0
    if wrapper._use_swapab(tokens):
        finalized = wrapper.get_workspace_size(tokens)
        if wrapper._swap_hybrid(tokens):
            # Hybrid finalize: 128-row sort groups with the dispatch work
            # lists and no permuted partial rows (the dense GEMM2 reduces
            # into the output); the deferred form keeps the swap layout.
            names = {f.name for f in wrapper._workspace_fields(tokens, True)[0]}
            assert {
                "swap_row_groups",
                "swap_row_group_count",
                "swap_wide_list",
                "swap_wide_count",
            } <= names
            assert "partial_rows" not in names
            assert finalized < deferred + rows * 7168 * 2
        elif (
            tokens > m.SWAP_ATOMIC_FINALIZE_MAX_TOKENS
            and wrapper.intermediate_shard <= m.SWAP_TWO_STAGE_MAX_SHARD
        ):
            # Two-stage finalize keeps the permuted rows in workspace; the
            # deferred caller owns that buffer instead.
            partial = rows * 7168 * 2
            assert finalized - deferred == partial + (-partial) % 256
        else:
            assert deferred == finalized
    with pytest.raises(ValueError):
        wrapper.get_deferred_output_rows(0)
    # The swap-AB chain is PDL-launched internally, so the deferred form is
    # available with the caller's PDL as well and sizes identically.
    pdl = m.CuteDslMxfp4MoEWrapper(
        896, 16, 7168, 3072, parallel_layout=layout, enable_pdl=True
    )
    assert pdl.get_deferred_output_rows(tokens) == wrapper.get_deferred_output_rows(
        tokens
    )
    assert pdl.get_workspace_size(tokens, do_finalize=False) == deferred


# ---------------------------------------------------------------------------
# Numerics (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 7, 16, 40, 300])
@pytest.mark.parametrize("distribution", ["balanced", "hot", "empty"])
def test_deferred_rows_match_fp64_and_finalized(tokens, distribution, record_property):
    """Gathered (token, slot) rows follow the FP64 row oracle and reduce to
    the finalized plan's output; non-local slots map to -1."""
    _require_blackwell()
    case = make_case(tokens=tokens, distribution=distribution)
    wrapper, plan, output, _ = _deferred_plan(case)
    assert plan.deferred
    plan.run()
    torch.cuda.synchronize()
    ids = case.topk_ids
    local = (ids >= case.local_expert_offset) & (
        ids < case.local_expert_offset + case.local_num_experts
    )
    mapping = plan.expanded_idx_to_permuted_idx
    assert mapping.shape == ids.shape
    assert bool(((mapping >= 0) == local).all())
    rows = mapping[local].to(torch.long)
    assert rows.numel() == int(local.sum())
    assert bool((rows < output.shape[0]).all())
    assert rows.unique().numel() == rows.numel(), "assignments share a row"
    assert torch.isfinite(output[rows]).all()
    torch.testing.assert_close(
        plan.route_weights.double(), case.topk_weights.double(), atol=0, rtol=0
    )

    gathered = _gathered(plan, output, case)
    oracles = reference_moe_rows(case)
    report = {name: error_metrics(gathered, ref) for name, ref in oracles.items()}
    # Development gate (not a customer tolerance), analogous to the finalized
    # paired tests: the rows' distance to the ideal FP64 oracle is bounded by
    # twice the reference MXFP8 middle-rounding distance (the kernel's own
    # UE8M0 requantization rounds differently) plus two BF16 output floors.
    ideal, quantized = oracles["ideal_fp64"], oracles["mxfp8_fp32"]
    assert report["ideal_fp64"]["finite"]
    assert _norm(gathered - ideal) <= 2 * _norm(quantized - ideal) + 2 * _norm_floor(
        ideal
    ), report

    # The caller's finalization of the rows is as close to the FP64 finalized
    # reference as the finalized plan (both round the same GEMM2 accumulators
    # to BF16; allow one BF16 floor for the different reduction order).
    finalized_plan, finalized_output, _ = prepare_candidate(case)
    finalized_plan.run()
    reduced = finalize_deferred_rows(output, mapping, plan.route_weights)
    reference = reference_moe(case, modes=("ideal_fp64",))["ideal_fp64"]
    report["finalized_from_rows"] = error_metrics(reduced, reference)
    report["finalized_plan"] = error_metrics(finalized_output, reference)
    assert report["finalized_from_rows"]["finite"]
    assert _norm(reduced - reference) <= _norm(
        finalized_output.double() - reference
    ) + _norm_floor(reference), report
    record_property("numerical_report", json.dumps(report))


@pytest.mark.parametrize("tokens", [1, 16, 128])
@pytest.mark.parametrize("distribution", ["balanced", "hot"])
def test_deferred_matches_trtllm_unfinalized(tokens, distribution, record_property):
    """Rows gathered through each side's map agree with the FP64 row oracle
    at least as well as trtllm-gen ``do_finalize=False`` (one BF16 floor)."""
    _require_blackwell()
    case = make_case(tokens=tokens, distribution=distribution)
    wrapper, plan, output, _ = _deferred_plan(case)
    baseline, holder = make_trt_baseline(case, do_finalize=False)
    plan.run()
    baseline()
    torch.cuda.synchronize()
    trt_rows, trt_weights, trt_map = holder["value"]
    top_k = case.topk_ids.shape[1]
    candidate = _gathered(plan, output, case).reshape(tokens * top_k, -1)
    trt = gather_deferred_rows(trt_rows, trt_map, tokens, top_k).reshape(
        tokens * top_k, -1
    )
    # Both maps mark exactly the local assignments.
    assert bool(
        (
            (plan.expanded_idx_to_permuted_idx >= 0)
            == (trt_map.reshape(tokens, top_k) >= 0)
        ).all()
    )
    torch.testing.assert_close(
        trt_weights.reshape(tokens, top_k).double(),
        plan.route_weights.double(),
        atol=1e-2,
        rtol=1e-2,
    )
    oracles = {
        name: rows.reshape(tokens * top_k, -1)
        for name, rows in reference_moe_rows(case).items()
    }
    report = {
        name: paired_accuracy(candidate, trt, ref) for name, ref in oracles.items()
    }
    record_property("numerical_report", json.dumps(report))
    assert report["ideal_fp64"]["candidate"]["finite"]
    assert report["ideal_fp64"]["baseline"]["finite"]
    ref = oracles["ideal_fp64"]
    assert _norm(candidate - ref) <= _norm(trt - ref) + _norm_floor(ref), report


@pytest.mark.parametrize("mode", ["expert_parallel", "moe_tensor_parallel"])
@pytest.mark.parametrize("tokens", [5, 16, 48])
def test_deferred_rank_sum_matches_unsharded(mode, tokens, record_property):
    """Finalizing every rank's deferred rows and summing the ranks reproduces
    the unsharded FP64 reference (EP rank sum / TP shard sum)."""
    _require_blackwell()
    unsharded = make_case(tokens=tokens, **SMALL)
    ids, weights = make_routing(tokens, 8, 2, 8, 0, "balanced")
    unsharded = replace(unsharded, topk_ids=ids, topk_weights=weights)
    partials, refs = [], []
    for layout in _layouts(mode, SMALL_WAYS):
        rank_case = shard_case(unsharded, layout)
        wrapper = _mxfp4().CuteDslMxfp4MoEWrapper(
            rank_case.num_experts,
            rank_case.topk_ids.shape[1],
            rank_case.hidden_size,
            unsharded.intermediate_size,
            parallel_layout=layout,
        )
        rows = wrapper.get_deferred_output_rows(tokens)
        output = torch.empty(
            (rows, rank_case.hidden_size), device="cuda", dtype=torch.bfloat16
        )
        workspace = torch.empty(
            wrapper.get_workspace_size(tokens, do_finalize=False),
            device="cuda",
            dtype=torch.uint8,
        )
        from .mxfp4_situ_reference import prepare_cute_weights

        plan = wrapper.plan(
            rank_case.x,
            rank_case.x_scale,
            rank_case.topk_ids,
            rank_case.topk_weights,
            *prepare_cute_weights(rank_case),
            beta=rank_case.beta,
            linear_beta=rank_case.linear_beta,
            workspace=workspace,
            output=output,
            do_finalize=False,
        )
        plan.run()
        partials.append(
            finalize_deferred_rows(
                output, plan.expanded_idx_to_permuted_idx, plan.route_weights
            ).to(torch.bfloat16)
        )
        refs.append(reference_moe(rank_case, modes=("ideal_fp64",))["ideal_fp64"])
        del plan
    whole, whole_output, _ = prepare_candidate(unsharded)
    whole.run()
    reference = reference_moe(unsharded, modes=("ideal_fp64",))["ideal_fp64"]
    report = check_partial_sum(
        partials, refs, reference, {"unsharded_candidate": whole_output}
    )
    record_property("numerical_report", json.dumps(report))


@pytest.mark.parametrize("mode", ["expert_parallel", "moe_tensor_parallel"])
@pytest.mark.parametrize("tokens", list(range(1, 17)))
def test_deferred_graph_replay(mode, tokens):
    """Capture once, replay under changed routing / SiTU parameters; the
    replayed rows and map equal eager execution and no recompilation happens."""
    _require_blackwell()
    layout = _layouts(mode, SMALL_WAYS)[1]
    rank_case = shard_case(make_case(tokens=tokens, **SMALL), layout)
    plan, output, _ = prepare_rank_candidate(rank_case, layout, do_finalize=False)
    plan.run()
    compiled_before = _compiled_counts()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    torch.cuda.current_stream().wait_stream(stream)
    for distribution, beta in (("hot", 2.5), ("empty", 4.0), ("balanced", 7.0)):
        ids, weights = make_routing(tokens, 8, 2, 8, 0, distribution)
        rank_case.topk_ids.copy_(ids)
        rank_case.topk_weights.copy_(weights)
        rank_case.beta.fill_(beta)
        plan.run()
        eager_rows = _gathered(plan, output, rank_case).clone()
        eager_map = plan.expanded_idx_to_permuted_idx.clone()
        eager_reduced = finalize_deferred_rows(
            output, plan.expanded_idx_to_permuted_idx, plan.route_weights
        )
        for _ in range(10):
            graph.replay()
        torch.testing.assert_close(plan.expanded_idx_to_permuted_idx, eager_map)
        replayed = _gathered(plan, output, rank_case)
        torch.testing.assert_close(replayed, eager_rows, atol=1e-2, rtol=1e-2)
        reference = reference_moe(rank_case, modes=("ideal_fp64",))["ideal_fp64"]
        reduced = finalize_deferred_rows(
            output, plan.expanded_idx_to_permuted_idx, plan.route_weights
        )
        assert torch.isfinite(reduced).all()
        assert _norm(reduced - reference) <= _norm(
            eager_reduced - reference
        ) + _norm_floor(reference)
    assert _compiled_counts() == compiled_before


def test_deferred_plan_rejects_bad_output_with_and_without_pdl():
    _require_blackwell()
    case = make_case(tokens=4)
    from flashinfer.fused_moe.cute_dsl.mxfp4 import CuteDslMxfp4MoEWrapper

    from .mxfp4_situ_reference import prepare_cute_weights

    wrapper = CuteDslMxfp4MoEWrapper(
        case.num_experts,
        case.topk_ids.shape[1],
        case.hidden_size,
        case.intermediate_size,
        num_local_experts=case.local_num_experts,
        local_expert_offset=case.local_expert_offset,
    )
    weights = prepare_cute_weights(case)
    workspace = torch.empty(
        wrapper.get_workspace_size(4, do_finalize=False),
        device="cuda",
        dtype=torch.uint8,
    )
    rows = wrapper.get_deferred_output_rows(4)
    short = torch.empty(
        (rows - 1, case.hidden_size), device="cuda", dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="deferred output"):
        wrapper.plan(
            case.x,
            case.x_scale,
            case.topk_ids,
            case.topk_weights,
            *weights,
            beta=case.beta,
            linear_beta=case.linear_beta,
            workspace=workspace,
            output=short,
            do_finalize=False,
        )
    pdl = CuteDslMxfp4MoEWrapper(
        case.num_experts,
        case.topk_ids.shape[1],
        case.hidden_size,
        case.intermediate_size,
        num_local_experts=case.local_num_experts,
        local_expert_offset=case.local_expert_offset,
        enable_pdl=True,
    )
    assert pdl.get_deferred_output_rows(4) == rows
    with pytest.raises(ValueError, match="deferred output"):
        pdl.plan(
            case.x,
            case.x_scale,
            case.topk_ids,
            case.topk_weights,
            *weights,
            beta=case.beta,
            linear_beta=case.linear_beta,
            workspace=workspace,
            output=short,
            do_finalize=False,
        )
