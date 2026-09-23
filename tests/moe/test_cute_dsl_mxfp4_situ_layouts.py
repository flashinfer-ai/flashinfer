"""Explicit EP/TP layouts for the planned MXFP4 SiTU MoE. Minimum GPU: SM100.

Expert parallelism: rank ``r`` owns global experts ``[r*E/8, (r+1)*E/8)``
and the sum of the eight rank outputs equals the unsharded result. MoE
tensor parallelism: every rank owns all experts and a ``3072/8 = 384``
intermediate shard; the sum of the eight shard outputs equals the unsharded
result. Both identities are checked against the same FP64 oracle as
``test_cute_dsl_mxfp4_situ.py``. Full Kimi geometry requires
``FLASHINFER_KIMI_K3_FULL=1`` on B300.
"""

from dataclasses import replace
import json
import os

import pytest
import torch

from .mxfp4_situ_reference import (
    error_metrics,
    make_case,
    make_routing,
    make_trt_baseline,
    paired_accuracy,
    parallel_routing_histogram,
    prepare_cute_weights,
    prepare_trt_weights,
    reference_moe,
    routing_histogram,
)
from .test_cute_dsl_mxfp4_situ import (
    _check_concurrent_caller_streams,
    _compiled_counts,
    _require_blackwell,
    prepare_candidate,
)

KIMI = dict(hidden_size=7168, intermediate_size=3072, num_experts=896, top_k=16)
# Small geometry: 8 experts, top-k 2, I=512 so that 4 TP shards stay 128-wide.
SMALL = dict(
    hidden=256,
    intermediate=512,
    num_experts=8,
    local_num_experts=8,
    local_expert_offset=0,
    top_k=2,
)
SMALL_WAYS = 4


def _mxfp4():
    return pytest.importorskip("flashinfer.fused_moe.cute_dsl.mxfp4")


def _layouts(mode, size):
    layout = _mxfp4().Mxfp4MoEParallelLayout
    return [layout(mode, size, rank) for rank in range(size)]


def _global_intermediate(case, layout):
    if layout.mode == "moe_tensor_parallel":
        return case.intermediate_size * layout.size
    return case.intermediate_size


def shard_case(unsharded, layout):
    """Rank-local weights and SiTU parameters; inputs and routing are shared."""
    from flashinfer.fused_moe.prepare import shard_cute_dsl_mxfp4_weights

    resolved = _mxfp4().resolve_mxfp4_moe_layout(
        unsharded.num_experts, unsharded.intermediate_size, parallel_layout=layout
    )
    sizes = (
        dict(ep_size=layout.size, ep_rank=layout.rank)
        if layout.mode == "expert_parallel"
        else dict(moe_tp_size=layout.size, moe_tp_rank=layout.rank)
        if layout.mode == "moe_tensor_parallel"
        else {}
    )
    w1, w1_scale, w2, w2_scale = shard_cute_dsl_mxfp4_weights(
        unsharded.w1, unsharded.w1_scale, unsharded.w2, unsharded.w2_scale, **sizes
    )

    def local_parameters(values):
        if values is None or values.numel() == 1:
            return values
        start = resolved.local_expert_offset
        return values[start : start + resolved.num_local_experts].contiguous()

    return replace(
        unsharded,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        beta=local_parameters(unsharded.beta),
        linear_beta=local_parameters(unsharded.linear_beta),
        local_expert_offset=resolved.local_expert_offset,
    )


def prepare_rank_candidate(
    case,
    layout,
    *,
    packed=False,
    prepared_weights=None,
    enable_pdl=False,
    do_finalize=True,
    **wrapper_kwargs,
):
    """Plan one rank from explicit layout metadata and its rank-local weights.

    ``do_finalize=False`` plans the deferred form; ``output`` is then the
    ``[rows, H]`` permuted-row buffer."""
    from .mxfp4_situ_reference import pack_topk

    wrapper = _mxfp4().CuteDslMxfp4MoEWrapper(
        case.num_experts,
        case.topk_ids.shape[1],
        case.hidden_size,
        _global_intermediate(case, layout),
        parallel_layout=layout,
        enable_pdl=enable_pdl,
        **wrapper_kwargs,
    )
    assert wrapper.num_local_experts == case.local_num_experts
    assert wrapper.local_expert_offset == case.local_expert_offset
    assert wrapper.intermediate_shard == case.intermediate_size
    weights = (
        prepare_cute_weights(case) if prepared_weights is None else prepared_weights
    )
    tokens = case.x.shape[0]
    rows = tokens if do_finalize else wrapper.get_deferred_output_rows(tokens)
    output = torch.empty(
        (rows, case.hidden_size), device=case.x.device, dtype=torch.bfloat16
    )
    workspace = torch.empty(
        wrapper.get_workspace_size(tokens, do_finalize),
        device=case.x.device,
        dtype=torch.uint8,
    )
    ids = pack_topk(case.topk_ids, case.topk_weights) if packed else case.topk_ids
    plan = wrapper.plan(
        case.x,
        case.x_scale,
        ids,
        None if packed else case.topk_weights,
        *weights,
        beta=case.beta,
        linear_beta=case.linear_beta,
        workspace=workspace,
        output=output,
        do_finalize=do_finalize,
    )
    return plan, output, workspace


def _norm(value):
    return torch.linalg.vector_norm(value)


def check_partial_sum(partials, partial_refs, reference, comparisons):
    """Assert the rank-sum identity against the FP64 oracle.

    The FP64 partial references must sum to the unsharded reference (this
    validates the slicing independently of the kernels). The BF16 rank
    outputs are summed in FP64; their error may exceed each comparison
    implementation's error by at most one BF16 representation floor per
    rounded partial. This generalizes the single-output development gate
    of the paired tests; it is not a customer acceptance rule.
    """
    total_ref = sum(partial_refs)
    torch.testing.assert_close(total_ref, reference, atol=1e-9, rtol=1e-9)
    total = sum(partial.double() for partial in partials)
    floors = sum(_norm(ref.to(torch.bfloat16).double() - ref) for ref in partial_refs)
    error = _norm(total - reference)
    report = {
        "rank_sum": error_metrics(total, reference),
        "partial_bf16_floor": float(floors),
        "partials": [
            error_metrics(p, r) for p, r in zip(partials, partial_refs, strict=False)
        ],
    }
    assert report["rank_sum"]["finite"]
    for name, output in comparisons.items():
        report[name] = error_metrics(output, reference)
        assert report[name]["finite"], report
        assert error <= _norm(output.double() - reference) + floors, (name, report)
    return report


def _run_ranks(unsharded, layouts, *, prepared=None, **wrapper_kwargs):
    partials, refs = [], []
    for index, layout in enumerate(layouts):
        rank_case = shard_case(unsharded, layout)
        plan, output, _ = prepare_rank_candidate(
            rank_case,
            layout,
            prepared_weights=None if prepared is None else prepared[index],
            **wrapper_kwargs,
        )
        plan.run()
        partials.append(output)
        refs.append(reference_moe(rank_case, modes=("ideal_fp64",))["ideal_fp64"])
        # Release this rank's prepared bank and FP64 scratch before the next
        # rank: eight full-Kimi ranks otherwise exhaust a 268 GiB device.
        del plan, rank_case
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return partials, refs


# ---------------------------------------------------------------------------
# Metadata (no GPU)
# ---------------------------------------------------------------------------


def test_capability_resolves_explicit_layouts():
    mx = _mxfp4()
    layout = mx.Mxfp4MoEParallelLayout
    ep = mx.mxfp4_moe_capability(
        gpu_arch=103, **KIMI, parallel_layout=layout("expert_parallel", 8, 3)
    )
    assert ep.supported and ep.cuda_graph
    assert ep.layout == mx.Mxfp4MoERankLayout(
        "expert_parallel", 896, 3072, 112, 336, 3072, 8, 3
    )
    tp = mx.mxfp4_moe_capability(
        gpu_arch=103, **KIMI, parallel_layout=layout("moe_tensor_parallel", 8, 5)
    )
    assert tp.supported and tp.cuda_graph
    assert tp.layout == mx.Mxfp4MoERankLayout(
        "moe_tensor_parallel", 896, 3072, 896, 0, 384, 8, 5
    )
    assert tp.layout.gemm1_n == 768 and tp.layout.gemm2_k == 384
    # The explicit expert-interval form is expert parallelism.
    explicit = mx.mxfp4_moe_capability(
        gpu_arch=103, **KIMI, num_local_experts=112, local_expert_offset=336
    )
    assert explicit.layout == ep.layout
    both = mx.mxfp4_moe_capability(
        gpu_arch=103,
        **KIMI,
        num_local_experts=112,
        local_expert_offset=336,
        parallel_layout=layout("expert_parallel", 8, 3),
    )
    assert both.layout == ep.layout
    single = mx.mxfp4_moe_capability(gpu_arch=103, **KIMI)
    assert single.layout.mode == "single"
    assert (single.layout.num_local_experts, single.layout.intermediate_shard) == (
        896,
        3072,
    )
    interval = mx.resolve_mxfp4_moe_layout(
        8, 512, num_local_experts=4, local_expert_offset=2
    )
    assert interval.mode == "expert_parallel"
    assert (interval.parallel_size, interval.parallel_rank) == (None, None)
    assert layout.from_sizes(ep_size=8, ep_rank=3) == layout("expert_parallel", 8, 3)
    assert layout.from_sizes(moe_tp_size=8, moe_tp_rank=1) == layout(
        "moe_tensor_parallel", 8, 1
    )
    assert layout.from_sizes() == layout()


def test_capability_rejects_hybrid_and_invalid_layouts():
    mx = _mxfp4()
    layout = mx.Mxfp4MoEParallelLayout
    with pytest.raises(ValueError, match="hybrid"):
        layout.from_sizes(ep_size=8, moe_tp_size=8)
    with pytest.raises(ValueError, match="rank"):
        layout("expert_parallel", 8, 8)
    with pytest.raises(ValueError, match="rank"):
        layout("moe_tensor_parallel", 8, -1)
    with pytest.raises(ValueError, match="mode"):
        layout("hybrid", 1, 0)
    with pytest.raises(ValueError, match="single"):
        layout("single", 2, 0)
    for bad, expected in (
        (layout("expert_parallel", 5, 0), "divisible"),
        (layout("moe_tensor_parallel", 5, 0), "divisible"),
        (layout("moe_tensor_parallel", 48, 0), "multiple of 128"),
    ):
        result = mx.mxfp4_moe_capability(gpu_arch=103, **KIMI, parallel_layout=bad)
        assert not result.supported and result.layout is None
        assert expected in result.reason
    # Explicit interval and derived layout must agree.
    for kwargs in (
        dict(num_local_experts=112, local_expert_offset=0),
        dict(num_local_experts=100),
        dict(local_expert_offset=224),
    ):
        result = mx.mxfp4_moe_capability(
            gpu_arch=103,
            **KIMI,
            parallel_layout=layout("expert_parallel", 8, 3),
            **kwargs,
        )
        assert not result.supported and "inconsistent" in result.reason
    result = mx.mxfp4_moe_capability(
        gpu_arch=103,
        **KIMI,
        num_local_experts=112,
        parallel_layout=layout("moe_tensor_parallel", 8, 0),
    )
    assert not result.supported and "inconsistent" in result.reason
    with pytest.raises(ValueError, match="inconsistent"):
        mx.CuteDslMxfp4MoEWrapper(
            896,
            16,
            7168,
            3072,
            num_local_experts=112,
            parallel_layout=layout("moe_tensor_parallel", 8, 2),
        )


def test_tp_workspace_size_follows_intermediate_shard():
    mx = _mxfp4()
    layout = mx.Mxfp4MoEParallelLayout
    dense = dict(swapab_max_tokens=0)
    tp = mx.CuteDslMxfp4MoEWrapper(
        896,
        16,
        7168,
        3072,
        parallel_layout=layout("moe_tensor_parallel", 8, 1),
        **dense,
    )
    single_shard = mx.CuteDslMxfp4MoEWrapper(896, 16, 7168, 384, **dense)
    ep = mx.CuteDslMxfp4MoEWrapper(
        896, 16, 7168, 3072, parallel_layout=layout("expert_parallel", 8, 3), **dense
    )
    assert (tp.parallel_mode, tp.intermediate_shard) == ("moe_tensor_parallel", 384)
    for tokens in (1, 16, 512, 4096):
        assert tp.get_workspace_size(tokens) == single_shard.get_workspace_size(tokens)
        assert tp.get_workspace_size(tokens) < ep.get_workspace_size(tokens)
    # Default swap-AB caps follow the intermediate shard (B300 crossover points:
    # the 384-wide MoE-TP shard continues in the hybrid form up to T=2048, the
    # 3072-wide expert-parallel rank switches to the dense path after T=1024)
    # and the swap workspace is smaller than the 128-row token-tile layout it
    # replaces; past the cap both wrappers carve the same dense workspace.
    tp_swap = mx.CuteDslMxfp4MoEWrapper(
        896, 16, 7168, 3072, parallel_layout=layout("moe_tensor_parallel", 8, 1)
    )
    ep_swap = mx.CuteDslMxfp4MoEWrapper(
        896, 16, 7168, 3072, parallel_layout=layout("expert_parallel", 8, 3)
    )
    assert (tp_swap.swapab_max_tokens, ep_swap.swapab_max_tokens) == (2048, 1024)
    for tokens in (1, 16):
        assert ep_swap.get_workspace_size(tokens) < ep.get_workspace_size(tokens)
        assert tp_swap.get_workspace_size(tokens) < tp.get_workspace_size(tokens)
    assert ep_swap.get_workspace_size(17) != ep.get_workspace_size(17)
    assert ep_swap.get_workspace_size(1025) == ep.get_workspace_size(1025)
    # The MoE-TP shard's hybrid form still carves its own workspace at T=1025.
    assert tp_swap.get_workspace_size(1025) != tp.get_workspace_size(1025)
    assert tp_swap.get_workspace_size(2049) == tp.get_workspace_size(2049)
    # Documented sizes of the 128-row layout; expert parallelism keeps it.
    assert [ep.get_workspace_size(t) for t in (1, 16, 4096)] == [
        6499072,
        45885440,
        253748224,
    ]
    assert [tp.get_workspace_size(t) for t in (1, 16, 512, 4096)] == [
        828160,
        13120000,
        48907264,
        72543744,
    ]


def test_shard_helper_slices_expected_rows_and_columns():
    from flashinfer.fused_moe.prepare import (
        prepare_cute_dsl_mxfp4_weights,
        shard_cute_dsl_mxfp4_weights,
    )

    experts, hidden, intermediate = 8, 256, 512
    generator = torch.Generator().manual_seed(0)
    bank = [
        torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        for shape in (
            (experts, 2 * intermediate, hidden // 2),
            (experts, 2 * intermediate, hidden // 32),
            (experts, hidden, intermediate // 2),
            (experts, hidden, intermediate // 32),
        )
    ]
    w1, w1_scale, w2, w2_scale = bank
    width = intermediate // 4
    for rank in range(4):
        shard = shard_cute_dsl_mxfp4_weights(*bank, moe_tp_size=4, moe_tp_rank=rank)
        up = slice(rank * width, (rank + 1) * width)
        gate = slice(intermediate + rank * width, intermediate + (rank + 1) * width)
        assert torch.equal(shard[0], torch.cat((w1[:, up], w1[:, gate]), dim=1))
        assert torch.equal(
            shard[1], torch.cat((w1_scale[:, up], w1_scale[:, gate]), dim=1)
        )
        assert torch.equal(
            shard[2], w2[:, :, rank * width // 2 : (rank + 1) * width // 2]
        )
        assert torch.equal(
            shard[3], w2_scale[:, :, rank * width // 32 : (rank + 1) * width // 32]
        )
        assert all(t.is_contiguous() for t in shard)
        # Tensor-parallel shards never alias the bank.
        assert shard[0].untyped_storage().data_ptr() != w1.untyped_storage().data_ptr()
        prepare_cute_dsl_mxfp4_weights(*shard)
        view = shard_cute_dsl_mxfp4_weights(*bank, ep_size=4, ep_rank=rank)
        assert torch.equal(view[0], w1[2 * rank : 2 * rank + 2])
        assert view[2].untyped_storage().data_ptr() == w2.untyped_storage().data_ptr()
        copy = shard_cute_dsl_mxfp4_weights(*bank, ep_size=4, ep_rank=rank, copy=True)
        assert torch.equal(copy[0], view[0])
        assert copy[0].untyped_storage().data_ptr() != w1.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="hybrid"):
        shard_cute_dsl_mxfp4_weights(*bank, ep_size=2, moe_tp_size=2)
    with pytest.raises(ValueError, match="divisible"):
        shard_cute_dsl_mxfp4_weights(*bank, ep_size=3)
    with pytest.raises(ValueError, match="multiples of 128"):
        shard_cute_dsl_mxfp4_weights(*bank, moe_tp_size=8)
    with pytest.raises(ValueError, match="rank"):
        shard_cute_dsl_mxfp4_weights(*bank, moe_tp_size=4, moe_tp_rank=4)


def test_remote_dominated_routing_and_rank_histogram():
    ids, _ = make_routing(129, 896, 16, 112, 336, "remote_dominated", device="cpu")
    local = (ids >= 336) & (ids < 448)
    assert (local.sum(dim=1) == 1).all()
    assert float(local.float().mean()) <= 0.1
    histogram = parallel_routing_histogram(ids, 896, 8)
    assert sum(histogram["per_rank_assignments"]) == 129 * 16
    assert histogram["per_rank_assignments"][3] == 129
    assert histogram["remote_fraction_per_rank"][3] >= 0.9
    assert len(histogram["per_rank"]) == 8 and len(histogram["per_rank"][0]) == 112
    counts = torch.bincount(ids.flatten().long(), minlength=896)
    assert histogram["global"] == counts.tolist()
    assert histogram["per_rank"][3] == counts[336:448].tolist()
    with pytest.raises(ValueError, match="divisible"):
        parallel_routing_histogram(ids, 896, 5)


# ---------------------------------------------------------------------------
# Small geometry (any SM100/SM103)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 16, 129])
@pytest.mark.parametrize("mode", ["expert_parallel", "moe_tensor_parallel"])
@pytest.mark.parametrize("distribution", ["balanced", "remote_dominated"])
def test_small_rank_sum_matches_unsharded(tokens, mode, distribution, record_property):
    _require_blackwell()
    unsharded = make_case(tokens=tokens, **SMALL)
    # remote_dominated is defined relative to EP rank 1 (experts 2..3).
    ids, weights = make_routing(tokens, 8, 2, 2, 2, distribution)
    unsharded = replace(unsharded, topk_ids=ids, topk_weights=weights)
    layouts = _layouts(mode, SMALL_WAYS)
    whole, whole_output, _ = prepare_candidate(unsharded)
    baseline, baseline_output = make_trt_baseline(unsharded)
    whole.run()
    baseline()
    partials, refs = _run_ranks(unsharded, layouts)
    reference = reference_moe(unsharded, modes=("ideal_fp64",))["ideal_fp64"]
    report = check_partial_sum(
        partials,
        refs,
        reference,
        {"unsharded_candidate": whole_output, "unsharded_trtllm_gen": baseline_output},
    )
    if mode == "expert_parallel":
        for rank, partial in enumerate(partials):
            owned = (ids >= 2 * rank) & (ids < 2 * rank + 2)
            if not owned.any():
                assert torch.count_nonzero(partial) == 0
        # The explicit expert-interval constructor is the same expert-parallel plan.
        explicit, explicit_output, _ = prepare_candidate(
            shard_case(unsharded, layouts[1])
        )
        explicit.run()
        torch.testing.assert_close(explicit_output, partials[1], atol=1e-2, rtol=1e-2)
    report["routing"] = parallel_routing_histogram(
        ids, 8, SMALL_WAYS if mode == "expert_parallel" else 1
    )
    record_property("numerical_report", json.dumps(report))


@pytest.mark.parametrize("tokens,tile", [(129, 8), (129, 16), (129, 32), (600, 32)])
@pytest.mark.parametrize("mode", ["expert_parallel", "moe_tensor_parallel"])
def test_small_rank_sum_swap_prefill_tiles(tokens, tile, mode, record_property):
    """The T > 16 swap-AB path (moe_sort row groups of ``tile`` rows) sums to
    the unsharded result for every offered group width on both layouts."""
    _require_blackwell()
    unsharded = make_case(tokens=tokens, **SMALL)
    ids, weights = make_routing(tokens, 8, 2, 2, 2, "hot")
    unsharded = replace(unsharded, topk_ids=ids, topk_weights=weights)
    layouts = _layouts(mode, SMALL_WAYS)
    swap = dict(swapab_max_tokens=1024, swapab_tile_policy=((1 << 62, tile),))
    partials, refs = _run_ranks(unsharded, layouts, **swap)
    whole, whole_output, _ = prepare_candidate(unsharded)
    whole.run()
    reference = reference_moe(unsharded, modes=("ideal_fp64",))["ideal_fp64"]
    report = check_partial_sum(
        partials, refs, reference, {"unsharded_candidate": whole_output}
    )
    record_property("numerical_report", json.dumps(report))


def test_plan_rejects_weights_of_another_layout():
    _require_blackwell()
    mx = _mxfp4()
    layout = mx.Mxfp4MoEParallelLayout
    unsharded = make_case(tokens=16, **SMALL)
    tp_case = shard_case(unsharded, layout("moe_tensor_parallel", SMALL_WAYS, 1))
    whole_weights = prepare_cute_weights(unsharded)
    tp_weights = prepare_cute_weights(tp_case)
    output = torch.empty_like(unsharded.x, dtype=torch.bfloat16)
    for wrapper_layout, weights, case, expected in (
        (
            layout("moe_tensor_parallel", SMALL_WAYS, 1),
            whole_weights,
            unsharded,
            "moe_tensor_parallel",
        ),
        (
            layout("expert_parallel", SMALL_WAYS, 1),
            tp_weights,
            tp_case,
            "expert_parallel",
        ),
        (layout(), tp_weights, tp_case, "single"),
    ):
        wrapper = mx.CuteDslMxfp4MoEWrapper(
            8, 2, 256, 512, parallel_layout=wrapper_layout
        )
        workspace = torch.empty(
            wrapper.get_workspace_size(16), device="cuda", dtype=torch.uint8
        )
        with pytest.raises(ValueError, match=expected):
            wrapper.plan(
                case.x,
                case.x_scale,
                case.topk_ids,
                case.topk_weights,
                *weights,
                beta=unsharded.beta,
                linear_beta=unsharded.linear_beta,
                workspace=workspace,
                output=output,
            )


@pytest.mark.parametrize("tokens", list(range(1, 17)))
def test_tp_decode_graph_replay(tokens):
    _require_blackwell()
    layout = _layouts("moe_tensor_parallel", SMALL_WAYS)[2]
    rank_case = shard_case(make_case(tokens=tokens, **SMALL), layout)
    plan, output, _ = prepare_rank_candidate(rank_case, layout)
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
        eager = output.clone()
        for _ in range(10):
            graph.replay()
        torch.testing.assert_close(output, eager, atol=1e-2, rtol=1e-2)
        reference = reference_moe(rank_case, modes=("ideal_fp64",))["ideal_fp64"]
        assert torch.isfinite(output).all()
        floor = _norm(reference.to(torch.bfloat16).double() - reference)
        assert (
            _norm(output.double() - reference)
            <= _norm(eager.double() - reference) + floor
        )
    assert _compiled_counts() == compiled_before


def test_tp_concurrent_caller_streams():
    _require_blackwell()
    layouts = [_layouts("moe_tensor_parallel", SMALL_WAYS)[rank] for rank in (0, 3)]
    cases = [
        shard_case(make_case(tokens=16, seed=seed, **SMALL), layout)
        for seed, layout in zip((101, 202), layouts, strict=False)
    ]
    by_case = {id(case): layout for case, layout in zip(cases, layouts, strict=False)}

    def prepare(case, prepared_weights=None):
        return prepare_rank_candidate(
            case, by_case[id(case)], prepared_weights=prepared_weights
        )

    _check_concurrent_caller_streams(cases, repeats=20, noise_size=512, prepare=prepare)


# ---------------------------------------------------------------------------
# Full Kimi geometry (B300, FLASHINFER_KIMI_K3_FULL=1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kimi_unsharded():
    """Unsharded 896-expert canonical bank on the GPU; ranks slice it."""
    if os.getenv("FLASHINFER_KIMI_K3_FULL") != "1":
        pytest.skip("set FLASHINFER_KIMI_K3_FULL=1 for full Kimi geometry")
    _require_blackwell()
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("full Kimi acceptance runs require B300")
    case = make_case(
        tokens=2048,
        hidden=7168,
        intermediate=3072,
        num_experts=896,
        local_num_experts=896,
        local_expert_offset=0,
        top_k=16,
    )
    device = case.x.device
    # Keep one canonical bank on the device: expert-parallel shards are
    # views of it and tensor-parallel shards are shard-sized copies.
    return replace(
        case,
        w1=case.w1.to(device),
        w1_scale=case.w1_scale.to(device),
        w2=case.w2.to(device),
        w2_scale=case.w2_scale.to(device),
        beta=torch.linspace(2.5, 7.0, 896, device=device, dtype=torch.float32),
        linear_beta=torch.linspace(7.0, 25.0, 896, device=device, dtype=torch.float32),
    )


@pytest.fixture(scope="module")
def kimi_trt_unsharded(kimi_unsharded):
    return prepare_trt_weights(kimi_unsharded)


@pytest.fixture(scope="module")
def kimi_tp8_rank3(kimi_unsharded):
    layout = _layouts("moe_tensor_parallel", 8)[3]
    rank_case = shard_case(kimi_unsharded, layout)
    return (
        layout,
        rank_case,
        prepare_cute_weights(rank_case),
        prepare_trt_weights(rank_case),
    )


def _kimi_tokens(base, tokens, ids, weights):
    return replace(
        base,
        x=base.x[:tokens],
        x_scale=base.x_scale[:tokens],
        topk_ids=ids,
        topk_weights=weights,
    )


@pytest.mark.parametrize("tokens", [16, 512])
@pytest.mark.parametrize("distribution", ["balanced", "remote_dominated"])
def test_full_kimi_ep8_rank_sum(
    kimi_unsharded, kimi_trt_unsharded, tokens, distribution, record_property
):
    # remote_dominated is defined relative to rank 3 (experts 336..447).
    ids, weights = make_routing(tokens, 896, 16, 112, 336, distribution)
    unsharded = _kimi_tokens(kimi_unsharded, tokens, ids, weights)
    baseline, baseline_output = make_trt_baseline(unsharded, kimi_trt_unsharded)
    baseline()
    partials, refs = _run_ranks(unsharded, _layouts("expert_parallel", 8))
    reference = reference_moe(unsharded, modes=("ideal_fp64",))["ideal_fp64"]
    report = check_partial_sum(
        partials, refs, reference, {"unsharded_trtllm_gen": baseline_output}
    )
    report["routing"] = parallel_routing_histogram(ids, 896, 8)
    report["layout"] = "expert_parallel/8"
    record_property("numerical_report", json.dumps(report))
    if distribution == "remote_dominated":
        assert report["routing"]["remote_fraction_per_rank"][3] >= 0.9


@pytest.mark.parametrize("tokens", [16, 512])
@pytest.mark.parametrize("distribution", ["balanced", "hot"])
def test_full_kimi_tp8_shard_sum(
    kimi_unsharded, kimi_trt_unsharded, tokens, distribution, record_property
):
    ids, weights = make_routing(tokens, 896, 16, 896, 0, distribution)
    unsharded = _kimi_tokens(kimi_unsharded, tokens, ids, weights)
    baseline, baseline_output = make_trt_baseline(unsharded, kimi_trt_unsharded)
    baseline()
    partials, refs = _run_ranks(unsharded, _layouts("moe_tensor_parallel", 8))
    reference = reference_moe(unsharded, modes=("ideal_fp64",))["ideal_fp64"]
    report = check_partial_sum(
        partials, refs, reference, {"unsharded_trtllm_gen": baseline_output}
    )
    report["routing"] = parallel_routing_histogram(ids, 896, 1)
    report["layout"] = "moe_tensor_parallel/8"
    record_property("numerical_report", json.dumps(report))


@pytest.mark.parametrize("tokens", [1, 16, 128, 512, 2048])
@pytest.mark.parametrize("distribution", ["balanced", "empty", "hot"])
def test_full_kimi_tp8_paired_fp64(
    kimi_tp8_rank3, tokens, distribution, record_property
):
    layout, base, cute_weights, trt_weights = kimi_tp8_rank3
    ids, weights = make_routing(tokens, 896, 16, 896, 0, distribution)
    case = _kimi_tokens(base, tokens, ids, weights)
    assert case.intermediate_size == 384 and case.local_num_experts == 896
    plan, output, _ = prepare_rank_candidate(
        case, layout, prepared_weights=cute_weights
    )
    # TRT-LLM Gen evaluates the same 384-wide shard: intermediate_size=384,
    # 896 local experts at offset 0.
    baseline, baseline_output = make_trt_baseline(case, trt_weights)
    plan.run()
    baseline()
    refs = reference_moe(case)
    report = {
        name: paired_accuracy(output, baseline_output, ref)
        for name, ref in refs.items()
    }
    report["routing"] = routing_histogram(case)
    report["layout"] = "moe_tensor_parallel/8 rank 3"
    report["trtllm_gen_intermediate_size"] = case.intermediate_size
    record_property("numerical_report", json.dumps(report))
    assert report["ideal_fp64"]["candidate"]["finite"]
    assert report["ideal_fp64"]["baseline"]["finite"]
    ref = refs["ideal_fp64"]
    floor = _norm(ref.to(torch.bfloat16).double() - ref)
    assert (
        _norm(output.double() - ref) <= _norm(baseline_output.double() - ref) + floor
    ), report


@pytest.mark.parametrize("tokens", list(range(1, 17)))
def test_full_kimi_tp8_graph_replay(kimi_tp8_rank3, tokens):
    layout, base, cute_weights, _ = kimi_tp8_rank3
    ids, weights = make_routing(tokens, 896, 16, 896, 0, "hot")
    case = _kimi_tokens(base, tokens, ids, weights)
    plan, output, _ = prepare_rank_candidate(
        case, layout, prepared_weights=cute_weights
    )
    plan.run()
    eager = output.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(100):
        graph.replay()
    torch.testing.assert_close(output, eager, atol=1e-2, rtol=1e-2)
