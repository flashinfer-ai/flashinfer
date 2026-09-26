"""Native MXFP4 W4A8 SiTU evaluation. Minimum GPU architecture: SM100.

Run the full Kimi dimensions with FLASHINFER_KIMI_K3_FULL=1 on B300.
Set FLASHINFER_KIMI_K3_ALL_LOCAL=1 for the separate 896-local-expert cases.
Numerical reports compare both backends to the same FP64 oracle. Historical
atol/rtol fractions are diagnostics, not a customer-approved acceptance rule.
"""

from dataclasses import replace
import importlib
import json
import os

import pytest
import torch

from .mxfp4_situ_reference import (
    MXFP4SiTUCase,
    absolute_error_quantiles,
    decode_mxfp4,
    decode_ue8m0,
    error_metrics,
    make_case,
    make_routing,
    make_trt_baseline,
    pack_topk,
    paired_accuracy,
    prepare_cute_weights,
    prepare_trt_weights,
    reference_moe,
    routing_histogram,
    situ_reference,
)


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("native W4A8 SiTU currently supports SM100/SM103")


def _compiled_counts():
    return tuple(
        len(
            getattr(
                importlib.import_module("flashinfer.fused_moe.cute_dsl." + module),
                cache,
            )
        )
        for module, cache in (
            (
                "blockscaled_contiguous_gather_grouped_gemm_act_fusion",
                "_gather_kernel_cache",
            ),
            (
                "blockscaled_contiguous_grouped_gemm_finalize_fusion",
                "_finalize_kernel_cache",
            ),
            ("mxfp4_routing", "_route_preprocess_kernel_cache"),
        )
    )


def prepare_candidate(case, packed=False, prepared_weights=None, enable_pdl=False):
    from flashinfer.fused_moe.cute_dsl.mxfp4 import CuteDslMxfp4MoEWrapper

    wrapper = CuteDslMxfp4MoEWrapper(
        case.num_experts,
        case.topk_ids.shape[1],
        case.hidden_size,
        case.intermediate_size,
        num_local_experts=case.local_num_experts,
        local_expert_offset=case.local_expert_offset,
        enable_pdl=enable_pdl,
    )
    weights = (
        prepare_cute_weights(case) if prepared_weights is None else prepared_weights
    )
    output = torch.empty(case.x.shape, device=case.x.device, dtype=torch.bfloat16)
    workspace = torch.empty(
        wrapper.get_workspace_size(case.x.shape[0]),
        device=case.x.device,
        dtype=torch.uint8,
    )
    ids = pack_topk(case.topk_ids, case.topk_weights) if packed else case.topk_ids
    route_weights = None if packed else case.topk_weights
    plan = wrapper.plan(
        case.x,
        case.x_scale,
        ids,
        route_weights,
        *weights,
        beta=case.beta,
        linear_beta=case.linear_beta,
        workspace=workspace,
        output=output,
    )
    return plan, output, workspace


def test_error_quantiles_above_torch_limit():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    count = 2**24 + 2
    values = torch.arange(count - 1, -1, -1, device="cuda", dtype=torch.float64)
    expected = torch.tensor([0.5, 0.95, 0.99], device="cuda", dtype=torch.float64) * (
        count - 1
    )
    torch.testing.assert_close(
        absolute_error_quantiles(values), expected, atol=0, rtol=0
    )


def test_e2m1_and_ue8m0_encoding():
    codes = torch.arange(16, dtype=torch.uint8).repeat(2)
    packed = (codes[::2] | (codes[1::2] << 4)).reshape(1, 16)
    actual = decode_mxfp4(packed, torch.tensor([[127]], dtype=torch.uint8))
    expected = (
        torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float64,
        )
        .repeat(2)
        .reshape(1, 32)
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    scale = decode_ue8m0(torch.tensor([0, 126, 127, 128, 254, 255], dtype=torch.uint8))
    torch.testing.assert_close(
        scale[:5],
        torch.tensor([2.0**-127, 0.5, 1.0, 2.0, 2.0**127], dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    assert torch.isnan(scale[-1])


@pytest.mark.parametrize(
    "distribution", ["balanced", "empty", "hot", "all_remote", "remote_dominated"]
)
def test_routing_global_ids(distribution):
    ids, weights = make_routing(128, 896, 16, 112, 336, distribution, device="cpu")
    assert ids.min() >= 0 and ids.max() < 896
    assert all(row.unique().numel() == 16 for row in ids)
    assert weights.dtype == torch.bfloat16
    assert torch.all(weights > 0)
    local = (ids >= 336) & (ids < 448)
    if distribution == "all_remote":
        assert not local.any()
    if distribution == "remote_dominated":
        assert local.sum() == 128 and (local.sum(dim=1) == 1).all()
        assert ids[local].unique().numel() == 112
    if distribution == "hot":
        assert (ids == 336).sum() == 128
    if distribution == "empty":
        assert ids[local].unique().numel() <= 1


def _assert_decode_route_semantics(buffers, ids, local_experts, offset, tile):
    """Check the public mapping contract independently of assignment ordering."""
    ids = ids.cpu().flatten().tolist()
    groups = [
        [q for q, expert in enumerate(ids) if expert == offset + local]
        for local in range(local_experts)
    ]
    arrays = {
        name: tensor.cpu().flatten().tolist()
        for name, tensor in buffers.items()
        if name != "out_expert_counts"
    }
    tile_experts, tile_limits = [], []
    base = 0
    valid_rows = set()
    expanded = arrays["out_expanded_idx_to_permuted_idx"]
    permuted = arrays["out_permuted_idx_to_expanded_idx"]
    for local, assignments in enumerate(groups):
        count = len(assignments)
        num_tiles = (count + tile - 1) // tile
        tile_experts.extend([local] * num_tiles)
        tile_limits.extend(
            min(base + (j + 1) * tile, base + count) for j in range(num_tiles)
        )
        assert sorted(permuted[base : base + count]) == assignments
        for q in assignments:
            row = expanded[q]
            assert base <= row < base + count
            assert permuted[row] == q
            assert row not in valid_rows
            valid_rows.add(row)
        base += num_tiles * tile
    active = len(tile_experts)
    assert arrays["out_tile_idx_to_expert_idx"][:active] == tile_experts
    assert arrays["out_tile_idx_to_mn_limit"][:active] == tile_limits
    assert arrays["out_num_non_exiting_tiles"] == [active]
    assert arrays["out_total_num_padded_tokens"] == [base]
    for q, expert in enumerate(ids):
        if not offset <= expert < offset + local_experts:
            assert expanded[q] == -1


_DECODE_ROUTE_CASES = [
    (
        tokens,
        112,
        336,
        16,
        ("balanced", "hot", "all_remote", "random")[(tokens - 1) % 4],
    )
    for tokens in range(1, 17)
] + [
    (1, 112, 336, 16, "hot"),
    (1, 112, 336, 16, "all_remote"),
    (16, 112, 336, 16, "balanced"),
    (16, 112, 336, 16, "hot"),
    (16, 112, 336, 16, "all_remote"),
    (16, 896, 0, 16, "hot"),
    (16, 4, 336, 32, "hot"),
    (16, 4, 336, 32, "all_remote"),
]


@pytest.mark.parametrize(
    "tokens,local_experts,offset,top_k,distribution", _DECODE_ROUTE_CASES
)
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp32"])
def test_decode_fused_routing_semantics_and_graph(
    tokens, local_experts, offset, top_k, distribution, mode
):
    # Full routing geometry, without allocating any expert weight bank.
    _require_blackwell()
    import cuda.bindings.driver as cuda
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        allocate_moe_sort_buffers,
        moe_sort,
    )
    from flashinfer.fused_moe.cute_dsl.mxfp4_routing import _plan_route_preprocess

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids, weights = make_routing(
            tokens,
            896,
            top_k,
            local_experts,
            offset,
            "balanced" if distribution == "random" else distribution,
        )
        if distribution == "random":
            generator = torch.Generator().manual_seed(123 + tokens)
            ids.copy_(
                torch.stack(
                    [
                        torch.randperm(896, generator=generator)[:top_k]
                        for _ in range(tokens)
                    ]
                ).to(device=ids.device, dtype=torch.int32)
            )
        if mode == "fp32":
            weights = weights.float() + 0.0003
        source_ids = pack_topk(ids, weights) if mode == "packed" else ids.clone()
        source_weights = None if mode == "packed" else weights.clone()
        route_ids = torch.empty_like(ids) if mode == "packed" else source_ids
        route_weights = (
            source_weights
            if mode == "fp32"
            else torch.empty_like(weights, dtype=torch.float32)
        )
        buffers = allocate_moe_sort_buffers(tokens, 896, top_k, local_experts)
        native_buffers = allocate_moe_sort_buffers(tokens, 896, top_k, local_experts)
        output = torch.full((tokens, 7168), 7, dtype=torch.bfloat16, device="cuda")
        plan = _plan_route_preprocess(
            source_ids,
            source_weights,
            output=output,
            route_ids=route_ids,
            route_weights=route_weights,
            moe_sort_buffers=buffers,
            num_experts=896,
            num_local_experts=local_experts,
            local_expert_offset=offset,
        )

        def check(current_ids, current_weights):
            moe_sort(
                current_ids,
                current_weights.float(),
                896,
                top_k,
                num_local_experts=local_experts,
                local_expert_offset=offset,
                **native_buffers,
            )
            for mappings in (buffers, native_buffers):
                _assert_decode_route_semantics(
                    mappings, current_ids, local_experts, offset, 128
                )
            torch.testing.assert_close(route_ids, current_ids, atol=0, rtol=0)
            torch.testing.assert_close(
                route_weights, current_weights.float(), atol=0, rtol=0
            )
            assert output.count_nonzero() == 0

        check(ids, weights)
        compiled_before = _compiled_counts()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run(cuda.CUstream(stream.cuda_stream))
        changed_ids = (ids + 1) % 896
        changed_weights = weights.flip(1).contiguous()
        if mode == "packed":
            source_ids.copy_(pack_topk(changed_ids, changed_weights))
        else:
            source_ids.copy_(changed_ids)
            source_weights.copy_(changed_weights)
        # A poisoned output and changed input contents must be consumed by the
        # existing capture, without an intervening eager forward or replan.
        for _ in range(3):
            output.fill_(7)
            graph.replay()
        check(changed_ids, changed_weights)
        assert _compiled_counts() == compiled_before


@pytest.mark.parametrize("tokens", [1, 16])
def test_decode_pdl_graph_matches_standard(tokens):
    _require_blackwell()
    case = make_case(tokens=tokens, distribution="hot")
    prepared = prepare_cute_weights(case)
    ordinary, expected, _ = prepare_candidate(
        case, packed=True, prepared_weights=prepared
    )
    pdl, output, _ = prepare_candidate(
        case, packed=True, prepared_weights=prepared, enable_pdl=True
    )
    ordinary.run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        pdl.run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


def test_decode_tile256_graph_matches_default_fp64(record_property):
    _require_blackwell()
    from flashinfer.fused_moe.cute_dsl.mxfp4 import CuteDslMxfp4MoEWrapper

    case = make_case(tokens=16, distribution="hot")
    prepared = prepare_cute_weights(case)
    baseline, baseline_output, _ = prepare_candidate(case, prepared_weights=prepared)
    # Both GEMMs use the supported 2-CTA M=256 layout. Sorting must produce
    # matching 256-row offsets, even though no expert has more than 16 rows.
    tactic = (256, ((256, 128), (2, 1), False), ((256, 128), (2, 1), False))
    wrapper = CuteDslMxfp4MoEWrapper(
        case.num_experts,
        case.topk_ids.shape[1],
        case.hidden_size,
        case.intermediate_size,
        num_local_experts=case.local_num_experts,
        local_expert_offset=case.local_expert_offset,
        enable_pdl=False,
        offline_tactics={16: tactic},
    )
    workspace = torch.empty(
        wrapper.get_workspace_size(16), device=case.x.device, dtype=torch.uint8
    )
    output = torch.empty_like(case.x, dtype=torch.bfloat16)
    plan = wrapper.plan(
        case.x,
        case.x_scale,
        case.topk_ids,
        case.topk_weights,
        *prepared,
        beta=case.beta,
        linear_beta=case.linear_beta,
        workspace=workspace,
        output=output,
    )
    compiled_before = _compiled_counts()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    torch.cuda.current_stream().wait_stream(stream)

    reports = {}
    previous_output = None
    for phase in ("initial", "changed_inputs"):
        if phase == "changed_inputs":
            case.x.copy_((-case.x.float()).to(case.x.dtype))
            case.topk_ids.copy_((case.topk_ids + 3) % case.num_experts)
            case.topk_weights.copy_(case.topk_weights.flip(1))
        # No eager candidate call after mutation: replay must read the new
        # activations and rebuild routes at the existing workspace addresses.
        for _ in range(3):
            graph.replay()
        baseline.run()
        ref = reference_moe(case, modes=("ideal_fp64",))["ideal_fp64"]
        report = paired_accuracy(output, baseline_output, ref)
        reports[phase] = report
        assert report["candidate"]["finite"] and report["baseline"]["finite"]
        floor = torch.linalg.vector_norm(ref.to(torch.bfloat16).double() - ref)
        candidate_error = torch.linalg.vector_norm(output.double() - ref)
        baseline_error = torch.linalg.vector_norm(baseline_output.double() - ref)
        # Keep the suite's paired FP64 non-regression criterion; the default
        # validated tactic is the same-input numerical anchor for this case.
        assert candidate_error <= baseline_error + floor, report
        if previous_output is not None:
            assert not torch.equal(output, previous_output), (
                "captured inputs were ignored"
            )
        previous_output = output.clone()
    assert _compiled_counts() == compiled_before
    record_property("numerical_report", json.dumps(reports))


@pytest.mark.parametrize("linear_beta", [None, 7.0])
def test_fp64_oracle_sparse_exact_case(linear_beta):
    # One active input, one active gate/up pair, and one output channel.
    x = torch.zeros((1, 32), dtype=torch.float32)
    x[0, 0] = 1
    w1 = torch.zeros((1, 64, 16), dtype=torch.uint8)
    w1[0, 0, 0] = 2  # up = 1
    w1[0, 32, 0] = 4  # gate = 2
    w2 = torch.zeros((1, 32, 16), dtype=torch.uint8)
    w2[0, 0, 0] = 2
    case = MXFP4SiTUCase(
        x.to(torch.float8_e4m3fn),
        torch.full((1, 1), 127, dtype=torch.uint8),
        w1,
        torch.full((1, 64, 1), 127, dtype=torch.uint8),
        w2,
        torch.full((1, 32, 1), 127, dtype=torch.uint8),
        torch.tensor([[5]], dtype=torch.int32),
        torch.tensor([[0.5]], dtype=torch.bfloat16),
        torch.tensor([2.5]),
        None if linear_beta is None else torch.tensor([linear_beta]),
        8,
        5,
    )
    out = reference_moe(case, modes=("ideal_fp64",))["ideal_fp64"]
    expected = torch.zeros_like(out)
    expected[0, 0] = (
        situ_reference(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
            2.5,
            linear_beta,
        )
        * 0.5
    )
    torch.testing.assert_close(out, expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("tokens", [1, 16, 128])
@pytest.mark.parametrize("distribution", ["balanced", "empty", "hot"])
def test_situ_paired_fp64_report(tokens, distribution, record_property):
    _require_blackwell()
    case = make_case(tokens=tokens, distribution=distribution)
    plan, output, _ = prepare_candidate(case)
    baseline, baseline_output = make_trt_baseline(case)
    plan.run()
    baseline()
    references = reference_moe(case)
    report = {
        name: paired_accuracy(output, baseline_output, ref)
        for name, ref in references.items()
    }
    report["routing"] = routing_histogram(case)
    record_property("numerical_report", json.dumps(report))
    assert report["ideal_fp64"]["candidate"]["finite"]
    assert report["ideal_fp64"]["baseline"]["finite"]
    # Development non-regression gate, not a claim of agreed customer tolerance:
    # allow one BF16 representation-error floor beyond the baseline L2 error.
    ref = references["ideal_fp64"]
    floor = torch.linalg.vector_norm(ref.to(torch.bfloat16).double() - ref)
    candidate_error = torch.linalg.vector_norm(output.double() - ref)
    baseline_error = torch.linalg.vector_norm(baseline_output.double() - ref)
    assert candidate_error <= baseline_error + floor, report


@pytest.mark.parametrize("tokens", list(range(1, 17)))
def test_decode_graph_replay(tokens):
    _require_blackwell()
    case = make_case(tokens=tokens)
    plan, output, _ = prepare_candidate(case)
    plan.run()
    compiled_before = _compiled_counts()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    torch.cuda.current_stream().wait_stream(stream)
    for distribution, beta in (("hot", 2.5), ("all_remote", 4.0), ("balanced", 7.0)):
        ids, weights = make_routing(tokens, 8, 2, 4, 2, distribution)
        case.topk_ids.copy_(ids)
        case.topk_weights.copy_(weights)
        case.beta.fill_(beta)
        plan.run()
        eager = output.clone()
        for _ in range(10):
            graph.replay()
        torch.testing.assert_close(output, eager, atol=1e-2, rtol=1e-2)
        if distribution == "all_remote":
            assert torch.count_nonzero(output) == 0
    assert _compiled_counts() == compiled_before


def test_packed_routing_matches_separate():
    _require_blackwell()
    case = make_case(tokens=16, distribution="hot")
    separate, output, _ = prepare_candidate(case)
    packed, output_packed, _ = prepare_candidate(case, packed=True)
    separate.run()
    packed.run()
    torch.testing.assert_close(output, output_packed, atol=1e-2, rtol=1e-2)


def test_native_weight_preparation_is_lossless():
    _require_blackwell()
    case = make_case(tokens=1)
    w1, sf1, w2, sf2 = prepare_cute_weights(case)
    e, rows, half_k = w1.shape
    canonical_w1 = (
        w1.reshape(e, rows // 128, 2, 64, half_k).transpose(1, 2).reshape_as(case.w1)
    )
    torch.testing.assert_close(canonical_w1.cpu(), case.w1, atol=0, rtol=0)
    torch.testing.assert_close(w2.cpu(), case.w2, atol=0, rtol=0)
    # MMA axes are (outer_m, inner_m, m_tile, inner_k, k_tile, expert).
    linear1 = sf1.permute(5, 2, 1, 0, 4, 3).contiguous().reshape_as(case.w1_scale)
    canonical_sf1 = (
        linear1.reshape(e, rows // 128, 2, 64, half_k // 16)
        .transpose(1, 2)
        .reshape_as(case.w1_scale)
    )
    linear2 = sf2.permute(5, 2, 1, 0, 4, 3).contiguous().reshape_as(case.w2_scale)
    torch.testing.assert_close(
        canonical_sf1.cpu().view(torch.uint8), case.w1_scale, atol=0, rtol=0
    )
    torch.testing.assert_close(
        linear2.cpu().view(torch.uint8), case.w2_scale, atol=0, rtol=0
    )


@pytest.mark.parametrize("linear_beta", [None, 7.0, 25.0])
def test_situ_sparse_kernel_matches_quantized_oracle(linear_beta):
    _require_blackwell()
    case = make_case(tokens=2, top_k=1, beta=2.5, linear_beta=linear_beta)
    case.w1.zero_()
    case.w2.zero_()
    case.w1_scale.fill_(127)
    case.w2_scale.fill_(127)
    case.w1[:, 0, 0] = 2
    case.w1[:, case.intermediate_size, 0] = 4
    case.w2[:, 0, 0] = 2
    source = torch.zeros(case.x.shape, device=case.x.device)
    source[:, 0] = 1.0
    case.x.copy_(source.to(case.x.dtype))
    case.x_scale.fill_(127)
    case.topk_ids.fill_(case.local_expert_offset)
    case.topk_weights.fill_(0.5)
    plan, output, _ = prepare_candidate(case)
    plan.run()
    reference = reference_moe(case, modes=("mxfp8_fp32",))["mxfp8_fp32"]
    torch.testing.assert_close(output.double(), reference, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("tokens", [127, 128, 129])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_partial_m_tile_matches_quantized_oracle(tokens, linear_beta):
    """Check the last valid row on either side of a 128-row CTA boundary."""
    _require_blackwell()
    case = make_case(
        tokens=tokens,
        hidden=256,
        intermediate=128,
        num_experts=1,
        local_num_experts=1,
        local_expert_offset=0,
        top_k=1,
        beta=2.5,
        linear_beta=linear_beta,
    )
    # Exactly represented sparse projections isolate the activation and row
    # mapping from GEMM accumulation error, as in the existing sparse test.
    case.w1.zero_()
    case.w2.zero_()
    case.w1_scale.fill_(127)
    case.w2_scale.fill_(127)
    case.w1[:, 0, 0] = 6  # E2M1 up = 4
    case.w1[:, case.intermediate_size, 0] = 7  # E2M1 gate = 6
    case.w2[:, 0, 0] = 2  # down projection = 1
    source = torch.zeros(case.x.shape, device=case.x.device)
    source[:, 0] = (
        torch.arange(tokens, device=case.x.device, dtype=torch.float32) % 4 + 1
    ) * 0.5
    case.x.copy_(source.to(case.x.dtype))
    case.x_scale.fill_(127)
    case.topk_ids.zero_()
    case.topk_weights.fill_(1.0)
    plan, output, _ = prepare_candidate(case)
    plan.run()
    reference = reference_moe(case, modes=("mxfp8_fp32",))["mxfp8_fp32"]
    torch.testing.assert_close(output.double(), reference, atol=1e-3, rtol=1e-2)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    torch.cuda.current_stream().wait_stream(stream)
    # A graph replay must retain the same row predicate while reading new
    # contents of the bound runtime parameter buffers.
    case.beta.fill_(5.0)
    if case.linear_beta is not None:
        case.linear_beta.fill_(7.0)
    for _ in range(3):
        graph.replay()
    reference = reference_moe(case, modes=("mxfp8_fp32",))["mxfp8_fp32"]
    torch.testing.assert_close(output.double(), reference, atol=1e-3, rtol=1e-2)


def _dual_tile_expectations(case, tile=128, alt=256, threshold_permille=1150):
    """Host model of the routing kernel's dual-tile rule for ``case``."""
    ids = case.topk_ids.to("cpu")
    local = (ids - case.local_expert_offset).clamp(min=-1)
    local = local[(local >= 0) & (local < case.local_num_experts)]
    counts = torch.bincount(local, minlength=case.local_num_experts)
    pad = lambda n: int((-(-counts // n) * n).sum())
    use_alt = pad(alt) * 1000 <= pad(tile) * threshold_permille
    rows = pad(alt) if use_alt else pad(tile)
    return {
        "use_alt": use_alt,
        "rows": rows,
        "base_tiles": rows // tile,
        "alt_tiles": rows // alt if use_alt else 0,
        "base_active": 0 if use_alt else rows // tile,
    }


def _set_routing(case, hot):
    """Deterministic routings: round-robin pairs (every local expert gets
    2 * tokens / 8 rows) or two experts holding every route."""
    tokens, top_k = case.topk_ids.shape
    assert top_k == 2 and case.local_num_experts == 8 and case.local_expert_offset == 0
    t = torch.arange(tokens, device=case.topk_ids.device, dtype=torch.int32)
    if hot:
        ids = torch.stack([torch.zeros_like(t), torch.ones_like(t)], dim=1)
    else:
        ids = torch.stack([t % 8, (t + 4) % 8], dim=1)
    case.topk_ids.copy_(ids)
    case.topk_weights.fill_(0.5)


def _make_dual_tile_wrapper(case, dense_dual_tile):
    from flashinfer.fused_moe.cute_dsl.mxfp4 import CuteDslMxfp4MoEWrapper

    return CuteDslMxfp4MoEWrapper(
        case.num_experts,
        case.topk_ids.shape[1],
        case.hidden_size,
        case.intermediate_size,
        num_local_experts=case.local_num_experts,
        local_expert_offset=case.local_expert_offset,
        swapab_max_tokens=0,  # dense path at every token count
        dense_dual_tile=dense_dual_tile,
    )


def _plan_dense(wrapper, case, weights):
    output = torch.empty(case.x.shape, device=case.x.device, dtype=torch.bfloat16)
    workspace = torch.empty(
        wrapper.get_workspace_size(case.x.shape[0]),
        device=case.x.device,
        dtype=torch.uint8,
    )
    plan = wrapper.plan(
        case.x,
        case.x_scale,
        case.topk_ids,
        case.topk_weights,
        *weights,
        beta=case.beta,
        linear_beta=case.linear_beta,
        workspace=workspace,
        output=output,
    )
    return plan, output


def _dual_tile_observed(plan):
    buffers = plan._kwargs["moe_sort_buffers"]
    return {
        "rows": int(buffers["out_total_num_padded_tokens"].item()),
        "base_tiles": int(buffers["out_num_non_exiting_tiles"].item()),
        "alt_tiles": int(buffers["out_alt_num_non_exiting_tiles"].item()),
        "base_active": int(buffers["out_base_active_num_non_exiting_tiles"].item()),
    }


def _assert_dual_tile_matches_single(case, dual_output, single_output, references):
    """Development gate of the dual-tile routing against the single-tile plan on
    the same routing: at most one BF16 representation-error floor beyond it."""
    ref = references["ideal_fp64"]
    assert torch.isfinite(dual_output).all()
    floor = torch.linalg.vector_norm(ref.to(torch.bfloat16).double() - ref)
    dual_error = torch.linalg.vector_norm(dual_output.double() - ref)
    single_error = torch.linalg.vector_norm(single_output.double() - ref)
    assert dual_error <= single_error + floor, (float(dual_error), float(single_error))


def test_dense_dual_tile_routing_follows_rows_per_expert():
    """The routing kernel pads to the 256-row tile where that pads no more rows
    than the 128-row tile (146-row experts) and to 128 where it would (73-row
    and 584-row experts); the GEMMs of the unchosen tile exit on the zero count
    and the output matches the single-tile plan. The choice is made on the
    device, so a CUDA-graph replay with a different routing flips it."""
    _require_blackwell()
    # 8 local experts, top_k 2: 584 tokens balanced = 146 rows per expert.
    case = make_case(
        tokens=584,
        hidden=256,
        intermediate=256,
        num_experts=8,
        local_num_experts=8,
        local_expert_offset=0,
        top_k=2,
    )
    _set_routing(case, hot=False)
    weights = prepare_cute_weights(case)
    dual_plan, dual_output = _plan_dense(
        _make_dual_tile_wrapper(case, True), case, weights
    )
    single_plan, single_output = _plan_dense(
        _make_dual_tile_wrapper(case, False), case, weights
    )
    dual_plan.run()
    single_plan.run()
    torch.cuda.synchronize()
    expected = _dual_tile_expectations(case)
    assert expected["use_alt"], expected
    observed = _dual_tile_observed(dual_plan)
    assert observed == {k: expected[k] for k in observed}, (observed, expected)
    _assert_dual_tile_matches_single(
        case, dual_output, single_output, reference_moe(case)
    )

    # Graph replay with the hot routing (two experts hold every route, 584 rows
    # each: 128-tile padding 1280 rows against 1536 for the 256 tile) flips the
    # device-side choice without re-planning.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        dual_plan.run()
    torch.cuda.current_stream().wait_stream(stream)
    _set_routing(case, hot=True)
    graph.replay()
    single_plan.run()
    torch.cuda.synchronize()
    expected = _dual_tile_expectations(case)
    assert not expected["use_alt"], expected
    observed = _dual_tile_observed(dual_plan)
    assert observed == {k: expected[k] for k in observed}, (observed, expected)
    _assert_dual_tile_matches_single(
        case, dual_output, single_output, reference_moe(case)
    )


@pytest.mark.parametrize("dense_dual_tile", [False, True])
def test_dense_gemm1_zero_fill_matches_memset(monkeypatch, dense_dual_tile):
    """With MXFP4_DENSE_FILL_IN_GEMM1 the gather GEMM1's epilogue warps zero the
    finalize output (dynamic chunks, counters reset by the last warp) and the
    memset launch disappears; the output matches the memset plan on repeated
    runs and graph replays (a stale counter or a missed chunk would leave the
    previous run's values in the reduce-add target), on the dual-tile chain
    (the unchosen variant leaves the fill to the chosen one) and when no route
    is local (both variants have zero tiles: the output must still be zero)."""
    _require_blackwell()
    from flashinfer.fused_moe.cute_dsl import mxfp4

    case = make_case(
        tokens=584,
        hidden=256,
        intermediate=256,
        num_experts=16,
        local_num_experts=8,
        local_expert_offset=0,
        top_k=2,
    )
    _set_routing(case, hot=False)
    weights = prepare_cute_weights(case)
    monkeypatch.setattr(mxfp4, "DENSE_FILL_IN_GEMM1", "0")
    plain_plan, plain_output = _plan_dense(
        _make_dual_tile_wrapper(case, dense_dual_tile), case, weights
    )
    monkeypatch.setattr(mxfp4, "DENSE_FILL_IN_GEMM1", "1")
    fill_plan, fill_output = _plan_dense(
        _make_dual_tile_wrapper(case, dense_dual_tile), case, weights
    )
    assert fill_plan._memset is None and fill_plan._aux_stream is None
    assert plain_plan._memset is not None
    counters = fill_plan._kwargs["zero_fill_counters"]
    for _ in range(3):
        fill_plan.run()
        plain_plan.run()
    torch.cuda.synchronize()
    assert counters.tolist() == [0, 0]
    _assert_dual_tile_matches_single(
        case, fill_output, plain_output, reference_moe(case)
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fill_plan.run()
    torch.cuda.current_stream().wait_stream(stream)
    for hot in (True, False, True):
        _set_routing(case, hot=hot)
        graph.replay()
        graph.replay()
        plain_plan.run()
        torch.cuda.synchronize()
        assert counters.tolist() == [0, 0]
        _assert_dual_tile_matches_single(
            case, fill_output, plain_output, reference_moe(case)
        )
    # Nothing local: every route goes to experts 8..15, both tile variants see
    # zero tiles and the output (previous values) must come back as zeros.
    tokens = case.topk_ids.shape[0]
    t = torch.arange(tokens, device=case.topk_ids.device, dtype=torch.int32)
    case.topk_ids.copy_(torch.stack([8 + t % 8, 8 + (t + 3) % 8], dim=1))
    graph.replay()
    torch.cuda.synchronize()
    assert counters.tolist() == [0, 0]
    assert not fill_output.any()


def test_dense_fill_in_gemm1_policy(monkeypatch):
    """``ep`` (expert-parallel ranks only) / ``1`` / ``0`` vocabulary of the
    in-GEMM1 zero-fill, mirroring DENSE_ASYNC_MEMSET."""
    from flashinfer.fused_moe.cute_dsl import mxfp4

    monkeypatch.setattr(mxfp4, "DENSE_FILL_IN_GEMM1", "ep")
    assert mxfp4._dense_fill_in_gemm1(896, 112)
    assert not mxfp4._dense_fill_in_gemm1(896, 896)
    monkeypatch.setattr(mxfp4, "DENSE_FILL_IN_GEMM1", "1")
    assert mxfp4._dense_fill_in_gemm1(896, 896)
    monkeypatch.setattr(mxfp4, "DENSE_FILL_IN_GEMM1", "0")
    assert not mxfp4._dense_fill_in_gemm1(896, 112)


@pytest.mark.parametrize("rows_per_expert", [73, 293])
def test_dense_dual_tile_keeps_128_where_256_pads(rows_per_expert):
    """73- and 293-row experts pad 2x / 1.33x under the 256 tile: the routing
    keeps the 128 tile and the alternate GEMMs run on an empty list."""
    _require_blackwell()
    case = make_case(
        tokens=rows_per_expert * 4,
        hidden=256,
        intermediate=256,
        num_experts=8,
        local_num_experts=8,
        local_expert_offset=0,
        top_k=2,
    )
    _set_routing(case, hot=False)
    weights = prepare_cute_weights(case)
    dual_plan, dual_output = _plan_dense(
        _make_dual_tile_wrapper(case, True), case, weights
    )
    single_plan, single_output = _plan_dense(
        _make_dual_tile_wrapper(case, False), case, weights
    )
    dual_plan.run()
    single_plan.run()
    torch.cuda.synchronize()
    expected = _dual_tile_expectations(case)
    assert not expected["use_alt"], expected
    observed = _dual_tile_observed(dual_plan)
    assert observed == {k: expected[k] for k in observed}, (observed, expected)
    _assert_dual_tile_matches_single(
        case, dual_output, single_output, reference_moe(case)
    )


@pytest.mark.parametrize("tokens", [16, 1025])
def test_run_uses_caller_output_without_torch_storage_allocation(tokens):
    _require_blackwell()
    case = make_case(tokens=tokens)
    plan, output, _ = prepare_candidate(case)
    plan.run()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        returned = plan.run()
        assert returned.data_ptr() == output.data_ptr()
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == allocated
    # This detects transient/cached PyTorch tensor-storage allocations, not
    # direct driver allocations or synchronization; those need runtime tracing.


def test_scalar_and_per_expert_runtime_betas():
    _require_blackwell()
    case = make_case(tokens=16, distribution="hot", beta=2.5, linear_beta=7.0)
    scalar_case = replace(
        case, beta=case.beta[:1].clone(), linear_beta=case.linear_beta[:1].clone()
    )
    per_expert, output, _ = prepare_candidate(case)
    scalar, scalar_output, _ = prepare_candidate(scalar_case)
    compiled_before = _compiled_counts()
    for beta, linear_beta in ((2.5, 7.0), (4.0, 25.0), (7.0, 3.0)):
        case.beta.fill_(beta)
        scalar_case.beta.fill_(beta)
        case.linear_beta.fill_(linear_beta)
        scalar_case.linear_beta.fill_(linear_beta)
        per_expert.run()
        scalar.run()
        torch.testing.assert_close(output, scalar_output, atol=1e-2, rtol=1e-2)
    assert _compiled_counts() == compiled_before


@pytest.mark.parametrize("with_linear_beta", [False, True])
def test_distinct_expert_situ_parameters_in_graph(with_linear_beta):
    """Use local expert parameters after global-ID translation on every replay."""
    _require_blackwell()
    case = make_case(
        tokens=8,
        top_k=1,
        num_experts=8,
        local_num_experts=4,
        local_expert_offset=3,
        linear_beta=7.0 if with_linear_beta else None,
    )
    # All experts have identical, exactly represented projections. Differences
    # between their outputs must therefore come from their SiTU parameters.
    case.w1.zero_()
    case.w2.zero_()
    case.w1_scale.fill_(127)
    case.w2_scale.fill_(127)
    case.w1[:, 0, 0] = 6  # E2M1 up = 4
    case.w1[:, case.intermediate_size, 0] = 7  # E2M1 gate = 6
    case.w2[:, 0, 0] = 2  # down projection = 1
    source = torch.zeros(case.x.shape, device=case.x.device)
    source[:, 0] = 1.0
    case.x.copy_(source.to(case.x.dtype))
    case.x_scale.fill_(127)
    # Visit every local expert twice, in an order different from local IDs.
    case.topk_ids.copy_(
        torch.tensor(
            [[6], [3], [5], [4], [3], [6], [4], [5]],
            device=case.x.device,
            dtype=torch.int32,
        )
    )
    case.topk_weights.fill_(1.0)
    initial_beta = torch.tensor(
        [0.75, 1.25, 2.5, 5.0], device=case.x.device, dtype=torch.float32
    )
    case.beta.copy_(initial_beta)
    if with_linear_beta:
        initial_linear_beta = torch.tensor(
            [1.0, 3.0, 7.0, 13.0], device=case.x.device, dtype=torch.float32
        )
        case.linear_beta.copy_(initial_linear_beta)

    plan, output, _ = prepare_candidate(case)
    plan.run()
    compiled_before = _compiled_counts()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    torch.cuda.current_stream().wait_stream(stream)

    previous_output = None
    phases = ("initial", "gate", "up") if with_linear_beta else ("initial", "gate")
    for phase in phases:
        if phase == "gate":
            case.beta.copy_(initial_beta.flip(0))
        elif phase == "up":
            case.linear_beta.copy_(initial_linear_beta.flip(0))
        # No eager forward intervenes after parameter mutation: the captured
        # graph must read the updated contents of the original device buffers.
        for _ in range(5):
            graph.replay()
        reference = reference_moe(case, modes=("mxfp8_fp32",))["mxfp8_fp32"]
        torch.testing.assert_close(output.double(), reference, atol=1e-3, rtol=1e-2)
        if previous_output is not None:
            assert not torch.equal(output, previous_output), (
                f"captured SiTU {phase} parameters were ignored"
            )
        previous_output = output.clone()
    assert _compiled_counts() == compiled_before


@pytest.mark.parametrize(
    "input_distribution", ["zero", "small", "outliers", "saturated"]
)
@pytest.mark.parametrize("linear_beta", [None, 7.0, 25.0])
def test_situ_runtime_parameters(input_distribution, linear_beta, record_property):
    _require_blackwell()
    case = make_case(
        tokens=16,
        distribution="hot",
        input_distribution=input_distribution,
        beta=2.5,
        linear_beta=linear_beta,
    )
    plan, output, _ = prepare_candidate(case)
    plan.run()
    first = output.clone()
    ref = reference_moe(case)
    report = {name: error_metrics(output, value) for name, value in ref.items()}
    record_property("numerical_report", json.dumps(report))
    assert report["ideal_fp64"]["finite"]
    if input_distribution == "zero":
        assert torch.count_nonzero(output) == 0
    elif input_distribution in ("outliers", "saturated"):
        case.beta.fill_(7.0)
        if case.linear_beta is not None:
            case.linear_beta.fill_(3.0)
        plan.run()
        assert not torch.equal(output, first), "runtime SiTU parameters were ignored"


def test_concurrent_caller_streams():
    _require_blackwell()
    cases = [make_case(tokens=16, seed=seed) for seed in (101, 202)]
    _check_concurrent_caller_streams(cases, repeats=20, noise_size=512)


def _check_concurrent_caller_streams(
    cases, *, repeats, noise_size, prepared_weights=None, prepare=None
):
    prepare = prepare_candidate if prepare is None else prepare
    prepared = [prepare(case, prepared_weights=prepared_weights) for case in cases]
    streams = [torch.cuda.Stream() for _ in range(len(cases) + 1)]
    expected = []
    for plan, output, _ in prepared:
        plan.run()
        expected.append(output.clone())
    source = torch.randn((noise_size, noise_size), device="cuda")
    target = torch.empty_like(source)
    gemm_out = torch.empty_like(source)
    checks = torch.empty((len(cases), repeats), dtype=torch.bool, device="cuda")
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for iteration in range(repeats):
        for index, (stream, (plan, output, _)) in enumerate(
            zip(streams, prepared, strict=False)
        ):
            with torch.cuda.stream(stream):
                plan.run()
                # Preserve each iteration's verdict without synchronizing the
                # host or retaining a full output snapshot per iteration.
                torch.all(
                    torch.isclose(output, expected[index], atol=1e-2, rtol=1e-2),
                    out=checks[index, iteration],
                )
        with torch.cuda.stream(streams[-1]):
            target.copy_(source)
            torch.mm(source, target, out=gemm_out)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert checks.all(), "at least one concurrent forward differed from eager output"
    for (_, output, _), eager in zip(prepared, expected, strict=True):
        torch.testing.assert_close(output, eager, atol=1e-2, rtol=1e-2)


@pytest.fixture(scope="module")
def full_kimi_case():
    if os.getenv("FLASHINFER_KIMI_K3_FULL") != "1":
        pytest.skip("set FLASHINFER_KIMI_K3_FULL=1 for full Kimi geometry")
    _require_blackwell()
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("full Kimi acceptance runs require B300")
    return make_case(
        tokens=2048,
        hidden=7168,
        intermediate=3072,
        num_experts=896,
        local_num_experts=112,
        local_expert_offset=336,
        top_k=16,
    )


@pytest.mark.parametrize("tokens", [1, 16, 128, 512, 2048])
@pytest.mark.parametrize(
    "distribution", ["balanced", "empty", "hot", "remote_dominated"]
)
def test_full_kimi_paired_fp64(full_kimi_case, tokens, distribution, record_property):
    base = full_kimi_case
    ids, weights = make_routing(tokens, 896, 16, 112, 336, distribution)
    case = replace(
        base,
        x=base.x[:tokens],
        x_scale=base.x_scale[:tokens],
        topk_ids=ids,
        topk_weights=weights,
    )
    plan, output, _ = prepare_candidate(case)
    baseline, baseline_output = make_trt_baseline(case)
    plan.run()
    baseline()
    refs = reference_moe(case)
    report = {
        name: paired_accuracy(output, baseline_output, ref)
        for name, ref in refs.items()
    }
    report["routing"] = routing_histogram(case)
    record_property("numerical_report", json.dumps(report))
    assert report["ideal_fp64"]["candidate"]["finite"]
    assert report["ideal_fp64"]["baseline"]["finite"]
    ref = refs["ideal_fp64"]
    floor = torch.linalg.vector_norm(ref.to(torch.bfloat16).double() - ref)
    assert (
        torch.linalg.vector_norm(output.double() - ref)
        <= torch.linalg.vector_norm(baseline_output.double() - ref) + floor
    ), report


@pytest.mark.parametrize("tokens", list(range(1, 17)))
def test_full_kimi_graph_replay(full_kimi_case, tokens):
    base = full_kimi_case
    ids, weights = make_routing(tokens, 896, 16, 112, 336, "hot")
    case = replace(
        base,
        x=base.x[:tokens],
        x_scale=base.x_scale[:tokens],
        topk_ids=ids,
        topk_weights=weights,
    )
    plan, output, _ = prepare_candidate(case)
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


def test_full_kimi_concurrent_caller_streams(full_kimi_case):
    base = full_kimi_case
    cases = []
    for tokens, distribution, beta in (
        (16, "hot", 2.5),
        (128, "empty", 4.0),
        (2048, "balanced", 7.0),
    ):
        ids, weights = make_routing(tokens, 896, 16, 112, 336, distribution)
        cases.append(
            replace(
                base,
                x=base.x[:tokens],
                x_scale=base.x_scale[:tokens],
                topk_ids=ids,
                topk_weights=weights,
                beta=torch.full_like(base.beta, beta),
            )
        )
    _check_concurrent_caller_streams(
        cases,
        repeats=100,
        noise_size=2048,
        prepared_weights=prepare_cute_weights(base),
    )


@pytest.fixture(scope="module")
def all_local_kimi_case_and_weights():
    if os.getenv("FLASHINFER_KIMI_K3_ALL_LOCAL") != "1":
        pytest.skip("set FLASHINFER_KIMI_K3_ALL_LOCAL=1 for 896 local experts")
    _require_blackwell()
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("all-local Kimi acceptance runs require B300")

    # Generate only the 16 nonzero experts, then place their distinct native
    # weights throughout the full bank, including its last expert. The unused
    # experts still occupy real contiguous storage and contain valid zeros.
    active_experts = [
        0,
        1,
        31,
        32,
        63,
        64,
        111,
        112,
        255,
        256,
        447,
        448,
        511,
        512,
        894,
        895,
    ]
    source = make_case(
        tokens=16,
        hidden=7168,
        intermediate=3072,
        num_experts=896,
        local_num_experts=16,
        local_expert_offset=0,
        top_k=16,
        distribution="empty",
    )
    banks = []
    for tensor, fill in (
        (source.w1, 0),
        (source.w1_scale, 127),
        (source.w2, 0),
        (source.w2_scale, 127),
    ):
        bank = torch.full(
            (896, *tensor.shape[1:]),
            fill,
            dtype=torch.uint8,
            device=source.x.device,
        )
        for index, expert in enumerate(active_experts):
            bank[expert].copy_(tensor[index])
        banks.append(bank)

    active = torch.tensor(active_experts, device=source.x.device, dtype=torch.int32)
    slots = torch.arange(16, device=source.x.device)
    ids = active[(slots[:, None] + slots[None, :]) % 16].contiguous()
    case = replace(
        source,
        w1=banks[0],
        w1_scale=banks[1],
        w2=banks[2],
        w2_scale=banks[3],
        topk_ids=ids,
        beta=torch.linspace(2.5, 7.0, 896, device=source.x.device, dtype=torch.float32),
        linear_beta=torch.linspace(
            7.0, 25.0, 896, device=source.x.device, dtype=torch.float32
        ),
    )
    # Keep the canonical bank on the GPU to avoid a 29 GiB CPU bank. The
    # reference decodes only routed experts, one at a time, into FP64.
    candidate_weights = prepare_cute_weights(case)
    baseline_weights = prepare_trt_weights(case)
    return case, candidate_weights, baseline_weights


@pytest.mark.parametrize("tokens", [1, 16])
def test_all_local_kimi_paired_fp64_and_graph(
    all_local_kimi_case_and_weights, tokens, record_property
):
    base, candidate_weights, baseline_weights = all_local_kimi_case_and_weights
    case = replace(
        base,
        x=base.x[:tokens],
        x_scale=base.x_scale[:tokens],
        topk_ids=base.topk_ids[:tokens],
        topk_weights=base.topk_weights[:tokens],
    )
    assert case.local_num_experts == case.num_experts == 896
    assert case.local_expert_offset == 0
    plan, output, workspace = prepare_candidate(
        case, packed=True, prepared_weights=candidate_weights
    )
    baseline, baseline_output = make_trt_baseline(
        case, prepared_weights=baseline_weights, packed=True
    )
    plan.run()
    baseline()
    eager = output.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(20):
        graph.replay()
    torch.testing.assert_close(output, eager, atol=1e-2, rtol=1e-2)

    refs = reference_moe(case)
    report = {
        name: paired_accuracy(output, baseline_output, reference)
        for name, reference in refs.items()
    }
    report["routing"] = routing_histogram(case)
    report["workspace_bytes"] = workspace.numel()
    report["routing_format"] = "packed"
    report["graph_replays"] = 20
    record_property("numerical_report", json.dumps(report))
    assert report["routing"]["empty_local_experts"] == 880
    assert report["routing"]["local_assignments"] == tokens * 16
    assert report["routing"]["local"][895] == tokens
    assert report["ideal_fp64"]["candidate"]["finite"]
    assert report["ideal_fp64"]["baseline"]["finite"]
    # Preserve the existing development gate; no separate all-local tolerance.
    ref = refs["ideal_fp64"]
    floor = torch.linalg.vector_norm(ref.to(torch.bfloat16).double() - ref)
    candidate_error = torch.linalg.vector_norm(output.double() - ref)
    baseline_error = torch.linalg.vector_norm(baseline_output.double() - ref)
    assert candidate_error <= baseline_error + floor, report


@pytest.mark.parametrize("order", [(17, 16, 17), (16, 17, 16)])
def test_decode_plan_cache_transition_graph(order, record_property):
    _require_blackwell()
    finalize = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl."
        "blockscaled_contiguous_grouped_gemm_finalize_fusion"
    )
    routing = importlib.import_module("flashinfer.fused_moe.cute_dsl.mxfp4_routing")
    # Test both first-compilation orders. Keep every plan alive within an order.
    finalize._finalize_kernel_cache.clear()
    routing._route_preprocess_kernel_cache.clear()
    stream = torch.cuda.Stream()
    reports = []
    with torch.no_grad(), torch.cuda.stream(stream):
        base = make_case(
            tokens=17,
            hidden=512,
            intermediate=128,
            num_experts=8,
            local_num_experts=4,
            local_expert_offset=2,
            top_k=2,
            seed=1917,
            distribution="hot",
        )
        cute_weights = prepare_cute_weights(base)
        trt_weights = prepare_trt_weights(base)
        plans = []
        for tokens in order:
            ids, weights = make_routing(tokens, 8, 2, 4, 2, "hot", seed=1917)
            case = replace(
                base,
                x=base.x[:tokens].clone(),
                x_scale=base.x_scale[:tokens].clone(),
                topk_ids=ids,
                topk_weights=weights,
                beta=torch.linspace(2.5, 7.0, 4, device="cuda"),
                linear_beta=torch.linspace(3.0, 25.0, 4, device="cuda"),
            )
            plan, output, workspace = prepare_candidate(
                case, packed=True, prepared_weights=cute_weights
            )
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan.run()
            plans.append((case, plan, output, workspace, graph))
        for index, (case, plan, output, _workspace, graph) in enumerate(plans):
            for distribution in ("hot", "all_remote", "balanced"):
                ids, weights = make_routing(
                    case.x.shape[0], 8, 2, 4, 2, distribution, seed=7401
                )
                case.topk_ids.copy_(ids)
                case.topk_weights.copy_(weights)
                case.x.copy_((-case.x.float()).to(case.x.dtype))
                case.beta.copy_(case.beta.flip(0) * 0.75 + 0.25)
                case.linear_beta.copy_(case.linear_beta.flip(0) * 0.5 + 0.5)
                plan._topk_ids.copy_(pack_topk(ids, weights))
                plan._kwargs["gemm1_out"].view(torch.uint8).fill_(0x7F)
                plan._kwargs["gemm1_out_scale"].view(torch.uint8).fill_(0xFF)
                output.fill_(float("nan"))
                graph.replay()
                baseline, baseline_output = make_trt_baseline(case, trt_weights)
                baseline()
                stream.synchronize()
                references = reference_moe(case)
                metrics = {
                    name: paired_accuracy(output, baseline_output, reference)
                    for name, reference in references.items()
                }
                for report in metrics.values():
                    assert report["candidate"]["finite"]
                    assert report["baseline"]["finite"]
                ideal = references["ideal_fp64"]
                floor = torch.linalg.vector_norm(
                    ideal.to(torch.bfloat16).double() - ideal
                )
                error = torch.linalg.vector_norm(output.double() - ideal)
                baseline_error = torch.linalg.vector_norm(
                    baseline_output.double() - ideal
                )
                assert error <= baseline_error + floor
                reports.append(
                    {
                        "plan_index": index,
                        "tokens": case.x.shape[0],
                        "distribution": distribution,
                        "metrics": metrics,
                    }
                )
        stream.synchronize()
    record_property("numerical_report", json.dumps(reports))


def test_decode_routing_opt_in_fallback_graph():
    _require_blackwell()
    import cuda.bindings.driver as cuda
    from flashinfer.fused_moe.cute_dsl.moe_utils import allocate_moe_sort_buffers
    from flashinfer.fused_moe.cute_dsl import mxfp4_routing as routing

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for order in ((128, 8), (8, 128)):
            routing._route_preprocess_kernel_cache.clear()
            ids, weights = make_routing(16, 8, 2, 4, 2, "hot", seed=1917)
            source_ids = pack_topk(ids, weights)
            plans = []
            for tile, opt_in in [(tile, True) for tile in order] + [(8, False)]:
                buffers = allocate_moe_sort_buffers(16, 8, 2, 4, tile)
                output = torch.full((16, 512), 7, device="cuda", dtype=torch.bfloat16)
                route_ids = torch.empty_like(ids)
                route_weights = torch.empty_like(weights, dtype=torch.float32)
                plan = routing._plan_route_preprocess(
                    source_ids,
                    None,
                    output=output,
                    route_ids=route_ids,
                    route_weights=route_weights,
                    moe_sort_buffers=buffers,
                    num_experts=8,
                    num_local_experts=4,
                    local_expert_offset=2,
                    tile_size=tile,
                    _single_tile_per_expert=opt_in,
                )
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    plan.run(cuda.CUstream(stream.cuda_stream))
                plans.append(
                    (tile, buffers, output, route_ids, route_weights, plan, graph)
                )
            for distribution in ("hot", "all_remote", "balanced"):
                ids, weights = make_routing(16, 8, 2, 4, 2, distribution, seed=7401)
                source_ids.copy_(pack_topk(ids, weights))
                for (
                    tile,
                    buffers,
                    output,
                    route_ids,
                    route_weights,
                    _plan,
                    graph,
                ) in plans:
                    for tensor in buffers.values():
                        tensor.fill_(-97)
                    output.fill_(7)
                    route_ids.fill_(-97)
                    route_weights.fill_(float("nan"))
                    graph.replay()
                    stream.synchronize()
                    _assert_decode_route_semantics(buffers, ids, 4, 2, tile)
                    torch.testing.assert_close(route_ids, ids, atol=0, rtol=0)
                    torch.testing.assert_close(
                        route_weights, weights.float(), atol=0, rtol=0
                    )
                    assert output.count_nonzero() == 0
        stream.synchronize()


@pytest.mark.parametrize(
    "tokens,local_experts,offset,distribution,narrow_tile,permille",
    [
        (128, 896, 0, "hot", 8, 0),
        (128, 896, 0, "hot", 8, 500),
        (128, 896, 0, "balanced", 8, 500),
        (128, 896, 0, "empty", 16, 500),
        (256, 896, 0, "balanced", 16, 0),
        (256, 896, 0, "empty", 16, 500),
        (256, 896, 0, "hot", 32, 500),
        (200, 112, 336, "balanced", 32, 0),
        (256, 112, 336, "empty", 8, 500),
    ],
)
def test_fused_routing_dispatch_lists_match_dispatch_kernel(
    tokens, local_experts, offset, distribution, narrow_tile, permille
):
    """The fused routing kernel's wide / narrow / all-sub-tile lists equal the
    ``swapab_dispatch`` lists built from the same sort groups, on the first
    run and after a graph replay over changed routing."""
    _require_blackwell()
    import cuda.bindings.driver as cuda
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        allocate_moe_sort_buffers,
        get_max_num_tiles,
    )
    from flashinfer.fused_moe.cute_dsl.mxfp4_routing import _plan_route_preprocess
    from flashinfer.fused_moe.cute_dsl.swapab_moe import swapab_dispatch

    group_rows, top_k = 128, 16
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids, weights = make_routing(
            tokens, 896, top_k, local_experts, offset, distribution
        )
        source_ids = pack_topk(ids, weights)
        route_ids = torch.empty_like(ids)
        route_weights = torch.empty_like(weights, dtype=torch.float32)
        buffers = allocate_moe_sort_buffers(
            tokens, 896, top_k, local_experts, group_rows
        )
        output = torch.zeros((tokens, 512), dtype=torch.bfloat16, device="cuda")
        tiles = get_max_num_tiles(tokens, top_k, local_experts, group_rows)
        sub = group_rows // narrow_tile

        def lists():
            return dict(
                wide_list=torch.full((tiles,), -7, dtype=torch.int32, device="cuda"),
                wide_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
                narrow_list=torch.full(
                    (tiles * sub,), -7, dtype=torch.int32, device="cuda"
                ),
                narrow_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
                all_list=torch.full(
                    (tiles * sub,), -7, dtype=torch.int32, device="cuda"
                ),
                all_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
            )

        fused, reference = lists(), lists()
        plan = _plan_route_preprocess(
            source_ids,
            None,
            output=output,
            route_ids=route_ids,
            route_weights=route_weights,
            moe_sort_buffers=buffers,
            num_experts=896,
            num_local_experts=local_experts,
            local_expert_offset=offset,
            tile_size=group_rows,
            _single_tile_per_expert=group_rows >= tokens,
            dispatch_lists=dict(
                fused,
                narrow_tile=narrow_tile,
                wide_min_rows=64,
                wide_min_permille=permille,
            ),
        )

        def check(expect_wide):
            swapab_dispatch(
                tile_idx_to_mn_limit=buffers["out_tile_idx_to_mn_limit"],
                num_non_exiting_tiles=buffers["out_num_non_exiting_tiles"],
                group_rows=group_rows,
                narrow_tile=narrow_tile,
                wide_min_rows=64,
                wide_min_permille=permille,
                **reference,
            )
            stream.synchronize()
            for name in ("wide", "narrow", "all"):
                count = int(reference[name + "_count"].item())
                assert int(fused[name + "_count"].item()) == count, name
                torch.testing.assert_close(
                    fused[name + "_list"][:count],
                    reference[name + "_list"][:count],
                    atol=0,
                    rtol=0,
                )
            total = int(buffers["out_num_non_exiting_tiles"].item())
            assert int(fused["wide_count"].item()) <= total
            if expect_wide:
                # Without the rule the full 128-row groups are wide.
                assert int(fused["wide_count"].item()) > 0

        check(expect_wide=permille == 0 and distribution in ("hot", "empty"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run(cuda.CUstream(stream.cuda_stream))
        changed_ids, changed_weights = make_routing(
            tokens,
            896,
            top_k,
            local_experts,
            offset,
            "balanced" if distribution != "balanced" else "hot",
            seed=4242,
        )
        source_ids.copy_(pack_topk(changed_ids, changed_weights))
        for buf in (fused, reference):
            for name in ("wide", "narrow", "all"):
                buf[name + "_list"].fill_(-7)
        graph.replay()
        # The replayed routing is balanced (or hot) over 896 experts: only
        # the equality with the dispatch kernel is asserted.
        check(expect_wide=False)


def _policy_wrapper(*, ep_size=1, ep_rank=0, moe_tp_size=1, moe_tp_rank=0):
    from flashinfer.fused_moe.cute_dsl.mxfp4 import (
        CuteDslMxfp4MoEWrapper,
        Mxfp4MoEParallelLayout,
    )

    layout = Mxfp4MoEParallelLayout.from_sizes(
        ep_size=ep_size,
        ep_rank=ep_rank,
        moe_tp_size=moe_tp_size,
        moe_tp_rank=moe_tp_rank,
    )
    # Kimi K3 geometry: 896 experts, top_k 16, hidden 7168, intermediate 3072.
    return CuteDslMxfp4MoEWrapper(896, 16, 7168, 3072, parallel_layout=layout)


def test_dense_dual_tile_default_follows_the_shard(monkeypatch):
    """The unchosen tile's two launches cost about 8 us, which the wide rank's
    empty routings (86-160 us at T=8192..32768) cannot absorb: the default rule
    applies to the MoE-TP shard (384 columns) only; ``dense_dual_tile=True``
    forces it, ``False`` disables it."""
    from flashinfer.fused_moe.cute_dsl import mxfp4

    monkeypatch.setattr(mxfp4, "DENSE_DUAL_TILE", True)
    monkeypatch.setattr(mxfp4, "DENSE_DUAL_TILE_MAX_SHARD", 512)
    tp8 = _policy_wrapper(moe_tp_size=8, moe_tp_rank=0)
    ep8 = _policy_wrapper(ep_size=8, ep_rank=3)
    assert tp8.intermediate_shard == 384 and ep8.intermediate_shard == 3072
    assert tp8._dual_enabled(8192) and not tp8._dual_enabled(4096)
    assert not ep8._dual_enabled(8192) and not ep8._dual_enabled(32768)
    monkeypatch.setattr(mxfp4, "DENSE_DUAL_TILE_MAX_SHARD", 4096)
    assert ep8._dual_enabled(8192)
    from flashinfer.fused_moe.cute_dsl.mxfp4 import (
        CuteDslMxfp4MoEWrapper,
        Mxfp4MoEParallelLayout,
    )

    layout = Mxfp4MoEParallelLayout.from_sizes(ep_size=8, ep_rank=3)
    assert CuteDslMxfp4MoEWrapper(
        896, 16, 7168, 3072, parallel_layout=layout, dense_dual_tile=True
    )._dual_enabled(1024)
    assert not CuteDslMxfp4MoEWrapper(
        896, 16, 7168, 3072, parallel_layout=layout, dense_dual_tile=False
    )._dual_enabled(32768)


def test_dense_gemm2_raster_policy(monkeypatch):
    """Dense GEMM2 raster: ``auto`` (default) hands the shard's GEMM2 both
    rasters from DENSE_GEMM2_RASTER_M_MIN_TOKENS tokens (device-side choice),
    ``1`` forces M-fastest with the swizzle when it divides the N tile count
    (7168 / 256 = 28 tiles), ``0`` keeps N-fastest; the wide expert-parallel
    rank never rasters along M."""
    from flashinfer.fused_moe.cute_dsl import mxfp4

    tp8 = _policy_wrapper(moe_tp_size=8, moe_tp_rank=0)
    ep8 = _policy_wrapper(ep_size=8, ep_rank=3)
    monkeypatch.setattr(mxfp4, "DENSE_GEMM2_SWIZZLE", 4)
    monkeypatch.setattr(mxfp4, "DENSE_GEMM2_RASTER_M", "0")
    assert tp8._gemm2_raster(32768, 256) == (False, 1)
    monkeypatch.setattr(mxfp4, "DENSE_GEMM2_RASTER_M", "1")
    assert tp8._gemm2_raster(128, 256) == (True, 4)
    assert ep8._gemm2_raster(8192, 256) == (True, 4)
    monkeypatch.setattr(mxfp4, "DENSE_GEMM2_SWIZZLE", 3)
    assert tp8._gemm2_raster(8192, 256) == (True, 1)  # 28 N tiles: no group of 3
    monkeypatch.setattr(mxfp4, "DENSE_GEMM2_SWIZZLE", 4)
    monkeypatch.setattr(mxfp4, "DENSE_GEMM2_RASTER_M", "auto")
    assert tp8._gemm2_raster(16384, 256) == ("auto", 4)
    assert tp8._gemm2_raster(32768, 256) == ("auto", 4)
    assert tp8._gemm2_raster(8192, 256) == (False, 1)
    assert ep8._gemm2_raster(32768, 256) == (False, 1)
    monkeypatch.setattr(mxfp4, "DENSE_GEMM2_RASTER_M_MIN_TOKENS", 8192)
    monkeypatch.setattr(mxfp4, "DENSE_GEMM2_SWIZZLE", 7)
    assert tp8._gemm2_raster(8192, 256) == ("auto", 7)


def test_swap_split_policy(monkeypatch):
    """Split form (policy-tile groups plus 128-row groups for the experts
    above SWAP_SPLIT_MIN_ROWS rows): fused-routing token counts from
    SWAP_SPLIT_MIN_TOKENS up, finalize only, MoE-TP shard by default; the
    workspace carries the wide slot arrays and a row capacity that covers
    any split of the routes into narrow and wide experts."""
    from flashinfer.fused_moe.cute_dsl import mxfp4
    from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_tiles

    monkeypatch.setattr(mxfp4, "SWAP_SPLIT", True)
    monkeypatch.setattr(mxfp4, "SWAP_SPLIT_EP", False)
    monkeypatch.setattr(mxfp4, "SWAP_MIXED", False)
    monkeypatch.setattr(mxfp4, "SWAP_SPLIT_MIN_ROWS", 64)
    tp8 = _policy_wrapper(moe_tp_size=8, moe_tp_rank=0)
    ep8 = _policy_wrapper(ep_size=8, ep_rank=3)
    assert tp8._swap_split(128) and tp8._swap_split(256) and tp8._swap_split(512)
    assert tp8._swap_split(1024)
    assert not tp8._swap_split(64)  # below SWAP_SPLIT_MIN_TOKENS
    assert not tp8._swap_split(2048)  # above SWAP_SPLIT_MAX_TOKENS (hybrid range)
    assert not tp8._swap_split(128, do_finalize=False)
    assert not ep8._swap_split(128)
    # The shard takes the 16384-route fused routing only for the split form.
    assert tp8._fused_route_cap() == mxfp4.FUSED_ROUTE_MAX_ROUTES
    assert tp8._fused_route_cap(1024) == mxfp4.FUSED_ROUTE_MAX_ROUTES_LARGE
    assert tp8._fused_route_cap(1024, do_finalize=False) == mxfp4.FUSED_ROUTE_MAX_ROUTES
    assert ep8._fused_route_cap() == mxfp4.FUSED_ROUTE_MAX_ROUTES_LARGE
    monkeypatch.setattr(mxfp4, "SWAP_SPLIT_EP", True)
    assert ep8._swap_split(128)
    for wrapper, local in ((tp8, 896), (ep8, 112)):
        for tokens in (128, 256, 512, 1024):
            tile = wrapper._swap_tile(tokens)
            groups, rows = wrapper._swap_split_capacity(tokens)
            narrow = get_max_num_tiles(tokens, 16, local, tile) * tile
            assert rows % 128 == 0 and rows >= -(-narrow // 128) * 128 + groups * 128
            # Worst case for the wide region: every wide expert holds
            # SWAP_SPLIT_MIN_ROWS + 1 rows (one padded 128-row group each).
            wide_experts = min(local, tokens * 16 // 65)
            assert groups >= wide_experts
            fields = {f.name: f.shape for f in wrapper._workspace_fields(tokens)[0]}
            assert fields["swap_wide_list"] == (rows // 128,)
            assert fields["swap_wide_expert"] == (rows // 128,)
            assert fields["swap_wide_limit"] == (rows // 128,)
            assert fields["out_permuted_idx_to_expanded_idx"] == (rows,)
            assert fields["gemm1_out"][0] == rows
    monkeypatch.setattr(mxfp4, "SWAP_SPLIT", False)
    names = {f.name for f in tp8._workspace_fields(128)[0]}
    assert "swap_wide_list" not in names
    assert not tp8._swap_split(128)


def _split_layout_reference(
    ids, local_experts, offset, tile, wide_tile, wide_min_rows, permille=0
):
    """Host model of the split layout: narrow experts (1..wide_min_rows rows)
    in ``tile``-row groups by local index, then the wide experts in
    ``wide_tile``-row groups from the next ``wide_tile`` multiple; no wide
    experts unless they hold ``permille`` of the local rows."""
    ids = ids.cpu().flatten().tolist()
    groups = [
        [q for q, expert in enumerate(ids) if expert == offset + local]
        for local in range(local_experts)
    ]
    total = sum(len(g) for g in groups)
    wide_rows = sum(len(g) for g in groups if len(g) > wide_min_rows)
    if wide_rows * 1000 < total * permille:
        wide_min_rows = total  # nothing is wide
    tile_experts, tile_limits, bases, wide_slots, wide_list = [], [], {}, {}, []
    base = 0
    for local, assignments in enumerate(groups):
        count = len(assignments)
        if not 0 < count <= wide_min_rows:
            continue
        num_tiles = -(-count // tile)
        bases[local] = base
        tile_experts.extend([local] * num_tiles)
        tile_limits.extend(
            min(base + (j + 1) * tile, base + count) for j in range(num_tiles)
        )
        base += num_tiles * tile
    narrow_tiles = len(tile_experts)
    base = -(-base // wide_tile) * wide_tile
    for local, assignments in enumerate(groups):
        count = len(assignments)
        if count <= wide_min_rows:
            continue
        num_tiles = -(-count // wide_tile)
        bases[local] = base
        for j in range(num_tiles):
            slot = base // wide_tile + j
            wide_slots[slot] = (local, min(base + (j + 1) * wide_tile, base + count))
            wide_list.append(slot)
        base += num_tiles * wide_tile
    return dict(
        groups=groups,
        tile_experts=tile_experts,
        tile_limits=tile_limits,
        narrow_tiles=narrow_tiles,
        bases=bases,
        wide_slots=wide_slots,
        wide_list=wide_list,
        padded=base,
    )


def _assert_split_layout(buffers, split, ids, local_experts, offset, tile):
    ref = _split_layout_reference(
        ids, local_experts, offset, tile, 128, 64, split["wide_min_permille"]
    )
    ints = ("wide_tile", "wide_min_rows", "wide_min_permille", "rows_capacity")
    arrays = {
        name: tensor.cpu().flatten().tolist()
        for name, tensor in list(buffers.items()) + list(split.items())
        if name != "out_expert_counts" and name not in ints
    }
    narrow = ref["narrow_tiles"]
    assert arrays["out_num_non_exiting_tiles"] == [narrow]
    assert arrays["out_tile_idx_to_expert_idx"][:narrow] == ref["tile_experts"]
    assert arrays["out_tile_idx_to_mn_limit"][:narrow] == ref["tile_limits"]
    assert arrays["out_total_num_padded_tokens"] == [ref["padded"]]
    assert arrays["wide_count"] == [len(ref["wide_list"])]
    assert arrays["wide_list"][: len(ref["wide_list"])] == ref["wide_list"]
    for slot, (local, limit) in ref["wide_slots"].items():
        assert arrays["wide_expert"][slot] == local, slot
        assert arrays["wide_limit"][slot] == limit, slot
    expanded = arrays["out_expanded_idx_to_permuted_idx"]
    permuted = arrays["out_permuted_idx_to_expanded_idx"]
    seen = set()
    for local, assignments in enumerate(ref["groups"]):
        if not assignments:
            continue
        base = ref["bases"][local]
        assert sorted(permuted[base : base + len(assignments)]) == assignments
        for q in assignments:
            row = expanded[q]
            assert base <= row < base + len(assignments)
            assert permuted[row] == q and row not in seen
            seen.add(row)
    for q, expert in enumerate(ids.cpu().flatten().tolist()):
        if not offset <= expert < offset + local_experts:
            assert expanded[q] == -1


@pytest.mark.parametrize("tokens", [128, 256])
@pytest.mark.parametrize("local_experts,offset", [(896, 0), (112, 336)])
@pytest.mark.parametrize("distribution", ["balanced", "hot", "empty"])
@pytest.mark.parametrize("permille", [0, 250])
def test_fused_routing_split_layout(
    tokens, local_experts, offset, distribution, permille
):
    """The fused routing kernel's split layout matches the host model on the
    first run and after a graph replay over changed routing (wide experts
    appearing and disappearing)."""
    _require_blackwell()
    import cuda.bindings.driver as cuda
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        allocate_moe_sort_buffers,
        get_max_num_tiles,
    )
    from flashinfer.fused_moe.cute_dsl.mxfp4_routing import _plan_route_preprocess

    tile, top_k = 16, 16
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids, weights = make_routing(
            tokens, 896, top_k, local_experts, offset, distribution
        )
        source_ids = pack_topk(ids, weights)
        route_ids = torch.empty_like(ids)
        route_weights = torch.empty_like(weights, dtype=torch.float32)
        buffers = allocate_moe_sort_buffers(tokens, 896, top_k, local_experts, tile)
        narrow_rows = get_max_num_tiles(tokens, top_k, local_experts, tile) * tile
        wide_groups = get_max_num_tiles(tokens, top_k, local_experts, 128)
        rows = -(-narrow_rows // 128) * 128 + wide_groups * 128
        buffers["out_permuted_idx_to_expanded_idx"] = torch.full(
            (rows,), -7, dtype=torch.int32, device="cuda"
        )
        split = dict(
            wide_expert=torch.full(
                (rows // 128,), -7, dtype=torch.int32, device="cuda"
            ),
            wide_limit=torch.full((rows // 128,), -7, dtype=torch.int32, device="cuda"),
            wide_list=torch.full((wide_groups,), -7, dtype=torch.int32, device="cuda"),
            wide_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
            wide_tile=128,
            wide_min_rows=64,
            wide_min_permille=permille,
            rows_capacity=rows,
        )
        output = torch.zeros((tokens, 512), dtype=torch.bfloat16, device="cuda")
        plan = _plan_route_preprocess(
            source_ids,
            None,
            output=output,
            route_ids=route_ids,
            route_weights=route_weights,
            moe_sort_buffers=buffers,
            num_experts=896,
            num_local_experts=local_experts,
            local_expert_offset=offset,
            tile_size=tile,
            split_layout=split,
        )
        stream.synchronize()
        _assert_split_layout(buffers, split, ids, local_experts, offset, tile)
        if local_experts == 896 and distribution == "empty":
            assert int(split["wide_count"].item()) > 0  # 16 experts x T rows
        if local_experts == 896 and distribution == "hot":
            # One expert with T of the 16 T local rows: wide only without the rule.
            assert (int(split["wide_count"].item()) > 0) == (permille == 0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run(cuda.CUstream(stream.cuda_stream))
        for changed in ("balanced", "hot", "empty"):
            if changed == distribution:
                continue
            changed_ids, changed_weights = make_routing(
                tokens, 896, top_k, local_experts, offset, changed
            )
            source_ids.copy_(pack_topk(changed_ids, changed_weights))
            for name in ("wide_expert", "wide_limit", "wide_list"):
                split[name].fill_(-7)
            graph.replay()
            stream.synchronize()
            _assert_split_layout(
                buffers, split, changed_ids, local_experts, offset, tile
            )


def test_finalize_rows_accumulate_and_skip_wide():
    """``accumulate`` adds the route-weighted narrow rows to the output rows
    (which hold the wide GEMM2's reduce-adds) and ``skip_wide`` leaves out
    the slots whose permuted row lies in the wide region, whose start the
    kernel derives from the narrow group count at run time."""
    _require_blackwell()
    import cuda.bindings.driver as cuda

    from flashinfer.fused_moe.cute_dsl.mxfp4_finalize import plan_finalize_rows

    torch.manual_seed(0)
    tokens, top_k, hidden = 37, 16, 256
    narrow_tile, wide_tile = 16, 128
    narrow_groups = 5  # 80 narrow rows -> the wide region starts at row 128
    rows_total = wide_tile + 2 * wide_tile
    rows = torch.randn(rows_total, hidden, device="cuda").to(torch.bfloat16)
    perm = torch.randint(
        0, rows_total, (tokens, top_k), device="cuda", dtype=torch.int32
    )
    perm[:, 3] = -1  # one slot per token that is not rank-local
    weights = torch.rand(tokens, top_k, device="cuda", dtype=torch.float32)
    prior = torch.randn(tokens, hidden, device="cuda").to(torch.bfloat16)
    out = prior.clone()
    count = torch.tensor([narrow_groups], device="cuda", dtype=torch.int32)
    wide_count = torch.tensor([2], device="cuda", dtype=torch.int32)
    plan = plan_finalize_rows(
        rows,
        perm,
        weights,
        out,
        accumulate=True,
        skip_wide=(count, wide_count, narrow_tile, wide_tile),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def expected(wide_row0, base):
        mask = ((perm >= 0) & (perm < wide_row0)).float()
        gathered = rows[perm.clamp(min=0).long()].float()
        return base.float() + (gathered * (weights * mask).unsqueeze(-1)).sum(1)

    out.copy_(prior)
    plan.run(stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(), expected(wide_tile, prior), atol=2e-2, rtol=1e-2
    )
    # A wider narrow region (device-side count) includes more slots without
    # re-planning, as a captured graph must; with no wide groups the output
    # rows are taken as zero (they were zero-filled) instead of read.
    count.fill_(rows_total // narrow_tile)
    wide_count.zero_()
    out.copy_(prior)
    plan.run(stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(),
        expected(rows_total, torch.zeros_like(prior)),
        atol=2e-2,
        rtol=1e-2,
    )
    # No narrow groups: the kernel leaves the (complete) output untouched.
    count.zero_()
    wide_count.fill_(3)
    out.copy_(prior)
    plan.run(stream)
    torch.cuda.synchronize()
    assert torch.equal(out, prior)


@pytest.mark.parametrize("tokens", [128, 192, 256])
@pytest.mark.parametrize("distribution", ["hot", "empty", "balanced"])
@pytest.mark.parametrize("dense_gemm2", [True, False], ids=["dense2", "swap2"])
@pytest.mark.parametrize("shard", ["tp", "ep"])
def test_swap_split_form_matches_default(
    monkeypatch, tokens, distribution, dense_gemm2, shard
):
    """The split form (wide experts through the 128-row launches: dense
    gather GEMM1, then either the dense finalize-fusion GEMM2 reduce-adding
    into the output with an accumulating finalize, or the 128-row swap GEMM2)
    matches the default swap form and stays within the BF16 floor of the
    FP64 reference; a captured graph follows routing changes that move
    experts between the narrow and the wide layout."""
    _require_blackwell()
    from flashinfer.fused_moe.cute_dsl import mxfp4

    monkeypatch.setattr(mxfp4, "SWAP_SPLIT", True)
    monkeypatch.setattr(mxfp4, "SWAP_MIXED", False)
    monkeypatch.setattr(mxfp4, "SWAP_SPLIT_MIN_ROWS", 64)
    monkeypatch.setattr(mxfp4, "SWAP_SPLIT_DENSE_GEMM2", dense_gemm2)
    monkeypatch.setattr(mxfp4, "SWAP_SPLIT_EP", shard == "ep")
    # The wide expert-parallel shard (intermediate > SWAP_TWO_STAGE_MAX_SHARD)
    # keeps the fused atomic finalize: no finalize kernel, the dense wide
    # GEMM2 reduce-adds next to the narrow swap GEMM2.
    kwargs = {"intermediate": 1024} if shard == "ep" else {}
    case = make_case(tokens=tokens, distribution=distribution, **kwargs)
    prepared = prepare_cute_weights(case)
    split, output, _ = prepare_candidate(case, prepared_weights=prepared)
    assert split.split
    assert split.split_dense is dense_gemm2
    assert split.two_stage is (shard == "tp")
    monkeypatch.setattr(mxfp4, "SWAP_SPLIT", False)
    default, expected, _ = prepare_candidate(case, prepared_weights=prepared)
    assert not default.split
    split.run()
    default.run()
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
    ref = reference_moe(case)["ideal_fp64"]
    floor = torch.linalg.vector_norm(ref.to(torch.bfloat16).double() - ref)
    assert torch.linalg.vector_norm(output.double() - ref) <= (
        torch.linalg.vector_norm(expected.double() - ref) + floor
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        split.run()
    torch.cuda.current_stream().wait_stream(stream)
    for changed in ("hot", "empty", "balanced"):
        if changed == distribution:
            continue
        ids, weights = make_routing(
            tokens,
            case.num_experts,
            case.topk_ids.shape[1],
            case.local_num_experts,
            case.local_expert_offset,
            changed,
        )
        case.topk_ids.copy_(ids)
        case.topk_weights.copy_(weights)
        default.run()
        for _ in range(5):
            graph.replay()
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
