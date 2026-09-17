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
        )
    )


def prepare_candidate(case, packed=False, prepared_weights=None):
    from flashinfer.fused_moe.cute_dsl.mxfp4 import CuteDslMxfp4MoEWrapper

    wrapper = CuteDslMxfp4MoEWrapper(
        case.num_experts,
        case.topk_ids.shape[1],
        case.hidden_size,
        case.intermediate_size,
        num_local_experts=case.local_num_experts,
        local_expert_offset=case.local_expert_offset,
        enable_pdl=False,
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


@pytest.mark.parametrize("distribution", ["balanced", "empty", "hot", "all_remote"])
def test_routing_global_ids(distribution):
    ids, weights = make_routing(128, 896, 16, 112, 336, distribution, device="cpu")
    assert ids.min() >= 0 and ids.max() < 896
    assert all(row.unique().numel() == 16 for row in ids)
    assert weights.dtype == torch.bfloat16
    assert torch.all(weights > 0)
    local = (ids >= 336) & (ids < 448)
    if distribution == "all_remote":
        assert not local.any()
    if distribution == "hot":
        assert (ids == 336).sum() == 128
    if distribution == "empty":
        assert ids[local].unique().numel() <= 1


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
    cases, *, repeats, noise_size, prepared_weights=None
):
    prepared = [
        prepare_candidate(case, prepared_weights=prepared_weights) for case in cases
    ]
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
@pytest.mark.parametrize("distribution", ["balanced", "empty", "hot"])
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
