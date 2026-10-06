"""Production standard-MoE coverage for the CuTe 4.6 migration corpus.

These tests deliberately enter through the public planned/bound serving API.
They use synthetic checkpoint tensors, a pure-Torch GPU oracle, fixed scratch,
and live-input CUDA-graph replay.  Together they force the production compile
IDs for direct micro, dynamic prefill, and both tiny-decode phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from b12x.moe._shared.kernels.reference import (
    compare_to_reference,
    moe_reference_nvfp4,
    moe_reference_w4a8_mx,
)

from b12x.testing.reference.helpers import (
    prepare_tp_moe_fp4_experts,
    require_b12x,
    swizzle_block_scale_reference,
)


_E = 4
_K = 512
_N = 128
_TOPK = 2


@dataclass(frozen=True)
class _Weights:
    w1_fp4: torch.Tensor
    w1_scale: torch.Tensor
    w1_alpha: torch.Tensor
    a1_scale: torch.Tensor
    w2_fp4: torch.Tensor
    w2_scale: torch.Tensor
    w2_alpha: torch.Tensor
    a2_scale: torch.Tensor


@dataclass(frozen=True)
class _Inputs:
    a: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True)
class _BoundCase:
    """Strongly own every allocation used by a captured serving launch."""

    source_weights: _Weights
    experts: object
    scratch_plan: object
    scratch: tuple[torch.Tensor, ...]
    binding: object


def _make_nvfp4_weights(
    device: torch.device,
    *,
    seed: int,
    num_experts: int = _E,
    hidden_size: int = _K,
    intermediate_size: int = _N,
    logical_intermediate_size: int | None = None,
) -> _Weights:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w1_fp4 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_fp4 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )

    # ModelOpt NVFP4 carries FlashInfer's vec16-swizzled E4M3 block scales.
    # A constant exact power of two keeps the synthetic layer well-conditioned
    # while the random FP4 payload still exercises every nibble value.
    w1_logical_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 16),
        2.0**-5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    w2_logical_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 16),
        2.0**-5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    if logical_intermediate_size is not None:
        logical_n = int(logical_intermediate_size)
        if not (0 < logical_n <= intermediate_size and logical_n % 16 == 0):
            raise ValueError(
                "logical_intermediate_size must be positive, 16-aligned, and "
                "no larger than intermediate_size"
            )
        w1_fp4[:, logical_n:intermediate_size].zero_()
        w1_fp4[:, intermediate_size + logical_n :].zero_()
        w2_fp4[:, :, logical_n // 2 :].zero_()
        w1_logical_scale[:, logical_n:intermediate_size].fill_(0)
        w1_logical_scale[:, intermediate_size + logical_n :].fill_(0)
        w2_logical_scale[:, :, logical_n // 16 :].fill_(0)
    w1_scale = swizzle_block_scale_reference(w1_logical_scale).contiguous()
    w2_scale = swizzle_block_scale_reference(w2_logical_scale).contiguous()
    w1_alpha = torch.linspace(0.5, 0.8, num_experts, dtype=torch.float32, device=device)
    w2_alpha = torch.linspace(0.6, 0.9, num_experts, dtype=torch.float32, device=device)
    unit = torch.ones(1, dtype=torch.float32, device=device)
    return _Weights(
        w1_fp4=w1_fp4,
        w1_scale=w1_scale,
        w1_alpha=w1_alpha,
        a1_scale=unit,
        w2_fp4=w2_fp4,
        w2_scale=w2_scale,
        w2_alpha=w2_alpha,
        a2_scale=unit.clone(),
    )


def _make_mxfp4_weights(device: torch.device, *, seed: int) -> _Weights:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w1_fp4 = torch.randint(
        0,
        256,
        (_E, 2 * _N, _K // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_fp4 = torch.randint(
        0,
        256,
        (_E, _K, _N // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    # E8M0 byte 122 is exactly 2^-5.  These are checkpoint-native logical
    # K/32 grids; production preparation repacks them for tiny decode.
    w1_scale = torch.full((_E, 2 * _N, _K // 32), 122, dtype=torch.uint8, device=device)
    w2_scale = torch.full((_E, _K, _N // 32), 122, dtype=torch.uint8, device=device)
    alpha = torch.ones(_E, dtype=torch.float32, device=device)
    unit = torch.ones(_E, dtype=torch.float32, device=device)
    return _Weights(
        w1_fp4=w1_fp4,
        w1_scale=w1_scale,
        w1_alpha=alpha,
        a1_scale=unit,
        w2_fp4=w2_fp4,
        w2_scale=w2_scale,
        w2_alpha=alpha.clone(),
        a2_scale=unit.clone(),
    )


def _make_inputs(
    device: torch.device,
    *,
    m: int,
    seed: int,
    route_shift: int,
    num_experts: int = _E,
    hidden_size: int = _K,
    topk: int = _TOPK,
) -> _Inputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = (
        torch.randn(
            m, hidden_size, dtype=torch.float32, device=device, generator=generator
        )
        * 0.35
    ).to(torch.bfloat16)
    token = torch.arange(m, dtype=torch.int32, device=device)
    topk_ids = torch.stack(
        tuple((token + route_shift + idx) % num_experts for idx in range(topk)),
        dim=1,
    ).contiguous()
    topk_weights = torch.rand(
        m, topk, dtype=torch.float32, device=device, generator=generator
    ).add_(0.25)
    topk_weights.div_(topk_weights.sum(dim=1, keepdim=True))

    assert topk_ids.dtype is torch.int32 and topk_ids.is_contiguous()
    assert bool(((topk_ids >= 0) & (topk_ids < num_experts)).all().item())
    if topk > 1:
        assert bool((topk_ids[:, :-1] != topk_ids[:, 1:]).all().item())
    assert bool((topk_weights > 0).all().item())
    torch.testing.assert_close(
        topk_weights.sum(dim=1),
        torch.ones(m, dtype=torch.float32, device=device),
        rtol=0,
        atol=1e-6,
    )
    return _Inputs(a=a, topk_ids=topk_ids, topk_weights=topk_weights)


def _nvfp4_oracle(
    weights: _Weights,
    inputs: _Inputs,
    *,
    quant_scale_math: str = "direct_division",
    num_experts: int = _E,
    hidden_size: int = _K,
    intermediate_size: int = _N,
) -> torch.Tensor:
    # This is the pure-Torch GPU oracle; it does not instantiate or call a CuTe
    # kernel and consumes the original checkpoint layout directly.
    active = (inputs.topk_ids >= 0) & (inputs.topk_ids < num_experts)
    oracle_ids = torch.where(
        active, inputs.topk_ids, torch.zeros_like(inputs.topk_ids)
    ).contiguous()
    oracle_weights = torch.where(
        active, inputs.topk_weights, torch.zeros_like(inputs.topk_weights)
    ).contiguous()
    return moe_reference_nvfp4(
        inputs.a,
        weights.w1_fp4,
        weights.w1_scale,
        weights.w1_alpha,
        weights.w2_fp4,
        weights.w2_scale,
        weights.w2_alpha,
        weights.a1_scale,
        weights.a2_scale,
        oracle_ids,
        oracle_weights,
        num_experts,
        hidden_size,
        intermediate_size,
        activation="silu",
        quant_scale_math=quant_scale_math,
    )


def _mxfp4_oracle(weights: _Weights, inputs: _Inputs) -> torch.Tensor:
    # No prepared/repacked tensor participates in this oracle.  It consumes the
    # checkpoint-native FP4 + E8M0 grids and emulates MXFP8 activation rounding.
    active = (inputs.topk_ids >= 0) & (inputs.topk_ids < _E)
    oracle_ids = torch.where(
        active, inputs.topk_ids, torch.zeros_like(inputs.topk_ids)
    ).contiguous()
    oracle_weights = torch.where(
        active, inputs.topk_weights, torch.zeros_like(inputs.topk_weights)
    ).contiguous()
    return moe_reference_w4a8_mx(
        inputs.a.float(),
        weights.w1_fp4,
        weights.w1_scale,
        None,
        weights.w1_alpha,
        weights.w2_fp4,
        weights.w2_scale,
        None,
        weights.w2_alpha,
        oracle_ids,
        oracle_weights,
        _E,
        _K,
        _N,
        activation="silu",
        w13_layout="w13",
    )


def _prepare_and_bind(
    weights: _Weights,
    inputs: _Inputs,
    *,
    quant_mode: str,
    source_format: str,
    num_topk: int = _TOPK,
    w13_layout: str = "w13",
    fast_math: bool | None = False,
    swiglu_limit: float | None = None,
) -> _BoundCase:
    """Prepare checkpoint weights and bind caller-owned launch storage."""

    from b12x.moe.fused_moe._impl import TPMoEScratchCaps, plan_tp_moe_scratch

    experts = prepare_tp_moe_fp4_experts(
        a=inputs.a,
        a1_gscale=weights.a1_scale,
        w1_fp4=weights.w1_fp4,
        w1_blockscale=weights.w1_scale,
        w1_alphas=weights.w1_alpha,
        a2_gscale=weights.a2_scale,
        w2_fp4=weights.w2_fp4,
        w2_blockscale=weights.w2_scale,
        w2_alphas=weights.w2_alpha,
        activation="silu",
        quant_mode=quant_mode,
        source_format=source_format,
        w13_layout=w13_layout,
    )
    scratch_plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=int(inputs.a.shape[0]),
            num_topk=num_topk,
            device=inputs.a.device,
            weight_plan=experts.plan,
            quant_mode=quant_mode,
            core_token_counts=(int(inputs.a.shape[0]),),
            route_num_experts=0,
            swiglu_limit=swiglu_limit,
            frozen=True,
        )
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in scratch_plan.scratch_specs()
    )
    output = torch.empty_like(inputs.a)
    binding = scratch_plan.bind(
        scratch=scratch,
        a=inputs.a,
        experts=experts,
        topk_weights=inputs.topk_weights,
        topk_ids=inputs.topk_ids,
        output=output,
        input_scales_static=True,
        fast_math=fast_math,
    )
    assert binding.output is output
    return _BoundCase(
        source_weights=weights,
        experts=experts,
        scratch_plan=scratch_plan,
        scratch=scratch,
        binding=binding,
    )


def _assert_oracle(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    context: str,
    min_cos: float,
    max_normalized_rmse: float,
) -> None:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    assert bool(actual_f32.isfinite().all().item()), (context, "non-finite output")
    actual_rms = actual_f32.square().mean().sqrt().item()
    reference_rms = reference_f32.square().mean().sqrt().item()
    assert actual_rms > 1e-5, (context, "all-zero output")
    assert reference_rms > 1e-5, (context, "all-zero reference")
    metrics = compare_to_reference(actual_f32, reference_f32)
    normalized_rmse = metrics.rmse / reference_rms
    assert metrics.cos >= min_cos, (context, metrics, normalized_rmse)
    assert normalized_rmse <= max_normalized_rmse, (
        context,
        metrics,
        normalized_rmse,
    )


def _run_live_graph_check(
    case: _BoundCase,
    *,
    initial: _Inputs,
    changed: _Inputs,
    initial_reference: torch.Tensor,
    changed_reference: torch.Tensor,
    context: str,
    min_cos: float,
    max_normalized_rmse: float,
    exact_zero_rows: slice | None = None,
    replay_count: int = 1,
    require_bit_exact_replay: bool = True,
    assert_no_replay_allocations: bool = False,
) -> None:
    """Validate eager execution and in-place graph replay against GPU oracles."""

    from b12x.moe.fused_moe._impl import b12x_moe_fp4

    binding = case.binding
    output = binding.output
    assert output is not None

    # Eager warmup resolves and compiles the production specialization before
    # capture.  No workspace or output allocation occurs inside the graph.
    b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    _assert_oracle(
        output,
        initial_reference,
        context=f"{context}:eager",
        min_cos=min_cos,
        max_normalized_rmse=max_normalized_rmse,
    )
    initial_output = output.clone()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        b12x_moe_fp4(binding=binding)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    _assert_oracle(
        output,
        initial_reference,
        context=f"{context}:capture",
        min_cos=min_cos,
        max_normalized_rmse=max_normalized_rmse,
    )

    # Mutate every live serving input in place. The replay may include inactive
    # route IDs, which must remain graph-safe without changing the binding.
    initial.a.copy_(changed.a)
    initial.topk_ids.copy_(changed.topk_ids)
    initial.topk_weights.copy_(changed.topk_weights)
    live_tensors = (
        *case.scratch,
        initial.a,
        initial.topk_ids,
        initial.topk_weights,
        output,
    )
    live_addresses = tuple(tensor.data_ptr() for tensor in live_tensors)
    replay_output = None
    for replay_idx in range(replay_count):
        output.fill_(37.0)  # Poison proves the captured launch owns output reset.
        if assert_no_replay_allocations:
            allocation_count_before = torch.cuda.memory_stats()[
                "allocation.all.allocated"
            ]
            allocated_bytes_before = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        if assert_no_replay_allocations:
            assert (
                torch.cuda.memory_stats()["allocation.all.allocated"]
                == allocation_count_before
            ), (context, replay_idx, "CUDA allocation during graph replay")
            assert torch.cuda.memory_allocated() == allocated_bytes_before, (
                context,
                replay_idx,
                "live CUDA bytes changed during graph replay",
            )
            assert (
                tuple(tensor.data_ptr() for tensor in live_tensors) == live_addresses
            ), (
                context,
                replay_idx,
                "serving tensor address changed during graph replay",
            )
        _assert_oracle(
            output,
            changed_reference,
            context=f"{context}:live-replay-{replay_idx}",
            min_cos=min_cos,
            max_normalized_rmse=max_normalized_rmse,
        )
        if exact_zero_rows is not None:
            assert torch.count_nonzero(output[exact_zero_rows]).item() == 0
        if replay_output is None:
            replay_output = output.clone()
        elif require_bit_exact_replay:
            assert torch.equal(output, replay_output), (
                context,
                "graph replay changed the output bit pattern",
                replay_idx,
            )

    changed_rmse = (
        (output.float() - initial_output.float()).square().mean().sqrt().item()
    )
    initial_rms = initial_output.float().square().mean().sqrt().item()
    assert changed_rmse > max(1e-4, 0.05 * initial_rms), (
        context,
        changed_rmse,
        initial_rms,
    )


def _reset_dispatch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("B12X_MICRO_DYNAMIC_CUTOVER_PAIRS", raising=False)
    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    monkeypatch.delenv("B12X_DYNAMIC_EXTERNAL_ROUTE_PLAN", raising=False)
    monkeypatch.delenv("B12X_W4A8_TINY_DECODE", raising=False)
    from b12x.moe.fused_moe._impl import clear_tp_moe_caches

    clear_tp_moe_caches()


def test_standard_moe_micro_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach ``integration.tp_moe.micro_direct`` through production dispatch."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=101)
    initial = _make_inputs(device, m=2, seed=102, route_shift=0)
    # Seed 103 intentionally exercises an FP4 RN-even boundary.  For token 0,
    # direct division produces exactly -2.5 while the micro kernel's explicit
    # reciprocal-then-multiply is just below -2.5 and rounds to the adjacent
    # FP4 value.  Preserve that boundary and model the production evaluation
    # order instead of selecting an input that happens not to expose it.
    changed = _make_inputs(device, m=2, seed=103, route_shift=2)
    initial_reference = _nvfp4_oracle(
        weights,
        initial,
        quant_scale_math="reciprocal_multiply",
    )
    changed_reference = _nvfp4_oracle(
        weights,
        changed,
        quant_scale_math="reciprocal_multiply",
    )
    direct_division_reference = _nvfp4_oracle(weights, changed)
    assert not torch.equal(changed_reference, direct_division_reference), (
        "seed 103 must retain the reciprocal/division FP4 tie boundary"
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    assert case.scratch_plan.caps.frozen
    assert case.scratch_plan.launch_plan.implementation == "micro"
    assert case.binding.implementation == "micro"
    assert 2 * _TOPK < 64
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-micro",
        min_cos=0.999,
        max_normalized_rmse=0.03,
    )


def test_standard_moe_glm53_m1_rowpair_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise GLM-5.3's native M1 row-pair FC2 path through dispatch."""

    num_experts = 288
    hidden_size = 4096
    intermediate_size = 512
    topk = 8
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(
        device,
        seed=104,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    initial = _make_inputs(
        device,
        m=1,
        seed=105,
        route_shift=0,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    changed = _make_inputs(
        device,
        m=1,
        seed=106,
        route_shift=5,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    initial_reference = _nvfp4_oracle(
        weights,
        initial,
        quant_scale_math="reciprocal_multiply",
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    changed_reference = _nvfp4_oracle(
        weights,
        changed,
        quant_scale_math="reciprocal_multiply",
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        num_topk=topk,
    )
    assert case.scratch_plan.launch_plan.implementation == "micro"
    assert case.binding.implementation == "micro"
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-glm53-m1-rowpair",
        min_cos=0.999,
        max_normalized_rmse=0.03,
        replay_count=3,
        assert_no_replay_allocations=True,
    )


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_standard_moe_qwen38_nvfp4_padding_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
) -> None:
    """Ignore a fully padded row at Qwen3.8 TP4 decode geometry."""

    num_experts = 512
    hidden_size = 2560
    intermediate_size = 192
    topk = 10
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(
        device,
        seed=111,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        logical_intermediate_size=160,
    )
    initial = _make_inputs(
        device,
        m=m,
        seed=112,
        route_shift=0,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    changed = _make_inputs(
        device,
        m=m,
        seed=113,
        route_shift=17,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    if m > 1:
        changed.topk_ids[-1].fill_(-1)
    initial_reference = _nvfp4_oracle(
        weights,
        initial,
        quant_scale_math="reciprocal_multiply",
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    changed_reference = _nvfp4_oracle(
        weights,
        changed,
        quant_scale_math="reciprocal_multiply",
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if m > 1:
        assert torch.count_nonzero(changed_reference[-1]).item() == 0
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        num_topk=topk,
    )
    assert tuple(initial.a.shape) == (m, hidden_size)
    assert tuple(weights.w1_fp4.shape) == (
        num_experts,
        2 * intermediate_size,
        hidden_size // 2,
    )
    assert tuple(weights.w2_fp4.shape) == (
        num_experts,
        hidden_size,
        intermediate_size // 2,
    )
    assert case.scratch_plan.launch_plan.implementation == "micro"
    assert case.binding.implementation == "micro"
    case.scratch[0].fill_(0xFF)
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-qwen38-nvfp4-padding",
        min_cos=0.999,
        max_normalized_rmse=0.03,
        exact_zero_rows=slice(m - 1, m) if m > 1 else None,
        replay_count=3,
    )


@pytest.mark.parametrize(
    ("m", "intermediate_size"),
    [(1, 208), (2, 208), (2, 272)],
)
def test_standard_moe_micro_masks_source_native_w2_tail(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
    intermediate_size: int,
) -> None:
    """Bound micro FC2 loads by logical W2 rows, not padded scale rows."""

    num_experts = 4
    hidden_size = 512
    topk = 4
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(
        device,
        seed=121,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    initial = _make_inputs(
        device,
        m=m,
        seed=122,
        route_shift=0,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    changed = _make_inputs(
        device,
        m=m,
        seed=123,
        route_shift=1,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    initial_reference = _nvfp4_oracle(
        weights,
        initial,
        quant_scale_math="reciprocal_multiply",
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    changed_reference = _nvfp4_oracle(
        weights,
        changed,
        quant_scale_math="reciprocal_multiply",
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        num_topk=topk,
    )
    assert case.scratch_plan.launch_plan.implementation == "micro"
    assert case.binding.implementation == "micro"
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context=f"standard-moe-micro-w2-tail-n{intermediate_size}-m{m}",
        min_cos=0.999,
        max_normalized_rmse=0.03,
    )


def test_standard_moe_dynamic_prefill_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach ``integration.tp_moe.dynamic`` at a prefill-sized standard M."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=201)
    initial = _make_inputs(device, m=128, seed=202, route_shift=0)
    changed = _make_inputs(device, m=128, seed=203, route_shift=2)
    initial_reference = _nvfp4_oracle(weights, initial)
    changed_reference = _nvfp4_oracle(weights, changed)
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    launch_plan = case.scratch_plan.launch_plan
    assert launch_plan.implementation == "dynamic"
    assert case.binding.implementation == "dynamic"
    assert launch_plan.execution.tile_m == 64
    assert launch_plan.execution.tile_n == 128
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-dynamic-prefill-m128",
        min_cos=0.999,
        max_normalized_rmse=0.03,
    )


@pytest.mark.parametrize("tile_m", [16, 32, 64, 128])
@pytest.mark.parametrize("intermediate_size", [112, 144])
def test_standard_moe_nvfp4_n16_tail_dynamic_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tile_m: int,
    intermediate_size: int,
) -> None:
    """Keep a 16-mod-32 gated FC1 half boundary exact in dynamic MoE."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    monkeypatch.setenv("B12X_DYNAMIC_TILE_MN", f"{tile_m}x128")
    monkeypatch.setenv("B12X_MICRO_DYNAMIC_CUTOVER_PAIRS", "0")
    weights = _make_nvfp4_weights(
        device,
        seed=204,
        intermediate_size=intermediate_size,
    )
    initial = _make_inputs(device, m=16, seed=205, route_shift=0)
    changed = _make_inputs(device, m=16, seed=206, route_shift=2)
    initial_reference = _nvfp4_oracle(
        weights,
        initial,
        intermediate_size=intermediate_size,
    )
    changed_reference = _nvfp4_oracle(
        weights,
        changed,
        intermediate_size=intermediate_size,
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    launch_plan = case.scratch_plan.launch_plan
    assert launch_plan.implementation == "dynamic"
    assert launch_plan.execution.tile_m == tile_m
    assert case.binding.implementation == "dynamic"
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context=f"standard-moe-dynamic-n{intermediate_size}-m{tile_m}",
        min_cos=0.999,
        max_normalized_rmse=0.03,
    )


@pytest.mark.parametrize("m", [4, 7])
@pytest.mark.parametrize("work_source", ["materialized_queue", "persistent_grid"])
def test_standard_moe_external_route_plan_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
    work_source: str,
) -> None:
    """Consume a Triton-planned grouped route layout under live replay."""

    num_experts = 32
    hidden_size = 512
    intermediate_size = 128
    topk = 10
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    monkeypatch.setenv("B12X_MICRO_DYNAMIC_CUTOVER_PAIRS", "0")
    monkeypatch.setenv("B12X_DYNAMIC_EXTERNAL_ROUTE_PLAN", "1")
    monkeypatch.setenv("B12X_DYNAMIC_WORK_SOURCE", work_source)
    weights = _make_nvfp4_weights(
        device,
        seed=211,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    initial = _make_inputs(
        device,
        m=m,
        seed=212,
        route_shift=0,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    changed = _make_inputs(
        device,
        m=m,
        seed=213,
        route_shift=13,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    changed.topk_ids[-1, -2:].fill_(-1)
    initial_reference = _nvfp4_oracle(
        weights,
        initial,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    changed_reference = _nvfp4_oracle(
        weights,
        changed,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        num_topk=topk,
    )
    launch_plan = case.scratch_plan.launch_plan
    assert launch_plan.implementation == "dynamic"
    assert case.binding.implementation == "dynamic"
    assert launch_plan.execution.tile_m == 16
    assert launch_plan.execution.tile_n == 128
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context=f"standard-moe-external-route-plan-m{m}",
        min_cos=0.999,
        max_normalized_rmse=0.03,
        replay_count=3,
        require_bit_exact_replay=False,
        assert_no_replay_allocations=True,
    )
    expected_counts = torch.bincount(
        changed.topk_ids[changed.topk_ids >= 0].to(torch.int64),
        minlength=num_experts,
    ).to(torch.int32)
    torch.testing.assert_close(case.binding.row_counts, expected_counts)
    expected_tiles = int(((expected_counts + 15) // 16).sum().item())
    assert int(case.binding.expert_tile_base[-1].item()) == expected_tiles


@pytest.mark.parametrize("layout", ["vector", "scalar", "strided"])
@pytest.mark.parametrize("work_source", ["materialized_queue", "persistent_grid"])
def test_standard_moe_dynamic_scale_graph_reads_canonical_owner_updates(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    work_source: str,
) -> None:
    from b12x.moe.fused_moe import run

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    monkeypatch.setenv("B12X_MICRO_DYNAMIC_CUTOVER_PAIRS", "0")
    monkeypatch.setenv("B12X_DYNAMIC_EXTERNAL_ROUTE_PLAN", "1")
    monkeypatch.setenv("B12X_DYNAMIC_WORK_SOURCE", work_source)
    geometry = dict(num_experts=32, hidden_size=512, intermediate_size=128)
    weights = _make_nvfp4_weights(device, seed=241, **geometry)
    if layout == "vector":
        weights = replace(
            weights,
            a1_scale=weights.a1_scale.expand(32).contiguous(),
            a2_scale=weights.a2_scale.expand(32).contiguous(),
        )
    elif layout == "strided":
        weights = replace(
            weights,
            a1_scale=weights.a1_scale.expand(64).contiguous()[::2],
            a2_scale=weights.a2_scale.expand(64).contiguous()[::2],
        )
    inputs = _make_inputs(
        device,
        m=4,
        seed=242,
        route_shift=0,
        num_experts=32,
        hidden_size=512,
        topk=10,
    )
    changed_weights = replace(
        weights,
        a1_scale=weights.a1_scale * 2,
        a2_scale=weights.a2_scale * 0.5,
    )
    references = [
        _nvfp4_oracle(source, inputs, **geometry)
        for source in (weights, changed_weights)
    ]
    case = _prepare_and_bind(
        weights,
        inputs,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        num_topk=10,
    )
    binding = case.binding
    assert (binding.input_gs.data_ptr() == case.experts.a1_gscale.data_ptr()) == (
        layout == "vector"
    )
    assert (
        binding.down_input_scale.data_ptr() == case.experts.a2_gscale.data_ptr()
    ) == (layout == "vector")
    run(binding=binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(binding=binding)
    for index, reference in enumerate(references):
        if index:
            case.experts.a1_gscale.copy_(changed_weights.a1_scale)
            case.experts.a2_gscale.copy_(changed_weights.a2_scale)
        allocation_count = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocation_count
        _assert_oracle(
            binding.output,
            reference,
            context=f"canonical-scale-values-{index}",
            min_cos=0.999,
            max_normalized_rmse=0.03,
        )
    assert not torch.equal(references[0], references[1])


def test_standard_moe_glm53_tp4_nvfp4_live_graph_replay(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Replay 42 bindings with GLM-5.3 TP4-local NVFP4 expert dimensions.

    One synthetic weight set with 288 experts, hidden width 4096, and local
    intermediate width 512 backs independent inputs and outputs. All bindings
    share scratch and run sequentially in one captured graph.
    """

    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe.fused_moe._impl import b12x_moe_fp4

    num_experts = 288
    hidden_size = 4096
    intermediate_size = 512
    topk = 8
    tokens = 128
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(
        device,
        seed=211,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    weights = _Weights(
        w1_fp4=weights.w1_fp4,
        w1_scale=weights.w1_scale,
        w1_alpha=weights.w1_alpha,
        a1_scale=torch.linspace(
            0.7, 1.3, num_experts, dtype=torch.float32, device=device
        ),
        w2_fp4=weights.w2_fp4,
        w2_scale=weights.w2_scale,
        w2_alpha=weights.w2_alpha,
        a2_scale=torch.linspace(
            0.8, 1.2, num_experts, dtype=torch.float32, device=device
        ),
    )
    initial = _make_inputs(
        device,
        m=tokens,
        seed=212,
        route_shift=0,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        num_topk=topk,
        w13_layout="w31",
        fast_math=None,
        swiglu_limit=10.0,
    )
    assert case.scratch_plan.launch_plan.implementation == "dynamic"
    assert case.binding.implementation == "dynamic"

    layer_inputs = [initial]
    bindings = [case.binding]
    for layer_idx in range(1, 42):
        layer_input = _make_inputs(
            device,
            m=tokens,
            seed=212 + layer_idx,
            route_shift=layer_idx * 7,
            num_experts=num_experts,
            hidden_size=hidden_size,
            topk=topk,
        )
        layer_output = torch.empty_like(layer_input.a)
        layer_binding = case.scratch_plan.bind(
            scratch=case.scratch,
            a=layer_input.a,
            experts=case.experts,
            topk_weights=layer_input.topk_weights,
            topk_ids=layer_input.topk_ids,
            output=layer_output,
            input_scales_static=True,
            fast_math=None,
        )
        assert layer_binding.implementation == "dynamic"
        layer_inputs.append(layer_input)
        bindings.append(layer_binding)

    for binding in bindings:
        b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    assert all(
        bool(binding.output.float().isfinite().all().item()) for binding in bindings
    )
    guard = kernel_resolution_guard("GLM-shaped MoE capture and live replay")
    guard.__enter__()
    request.addfinalizer(lambda: guard.__exit__(None, None, None))

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        for binding in bindings:
            b12x_moe_fp4(binding=binding)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    for layer_idx, layer_input in enumerate(layer_inputs):
        layer_changed = _make_inputs(
            device,
            m=tokens,
            seed=413 + layer_idx,
            route_shift=97 + layer_idx * 11,
            num_experts=num_experts,
            hidden_size=hidden_size,
            topk=topk,
        )
        layer_input.a.copy_(layer_changed.a)
        layer_input.topk_ids.copy_(layer_changed.topk_ids)
        layer_input.topk_weights.copy_(layer_changed.topk_weights)
    for binding in bindings:
        b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    expected = tuple(binding.output.clone() for binding in bindings)
    assert all(bool(output.float().isfinite().all().item()) for output in expected)

    live_tensors = list(case.scratch)
    for layer_input, binding in zip(layer_inputs, bindings, strict=True):
        live_tensors.extend(
            (
                layer_input.a,
                layer_input.topk_ids,
                layer_input.topk_weights,
                binding.output,
            )
        )
    live_addresses = tuple(tensor.data_ptr() for tensor in live_tensors)

    for replay_idx in range(3):
        for binding in bindings:
            binding.output.fill_(37.0)
        allocation_count_before = torch.cuda.memory_stats(device)[
            "allocation.all.allocated"
        ]
        allocated_bytes_before = torch.cuda.memory_allocated(device)
        graph.replay()
        torch.cuda.synchronize()
        assert (
            torch.cuda.memory_stats(device)["allocation.all.allocated"]
            == allocation_count_before
        ), (replay_idx, "CUDA allocation during 42-layer graph replay")
        assert torch.cuda.memory_allocated(device) == allocated_bytes_before, (
            replay_idx,
            "live CUDA bytes changed during 42-layer graph replay",
        )
        assert tuple(tensor.data_ptr() for tensor in live_tensors) == live_addresses, (
            replay_idx,
            "serving tensor address changed during 42-layer graph replay",
        )
        for layer_idx, (binding, layer_expected) in enumerate(
            zip(bindings, expected, strict=True)
        ):
            assert bool(binding.output.float().isfinite().all().item()), (
                replay_idx,
                layer_idx,
            )
            error = (
                (binding.output.float() - layer_expected.float()).square().mean().sqrt()
            )
            reference_rms = layer_expected.float().square().mean().sqrt()
            assert float(error / reference_rms) < 0.01, (replay_idx, layer_idx)


def test_standard_moe_glm53_tp4_nvfp4_multishape_graph_replay(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Replay token widths 8 through 128 against one shared CUDA workspace.

    Each eight-token increment has an independent captured graph. One uint8
    allocation backs every graph's scratch after a 256-byte-aligned reservation
    of tokens * hidden_size * sizeof(BF16) bytes. Outputs have separate storage.
    Full and half-active route sets reuse the prewarmed plans with kernel
    resolution frozen.
    """

    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe.fused_moe._impl import (
        TPMoEScratchCaps,
        b12x_moe_fp4,
        plan_tp_moe_scratch,
    )

    num_experts = 288
    hidden_size = 4096
    intermediate_size = 512
    topk = 8
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(
        device,
        seed=251,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    weights = _Weights(
        w1_fp4=weights.w1_fp4,
        w1_scale=weights.w1_scale,
        w1_alpha=weights.w1_alpha,
        a1_scale=torch.linspace(
            0.7, 1.3, num_experts, dtype=torch.float32, device=device
        ),
        w2_fp4=weights.w2_fp4,
        w2_scale=weights.w2_scale,
        w2_alpha=weights.w2_alpha,
        a2_scale=torch.linspace(
            0.8, 1.2, num_experts, dtype=torch.float32, device=device
        ),
    )
    owner_input = _make_inputs(
        device,
        m=128,
        seed=252,
        route_shift=0,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    owner = _prepare_and_bind(
        weights,
        owner_input,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        num_topk=topk,
        w13_layout="w31",
        fast_math=None,
        swiglu_limit=10.0,
    )

    plans = []
    workspace_sizes = []
    for tokens in range(8, 129, 8):
        plan = plan_tp_moe_scratch(
            TPMoEScratchCaps(
                max_tokens=tokens,
                num_topk=topk,
                device=owner.experts.w1_fp4.device,
                weight_plan=owner.experts.plan,
                quant_mode="nvfp4",
                core_token_counts=(tokens,),
                route_num_experts=0,
                swiglu_limit=10.0,
                frozen=True,
            )
        )
        output_nbytes = tokens * hidden_size * torch.bfloat16.itemsize
        output_workspace_nbytes = (output_nbytes + 255) // 256 * 256
        scratch_nbytes = plan.scratch_specs()[0].nbytes
        assert plan.launch_plan.implementation == "dynamic", tokens
        plans.append((tokens, plan, output_workspace_nbytes))
        workspace_sizes.append(output_workspace_nbytes + scratch_nbytes)

    workspace = torch.empty(max(workspace_sizes), dtype=torch.uint8, device=device)
    graphs = []
    graph_owners = []
    for graph_idx, (tokens, plan, scratch_offset) in enumerate(plans):
        inputs = _make_inputs(
            device,
            m=tokens,
            seed=253 + graph_idx,
            route_shift=graph_idx * 13,
            num_experts=num_experts,
            hidden_size=hidden_size,
            topk=topk,
        )
        output = torch.empty_like(inputs.a)
        scratch = workspace.narrow(
            0,
            scratch_offset,
            workspace.numel() - scratch_offset,
        )
        binding = plan.bind(
            scratch=scratch,
            a=inputs.a,
            experts=owner.experts,
            topk_weights=inputs.topk_weights,
            topk_ids=inputs.topk_ids,
            output=output,
            input_scales_static=True,
            fast_math=None,
        )
        assert binding.implementation == "dynamic", tokens
        b12x_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
            workspace.fill_(0xA5)
            captured_binding = plan.bind(
                scratch=scratch,
                a=inputs.a,
                experts=owner.experts,
                topk_weights=inputs.topk_weights,
                topk_ids=inputs.topk_ids,
                output=output,
                input_scales_static=True,
                fast_math=None,
            )
            assert captured_binding.implementation == "dynamic", tokens
            b12x_moe_fp4(binding=captured_binding)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        graphs.append(graph)
        graph_owners.append(
            (inputs, output, scratch, binding, captured_binding, capture_stream)
        )

    live_tensors = [workspace]
    for inputs, output, scratch, _binding, _captured_binding, _stream in graph_owners:
        live_tensors.extend(
            (inputs.a, inputs.topk_ids, inputs.topk_weights, output, scratch)
        )
    live_addresses = tuple(tensor.data_ptr() for tensor in live_tensors)
    guard = kernel_resolution_guard("GLM-shaped MoE replay with full and inactive routes")
    guard.__enter__()
    request.addfinalizer(lambda: guard.__exit__(None, None, None))

    for replay_idx in range(3):
        graph_order = (
            range(len(graphs)) if replay_idx % 2 == 0 else reversed(range(len(graphs)))
        )
        for graph_idx in graph_order:
            inputs, output, _scratch, binding, _captured_binding, _capture_stream = (
                graph_owners[graph_idx]
            )
            inputs.topk_ids.copy_(
                torch.arange(topk, dtype=torch.int32, device=device)
                .unsqueeze(0)
                .expand_as(inputs.topk_ids)
            )
            inputs.topk_weights.fill_(1.0 / topk)
            live_tokens = inputs.topk_ids.shape[0]
            if replay_idx == 2:
                live_tokens = max(1, inputs.topk_ids.shape[0] // 2)
                inputs.topk_ids[live_tokens:].fill_(-1)
                inputs.topk_weights[live_tokens:].zero_()

            workspace.fill_(0xA5)
            b12x_moe_fp4(binding=binding)
            torch.cuda.synchronize()
            expected = output.clone()

            output.fill_(37.0)
            allocation_count_before = torch.cuda.memory_stats(device)[
                "allocation.all.allocated"
            ]
            allocated_bytes_before = torch.cuda.memory_allocated(device)
            graphs[graph_idx].replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"]
                == allocation_count_before
            ), (replay_idx, graph_idx, "CUDA allocation during graph replay")
            assert torch.cuda.memory_allocated(device) == allocated_bytes_before, (
                replay_idx,
                graph_idx,
                "live CUDA bytes changed during graph replay",
            )
            assert (
                tuple(tensor.data_ptr() for tensor in live_tensors) == live_addresses
            ), (
                replay_idx,
                graph_idx,
                "serving tensor address changed during graph replay",
            )
            _assert_oracle(
                output,
                expected,
                context=f"glm53-tp4-nvfp4-multishape-m{output.shape[0]}-replay-{replay_idx}",
                min_cos=0.999,
                max_normalized_rmse=0.01,
            )
            assert torch.count_nonzero(output[:live_tokens]).item() > 0
            if live_tokens < inputs.topk_ids.shape[0]:
                assert (
                    torch.count_nonzero(inputs.topk_ids[live_tokens:] + 1).item() == 0
                )
                assert torch.count_nonzero(output[live_tokens:]).item() == 0


@pytest.mark.parametrize("m", [80, 96])
def test_standard_moe_nvfp4_tp4_padded_prefill_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
) -> None:
    """Exercise the TP4 N160-to-N192 padded expert prefill contract."""

    num_experts = 512
    hidden_size = 2560
    intermediate_size = 192
    topk = 10
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(
        device,
        seed=221,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        logical_intermediate_size=160,
    )
    initial = _make_inputs(
        device,
        m=m,
        seed=222,
        route_shift=0,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    changed = _make_inputs(
        device,
        m=m,
        seed=223,
        route_shift=113,
        num_experts=num_experts,
        hidden_size=hidden_size,
        topk=topk,
    )
    initial_reference = _nvfp4_oracle(
        weights,
        initial,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    changed_reference = _nvfp4_oracle(
        weights,
        changed,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        num_topk=topk,
    )
    launch_plan = case.scratch_plan.launch_plan
    assert launch_plan.implementation == "dynamic"
    assert case.binding.implementation == "dynamic"
    assert launch_plan.execution.tile_m == 16
    assert launch_plan.execution.tile_n == 128
    case.scratch[0].view(torch.uint8).fill_(0xFF)
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context=f"standard-moe-nvfp4-tp4-padded-prefill-m{m}",
        min_cos=0.999,
        max_normalized_rmse=0.03,
        replay_count=3,
        require_bit_exact_replay=False,
        assert_no_replay_allocations=True,
    )


def test_standard_moe_dynamic_inactive_route_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore inactive routes in the production NVFP4 dynamic graph path."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_nvfp4_weights(device, seed=201)
    initial = _make_inputs(device, m=40, seed=202, route_shift=0)
    changed = _make_inputs(device, m=40, seed=203, route_shift=2)
    changed.topk_ids[-8:].fill_(-1)
    initial_reference = _nvfp4_oracle(weights, initial)
    changed_reference = _nvfp4_oracle(weights, changed)
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    launch_plan = case.scratch_plan.launch_plan
    assert launch_plan.implementation == "dynamic"
    assert case.binding.implementation == "dynamic"
    assert launch_plan.execution.tile_m == 32
    assert launch_plan.execution.tile_n == 128
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-dynamic-graph-m40",
        min_cos=0.999,
        max_normalized_rmse=0.03,
    )
    assert case.binding.output is not None
    assert torch.count_nonzero(case.binding.output[-8:]).item() == 0


def test_standard_moe_tiny_decode_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach both ``integration.tp_moe.tiny_decode`` production phases."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_mxfp4_weights(device, seed=301)
    initial = _make_inputs(device, m=2, seed=302, route_shift=0)
    changed = _make_inputs(device, m=2, seed=303, route_shift=2)

    # Preparation destructively transfers/re-packs the source allocation, so
    # calculate both independent checkpoint-layout references first.
    initial_reference = _mxfp4_oracle(weights, initial)
    changed_reference = _mxfp4_oracle(weights, changed)
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
    )
    assert case.scratch_plan.launch_plan.implementation == "micro"
    assert case.binding.implementation == "micro"
    assert case.binding.quant_mode == "w4a8_mx"
    assert 1 <= int(initial.a.shape[0]) <= 4
    assert _K % 256 == 0 and _N % 32 == 0 and (_K // 128) % 4 == 0
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-tiny-decode-phases-1-2",
        min_cos=0.998,
        max_normalized_rmse=0.05,
    )


def test_standard_moe_tiny_decode_inactive_route_live_graph_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore scheduler-padding routes during tiny-decode graph replay."""

    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_mxfp4_weights(device, seed=301)
    initial = _make_inputs(device, m=2, seed=302, route_shift=0)
    changed = _make_inputs(device, m=2, seed=303, route_shift=2)
    changed.topk_ids[1, 1] = -1

    initial_reference = _mxfp4_oracle(weights, initial)
    changed_reference = _mxfp4_oracle(weights, changed)
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
    )
    assert case.scratch_plan.launch_plan.implementation == "micro"
    assert case.binding.implementation == "micro"
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context="standard-moe-tiny-decode-inactive-route",
        min_cos=0.998,
        max_normalized_rmse=0.05,
    )
