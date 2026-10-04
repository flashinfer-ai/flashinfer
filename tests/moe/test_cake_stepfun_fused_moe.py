"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Cake StepFun fused MoE through the unified MoE API.

``CakeStepFunConfig(backend="cake")`` runs trtllm-gen routing, GEMM2 and
finalize with the exported Cake StepFun FC1 kernels in the GEMM1 slot. These
tests compare that backend with the trtllm-gen fused-MoE test reference
(``FP4Moe.compute_reference``) at the StepFun serving geometry, exercise every
FC1 tactic the backend enumerates, and replay a forward through a CUDA graph.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CakeStepFunConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    SwiGLU,
    SwiGLUStep,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.runners import CakeStepFunRunner, TrtllmFp4RoutedRunner
from flashinfer.utils import get_compute_capability
from tests.moe.trtllm_gen_fused_moe_utils import (
    ActivationType,
    FP4Moe,
    QuantMode,
    RoutingMethodType,
    WeightLayout,
    check_accuracy,
    moe_args,
    routing_reference_no_aux,
    routing_reference_renormalize,
)

pytestmark = pytest.mark.long_running

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1536
NUM_EXPERTS = 64
TOP_K = 8
PADDING = 8
TOKENS = (8, 64, 512, 2048)
# Matches the trtllm-gen StepFun tests: both limits become effective.
HIDDEN_STATE_AMPLITUDE = 12.0
NVFP4 = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
CAKE = BackendOptions(candidates=(CakeStepFunConfig(backend="cake"),))


@pytest.fixture(scope="module")
def cache_permute_indices():
    return {}


def _require_cake_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Cake StepFun fused MoE requires CUDA.")
    device = torch.device("cuda", torch.cuda.current_device())
    compute_capability = get_compute_capability(device)
    if compute_capability not in ((10, 0), (10, 3)):
        pytest.skip(
            "Cake StepFun fused MoE requires exact SM100 or SM103, got "
            f"SM{compute_capability[0]}{compute_capability[1]}."
        )
    return device


def _config(
    *,
    num_tokens: int,
    activation=None,
    num_fused_shared_experts: int = 0,
    routing_method=RoutingMethodType.Renormalize,
    n_groups=None,
    top_k_groups=None,
    routed_scaling=None,
    backend=CAKE,
    enable_pdl=True,
) -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            method=routing_method,
            n_group=n_groups,
            topk_group=top_k_groups,
            routed_scaling_factor=routed_scaling,
        ),
        quant=NVFP4,
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE_SIZE,
            local_num_experts=NUM_EXPERTS,
            num_fused_shared_experts=num_fused_shared_experts,
        ),
        activation=SwiGLUStep() if activation is None else activation,
        backend=backend,
        execution=ExecutionConfig(enable_pdl=enable_pdl, tune_max_num_tokens=num_tokens),
    )


def _build_case(
    cache_permute_indices,
    *,
    num_tokens: int,
    limits: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routing_method=RoutingMethodType.Renormalize,
    n_groups=None,
    top_k_groups=None,
    routed_scaling=None,
    has_routing_bias: bool = False,
    seed: int = 0,
):
    """Generate the trtllm-gen StepFun test data, its reference and the unified-API inputs.

    The weights, activations and routing follow ``run_moe_test`` (seed 0,
    amplitude 12). The kernel weight view is the harness's shuffled MajorK
    preparation, so the Cake backend consumes exactly the tensors the trtllm-gen
    StepFun tests feed the native kernels; ``gemm1_clamp_limit`` is the logical
    limit divided by the FC1 gate dequant scale.
    """
    torch.random.manual_seed(seed)
    total_experts = NUM_EXPERTS + num_fused_shared_experts
    expert_logits = torch.randn((num_tokens, NUM_EXPERTS), device="cuda").to(
        torch.bfloat16
    )
    routing_bias = (
        torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
        if has_routing_bias
        else None
    )
    hidden_states = HIDDEN_STATE_AMPLITUDE * torch.randn(
        (num_tokens, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (total_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    ) / math.sqrt(HIDDEN_SIZE)
    gemm2_weights = torch.randn(
        (total_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    ) / math.sqrt(INTERMEDIATE_SIZE)

    if routing_method == RoutingMethodType.Renormalize:
        permute_info, scores = routing_reference_renormalize(
            expert_logits, TOP_K, NUM_EXPERTS, PADDING
        )
    elif routing_method == RoutingMethodType.DeepSeekV3:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            TOP_K,
            n_groups,
            top_k_groups,
            routed_scaling,
            PADDING,
            num_fused_shared_experts=num_fused_shared_experts,
        )
    else:
        raise NotImplementedError(routing_method)

    moe_impl = FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4)
    moe_impl._cache_permute_indices = cache_permute_indices
    weights_data = moe_impl.quantize_weights(gemm1_weights, gemm2_weights, hidden_states)
    hidden_states_scale_global = weights_data["hidden_states_scale_global"]
    inputs_data = moe_impl.quantize_inputs(hidden_states, hidden_states_scale_global)
    args = moe_args(
        num_tokens,
        total_experts,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        TOP_K + num_fused_shared_experts,
        PADDING,
        inputs_data["hidden_states"],
        inputs_data["hidden_states_scale"],
        hidden_states_scale_global,
        scores,
        weights_data["gemm1_weights"],
        weights_data["gemm1_scales"],
        weights_data["gemm1_scales_global"],
        weights_data["gemm2_weights"],
        weights_data["gemm2_scales"],
        weights_data["gemm2_scales_global"],
        permute_info,
        False,
        ActivationType.SwigluStep,
        gemm1_clamp_limit=limits,
    )
    reference, args_dequant = moe_impl.compute_reference(args)
    assert reference is not None
    static = moe_impl.prepare_static_weights_for_kernel(
        args_dequant,
        args,
        gemm1_weights,
        gemm2_weights,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        total_experts,
        {"use_shuffled_weight": True, "layout": WeightLayout.MajorK},
    )
    view = {
        "gemm1_weights": static["gemm1_weights_fp4_shuffled"],
        "gemm1_weights_scale": static["gemm1_scales_fp4_shuffled"],
        "gemm2_weights": static["gemm2_weights_fp4_shuffled"],
        "gemm2_weights_scale": static["gemm2_scales_fp4_shuffled"],
        "output1_scale_scalar": static["scale_c_fc1"],
        "output1_scale_gate_scalar": static["scale_gate_fc1"],
        "output2_scale_scalar": static["scale_c_fc2"],
        "gemm1_clamp_limit": (limits / static["scale_gate_fc1"]).contiguous(),
    }
    weights = MoEWeightPack()
    weights.prepare_for("cake", view)
    linear = moe_impl.quantize_inputs(
        hidden_states, hidden_states_scale_global, is_swizzling=False
    )
    act = MoEActivationPack(
        hidden_states_q=linear["hidden_states"],
        hidden_states_scale=linear["hidden_states_scale"],
        routing_logits=expert_logits,
        routing_bias=routing_bias,
        routing_input_mode=RoutingInputMode.FromLogits,
    )
    config = _config(
        num_tokens=num_tokens,
        activation=SwiGLUStep(limit=float(limits[0].item())),
        num_fused_shared_experts=num_fused_shared_experts,
        routing_method=routing_method,
        n_groups=n_groups,
        top_k_groups=top_k_groups,
        routed_scaling=routed_scaling,
    )
    return SimpleNamespace(
        reference=reference.float(),
        tolerances=moe_impl.get_tolerances(),
        act=act,
        weights=weights,
        config=config,
    )


def _cake_runner(case, device):
    layer = MoELayer(case.config, device)
    assert len(layer.runners) == 1
    runner = layer.runners[0]
    assert isinstance(runner, CakeStepFunRunner)
    runner.check_support()
    runner.build()
    packed = runner.pack_inputs(case.act, case.weights)
    kwargs = runner.launch_kwargs_for(packed)
    tactics = [list(map(int, tactic)) for tactic in runner.get_valid_tactics(packed, None)]
    assert tactics, "the Cake StepFun backend enumerated no tactics"
    return layer, runner, packed, kwargs, tactics


def _forward(runner, packed, kwargs, tactic):
    output = runner.forward(packed, tactic=tactic, do_preparation=True, **kwargs)
    torch.cuda.synchronize()
    return output.clone()


def test_config_is_explicit_exact_sm100_sm103_and_nvfp4_stepfun_only():
    config = CakeStepFunConfig(backend="cake")
    assert repr(config) == "CakeStepFunConfig(backend='cake')"
    assert config == CakeStepFunConfig()
    with pytest.raises(ValueError, match="must be 'cake'"):
        CakeStepFunConfig(backend="trtllm")
    assert CakeStepFunConfig.supported(100)
    assert CakeStepFunConfig.supported(103)
    assert not CakeStepFunConfig.supported(107)
    assert not CakeStepFunConfig.supported(120)
    assert CakeStepFunRunner.backend_key == "cake"
    assert CakeStepFunRunner.supports_quant(NVFP4)
    assert not CakeStepFunRunner.supports_quant(
        QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16)
    )
    assert not CakeStepFunRunner.supports_quant(
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
    )


def test_runner_rejects_non_stepfun_activations():
    runner = CakeStepFunRunner(
        _config(num_tokens=8, activation=SwiGLU()), torch.device("cpu")
    )
    with pytest.raises(NotImplementedError, match="SwiGLUStep"):
        runner.check_support()


def test_prepare_weights_requires_nvfp4_stepfun():
    w1 = torch.zeros(2, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
    w2 = torch.zeros(2, HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.bfloat16)
    common = dict(
        num_local_experts=2, hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE
    )
    with pytest.raises(ValueError, match="NVFP4"):
        CakeStepFunConfig.prepare_weights(
            w1,
            w2,
            quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
            **common,
        )
    with pytest.raises(ValueError, match="SwiGLUStep"):
        CakeStepFunConfig.prepare_weights(w1, w2, activation=SwiGLU(), **common)


def test_prepare_weights_adds_raw_step_limits_and_matches_trtllm_backend():
    """The shared NVFP4 view carries raw limits; Cake and trtllm agree on it."""
    device = _require_cake_device()
    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size, num_tokens = 4, 512, 512, 8
    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
    w1 /= math.sqrt(hidden_size)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16)
    w2 /= math.sqrt(intermediate_size)
    step_limits = torch.tensor([7.0, 16.0, 7.0, 16.0], device=device)
    common = dict(
        quant=NVFP4,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    view = CakeStepFunConfig.prepare_weights(w1, w2, activation=SwiGLUStep(limit=7.0), step_limits=step_limits, **common)
    expected = step_limits / view["output1_scale_gate_scalar"]
    assert torch.equal(view["gemm1_clamp_limit"], expected)
    default_view = CakeStepFunConfig.prepare_weights(w1, w2, **common)
    assert torch.equal(
        default_view["gemm1_clamp_limit"],
        torch.full_like(default_view["output1_scale_gate_scalar"], 7.0)
        / default_view["output1_scale_gate_scalar"],
    )
    with pytest.raises(ValueError, match="one limit per physical expert row"):
        CakeStepFunConfig.prepare_weights(w1, w2, step_limits=step_limits[:2], **common)

    hidden = 12.0 * torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    quantized, scales = CakeStepFunConfig.prepare_activations(hidden, quant=NVFP4)
    topk_ids = (
        torch.arange(num_tokens * 2, device=device, dtype=torch.int32).reshape(num_tokens, 2)
        % num_experts
    )
    topk_weights = torch.full((num_tokens, 2), 0.5, device=device, dtype=torch.float32)
    act = MoEActivationPack(
        hidden_states_q=quantized,
        hidden_states_scale=scales,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cake", view)
    weights.prepare_for("trtllm_fp4_routed", view)
    outputs = {}
    for backend, runner_cls in (
        (CakeStepFunConfig(backend="cake"), CakeStepFunRunner),
        (TrtllmFp4Config(), TrtllmFp4RoutedRunner),
    ):
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=2),
            quant=NVFP4,
            experts=ExpertConfig(intermediate_size=intermediate_size, local_num_experts=num_experts),
            activation=SwiGLUStep(limit=7.0),
            backend=BackendOptions(candidates=(backend,)),
            execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=num_tokens),
        )
        runner = runner_cls(config, device)
        runner.check_support()
        runner.build()
        packed = runner.pack_inputs(act, weights)
        outputs[runner.backend_key] = _forward(runner, packed, runner.launch_kwargs_for(packed), -1)
    tolerances = FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4).get_tolerances()
    check_accuracy(outputs["trtllm_fp4_routed"].float(), outputs["cake"].float(), **tolerances)


@pytest.mark.parametrize("limit", [7.0, 16.0])
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_stepfun_matches_trtllm_reference_for_every_tactic(
    num_tokens, limit, cache_permute_indices
):
    device = _require_cake_device()
    limits = torch.full((NUM_EXPERTS,), limit, device="cuda", dtype=torch.float32)
    case = _build_case(cache_permute_indices, num_tokens=num_tokens, limits=limits)
    layer, runner, packed, kwargs, tactics = _cake_runner(case, device)
    for tactic in tactics:
        output = _forward(runner, packed, kwargs, tactic)
        check_accuracy(case.reference, output.float(), **case.tolerances)
    output = layer(case.act, case.weights)
    torch.cuda.synchronize()
    check_accuracy(case.reference, output.float(), **case.tolerances)


def test_stepfun_mixed_expert_limits_with_fused_shared_expert(cache_permute_indices):
    """Routed experts alternate 7/16 limits; the fused shared expert reads 16."""
    device = _require_cake_device()
    limits = torch.tensor(
        [7.0 if expert % 2 == 0 else 16.0 for expert in range(NUM_EXPERTS)] + [16.0],
        device="cuda",
        dtype=torch.float32,
    )
    case = _build_case(
        cache_permute_indices,
        num_tokens=64,
        limits=limits,
        num_fused_shared_experts=1,
        routing_method=RoutingMethodType.DeepSeekV3,
        n_groups=8,
        top_k_groups=4,
        routed_scaling=2.5,
        has_routing_bias=True,
    )
    layer, runner, packed, kwargs, tactics = _cake_runner(case, device)
    for tactic in tactics:
        output = _forward(runner, packed, kwargs, tactic)
        check_accuracy(case.reference, output.float(), **case.tolerances)
    output = layer(case.act, case.weights)
    torch.cuda.synchronize()
    check_accuracy(case.reference, output.float(), **case.tolerances)


@pytest.mark.parametrize("num_tokens", [8, 512])
def test_stepfun_cuda_graph_replay(num_tokens, cache_permute_indices):
    device = _require_cake_device()
    limits = torch.full((NUM_EXPERTS,), 7.0, device="cuda", dtype=torch.float32)
    case = _build_case(cache_permute_indices, num_tokens=num_tokens, limits=limits)
    _, runner, packed, kwargs, tactics = _cake_runner(case, device)
    tactic = tactics[0]
    eager = _forward(runner, packed, kwargs, tactic)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        runner.forward(packed, tactic=tactic, **kwargs)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            replayed = runner.forward(packed, tactic=tactic, **kwargs)
    torch.cuda.synchronize()
    replayed.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, eager)
    check_accuracy(case.reference, replayed.float(), **case.tolerances)
