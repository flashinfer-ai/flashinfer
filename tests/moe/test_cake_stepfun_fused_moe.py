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
finalize with the exported Cake StepFun FC1 kernels in the GEMM1 slot, for
NVFP4 (E2m1 output and the per-token bf16-output variant), BF16, per-tensor
FP8 and MXFP8. These tests compare that backend with the trtllm-gen fused-MoE
test references at the StepFun serving geometry, exercise every FC1 tactic the
backend enumerates, replay forwards through CUDA graphs (including the first
replay of fresh processes), and, for the families whose Cake kernels reproduce
the native FC1 bitwise, require the native outputs byte for byte.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
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
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
)
from flashinfer.fused_moe.runners import (
    CakeStepFunBf16Runner,
    CakeStepFunFp8PerTensorRunner,
    CakeStepFunMxfp8Runner,
    CakeStepFunNvfp4Runner,
    CakeStepFunRunner,
    TrtllmBf16RoutedRunner,
    TrtllmFp4RoutedRunner,
    TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorRunner,
)
from flashinfer.utils import get_compute_capability
from tests.moe.trtllm_gen_fused_moe_utils import (
    ActivationType,
    BF16Moe,
    FP4Moe,
    FP8BlockScaleMoe,
    FP8PerTensorMoe,
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
NVFP4_PER_TOKEN = QuantConfig(
    weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4, per_token_scale=True
)
BF16 = QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16)
FP8 = QuantConfig(weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor)
MXFP8 = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
CAKE = BackendOptions(candidates=(CakeStepFunConfig(backend="cake"),))

# One entry per FC1 family the backend serves: the quantization, the trtllm-gen
# test harness that produces the data and the reference, the weight layout the
# native kernels (and therefore the Cake kernels) consume, the native runner the
# Cake runner mirrors, whether the Cake kernels reproduce the native FC1 bitwise
# (bf16, per-tensor fp8 and mxfp8 do; the nvfp4 kernels accumulate in a
# different order) and whether the family supports fused shared experts.
PRECISIONS = {
    "nvfp4": SimpleNamespace(
        quant=NVFP4,
        make_impl=lambda: FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
        layout=WeightLayout.MajorK,
        family="nvfp4",
        runner_cls=CakeStepFunNvfp4Runner,
        native_config=TrtllmFp4Config,
        native_runner=TrtllmFp4RoutedRunner,
        bitwise_twin=False,
        fused_shared_experts=True,
    ),
    "bf16": SimpleNamespace(
        quant=BF16,
        make_impl=BF16Moe,
        layout=WeightLayout.BlockMajorK,
        family="bf16",
        runner_cls=CakeStepFunBf16Runner,
        native_config=TrtllmBf16Config,
        native_runner=TrtllmBf16RoutedRunner,
        bitwise_twin=True,
        fused_shared_experts=False,
    ),
    "fp8": SimpleNamespace(
        quant=FP8,
        make_impl=FP8PerTensorMoe,
        layout=WeightLayout.MajorK,
        family="fp8",
        runner_cls=CakeStepFunFp8PerTensorRunner,
        native_config=TrtllmFp8PerTensorConfig,
        native_runner=TrtllmFp8PerTensorRunner,
        bitwise_twin=True,
        fused_shared_experts=False,
    ),
    "mxfp8": SimpleNamespace(
        quant=MXFP8,
        make_impl=lambda: FP8BlockScaleMoe(
            fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8
        ),
        layout=WeightLayout.MajorK,
        family="mxfp8",
        runner_cls=CakeStepFunMxfp8Runner,
        native_config=TrtllmFp8BlockConfig,
        native_runner=TrtllmFp8BlockRunner,
        bitwise_twin=True,
        fused_shared_experts=True,
    ),
}
# Rows whose FC1 tile candidates all have a bitwise Cake twin: per-tensor FP8
# ships tiles 8/16/32 only, so its larger token counts run the native tiles.
# (precision, num_tokens) seed rows whose FC1 tile candidates have no exported Cake kernel yet and
# therefore run the native FC1: the per-tensor fp8 kernels cover tiles 8/16/32 only (the tile-64
# persistent fp8 kernel carries no device tile count), so T=2048 (tiles 64/128) falls back.
KNOWN_NATIVE_FALLBACK = {("fp8", 2048)}

BITWISE_ROWS = [
    ("bf16", 8),
    ("bf16", 64),
    ("bf16", 512),
    ("bf16", 2048),
    ("fp8", 8),
    ("fp8", 64),
    ("mxfp8", 8),
    ("mxfp8", 512),
    ("mxfp8", 2048),
]


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
    quant: QuantConfig = NVFP4,
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
        quant=quant,
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE_SIZE,
            local_num_experts=NUM_EXPERTS,
            num_fused_shared_experts=num_fused_shared_experts,
        ),
        activation=SwiGLUStep() if activation is None else activation,
        backend=backend,
        execution=ExecutionConfig(enable_pdl=enable_pdl, tune_max_num_tokens=num_tokens),
    )


def _kernel_view(precision: str, static: dict, limits: torch.Tensor) -> dict:
    """The runner weight view from the harness's kernel-ready static data.

    ``gemm1_clamp_limit`` is stored in the units the FC1 kernels clamp: the
    logical limit over the FC1 gate dequant scale for NVFP4 and per-tensor FP8
    (as ``CakeStepFunConfig.prepare_weights`` does), physical units otherwise.
    """
    if precision == "nvfp4":
        return {
            "gemm1_weights": static["gemm1_weights_fp4_shuffled"],
            "gemm1_weights_scale": static["gemm1_scales_fp4_shuffled"],
            "gemm2_weights": static["gemm2_weights_fp4_shuffled"],
            "gemm2_weights_scale": static["gemm2_scales_fp4_shuffled"],
            "output1_scale_scalar": static["scale_c_fc1"],
            "output1_scale_gate_scalar": static["scale_gate_fc1"],
            "output2_scale_scalar": static["scale_c_fc2"],
            "gemm1_clamp_limit": (limits / static["scale_gate_fc1"]).contiguous(),
        }
    if precision == "bf16":
        return {
            "gemm1_weights": static["gemm1_weights"],
            "gemm2_weights": static["gemm2_weights"],
            "gemm1_clamp_limit": limits.contiguous(),
        }
    if precision == "fp8":
        return {
            "gemm1_weights": static["gemm1_weights"],
            "gemm2_weights": static["gemm2_weights"],
            "output1_scales_scalar": static["scale_c_fc1"],
            "output1_scales_gate_scalar": static["scale_gate_fc1"],
            "output2_scales_scalar": static["scale_c_fc2"],
            "gemm1_clamp_limit": (limits / static["scale_gate_fc1"]).contiguous(),
        }
    if precision == "mxfp8":
        # The harness keeps the swizzled UE8M0 weight scales flattened per expert; the runner
        # validates them as [experts, rows, hidden / 32] (same bytes).
        experts = static["gemm1_weights"].shape[0]
        return {
            "gemm1_weights": static["gemm1_weights"],
            "gemm1_weights_scale": static["gemm1_scales"].reshape(
                experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 32
            ),
            "gemm2_weights": static["gemm2_weights"],
            "gemm2_weights_scale": static["gemm2_scales"].reshape(
                experts, HIDDEN_SIZE, INTERMEDIATE_SIZE // 32
            ),
            "gemm1_clamp_limit": limits.contiguous(),
        }
    raise KeyError(precision)


def _kernel_activations(precision, moe_impl, hidden_states, hidden_states_scale_global, inputs):
    """The (hidden_states_q, hidden_states_scale) pair the runners consume (linear scale layouts)."""
    if precision == "nvfp4":
        linear = moe_impl.quantize_inputs(
            hidden_states, hidden_states_scale_global, is_swizzling=False
        )
        return linear["hidden_states"], linear["hidden_states_scale"]
    if precision == "bf16":
        return hidden_states, None
    if precision == "fp8":
        return inputs["hidden_states"], None
    return inputs["hidden_states"], inputs["hidden_states_scale"]


def _build_case(
    cache_permute_indices,
    *,
    precision: str = "nvfp4",
    num_tokens: int,
    limits: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routing_method=RoutingMethodType.Renormalize,
    n_groups=None,
    top_k_groups=None,
    routed_scaling=None,
    has_routing_bias: bool = False,
    seed: int = 0,
    enable_pdl: bool = True,
):
    """Generate the trtllm-gen StepFun test data, its reference and the unified-API inputs.

    The weights, activations and routing follow ``run_moe_test`` (seed 0,
    amplitude 12) for the harness of ``precision``. The kernel weight view is
    the harness's shuffled preparation in the layout the native kernels read,
    so the Cake backend consumes exactly the tensors the trtllm-gen StepFun
    tests feed the native kernels. The view is registered for the Cake backend
    and for the native runner it mirrors.
    """
    spec = PRECISIONS[precision]
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

    moe_impl = spec.make_impl()
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
        {"use_shuffled_weight": True, "layout": spec.layout},
    )
    view = _kernel_view(precision, static, limits)
    weights = MoEWeightPack()
    weights.prepare_for("cake", view)
    weights.prepare_for(spec.native_runner.backend_key, view)
    hidden_states_q, hidden_states_scale = _kernel_activations(
        precision, moe_impl, hidden_states, hidden_states_scale_global, inputs_data
    )
    act = MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        routing_logits=expert_logits,
        routing_bias=routing_bias,
        routing_input_mode=RoutingInputMode.FromLogits,
    )
    common = dict(
        num_tokens=num_tokens,
        quant=spec.quant,
        activation=SwiGLUStep(limit=float(limits[0].item())),
        num_fused_shared_experts=num_fused_shared_experts,
        routing_method=routing_method,
        n_groups=n_groups,
        top_k_groups=top_k_groups,
        routed_scaling=routed_scaling,
        enable_pdl=enable_pdl,
    )
    return SimpleNamespace(
        precision=precision,
        reference=reference.float(),
        tolerances=moe_impl.get_tolerances(),
        act=act,
        weights=weights,
        config=_config(**common),
        native_config=_config(
            backend=BackendOptions(candidates=(spec.native_config(),)), **common
        ),
    )


def _layer_runner(config, device, runner_cls, act, weights):
    layer = MoELayer(config, device)
    assert len(layer.runners) == 1
    runner = layer.runners[0]
    assert isinstance(runner, runner_cls), type(runner)
    runner.check_support()
    runner.build()
    packed = runner.pack_inputs(act, weights)
    kwargs = runner.launch_kwargs_for(packed)
    tactics = [list(map(int, tactic)) for tactic in runner.get_valid_tactics(packed, None)]
    assert tactics, f"{runner_cls.__name__} enumerated no tactics"
    return layer, runner, packed, kwargs, tactics


def _cake_runner(case, device):
    return _layer_runner(
        case.config, device, PRECISIONS[case.precision].runner_cls, case.act, case.weights
    )


def _native_runner(case, device):
    return _layer_runner(
        case.native_config,
        device,
        PRECISIONS[case.precision].native_runner,
        case.act,
        case.weights,
    )


def _cake_tiles(runner, family: str) -> set:
    return {int(tile) for tile in runner._module.moe_op.cake_stepfun_fc1_tiles(family)}


def _forward(runner, packed, kwargs, tactic):
    output = runner.forward(packed, tactic=tactic, do_preparation=True, **kwargs)
    torch.cuda.synchronize()
    return output.clone()


def _capture_and_replay(runner, packed, kwargs, tactic):
    """Capture one forward in a CUDA graph (after a warm-up on the capture stream) and replay it once."""
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
    return replayed, graph


def test_config_is_explicit_exact_sm100_sm103_and_stepfun_only():
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
    for spec in PRECISIONS.values():
        assert CakeStepFunRunner.supports_quant(spec.quant)
        assert CakeStepFunRunner.runner_class_for(spec.quant) is spec.runner_cls
        assert spec.runner_cls.backend_key == "cake"
        assert spec.runner_cls.supported_activation_classes_by_quant[spec.quant.pair] == (
            SwiGLUStep,
        )
    assert CakeStepFunRunner.supports_quant(NVFP4_PER_TOKEN)
    assert CakeStepFunRunner.runner_class_for(NVFP4_PER_TOKEN) is CakeStepFunNvfp4Runner
    for unsupported in (
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8),
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.BF16),
    ):
        assert not CakeStepFunRunner.supports_quant(unsupported)
        with pytest.raises(NotImplementedError, match="no FC1 family"):
            CakeStepFunRunner.runner_class_for(unsupported)
    # Instantiation dispatches to the family runner.
    for spec in PRECISIONS.values():
        runner = CakeStepFunRunner(
            _config(num_tokens=8, quant=spec.quant), torch.device("cpu")
        )
        assert type(runner) is spec.runner_cls


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_runner_rejects_non_stepfun_activations(precision):
    runner = CakeStepFunRunner(
        _config(num_tokens=8, quant=PRECISIONS[precision].quant, activation=SwiGLU()),
        torch.device("cpu"),
    )
    with pytest.raises(NotImplementedError, match="SwiGLUStep"):
        runner.check_support()


def test_prepare_weights_requires_supported_quant_and_stepfun():
    w1 = torch.zeros(2, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
    w2 = torch.zeros(2, HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.bfloat16)
    common = dict(
        num_local_experts=2, hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE
    )
    with pytest.raises(ValueError, match="supports NVFP4xNVFP4"):
        CakeStepFunConfig.prepare_weights(
            w1,
            w2,
            quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
            **common,
        )
    with pytest.raises(ValueError, match="SwiGLUStep"):
        CakeStepFunConfig.prepare_weights(w1, w2, activation=SwiGLU(), **common)
    with pytest.raises(ValueError, match="hidden_states_scale_global"):
        CakeStepFunConfig.prepare_weights(w1, w2, quant=FP8, **common)
    with pytest.raises(ValueError, match="supports NVFP4xNVFP4"):
        CakeStepFunConfig.prepare_activations(
            w1[0], quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
        )


def _fp8_global_scales(hidden, limit, device):
    hidden_global = torch.tensor(
        [448.0 / float(hidden.float().abs().max().item())], device=device, dtype=torch.float32
    )
    intermediate_global = torch.tensor([448.0 / (2.0 * limit)], device=device, dtype=torch.float32)
    return hidden_global, intermediate_global


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_prepare_weights_adds_step_limits_and_matches_trtllm_backend(precision):
    """The shared view carries limits in kernel units; Cake and trtllm agree on it."""
    device = _require_cake_device()
    spec = PRECISIONS[precision]
    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size, num_tokens = 4, 512, 512, 8
    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
    w1 /= math.sqrt(hidden_size)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16)
    w2 /= math.sqrt(intermediate_size)
    hidden = 12.0 * torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    step_limits = torch.tensor([7.0, 16.0, 7.0, 16.0], device=device)
    common = dict(
        quant=spec.quant,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    activation_kwargs = {}
    if precision == "fp8":
        hidden_global, intermediate_global = _fp8_global_scales(hidden, 7.0, device)
        common.update(
            hidden_states_scale_global=hidden_global, intermediate_scale_global=intermediate_global
        )
        activation_kwargs["hidden_states_scale_global"] = hidden_global
    view = CakeStepFunConfig.prepare_weights(
        w1, w2, activation=SwiGLUStep(limit=7.0), step_limits=step_limits, **common
    )
    gate_key = {"nvfp4": "output1_scale_gate_scalar", "fp8": "output1_scales_gate_scalar"}.get(
        precision
    )
    expected = step_limits / view[gate_key] if gate_key else step_limits
    assert torch.equal(view["gemm1_clamp_limit"], expected)
    default_view = CakeStepFunConfig.prepare_weights(w1, w2, **common)
    default_expected = torch.full((num_experts,), 7.0, device=device)
    if gate_key:
        default_expected = default_expected / default_view[gate_key]
    assert torch.equal(default_view["gemm1_clamp_limit"], default_expected)
    with pytest.raises(ValueError, match="one limit per physical expert row"):
        CakeStepFunConfig.prepare_weights(w1, w2, step_limits=step_limits[:2], **common)

    quantized, scales = CakeStepFunConfig.prepare_activations(
        hidden, quant=spec.quant, **activation_kwargs
    )
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.bfloat16)
    act = MoEActivationPack(
        hidden_states_q=quantized,
        hidden_states_scale=scales,
        routing_logits=logits,
        routing_input_mode=RoutingInputMode.FromLogits,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cake", view)
    weights.prepare_for(spec.native_runner.backend_key, view)
    outputs = {}
    for backend, runner_cls in (
        (CakeStepFunConfig(backend="cake"), spec.runner_cls),
        (spec.native_config(), spec.native_runner),
    ):
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=2),
            quant=spec.quant,
            experts=ExpertConfig(intermediate_size=intermediate_size, local_num_experts=num_experts),
            activation=SwiGLUStep(limit=7.0),
            backend=BackendOptions(candidates=(backend,)),
            execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=num_tokens),
        )
        runner = runner_cls(config, device)
        runner.check_support()
        runner.build()
        packed = runner.pack_inputs(act, weights)
        outputs[runner_cls.__name__] = _forward(runner, packed, runner.launch_kwargs_for(packed), -1)
    tolerances = spec.make_impl().get_tolerances()
    check_accuracy(
        outputs[spec.native_runner.__name__].float(),
        outputs[spec.runner_cls.__name__].float(),
        **tolerances,
    )


@pytest.mark.parametrize("limit", [7.0, 16.0])
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_stepfun_matches_trtllm_reference_for_every_tactic(
    precision, num_tokens, limit, cache_permute_indices
):
    device = _require_cake_device()
    limits = torch.full((NUM_EXPERTS,), limit, device="cuda", dtype=torch.float32)
    case = _build_case(
        cache_permute_indices, precision=precision, num_tokens=num_tokens, limits=limits
    )
    layer, runner, packed, kwargs, tactics = _cake_runner(case, device)
    cake_tiles = _cake_tiles(runner, PRECISIONS[precision].family)
    if not any(tactic[0] in cake_tiles for tactic in tactics):
        message = (
            f"no FC1 tile candidate of T={num_tokens} has a Cake {precision} kernel: "
            f"tactics {tactics}, Cake tiles {sorted(cake_tiles)}"
        )
        if (precision, num_tokens) in KNOWN_NATIVE_FALLBACK:
            pytest.xfail(message)
        pytest.fail(message)
    for tactic in tactics:
        output = _forward(runner, packed, kwargs, tactic)
        check_accuracy(case.reference, output.float(), **case.tolerances)
    output = layer(case.act, case.weights)
    torch.cuda.synchronize()
    check_accuracy(case.reference, output.float(), **case.tolerances)


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_stepfun_mixed_expert_limits(precision, cache_permute_indices):
    """Routed experts alternate 7/16 limits; where supported, a fused shared expert reads 16."""
    device = _require_cake_device()
    spec = PRECISIONS[precision]
    shared = 1 if spec.fused_shared_experts else 0
    limits = torch.tensor(
        [7.0 if expert % 2 == 0 else 16.0 for expert in range(NUM_EXPERTS)] + [16.0] * shared,
        device="cuda",
        dtype=torch.float32,
    )
    routing = (
        dict(
            routing_method=RoutingMethodType.DeepSeekV3,
            n_groups=8,
            top_k_groups=4,
            routed_scaling=2.5,
            has_routing_bias=True,
        )
        if shared
        else {}
    )
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=64,
        limits=limits,
        num_fused_shared_experts=shared,
        **routing,
    )
    layer, runner, packed, kwargs, tactics = _cake_runner(case, device)
    for tactic in tactics:
        output = _forward(runner, packed, kwargs, tactic)
        check_accuracy(case.reference, output.float(), **case.tolerances)
    output = layer(case.act, case.weights)
    torch.cuda.synchronize()
    check_accuracy(case.reference, output.float(), **case.tolerances)


@pytest.mark.parametrize("num_tokens", [8, 512])
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_stepfun_cuda_graph_replay(precision, num_tokens, cache_permute_indices):
    device = _require_cake_device()
    limits = torch.full((NUM_EXPERTS,), 7.0, device="cuda", dtype=torch.float32)
    case = _build_case(
        cache_permute_indices, precision=precision, num_tokens=num_tokens, limits=limits
    )
    _, runner, packed, kwargs, tactics = _cake_runner(case, device)
    cake_tiles = _cake_tiles(runner, PRECISIONS[precision].family)
    for tactic in [t for t in tactics if t[0] in cake_tiles][:2]:
        eager = _forward(runner, packed, kwargs, tactic)
        replayed, graph = _capture_and_replay(runner, packed, kwargs, tactic)
        assert torch.equal(replayed, eager), f"tactic {tactic}: first replay differs from eager"
        replayed.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replayed, eager), f"tactic {tactic}: second replay differs from eager"
        check_accuracy(case.reference, replayed.float(), **case.tolerances)


@pytest.mark.parametrize("precision,num_tokens", BITWISE_ROWS)
def test_stepfun_reproduces_native_fc1_twin_bitwise(precision, num_tokens, cache_permute_indices):
    """Every Cake tactic reproduces some native tactic of the same FC1 tile byte for byte.

    The BF16, per-tensor FP8 and MXFP8 Cake FC1 kernels write the same bytes as
    their trtllm-gen twins; routing, GEMM2 and finalize are the native kernels,
    so the fused-MoE output must equal the native output for the matching GEMM2
    configuration exactly.
    """
    device = _require_cake_device()
    spec = PRECISIONS[precision]
    assert spec.bitwise_twin
    limits = torch.full((NUM_EXPERTS,), 7.0, device="cuda", dtype=torch.float32)
    case = _build_case(
        cache_permute_indices, precision=precision, num_tokens=num_tokens, limits=limits
    )
    _, cake, cake_packed, cake_kwargs, cake_tactics = _cake_runner(case, device)
    _, native, native_packed, native_kwargs, native_tactics = _native_runner(case, device)
    cake_tiles = _cake_tiles(cake, spec.family)
    native_outputs = {}
    for tactic in native_tactics:
        native_outputs.setdefault(tactic[0], []).append(
            (tactic, _forward(native, native_packed, native_kwargs, tactic))
        )
    checked = 0
    for tactic in cake_tactics:
        if tactic[0] not in cake_tiles:
            continue
        output = _forward(cake, cake_packed, cake_kwargs, tactic)
        check_accuracy(case.reference, output.float(), **case.tolerances)
        candidates = native_outputs.get(tactic[0], [])
        matches = [nt for nt, nout in candidates if torch.equal(output, nout)]
        closest = min(
            ((nt, (output.float() - nout.float()).abs().max().item()) for nt, nout in candidates),
            key=lambda item: item[1],
            default=None,
        )
        assert matches, (
            f"{precision} T={num_tokens}: Cake tactic {tactic} matches no native tactic of tile "
            f"{tactic[0]} bitwise ({len(candidates)} candidates; closest {closest})"
        )
        checked += 1
    assert checked, f"no Cake tactic at T={num_tokens}: {cake_tactics} vs tiles {sorted(cake_tiles)}"


@pytest.mark.parametrize("num_tokens", [8, 512, 2048])
def test_stepfun_per_token_nvfp4_matches_native(num_tokens):
    """Per-token NVFP4 (bf16 FC1 output, FlashInfer quantizes GEMM2's input) agrees with trtllm."""
    from flashinfer.quantization import SfLayout, nvfp4_quantize
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )

    device = _require_cake_device()
    torch.manual_seed(0)
    w1 = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    w1 /= math.sqrt(HIDDEN_SIZE)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16)
    w2 /= math.sqrt(INTERMEDIATE_SIZE)
    hidden = HIDDEN_STATE_AMPLITUDE * torch.randn(
        num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    logits = torch.randn(num_tokens, NUM_EXPERTS, device=device, dtype=torch.bfloat16)
    global_scale = make_nvfp4_global_scale(
        hidden, per_token_activation=True, nvfp4_4over6_config=current_nvfp4_4over6_config()
    )
    quantized, block_scales, per_token_scale = nvfp4_quantize(
        hidden,
        global_scale,
        sfLayout=SfLayout.layout_linear,
        sf_vec_size=16,
        backend="cuda",
        per_token_activation=True,
    )
    activation = SwiGLUStep(limit=7.0)
    view = CakeStepFunConfig.prepare_weights(
        w1,
        w2,
        quant=NVFP4_PER_TOKEN,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        activation=activation,
        device=device,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cake", view)
    weights.prepare_for(TrtllmFp4RoutedRunner.backend_key, view)
    act = MoEActivationPack(
        hidden_states_q=quantized,
        hidden_states_scale=block_scales.view(torch.float8_e4m3fn).reshape(num_tokens, -1),
        per_token_scale=per_token_scale.contiguous(),
        routing_logits=logits,
        routing_input_mode=RoutingInputMode.FromLogits,
    )
    runners = {}
    for backend, runner_cls in (
        (CakeStepFunConfig(backend="cake"), CakeStepFunNvfp4Runner),
        (TrtllmFp4Config(), TrtllmFp4RoutedRunner),
    ):
        config = _config(
            num_tokens=num_tokens,
            quant=NVFP4_PER_TOKEN,
            activation=activation,
            backend=BackendOptions(candidates=(backend,)),
        )
        runners[runner_cls] = _layer_runner(config, device, runner_cls, act, weights)
    _, cake, cake_packed, cake_kwargs, cake_tactics = runners[CakeStepFunNvfp4Runner]
    _, native, native_packed, native_kwargs, _ = runners[TrtllmFp4RoutedRunner]
    assert cake._inner.use_per_token_scaling is True
    assert native._inner.use_per_token_scaling is True
    per_token_tiles = _cake_tiles(cake, "nvfp4_bf16tok")
    assert per_token_tiles
    reference = _forward(native, native_packed, native_kwargs, -1).float()
    tolerances = FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4).get_tolerances()
    checked = 0
    for tactic in cake_tactics:
        if tactic[0] not in per_token_tiles:
            continue
        output = _forward(cake, cake_packed, cake_kwargs, tactic)
        check_accuracy(reference, output.float(), **tolerances)
        checked += 1
    assert checked, f"no per-token Cake tactic at T={num_tokens}: {cake_tactics} vs {sorted(per_token_tiles)}"
    check_accuracy(reference, _forward(cake, cake_packed, cake_kwargs, -1).float(), **tolerances)


def _first_replay_probe(precision: str, num_tokens: int, tactic_index: int) -> dict:
    """Fresh-process probe: eager forward, graph capture, one replay; bitwise agreement."""
    device = torch.device("cuda", torch.cuda.current_device())
    limits = torch.full((NUM_EXPERTS,), 7.0, device="cuda", dtype=torch.float32)
    case = _build_case({}, precision=precision, num_tokens=num_tokens, limits=limits)
    _, runner, packed, kwargs, tactics = _cake_runner(case, device)
    cake_tiles = _cake_tiles(runner, PRECISIONS[precision].family)
    cake_tactics = [t for t in tactics if t[0] in cake_tiles]
    tactic = cake_tactics[min(tactic_index, len(cake_tactics) - 1)]
    eager = _forward(runner, packed, kwargs, tactic)
    replayed, _ = _capture_and_replay(runner, packed, kwargs, tactic)
    return {
        "tactic": tactic,
        "bitwise": bool(torch.equal(replayed, eager)),
        "max_abs_diff": float((replayed.float() - eager.float()).abs().max().item()),
    }


_PROBE_SCRIPT = """
import json, sys
sys.path.insert(0, {root!r})
import torch
from tests.moe import test_cake_stepfun_fused_moe as t
results = {{}}
for precision in {precisions!r}:
    for tactic_index in {tactic_indices!r}:
        results[f"{{precision}}:{{tactic_index}}"] = t._first_replay_probe(precision, {num_tokens}, tactic_index)
print("FIRST_REPLAY_RESULT " + json.dumps(results), flush=True)
"""


def test_stepfun_first_graph_replay_in_fresh_processes():
    """The first CUDA-graph replay of a fresh process matches eager for every family.

    Each subprocess builds the T=8 case for every precision, runs eager, captures
    a graph with PDL enabled (the configuration under which a stale first replay
    was observed) and replays once. ``CAKE_STEPFUN_GRAPH_STRESS_PROCESSES`` sets
    the number of processes (default 4); every process is a fresh module load.
    """
    _require_cake_device()
    processes = int(os.environ.get("CAKE_STEPFUN_GRAPH_STRESS_PROCESSES", "4"))
    root = str(Path(__file__).resolve().parents[2])
    script = _PROBE_SCRIPT.format(
        root=root, precisions=list(PRECISIONS), tactic_indices=[0, 1], num_tokens=8
    )
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    failures = []
    checked = 0
    for index in range(processes):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        marker = [line for line in proc.stdout.splitlines() if line.startswith("FIRST_REPLAY_RESULT ")]
        assert proc.returncode == 0 and marker, (
            f"probe process {index} failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
        results = json.loads(marker[-1][len("FIRST_REPLAY_RESULT ") :])
        for key, result in results.items():
            checked += 1
            if not result["bitwise"]:
                failures.append((index, key, result))
    assert not failures, f"{len(failures)}/{checked} first replays differ from eager: {failures}"
    assert checked == processes * len(PRECISIONS) * 2
