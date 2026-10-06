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

``CakeStepFunConfig(backend="cake_stepfun")`` runs trtllm-gen routing, GEMM2 and
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
import re
import subprocess
import sys
from collections import Counter
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
    MoEFinalizeConfig,
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
CAKE = BackendOptions(candidates=(CakeStepFunConfig(backend="cake_stepfun"),))

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
# The Cake backend has no native FC1 fallback: a token count whose FC1 tile window has no exported
# Cake kernel is a hard failure of the inventory, never an excused row.
LIMIT_MODES = ("7", "16", "mixed")
ROUTING_REGIMES = ("uniform", "ragged", "randperm")
ROUTING_SEED = 20261004

# Rows whose FC1 tile candidates all have a bitwise Cake twin (bf16, per-tensor fp8 and mxfp8
# reproduce the native FC1 twin bit for bit; nvfp4 does not and is checked by tolerance).
# fp8 tile 64: bitwise at one tile per expert (T=512); <= 1 e4m3 ulp when an expert spans several
# tiles (T=2048, measured 0.25 at magnitude 4-8), so that row is covered by the tolerance test only.
BITWISE_ROWS = [
    ("bf16", 8),
    ("bf16", 64),
    ("bf16", 512),
    ("bf16", 2048),
    ("fp8", 8),
    ("fp8", 64),
    ("fp8", 512),
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
    do_finalize=True,
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
        execution=ExecutionConfig(
            enable_pdl=enable_pdl, tune_max_num_tokens=num_tokens
        ),
        finalize=MoEFinalizeConfig(do_finalize=do_finalize),
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


def _kernel_activations(
    precision, moe_impl, hidden_states, hidden_states_scale_global, inputs
):
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
    logits: torch.Tensor | None = None,
    do_finalize: bool = True,
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
    if logits is not None:
        expert_logits = logits.to(device="cuda", dtype=torch.bfloat16)
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
    weights_data = moe_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )
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
    weights.prepare_for("cake_stepfun", view)
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
        do_finalize=do_finalize,
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


def _layer_runner(config, device, runner_cls, act, weights, *, native=False):
    layer = MoELayer(config, device)
    assert len(layer.runners) == 1
    runner = layer.runners[0]
    assert isinstance(runner, runner_cls), type(runner)
    try:
        runner.check_support()
    except NotImplementedError as error:
        if native and "SwiGLUStep" in str(error):
            # The public trtllm-gen artifact ships no StepFun kernels, so the native
            # runners do not advertise SwiGLUStep; the comparison needs a tree with
            # the native StepFun path.
            pytest.skip(
                f"native {runner_cls.__name__} does not advertise SwiGLUStep "
                f"(no native StepFun path in this tree): {error}"
            )
        raise
    runner.build()
    packed = runner.pack_inputs(act, weights)
    kwargs = runner.launch_kwargs_for(packed)
    tactics = [
        list(map(int, tactic)) for tactic in runner.get_valid_tactics(packed, None)
    ]
    assert tactics, f"{runner_cls.__name__} enumerated no tactics"
    return layer, runner, packed, kwargs, tactics


def _cake_runner(case, device):
    return _layer_runner(
        case.config,
        device,
        PRECISIONS[case.precision].runner_cls,
        case.act,
        case.weights,
    )


def _native_runner(case, device):
    return _layer_runner(
        case.native_config,
        device,
        PRECISIONS[case.precision].native_runner,
        case.act,
        case.weights,
        native=True,
    )


def _target(device) -> str:
    major, minor = get_compute_capability(device)
    return f"sm_{major}{minor}a"


def _full_path_unavailable_reason(device) -> str | None:
    """None when the module for this device runs the full Cake path, else why not."""
    from flashinfer.jit.cake_stepfun_moe import (
        cake_stepfun_missing_stages,
        resolve_cake_stepfun_full_path,
    )

    target = _target(device)
    missing = cake_stepfun_missing_stages(target)
    if missing:
        return (
            f"Cake StepFun full path unavailable for {target}: the generated inventory "
            f"has no {', '.join(missing)} kernel(s)"
        )
    if not resolve_cake_stepfun_full_path(target):
        return "Cake StepFun full path disabled by FLASHINFER_CAKE_STEPFUN_FULL_PATH"
    return None


def _require_full_path() -> torch.device:
    device = _require_cake_device()
    reason = _full_path_unavailable_reason(device)
    if reason is not None:
        pytest.skip(reason)
    return device


def _limits(mode: str, num_fused_shared_experts: int = 0) -> torch.Tensor:
    if mode == "mixed":
        values = [7.0 if expert % 2 == 0 else 16.0 for expert in range(NUM_EXPERTS)]
    else:
        values = [float(mode)] * NUM_EXPERTS
    values += [16.0] * num_fused_shared_experts
    return torch.tensor(values, device="cuda", dtype=torch.float32)


def _routing_logits(
    num_tokens: int, regime: str, seed: int = ROUTING_SEED
) -> torch.Tensor:
    """Routing logits of a load regime.

    ``uniform``: i.i.d. normal logits (balanced expert loads). ``ragged``: token ``t`` prefers
    the experts around ``(t * E / T)^2 / E`` so the per-expert loads are ragged and ascend
    with the token index (expert 0 heavy, high experts sparse). ``randperm``: the ragged
    rows in a random token order (same multiset of tokens, scrambled permutation).
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", generator=generator)
    if regime == "uniform":
        return base.to(torch.bfloat16)
    centers = (
        torch.arange(num_tokens, device="cuda", dtype=torch.float32)
        * NUM_EXPERTS
        / num_tokens
    )
    centers = (centers * centers / NUM_EXPERTS).floor()
    experts = torch.arange(NUM_EXPERTS, device="cuda", dtype=torch.float32)
    logits = base - 0.25 * (experts[None, :] - centers[:, None]).abs()
    if regime == "randperm":
        logits = logits[torch.randperm(num_tokens, device="cuda", generator=generator)]
    elif regime != "ragged":
        raise KeyError(regime)
    return logits.to(torch.bfloat16)


def _unfinalized_forward(runner, packed, kwargs, tactic):
    """[gemm2_output (permuted rows), expert_weights, expanded_idx_to_permuted_idx] of one forward."""
    result = runner.forward(packed, tactic=tactic, do_preparation=True, **kwargs)
    torch.cuda.synchronize()
    assert isinstance(result, (list, tuple)) and len(result) >= 3, type(result)
    return [tensor.clone() for tensor in result[:3]]


def _cake_tiles(runner, family: str) -> set:
    return {int(tile) for tile in runner._module.moe_op.cake_stepfun_fc1_tiles(family)}


def _factorized(runner, packed):
    """The runner's factorized tactic space: (tile, config) -> (FC1, FC2) components and the
    per-tile anchor, i.e. the tile's default (FC1, FC2) configuration pair."""
    return runner._inner.get_factorized_tactic_space(packed)


def _native_space_for_tile(native, native_packed, tile: int):
    """Native factorized tactic space whose candidate window contains ``tile``.

    The identities of a tile's anchor and (FC1, FC2) components do not depend on the token
    count; the candidate window does (the Cake module windows over its exported tiles, the
    native one over the full ladder), so a tile outside the native window of the test's token
    count is queried at the token count that centres the native window on it.
    """
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

    space = _factorized(native, native_packed)
    if tile in space.tiles:
        return space
    inputs = MoeRunnerInputs.from_list(list(native_packed))
    hidden = inputs.hidden_states
    inputs.hidden_states = hidden.new_empty(
        (tile * NUM_EXPERTS // TOP_K,) + tuple(hidden.shape[1:])
    )
    space = _factorized(native, inputs.to_list())
    assert tile in space.tiles, (
        f"the native runner enumerates no tactic of tile {tile} (tiles {space.tiles})"
    )
    return space


def _native_twin_tactic(native, native_packed, cake_space, tactic, full_path: bool):
    """The native tactic twinned with the Cake tactic ``tactic`` (same tile).

    The exported Cake FC1 and FC2 kernels are ports of the trtllm-gen configurations the
    native dispatcher pairs by default with the tile (the tile's anchor), so on the full Cake
    path the twin is the anchor tactic. With the native FC2 (FC1-only module) the twin pairs
    the tile's default FC1 configuration with the very FC2 configuration the Cake tactic runs.
    """
    tile = int(tactic[0])
    native_space = _native_space_for_tile(native, native_packed, tile)
    anchor = native_space.anchor(tile)
    if full_path:
        return [int(v) for v in anchor.tactic]
    cake_anchor = cake_space.anchor(tile)
    identity = (tile, int(tactic[1]))
    fc2 = next(
        t.fc2
        for t in cake_space.fc2_sweep(tile, cake_anchor.fc1)
        if tuple(int(v) for v in t.tactic) == identity
    )
    try:
        twin = native_space.compose(tile, anchor.fc1, fc2)
    except KeyError:
        pytest.fail(
            f"no native tactic pairs the default FC1 configuration {anchor.fc1} of tile {tile} "
            f"with FC2 configuration {fc2} (Cake tactic {list(tactic)})"
        )
    return [int(v) for v in twin.tactic]


def _bitwise_matches(native, native_packed, native_kwargs, tile, output):
    """Native tactics of ``tile`` (every FC1 x FC2 pair) whose output equals ``output`` byte for
    byte, as (tactic, FC1, FC2) triples (failure diagnostics)."""
    native_space = _native_space_for_tile(native, native_packed, tile)
    anchor = native_space.anchor(tile)
    matches = []
    for fc1 in sorted({t.fc1 for t in native_space.fc1_sweep(tile, anchor.fc2)}):
        for t in native_space.fc2_sweep(tile, fc1):
            nout = _forward(native, native_packed, native_kwargs, list(t.tactic))
            if torch.equal(output, nout):
                matches.append(([int(v) for v in t.tactic], t.fc1, t.fc2))
    return matches


def _require_routing_input(device, kind: str) -> None:
    from flashinfer.jit.cake_stepfun_moe import cake_stepfun_routing_inputs

    exported = cake_stepfun_routing_inputs(_target(device))
    if kind not in exported:
        pytest.skip(
            f"Cake StepFun routing for {_target(device)}: the generated inventory has no "
            f"routing kernel reading {kind} (exported: {sorted(exported) or 'none'})"
        )


def _require_finalize_weight_dtype(device, dtype: str) -> None:
    from flashinfer.jit.cake_stepfun_moe import cake_stepfun_finalize_weight_dtypes

    exported = cake_stepfun_finalize_weight_dtypes(_target(device))
    if dtype not in exported:
        pytest.skip(
            f"Cake StepFun finalize for {_target(device)}: the generated inventory has no "
            f"finalize kernel reading {dtype} expert weights (exported: "
            f"{sorted(exported) or 'none'})"
        )


def _precomputed_routing(native_op, case, num_tokens: int, logits: torch.Tensor):
    """(topk_ids int32 [T, K], topk_weights bf16 [T, K]) of the native router for ``logits``
    (every expert local, so each expanded slot maps to one expert tile)."""
    tables = _staged_routing(native_op, case, PADDING, num_tokens, logits)
    expanded = tables["expanded_idx_to_permuted_idx"]
    assert bool((expanded >= 0).all()), "every expert is local in this configuration"
    ids = tables["cta_idx_xy_to_batch_idx"][expanded.long() // PADDING]
    return (
        ids.reshape(num_tokens, TOP_K).to(torch.int32).contiguous(),
        tables["expert_weights"].reshape(num_tokens, TOP_K).contiguous(),
    )


def _precomputed_pack(case, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    """``case.act`` re-expressed in the unpacked pre-routed protocol of the benchmark."""
    return MoEActivationPack(
        hidden_states_q=case.act.hidden_states_q,
        hidden_states_scale=case.act.hidden_states_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


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
    config = CakeStepFunConfig(backend="cake_stepfun")
    assert repr(config) == "CakeStepFunConfig(backend='cake_stepfun')"
    assert config == CakeStepFunConfig()
    with pytest.raises(ValueError, match="must be 'cake_stepfun'"):
        CakeStepFunConfig(backend="trtllm")
    assert CakeStepFunConfig.supported(100)
    assert CakeStepFunConfig.supported(103)
    assert not CakeStepFunConfig.supported(107)
    assert not CakeStepFunConfig.supported(120)
    assert CakeStepFunRunner.backend_key == "cake_stepfun"
    for spec in PRECISIONS.values():
        assert CakeStepFunRunner.supports_quant(spec.quant)
        assert CakeStepFunRunner.runner_class_for(spec.quant) is spec.runner_cls
        assert spec.runner_cls.backend_key == "cake_stepfun"
        assert spec.runner_cls.supported_activation_classes_by_quant[
            spec.quant.pair
        ] == (SwiGLUStep,)
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
def test_fused_shared_experts_follow_the_native_family(precision):
    """bf16 / per-tensor fp8 reject S > 0 like their trtllm-gen runners; nvfp4 / mxfp8 accept it."""
    spec = PRECISIONS[precision]
    # MoEConfig only admits fused shared experts under DeepSeekV3 routing.
    runner = CakeStepFunRunner(
        _config(
            num_tokens=8,
            quant=spec.quant,
            num_fused_shared_experts=1,
            routing_method=RoutingMethodType.DeepSeekV3,
            n_groups=8,
            top_k_groups=4,
            routed_scaling=2.5,
        ),
        torch.device("cpu"),
    )
    assert type(runner) is spec.runner_cls
    assert type(runner).supports_fused_shared_experts is spec.fused_shared_experts
    assert spec.native_runner.supports_fused_shared_experts is spec.fused_shared_experts
    if spec.fused_shared_experts:
        runner._assert_shared_experts_supported()
    else:
        with pytest.raises(NotImplementedError, match="fused shared experts"):
            runner.check_support()


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
        num_local_experts=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
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
            w1[0],
            quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        )


def _fp8_global_scales(hidden, limit, device):
    hidden_global = torch.tensor(
        [448.0 / float(hidden.float().abs().max().item())],
        device=device,
        dtype=torch.float32,
    )
    intermediate_global = torch.tensor(
        [448.0 / (2.0 * limit)], device=device, dtype=torch.float32
    )
    return hidden_global, intermediate_global


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_prepare_weights_adds_step_limits_and_matches_trtllm_backend(precision):
    """The shared view carries limits in kernel units; Cake and trtllm agree on it."""
    device = _require_cake_device()
    spec = PRECISIONS[precision]
    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size, num_tokens = 4, 512, 512, 8
    w1 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    w1 /= math.sqrt(hidden_size)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16
    )
    w2 /= math.sqrt(intermediate_size)
    hidden = 12.0 * torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
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
            hidden_states_scale_global=hidden_global,
            intermediate_scale_global=intermediate_global,
        )
        activation_kwargs["hidden_states_scale_global"] = hidden_global
    view = CakeStepFunConfig.prepare_weights(
        w1, w2, activation=SwiGLUStep(limit=7.0), step_limits=step_limits, **common
    )
    gate_key = {
        "nvfp4": "output1_scale_gate_scalar",
        "fp8": "output1_scales_gate_scalar",
    }.get(precision)
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
    weights.prepare_for("cake_stepfun", view)
    weights.prepare_for(spec.native_runner.backend_key, view)
    outputs = {}
    for backend, runner_cls in (
        (CakeStepFunConfig(backend="cake_stepfun"), spec.runner_cls),
        (spec.native_config(), spec.native_runner),
    ):
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=2),
            quant=spec.quant,
            experts=ExpertConfig(
                intermediate_size=intermediate_size, local_num_experts=num_experts
            ),
            activation=SwiGLUStep(limit=7.0),
            backend=BackendOptions(candidates=(backend,)),
            execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=num_tokens),
        )
        runner = runner_cls(config, device)
        try:
            runner.check_support()
        except NotImplementedError as error:
            if runner_cls is spec.native_runner and "SwiGLUStep" in str(error):
                pytest.skip(
                    f"native {runner_cls.__name__} does not advertise SwiGLUStep "
                    f"(no native StepFun path in this tree): {error}"
                )
            raise
        runner.build()
        packed = runner.pack_inputs(act, weights)
        outputs[runner_cls.__name__] = _forward(
            runner, packed, runner.launch_kwargs_for(packed), -1
        )
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
    assert tactics and all(tactic[0] in cake_tiles for tactic in tactics), (
        f"every tactic of T={num_tokens} must run an exported Cake {precision} kernel: "
        f"tactics {tactics}, Cake tiles {sorted(cake_tiles)}"
    )
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
        [7.0 if expert % 2 == 0 else 16.0 for expert in range(NUM_EXPERTS)]
        + [16.0] * shared,
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
        assert torch.equal(replayed, eager), (
            f"tactic {tactic}: first replay differs from eager"
        )
        replayed.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replayed, eager), (
            f"tactic {tactic}: second replay differs from eager"
        )
        check_accuracy(case.reference, replayed.float(), **case.tolerances)


@pytest.mark.parametrize("precision,num_tokens", BITWISE_ROWS)
def test_stepfun_reproduces_native_fc1_twin_bitwise(
    precision, num_tokens, cache_permute_indices
):
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
    _, native, native_packed, native_kwargs, _ = _native_runner(case, device)
    cake_tiles = _cake_tiles(cake, spec.family)
    cake_space = _factorized(cake, cake_packed)
    checked = 0
    for tactic in cake_tactics:
        assert tactic[0] in cake_tiles, (tactic, sorted(cake_tiles))
        output = _forward(cake, cake_packed, cake_kwargs, tactic)
        check_accuracy(case.reference, output.float(), **case.tolerances)
        # The native arm is pinned to the twinned (FC1, FC2) configurations; a mismatch lists
        # the native tactics of the tile that do match, if any.
        twin = _native_twin_tactic(
            native, native_packed, cake_space, tactic, cake.full_path
        )
        twin_output = _forward(native, native_packed, native_kwargs, twin)
        if not torch.equal(output, twin_output):
            diff = (output.float() - twin_output.float()).abs().max().item()
            matches = _bitwise_matches(
                native, native_packed, native_kwargs, tactic[0], output
            )
            pytest.fail(
                f"{precision} T={num_tokens}: Cake tactic {tactic} differs from its native twin "
                f"{twin} (max |diff| {diff}); native (tactic, FC1, FC2) of tile {tactic[0]} "
                f"matching bitwise: {matches or 'none'}"
            )
        checked += 1
    assert checked, (
        f"no Cake tactic at T={num_tokens}: {cake_tactics} vs tiles {sorted(cake_tiles)}"
    )


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
    w1 = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    w1 /= math.sqrt(HIDDEN_SIZE)
    w2 = torch.randn(
        NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16
    )
    w2 /= math.sqrt(INTERMEDIATE_SIZE)
    hidden = HIDDEN_STATE_AMPLITUDE * torch.randn(
        num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    logits = torch.randn(num_tokens, NUM_EXPERTS, device=device, dtype=torch.bfloat16)
    global_scale = make_nvfp4_global_scale(
        hidden,
        per_token_activation=True,
        nvfp4_4over6_config=current_nvfp4_4over6_config(),
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
    weights.prepare_for("cake_stepfun", view)
    weights.prepare_for(TrtllmFp4RoutedRunner.backend_key, view)
    act = MoEActivationPack(
        hidden_states_q=quantized,
        hidden_states_scale=block_scales.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        ),
        per_token_scale=per_token_scale.contiguous(),
        routing_logits=logits,
        routing_input_mode=RoutingInputMode.FromLogits,
    )
    runners = {}
    for backend, runner_cls in (
        (CakeStepFunConfig(backend="cake_stepfun"), CakeStepFunNvfp4Runner),
        (TrtllmFp4Config(), TrtllmFp4RoutedRunner),
    ):
        config = _config(
            num_tokens=num_tokens,
            quant=NVFP4_PER_TOKEN,
            activation=activation,
            backend=BackendOptions(candidates=(backend,)),
        )
        runners[runner_cls] = _layer_runner(
            config,
            device,
            runner_cls,
            act,
            weights,
            native=runner_cls is TrtllmFp4RoutedRunner,
        )
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
        if cake.full_path:
            _assert_full_path_kernel_set(
                cake,
                cake_packed,
                cake_kwargs,
                tactic,
                family="nvfp4_bf16tok",
                num_tokens=num_tokens,
                routing_input="scores",
                logits_dtype="bfloat16",
                expert_weights_dtype="bfloat16",
                context=f"nvfp4_bf16tok T={num_tokens} tactic {tactic}",
            )
        checked += 1
    assert checked, (
        f"no per-token Cake tactic at T={num_tokens}: {cake_tactics} vs {sorted(per_token_tiles)}"
    )
    check_accuracy(
        reference, _forward(cake, cake_packed, cake_kwargs, -1).float(), **tolerances
    )


# ---------------------------------------------------------------------------------------------
# Module variants, inventory and full-path resolution (CPU)
# ---------------------------------------------------------------------------------------------


def test_inventory_v3_lists_a_stage_per_kernel():
    """Every inventory record names its pipeline stage; fc1 is present for every target."""
    from flashinfer.jit.cake_stepfun_moe import (
        CAKE_STEPFUN_STAGES,
        load_cake_stepfun_inventory,
    )

    inventory = load_cake_stepfun_inventory()
    records = json.loads(inventory.path.read_text(encoding="utf-8"))["kernels"]
    assert records and all(record["stage"] in CAKE_STEPFUN_STAGES for record in records)
    for target, stages in inventory.stages.items():
        if inventory.device_sources[target]:
            assert "fc1" in stages, (target, sorted(stages))
            assert set(inventory.missing_stages(target)).isdisjoint(stages)


def test_inventory_routing_and_finalize_records_are_typed():
    """Routing records name their input kind (and a listed pre-kernel unit, if any); finalize
    records their expert-weight dtype; the loader's accessors agree with the records."""
    from flashinfer.jit.cake_stepfun_moe import (
        CAKE_STEPFUN_FINALIZE_WEIGHT_DTYPES,
        CAKE_STEPFUN_ROUTING_INPUTS,
        cake_stepfun_finalize_weight_dtypes,
        cake_stepfun_routing_inputs,
        load_cake_stepfun_inventory,
    )

    inventory = load_cake_stepfun_inventory()
    raw = json.loads(inventory.path.read_text(encoding="utf-8"))
    inputs = {t: set() for t in inventory.stages}
    dtypes = {t: set() for t in inventory.stages}
    for record in raw["kernels"]:
        if record["stage"] == "routing":
            assert record["input"] in CAKE_STEPFUN_ROUTING_INPUTS, record[
                "kernel_symbol"
            ]
            inputs[record["arch"]].add(record["input"])
            pre = record.get("pre_kernel")
            if pre is not None:
                assert pre["device"] in raw["files"], pre
                assert (
                    inventory.repo_root / pre["device"]
                ).resolve() in inventory.device_sources[record["arch"]]
        elif record["stage"] == "finalize":
            assert (
                record["expert_weights_dtype"] in CAKE_STEPFUN_FINALIZE_WEIGHT_DTYPES
            ), record["kernel_symbol"]
            dtypes[record["arch"]].add(record["expert_weights_dtype"])
    for target in inventory.stages:
        assert cake_stepfun_routing_inputs(target) == frozenset(inputs[target])
        assert cake_stepfun_finalize_weight_dtypes(target) == frozenset(dtypes[target])


@pytest.mark.parametrize("target", ["sm_100a", "sm_103a"])
def test_full_path_resolution_names_missing_stages(target, monkeypatch):
    """Requesting the full path with a stage missing fails naming the stage; auto never does."""
    from flashinfer.jit.cake_stepfun_moe import (
        CAKE_STEPFUN_FULL_PATH_ENV,
        cake_stepfun_missing_stages,
        get_cake_stepfun_fused_moe_uri,
        resolve_cake_stepfun_full_path,
    )

    missing = cake_stepfun_missing_stages(target)
    monkeypatch.delenv(CAKE_STEPFUN_FULL_PATH_ENV, raising=False)
    assert resolve_cake_stepfun_full_path(target) is (not missing)
    assert resolve_cake_stepfun_full_path(target, False) is False
    monkeypatch.setenv(CAKE_STEPFUN_FULL_PATH_ENV, "0")
    assert resolve_cake_stepfun_full_path(target) is False
    monkeypatch.setenv(CAKE_STEPFUN_FULL_PATH_ENV, "1")
    if missing:
        with pytest.raises(ValueError) as excinfo:
            resolve_cake_stepfun_full_path(target)
        for stage in missing:
            assert stage in str(excinfo.value)
        with pytest.raises(ValueError):
            resolve_cake_stepfun_full_path(target, True)
    else:
        assert resolve_cake_stepfun_full_path(target) is True
    monkeypatch.setenv(CAKE_STEPFUN_FULL_PATH_ENV, "sometimes")
    with pytest.raises(ValueError, match="auto, 0 or 1"):
        resolve_cake_stepfun_full_path(target)
    suffix = "sm" + target[3:6]
    assert get_cake_stepfun_fused_moe_uri(target, False).endswith(suffix)
    assert get_cake_stepfun_fused_moe_uri(target, True).endswith("_full_" + suffix)


def test_module_build_matches_the_resolved_variant():
    """The loaded module reports the variant the loader resolved and the stages it serves."""
    device = _require_cake_device()
    limits = torch.full((NUM_EXPERTS,), 7.0, device="cuda", dtype=torch.float32)
    case = _build_case({}, precision="bf16", num_tokens=8, limits=limits)
    _, runner, _, _, _ = _cake_runner(case, device)
    moe_op = runner._module.moe_op
    assert bool(moe_op.cake_stepfun_full_path()) is runner.full_path
    stages = [str(stage) for stage in moe_op.cake_stepfun_stages()]
    expected = (
        ["routing", "fc1", "requant", "fc2", "finalize"]
        if runner.full_path
        else ["fc1"]
    )
    assert stages == expected
    if runner.full_path:
        from flashinfer.jit.cake_stepfun_moe import (
            cake_stepfun_finalize_weight_dtypes,
            cake_stepfun_routing_inputs,
        )

        target = _target(device)
        assert {str(k) for k in moe_op.cake_stepfun_routing_inputs()} == set(
            cake_stepfun_routing_inputs(target)
        )
        assert {str(d) for d in moe_op.cake_stepfun_finalize_weight_dtypes()} == set(
            cake_stepfun_finalize_weight_dtypes(target)
        )


# ---------------------------------------------------------------------------------------------
# Full Cake path (routing + FC1 + requantization + FC2 + finalize on Cake kernels)
# ---------------------------------------------------------------------------------------------


def _full_path_case(
    cache_permute_indices, precision, num_tokens, limit_mode, regime, input_set
):
    return _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limits=_limits(limit_mode),
        seed=input_set,
        logits=_routing_logits(num_tokens, regime, seed=ROUTING_SEED + input_set),
    )


@pytest.mark.parametrize("regime", ROUTING_REGIMES)
@pytest.mark.parametrize("limit_mode", LIMIT_MODES)
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_full_path_matches_native_pipeline(
    precision, num_tokens, limit_mode, regime, cache_permute_indices
):
    """Every Cake full-path tactic reproduces the native pipeline: bitwise where the FC1 twin
    is bitwise (some native tactic of the same tile), within the harness tolerance otherwise,
    on two input sets per (precision, T, limit, routing regime)."""
    device = _require_full_path()
    spec = PRECISIONS[precision]
    bitwise = (precision, num_tokens) in BITWISE_ROWS
    for input_set in (0, 1):
        case = _full_path_case(
            cache_permute_indices, precision, num_tokens, limit_mode, regime, input_set
        )
        _, cake, cake_packed, cake_kwargs, cake_tactics = _cake_runner(case, device)
        _, native, native_packed, native_kwargs, _ = _native_runner(case, device)
        assert cake.full_path
        cake_tiles = _cake_tiles(cake, spec.family)
        assert cake_tactics and all(t[0] in cake_tiles for t in cake_tactics)
        cake_space = _factorized(cake, cake_packed)
        twin_outputs = {}
        for tactic in cake_tactics:
            output = _forward(cake, cake_packed, cake_kwargs, tactic)
            check_accuracy(case.reference, output.float(), **case.tolerances)
            # The native arm runs the twinned (FC1, FC2) configurations of the tile.
            twin = tuple(
                _native_twin_tactic(native, native_packed, cake_space, tactic, True)
            )
            if twin not in twin_outputs:
                twin_outputs[twin] = _forward(
                    native, native_packed, native_kwargs, list(twin)
                )
            twin_output = twin_outputs[twin]
            if bitwise:
                assert torch.equal(output, twin_output), (
                    f"{precision} T={num_tokens} limit={limit_mode} {regime} set {input_set}: "
                    f"Cake tactic {tactic} differs from its native twin {list(twin)} (max "
                    f"|diff| {(output.float() - twin_output.float()).abs().max().item()})"
                )
            else:
                check_accuracy(twin_output.float(), output.float(), **case.tolerances)
        default = _forward(cake, cake_packed, cake_kwargs, -1)
        check_accuracy(case.reference, default.float(), **case.tolerances)


def _staged_routing(moe_op, case, tile: int, num_tokens: int, logits: torch.Tensor):
    """Routing tables of the module's staged bf16 routing op for one FC1 tile."""
    view = case.weights.get_view("cake_stepfun")
    hidden = case.act.hidden_states_q
    empty_ids = hidden.new_empty((0,), dtype=torch.int32)
    empty_weights = hidden.new_empty((0,), dtype=torch.bfloat16)
    out = moe_op.trtllm_moe_run_routing(
        logits,
        None,
        empty_ids,
        empty_weights,
        hidden,
        view["gemm1_weights"],
        view["gemm2_weights"],
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        INTERMEDIATE_SIZE,
        0,
        NUM_EXPERTS,
        None,
        int(RoutingMethodType.Renormalize),
        True,
        int(WeightLayout.BlockMajorK),
        True,
        # A config of -1 would make the launcher fall back to the default tile of the token
        # count; the staged routing op ignores the config itself, so 0 pins the tile.
        [tile, 0],
        int(ActivationType.Swiglu),
        True,
        None,
    )
    tensors = [t if isinstance(t, torch.Tensor) else torch.from_dlpack(t) for t in out]
    torch.cuda.synchronize()
    (
        expert_weights,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_token_idx,
        tile_idx,
        mn_limit,
        num_non_exiting_ctas,
        total_num_padded_tokens,
    ) = tensors[:7]
    num_ctas = int(num_non_exiting_ctas.item())
    num_padded = int(total_num_padded_tokens.item())
    return {
        "expert_weights": expert_weights.clone(),
        "expanded_idx_to_permuted_idx": expanded_idx_to_permuted_idx.clone(),
        "permuted_idx_to_token_idx": permuted_idx_to_token_idx[:num_padded].clone(),
        "cta_idx_xy_to_batch_idx": tile_idx[:num_ctas].clone(),
        "cta_idx_xy_to_mn_limit": mn_limit[:num_ctas].clone(),
        "num_non_exiting_ctas": num_non_exiting_ctas.clone(),
        "total_num_padded_tokens": total_num_padded_tokens.clone(),
        "tile": tile,
    }


# Routing tables the trtllm-gen router writes deterministically. The two permutation maps are
# assigned by atomic arrival order inside the cluster / cooperative kernels (T >= 17), so the
# native router is not bitwise reproducible against itself there; they are compared canonically.
_DETERMINISTIC_ROUTING_TABLES = (
    "expert_weights",
    "cta_idx_xy_to_batch_idx",
    "cta_idx_xy_to_mn_limit",
    "num_non_exiting_ctas",
    "total_num_padded_tokens",
)


def _assert_routing_tables_match(ours, theirs, num_tokens: int, context: str) -> None:
    for name in _DETERMINISTIC_ROUTING_TABLES:
        assert torch.equal(ours[name], theirs[name]), (
            f"{context}: {name} differs (ours {ours[name].flatten()[:8].tolist()} vs native "
            f"{theirs[name].flatten()[:8].tolist()})"
        )
    e2p_ours, e2p_theirs = (
        ours["expanded_idx_to_permuted_idx"],
        theirs["expanded_idx_to_permuted_idx"],
    )
    p2t_ours, p2t_theirs = (
        ours["permuted_idx_to_token_idx"],
        theirs["permuted_idx_to_token_idx"],
    )
    if num_tokens <= 16:
        assert torch.equal(e2p_ours, e2p_theirs), (
            f"{context}: expanded_idx_to_permuted_idx differs"
        )
        assert torch.equal(p2t_ours, p2t_theirs), (
            f"{context}: permuted_idx_to_token_idx differs"
        )
        return
    assert torch.equal(e2p_ours < 0, e2p_theirs < 0), (
        f"{context}: locality mask differs"
    )
    assert torch.equal(p2t_ours < 0, p2t_theirs < 0), f"{context}: padding mask differs"
    tile = ours["tile"]
    for label, e2p, p2t in (
        ("ours", e2p_ours, p2t_ours),
        ("native", e2p_theirs, p2t_theirs),
    ):
        valid = (e2p >= 0).nonzero().flatten()
        slots = e2p[valid].long()
        assert torch.equal(p2t[slots], (valid // TOP_K).to(p2t.dtype)), (
            f"{context}: {label} permutation maps are inconsistent"
        )
        assert slots.unique().numel() == slots.numel(), (
            f"{context}: {label} slots collide"
        )
    experts_ours = ours["cta_idx_xy_to_batch_idx"][
        e2p_ours[e2p_ours >= 0].long() // tile
    ]
    experts_theirs = theirs["cta_idx_xy_to_batch_idx"][
        e2p_theirs[e2p_theirs >= 0].long() // tile
    ]
    assert torch.equal(experts_ours, experts_theirs), (
        f"{context}: (token, k) -> expert assignment differs"
    )


@pytest.mark.parametrize("regime", ROUTING_REGIMES)
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_full_path_routing_tables_match_native(
    num_tokens, regime, cache_permute_indices
):
    """The Cake router writes the trtllm-gen routing tables byte for byte (every exported tile)."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    device = _require_full_path()
    limits = torch.full((NUM_EXPERTS,), 7.0, device="cuda", dtype=torch.float32)
    logits = _routing_logits(num_tokens, regime)
    case = _build_case(
        cache_permute_indices,
        precision="bf16",
        num_tokens=num_tokens,
        limits=limits,
        logits=logits,
    )
    _, cake, _, _, _ = _cake_runner(case, device)
    native_op = get_trtllm_moe_sm100_module().moe_op
    tiles = sorted(_cake_tiles(cake, "bf16"))
    for tile in tiles:
        ours = _staged_routing(cake._module.moe_op, case, tile, num_tokens, logits)
        theirs = _staged_routing(native_op, case, tile, num_tokens, logits)
        _assert_routing_tables_match(
            ours, theirs, num_tokens, f"T={num_tokens} {regime} tile {tile}"
        )


@pytest.mark.parametrize("num_tokens", [8, 512])
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_full_path_fc2_permuted_output_matches_native(
    precision, num_tokens, cache_permute_indices
):
    """With do_finalize=False the permuted FC2 output and the routing weights/map match native."""
    device = _require_full_path()
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limits=_limits("mixed"),
        logits=_routing_logits(num_tokens, "ragged"),
        do_finalize=False,
    )
    try:
        _, cake, cake_packed, cake_kwargs, cake_tactics = _cake_runner(case, device)
        _, native, native_packed, native_kwargs, _ = _native_runner(case, device)
    except NotImplementedError as error:
        if "finalize" in str(error).lower():
            pytest.skip(f"unfinalized output unsupported: {error}")
        raise
    cake_space = _factorized(cake, cake_packed)
    bitwise = (precision, num_tokens) in BITWISE_ROWS
    twin_results = {}
    for tactic in cake_tactics:
        gemm2_output, expert_weights, expanded = _unfinalized_forward(
            cake, cake_packed, cake_kwargs, tactic
        )
        twin = tuple(
            _native_twin_tactic(native, native_packed, cake_space, tactic, True)
        )
        if twin not in twin_results:
            twin_results[twin] = _unfinalized_forward(
                native, native_packed, native_kwargs, list(twin)
            )
        theirs_out, theirs_weights, theirs_expanded = twin_results[twin]
        # The expert weights and the locality mask are deterministic; the slot of a (token, k)
        # inside its expert segment is not (atomic arrival order), so rows are compared in
        # expanded-index order through each arm's own permutation map.
        assert torch.equal(expert_weights, theirs_weights)
        assert torch.equal(expanded >= 0, theirs_expanded >= 0)
        valid = (expanded >= 0).nonzero().flatten()
        ours = gemm2_output[expanded[valid].long()]
        theirs = theirs_out[theirs_expanded[valid].long()]
        if bitwise:
            assert torch.equal(ours, theirs), (
                f"{precision} T={num_tokens}: permuted FC2 rows of Cake tactic {tactic} differ "
                f"from its native twin {list(twin)}"
            )
        else:
            check_accuracy(theirs.float(), ours.float(), **case.tolerances)


@pytest.mark.parametrize("weights_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_full_path_precomputed_ids_matches_native_pipeline(
    precision, num_tokens, weights_dtype, cache_permute_indices
):
    """The benchmark protocol (unpacked pre-computed top-k ids + weights) on the full Cake path:
    the Cake router's table path and finalize reproduce the native pipeline pinned to the
    twinned configurations, with bf16 and fp32 expert weights."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    device = _require_full_path()
    _require_routing_input(device, "topk_ids")
    _require_finalize_weight_dtype(device, weights_dtype)
    bitwise = (precision, num_tokens) in BITWISE_ROWS
    logits = _routing_logits(num_tokens, "ragged")
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limits=_limits("mixed"),
        logits=logits,
    )
    topk_ids, topk_weights = _precomputed_routing(
        get_trtllm_moe_sm100_module().moe_op, case, num_tokens, logits
    )
    if weights_dtype == "float32":
        topk_weights = topk_weights.float()
    act = _precomputed_pack(case, topk_ids, topk_weights)
    spec = PRECISIONS[precision]
    _, cake, cake_packed, cake_kwargs, cake_tactics = _layer_runner(
        case.config, device, spec.runner_cls, act, case.weights
    )
    _, native, native_packed, native_kwargs, _ = _layer_runner(
        case.native_config, device, spec.native_runner, act, case.weights, native=True
    )
    assert cake.full_path
    cake_space = _factorized(cake, cake_packed)
    for tactic in cake_tactics:
        output = _forward(cake, cake_packed, cake_kwargs, tactic)
        # The pre-computed tables are the native router's own, so the reference still holds.
        check_accuracy(case.reference, output.float(), **case.tolerances)
        twin = _native_twin_tactic(native, native_packed, cake_space, tactic, True)
        twin_output = _forward(native, native_packed, native_kwargs, twin)
        if bitwise:
            assert torch.equal(output, twin_output), (
                f"{precision} T={num_tokens} {weights_dtype} weights: Cake tactic {tactic} "
                f"differs from its native twin {twin} (max |diff| "
                f"{(output.float() - twin_output.float()).abs().max().item()})"
            )
        else:
            check_accuracy(twin_output.float(), output.float(), **case.tolerances)
    replayed, _ = _capture_and_replay(cake, cake_packed, cake_kwargs, cake_tactics[0])
    assert torch.equal(
        replayed, _forward(cake, cake_packed, cake_kwargs, cake_tactics[0])
    )


@pytest.mark.parametrize("num_tokens", TOKENS)
def test_full_path_finalize_matches_native_kernel(num_tokens, cache_permute_indices):
    """The standalone Cake finalize equals the public finalize op (scalar and vector variants),
    for every exported expert-weight dtype."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.jit.cake_stepfun_moe import cake_stepfun_finalize_weight_dtypes

    device = _require_full_path()
    limits = torch.full((NUM_EXPERTS,), 7.0, device="cuda", dtype=torch.float32)
    logits = _routing_logits(num_tokens, "ragged")
    case = _build_case(
        cache_permute_indices,
        precision="bf16",
        num_tokens=num_tokens,
        limits=limits,
        logits=logits,
    )
    _, cake, _, _, _ = _cake_runner(case, device)
    native_op = get_trtllm_moe_sm100_module().moe_op
    tile = min(_cake_tiles(cake, "bf16"))
    tables = _staged_routing(native_op, case, tile, num_tokens, logits)
    max_padded = int(tables["total_num_padded_tokens"].item())
    expanded = tables["expanded_idx_to_permuted_idx"]
    expert_weights = tables["expert_weights"].reshape(num_tokens, TOP_K)
    torch.manual_seed(ROUTING_SEED + num_tokens)
    gemm2_output = torch.randn(
        max_padded, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    theirs = torch.full(
        (num_tokens, HIDDEN_SIZE), float("nan"), device=device, dtype=torch.bfloat16
    )
    native_op.trtllm_moe_run_finalize(
        gemm2_output,
        theirs,
        expert_weights,
        expanded,
        tables["total_num_padded_tokens"],
        num_tokens,
        NUM_EXPERTS,
        TOP_K,
        HIDDEN_SIZE,
        True,
        False,
    )
    torch.cuda.synchronize()
    assert not torch.isnan(theirs).any()
    checked = 0
    # fp32 expert weights holding the same bf16 values reduce to the same bytes (both kernels
    # accumulate in fp32), so the native bf16 result is the reference for both dtypes.
    for dtype_name in sorted(cake_stepfun_finalize_weight_dtypes(_target(device))):
        weights = expert_weights.float() if dtype_name == "float32" else expert_weights
        ours = torch.full_like(theirs, float("nan"))
        cake._module.moe_op.cake_stepfun_finalize(
            gemm2_output,
            weights,
            expanded,
            tables["total_num_padded_tokens"],
            ours,
            NUM_EXPERTS,
            True,
        )
        torch.cuda.synchronize()
        assert torch.equal(ours, theirs), (
            f"T={num_tokens} {dtype_name} expert weights: finalize differs, max |diff| "
            f"{(ours.float() - theirs.float()).abs().max().item()}"
        )
        checked += 1
    assert checked


@pytest.mark.parametrize("num_tokens", [8, 512])
def test_full_path_requant_matches_native_kernel(num_tokens, cache_permute_indices):
    """The standalone Cake NVFP4 per-token requantization equals the public per-token quant op
    in the block-scale layout the exported FC2 kernels read."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.quantization import SfLayout
    from flashinfer.quantization.fp4_quantization import get_fp4_quantization_module

    device = _require_full_path()
    if os.environ.get("FLASHINFER_NVFP4_4OVER6", "0") not in ("", "0"):
        pytest.skip("the comparison fixes the default NVFP4 recipe (448 * 6)")
    limits = torch.full((NUM_EXPERTS,), 7.0, device="cuda", dtype=torch.float32)
    logits = _routing_logits(num_tokens, "ragged")
    case = _build_case(
        cache_permute_indices,
        precision="bf16",
        num_tokens=num_tokens,
        limits=limits,
        logits=logits,
    )
    _, cake, _, _, _ = _cake_runner(case, device)
    moe_op = cake._module.moe_op
    native_op = get_trtllm_moe_sm100_module().moe_op
    quant = get_fp4_quantization_module("100")
    layouts = {
        "linear": SfLayout.layout_linear.value,
        "r8c4": SfLayout.layout_8x4.value,
        "r128c4": SfLayout.layout_128x4.value,
    }
    checked = 0
    for tile in sorted(int(t) for t in moe_op.cake_stepfun_fc2_tiles("nvfp4_bf16tok")):
        layout = str(
            moe_op.cake_stepfun_fc2_activation_sf_layout("nvfp4_bf16tok", tile)
        )
        tables = _staged_routing(native_op, case, tile, num_tokens, logits)
        expanded = tables["expanded_idx_to_permuted_idx"]
        max_padded = int(tables["total_num_padded_tokens"].item())
        torch.manual_seed(ROUTING_SEED + tile)
        gemm1_output = torch.randn(
            max_padded, INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16
        )
        scale_inv = 1.0 / (448.0 * 6.0)
        ref_out, ref_scale, ref_token = quant.nvfp4_quant_and_per_token_scale_sm100(
            gemm1_output, scale_inv, expanded, layouts[layout]
        )
        out = torch.zeros_like(ref_out)
        out_scale = torch.zeros_like(ref_scale)
        token_scale = torch.zeros_like(ref_token)
        moe_op.cake_stepfun_requant(
            gemm1_output, expanded, out, out_scale, token_scale, layout, True
        )
        torch.cuda.synchronize()
        valid = expanded[expanded >= 0].long()
        assert torch.equal(out[valid], ref_out[valid]), (
            f"tile {tile} {layout}: fp4 rows differ"
        )
        assert torch.equal(token_scale[valid], ref_token[valid]), (
            f"tile {tile} {layout}: per-token scales differ"
        )
        assert torch.equal(out_scale, ref_scale), (
            f"tile {tile} {layout}: block scales differ"
        )
        checked += 1
    assert checked


_CAKE_KERNEL_SYMBOL = re.compile(
    r"\b(kernel_cake_stepfun_moe_[0-9a-f]+|cake_stepfun_routing_tail_kernel)\b"
)


def _inventory_records(target: str) -> list[dict]:
    from flashinfer.jit.cake_stepfun_moe import load_cake_stepfun_inventory

    inventory = load_cake_stepfun_inventory()
    records = json.loads(inventory.path.read_text(encoding="utf-8"))["kernels"]
    return [record for record in records if record["arch"] == target]


def _expected_full_path_kernels(
    moe_op,
    target: str,
    *,
    family: str,
    tile: int,
    num_tokens: int,
    routing_input: str,
    logits_dtype: str,
    expert_weights_dtype: str,
) -> dict[str, list[str]]:
    """The exact Cake kernel set of one full-path forward, per stage, resolved from the
    inventory the way the host runners resolve it (``cake_stepfun_stages.cu``): the first
    routing record (table order) of the input kind / logits dtype whose token range covers
    ``num_tokens`` plus its leading histogram kernel, the ``(family, tile)`` FC1 and FC2 units,
    the requantization unit of the FC2 unit's activation scale layout (per-token family) and
    the finalize variant of the native dispatcher's CTA-count rule for the expert-weight dtype."""
    records = _inventory_records(target)

    def one(stage: str, **match) -> dict:
        found = [
            record
            for record in records
            if record["stage"] == stage
            and all(record.get(key) == value for key, value in match.items())
        ]
        assert len(found) == 1, (stage, match, [r["kernel_symbol"] for r in found])
        return found[0]

    routing = next(
        (
            record
            for record in records
            if record["stage"] == "routing"
            and record["input"] == routing_input
            and (routing_input != "scores" or record["logits_dtype"] == logits_dtype)
            and record["min_tokens"] <= num_tokens <= record["max_tokens"]
        ),
        None,
    )
    assert routing is not None, (target, routing_input, logits_dtype, num_tokens)
    expected = {"routing": []}
    if routing.get("pre_kernel"):
        expected["routing"].append(routing["pre_kernel"]["kernel_symbol"])
    expected["routing"].append(routing["kernel_symbol"])
    expected["fc1"] = [one("fc1", family=family, tile_n=tile)["kernel_symbol"]]
    if family == "nvfp4_bf16tok":
        layout = str(moe_op.cake_stepfun_fc2_activation_sf_layout(family, tile))
        expected["requant"] = [one("requant", sf_layout=layout)["kernel_symbol"]]
    expected["fc2"] = [one("fc2", family=family, tile_n=tile)["kernel_symbol"]]
    # Same variant rule as the native finalize dispatcher (and Cake's host): the scalar kernel
    # below 1184 CTAs of (ceil(H / 256) x min(8192, T)), the vector-load kernel otherwise.
    blocks = ((HIDDEN_SIZE - 1 + 256) // 256) * min(8192, num_tokens)
    variant = "scalar" if blocks < 1184 else "vector"
    expected["finalize"] = [
        one(
            "finalize",
            kernel_variant=variant,
            expert_weights_dtype=expert_weights_dtype,
        )["kernel_symbol"]
    ]
    return expected


def _profiled_kernel_names(runner, packed, kwargs, tactic) -> list[str]:
    """CUDA kernel names of one forward (memcpy / memset activities excluded)."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        runner.forward(packed, tactic=tactic, **kwargs)
        torch.cuda.synchronize()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith(("Memcpy", "Memset"))
    ]


def _assert_full_path_kernel_set(
    runner,
    packed,
    kwargs,
    tactic,
    *,
    family: str,
    num_tokens: int,
    routing_input: str,
    logits_dtype: str,
    expert_weights_dtype: str,
    context: str,
) -> dict[str, list[str]]:
    """One forward of ``tactic`` launches exactly the Cake kernels of its stages, each once:
    no trtllm-gen symbol and no routing-tail padding kernel (every exported FC1 / FC2 kernel
    consumes ``num_non_exiting_ctas`` like the native kernels, so the runners pad no tail)."""
    device = torch.device("cuda", torch.cuda.current_device())
    expected = _expected_full_path_kernels(
        runner._module.moe_op,
        _target(device),
        family=family,
        tile=int(tactic[0]),
        num_tokens=num_tokens,
        routing_input=routing_input,
        logits_dtype=logits_dtype,
        expert_weights_dtype=expert_weights_dtype,
    )
    names = _profiled_kernel_names(runner, packed, kwargs, tactic)
    assert names, f"{context}: the profiler captured no kernels"
    symbols: list[str] = []
    offenders: list[str] = []
    for name in names:
        match = _CAKE_KERNEL_SYMBOL.search(name)
        if match is None:
            offenders.append(name)
        else:
            symbols.append(match.group(1))
    assert not offenders, (
        f"{context}: non-Cake kernels in a Cake full-path forward: {sorted(set(offenders))}"
    )
    want = [symbol for stage in expected.values() for symbol in stage]
    assert Counter(symbols) == Counter(want), (
        f"{context}: kernel set {sorted(symbols)} != the expected per-stage set {expected} "
        "(an extra cake_stepfun_routing_tail_kernel means a runner padded the routing tail)"
    )
    return expected


@pytest.mark.parametrize("routing_input", ("scores", "topk_ids"))
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_full_path_launches_only_cake_kernels(
    precision, num_tokens, routing_input, cache_permute_indices
):
    """A full-path forward launches exactly the Cake kernels of its stages -- the routing
    kernel(s) of the token count and input kind, the FC1 unit of the tactic's tile, the FC2
    unit and the finalize variant -- and nothing else: no trtllm-gen routing, GEMM or finalize
    kernel and no routing-tail padding kernel, for every Cake tactic of every (precision, T)."""
    device = _require_full_path()
    spec = PRECISIONS[precision]
    logits = _routing_logits(num_tokens, "ragged")
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limits=_limits("mixed"),
        logits=logits,
    )
    if routing_input == "topk_ids":
        from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

        _require_routing_input(device, "topk_ids")
        topk_ids, topk_weights = _precomputed_routing(
            get_trtllm_moe_sm100_module().moe_op, case, num_tokens, logits
        )
        act = _precomputed_pack(case, topk_ids, topk_weights)
        _, runner, packed, kwargs, tactics = _layer_runner(
            case.config, device, spec.runner_cls, act, case.weights
        )
    else:
        _, runner, packed, kwargs, tactics = _cake_runner(case, device)
    assert runner.full_path
    cake_tiles = _cake_tiles(runner, spec.family)
    checked = []
    for tactic in tactics:
        assert tactic[0] in cake_tiles, (tactic, sorted(cake_tiles))
        _forward(
            runner, packed, kwargs, tactic
        )  # module load / lazy state before profiling
        checked.append(
            _assert_full_path_kernel_set(
                runner,
                packed,
                kwargs,
                tactic,
                family=spec.family,
                num_tokens=num_tokens,
                routing_input=routing_input,
                logits_dtype="bfloat16",
                expert_weights_dtype="bfloat16",
                context=f"{precision} T={num_tokens} {routing_input} tactic {tactic}",
            )
        )
    assert len(checked) == len(tactics)


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
    env = dict(
        os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
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
        marker = [
            line
            for line in proc.stdout.splitlines()
            if line.startswith("FIRST_REPLAY_RESULT ")
        ]
        assert proc.returncode == 0 and marker, (
            f"probe process {index} failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
        results = json.loads(marker[-1][len("FIRST_REPLAY_RESULT ") :])
        for key, result in results.items():
            checked += 1
            if not result["bitwise"]:
                failures.append((index, key, result))
    assert not failures, (
        f"{len(failures)}/{checked} first replays differ from eager: {failures}"
    )
    assert checked == processes * len(PRECISIONS) * 2
