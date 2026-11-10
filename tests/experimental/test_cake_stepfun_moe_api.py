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

Public-route tests of the Cake StepFun fused MoE: ``CakeStepFunConfig(
backend="cake_stepfun")`` through the unified MoE API (``flashinfer.fused_moe``)
runs routing, FC1 with the StepFun epilogue, the NVFP4 per-token
requantization, FC2 and finalize on the exported Cake kernels of
``flashinfer.experimental.cake_stepfun_moe`` for NVFP4 (E2m1 output and the
per-token bf16-output variant), BF16, per-tensor FP8 and MXFP8, on exact SM100
/ SM103.

The suite compares that route with the trtllm-gen fused-MoE test references
and with the native trtllm-gen pipeline on the StepFun serving geometry
(H=4096, I=1536, E=64, top-k 8; T in 8 / 64 / 512 / 2048; limit modes 7 / 16
/ mixed / 0.5 / inf; routing regimes uniform / ragged / randperm / empty), for
every plan the facade enumerates: the final bf16 output at the per-family
tolerances of ``tests/test_helpers/cake_stepfun_moe_reference.py`` (every
element, with the e4m3 / e2m1 one-ulp block rules; against the native default
tactic, or the native tactic that the plan reproduces bitwise, see
``_native_target``), the routing tables
bitwise / canonically, the unfinalized permuted FC2 rows, the precomputed-ids
protocol, the standalone finalize and requantization against the public native
ops, CUDA-graph replays, and the kernel set of a forward (Cake kernels only).
The last test writes the per-arch error table
``cake_stepfun_error_table.<arch>.json`` (schema ``cake_stepfun_error_table/1``)
into ``CAKE_STEPFUN_ERROR_TABLE_DIR``.

The numerical contract of the backend package alone (stage by stage against
the torch-only fp32 reference) lives in ``test_cake_stepfun_moe.py``; this
module needs an SM100 / SM103 device and skips as a whole elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer.experimental.cake_stepfun_moe import cake_backend as cb
from flashinfer.experimental.cake_stepfun_moe import cake_build
from flashinfer.experimental.cake_stepfun_moe import cake_jit as loader
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
from flashinfer.jit.env import has_flashinfer_jit_cache
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
    routing_reference_renormalize,
)
from tests.test_helpers import cake_stepfun_moe_reference as ref

H, I, E, K = ref.HIDDEN_SIZE, ref.INTERMEDIATE_SIZE, ref.NUM_EXPERTS, ref.TOP_K
PADDING = 8
TOKENS = ref.TOKENS
NVFP4 = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
NVFP4_PER_TOKEN = QuantConfig(
    weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4, per_token_scale=True
)
BF16 = QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16)
FP8 = QuantConfig(weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor)
MXFP8 = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
CAKE = BackendOptions(candidates=(CakeStepFunConfig(backend="cake_stepfun"),))
CAKE_KERNEL_SYMBOL = re.compile(r"\bkernel_cake_stepfun_moe_[0-9a-f]+\b")
ERROR_TABLE_DIR_ENV = "CAKE_STEPFUN_ERROR_TABLE_DIR"

# One entry per public precision: the quantization, the trtllm-gen test harness that produces
# the data and the reference, the weight layout the native kernels (and the Cake kernels)
# consume, the Cake family runner and the native runner the Cake route is compared with.
PRECISIONS = {
    "nvfp4": SimpleNamespace(
        quant=NVFP4,
        make_impl=lambda: FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
        layout=WeightLayout.MajorK,
        family="nvfp4",
        runner_cls=CakeStepFunNvfp4Runner,
        native_config=TrtllmFp4Config,
        native_runner=TrtllmFp4RoutedRunner,
    ),
    "bf16": SimpleNamespace(
        quant=BF16,
        make_impl=BF16Moe,
        layout=WeightLayout.BlockMajorK,
        family="bf16",
        runner_cls=CakeStepFunBf16Runner,
        native_config=TrtllmBf16Config,
        native_runner=TrtllmBf16RoutedRunner,
    ),
    "fp8": SimpleNamespace(
        quant=FP8,
        make_impl=FP8PerTensorMoe,
        layout=WeightLayout.MajorK,
        family="fp8",
        runner_cls=CakeStepFunFp8PerTensorRunner,
        native_config=TrtllmFp8PerTensorConfig,
        native_runner=TrtllmFp8PerTensorRunner,
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
    ),
}
# The per-tensor FP8 harness keeps the native plural key spelling for the scale scalars.
FP8_VIEW_KEYS = {
    "output1_scale_scalar": "output1_scales_scalar",
    "output1_scale_gate_scalar": "output1_scales_gate_scalar",
    "output2_scale_scalar": "output2_scales_scalar",
}
# Error-table rows collected by the comparison tests of this process; written by the last test.
ERROR_ROWS: list[dict] = []


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _arch_or_none():
    if not torch.cuda.is_available():
        return None
    return cb.SUPPORTED_COMPUTE_CAPABILITIES.get(get_compute_capability(_device()))


def _supported() -> bool:
    return _arch_or_none() is not None and bool(loader.PROGRAMS)


pytestmark = [
    pytest.mark.long_running,
    pytest.mark.skipif(
        not _supported(),
        reason="cake_stepfun_moe needs an SM100 / SM103 device and a filled registry",
    ),
]


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        flashinfer.jit.build_jit_specs(
            loader.specs_for_arch(_arch_or_none()), verbose=False
        )
    yield


@pytest.fixture(scope="module")
def cache_permute_indices():
    return {}


@pytest.fixture(autouse=True, scope="module")
def highest_matmul_precision():
    # The harness reference is torch fp32 on the device; pin the matmul precision so the error
    # table's reference column does not depend on the process' TF32 setting.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


def _arch() -> str:
    return _arch_or_none()


def _record(**row) -> dict:
    row.setdefault("arch", _arch())
    record = ref.error_row(**row)
    ERROR_ROWS.append(record)
    return record


def _sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------
# Case construction (trtllm-gen harness data, unified-API inputs)
# ---------------------------------------------------------------------------------------------


def _config(
    *,
    num_tokens: int,
    quant: QuantConfig = NVFP4,
    activation=None,
    backend=CAKE,
    enable_pdl=True,
    do_finalize=True,
) -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(
            num_experts=E, top_k=K, method=RoutingMethodType.Renormalize
        ),
        quant=quant,
        experts=ExpertConfig(intermediate_size=I, local_num_experts=E),
        activation=SwiGLUStep() if activation is None else activation,
        backend=backend,
        execution=ExecutionConfig(
            enable_pdl=enable_pdl, tune_max_num_tokens=num_tokens
        ),
        finalize=MoEFinalizeConfig(do_finalize=do_finalize),
    )


def _kernel_view(precision: str, static: dict, limits: torch.Tensor) -> dict:
    """The runner weight view from the harness's kernel-ready static data; ``gemm1_clamp_limit`` in
    the units the FC1 kernels clamp (logical limit over the FC1 gate dequant scale for NVFP4 and
    per-tensor FP8, physical units otherwise).  ``inf`` limits flow through unchanged."""
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
        experts = static["gemm1_weights"].shape[0]
        return {
            "gemm1_weights": static["gemm1_weights"],
            "gemm1_weights_scale": static["gemm1_scales"].reshape(
                experts, 2 * I, H // 32
            ),
            "gemm2_weights": static["gemm2_weights"],
            "gemm2_weights_scale": static["gemm2_scales"].reshape(experts, H, I // 32),
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
    precision: str,
    num_tokens: int,
    limit_mode: str = "7",
    seed: int = 0,
    logits: torch.Tensor | None = None,
    do_finalize: bool = True,
):
    """The trtllm-gen StepFun test data of ``precision`` (seed, amplitude 12), its fp32 reference and
    the unified-API inputs.  The per-expert limits of ``limit_mode`` enter the kernel view and the
    reference as data (``inf`` included); the config nominal is the finite ``SwiGLUStep`` limit
    (``FLT_MAX`` for ``inf``).  The view is registered for the Cake backend and for the native runner."""
    spec = PRECISIONS[precision]
    device = _device()
    limits = ref.limits_tensor(limit_mode, device)
    torch.random.manual_seed(seed)
    expert_logits = torch.randn((num_tokens, E), device=device).to(torch.bfloat16)
    if logits is not None:
        expert_logits = logits.to(device=device, dtype=torch.bfloat16)
    hidden_states = ref.HIDDEN_STATE_AMPLITUDE * torch.randn(
        (num_tokens, H), device=device, dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (E, 2 * I, H), device=device, dtype=torch.bfloat16
    ) / math.sqrt(H)
    gemm2_weights = torch.randn(
        (E, H, I), device=device, dtype=torch.bfloat16
    ) / math.sqrt(I)
    permute_info, scores = routing_reference_renormalize(expert_logits, K, E, PADDING)
    moe_impl = spec.make_impl()
    moe_impl._cache_permute_indices = cache_permute_indices
    weights_data = moe_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )
    hidden_states_scale_global = weights_data["hidden_states_scale_global"]
    inputs_data = moe_impl.quantize_inputs(hidden_states, hidden_states_scale_global)
    args = moe_args(
        num_tokens,
        E,
        H,
        I,
        K,
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
        H,
        I,
        E,
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
        routing_input_mode=RoutingInputMode.FromLogits,
    )
    common = dict(
        num_tokens=num_tokens,
        quant=spec.quant,
        activation=SwiGLUStep(limit=ref.limit_scalar(limit_mode)),
        do_finalize=do_finalize,
    )
    return SimpleNamespace(
        precision=precision,
        family=spec.family,
        num_tokens=num_tokens,
        limit_mode=limit_mode,
        limits=limits,
        logits=expert_logits,
        reference=reference.float(),
        tolerances=moe_impl.get_tolerances(),
        act=act,
        view=view,
        weights=weights,
        config=_config(**common),
        native_config=_config(
            backend=BackendOptions(candidates=(spec.native_config(),)), **common
        ),
    )


def _require_runner_support(runner_cls, config, device, *, unavailable: str) -> None:
    """Skip when ``runner_cls`` rejects ``config`` for ``unavailable`` (a path this tree does not
    provide); any other rejection is a failure."""
    try:
        runner_cls(config, device).check_support()
    except NotImplementedError as error:
        if unavailable in str(error):
            pytest.skip(
                f"{runner_cls.__name__} does not serve this configuration in this tree "
                f"({unavailable}): {error}"
            )
        raise


def _tactic_key(tactic):
    """A runner tactic as a hashable record: the Cake runner's tactic is the int plan tile
    (``get_valid_tactics`` lists ``plan.tile_n``, ``forward`` runs ``plan=<tile>``); the native
    trtllm runners' is the ``(tile_tokens_dim, config index)`` pair of the batched-GEMM tactic
    space, delivered as a tvm-ffi ``Array`` (a ``Sequence``, not a ``list``/``tuple``) and kept
    here as a tuple of ints.  Anything that is not a Python int is read as a sequence."""
    if isinstance(tactic, bool):
        raise TypeError(f"tactic must be an int or a sequence of ints, got {tactic!r}")
    if isinstance(tactic, int):
        return int(tactic)
    return tuple(int(v) for v in tactic)


def _tactic_tile(tactic) -> int:
    """The FC1 / FC2 tile a tactic names: the Cake int itself, a native pair's first entry."""
    key = _tactic_key(tactic)
    return key if isinstance(key, int) else int(key[0])


def _layer_runner(config, device, runner_cls, act, weights, *, native=False):
    if native:
        _require_runner_support(runner_cls, config, device, unavailable="SwiGLUStep")
    layer = MoELayer(config, device)
    assert len(layer.runners) == 1
    runner = layer.runners[0]
    assert isinstance(runner, runner_cls), type(runner)
    runner.check_support()
    runner.build()
    packed = runner.pack_inputs(act, weights)
    kwargs = runner.launch_kwargs_for(packed)
    tactics = [_tactic_key(tactic) for tactic in runner.get_valid_tactics(packed, None)]
    assert tactics, f"{runner_cls.__name__} enumerated no tactics"
    return layer, runner, packed, kwargs, tactics


def _cake_runner(case, device, act=None):
    return _layer_runner(
        case.config,
        device,
        PRECISIONS[case.precision].runner_cls,
        case.act if act is None else act,
        case.weights,
    )


def _native_runner(case, device, act=None):
    return _layer_runner(
        case.native_config,
        device,
        PRECISIONS[case.precision].native_runner,
        case.act if act is None else act,
        case.weights,
        native=True,
    )


def _forward(runner, packed, kwargs, tactic):
    output = runner.forward(packed, tactic=tactic, do_preparation=True, **kwargs)
    _sync()
    return output.clone()


def _unfinalized_forward(runner, packed, kwargs, tactic):
    """[gemm2_output (permuted rows), expert_weights, expanded_idx_to_permuted_idx] of one forward."""
    result = runner.forward(packed, tactic=tactic, do_preparation=True, **kwargs)
    _sync()
    assert isinstance(result, (list, tuple)) and len(result) >= 3, type(result)
    return [tensor.clone() for tensor in result[:3]]


def _capture_and_replay(runner, packed, kwargs, tactic):
    """Capture one forward in a CUDA graph (after a warm-up on the capture stream) and replay it once."""
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        runner.forward(packed, tactic=tactic, **kwargs)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            replayed = runner.forward(packed, tactic=tactic, **kwargs)
    _sync()
    replayed.zero_()
    graph.replay()
    _sync()
    return replayed, graph


def _assert_plan_tactics(case, tactics) -> None:
    """Every facade tactic (an int plan tile) names an exported plan tile of the family (the facade
    may narrow the ladder to the native autotune window, never widen it)."""
    tiles = {plan.tile_n for plan in cb.plans_for(case.family, case.num_tokens, H, I)}
    assert tactics and {_tactic_tile(t) for t in tactics} <= tiles, (
        tactics,
        sorted(tiles),
    )


def _closest_native_tactic(native, packed, kwargs, output):
    """Diagnostic: the native tactic whose output is closest (max-abs) to ``output``."""
    best = None
    for tactic in native.get_valid_tactics(packed, None):
        tactic = _tactic_key(tactic)
        diff = float(
            (output.float() - _forward(native, packed, kwargs, tactic).float())
            .abs()
            .max()
        )
        if best is None or diff < best[1]:
            best = (tactic, diff)
    return best


def _identical_native_tactic(native, packed, kwargs, output, tile: int):
    """The first native tactic whose output equals ``output`` bitwise, as ``(tactic, output)``:
    the tactics of FC1 / FC2 tile ``tile`` first, then the others (the token tile does not enter
    an element's reduction order, so a plan reproduces native tactics of other tiles as well);
    ``None`` when no native tactic reproduces it."""
    tactics = [_tactic_key(t) for t in native.get_valid_tactics(packed, None)]
    same = [t for t in tactics if _tactic_tile(t) == tile]
    other = [t for t in tactics if _tactic_tile(t) != tile]
    for tactic in same + other:
        candidate = _forward(native, packed, kwargs, tactic)
        if torch.equal(output, candidate):
            return tactic, candidate
    return None


def _native_target(case, native, packed, kwargs, native_output, output, tactic):
    """The native output a Cake plan is gated against: the native default tactic's, or --
    when that output is not within ``TOL[family]`` of the plan's and a native tactic (the
    plan's tile first) reproduces the plan bitwise -- that tactic's.  trtllm-gen's own tactics of
    one family differ by more than the tolerance at single elements (an fp32
    accumulation-order flip across an e4m3 / e2m1 block-scale edge), so the default's
    output is not the only native output, and a plan that equals one of them is as correct
    as the native pipeline.  Returns ``(target, name, extra error-row fields)``;
    ``native_default_max_abs`` records the spread between the two native outputs."""
    atol, rtol = ref.TOL[case.family]
    if not ref.errors(output, native_output, atol, rtol)["n_outside"]:
        return native_output, "default:-1", {}
    identical = _identical_native_tactic(
        native, packed, kwargs, output, _tactic_tile(tactic)
    )
    if identical is None:
        return native_output, "default:-1", {}
    native_tactic, target = identical
    spread = float((target.float() - native_output.float()).abs().max().item())
    return target, f"native:{native_tactic}", {"native_default_max_abs": spread}


def _compare_outputs(case, cake_output, native_output, *, label, **row):
    """Option-A gate of the final bf16 output (Cake vs native at ``TOL[family]``), the harness gate
    against the trtllm-gen reference, and the error-table row."""
    atol, rtol = ref.TOL[case.family]
    cake_vs_ref = ref.errors(cake_output, case.reference, atol, rtol)
    native_vs_ref = ref.errors(native_output, case.reference, atol, rtol)
    cake_vs_ref.pop("exceeds")
    native_vs_ref.pop("exceeds")
    try:
        check_accuracy(case.reference, cake_output.float(), **case.tolerances)
        harness = True
    except AssertionError:
        harness = False
    try:
        cake_vs_native = ref.assert_family_close(
            cake_output, native_output, case.family, label=label
        )
        passed = True
    except AssertionError as error:
        cake_vs_native = ref.errors(cake_output, native_output, atol, rtol)
        cake_vs_native.pop("exceeds")
        passed = False
        failure = error
    record = _record(
        stage="output",
        family=case.family,
        T=case.num_tokens,
        cake=cake_vs_ref,
        native=native_vs_ref,
        cake_vs_native=cake_vs_native,
        atol=atol,
        rtol=rtol,
        passed=passed,
        kind="graph",
        dtype=str(cake_output.dtype).replace("torch.", ""),
        numel=int(cake_output.numel()),
        n_outside_bf16_band=ref.bf16_band_count(cake_output, native_output),
        bitwise=bool(torch.equal(cake_output, native_output)),
        check_accuracy_passed=harness,
        **row,
    )
    if not passed:
        raise failure
    assert harness, f"{label}: check_accuracy against the trtllm-gen reference failed"
    return record


# ---------------------------------------------------------------------------------------------
# A1 -- config, registry, rejections
# ---------------------------------------------------------------------------------------------


def test_config_is_explicit_exact_sm100_sm103_and_stepfun_only():
    config = CakeStepFunConfig(backend="cake_stepfun")
    assert repr(config).startswith("CakeStepFunConfig(backend='cake_stepfun'")
    assert config == CakeStepFunConfig()
    assert config.full_path is None
    assert CakeStepFunConfig(full_path=True).full_path is True
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
        with pytest.raises(NotImplementedError, match="has no family"):
            CakeStepFunRunner.runner_class_for(unsupported)
    for spec in PRECISIONS.values():
        runner = CakeStepFunRunner(
            _config(num_tokens=8, quant=spec.quant), torch.device("cpu")
        )
        assert type(runner) is spec.runner_cls
    assert set(CakeStepFunRunner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.UnpackedPrecomputed,
    }
    assert not hasattr(CakeStepFunRunner, "native_fc2_twin")


@pytest.mark.parametrize("precision", list(PRECISIONS))
@pytest.mark.parametrize(
    "activation", [SwiGLUStep(), SwiGLU()], ids=["swiglu_step", "swiglu"]
)
def test_fused_shared_experts_are_rejected_by_every_family(precision, activation):
    """The Cake router serves routed experts only: no family accepts fused shared experts.

    The rejection names the shared experts whatever activation the configuration
    carries: ``tests/moe/test_unified_moe.py`` checks every non-supporting backend
    with the default ``SwiGLU``, which the Cake family does not serve either."""
    spec = PRECISIONS[precision]
    assert spec.runner_cls.supports_fused_shared_experts is False
    assert CakeStepFunRunner.supports_fused_shared_experts is False
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=E,
            top_k=K,
            method=RoutingMethodType.DeepSeekV3,
            n_group=8,
            topk_group=4,
            routed_scaling_factor=2.5,
        ),
        quant=spec.quant,
        experts=ExpertConfig(
            intermediate_size=I, local_num_experts=E, num_fused_shared_experts=1
        ),
        activation=activation,
        backend=CAKE,
        execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=8),
    )
    runner = CakeStepFunRunner(config, torch.device("cpu"))
    assert type(runner) is spec.runner_cls
    with pytest.raises(
        NotImplementedError, match="does not support fused shared experts"
    ):
        runner.check_support()


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_per_token_scale_follows_the_native_family(precision):
    """Only nvfp4 consumes the pack's per-token scale, like its trtllm-gen runner."""
    spec = PRECISIONS[precision]
    expected = spec.runner_cls is CakeStepFunNvfp4Runner
    assert spec.native_runner.supports_per_token_scale is expected
    assert spec.runner_cls.supports_per_token_scale is expected
    assert CakeStepFunRunner.supports_per_token_scale is True
    per_token = QuantConfig(
        weight=spec.quant.weight, activation=spec.quant.activation, per_token_scale=True
    )
    runner = CakeStepFunRunner(
        _config(num_tokens=8, quant=per_token), torch.device("cpu")
    )
    assert type(runner) is spec.runner_cls
    if expected:
        runner._assert_per_token_scale_supported()
    else:
        with pytest.raises(NotImplementedError, match="per_token_scale=True"):
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
    w1 = torch.zeros(2, 2 * I, H, dtype=torch.bfloat16)
    w2 = torch.zeros(2, H, I, dtype=torch.bfloat16)
    common = dict(num_local_experts=2, hidden_size=H, intermediate_size=I)
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


# ---------------------------------------------------------------------------------------------
# A2 -- environment knobs are resolved by the facade, never by the package
# ---------------------------------------------------------------------------------------------


def test_full_path_environment_is_resolved_by_the_facade(monkeypatch):
    """``FLASHINFER_CAKE_STEPFUN_FULL_PATH`` is read by ``flashinfer.fused_moe.api``: unset, blank,
    ``auto`` and ``1`` resolve to the full Cake path, ``0`` raises ``NotImplementedError`` (the
    FC1-only hybrid no longer exists), any other value ``ValueError`` naming the variable; an
    explicit ``CakeStepFunConfig(full_path=...)`` wins over the variable; the reserved-SM count and
    the NVFP4 recipe resolve next to it; the package itself reads no environment variable."""
    from flashinfer.fused_moe import api as facade

    env = facade.CAKE_STEPFUN_FULL_PATH_ENV
    assert env == "FLASHINFER_CAKE_STEPFUN_FULL_PATH"
    resolve = facade._resolve_cake_stepfun_knobs
    config = _config(num_tokens=8, quant=BF16)
    monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("FLASHINFER_TRTLLM_MOE_OVERLAP_RESERVED_SMS", raising=False)
    knobs = resolve(config)
    assert set(knobs) == {"full_path", "reserved_sms", "global_scale_inv"}
    assert knobs["full_path"] is True and knobs["reserved_sms"] == 8
    assert knobs["global_scale_inv"] == pytest.approx(ref.GLOBAL_SCALE_INV)
    for value in ("", " ", "auto", "1", " 1 "):
        monkeypatch.setenv(env, value)
        assert resolve(config)["full_path"] is True, value
    monkeypatch.setenv(env, "0")
    with pytest.raises(NotImplementedError):
        resolve(config)
    monkeypatch.setenv(env, "sometimes")
    with pytest.raises(ValueError, match=env):
        resolve(config)
    explicit = _config(
        num_tokens=8,
        quant=BF16,
        backend=BackendOptions(candidates=(CakeStepFunConfig(full_path=True),)),
    )
    assert resolve(explicit)["full_path"] is True
    monkeypatch.delenv(env)
    disabled = _config(
        num_tokens=8,
        quant=BF16,
        backend=BackendOptions(candidates=(CakeStepFunConfig(full_path=False),)),
    )
    with pytest.raises(NotImplementedError):
        resolve(disabled)
    monkeypatch.setenv("FLASHINFER_TRTLLM_MOE_OVERLAP_RESERVED_SMS", "4")
    assert resolve(config)["reserved_sms"] == 4
    package_dir = Path(cb.__file__).resolve().parent
    for name in ("__init__.py", "cake_backend.py", "cake_jit.py", "cake_launch.py"):
        text = (package_dir / name).read_text()
        assert "os.environ" not in text and "getenv" not in text, name
        assert "flashinfer.fused_moe" not in text or name == "__init__.py", name


# ---------------------------------------------------------------------------------------------
# A3 -- prepare_weights, the limit encodings, Cake vs native on the public view
# ---------------------------------------------------------------------------------------------


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
def test_prepare_weights_adds_step_limits_and_matches_the_native_route(precision):
    """The shared view carries the limits in kernel units; ``FLT_MAX`` (the Cake no-clamp value) is
    accepted, ``inf`` is rejected at the two validated public sites; the Cake route agrees with the
    native route on the view (elementwise tolerances and the harness gate)."""
    device = _device()
    spec = PRECISIONS[precision]
    torch.manual_seed(0)
    num_tokens = 8
    w1 = torch.randn(E, 2 * I, H, device=device, dtype=torch.bfloat16) / math.sqrt(H)
    w2 = torch.randn(E, H, I, device=device, dtype=torch.bfloat16) / math.sqrt(I)
    hidden = 12.0 * torch.randn(num_tokens, H, device=device, dtype=torch.bfloat16)
    step_limits = ref.limits_tensor("mixed", device)
    common = dict(
        quant=spec.quant,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
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
    default_expected = torch.full((E,), 7.0, device=device)
    if gate_key:
        default_expected = default_expected / default_view[gate_key]
    assert torch.equal(default_view["gemm1_clamp_limit"], default_expected)
    with pytest.raises(ValueError, match="one limit per physical expert row"):
        CakeStepFunConfig.prepare_weights(
            w1, w2, step_limits=step_limits[: E // 2], **common
        )
    # Encodings: the PUBLIC step_limits must be finite and positive (FLT_MAX spells "no clamp");
    # the kernel-unit value ``limits / gate_scale`` may overflow to +inf (fp8: gate scale < 1),
    # which the kernels treat as no clamp (the backend suite's L=inf rows encode it the same way).
    no_clamp = CakeStepFunConfig.prepare_weights(
        w1,
        w2,
        activation=SwiGLUStep(limit=ref.FLT_MAX),
        step_limits=ref.public_step_limits("inf", device),
        **common,
    )
    no_clamp_expected = ref.public_step_limits("inf", device)
    if gate_key:
        no_clamp_expected = no_clamp_expected / no_clamp[gate_key]
    assert torch.equal(no_clamp["gemm1_clamp_limit"], no_clamp_expected)
    with pytest.raises(ValueError, match="finite"):
        CakeStepFunConfig.prepare_weights(
            w1, w2, step_limits=torch.full((E,), float("inf"), device=device), **common
        )
    with pytest.raises(ValueError, match="finite"):
        SwiGLUStep(limit=float("inf"))
    assert SwiGLUStep(limit=ref.FLT_MAX).limit == ref.FLT_MAX
    assert ref.limit_scalar("inf") == ref.FLT_MAX and ref.limit_scalar("0.5") == 0.5

    quantized, scales = CakeStepFunConfig.prepare_activations(
        hidden, quant=spec.quant, **activation_kwargs
    )
    logits = torch.randn(num_tokens, E, device=device, dtype=torch.bfloat16)
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
        config = _config(
            num_tokens=num_tokens,
            quant=spec.quant,
            activation=SwiGLUStep(limit=7.0),
            backend=BackendOptions(candidates=(backend,)),
        )
        if runner_cls is spec.native_runner:
            _require_runner_support(
                runner_cls, config, device, unavailable="SwiGLUStep"
            )
        runner = runner_cls(config, device)
        runner.check_support()
        runner.build()
        packed = runner.pack_inputs(act, weights)
        outputs[runner_cls.__name__] = _forward(
            runner, packed, runner.launch_kwargs_for(packed), -1
        )
    cake, native = (
        outputs[spec.runner_cls.__name__],
        outputs[spec.native_runner.__name__],
    )
    check_accuracy(native.float(), cake.float(), **spec.make_impl().get_tolerances())
    ref.assert_family_close(
        cake, native, spec.family, label=f"{precision} prepare_weights view"
    )


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_reference_layouts_match_prepare_weights(precision):
    """The torch-only reference builds the same kernel view as ``CakeStepFunConfig.prepare_weights``
    on the same canonical experts: bf16 byte for byte (row shuffles and BlockMajorK), the
    quantized families in shape and dtype (their quantizers are the reference's own)."""
    device = _device()
    spec = PRECISIONS[precision]
    w1, w2 = ref.canonical_weights(0, device)
    common = dict(
        quant=spec.quant,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
        device=device,
    )
    if precision == "fp8":
        hidden = ref.canonical_hidden(8, 0, device)
        hidden_global, intermediate_global = _fp8_global_scales(hidden, 7.0, device)
        common.update(
            hidden_states_scale_global=hidden_global,
            intermediate_scale_global=intermediate_global,
        )
    public = CakeStepFunConfig.prepare_weights(
        w1, w2, activation=SwiGLUStep(limit=7.0), **common
    )
    helper = ref.build_weights(spec.family, 0, device).view
    for key, tensor in helper.items():
        public_key = FP8_VIEW_KEYS.get(key, key) if precision == "fp8" else key
        assert public_key in public, key
        assert tuple(public[public_key].shape) == tuple(tensor.shape), key
        assert public[public_key].dtype == tensor.dtype, key
        if precision == "bf16":
            assert torch.equal(public[public_key], tensor), key


# ---------------------------------------------------------------------------------------------
# A4 -- the Cake route vs the trtllm-gen reference for every plan
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("limit_mode", ("7", "16", "0.5", "inf"))
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_matches_trtllm_reference_for_every_plan(
    precision, num_tokens, limit_mode, cache_permute_indices
):
    """Every facade tactic (plan) and the ``MoELayer`` call reproduce the trtllm-gen fp32 reference
    at the harness tolerances and at ``TOL[family]`` (information rows of the error table)."""
    device = _device()
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limit_mode=limit_mode,
    )
    layer, runner, packed, kwargs, tactics = _cake_runner(case, device)
    _assert_plan_tactics(case, tactics)
    atol, rtol = ref.TOL[case.family]
    for tactic in tactics:
        output = _forward(runner, packed, kwargs, tactic)
        check_accuracy(case.reference, output.float(), **case.tolerances)
        record = ref.errors(output, case.reference, atol, rtol)
        record.pop("exceeds")
        _record(
            stage="output",
            family=case.family,
            T=num_tokens,
            limit_mode=limit_mode,
            regime="uniform",
            input_set=0,
            status="info",
            cake=record,
            atol=atol,
            rtol=rtol,
            kind="graph_vs_reference",
            plan=_tactic_tile(tactic),
            check_accuracy_passed=True,
        )
    output = layer(case.act, case.weights)
    _sync()
    check_accuracy(case.reference, output.float(), **case.tolerances)


# ---------------------------------------------------------------------------------------------
# A5 -- the full Cake path vs the native pipeline
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("limit_mode,regime", ref.API_GRAPH_CELLS)
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_full_path_matches_native_pipeline(
    precision, num_tokens, limit_mode, regime, cache_permute_indices
):
    """Every Cake plan reproduces the native pipeline within ``TOL[family]`` on the final bf16
    output (the native default tactic, or the native tactic it equals bitwise:
    ``_native_target``), on two input sets per (precision, T, limit mode, regime);
    the harness gate holds for both arms; whole-graph error rows are recorded."""
    device = _device()
    for input_set in ref.INPUT_SETS:
        case = _build_case(
            cache_permute_indices,
            precision=precision,
            num_tokens=num_tokens,
            limit_mode=limit_mode,
            seed=input_set,
            logits=ref.routing_logits(
                num_tokens, regime, seed=ref.ROUTING_SEED + input_set, device=device
            ),
        )
        _, cake, cake_packed, cake_kwargs, tactics = _cake_runner(case, device)
        _assert_plan_tactics(case, tactics)
        outputs = {t: _forward(cake, cake_packed, cake_kwargs, t) for t in tactics}
        default = _forward(cake, cake_packed, cake_kwargs, -1)
        default_tactic = _tactic_key(cake.default_tactic(cake_packed))
        assert torch.equal(default, outputs[default_tactic]), (
            "tactic -1 does not run the default plan"
        )
        _, native, native_packed, native_kwargs, _ = _native_runner(case, device)
        native_output = _forward(native, native_packed, native_kwargs, -1)
        check_accuracy(case.reference, native_output.float(), **case.tolerances)
        for tactic, output in outputs.items():
            label = f"{precision} T={num_tokens} L={limit_mode} {regime} set {input_set} plan {tactic}"
            target, target_name, extra = _native_target(
                case,
                native,
                native_packed,
                native_kwargs,
                native_output,
                output,
                tactic,
            )
            try:
                _compare_outputs(
                    case,
                    output,
                    target,
                    label=label,
                    limit_mode=limit_mode,
                    regime=regime,
                    input_set=input_set,
                    plan=_tactic_tile(tactic),
                    native_tactic=target_name,
                    **extra,
                )
            except AssertionError as error:
                closest = _closest_native_tactic(
                    native, native_packed, native_kwargs, output
                )
                pytest.fail(f"{error}\nclosest native tactic by max-abs: {closest}")


# ---------------------------------------------------------------------------------------------
# A6 -- routing tables vs the native router
# ---------------------------------------------------------------------------------------------

# Staged routing op of each family (``csrc/trtllm_fused_moe_kernel_launcher.cu``).
_STAGED_ROUTING_OPS = {
    "bf16": "trtllm_moe_run_routing",
    "fp8": "trtllm_moe_run_routing_fp8_per_tensor",
    "nvfp4": "trtllm_moe_run_routing_fp4_nvfp4",
    "mxfp8": "trtllm_moe_run_routing_fp8_block_scale",
}


def _staged_routing_args(
    precision: str, inner, packed, tile: int, logits: torch.Tensor
):
    """``(op name, positional arguments)`` of the family's staged routing op for one FC1 tile, from
    the native runner's own packed call (``MoeRunnerInputs`` list plus static launch kwargs) and
    the trtllm-gen ``MoERunner`` it drives.  ``[tile, 0]`` pins the tile."""
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.prims_ts.moe.runner import _staged_routing_io

    moe_inputs = MoeRunnerInputs.from_list(list(packed))
    kw = dict(packed.launch_state.static_kwargs)
    topk_ids, expert_weights_in = _staged_routing_io(moe_inputs, kw)
    head = (
        logits,
        kw["routing_bias"],
        topk_ids,
        expert_weights_in,
        moe_inputs.hidden_states,
    )
    experts = (
        kw["num_experts"],
        inner.top_k,
        kw["n_group"],
        kw["topk_group"],
        inner.intermediate_size,
        kw["local_expert_offset"],
        inner.num_local_experts,
        kw["routed_scaling_factor"],
        kw["routing_method_type"],
    )
    tactic = [tile, 0]
    layout = int(inner.weight_layout)
    activation = int(inner.activation_type)
    norm_topk_prob = kw.get("norm_topk_prob", True)
    replay = kw.get("routing_replay_out")
    if precision == "bf16":
        args = (
            *head,
            kw["gemm1_weights"],
            kw["gemm2_weights"],
            *experts,
            bool(inner.use_shuffled_weight),
            layout,
            kw["enable_pdl"],
            tactic,
            activation,
            norm_topk_prob,
            replay,
        )
    elif precision == "fp8":
        args = (
            *head,
            kw["gemm1_weights"],
            kw["output1_scales_scalar"],
            kw["output1_scales_gate_scalar"],
            kw["gemm2_weights"],
            kw["output2_scales_scalar"],
            *experts,
            kw["use_routing_scales_on_input"],
            kw["enable_pdl"],
            tactic,
            layout,
            activation,
            norm_topk_prob,
            replay,
        )
    elif precision == "nvfp4":
        args = (
            *head,
            moe_inputs.hidden_states_scale,
            kw["gemm1_weights"],
            kw["gemm1_weights_scale"],
            kw["gemm2_weights"],
            kw["gemm2_weights_scale"],
            kw["output1_scale_scalar"],
            kw["output1_scale_gate_scalar"],
            kw["output2_scale_scalar"],
            *experts,
            kw["enable_pdl"],
            tactic,
            layout,
            activation,
            norm_topk_prob,
            replay,
        )
    elif precision == "mxfp8":
        args = (
            *head,
            moe_inputs.hidden_states_scale,
            kw["gemm1_weights"],
            kw["gemm1_weights_scale"],
            kw["gemm2_weights"],
            kw["gemm2_weights_scale"],
            *experts,
            kw["enable_pdl"],
            tactic,
            layout,
            activation,
            int(inner.fp8_quantization_type),
            norm_topk_prob,
            replay,
        )
    else:
        raise KeyError(precision)
    return _STAGED_ROUTING_OPS[precision], args


def _native_staged_routing(case, device, tile: int, logits: torch.Tensor):
    """Routing tables of the public native module's staged routing op of the case's family for one
    FC1 tile (arguments from the native runner's packed call)."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    _, native, packed, _, _ = _native_runner(case, device)
    name, args = _staged_routing_args(
        case.precision, native._inner, packed, tile, logits
    )
    out = getattr(get_trtllm_moe_sm100_module().moe_op, name)(*args)
    tensors = [t if isinstance(t, torch.Tensor) else torch.from_dlpack(t) for t in out]
    _sync()
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
        "topk_weights": expert_weights.clone(),
        "expanded_idx_to_permuted_idx": expanded_idx_to_permuted_idx.clone(),
        "permuted_idx_to_token_idx": permuted_idx_to_token_idx[:num_padded].clone(),
        "cta_idx_xy_to_batch_idx": tile_idx[:num_ctas].clone(),
        "cta_idx_xy_to_mn_limit": mn_limit[:num_ctas].clone(),
        "num_non_exiting_ctas": num_ctas,
        "total_num_padded_tokens": num_padded,
        "tile": tile,
    }


def _cake_routing(logits: torch.Tensor, num_tokens: int, tile: int, device):
    buffers = ref.routing_buffers(num_tokens, tile, device)
    cb.prepare_routing(
        routing_logits=logits,
        num_experts=E,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=E,
        tile_tokens_dim=tile,
        **buffers,
    ).launch()
    _sync()
    return ref.live_tables(buffers, tile=tile)


@pytest.mark.parametrize("regime", ref.ROUTING_REGIMES)
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_full_path_routing_tables_match_native(
    num_tokens, regime, cache_permute_indices
):
    """The Cake router writes the trtllm-gen routing tables byte for byte (permutation maps
    canonically above 16 tokens) for every exported bf16 FC1 tile; the ``empty`` regime leaves
    experts 13 and 41 without a CTA on both arms."""
    device = _device()
    logits = ref.routing_logits(num_tokens, regime, device=device)
    case = _build_case(
        cache_permute_indices, precision="bf16", num_tokens=num_tokens, logits=logits
    )
    for tile in cb.exported_tiles("fc1", "bf16"):
        ours = _cake_routing(logits, num_tokens, tile, device)
        theirs = _native_staged_routing(case, device, tile, logits)
        context = f"T={num_tokens} {regime} tile {tile}"
        bitwise = ref.assert_routing_tables_match(
            ours, theirs, num_tokens=num_tokens, tile=tile, top_k=K, context=context
        )
        if regime == "empty":
            ref.assert_empty_experts(ours, tile=tile, context=context)
            ref.assert_empty_experts(theirs, tile=tile, context=context + " (native)")
        _record(
            stage="routing",
            family="bf16",
            T=num_tokens,
            limit_mode="7",
            regime=regime,
            input_set=0,
            passed=True,
            kind="stage",
            plan=tile,
            bitwise=bitwise,
        )


# ---------------------------------------------------------------------------------------------
# A7 -- unfinalized (permuted FC2) output vs native
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,limit_mode,regime,input_set", ref.API_FC2_PERMUTED_CELLS
)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_full_path_fc2_permuted_output_matches_native(
    precision, num_tokens, limit_mode, regime, input_set, cache_permute_indices
):
    """With ``do_finalize=False`` the permuted FC2 output, the routing weights and the map match
    native: ``expert_weights`` and the locality mask bitwise, the rows gathered in expanded order
    at ``TOL[family]``, ``-1`` slots never gathered, and ``gemm2_output`` sized by the native
    ``maybeGetMinTokenCount`` rule."""
    device = _device()
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limit_mode=limit_mode,
        seed=input_set,
        logits=ref.routing_logits(
            num_tokens, regime, seed=ref.ROUTING_SEED + input_set, device=device
        ),
        do_finalize=False,
    )
    _, cake, cake_packed, cake_kwargs, tactics = _cake_runner(case, device)
    _, native, native_packed, native_kwargs, native_tactics = _native_runner(
        case, device
    )
    atol, rtol = ref.TOL[case.family]
    for tactic in tactics:
        tile = _tactic_tile(tactic)
        gemm2_output, expert_weights, expanded = _unfinalized_forward(
            cake, cake_packed, cake_kwargs, tactic
        )
        same_tile = [t for t in native_tactics if _tactic_tile(t) == tile]
        native_tactic = same_tile[0] if same_tile else -1
        theirs_out, theirs_weights, theirs_expanded = _unfinalized_forward(
            native, native_packed, native_kwargs, native_tactic
        )
        max_padded = ref.max_padded_tokens(num_tokens, K, E, tile)
        assert gemm2_output.shape == (ref.gemm2_output_rows(max_padded, H), H), (
            gemm2_output.shape
        )
        if same_tile:
            assert gemm2_output.shape == theirs_out.shape
        assert torch.equal(expert_weights, theirs_weights)
        assert torch.equal(expanded >= 0, theirs_expanded >= 0)
        valid = (expanded >= 0).nonzero().flatten()
        ours = gemm2_output[expanded[valid].long()]
        theirs = theirs_out[theirs_expanded[valid].long()]
        label = f"{precision} T={num_tokens} L={limit_mode} {regime} set {input_set} plan {tactic}"
        record = ref.assert_family_close(ours, theirs, case.family, label=label)
        if regime == "empty":
            tables = {
                "num_non_exiting_ctas": max_padded // tile,
                "cta_idx_xy_to_batch_idx": torch.empty(
                    0, dtype=torch.int32, device=device
                ),
                "permuted_idx_to_token_idx": torch.full(
                    (max_padded,), -1, dtype=torch.int32, device=device
                ),
                "expanded_idx_to_permuted_idx": expanded,
            }
            assert int(valid.numel()) == num_tokens * K
            assert tables["num_non_exiting_ctas"] * tile == max_padded
        _record(
            stage="fc2_permuted",
            family=case.family,
            T=num_tokens,
            limit_mode=limit_mode,
            regime=regime,
            input_set=input_set,
            cake_vs_native=record,
            atol=atol,
            rtol=rtol,
            passed=True,
            kind="stage",
            plan=tile,
            native_tactic=native_tactic if native_tactic == -1 else list(native_tactic),
            dtype="bfloat16",
            numel=int(ours.numel()),
            bitwise=bool(torch.equal(ours, theirs)),
            n_outside_bf16_band=ref.bf16_band_count(ours, theirs),
        )


# ---------------------------------------------------------------------------------------------
# A8 -- the benchmark protocol (precomputed ids + weights)
# ---------------------------------------------------------------------------------------------


def _precomputed_routing(case, device, num_tokens: int, logits: torch.Tensor):
    """(topk_ids int32 [T, K], topk_weights bf16 [T, K]) of the native router for ``logits``."""
    tables = _native_staged_routing(case, device, PADDING, logits)
    expanded = tables["expanded_idx_to_permuted_idx"]
    assert bool((expanded >= 0).all()), "every expert is local in this configuration"
    ids = tables["cta_idx_xy_to_batch_idx"][expanded.long() // PADDING]
    return (
        ids.reshape(num_tokens, K).to(torch.int32).contiguous(),
        tables["expert_weights"].reshape(num_tokens, K).contiguous(),
    )


def _precomputed_pack(case, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    return MoEActivationPack(
        hidden_states_q=case.act.hidden_states_q,
        hidden_states_scale=case.act.hidden_states_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


@pytest.mark.parametrize("weights_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_full_path_precomputed_ids_matches_native_pipeline(
    precision, num_tokens, weights_dtype, cache_permute_indices
):
    """Unpacked pre-computed top-k ids + weights (the benchmark protocol) on the full Cake path: the
    Cake router's table path and finalize reproduce the native pipeline within ``TOL[family]``
    (``_native_target``) with bf16 and fp32 expert weights; a CUDA-graph replay equals eager."""
    device = _device()
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limit_mode="mixed",
        logits=logits,
    )
    topk_ids, topk_weights = _precomputed_routing(case, device, num_tokens, logits)
    if weights_dtype == "float32":
        topk_weights = topk_weights.float()
    act = _precomputed_pack(case, topk_ids, topk_weights)
    _, cake, cake_packed, cake_kwargs, tactics = _cake_runner(case, device, act=act)
    _assert_plan_tactics(case, tactics)
    _, native, native_packed, native_kwargs, _ = _native_runner(case, device, act=act)
    native_output = _forward(native, native_packed, native_kwargs, -1)
    for tactic in tactics:
        output = _forward(cake, cake_packed, cake_kwargs, tactic)
        check_accuracy(case.reference, output.float(), **case.tolerances)
        target, target_name, _ = _native_target(
            case, native, native_packed, native_kwargs, native_output, output, tactic
        )
        ref.assert_family_close(
            output,
            target,
            case.family,
            label=f"{precision} T={num_tokens} {weights_dtype} plan {tactic} vs {target_name}",
        )
    replayed, _ = _capture_and_replay(cake, cake_packed, cake_kwargs, tactics[0])
    assert torch.equal(replayed, _forward(cake, cake_packed, cake_kwargs, tactics[0]))


# ---------------------------------------------------------------------------------------------
# A9 -- finalize vs the public native op
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", TOKENS)
def test_full_path_finalize_matches_native_kernel(num_tokens, cache_permute_indices):
    """The standalone Cake finalize equals the public finalize op (scalar and vector variants) at
    ``TOL["bf16"]`` for bf16 and fp32 expert weights; ``bitwise`` is recorded (expected true: both
    accumulate the eight products in fp32 in ``k`` order)."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    device = _device()
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    case = _build_case(
        cache_permute_indices, precision="bf16", num_tokens=num_tokens, logits=logits
    )
    tile = cb.exported_tiles("fc1", "bf16")[0]
    tables = _native_staged_routing(case, device, tile, logits)
    max_padded = tables["total_num_padded_tokens"]
    expanded = tables["expanded_idx_to_permuted_idx"]
    expert_weights = tables["expert_weights"].reshape(num_tokens, K)
    torch.manual_seed(ref.ROUTING_SEED + num_tokens)
    gemm2_output = torch.randn(max_padded, H, device=device, dtype=torch.bfloat16)
    theirs = torch.full(
        (num_tokens, H), float("nan"), device=device, dtype=torch.bfloat16
    )
    get_trtllm_moe_sm100_module().moe_op.trtllm_moe_run_finalize(
        gemm2_output,
        theirs,
        expert_weights,
        expanded,
        torch.tensor([max_padded], dtype=torch.int32, device=device),
        num_tokens,
        E,
        K,
        H,
        True,
        False,
    )
    _sync()
    assert not torch.isnan(theirs).any()
    for dtype_name in cb.finalize_weight_dtypes():
        weights = expert_weights.float() if dtype_name == "float32" else expert_weights
        ours = torch.full_like(theirs, float("nan"))
        cb.prepare_finalize(
            gemm2_output=gemm2_output,
            expert_weights=weights.contiguous(),
            expanded_idx_to_permuted_idx=expanded.reshape(-1).contiguous(),
            total_num_padded_tokens=torch.tensor(
                [max_padded], dtype=torch.int32, device=device
            ),
            output=ours,
            num_experts=E,
        ).launch()
        _sync()
        record = ref.assert_family_close(
            ours, theirs, "bf16", label=f"finalize T={num_tokens} {dtype_name}"
        )
        _record(
            stage="finalize",
            family="bf16",
            T=num_tokens,
            limit_mode="7",
            regime="ragged",
            input_set=0,
            cake_vs_native=record,
            atol=ref.TOL["bf16"][0],
            rtol=ref.TOL["bf16"][1],
            passed=True,
            kind="stage",
            dtype=dtype_name,
            numel=int(ours.numel()),
            bitwise=bool(torch.equal(ours, theirs)),
            note=f"finalize variant {ref.finalize_variant(H, num_tokens)}",
        )


# ---------------------------------------------------------------------------------------------
# A10 -- requantization vs the public per-token quant op
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens,regime", ref.API_REQUANT_CELLS)
def test_full_path_requant_matches_native_kernel(
    num_tokens, regime, cache_permute_indices
):
    """The standalone Cake NVFP4 per-token requantization equals the public per-token quant op in
    every block-scale layout the exported FC2 units read: the dequantized product within
    ``TOL["nvfp4"]`` (e2m1 one-code rule), the per-token scales bitwise; ``sf_mismatch_bytes`` and
    ``e2m1_one_code_flips`` are recorded; rows without an expanded index stay untouched."""
    from flashinfer.quantization import SfLayout
    from flashinfer.quantization.fp4_quantization import get_fp4_quantization_module

    device = _device()
    if os.environ.get("FLASHINFER_NVFP4_4OVER6", "0") not in ("", "0"):
        pytest.skip("the comparison fixes the default NVFP4 recipe (448 * 6)")
    logits = ref.routing_logits(num_tokens, regime, device=device)
    case = _build_case(
        cache_permute_indices, precision="bf16", num_tokens=num_tokens, logits=logits
    )
    major, minor = get_compute_capability(device)
    quant = get_fp4_quantization_module(f"{major}{minor}")
    layouts = {
        "linear": SfLayout.layout_linear.value,
        "r8c4": SfLayout.layout_8x4.value,
        "r128c4": SfLayout.layout_128x4.value,
    }
    expanded_rows = num_tokens * K
    checked = 0
    for tile in cb.exported_tiles("fc2", "nvfp4_bf16tok"):
        layout = cb.fc2_activation_sf_layout("nvfp4_bf16tok", tile)
        tables = _native_staged_routing(case, device, tile, logits)
        expanded = tables["expanded_idx_to_permuted_idx"].reshape(-1).contiguous()
        assert expanded.numel() == expanded_rows, (tile, expanded.shape)
        # ``prepare_requant`` reads max_padded from ``gemm1_output`` and needs the geometry's row
        # count (a multiple of the layout's block rows); its block scales hold
        # ceil(max_padded / 128) * 128 rows for either layout, as the full path allocates them.
        max_padded = ref.max_padded_tokens(num_tokens, K, E, tile)
        assert max_padded % ref.SF_BLOCK_ROWS[layout] == 0, (layout, tile, max_padded)
        torch.manual_seed(ref.ROUTING_SEED + tile)
        gemm1_output = torch.randn(max_padded, I, device=device, dtype=torch.bfloat16)
        scale_inv = ref.GLOBAL_SCALE_INV
        sf_shape = (-(-max_padded // 128) * 128, -(-(I // 16) // 4) * 4)

        def buffers():
            return (
                torch.full(
                    (max_padded, I // 2), 0xAB, device=device, dtype=torch.uint8
                ),
                torch.full(sf_shape, 0xAB, device=device, dtype=torch.uint8),
                torch.full((max_padded,), -1.0, device=device, dtype=torch.float32),
            )

        ref_out, ref_scale, ref_token = buffers()
        quant.nvfp4_quant_and_per_token_scale_out_sm100(
            torch.as_strided(gemm1_output, (expanded_rows, I), gemm1_output.stride()),
            scale_inv,
            ref_out,
            ref_scale,
            ref_token,
            expanded,
            layouts[layout],
        )
        out, out_scale, token_scale = buffers()
        cb.prepare_requant(
            gemm1_output=gemm1_output,
            expanded_idx_to_permuted_idx=expanded,
            output=out,
            output_scale=out_scale.reshape(-1),
            per_token_scale=token_scale,
            sf_layout=layout,
        ).launch()
        _sync()
        live = expanded[expanded >= 0].long().unique()
        cols = I // ref.NVFP4_SF_VEC
        record = ref.nvfp4_rows_compare(
            packed_a=out[live],
            sf_a=ref.unswizzle_sf(out_scale, max_padded, cols, layout)[live],
            packed_e=ref_out[live],
            sf_e=ref.unswizzle_sf(ref_scale, max_padded, cols, layout)[live],
            token_scale_a=token_scale[live],
            token_scale_e=ref_token[live],
            label=f"requant tile {tile} {layout} T={num_tokens} {regime}",
        )
        assert record["token_scale_bitwise"], record
        untouched = torch.ones(max_padded, dtype=torch.bool, device=device)
        untouched[live] = False
        assert bool((out[untouched] == 0xAB).all()) and bool(
            (token_scale[untouched] == -1.0).all()
        )
        _record(
            stage="requant",
            family="nvfp4_bf16tok",
            T=num_tokens,
            limit_mode="7",
            regime=regime,
            input_set=0,
            cake_vs_native=record,
            atol=ref.TOL["nvfp4"][0],
            rtol=ref.TOL["nvfp4"][1],
            passed=True,
            kind="stage",
            plan=tile,
            dtype="nvfp4",
            numel=int(live.numel()) * I,
            bitwise=record["bitwise"],
            sf_mismatch_bytes=record["sf_mismatch_bytes"],
            e2m1_one_code_flips=record["e2m1_one_code_flips"],
            note=f"sf layout {layout}; padded rows excluded: {int(untouched.sum())}",
        )
        checked += 1
    assert checked


# ---------------------------------------------------------------------------------------------
# A12 -- a topk_ids forward launches only Cake kernels
# ---------------------------------------------------------------------------------------------


def _profiled_kernel_names(runner, packed, kwargs, tactic) -> list[str]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
    ) as profile:
        runner.forward(packed, tactic=tactic, **kwargs)
        _sync()
        profile.step()
        runner.forward(packed, tactic=tactic, **kwargs)
        _sync()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith(("Memcpy", "Memset"))
    ]


@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_full_path_launches_only_cake_kernels_from_topk_ids(
    precision, num_tokens, cache_permute_indices
):
    """A forward from pre-computed ids launches exactly the Cake kernels of its stages -- the
    ``topk_ids`` routing kernel(s) of the token count, the FC1 and FC2 units of the plan's tile and
    the finalize variant of the bf16 weights -- each once, and no trtllm-gen kernel."""
    device = _device()
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limit_mode="mixed",
        logits=logits,
    )
    topk_ids, topk_weights = _precomputed_routing(case, device, num_tokens, logits)
    act = _precomputed_pack(case, topk_ids, topk_weights)
    _, runner, packed, kwargs, tactics = _cake_runner(case, device, act=act)
    for tactic in tactics:
        tile = _tactic_tile(tactic)
        _forward(runner, packed, kwargs, tactic)
        names = _profiled_kernel_names(runner, packed, kwargs, tactic)
        assert names, "the profiler captured no kernel"
        offenders = [name for name in names if CAKE_KERNEL_SYMBOL.search(name) is None]
        assert not offenders, (
            f"non-Cake kernels in a Cake forward: {sorted(set(offenders))}"
        )
        symbols = [CAKE_KERNEL_SYMBOL.search(name).group(0) for name in names]
        programs = list(cb.routing_plan(num_tokens, "topk_ids", None))
        programs += [
            cb.fc1_program(case.family, tile),
            cb.fc2_program(case.family, tile),
        ]
        programs.append(
            cb.finalize_program(ref.finalize_variant(H, num_tokens), "bfloat16")
        )
        expected = [loader.PROGRAMS[p]["kernel"] for p in programs]
        assert Counter(symbols) == Counter(expected), (symbols, expected)


# ---------------------------------------------------------------------------------------------
# A13 -- the facade runs the backend's prepared launch
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_facade_forward_equals_backend_forward(precision, cache_permute_indices):
    """The facade's default tactic produces the bytes of ``prepare_stepfun_moe`` on the same view,
    activations and logits with the plan of that tactic's tile."""
    device = _device()
    num_tokens = 64
    logits = ref.routing_logits(num_tokens, "ragged", device=device)
    case = _build_case(
        cache_permute_indices,
        precision=precision,
        num_tokens=num_tokens,
        limit_mode="mixed",
        logits=logits,
    )
    _, runner, packed, kwargs, _ = _cake_runner(case, device)
    tactic = _tactic_key(runner.default_tactic(packed))
    facade_output = _forward(runner, packed, kwargs, tactic)
    view = {}
    for key, value in case.view.items():
        backend_key = key
        if precision == "fp8":
            backend_key = {v: k for k, v in FP8_VIEW_KEYS.items()}.get(key, key)
        view[backend_key] = value
    launch = cb.prepare_stepfun_moe(
        family=case.family,
        hidden_states=case.act.hidden_states_q,
        hidden_states_scale=case.act.hidden_states_scale,
        routing_logits=logits,
        num_experts=E,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=E,
        plan=_tactic_tile(tactic),
        **view,
    )
    launch.launch()
    _sync()
    (backend_output,) = launch.outputs()
    assert torch.equal(facade_output, backend_output)


# ---------------------------------------------------------------------------------------------
# A14 -- the package entries are marked experimental
# ---------------------------------------------------------------------------------------------


def test_public_api_is_marked_experimental():
    """The one-shot package helpers carry ``@flashinfer_experimental_api`` (``is_experimental``);
    naming the backend in ``flashinfer.fused_moe`` is the explicit opt-in."""
    package = flashinfer.experimental.cake_stepfun_moe
    for name in ("stepfun_moe", "routing", "fc1", "requant", "fc2", "finalize"):
        assert getattr(package, name).is_experimental is True, name
    assert package.prepare_stepfun_moe is cb.prepare_stepfun_moe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert CakeStepFunConfig(backend="cake_stepfun").backend == "cake_stepfun"


# ---------------------------------------------------------------------------------------------
# A15 -- the error table (last)
# ---------------------------------------------------------------------------------------------


def _command_output(*command) -> str:
    try:
        return subprocess.run(
            command, text=True, capture_output=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


# The measurement identity of build.json (Cake ``cake_build_manifest.identity_digest``): sha256 of the
# deterministic JSON (sorted keys, indent 2, ASCII, trailing newline) of the document without its
# ``generator`` block.
MANIFEST_IDENTITY_METHOD = "manifest_without_generator_sha256"


def _manifest_identity(manifest: dict) -> str:
    document = {key: value for key, value in manifest.items() if key != "generator"}
    text = (
        json.dumps(
            document, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_zz_error_table_is_complete_and_written(tmp_path_factory):
    """Every expected (stage, family, T, limit mode, regime, input set) key of this arch was
    recorded by the comparison tests of this process, every gated row passed, the ``fc1`` rows are
    ``n/a``, and ``cake_stepfun_error_table.<arch>.json`` is written into
    ``CAKE_STEPFUN_ERROR_TABLE_DIR`` (default: the pytest base temporary directory)."""
    arch = _arch()
    for family in ref.PRECISION_FAMILY.values():
        for num_tokens in TOKENS:
            _record(
                stage="fc1",
                family=family,
                T=num_tokens,
                limit_mode="mixed",
                regime="ragged",
                input_set=0,
                status="n/a",
                kind="stage",
                note="no native FC1-only output on the full Cake path (no FC1-only hybrid)",
            )
    recorded = {ref.error_key(row) for row in ERROR_ROWS}
    missing = sorted(ref.expected_error_table_keys() - recorded)
    if missing:
        pytest.skip(
            f"{len(missing)} of {len(ref.expected_error_table_keys())} error-table cells did not run in "
            f"this process (first missing: {missing[:3]}); run the whole module to write the table"
        )
    gated = [row for row in ERROR_ROWS if row["status"] == "gated"]
    assert gated and all(row["pass"] is True for row in gated)
    assert all(
        row["status"] == "n/a" and row["pass"] is None
        for row in ERROR_ROWS
        if row["stage"] == "fc1"
    )
    for row in ERROR_ROWS:
        assert set(ref.ERROR_ROW_REQUIRED_KEYS) <= set(row)
        assert row["stage"] in ref.ERROR_STAGES and row["limit_mode"] in ref.LIMIT_MODES
        assert (
            row["regime"] in ref.ROUTING_REGIMES and row["input_set"] in ref.INPUT_SETS
        )
    manifest = cake_build.manifest()
    nvcc = _command_output("nvcc", "--version").splitlines()
    header = {
        "arch": arch,
        "device": torch.cuda.get_device_name(_device()),
        "compute_capability": list(get_compute_capability(_device())),
        "fi_commit": _command_output(
            "git",
            "-C",
            str(Path(flashinfer.__file__).resolve().parent.parent),
            "rev-parse",
            "HEAD",
        ),
        "manifest_identity": _manifest_identity(manifest),
        "manifest_identity_method": MANIFEST_IDENTITY_METHOD,
        "cake_revision": manifest.get("generator", {}).get("cake_revision"),
        "toolkit": nvcc[-1] if nvcc else None,
        "torch": torch.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": "fi_trtllm_gen_fp32_torch",
        "reference_fn": "tests/moe/trtllm_gen_fused_moe_utils.py:run_moe_dequant",
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "rows_total": len(ERROR_ROWS),
    }
    directory = os.environ.get(ERROR_TABLE_DIR_ENV) or str(
        tmp_path_factory.getbasetemp()
    )
    path = ref.write_error_table(directory, arch, header, ERROR_ROWS)
    document = json.loads(path.read_text())
    assert document["schema"] == ref.ERROR_TABLE_SCHEMA and document["arch"] == arch
    assert len(document["rows"]) == len(ERROR_ROWS)
    assert document["tolerances"]["nvfp4"] == [1.0, 0.1]
    print(f"error table written: {path} ({len(ERROR_ROWS)} rows)")
