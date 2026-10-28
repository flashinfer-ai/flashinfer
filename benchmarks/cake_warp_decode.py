#!/usr/bin/env python3
"""SM100/SM103 Cake warp-decode cold-L2 CUPTI benchmark harness.

The comparison arms share one prepared TRTLLM NVFP4 representation: activation,
routing, packed-weight, and block-scale tensors. Parameterized SwiGLU uses
separate official beta/clamp buffers divided by the gate scale; alpha remains
shared. SwiGLU and SiTU rows use the public
``trtllm_fp4_block_scale_routed_moe`` baseline. Standalone SiLU has no supported
official peer and reports exported Cake absolute timing only. Benchmark
preparation stays outside the timed regions. The correctness tests live in
``tests/moe/test_cake_warp_decode.py``,
``tests/moe/test_cake_warp_decode_clamped_gpu.py`` and
``tests/moe/test_cake_warp_decode_workspace_gpu.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import weakref
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from flashinfer.fused_moe import (
    ActivationConfig,
    CakeWarpDecodeConfig,
    QuantConfig,
    QuantFormat,
    RoutingMethodType,
    SiLU,
    SiTU,
    SwiGLU,
    TrtllmFp4Config,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.jit.cake_fused_moe_warp_decode import (
    get_cake_fused_moe_warp_decode_module,
)
from flashinfer.testing.utils import bench_gpu_time


ATOL = 1e-2
RTOL = 1e-2
MAX_TOKENS = 32
PAIR_PATTERNS = (
    ("ABBA", ("exported", "baseline", "baseline", "exported")),
    ("BAAB", ("baseline", "exported", "exported", "baseline")),
)
# A failed release leaves receipt ownership ambiguous: the binding may have
# rejected the call outright or failed while waiting for the generation's
# recorded submissions.  Hold the backing allocation for process lifetime on
# that exceptional path so its address cannot be recycled unsafely.
_WORKSPACE_QUARANTINE: dict[int, torch.Tensor] = {}


def _quarantine_workspace(workspace: torch.Tensor) -> int:
    identity = id(workspace)
    existing = _WORKSPACE_QUARANTINE.get(identity)
    if existing is not None and existing is not workspace:
        raise AssertionError("workspace quarantine identity collision")
    _WORKSPACE_QUARANTINE[identity] = workspace
    return identity


def _unquarantine_workspace(identity: int, workspace: torch.Tensor) -> None:
    retained = _WORKSPACE_QUARANTINE.pop(identity, None)
    if retained is not workspace:
        raise AssertionError("workspace quarantine ownership mismatch")


def _release_workspace_receipt_fail_closed(
    module: Any, workspace: torch.Tensor, receipt: int
) -> None:
    """Release once, retaining storage permanently if retirement is uncertain."""
    identity = _quarantine_workspace(workspace)
    try:
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
    except Exception:
        raise
    else:
        _unquarantine_workspace(identity, workspace)


def _finalize_workspace_receipt(
    module: Any, workspace: torch.Tensor, receipt: int
) -> None:
    with suppress(Exception):
        _release_workspace_receipt_fail_closed(module, workspace, receipt)
        # Finalizers cannot report errors usefully. The helper has retained the
        # allocation so its address cannot be recycled beneath uncertain work.


def _detach_and_release_workspace_receipt(releaser: weakref.finalize) -> None:
    detached = releaser.detach()
    if detached is None:
        raise AssertionError("workspace receipt finalizer lost ownership")
    _, callback, args, kwargs = detached
    if callback is not _finalize_workspace_receipt or kwargs or len(args) != 3:
        raise AssertionError("workspace receipt finalizer has an invalid callback")
    module, workspace, receipt = args
    _release_workspace_receipt_fail_closed(module, workspace, receipt)


@dataclass(frozen=True)
class Geometry:
    """One exact warp-decode model geometry."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    selector_boundaries: tuple[int, ...]
    activation: ActivationConfig = SwiGLU()


GEOMETRIES = (
    Geometry("e512_i512_k10", 2048, 512, 512, 10, (1, 3, 4, 32)),
    Geometry("e60_i1536_k4", 2048, 1536, 60, 4, (1, 4, 5, 10, 11, 12, 14, 15, 32)),
    Geometry("e192_i1536_k4_silu", 6144, 1536, 192, 4, (1, 32), SiLU()),
    Geometry(
        "e256_i2048_k6_clamp10",
        4096,
        2048,
        256,
        6,
        (1, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 32),
        SwiGLU(limit=10.0),
    ),
    Geometry("e384_i768_k4", 2560, 768, 384, 4, (1, 32)),
    Geometry("e128_i768_k8", 2048, 768, 128, 8, (1, 7, 8, 15, 16, 32)),
    Geometry("e128_i1536_k8", 4096, 1536, 128, 8, (1, 6, 11, 12, 32)),
    Geometry("e256_i512_k8", 2048, 512, 256, 8, (1, 15, 16, 18, 19, 30, 32)),
    Geometry("e512_i1024_k10", 4096, 1024, 512, 10, (1, 2, 5, 6, 8, 25, 26, 32)),
    Geometry("e256_i1536_k8", 3072, 1536, 256, 8, (1, 15, 16, 32)),
    Geometry(
        "e128_i3072_k4_swiglu_oa",
        6144,
        3072,
        128,
        4,
        (1, 3, 4, 15, 16, 32),
        SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
    ),
    Geometry(
        "e896_i3072_k16_situ",
        3584,
        3072,
        896,
        16,
        (1, 2, 32),
        SiTU(gate_scale=4.0, linear_scale=25.0),
    ),
    # Sharded per-partition slices (fused route packing on every token count).
    Geometry("h4096_e512_i512_k10", 4096, 512, 512, 10, (1, 2, 16, 32)),
    Geometry("h4096_e512_i256_k10", 4096, 256, 512, 10, (1, 2, 16, 32)),
    Geometry("h3072_e256_i768_k8", 3072, 768, 256, 8, (1, 2, 16, 32)),
    Geometry("h3072_e256_i384_k8", 3072, 384, 256, 8, (1, 2, 8, 9, 10, 16, 32)),
)


def _activation_name(geometry: Geometry) -> str:
    return geometry.activation.type.name.lower()


def _has_official_baseline(geometry: Geometry) -> bool:
    return isinstance(geometry.activation, (SwiGLU, SiTU))


def _uses_activation_params(geometry: Geometry) -> bool:
    return geometry.activation.is_gated and geometry.activation != SwiGLU()


def _selector_bucket(geometry: Geometry, num_tokens: int) -> str:
    """Name the fixed schedule/route-packer bucket exercised by a row.

    The e60_i1536_k4 buckets name the sm_103a selection; the other geometries
    report a coarse label.
    """
    if geometry.activation == SwiGLU(limit=10.0):
        # Names describe the semantic selection; generated inventory supplies
        # exact kernel identities, not a second heuristic selector.
        if num_tokens <= 6:
            return "direct"
        if num_tokens in (7, 8):
            return "direct_expert_order"
        if num_tokens in (9, 10):
            return "packed_sorted_short"
        if num_tokens in (11, 23):
            return "packed_count_rank"
        return "packed_general"
    key = (
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.num_experts,
        geometry.top_k,
    )
    if isinstance(geometry.activation, SiTU):
        return "static_direct"
    if key != (2048, 1536, 60, 4):
        return "persistent_direct"

    if num_tokens <= 4:
        return "persistent_direct"
    if num_tokens <= 10:
        return "persistent_direct_padded_sfb"
    if num_tokens == 11:
        return "persistent_e64_scan1_padded_sfb"
    return "persistent_fused_route_packer"


@dataclass
class PhysicalFixture:
    """TRTLLM NVFP4 physical tensors for one geometry."""

    geometry: Geometry
    hidden_states_q: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    weight_view: Mapping[str, torch.Tensor]

    def case(self, num_tokens: int) -> "PhysicalCase":
        if not 1 <= num_tokens <= MAX_TOKENS:
            raise ValueError(f"num_tokens must be in [1, {MAX_TOKENS}]")
        return PhysicalCase(
            geometry=self.geometry,
            num_tokens=num_tokens,
            hidden_states_q=self.hidden_states_q[:num_tokens],
            hidden_states_scale=self.hidden_states_scale[:num_tokens],
            topk_ids=self.topk_ids[:num_tokens],
            topk_weights=self.topk_weights[:num_tokens],
            weight_view=self.weight_view,
        )


@dataclass(frozen=True)
class PhysicalCase:
    """The exact physical inputs shared by every implementation for one T."""

    geometry: Geometry
    num_tokens: int
    hidden_states_q: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    weight_view: Mapping[str, torch.Tensor]

    @property
    def device(self) -> torch.device:
        return self.hidden_states_q.device

    def benchmark_inputs(self) -> tuple[torch.Tensor, ...]:
        """Expose live tensors so bench_gpu_time selects the correct CUDA device."""
        view = self.weight_view
        return (
            self.hidden_states_q,
            self.hidden_states_scale,
            self.topk_ids,
            self.topk_weights,
            view["gemm1_weights"],
            view["gemm1_weights_scale"],
            view["gemm2_weights"],
            view["gemm2_weights_scale"],
            view["output1_scale_scalar"],
            view["output1_scale_gate_scalar"],
            view["output2_scale_scalar"],
        )


@dataclass
class PreparedCall:
    """A launch-only callable whose initialization has already completed."""

    name: str
    output: torch.Tensor
    invoke: Callable[[], torch.Tensor]
    workspace: Any = None
    workspace_receipt: Optional[int] = None
    receipt_releaser: Optional[weakref.finalize] = None

    def close(self) -> None:
        """Release a public workspace receipt exactly once, if this call owns one."""
        if self.receipt_releaser is not None and self.receipt_releaser.alive:
            _detach_and_release_workspace_receipt(self.receipt_releaser)
            self.workspace_receipt = None


def _make_routing(geometry: Geometry) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    tokens = torch.arange(MAX_TOKENS, device=device, dtype=torch.int64)[:, None]
    ranks = torch.arange(geometry.top_k, device=device, dtype=torch.int64)[None, :]
    ids = ((tokens * 17 + ranks * 29) % (geometry.num_experts - 1) + 1).to(torch.int32)
    ids[:, 0] = 0
    raw_weights = (geometry.top_k + 1 - ranks).expand(MAX_TOKENS, -1).float()
    raw_weights = raw_weights + (tokens % 5).float() * 0.03125
    weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    return ids.contiguous(), weights.contiguous()


def _prepare_fixture(geometry: Geometry, seed: int) -> PhysicalFixture:
    """Use the public TRTLLM preparation API to produce one physical fixture."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden = torch.randn(
        MAX_TOKENS, geometry.hidden_size, device=device, dtype=torch.bfloat16
    )
    w1 = (
        torch.randn(
            geometry.num_experts,
            geometry.intermediate_size * (2 if geometry.activation.is_gated else 1),
            geometry.hidden_size,
            device=device,
        )
        * 0.02
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            geometry.num_experts,
            geometry.hidden_size,
            geometry.intermediate_size,
            device=device,
        )
        * 0.02
    ).to(torch.bfloat16)
    if geometry.activation == SwiGLU(limit=10.0):
        hidden.mul_(16.0)  # Exercise both upper-gate and symmetric-linear clamp.
    hidden_q, hidden_scale = TrtllmFp4Config.prepare_activations(
        hidden,
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
    )
    weight_view = CakeWarpDecodeConfig.prepare_weights(
        w1,
        w2,
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        num_local_experts=geometry.num_experts,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        activation=geometry.activation,
        device=device,
    )
    if geometry.activation == SwiGLU(limit=10.0):
        # Both arms consume these same physical tensors. Non-unit gate scales
        # exercise the raw-accumulator clamp conversion used by calibrated views.
        gate_scale = torch.linspace(
            0.5, 1.5, geometry.num_experts, dtype=torch.float32, device=device
        )
        weight_view["output1_scale_gate_scalar"] = gate_scale
        weight_view["gemm1_clamp_limit"] = (
            torch.full_like(gate_scale, 10.0) / gate_scale
        )
    topk_ids, topk_weights = _make_routing(geometry)
    return PhysicalFixture(
        geometry=geometry,
        hidden_states_q=hidden_q,
        hidden_states_scale=hidden_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        weight_view=weight_view,
    )


def _normalize_result(result: Any) -> torch.Tensor:
    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("MoE implementation returned an empty result sequence")
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError(
            f"MoE implementation returned {type(result).__name__}, not Tensor"
        )
    return result


def _prepare_baseline(case: PhysicalCase) -> PreparedCall:
    geometry = case.geometry
    if not _has_official_baseline(geometry):
        raise ValueError(
            "the official TRT-LLM NVFP4 runner does not support standalone SiLU"
        )
    view = case.weight_view
    official_beta = view.get("gemm1_beta")
    official_clamp_limit = view.get("gemm1_clamp_limit")
    if geometry.name == "e128_i3072_k4_swiglu_oa":
        # This baseline consumes beta and clamp in raw-accumulator units.
        # Keep the logical parameter tensors supplied to warp-decode intact.
        official_beta = view["gemm1_beta"] / view["output1_scale_gate_scalar"]
        official_clamp_limit = (
            view["gemm1_clamp_limit"] / view["output1_scale_gate_scalar"]
        )
    output = torch.empty(
        case.num_tokens,
        geometry.hidden_size,
        dtype=torch.bfloat16,
        device=case.device,
    )

    def invoke() -> torch.Tensor:
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(case.topk_ids, case.topk_weights),
            routing_bias=None,
            hidden_states=case.hidden_states_q,
            hidden_states_scale=case.hidden_states_scale,
            gemm1_weights=view["gemm1_weights"],
            gemm1_weights_scale=view["gemm1_weights_scale"],
            gemm1_bias=None,
            gemm1_alpha=view.get("gemm1_alpha"),
            gemm1_beta=official_beta,
            gemm1_clamp_limit=official_clamp_limit,
            gemm2_weights=view["gemm2_weights"],
            gemm2_weights_scale=view["gemm2_weights_scale"],
            gemm2_bias=None,
            output1_scale_scalar=view["output1_scale_scalar"],
            output1_scale_gate_scalar=view["output1_scale_gate_scalar"],
            output2_scale_scalar=view["output2_scale_scalar"],
            num_experts=geometry.num_experts,
            top_k=geometry.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=geometry.intermediate_size,
            local_expert_offset=0,
            local_num_experts=geometry.num_experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.TopK.value,
            do_finalize=True,
            enable_pdl=True,
            activation_type=geometry.activation.type.value,
            per_token_scale=None,
            output=output,
            tune_max_num_tokens=MAX_TOKENS,
        )
        result_tensor = _normalize_result(result)
        if result_tensor.data_ptr() != output.data_ptr():
            raise RuntimeError("TRTLLM baseline did not honor its caller output tensor")
        return output

    return PreparedCall("flashinfer_trtllm_nvfp4", output, invoke)


def _cake_target(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    try:
        return {(10, 0): "sm100a", (10, 3): "sm103a"}[(major, minor)]
    except KeyError as error:
        raise RuntimeError(
            "the warp-decode harness requires exact SM100 or SM103, "
            f"got SM{major}{minor}"
        ) from error


def _get_cake_module(device: torch.device) -> Any:
    return get_cake_fused_moe_warp_decode_module(_cake_target(device), device=device)


def _cake_shape(case: PhysicalCase) -> tuple[int, int, int, int, int]:
    geometry = case.geometry
    return (
        case.num_tokens,
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.num_experts,
        geometry.top_k,
    )


def _invoke_cake(
    module: Any,
    case: PhysicalCase,
    output: torch.Tensor,
    workspace: torch.Tensor,
    workspace_receipt: int,
) -> torch.Tensor:
    view = case.weight_view
    inputs = (
        output,
        workspace,
        case.hidden_states_q,
        case.hidden_states_scale,
        case.topk_ids,
        case.topk_weights,
        view["gemm1_weights"],
        view["gemm1_weights_scale"],
        view["gemm2_weights"],
        view["gemm2_weights_scale"],
        view["output1_scale_scalar"],
        view["output1_scale_gate_scalar"],
        view["output2_scale_scalar"],
    )
    if case.geometry.activation == SwiGLU(limit=10.0):
        module.cake_fused_moe_warp_decode_clamped_swiglu(
            *inputs,
            view["gemm1_alpha"],
            view["gemm1_beta"],
            view["gemm1_clamp_limit"],
            workspace_receipt,
            True,
        )
    elif _uses_activation_params(case.geometry):
        module.cake_fused_moe_warp_decode_with_activation_params(
            *inputs,
            view.get("gemm1_alpha"),
            view.get("gemm1_beta"),
            view.get("gemm1_clamp_limit"),
            workspace_receipt,
            True,
        )
    else:
        module.cake_fused_moe_warp_decode(*inputs, workspace_receipt, True)
    return output


def _prepare_cake(case: PhysicalCase) -> PreparedCall:
    geometry = case.geometry
    module = _get_cake_module(case.device)
    shape = _cake_shape(case)
    workspace_size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
    if workspace_size <= 0:
        raise RuntimeError("Cake workspace query returned a non-positive byte count")
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=case.device)
    output = torch.empty(
        case.num_tokens,
        geometry.hidden_size,
        dtype=torch.bfloat16,
        device=case.device,
    )
    workspace_receipt = int(
        module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
    )
    if workspace_receipt <= 0:
        raise RuntimeError("Cake workspace preparation returned an invalid receipt")

    def invoke() -> torch.Tensor:
        return _invoke_cake(
            module,
            case,
            output,
            workspace,
            workspace_receipt,
        )

    receipt_releaser = weakref.finalize(
        output,
        _finalize_workspace_receipt,
        module,
        workspace,
        workspace_receipt,
    )
    receipt_releaser.atexit = False
    return PreparedCall(
        "cake_warp_decode",
        output,
        invoke,
        workspace,
        workspace_receipt,
        receipt_releaser,
    )


def _distinct_nondefault_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    current = torch.cuda.current_stream(device)
    prepare_stream = torch.cuda.Stream(device=device)
    run_stream = torch.cuda.Stream(device=device)
    handles = {
        int(current.cuda_stream),
        int(prepare_stream.cuda_stream),
        int(run_stream.cuda_stream),
    }
    if len(handles) != 3:
        raise RuntimeError("failed to create distinct prepare and run CUDA streams")
    prepare_stream.wait_stream(current)
    run_stream.wait_stream(current)
    return prepare_stream, run_stream


def _diagnostic(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise TypeError(
            "warp-decode correctness compares BF16 outputs; got "
            f"{actual.dtype} and {expected.dtype}"
        )
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    abs_error = (actual_f32 - expected_f32).abs()
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "max_abs": float(abs_error.max().item()),
        "mean_abs": float(abs_error.mean().item()),
        "finite": bool(torch.isfinite(actual_f32).all().item()),
    }


def run_sanitizer(
    geometries: Sequence[Geometry],
    *,
    seed: int,
    sanitizer_tokens: Optional[Sequence[int]],
) -> dict[str, Any]:
    """Issue launch-only boundary cases for compute-sanitizer wrappers."""
    rows: list[dict[str, Any]] = []
    for geometry_index, geometry in enumerate(geometries):
        fixture = _prepare_fixture(geometry, seed + geometry_index)
        tokens = (
            tuple(sanitizer_tokens)
            if sanitizer_tokens is not None
            else geometry.selector_boundaries
        )
        for num_tokens in tokens:
            case = fixture.case(num_tokens)
            prepare_stream, run_stream = _distinct_nondefault_streams(case.device)
            with torch.cuda.stream(prepare_stream):
                cake = _prepare_cake(case)
            with torch.cuda.stream(run_stream):
                cake.invoke()
            run_stream.synchronize()
            cake.close()
            rows.append(
                {
                    "geometry": geometry.name,
                    "activation": _activation_name(geometry),
                    "num_tokens": num_tokens,
                    "selector_bucket": _selector_bucket(geometry, num_tokens),
                    "prepare_stream": "non_default",
                    "run_stream": "different_non_default",
                    "cross_stream_prepare_run": True,
                    "workspace_receipt_released": True,
                    "status": "launched",
                }
            )
    return {"mode": "sanitizer", "rows": rows, "status": "pass"}


def _require_cupti() -> str:
    try:
        from cupti import cupti  # noqa: F401

        cupti_version = version("cupti-python")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "cupti-python >= 13 is required; benchmark fallback is not permitted"
        ) from error
    if int(cupti_version.split(".", 1)[0]) < 13:
        raise RuntimeError(
            f"cupti-python >= 13 is required, found {cupti_version}; "
            "benchmark fallback is not permitted"
        )
    return cupti_version


def _validate_prepared_call(call: PreparedCall, case: PhysicalCase) -> None:
    expected_shape = (case.num_tokens, case.geometry.hidden_size)
    if call.output.dtype != torch.bfloat16:
        raise TypeError(f"{call.name} output must be BF16, got {call.output.dtype}")
    if tuple(call.output.shape) != expected_shape:
        raise ValueError(
            f"{call.name} output shape {tuple(call.output.shape)} != {expected_shape}"
        )
    if call.output.device != case.device or not call.output.is_contiguous():
        raise ValueError(f"{call.name} output must be contiguous on {case.device}")


def _benchmark_call(
    call: PreparedCall,
    case: PhysicalCase,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    _validate_prepared_call(call, case)

    def launch_with_live_inputs(*_live_inputs: torch.Tensor) -> torch.Tensor:
        return call.invoke()

    call.invoke()
    torch.cuda.synchronize(case.device)
    samples = bench_gpu_time(
        fn=launch_with_live_inputs,
        input_args=case.benchmark_inputs(),
        enable_cupti=True,
        cold_l2_cache=True,
        use_cuda_graph=False,
        dry_run_iters=warmup,
        repeat_iters=repetitions,
    )
    if not samples or any(
        not math.isfinite(float(sample)) or float(sample) <= 0.0 for sample in samples
    ):
        raise RuntimeError(
            f"{call.name} returned non-positive or non-finite CUPTI samples"
        )
    return {
        "name": call.name,
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "samples": len(samples),
    }


def _paired_benchmark_round(
    exported: PreparedCall,
    baseline: PreparedCall,
    case: PhysicalCase,
    *,
    round_index: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    """Measure one exported/baseline ABBA or BAAB CUPTI round."""
    pattern, order = PAIR_PATTERNS[round_index % len(PAIR_PATTERNS)]
    calls = {"exported": exported, "baseline": baseline}
    measurements: list[dict[str, Any]] = []
    per_arm: dict[str, list[float]] = {"exported": [], "baseline": []}
    for position, arm in enumerate(order):
        measurement = _benchmark_call(
            calls[arm],
            case,
            warmup=warmup,
            repetitions=repetitions,
        )
        measurements.append(
            {
                "position": position,
                "arm": arm,
                "measurement": measurement,
            }
        )
        per_arm[arm].append(float(measurement["median_ms"]))
    exported_ms = float(statistics.median(per_arm["exported"]))
    baseline_ms = float(statistics.median(per_arm["baseline"]))
    return {
        "round_index": round_index,
        "pattern": pattern,
        "order": list(order),
        "measurements": measurements,
        "exported_median_ms": exported_ms,
        "baseline_median_ms": baseline_ms,
        "exported_baseline_ratio": exported_ms / baseline_ms,
    }


def _paired_arm_summary(
    name: str,
    rounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    session_medians = [
        float(item["measurement"]["median_ms"])
        for round_record in rounds
        for item in round_record["measurements"]
        if item["arm"] == name
    ]
    return {
        "name": name,
        "median_ms": float(statistics.median(session_medians)),
        "session_count": len(session_medians),
        "session_medians_ms": session_medians,
    }


def run_benchmark(
    geometries: Sequence[Geometry],
    *,
    seed: int,
    benchmark_tokens: Optional[Sequence[int]],
    warmup: int,
    repetitions: int,
    paired_rounds: int,
) -> dict[str, Any]:
    cupti_version = _require_cupti()
    rows: list[dict[str, Any]] = []
    for geometry_index, geometry in enumerate(geometries):
        fixture = _prepare_fixture(geometry, seed + geometry_index)
        tokens = (
            tuple(benchmark_tokens)
            if benchmark_tokens is not None
            else tuple(range(1, MAX_TOKENS + 1))
            if geometry.activation == SwiGLU(limit=10.0)
            else geometry.selector_boundaries
        )
        for num_tokens in tokens:
            case = fixture.case(num_tokens)
            prepare_stream, benchmark_stream = _distinct_nondefault_streams(case.device)
            with torch.cuda.stream(prepare_stream):
                exported = _prepare_cake(case)
            output_ptr = exported.output.data_ptr()
            workspace_ptr = exported.workspace.data_ptr()
            workspace_bytes = exported.workspace.numel()
            with torch.cuda.stream(benchmark_stream):
                exported_output = exported.invoke().clone()
                torch.cuda.synchronize(case.device)
                row: dict[str, Any] = {
                    "geometry": geometry.name,
                    "activation": _activation_name(geometry),
                    "num_tokens": num_tokens,
                    "selector_bucket": _selector_bucket(geometry, num_tokens),
                    "prepare_stream": "non_default",
                    "benchmark_stream": "different_non_default",
                    "cross_stream_prepare_run": True,
                    "exported_parity": None,
                    "exported": None,
                    "flashinfer_baseline": None,
                    "paired_rounds": [],
                    "ratios": {},
                    "status": "pass",
                }
                if _has_official_baseline(geometry):
                    baseline = _prepare_baseline(case)
                    baseline_output = baseline.invoke().clone()
                    torch.cuda.synchronize(case.device)
                    row["exported_parity"] = _diagnostic(
                        exported_output, baseline_output
                    )
                    pair_records = [
                        _paired_benchmark_round(
                            exported,
                            baseline,
                            case,
                            round_index=round_index,
                            warmup=warmup,
                            repetitions=repetitions,
                        )
                        for round_index in range(paired_rounds)
                    ]
                    exported_summary = _paired_arm_summary("exported", pair_records)
                    baseline_summary = _paired_arm_summary("baseline", pair_records)
                    row["timing_mode"] = "paired_official_baseline"
                    row["exported"] = exported_summary
                    row["flashinfer_baseline"] = baseline_summary
                    row["paired_rounds"] = pair_records
                    row["ratios"] = {
                        "exported_over_flashinfer_baseline": (
                            float(exported_summary["median_ms"])
                            / float(baseline_summary["median_ms"])
                        ),
                        "worst_round_exported_over_flashinfer_baseline": max(
                            float(record["exported_baseline_ratio"])
                            for record in pair_records
                        ),
                    }
                else:
                    row["timing_mode"] = "exported_absolute"
                    row["exported_parity"] = {
                        "status": "N/A",
                        "reason": "source/export parity is a separate release receipt",
                    }
                    row["exported"] = _benchmark_call(
                        exported,
                        case,
                        warmup=warmup,
                        repetitions=repetitions,
                    )
                    row["flashinfer_baseline"] = {
                        "status": "N/A",
                        "reason": (
                            "the official TRT-LLM NVFP4 runner does not support "
                            "standalone SiLU"
                        ),
                    }
                    row["ratios"] = {
                        "exported_over_flashinfer_baseline": "N/A",
                    }
            benchmark_stream.synchronize()
            if exported.output.data_ptr() != output_ptr:
                raise AssertionError("benchmark replaced the caller output tensor")
            if (
                exported.workspace.data_ptr() != workspace_ptr
                or exported.workspace.numel() != workspace_bytes
            ):
                raise AssertionError("benchmark replaced the caller workspace")
            exported.close()
            row["output_reused"] = True
            row["workspace_reused"] = True
            row["workspace_receipt_released"] = True
            rows.append(row)
    return {
        "mode": "benchmark",
        "timing": {
            "backend": "cupti",
            "cupti_python": cupti_version,
            "cold_l2_cache": True,
            "use_cuda_graph": False,
            "warmup": warmup,
            "repetitions": repetitions,
            "paired_rounds": paired_rounds,
            "pair_patterns": [name for name, _ in PAIR_PATTERNS],
            "standalone_silu": "exported Cake absolute timing only",
        },
        "rows": rows,
        "status": "pass",
    }


def _selected_geometries(name: str) -> tuple[Geometry, ...]:
    if name == "all":
        return tuple(
            geometry
            for geometry in GEOMETRIES
            if geometry.activation != SwiGLU(limit=10.0)
            or torch.cuda.get_device_capability() == (10, 0)
        )
    selected = tuple(geometry for geometry in GEOMETRIES if geometry.name == name)
    if not selected:
        raise ValueError(f"unknown geometry {name!r}")
    if any(g.activation == SwiGLU(limit=10.0) for g in selected):
        if torch.cuda.get_device_capability() != (10, 0):
            raise ValueError("Clamped E256 warp decode requires exact SM100.")
    return selected


def _validate_environment(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError("the warp-decode harness requires a CUDA device")
    return _cake_target(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("benchmark", "sanitizer"), default="benchmark"
    )
    parser.add_argument(
        "--geometry",
        choices=("all", *(geometry.name for geometry in GEOMETRIES)),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--benchmark-tokens",
        type=int,
        nargs="+",
        help="token counts for benchmark mode; defaults to selector boundaries",
    )
    parser.add_argument(
        "--sanitizer-tokens",
        type=int,
        nargs="+",
        help="token counts for sanitizer mode; defaults to selector boundaries",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--paired-rounds",
        type=int,
        default=2,
        help="even number of alternating ABBA/BAAB exported/baseline rounds",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.benchmark_tokens is not None and any(
        token < 1 or token > MAX_TOKENS for token in args.benchmark_tokens
    ):
        parser.error(f"--benchmark-tokens values must be in [1, {MAX_TOKENS}]")
    if args.sanitizer_tokens is not None and any(
        token < 1 or token > MAX_TOKENS for token in args.sanitizer_tokens
    ):
        parser.error(f"--sanitizer-tokens values must be in [1, {MAX_TOKENS}]")
    if args.warmup < 1 or args.repetitions < 1:
        parser.error("--warmup and --repetitions must be positive")
    if args.paired_rounds < 2 or args.paired_rounds % 2:
        parser.error("--paired-rounds must be an even integer of at least 2")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    target = _validate_environment(device)
    major, minor = torch.cuda.get_device_capability(device)
    geometries = _selected_geometries(args.geometry)
    report: dict[str, Any] = {
        "device": str(device),
        "compute_capability": f"sm_{major}{minor}",
        "target": target,
        "geometry": args.geometry,
        "seed": args.seed,
    }
    if args.mode == "benchmark":
        report["benchmark"] = run_benchmark(
            geometries,
            seed=args.seed,
            benchmark_tokens=args.benchmark_tokens,
            warmup=args.warmup,
            repetitions=args.repetitions,
            paired_rounds=args.paired_rounds,
        )
    if args.mode == "sanitizer":
        report["sanitizer"] = run_sanitizer(
            geometries,
            seed=args.seed,
            sanitizer_tokens=args.sanitizer_tokens,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    benchmark = report.get("benchmark")
    if benchmark is not None and benchmark["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
