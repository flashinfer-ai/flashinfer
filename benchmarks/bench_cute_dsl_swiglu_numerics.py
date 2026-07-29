# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare current and strict CuTe DSL SwiGLU arithmetic with a Megatron mirror.

This is an activation-only numerical experiment. Inputs use a plain contiguous
``[model_hidden, 2 * expert_hidden]`` numeric grid with gate values in the first
half and up values in the second half, matching Megatron's ``torch.chunk``
order. The model-hidden extent is only a requested workload dimension here; in
the real MoE path it is the excluded GEMM K dimension, not a post-FC1 activation
axis. The experiment does not exercise GEMM, quantization, routing, permutation,
combine, or a FlashInfer MoE tensor layout. It mirrors Megatron's two-line
SwiGLU expression locally and does not import or depend on Megatron.

The default plain activation shape is DeepSeek-V3-sized:
``[7168, 2 * 2048]`` input and ``[7168, 2048]`` output.

Example:

    python benchmarks/bench_cute_dsl_swiglu_numerics.py \
        --dtype all \
        --case all \
        --json-out /sgl-workspace/logs/<run-id>/results.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass.cute.typing import Float32, Int32

from flashinfer.cute_dsl.fp4_common import fadd_rn, fmul_rn
from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    _strict_swiglu_f32,
)
from flashinfer.cute_dsl.utils import (
    current_cuda_stream,
    is_cute_dsl_available,
    torch_to_cutlass_dtype,
)


_LOG2_E = 1.4426950408889634
_THREADS = 256
_DEEPSEEK_V3_MODEL_HIDDEN = 7168
_DEEPSEEK_V3_EXPERT_HIDDEN = 2048
_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
_CASES = ("edge", "sweep", "normal", "wide")
_EDGE_GATES = (
    -20.0,
    -10.0,
    -5.0,
    -1.0,
    -0.5,
    -(2.0**-20),
    -0.0,
    0.0,
    2.0**-20,
    0.5,
    1.0,
    5.0,
    10.0,
    20.0,
)
_EDGE_UPS = (
    -4.0,
    -1.0,
    -0.25,
    0.25,
    1.0,
    4.0,
    -1.0,
    1.0,
    -4.0,
    4.0,
    -1.0,
    1.0,
    -0.25,
    0.25,
)


class _SwiGLUNumericsKernel:
    """Plain-linear pointwise kernel emitting current-fast and strict results."""

    def __init__(
        self,
        *,
        rows: int,
        width: int,
        element_dtype: type[cutlass.Numeric],
    ) -> None:
        self.rows = rows
        self.width = width
        self.numel = rows * width
        self.element_dtype = element_dtype

    @cute.jit
    def __call__(
        self,
        fc1_flat: cute.Tensor,
        fast_output_flat: cute.Tensor,
        strict_output_flat: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        grid = ((self.numel + _THREADS - 1) // _THREADS, 1, 1)
        self.kernel(fc1_flat, fast_output_flat, strict_output_flat).launch(
            grid=grid,
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        fc1_flat: cute.Tensor,
        fast_output_flat: cute.Tensor,
        strict_output_flat: cute.Tensor,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        idx = Int32(block_idx) * Int32(_THREADS) + Int32(thread_idx)
        if idx < Int32(self.numel):
            row = idx // Int32(self.width)
            col = idx - row * Int32(self.width)
            row_base = row * Int32(2 * self.width)
            gate = fc1_flat[row_base + col].to(Float32)
            up = fc1_flat[row_base + Int32(self.width) + col].to(Float32)

            # Mirror the current SM100 vectorized MoE epilogue:
            # exp2(..., fastmath=True), approximate reciprocal, then two
            # round-to-nearest FP32 multiplies.
            fast_exp_arg = fmul_rn(gate, Float32(-_LOG2_E))
            fast_exp = cute.math.exp2(fast_exp_arg, fastmath=True)
            fast_denom = fadd_rn(fast_exp, Float32(1.0))
            fast_sigmoid = cute.arch.rcp_approx(fast_denom)
            fast_silu = fmul_rn(fast_sigmoid, gate)
            fast_result = fmul_rn(fast_silu, up)

            # Production strict arithmetic. This follows Megatron/PyTorch's
            # SiLU operation order and avoids both fast exp2 and rcp_approx.
            strict_result = _strict_swiglu_f32(
                gate,
                up,
                Float32(1.0),
                Float32(0.0),
            )

            fast_output_flat[idx] = self.element_dtype(fast_result)
            strict_output_flat[idx] = self.element_dtype(strict_result)


def _compile_kernel(rows: int, width: int, dtype: torch.dtype):
    element_dtype = torch_to_cutlass_dtype(dtype)
    input_fake = cute.runtime.make_fake_compact_tensor(
        element_dtype,
        (rows * 2 * width,),
        assumed_align=16,
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        element_dtype,
        (rows * width,),
        assumed_align=16,
    )
    kernel = _SwiGLUNumericsKernel(
        rows=rows,
        width=width,
        element_dtype=element_dtype,
    )
    return cute.compile(
        kernel,
        input_fake,
        output_fake,
        output_fake,
        current_cuda_stream(),
    )


def _make_case(
    case: str,
    *,
    rows: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    numel = rows * width
    if case == "edge":
        gate_pattern = torch.tensor(_EDGE_GATES, dtype=torch.float32)
        up_pattern = torch.tensor(_EDGE_UPS, dtype=torch.float32)
        pattern_idx = torch.arange(numel) % gate_pattern.numel()
        gate = gate_pattern[pattern_idx]
        up = up_pattern[pattern_idx]
    elif case == "sweep":
        gate = torch.linspace(-20.0, 20.0, numel, dtype=torch.float32)
        up_pattern = torch.tensor(
            (-4.0, -1.0, -0.25, 0.25, 1.0, 4.0),
            dtype=torch.float32,
        )
        up = up_pattern[torch.arange(numel) % up_pattern.numel()]
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + _CASES.index(case))
        gate = torch.randn(numel, generator=generator, dtype=torch.float32)
        up = torch.randn(numel, generator=generator, dtype=torch.float32)
        if case == "wide":
            gate.mul_(6.0)
            up.mul_(3.0)

    gate = gate.reshape(rows, width).to(dtype=dtype, device=device)
    up = up.reshape(rows, width).to(dtype=dtype, device=device)
    return gate, up


def _megatron_swiglu_reference(fc1: torch.Tensor) -> torch.Tensor:
    """Mirror Megatron's dependency-free SwiGLU forward expression."""
    gate, up = torch.chunk(fc1, 2, dim=-1)
    return F.silu(gate) * up


def _ordered_float_bits(value: torch.Tensor) -> torch.Tensor:
    """Map finite FP32/BF16 values to monotonic integers for ULP distance."""
    if value.dtype == torch.float32:
        raw = value.contiguous().view(torch.int32).to(torch.int64)
        sign_mask = 0x80000000
        full_mask = 0xFFFFFFFF
    elif value.dtype == torch.bfloat16:
        raw = value.contiguous().view(torch.int16).to(torch.int64)
        sign_mask = 0x8000
        full_mask = 0xFFFF
    else:
        raise TypeError(f"ULP distance is not implemented for {value.dtype}")

    raw = raw & full_mask
    magnitude = raw & (sign_mask - 1)
    # Sign-stripped magnitude maps +0 and -0 to the same ordered value and
    # keeps adjacent finite values one integer apart across zero.
    return torch.where(
        (raw & sign_mask) != 0,
        sign_mask - magnitude,
        sign_mask + magnitude,
    )


def _error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> dict[str, Any]:
    actual_f64 = actual.to(torch.float64).flatten()
    reference_f64 = reference.to(torch.float64).flatten()
    error = (actual_f64 - reference_f64).abs()
    finite = torch.isfinite(actual_f64) & torch.isfinite(reference_f64)
    if not bool(finite.all()):
        raise RuntimeError(
            "Non-finite output encountered for finite experiment inputs: "
            f"{int((~finite).sum().item())} elements"
        )

    squared_error = error.square()
    reference_norm = torch.linalg.vector_norm(reference_f64)
    relative_l2 = torch.linalg.vector_norm(error) / reference_norm
    quantiles = torch.quantile(
        error,
        torch.tensor(
            (0.5, 0.95, 0.99, 0.999),
            dtype=error.dtype,
            device=error.device,
        ),
    )
    worst_idx = int(error.argmax().item())
    exact = actual.flatten() == reference.flatten()
    ulp_error = (
        _ordered_float_bits(actual.flatten()) - _ordered_float_bits(reference.flatten())
    ).abs()
    within_one_ulp = ulp_error <= 1

    return {
        "numel": actual.numel(),
        "mismatch_count": int((~exact).sum().item()),
        "exact_fraction": float(exact.to(torch.float64).mean().item()),
        "mean_abs": float(error.mean().item()),
        "rmse": float(squared_error.mean().sqrt().item()),
        "max_abs": float(error[worst_idx].item()),
        "relative_l2": float(relative_l2.item()),
        "abs_error_quantiles": {
            "p50": float(quantiles[0].item()),
            "p95": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "p999": float(quantiles[3].item()),
        },
        "ulp_error": {
            "max": int(ulp_error.max().item()),
            "mean": float(ulp_error.to(torch.float64).mean().item()),
            "within_one_count": int(within_one_ulp.sum().item()),
            "within_one_fraction": float(
                within_one_ulp.to(torch.float64).mean().item()
            ),
        },
        "worst": {
            "flat_index": worst_idx,
            "gate": float(gate.flatten()[worst_idx].item()),
            "up": float(up.flatten()[worst_idx].item()),
            "reference": float(reference.flatten()[worst_idx].item()),
            "actual": float(actual.flatten()[worst_idx].item()),
        },
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _gap_summary(
    fast: torch.Tensor,
    strict: torch.Tensor,
    reference: torch.Tensor,
    fast_metrics: dict[str, Any],
    strict_metrics: dict[str, Any],
) -> dict[str, Any]:
    fast_exact = fast == reference
    strict_exact = strict == reference
    strict_better = (
        strict_metrics["rmse"] < fast_metrics["rmse"]
        and strict_metrics["relative_l2"] < fast_metrics["relative_l2"]
        and strict_metrics["max_abs"] <= fast_metrics["max_abs"]
    )
    fast_gap_present = fast_metrics["mismatch_count"] > 0
    strict_value_matches = strict_metrics["mismatch_count"] == 0
    return {
        "recovered_exact_count": int(((~fast_exact) & strict_exact).sum().item()),
        "regressed_exact_count": int((fast_exact & (~strict_exact)).sum().item()),
        "fast_approximation_gap_present": fast_gap_present,
        "strict_value_matches_reference": strict_value_matches,
        "observed_fast_gap_closed_by_strict": (
            strict_value_matches if fast_gap_present else None
        ),
        "strict_improved_all_primary_metrics": strict_better,
        "fast_over_strict_rmse": _ratio(fast_metrics["rmse"], strict_metrics["rmse"]),
        "fast_over_strict_relative_l2": _ratio(
            fast_metrics["relative_l2"], strict_metrics["relative_l2"]
        ),
        "fast_over_strict_max_abs": _ratio(
            fast_metrics["max_abs"], strict_metrics["max_abs"]
        ),
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@torch.no_grad()
def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not is_cute_dsl_available():
        raise RuntimeError("nvidia-cutlass-dsl is required for this experiment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            f"This experiment targets B200 (SM100), got compute capability {capability}"
        )

    selected_dtypes = (
        tuple(_DTYPES.items())
        if args.dtype == "all"
        else ((args.dtype, _DTYPES[args.dtype]),)
    )
    selected_cases = _CASES if args.case == "all" else (args.case,)

    result: dict[str, Any] = {
        "schema_version": 3,
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvidia_cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
        },
        "config": {
            "model_hidden": args.model_hidden,
            "expert_hidden": args.expert_hidden,
            "input_shape": [args.model_hidden, 2 * args.expert_hidden],
            "output_shape": [args.model_hidden, args.expert_hidden],
            "numel_per_half": args.model_hidden * args.expert_hidden,
            "shape_semantics": (
                "activation-only numeric grid; model_hidden is a workload "
                "extent, while the real post-FC1 leading axis is token/expert rows"
            ),
            "seed": args.seed,
            "dtypes": [name for name, _ in selected_dtypes],
            "cases": list(selected_cases),
            "layout": "plain contiguous [gate, up] halves",
            "reference": "torch.nn.functional.silu(first_half) * second_half",
            "closure_domain": (
                "FP32 only; BF16 is a source-expression/storage diagnostic"
            ),
        },
        "cases": [],
    }

    for dtype_name, dtype in selected_dtypes:
        compiled = _compile_kernel(
            args.model_hidden,
            args.expert_hidden,
            dtype,
        )
        for case in selected_cases:
            gate, up = _make_case(
                case,
                rows=args.model_hidden,
                width=args.expert_hidden,
                dtype=dtype,
                device=device,
                seed=args.seed,
            )
            fc1 = torch.cat((gate, up), dim=-1).contiguous()
            fast = torch.empty_like(gate)
            strict = torch.empty_like(gate)

            reference = _megatron_swiglu_reference(fc1)
            compiled(
                fc1.view(-1),
                fast.view(-1),
                strict.view(-1),
                current_cuda_stream(),
            )
            torch.cuda.synchronize(device)

            fast_metrics = _error_metrics(
                fast,
                reference,
                gate=gate,
                up=up,
            )
            strict_metrics = _error_metrics(
                strict,
                reference,
                gate=gate,
                up=up,
            )
            case_result = {
                "dtype": dtype_name,
                "case": case,
                "current_fast_vs_reference": fast_metrics,
                "strict_vs_reference": strict_metrics,
                "gap": _gap_summary(
                    fast,
                    strict,
                    reference,
                    fast_metrics,
                    strict_metrics,
                ),
            }
            result["cases"].append(case_result)
            print(json.dumps({"event": "case_result", **case_result}, sort_keys=True))

    fp32_cases = [case for case in result["cases"] if case["dtype"] == "float32"]
    fp32_gapped_cases = [
        case for case in fp32_cases if case["gap"]["fast_approximation_gap_present"]
    ]
    result["fp32_closure"] = {
        "evaluated": bool(fp32_cases),
        "observed_fast_gap_case_count": len(fp32_gapped_cases),
        "strict_value_matches_reference_on_all_cases": (
            all(case["gap"]["strict_value_matches_reference"] for case in fp32_cases)
            if fp32_cases
            else None
        ),
        "all_observed_fast_gaps_closed": (
            all(
                case["gap"]["observed_fast_gap_closed_by_strict"]
                for case in fp32_gapped_cases
            )
            if fp32_gapped_cases
            else None
        ),
        "fast_gap_demonstrated_and_closed": bool(fp32_gapped_cases)
        and all(
            case["gap"]["observed_fast_gap_closed_by_strict"]
            for case in fp32_gapped_cases
        )
        and all(case["gap"]["strict_value_matches_reference"] for case in fp32_cases),
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-hidden",
        type=int,
        default=_DEEPSEEK_V3_MODEL_HIDDEN,
        help=(
            "Requested leading workload extent (default: DeepSeek-V3 model hidden size)"
        ),
    )
    parser.add_argument(
        "--expert-hidden",
        type=int,
        default=_DEEPSEEK_V3_EXPERT_HIDDEN,
        help="Gate/up width (default: DeepSeek-V3 expert hidden size)",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("all", *_DTYPES),
        default="all",
    )
    parser.add_argument(
        "--case",
        choices=("all", *_CASES),
        default="all",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for the complete machine-readable result.",
    )
    args = parser.parse_args()
    if args.model_hidden <= 0 or args.expert_hidden <= 0:
        parser.error("--model-hidden and --expert-hidden must be positive")
    return args


def main() -> None:
    args = _parse_args()
    result = _run(args)
    print(json.dumps({"event": "summary", **result}, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
