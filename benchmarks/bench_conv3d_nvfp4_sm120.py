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
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from flashinfer import conv3d_nvfp4, prepare_nvfp4_conv3d_weight
from flashinfer.conv.nvfp4 import _quantize_nvfp4_conv3d_activation
from flashinfer.conv.nvfp4_sm120 import run_sm120_nvfp4_conv3d


_PADDING = (0, 1, 1)
_VALUE = 0.125


@dataclass(frozen=True)
class Conv3dCase:
    input_channels: int
    output_channels: int
    depth: int
    height: int
    width: int
    calls: int

    @property
    def name(self) -> str:
        return (
            f"c{self.input_channels}_k{self.output_channels}_d{self.depth}_"
            f"h{self.height}_w{self.width}"
        )

    @property
    def input_shape(self) -> tuple[int, int, int, int, int]:
        return (1, self.input_channels, self.depth, self.height, self.width)

    @property
    def weight_shape(self) -> tuple[int, int, int, int, int]:
        return (self.output_channels, self.input_channels, 3, 3, 3)

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        return (
            1,
            self.output_channels,
            self.depth - 2,
            self.height,
            self.width,
        )

    @property
    def flops_per_call(self) -> int:
        _, output_channels, output_depth, output_height, output_width = (
            self.output_shape
        )
        return (
            2
            * output_depth
            * output_height
            * output_width
            * output_channels
            * self.input_channels
            * 27
        )


_CASES = (
    Conv3dCase(512, 512, 6, 176, 320, 100),
    Conv3dCase(256, 256, 6, 352, 640, 100),
    Conv3dCase(1024, 1024, 4, 88, 160, 120),
    Conv3dCase(1024, 512, 6, 176, 320, 20),
    Conv3dCase(512, 256, 6, 352, 640, 20),
    Conv3dCase(1024, 1024, 3, 44, 80, 210),
    Conv3dCase(1024, 1024, 3, 88, 160, 6),
    Conv3dCase(512, 512, 3, 176, 320, 5),
    Conv3dCase(256, 256, 3, 352, 640, 5),
    Conv3dCase(1024, 512, 3, 176, 320, 1),
    Conv3dCase(512, 256, 3, 352, 640, 1),
)


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _summarize_samples(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "median_us": statistics.median(samples_ms) * 1000.0,
        "p95_us": _percentile(samples_ms, 0.95) * 1000.0,
        "min_us": min(samples_ms) * 1000.0,
        "max_us": max(samples_ms) * 1000.0,
        "samples_us": [sample * 1000.0 for sample in samples_ms],
    }


def _benchmark(
    launch: Callable[[], Any],
    *,
    warmup_iterations: int,
    iterations: int,
    repeats: int,
    cold_l2: bool,
    l2_flush: torch.Tensor,
) -> dict[str, Any]:
    for _ in range(warmup_iterations):
        launch()
    torch.cuda.synchronize()

    samples_ms = []
    if cold_l2:
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(repeats * iterations):
            l2_flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        samples_ms = [start.elapsed_time(end) for start, end in events]
    else:
        events = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                launch()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        samples_ms = [start.elapsed_time(end) / iterations for start, end in events]
    result = _summarize_samples(samples_ms)
    result.update(
        {
            "warmup_iterations": warmup_iterations,
            "iterations": iterations,
            "repeats": repeats,
            "l2": "cold" if cold_l2 else "warm",
        }
    )
    return result


def _benchmark_warm_and_cold(
    launch: Callable[[], Any],
    *,
    args: argparse.Namespace,
    l2_flush: torch.Tensor,
) -> dict[str, Any]:
    result = {
        "warm_l2": _benchmark(
            launch,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            repeats=args.repeats,
            cold_l2=False,
            l2_flush=l2_flush,
        ),
    }
    if not args.warm_only:
        result["cold_l2"] = _benchmark(
            launch,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            repeats=args.repeats,
            cold_l2=True,
            l2_flush=l2_flush,
        )
    return result


def _prepare_cudnn_graph(graph: Any) -> None:
    import cudnn

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.ALL)


def _select_cudnn_plan(
    graph: Any,
    variant_pack: dict[Any, torch.Tensor],
    handle: Any,
    *,
    args: argparse.Namespace,
    l2_flush: torch.Tensor,
) -> tuple[Callable[[], None], dict[str, Any], torch.Tensor]:
    candidates = []
    selected: tuple[float, int, str, int, torch.Tensor] | None = None
    for index in range(int(graph.get_execution_plan_count())):
        name = graph.get_plan_name_at_index(index)
        workspace_bytes = int(graph.get_workspace_size_plan_at_index(index))
        try:
            workspace = torch.empty(
                workspace_bytes,
                device="cuda",
                dtype=torch.uint8,
            )

            def launch(
                index: int = index,
                workspace: torch.Tensor = workspace,
            ) -> None:
                graph.execute_plan_at_index(
                    variant_pack,
                    workspace,
                    index=index,
                    handle=handle,
                )

            timing = _benchmark(
                launch,
                warmup_iterations=args.tune_warmup_iterations,
                iterations=args.tune_iterations,
                repeats=1,
                cold_l2=False,
                l2_flush=l2_flush,
            )
            latency_us = float(timing["median_us"])
            candidates.append(
                {
                    "index": index,
                    "name": name,
                    "workspace_bytes": workspace_bytes,
                    "median_us": latency_us,
                    "succeeded": True,
                }
            )
            if selected is None or latency_us < selected[0]:
                selected = (
                    latency_us,
                    index,
                    name,
                    workspace_bytes,
                    workspace,
                )
        except Exception as error:
            candidates.append(
                {
                    "index": index,
                    "name": name,
                    "workspace_bytes": workspace_bytes,
                    "succeeded": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    if selected is None:
        raise RuntimeError("no cuDNN execution plan completed")
    _, selected_index, selected_name, workspace_bytes, workspace = selected

    def selected_launch() -> None:
        graph.execute_plan_at_index(
            variant_pack,
            workspace,
            index=selected_index,
            handle=handle,
        )

    plan = {
        "candidate_count": len(candidates),
        "selected_index": selected_index,
        "selected_name": selected_name,
        "workspace_bytes": workspace_bytes,
        "candidates": candidates,
    }
    return selected_launch, plan, workspace


def _benchmark_cudnn_16bit(
    case: Conv3dCase,
    handle: Any,
    *,
    torch_dtype: torch.dtype,
    cudnn_dtype: Any,
    label: str,
    args: argparse.Namespace,
    l2_flush: torch.Tensor,
) -> dict[str, Any]:
    import cudnn

    input = torch.empty(
        case.input_shape,
        device="cuda",
        dtype=torch_dtype,
        memory_format=torch.channels_last_3d,
    )
    weight = torch.empty(
        case.weight_shape,
        device="cuda",
        dtype=torch_dtype,
        memory_format=torch.channels_last_3d,
    )
    input.fill_(_VALUE)
    weight.fill_(_VALUE)
    output = torch.empty(
        case.output_shape,
        device="cuda",
        dtype=torch_dtype,
        memory_format=torch.channels_last_3d,
    )
    graph = cudnn.pygraph(
        io_data_type=cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    input_uid = graph.tensor(
        name="input",
        dim=list(input.shape),
        stride=list(input.stride()),
        data_type=cudnn_dtype,
    )
    weight_uid = graph.tensor(
        name="weight",
        dim=list(weight.shape),
        stride=list(weight.stride()),
        data_type=cudnn_dtype,
    )
    output_uid = graph.conv_fprop(
        image=input_uid,
        weight=weight_uid,
        padding=_PADDING,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
        name=f"conv3d_{label}",
    )
    output_uid.set_output(True).set_data_type(cudnn_dtype)
    _prepare_cudnn_graph(graph)
    launch, plan, workspace = _select_cudnn_plan(
        graph,
        {input_uid: input, weight_uid: weight, output_uid: output},
        handle,
        args=args,
        l2_flush=l2_flush,
    )
    timing = _benchmark_warm_and_cold(
        launch,
        args=args,
        l2_flush=l2_flush,
    )
    expected = case.input_channels * 27 * _VALUE * _VALUE
    center = float(
        output[
            0,
            0,
            output.shape[2] // 2,
            output.shape[3] // 2,
            output.shape[4] // 2,
        ]
    )
    result = {
        "backend": "cudnn_frontend_graph",
        "dtype": label,
        "timing": timing,
        "plan": plan,
        "validation": {
            "center_value": center,
            "expected_center_value": expected,
            "passed": math.isclose(
                center,
                expected,
                rel_tol=0.01,
                abs_tol=0.125,
            )
            and bool(torch.isfinite(output).all()),
        },
    }
    del graph, workspace, input, weight, output
    return result


def _benchmark_cudnn_fp8(
    case: Conv3dCase,
    handle: Any,
    *,
    args: argparse.Namespace,
    l2_flush: torch.Tensor,
) -> dict[str, Any]:
    import cudnn

    input_fp8 = torch.empty(
        case.input_shape,
        device="cuda",
        dtype=torch.float8_e4m3fn,
        memory_format=torch.channels_last_3d,
    )
    weight_fp8 = torch.empty(
        case.weight_shape,
        device="cuda",
        dtype=torch.float8_e4m3fn,
        memory_format=torch.channels_last_3d,
    )
    input_fp8.fill_(_VALUE)
    weight_fp8.fill_(_VALUE)
    input_storage = input_fp8.view(torch.int8)
    weight_storage = weight_fp8.view(torch.int8)
    input_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    weight_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    output = torch.empty(
        case.output_shape,
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    )
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FAST_FLOAT_FOR_FP8,
        handle=handle,
    )
    input_uid = graph.tensor(
        name="input_fp8",
        dim=list(input_storage.shape),
        stride=list(input_storage.stride()),
        data_type=cudnn.data_type.FP8_E4M3,
    )
    weight_uid = graph.tensor(
        name="weight_fp8",
        dim=list(weight_storage.shape),
        stride=list(weight_storage.stride()),
        data_type=cudnn.data_type.FP8_E4M3,
    )
    scalar_dim = [1, 1, 1, 1, 1]
    scalar_stride = [1, 1, 1, 1, 1]
    input_descale_uid = graph.tensor(
        name="input_descale",
        dim=scalar_dim,
        stride=scalar_stride,
        data_type=cudnn.data_type.FLOAT,
    )
    weight_descale_uid = graph.tensor(
        name="weight_descale",
        dim=scalar_dim,
        stride=scalar_stride,
        data_type=cudnn.data_type.FLOAT,
    )
    conv_uid = graph.conv_fprop(
        image=input_uid,
        weight=weight_uid,
        padding=_PADDING,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        compute_data_type=cudnn.data_type.FAST_FLOAT_FOR_FP8,
        name="conv3d_fp8",
    )
    after_input_descale = graph.mul(
        conv_uid,
        input_descale_uid,
        compute_data_type=cudnn.data_type.FLOAT,
        name="input_descale_mul",
    )
    output_uid = graph.mul(
        after_input_descale,
        weight_descale_uid,
        compute_data_type=cudnn.data_type.FLOAT,
        name="weight_descale_mul",
    )
    output_uid.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    _prepare_cudnn_graph(graph)
    launch, plan, workspace = _select_cudnn_plan(
        graph,
        {
            input_uid: input_storage,
            weight_uid: weight_storage,
            input_descale_uid: input_descale,
            weight_descale_uid: weight_descale,
            output_uid: output,
        },
        handle,
        args=args,
        l2_flush=l2_flush,
    )
    timing = _benchmark_warm_and_cold(
        launch,
        args=args,
        l2_flush=l2_flush,
    )
    expected = case.input_channels * 27 * _VALUE * _VALUE
    center = float(
        output[
            0,
            0,
            output.shape[2] // 2,
            output.shape[3] // 2,
            output.shape[4] // 2,
        ]
    )
    result = {
        "backend": "cudnn_frontend_fused_graph",
        "timing": timing,
        "plan": plan,
        "validation": {
            "center_value": center,
            "expected_center_value": expected,
            "passed": math.isclose(
                center,
                expected,
                rel_tol=0.01,
                abs_tol=0.125,
            )
            and bool(torch.isfinite(output).all()),
        },
    }
    del graph, workspace, input_fp8, weight_fp8, output
    return result


def _benchmark_nvfp4(
    case: Conv3dCase,
    *,
    args: argparse.Namespace,
    l2_flush: torch.Tensor,
) -> dict[str, Any]:
    input = torch.full(
        case.input_shape,
        _VALUE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.full(
        case.weight_shape,
        _VALUE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    input_global_scale = torch.tensor(
        [448.0 * 6.0 / _VALUE],
        device="cuda",
        dtype=torch.float32,
    )

    torch.cuda.synchronize()
    prepare_started = time.perf_counter()
    packed_weight, weight_scale, weight_global_scale = prepare_nvfp4_conv3d_weight(
        weight
    )
    torch.cuda.synchronize()
    weight_prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

    packed_input, input_scale = _quantize_nvfp4_conv3d_activation(
        input,
        input_global_scale,
        _PADDING,
    )
    raw_no_alpha_output = torch.empty(
        (
            1,
            case.depth - 2,
            case.height,
            case.width,
            case.output_channels,
        ),
        device="cuda",
        dtype=torch.bfloat16,
    )
    raw_with_alpha_output = torch.empty_like(raw_no_alpha_output)
    full_output = torch.empty(
        case.output_shape,
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    )
    alpha = torch.reciprocal(input_global_scale * weight_global_scale)

    def raw_no_alpha_launch() -> None:
        run_sm120_nvfp4_conv3d(
            packed_input,
            packed_weight,
            input_scale,
            weight_scale,
            alpha,
            raw_no_alpha_output,
            fuse_alpha=False,
            fuse_bias=False,
        )

    torch.cuda.synchronize()
    first_raw_started = time.perf_counter()
    raw_no_alpha_launch()
    torch.cuda.synchronize()
    first_raw_wall_ms = (time.perf_counter() - first_raw_started) * 1000.0

    def raw_with_alpha_launch() -> None:
        run_sm120_nvfp4_conv3d(
            packed_input,
            packed_weight,
            input_scale,
            weight_scale,
            alpha,
            raw_with_alpha_output,
            fuse_alpha=True,
            fuse_bias=False,
        )

    producer_packed = torch.empty_like(packed_input)
    producer_scale = torch.empty_like(input_scale)

    def producer_launch() -> None:
        _quantize_nvfp4_conv3d_activation(
            input,
            input_global_scale,
            _PADDING,
            packed_out=producer_packed,
            scale_out=producer_scale,
        )

    def full_launch() -> None:
        conv3d_nvfp4(
            input,
            packed_weight,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            padding=_PADDING,
            out=full_output,
        )

    raw_no_alpha_timing = _benchmark_warm_and_cold(
        raw_no_alpha_launch,
        args=args,
        l2_flush=l2_flush,
    )
    raw_with_alpha_timing = _benchmark_warm_and_cold(
        raw_with_alpha_launch,
        args=args,
        l2_flush=l2_flush,
    )
    producer_timing = _benchmark_warm_and_cold(
        producer_launch,
        args=args,
        l2_flush=l2_flush,
    )
    full_timing = _benchmark_warm_and_cold(
        full_launch,
        args=args,
        l2_flush=l2_flush,
    )
    expected = case.input_channels * 27 * _VALUE * _VALUE
    raw_no_alpha_center = float(
        raw_no_alpha_output[
            0,
            raw_no_alpha_output.shape[1] // 2,
            raw_no_alpha_output.shape[2] // 2,
            raw_no_alpha_output.shape[3] // 2,
            0,
        ]
    )
    raw_with_alpha_center = float(
        raw_with_alpha_output[
            0,
            raw_with_alpha_output.shape[1] // 2,
            raw_with_alpha_output.shape[2] // 2,
            raw_with_alpha_output.shape[3] // 2,
            0,
        ]
    )
    full_center = float(
        full_output[
            0,
            0,
            full_output.shape[2] // 2,
            full_output.shape[3] // 2,
            full_output.shape[4] // 2,
        ]
    )
    validation = {
        "raw_no_alpha_center_value": raw_no_alpha_center,
        "raw_no_alpha_center_after_alpha": (raw_no_alpha_center * float(alpha.item())),
        "raw_with_alpha_center_value": raw_with_alpha_center,
        "full_center_value": full_center,
        "expected_center_value": expected,
        "raw_no_alpha_finite": bool(torch.isfinite(raw_no_alpha_output).all()),
        "raw_with_alpha_finite": bool(torch.isfinite(raw_with_alpha_output).all()),
        "full_finite": bool(torch.isfinite(full_output).all()),
    }
    validation["passed"] = (
        math.isclose(
            validation["raw_no_alpha_center_after_alpha"],
            expected,
            rel_tol=0.01,
            abs_tol=0.125,
        )
        and math.isclose(
            raw_with_alpha_center,
            expected,
            rel_tol=0.01,
            abs_tol=0.125,
        )
        and math.isclose(
            full_center,
            expected,
            rel_tol=0.01,
            abs_tol=0.125,
        )
        and validation["raw_no_alpha_finite"]
        and validation["raw_with_alpha_finite"]
        and validation["full_finite"]
    )
    result = {
        "raw_kernel_no_alpha": {
            "timing": raw_no_alpha_timing,
            "first_call_wall_ms": first_raw_wall_ms,
        },
        "raw_kernel_with_alpha": {
            "timing": raw_with_alpha_timing,
        },
        "activation_producer": {
            "timing": producer_timing,
            "output_buffers_reused": True,
        },
        "full_operation": {
            "timing": full_timing,
            "output_buffer_reused": True,
            "includes_activation_preparation": True,
        },
        "weight_preparation": {
            "wall_ms": weight_prepare_ms,
            "steady_state": False,
        },
        "validation": validation,
    }
    return result


def _aggregate(cases: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    weighted_us = 0.0
    total_flops = 0
    for case in cases:
        value: Any = case
        for key in path:
            value = value[key]
        weighted_us += case["calls"] * float(value)
        total_flops += case["calls"] * case["flops_per_call"]
    return {
        "weighted_ms": weighted_us / 1000.0,
        "effective_tflops": total_flops / (weighted_us * 1.0e6),
    }


def _select_cases(names: list[str]) -> tuple[Conv3dCase, ...]:
    if not names:
        return _CASES
    by_name = {case.name: case for case in _CASES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; choices are {sorted(by_name)}")
    return tuple(by_name[name] for name in names)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SM120 NVFP4 Conv3d against cuDNN baselines."
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tune-warmup-iterations", type=int, default=1)
    parser.add_argument("--tune-iterations", type=int, default=3)
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument(
        "--warm-only",
        action="store_true",
        help="Skip cold-L2 measurements.",
    )
    parser.add_argument("--skip-cudnn", action="store_true")
    parser.add_argument(
        "--fp16-only",
        action="store_true",
        help="Benchmark only the cuDNN FP16 baseline.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 is required")
    if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
        raise RuntimeError("CUDA 13 or newer is required")
    if args.fp16_only and args.skip_cudnn:
        raise ValueError("--fp16-only and --skip-cudnn are mutually exclusive")
    cases = _select_cases(args.case)
    l2_flush = torch.empty(
        args.l2_flush_mib * 1024 * 1024,
        device="cuda",
        dtype=torch.uint8,
    )

    cudnn_handle = None
    cudnn_version = None
    if not args.skip_cudnn:
        import cudnn

        cudnn_handle = cudnn.create_handle()
        cudnn.set_stream(
            handle=cudnn_handle,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        cudnn_version = {
            "frontend": getattr(cudnn, "__version__", None),
            "backend": cudnn.backend_version_string(),
        }

    results = []
    expected_case_count = len(cases)
    expected_call_count = sum(case.calls for case in cases)
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}", flush=True)
        if args.fp16_only:
            result = {
                "name": case.name,
                "input_shape": list(case.input_shape),
                "weight_shape": list(case.weight_shape),
                "output_shape": list(case.output_shape),
                "calls": case.calls,
                "flops_per_call": case.flops_per_call,
                "cudnn_fp16": _benchmark_cudnn_16bit(
                    case,
                    cudnn_handle,
                    torch_dtype=torch.float16,
                    cudnn_dtype=cudnn.data_type.HALF,
                    label="fp16",
                    args=args,
                    l2_flush=l2_flush,
                ),
            }
            results.append(result)
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "name": case.name,
                        "cudnn_fp16_us": result["cudnn_fp16"]["timing"]["warm_l2"][
                            "median_us"
                        ],
                    }
                ),
                flush=True,
            )
            continue

        nvfp4 = _benchmark_nvfp4(case, args=args, l2_flush=l2_flush)
        result = {
            "name": case.name,
            "input_shape": list(case.input_shape),
            "weight_shape": list(case.weight_shape),
            "output_shape": list(case.output_shape),
            "calls": case.calls,
            "flops_per_call": case.flops_per_call,
            "nvfp4": nvfp4,
        }
        if cudnn_handle is not None:
            result["cudnn_bf16"] = _benchmark_cudnn_16bit(
                case,
                cudnn_handle,
                torch_dtype=torch.bfloat16,
                cudnn_dtype=cudnn.data_type.BFLOAT16,
                label="bf16",
                args=args,
                l2_flush=l2_flush,
            )
            result["cudnn_fp16"] = _benchmark_cudnn_16bit(
                case,
                cudnn_handle,
                torch_dtype=torch.float16,
                cudnn_dtype=cudnn.data_type.HALF,
                label="fp16",
                args=args,
                l2_flush=l2_flush,
            )
            result["cudnn_fp8"] = _benchmark_cudnn_fp8(
                case,
                cudnn_handle,
                args=args,
                l2_flush=l2_flush,
            )
        results.append(result)
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "name": case.name,
                    "nvfp4_raw_no_alpha_us": nvfp4["raw_kernel_no_alpha"]["timing"][
                        "warm_l2"
                    ]["median_us"],
                    "nvfp4_raw_with_alpha_us": nvfp4["raw_kernel_with_alpha"]["timing"][
                        "warm_l2"
                    ]["median_us"],
                    "nvfp4_producer_us": nvfp4["activation_producer"]["timing"][
                        "warm_l2"
                    ]["median_us"],
                    "nvfp4_full_us": nvfp4["full_operation"]["timing"]["warm_l2"][
                        "median_us"
                    ],
                    "cudnn_bf16_us": (
                        result["cudnn_bf16"]["timing"]["warm_l2"]["median_us"]
                        if "cudnn_bf16" in result
                        else None
                    ),
                    "cudnn_fp16_us": (
                        result["cudnn_fp16"]["timing"]["warm_l2"]["median_us"]
                        if "cudnn_fp16" in result
                        else None
                    ),
                    "cudnn_fp8_us": (
                        result["cudnn_fp8"]["timing"]["warm_l2"]["median_us"]
                        if "cudnn_fp8" in result
                        else None
                    ),
                }
            ),
            flush=True,
        )

    if args.fp16_only:
        aggregates = {
            "cudnn_fp16": _aggregate(
                results,
                ("cudnn_fp16", "timing", "warm_l2", "median_us"),
            )
        }
        payload = {
            "schema_version": 1,
            "scope": {
                "backend": "cudnn_fp16",
                "case_count": len(results),
                "calls": sum(case["calls"] for case in results),
                "padding": list(_PADDING),
            },
            "environment": {
                "device": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": cudnn_version,
            },
            "timing": {
                "method": "CUDA events",
                "warmup_iterations": args.warmup_iterations,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "l2_flush_mib": args.l2_flush_mib,
            },
            "cases": results,
            "aggregates": aggregates,
            "validation": {
                "all_outputs_passed": all(
                    case["cudnn_fp16"]["validation"]["passed"] for case in results
                ),
                "all_selected_shapes": len(results) == expected_case_count,
                "all_selected_calls": (
                    sum(case["calls"] for case in results) == expected_call_count
                ),
            },
            "wall_seconds": time.perf_counter() - started,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps({"aggregates": aggregates}, indent=2))
        return 0 if payload["validation"]["all_outputs_passed"] else 1

    aggregates = {
        "nvfp4_raw_no_alpha": _aggregate(
            results,
            (
                "nvfp4",
                "raw_kernel_no_alpha",
                "timing",
                "warm_l2",
                "median_us",
            ),
        ),
        "nvfp4_raw_with_alpha": _aggregate(
            results,
            (
                "nvfp4",
                "raw_kernel_with_alpha",
                "timing",
                "warm_l2",
                "median_us",
            ),
        ),
        "nvfp4_activation_producer": _aggregate(
            results,
            (
                "nvfp4",
                "activation_producer",
                "timing",
                "warm_l2",
                "median_us",
            ),
        ),
        "nvfp4_full": _aggregate(
            results,
            ("nvfp4", "full_operation", "timing", "warm_l2", "median_us"),
        ),
    }
    comparisons = {}
    if cudnn_handle is not None:
        aggregates["cudnn_bf16"] = _aggregate(
            results,
            ("cudnn_bf16", "timing", "warm_l2", "median_us"),
        )
        aggregates["cudnn_fp16"] = _aggregate(
            results,
            ("cudnn_fp16", "timing", "warm_l2", "median_us"),
        )
        aggregates["cudnn_fp8"] = _aggregate(
            results,
            ("cudnn_fp8", "timing", "warm_l2", "median_us"),
        )
        comparisons = {
            "raw_no_alpha_speedup_vs_fp8": (
                aggregates["cudnn_fp8"]["weighted_ms"]
                / aggregates["nvfp4_raw_no_alpha"]["weighted_ms"]
            ),
            "raw_no_alpha_speedup_vs_bf16": (
                aggregates["cudnn_bf16"]["weighted_ms"]
                / aggregates["nvfp4_raw_no_alpha"]["weighted_ms"]
            ),
            "raw_no_alpha_speedup_vs_fp16": (
                aggregates["cudnn_fp16"]["weighted_ms"]
                / aggregates["nvfp4_raw_no_alpha"]["weighted_ms"]
            ),
            "raw_with_alpha_speedup_vs_fp8": (
                aggregates["cudnn_fp8"]["weighted_ms"]
                / aggregates["nvfp4_raw_with_alpha"]["weighted_ms"]
            ),
            "full_speedup_vs_bf16": (
                aggregates["cudnn_bf16"]["weighted_ms"]
                / aggregates["nvfp4_full"]["weighted_ms"]
            ),
            "full_speedup_vs_fp16": (
                aggregates["cudnn_fp16"]["weighted_ms"]
                / aggregates["nvfp4_full"]["weighted_ms"]
            ),
            "raw_no_alpha_meets_2x_fp8_gate": (
                aggregates["cudnn_fp8"]["weighted_ms"]
                >= 2.0 * aggregates["nvfp4_raw_no_alpha"]["weighted_ms"]
            ),
            "full_beats_bf16": (
                aggregates["nvfp4_full"]["weighted_ms"]
                < aggregates["cudnn_bf16"]["weighted_ms"]
            ),
            "full_beats_fp16": (
                aggregates["nvfp4_full"]["weighted_ms"]
                < aggregates["cudnn_fp16"]["weighted_ms"]
            ),
        }

    payload = {
        "schema_version": 1,
        "scope": {
            "case_count": len(results),
            "calls": sum(case["calls"] for case in results),
            "padding": list(_PADDING),
            "raw_no_alpha_excludes_global_scale_epilogue": True,
            "raw_with_alpha_includes_global_scale_epilogue": True,
            "both_raw_paths_exclude_activation_quantization": True,
            "full_includes_activation_quantization": True,
            "weight_preparation_excluded_from_steady_state": True,
        },
        "environment": {
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": cudnn_version,
        },
        "timing": {
            "method": "CUDA events",
            "warmup_iterations": args.warmup_iterations,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "l2_flush_mib": args.l2_flush_mib,
        },
        "cases": results,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "validation": {
            "all_outputs_passed": all(
                case["nvfp4"]["validation"]["passed"]
                and (
                    "cudnn_bf16" not in case
                    or case["cudnn_bf16"]["validation"]["passed"]
                )
                and (
                    "cudnn_fp16" not in case
                    or case["cudnn_fp16"]["validation"]["passed"]
                )
                and (
                    "cudnn_fp8" not in case or case["cudnn_fp8"]["validation"]["passed"]
                )
                for case in results
            ),
            "all_selected_shapes": len(results) == expected_case_count,
            "all_selected_calls": (
                sum(case["calls"] for case in results) == expected_call_count
            ),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregates": aggregates, "comparisons": comparisons}, indent=2))
    return 0 if payload["validation"]["all_outputs_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
