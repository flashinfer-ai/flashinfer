"""Qualify exact Wan hybrid attention against production SGLang FA4.

The candidate timing starts from raw post-RoPE BF16 NHD Q/K/V and includes
value quantization/packing, attention, and materialization into caller-owned
BF16 output. Both implementations run in one process on one CUDA stream with
cold-L2 CUPTI activity timing in forward and reverse paired orders.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from importlib.metadata import version as distribution_version
from typing import Callable

import torch
import torch.nn.functional as F

from flashinfer import (
    WanHybridAttentionWorkspace,
    is_wan_hybrid_attention_available,
    wan_hybrid_attention,
)
from flashinfer.testing import bench_gpu_time


_SHAPE = (1, 4800, 40, 128)
_WARMUP_RUNS = 2
_MEASURE_RUNS = 5
_PAIRED_ORDERS = (("C", "F", "F", "C"), ("F", "C", "C", "F"))


def _require_cupti() -> str:
    try:
        from cupti import cupti as _cupti  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError("cupti-python is required for this benchmark") from error
    cupti_version = distribution_version("cupti-python")
    if int(cupti_version.split(".", maxsplit=1)[0]) < 13:
        raise RuntimeError(f"cupti-python>=13 is required, found {cupti_version}")
    return cupti_version


def _load_production_fa4():
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except ImportError as error:
        raise RuntimeError(
            "the flash-attn-4 package used by production SGLang FA4 is required"
        ) from error
    return _flash_attn_fwd


def _measure_leg(fn: Callable[[], None]) -> list[float]:
    return [
        float(sample)
        for sample in bench_gpu_time(
            fn=fn,
            dry_run_iters=_WARMUP_RUNS,
            repeat_iters=_MEASURE_RUNS,
            enable_cupti=True,
            use_cuda_graph=False,
            cold_l2_cache=True,
        )
    ]


def _measure_order(
    order: tuple[str, str, str, str],
    candidate_fn: Callable[[], None],
    fa4_fn: Callable[[], None],
) -> dict:
    functions = {"C": candidate_fn, "F": fa4_fn}
    pooled = {"C": [], "F": []}
    legs = []
    for leg_index, label in enumerate(order):
        process_id = os.getpid()
        stream_id = int(torch.cuda.current_stream().cuda_stream)
        samples = _measure_leg(functions[label])
        pooled[label].extend(samples)
        legs.append(
            {
                "leg": leg_index,
                "provider": "candidate" if label == "C" else "production_fa4",
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
                "process_id": process_id,
                "stream_id": stream_id,
                "timing_backend": "CUPTI activity span",
                "cold_l2": True,
            }
        )
    candidate_ms = statistics.median(pooled["C"])
    production_fa4_ms = statistics.median(pooled["F"])
    speedup = production_fa4_ms / candidate_ms
    return {
        "order": "/".join(order),
        "legs": legs,
        "candidate_median_ms": candidate_ms,
        "production_fa4_median_ms": production_fa4_ms,
        "candidate_minus_production_fa4_ms": candidate_ms - production_fa4_ms,
        "production_fa4_minus_candidate_ms": production_fa4_ms - candidate_ms,
        "absolute_delta_ms": abs(candidate_ms - production_fa4_ms),
        "speedup": speedup,
        "passed_speedup_ge_1": speedup >= 1.0,
    }


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = (actual_f32 - expected_f32).abs()
    cosine = F.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    ).item()
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "allclose_atol_1_rtol_0_1": bool(
            torch.allclose(actual, expected, atol=1.0, rtol=0.1)
        ),
        "cosine": cosine,
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
        "passed": bool(
            torch.isfinite(actual).all().item()
            and torch.allclose(actual, expected, atol=1.0, rtol=0.1)
            and cosine >= 0.995
            and delta.mean().item() <= 0.025
        ),
    }


def _qualification_passed(
    quality: dict,
    production_fa4_quality: dict,
    repeat_bitwise: bool,
    allocation_stable: bool,
    orders: list[dict],
) -> bool:
    return bool(
        quality["passed"]
        and production_fa4_quality["passed"]
        and repeat_bitwise
        and allocation_stable
        and all(order["passed_speedup_ge_1"] for order in orders)
    )


def _callable_provenance(distribution: str, fn: Callable[..., object]) -> dict:
    module_name = getattr(fn, "__module__", None)
    if not module_name:
        raise RuntimeError(f"{distribution} callable has no module")
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{distribution} callable module has no source file")
    source = Path(module_file).resolve(strict=True)
    if not source.is_file():
        raise RuntimeError(f"{distribution} callable source is not a file: {source}")
    return {
        "distribution": distribution,
        "distribution_version": distribution_version(distribution),
        "callable_module": module_name,
        "callable_qualified_name": getattr(
            fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__)
        ),
        "module_source_path": str(source),
        "module_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _production_fa4_provenance(fn: Callable[..., object]) -> dict:
    return {
        **_callable_provenance("flash-attn-4", fn),
        "sglang_distribution_version": distribution_version("sglang"),
        "sglang_backend": "FA4",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the qualification report to this JSON path",
    )
    args = parser.parse_args()
    started = time.monotonic()
    cupti_version = _require_cupti()
    device = torch.device("cuda", torch.cuda.current_device())
    if not is_wan_hybrid_attention_available(device):
        raise RuntimeError("wan_hybrid attention is unavailable on this device")

    generator = torch.Generator(device=device)
    generator.manual_seed(4254)
    q, k, v = (
        torch.randn(
            _SHAPE,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        for _ in range(3)
    )
    out_candidate = torch.empty_like(q)
    out_fa4 = torch.empty_like(q)
    workspace = WanHybridAttentionWorkspace(device)
    softmax_scale = 1.0 / math.sqrt(_SHAPE[-1])
    production_fa4 = _load_production_fa4()

    def candidate_fn() -> None:
        result = wan_hybrid_attention(
            q,
            k,
            v,
            out=out_candidate,
            workspace=workspace,
            sm_scale=softmax_scale,
            qkv_layout="NHD",
            causal=False,
        )
        if result is not out_candidate or result.data_ptr() != out_candidate.data_ptr():
            raise RuntimeError("candidate did not preserve caller-owned output")

    def fa4_fn() -> None:
        returned_out, returned_lse = production_fa4(
            q,
            k,
            v,
            out=out_fa4,
            pack_gqa=False,
        )
        if returned_out.data_ptr() != out_fa4.data_ptr() or returned_lse is not None:
            raise RuntimeError("production FA4 did not preserve its output contract")

    fa4_fn()
    candidate_fn()
    torch.cuda.synchronize(device)
    first_candidate = out_candidate.clone()
    out_candidate.fill_(float("nan"))
    candidate_fn()
    torch.cuda.synchronize(device)
    repeat_bitwise = bool(torch.equal(out_candidate, first_candidate))

    q_hnd = q.permute(0, 2, 1, 3)
    k_hnd = k.permute(0, 2, 1, 3)
    v_hnd = v.permute(0, 2, 1, 3)
    reference = (
        F.scaled_dot_product_attention(
            q_hnd,
            k_hnd,
            v_hnd,
            scale=softmax_scale,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    quality = _quality(out_candidate, reference)
    fa4_quality = _quality(out_fa4, reference)

    allocated_before = torch.cuda.memory_allocated(device)
    for _ in range(10):
        candidate_fn()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)
    allocation_stable = allocated_after == allocated_before

    orders = [_measure_order(order, candidate_fn, fa4_fn) for order in _PAIRED_ORDERS]
    passed = _qualification_passed(
        quality,
        fa4_quality,
        repeat_bitwise,
        allocation_stable,
        orders,
    )
    candidate_median_ms = statistics.median(
        order["candidate_median_ms"] for order in orders
    )
    production_fa4_median_ms = statistics.median(
        order["production_fa4_median_ms"] for order in orders
    )
    properties = torch.cuda.get_device_properties(device)
    gpu_uuid = getattr(properties, "uuid", None)
    if gpu_uuid is None:
        raise RuntimeError("PyTorch did not expose the GPU UUID")
    report = {
        "passed": passed,
        "shape": list(_SHAPE),
        "layout": "NHD",
        "causal": False,
        "dtype": "bfloat16",
        "seed": 4254,
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "multi_processor_count": properties.multi_processor_count,
        "gpu_uuid": str(gpu_uuid),
        "process_id": os.getpid(),
        "stream": int(torch.cuda.current_stream(device).cuda_stream),
        "timing": {
            "backend": "CUPTI activity span",
            "cold_l2": True,
            "cuda_graph": False,
            "warmup_runs_per_leg": _WARMUP_RUNS,
            "measure_runs_per_leg": _MEASURE_RUNS,
            "candidate_scope": [
                "raw BF16 V quantize and pack",
                "hybrid attention",
                "caller-owned BF16 output materialization",
            ],
        },
        "cupti_python_version": cupti_version,
        "quality_vs_bf16_reference": quality,
        "production_fa4_quality_vs_bf16_reference": fa4_quality,
        "candidate_median_ms": candidate_median_ms,
        "production_fa4_median_ms": production_fa4_median_ms,
        "candidate_minus_production_fa4_ms": (
            candidate_median_ms - production_fa4_median_ms
        ),
        "production_fa4_minus_candidate_ms": (
            production_fa4_median_ms - candidate_median_ms
        ),
        "absolute_delta_ms": abs(candidate_median_ms - production_fa4_median_ms),
        "overall_speedup": production_fa4_median_ms / candidate_median_ms,
        "repeatability_bitwise": repeat_bitwise,
        "allocation_stable": allocation_stable,
        "memory_allocated_before": allocated_before,
        "memory_allocated_after": allocated_after,
        "orders": orders,
        "provenance": {
            "candidate": _callable_provenance(
                "flashinfer-python", wan_hybrid_attention
            ),
            "production_fa4": _production_fa4_provenance(production_fa4),
        },
        "benchmark_process_runtime_seconds": time.monotonic() - started,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not passed:
        raise RuntimeError("exact Wan hybrid qualification failed")


if __name__ == "__main__":
    main()
