"""Strict cold-L2 CUPTI benchmark for grouped MXFP8 public backends.

The timed boundary is one complete eager ``mxfp8_grouped_quantize`` call,
including output allocation and every GPU activity attributed to that call.
CUDA-event and CUDA-graph timing fallbacks are deliberately rejected.
"""

import argparse
import json
import math
import statistics
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import torch

import flashinfer
import flashinfer.testing.utils as timing_utils
from flashinfer.cutile import is_cuda_tile_available
from flashinfer.jit.cake_grouped_mxfp8_quantize import (
    is_cake_grouped_mxfp8_quantize_available,
)


DEFAULT_SHAPES = ((2, 256, 4096),)


def _parse_shape(text: str) -> tuple[int, int, int]:
    try:
        batch, rows, columns = (int(value) for value in text.lower().split("x"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must use BxMxK syntax") from error
    if batch <= 0 or rows <= 0 or columns <= 0 or columns % 32:
        raise argparse.ArgumentTypeError("B/M/K must be positive and K divisible by 32")
    return batch, rows, columns


def _require_environment() -> tuple[tuple[int, int], str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            "this benchmark requires exact compute capability 10.0 or 10.3, "
            f"got {capability[0]}.{capability[1]}"
        )
    try:
        from cupti import cupti  # noqa: F401

        cupti_version = version("cupti-python")
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("reportable timings require cupti-python >= 13") from error
    if int(cupti_version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"reportable timings require cupti-python >= 13, got {cupti_version}")
    if not is_cake_grouped_mxfp8_quantize_available(
        torch.bfloat16, torch.device("cuda", torch.cuda.current_device())
    ):
        raise RuntimeError("an exported Cake BF16 grouped MXFP8 profile is not installed")
    if not is_cuda_tile_available():
        raise RuntimeError("the cuTile grouped MXFP8 baseline is not available")
    return capability, cupti_version


def _measure_strict_cupti(call, *, warmup_ms: int, benchmark_ms: int) -> dict:
    def reject_fallback(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("CUPTI fallback is forbidden for reportable timings")

    original_event = timing_utils.bench_gpu_time_with_cuda_event
    original_graph = timing_utils.bench_gpu_time_with_cudagraph
    timing_utils.bench_gpu_time_with_cuda_event = reject_fallback
    timing_utils.bench_gpu_time_with_cudagraph = reject_fallback
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            measured = timing_utils.bench_gpu_time(
                call,
                enable_cupti=True,
                cold_l2_cache=True,
                use_cuda_graph=False,
                dry_run_iters=None,
                repeat_iters=None,
                dry_run_time_ms=warmup_ms,
                repeat_time_ms=benchmark_ms,
            )
    finally:
        timing_utils.bench_gpu_time_with_cudagraph = original_graph
        timing_utils.bench_gpu_time_with_cuda_event = original_event

    fallback = [item for item in caught if "Falling back" in str(item.message)]
    if fallback:
        raise RuntimeError(str(fallback[0].message))
    samples = [float(value) for value in measured]
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise RuntimeError(f"invalid CUPTI samples: {samples}")
    return {
        "timing_backend": "CUPTI activity",
        "cold_l2": True,
        "cuda_graph": False,
        "activity_scope": "first_to_last_correlated_GPU_activity_in_one_public_call",
        "samples_ms": samples,
        "sample_count": len(samples),
        "median_ms": float(statistics.median(samples)),
    }


def _run_shape(shape: tuple[int, int, int], warmup_ms: int, benchmark_ms: int) -> dict:
    batch, rows, columns = shape
    generator = torch.Generator(device="cuda").manual_seed(20260829 + sum(shape))
    x = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
    mask = torch.full((batch,), rows, dtype=torch.int32, device="cuda")

    cutile_q, cutile_sf = flashinfer.mxfp8_grouped_quantize(x, mask, backend="cutile")
    cake_q, cake_sf = flashinfer.mxfp8_grouped_quantize(x, mask, backend="cake")
    torch.cuda.synchronize()
    torch.testing.assert_close(
        cake_q.contiguous().view(torch.uint8),
        cutile_q.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )
    padded_k = (columns + 127) // 128 * 128

    def valid_scales(sf):
        physical = sf.permute(5, 2, 4, 0, 1, 3)
        m_tiles = (rows + 127) // 128
        k_tiles = padded_k // 128
        unswizzled = physical.reshape(batch, m_tiles, k_tiles, 32, 4, 4)
        unswizzled = unswizzled.transpose(2, 4)
        return unswizzled.reshape(batch, m_tiles * 128, k_tiles * 4)[:, :rows]

    torch.testing.assert_close(
        valid_scales(cake_sf), valid_scales(cutile_sf), rtol=0, atol=0
    )

    retained = {}

    def make_call(backend):
        def call():
            retained[backend] = flashinfer.mxfp8_grouped_quantize(
                x, mask, backend=backend
            )

        return call

    arms = {}
    for backend in ("cutile", "cake"):
        arms[backend] = _measure_strict_cupti(
            make_call(backend), warmup_ms=warmup_ms, benchmark_ms=benchmark_ms
        )
    return {
        "shape": {"B": batch, "M": rows, "K": columns},
        "dtype": "bfloat16",
        "mask": "full",
        "correctness": "exact_bits_and_scales",
        "timed_boundary": "one_complete_eager_public_API_call",
        "arms": arms,
        "cake_speedup": arms["cutile"]["median_ms"] / arms["cake"]["median_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--benchmark-ms", type=int, default=1000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.warmup_ms < 100 or args.benchmark_ms < 1000:
        raise ValueError("reportable runs require >=100 ms warmup and >=1000 ms measurement")

    wall_start = time.perf_counter()
    capability, cupti_version = _require_environment()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    shapes = tuple(args.shape) if args.shape else DEFAULT_SHAPES
    rows = [_run_shape(shape, args.warmup_ms, args.benchmark_ms) for shape in shapes]
    report = {
        "schema": "flashinfer-grouped-mxfp8-backend-comparison-v1",
        "gpu": properties.name,
        "compute_capability": list(capability),
        "clock_rate_khz": properties.clock_rate,
        "cupti_python_version": cupti_version,
        "timing_backend": "CUPTI activity",
        "cold_l2": True,
        "warmup_ms_per_arm": args.warmup_ms,
        "benchmark_ms_per_arm": args.benchmark_ms,
        "rows": rows,
        "physical_wall_time_s": time.perf_counter() - wall_start,
    }
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
