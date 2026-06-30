"""NVTX ranges for MegaMoE EP nsys profiling (trace benchmarks)."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from bench_deepseek_v4_mega_moe_common import BenchTiming

# Matched by bench_deepseek_v4_mega_moe_nsys.sh (--capture-range=cudaProfilerApi).
NVTX_DOMAIN = "default"
NVTX_STEADY_CAPTURE = "steady_capture"
NVTX_COLD_START = "cold_start"
NVTX_WARMUP = "warmup"
NVTX_STEADY_ITER = "steady_iter"
NVTX_FORWARD = "forward"
NVTX_SETUP = "setup"

# Kept for manual nsys experiments; PyTorch range_push uses the NULL domain.
NVTX_CAPTURE_SPEC = f"@{NVTX_STEADY_CAPTURE}"


def _cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def steady_capture_profile():
    """NVTX label + cudaProfilerStart/Stop for nsys capture-range."""
    torch.cuda.synchronize()
    with nvtx_range(NVTX_STEADY_CAPTURE):
        _cuda_profiler_start()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            _cuda_profiler_stop()


def bench_forward_ms_nvtx(
    run_once,
    *,
    warmup: int,
    repeat: int,
    cold_start: bool,
) -> BenchTiming:
    def timed_forward() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with nvtx_range(NVTX_FORWARD):
            start.record()
            run_once()
            end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    cold_start_ms = None
    if cold_start:
        with nvtx_range(NVTX_COLD_START):
            cold_start_ms = timed_forward()

    with nvtx_range(NVTX_WARMUP):
        for _ in range(warmup):
            timed_forward()

    with steady_capture_profile():
        elapsed_ms = [timed_forward() for _ in range(repeat)]

    return BenchTiming(
        steady_avg_ms=sum(elapsed_ms) / len(elapsed_ms),
        steady_first_ms=elapsed_ms[0],
        steady_min_ms=min(elapsed_ms),
        steady_max_ms=max(elapsed_ms),
        cold_start_ms=cold_start_ms,
    )
