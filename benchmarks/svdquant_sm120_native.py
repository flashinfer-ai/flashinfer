"""Native NVFP4 baseline arms for the SM120 SVDQuant backend comparison.

Ported from the PR #5228 native campaign (georgeliu95/svdq-kernel-optimization
648ea251d, benchmarks/bench_nvfp4_svdquant_gemm.py): bias-fused CUTLASS and
cuBLASLt NVFP4 GEMMs. Linear quantizes the *unsmoothed* activation with the
public CUDA quantizer before the GEMM; smoothing and LoRA belong to SVDQ, so
native is a lower-work reference, not an equal-function one. GEMM times the
bias-fused GEMM alone. Every tensor-wide scale is derived outside timing.

Each arm is pinned to the tactic that is fastest under isolated cold-L2 graph
replay, measured here with the same timer the comparison uses.
"""

import functools
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch

from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.gemm.gemm_svdquant import DEFAULT_WORKSPACE_SIZE
from flashinfer.jit.gemm.nvfp4_cublaslt_bias import gen_nvfp4_cublaslt_bias_module
from flashinfer.jit.gemm.nvfp4_cutlass_bias_sm120 import (
    gen_nvfp4_cutlass_bias_sm120_module,
)
from flashinfer.testing.utils import bench_gpu_time_with_cuda_event

NATIVE_BACKENDS = ("native-cutlass", "native-cublaslt")
CUBLASLT_ALGORITHM_BYTES = 64
CUBLASLT_MAX_ALGORITHMS = 100


@functools.cache
def _cutlass_module():
    return gen_nvfp4_cutlass_bias_sm120_module().build_and_load()


@functools.cache
def _cublaslt_module():
    return gen_nvfp4_cublaslt_bias_module().build_and_load()


@dataclass(frozen=True, slots=True)
class NativeOperands:
    x: torch.Tensor
    global_scale: torch.Tensor
    alpha: torch.Tensor
    beta_zero: torch.Tensor
    xq: torch.Tensor
    x_sf: torch.Tensor
    wq: torch.Tensor
    w_sf: torch.Tensor
    bias: torch.Tensor


def _quantize(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed, sf = nvfp4_quantize(
        x, scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
        sf_vec_size=16, backend="cuda",
    )
    return packed.view(torch.uint8).contiguous(), sf.view(torch.uint8).contiguous()


def native_operands(case) -> NativeOperands:
    """Native scales come from the raw activation; the SVDQ weight is reused."""
    peak = case.x.float().abs().nan_to_num().max()
    global_scale = ((448.0 * 6.0) / peak).reshape(1).contiguous()
    # SVDQ alpha = 1 / (smoothed_global * weight_global); recover weight_global.
    weight_global = (case.alpha.float() * case.global_sf.float()).reciprocal()
    alpha = (1.0 / (global_scale * weight_global)).reshape(1).float()
    xq, x_sf = _quantize(case.x, global_scale)
    bias = case.bias if case.bias is not None else torch.zeros(
        case.wq.shape[0], dtype=torch.bfloat16, device=case.x.device
    )
    return NativeOperands(
        case.x, global_scale, alpha,
        torch.zeros(1, dtype=torch.float32, device=case.x.device),
        xq, x_sf, case.wq, case.w_sf.view(torch.uint8), bias,
    )


def _launchers(backend: str, ops: NativeOperands, out: torch.Tensor, quantize: bool):
    m, k = ops.x.shape
    n = ops.wq.shape[0]
    device = ops.x.device

    def prefix() -> tuple[torch.Tensor, torch.Tensor]:
        return _quantize(ops.x, ops.global_scale) if quantize else (ops.xq, ops.x_sf)

    if backend == "native-cutlass":
        module = _cutlass_module()
        tactics = list(range(int(module.nvfp4_cutlass_gemm_bias_tactic_num())))
        ws = torch.empty(
            max(DEFAULT_WORKSPACE_SIZE, *(int(module.nvfp4_cutlass_gemm_bias_workspace_size(m, n, k, t)) for t in tactics)),
            dtype=torch.uint8, device=device,
        )

        def factory(t: int) -> Callable[[], torch.Tensor]:
            def launch() -> torch.Tensor:
                xq, x_sf = prefix()
                module.nvfp4_cutlass_gemm_bias(xq, ops.wq, x_sf, ops.w_sf, ops.alpha, ops.bias, out, ws, t)
                return out
            return launch

        return tactics, factory
    module = _cublaslt_module()
    ws = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    algos = torch.empty(CUBLASLT_MAX_ALGORITHMS * CUBLASLT_ALGORITHM_BYTES, dtype=torch.uint8, device="cpu")
    count = int(module.nvfp4_cublaslt_bias_get_algorithms(ops.xq, ops.wq, ops.x_sf, ops.w_sf, ops.bias, out, ws, algos))
    if count <= 0:
        raise RuntimeError("cuBLASLt returned no algorithm for the bias-enabled descriptor")

    def factory(t: int) -> Callable[[], torch.Tensor]:
        def launch() -> torch.Tensor:
            xq, x_sf = prefix()
            module.nvfp4_cublaslt_bias_run_with_algorithm(
                xq, ops.wq, x_sf, ops.w_sf, ops.alpha, ops.beta_zero, ops.bias, out, ws, algos, count, t)
            return out
        return launch

    return list(range(count)), factory


def _cold_graph_us(launch: Callable[[], torch.Tensor]) -> float:
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    samples = bench_gpu_time_with_cuda_event(
        graph.replay, dry_run_time_ms=20, repeat_time_ms=60, cold_l2_cache=True
    )
    return statistics.median(samples) * 1000.0


def native_call(backend: str, ops: NativeOperands, quantize: bool) -> tuple[Callable[[], torch.Tensor], int]:
    """Return a launch pinned to the fastest runnable tactic, and that tactic."""
    out = torch.empty(ops.x.shape[0], ops.wq.shape[0], dtype=torch.bfloat16, device=ops.x.device)
    tactics, factory = _launchers(backend, ops, out, quantize)
    best: tuple[float, int] | None = None
    for t in tactics:
        try:
            us = _cold_graph_us(factory(t))
        except (RuntimeError, ValueError):
            continue
        if best is None or us < best[0]:
            best = (us, t)
    if best is None:
        raise RuntimeError(f"no runnable {backend} tactic")
    return factory(best[1]), best[1]
