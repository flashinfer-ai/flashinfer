# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Persistent BF16 GEMM for ``mm_bf16(backend="cute-dsl")`` above the low-M range.

Runs the SM100 persistent dense GEMM from ``bmm_fp8_blackwell`` (the CUTLASS
``dense_gemm_persistent`` example, also ported by TensorRT-LLM) with BF16
operands and a unit output scale. Each tactic compiles once with symbolic
M/N/K, so one kernel serves every shape.
"""

import functools

import cutlass
import torch

from .bmm_fp8_blackwell import PersistentDenseGemmKernel
from .bmm_fp8_wrapper import _compile_and_create_tensor_api, _create_fake_tensors

# (use_2cta_instrs, tile_m, tile_n, cluster_m, cluster_n); index 0 is the
# untuned default.
TACTICS = (
    (False, 128, 128, 1, 1),
    (False, 64, 128, 1, 1),
    (False, 128, 256, 1, 1),
    (False, 128, 128, 1, 2),
    (False, 128, 128, 2, 1),
    (True, 256, 128, 2, 1),
    (True, 256, 256, 2, 1),
    (True, 256, 128, 2, 2),
    (True, 256, 128, 4, 1),
    (True, 256, 256, 4, 1),
)


def supports(a, b, bias, out) -> bool:
    """Every tactic applies: the TMA store clips M/N tails, leaving 16-byte alignment."""
    return (
        bias is None
        and a.shape[1] % 8 == 0
        and b.shape[1] % 8 == 0
        and all(t.data_ptr() % 16 == 0 for t in (a, b, out) if t is not None)
    )


@functools.cache
def _unit_scale(device_index: int) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=f"cuda:{device_index}")


@functools.cache
def _get_compiled_kernel(device_index: int, tactic: tuple, use_pdl: bool):
    use_2cta, tile_m, tile_n, cluster_m, cluster_n = tactic
    kernel = PersistentDenseGemmKernel(
        cutlass.Float32,
        use_2cta,
        (tile_m, tile_n),
        (cluster_m, cluster_n),
        use_tma_store=True,
        use_pdl=use_pdl,
    )
    with torch.cuda.device(device_index):
        return _compile_and_create_tensor_api(
            kernel,
            *_create_fake_tensors(cutlass.BFloat16, cutlass.BFloat16, "k", "k", "n"),
            (cluster_m, cluster_n),
        )


def run_persistent_dense(a, b, out, pdl: bool, tactic: tuple = TACTICS[0]):
    """Run ``out = a @ b`` with row-major A/out and column-major B."""
    device_index = a.get_device()
    with torch.cuda.device(device_index):
        _get_compiled_kernel(device_index, tuple(tactic), bool(pdl))(
            a.unsqueeze(0),
            # (1, K, N) with compact batch stride, matching the fake tensor.
            b.T.unsqueeze(0).transpose(1, 2),
            out.unsqueeze(0),
            _unit_scale(device_index),
        )
    return out
