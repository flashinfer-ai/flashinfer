# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Persistent BF16 GEMM for ``mm_bf16(backend="cute-dsl")`` above the low-M range.

Runs the SM100 persistent dense GEMM from ``bmm_fp8_blackwell`` (the CUTLASS
``dense_gemm_persistent`` example, also ported by TensorRT-LLM) with BF16
operands and a unit output scale. Each tactic compiles once with symbolic
M/N/K, so one kernel serves every shape.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import cutlass
import torch

from .bmm_fp8_blackwell import PersistentDenseGemmKernel
from .bmm_fp8_wrapper import _compile_and_create_tensor_api, _create_fake_tensors


class PersistentTactic(NamedTuple):
    use_2cta_instrs: bool
    tile_m: int
    tile_n: int
    cluster_m: int
    cluster_n: int


# Trimmed from TensorRT-LLM's BF16 runner space; index 0 is the untuned default.
_TACTICS = (
    PersistentTactic(False, 128, 128, 1, 1),
    PersistentTactic(False, 64, 128, 1, 1),
    PersistentTactic(False, 128, 256, 1, 1),
    PersistentTactic(False, 128, 128, 1, 2),
    PersistentTactic(False, 128, 128, 2, 1),
    PersistentTactic(True, 256, 128, 2, 1),
    PersistentTactic(True, 256, 256, 2, 1),
    PersistentTactic(True, 256, 128, 2, 2),
    PersistentTactic(True, 256, 128, 4, 1),
    PersistentTactic(True, 256, 256, 4, 1),
)


def _make_kernel(tactic: PersistentTactic, use_pdl: bool = False):
    return PersistentDenseGemmKernel(
        cutlass.Float32,
        tactic.use_2cta_instrs,
        (tactic.tile_m, tactic.tile_n),
        (tactic.cluster_m, tactic.cluster_n),
        use_tma_store=True,
        use_pdl=use_pdl,
    )


# Only N/K alignment matters (TMA store clips M/N tails), so M stays out of the key.
@functools.cache
def _can_implement(tactic: PersistentTactic, n: int, k: int) -> bool:
    return _make_kernel(tactic).can_implement(
        (1, n, k, 1),
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.BFloat16,
        "k",
        "k",
        "n",
    )


def autotune_tactics(m: int, n: int, k: int) -> list[PersistentTactic]:
    """Return the tactics that can serve ``(m, n, k)``, default first."""
    return [t for t in _TACTICS if _can_implement(t, n, k)]


def default_tactic(m: int, n: int, k: int) -> PersistentTactic:
    tactics = autotune_tactics(m, n, k)
    if not tactics:
        raise ValueError(
            f"persistent BF16 GEMM cannot serve M={m}, N={n}, K={k} "
            "(N and K must be multiples of 8)"
        )
    return tactics[0]


@functools.cache
def _unit_scale(device_index: int) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=f"cuda:{device_index}")


@functools.cache
def _get_compiled_kernel(device_index: int, tactic: PersistentTactic, use_pdl: bool):
    with torch.cuda.device(device_index):
        return _compile_and_create_tensor_api(
            _make_kernel(tactic, use_pdl),
            *_create_fake_tensors(cutlass.BFloat16, cutlass.BFloat16, "k", "k", "n"),
            (tactic.cluster_m, tactic.cluster_n),
        )


def run_persistent_dense(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    pdl: bool,
    tactic: PersistentTactic,
) -> torch.Tensor:
    """Run ``out = a @ b`` with the ``mm_bf16`` layouts (row-major A/out, column-major B)."""
    m, k = a.shape
    n = b.shape[1]
    if not (a.is_contiguous() and b.T.is_contiguous() and out.is_contiguous()):
        raise ValueError("persistent GEMM requires row-major A/out and column-major B")
    if any(t.data_ptr() % 16 for t in (a, b, out)):
        raise ValueError("persistent GEMM requires 16-byte aligned A, B and out")
    if not _can_implement(tactic, n, k):
        raise ValueError(f"tactic {tactic} cannot serve M={m}, N={n}, K={k}")
    device_index = a.get_device()
    with torch.cuda.device(device_index):
        _get_compiled_kernel(device_index, tactic, bool(pdl))(
            a.unsqueeze(0),
            # (1, K, N) with compact batch stride, matching the fake tensor.
            b.T.unsqueeze(0).transpose(1, 2),
            out.unsqueeze(0),
            _unit_scale(device_index),
        )
    return out


__all__ = [
    "PersistentTactic",
    "autotune_tactics",
    "default_tactic",
    "run_persistent_dense",
]
