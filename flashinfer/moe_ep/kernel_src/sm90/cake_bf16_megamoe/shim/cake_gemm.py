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

import importlib
import os
from typing import Any, Protocol

import torch

__all__ = [
    "DEV_LAUNCHER_ENV",
    "MAX_OUTPUT_ELEMENTS",
    "ExportedGroupedGemm",
    "GroupedGemm",
    "create_grouped_gemm",
    "output_row_capacity",
    "validate_grouped_gemm_geometry",
]

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
# The kernels form the output address as a 32-bit ``row * N + col``; the row
# capacity times the output width must stay inside that range.
MAX_OUTPUT_ELEMENTS = 1 << 32


def output_row_capacity(m_cap: int) -> int:
    """Activation rows allocated for ``m_cap`` routed tokens (BLOCK_M-padded)."""
    return (int(m_cap) + BLOCK_M - 1) // BLOCK_M * BLOCK_M


class GroupedGemm(Protocol):
    """Expert-grouped BF16 GEMMs over expert-contiguous activation rows.

    ``offsets`` is the ``int64 [E + 1]`` expert-major row prefix written by the
    protocol's prefix kernel; rows beyond ``offsets[E]`` are never read as
    valid output (the kernels guard stores by the per-expert row bound).
    """

    def fc1(
        self,
        a: torch.Tensor,
        w13: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """``out[m, I] = silu(a @ gate.T) * (a @ up.T)`` with gate/up-interleaved ``w13``."""

    def fc2(
        self,
        h: torch.Tensor,
        w2: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """``out[m, H] = h @ w2[e].T``."""

    def destroy(self) -> None: ...


def validate_grouped_gemm_geometry(
    *,
    hidden_size: int,
    intermediate_size: int,
    gate_up_group: int,
    row_capacity: int | None = None,
) -> None:
    """Shape constraints of the Cake GEMM kernels (FC1 N = 2I, K = H; FC2 N = H, K = I).

    ``row_capacity`` is the BLOCK_M-padded activation row count of the runner
    (``output_row_capacity(m_cap)``); the kernels index their output with a
    32-bit ``row * N + col``, so ``row_capacity * max(H, I)`` must fit.
    """
    from .....core.validation.common import MoEEpConfigError

    if hidden_size % BLOCK_K != 0 or intermediate_size % BLOCK_K != 0:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake requires hidden_size and intermediate_size to be "
            f"multiples of {BLOCK_K}, got hidden={hidden_size}, intermediate={intermediate_size}"
        )
    if (2 * intermediate_size) % BLOCK_N != 0 or hidden_size % BLOCK_N != 0:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake requires 2 * intermediate_size and hidden_size to be "
            f"multiples of {BLOCK_N}, got hidden={hidden_size}, intermediate={intermediate_size}"
        )
    if intermediate_size % gate_up_group != 0:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake requires intermediate_size to be a multiple of the "
            f"gate/up interleave group {gate_up_group}, got {intermediate_size}"
        )
    if row_capacity is not None:
        widest = max(hidden_size, intermediate_size)
        if row_capacity * widest > MAX_OUTPUT_ELEMENTS:
            raise MoEEpConfigError(
                "sm90_bf16_bf16_bf16_push_cake indexes the expert GEMM outputs with 32-bit "
                f"row * N + col; row capacity {row_capacity} x {widest} columns exceeds "
                f"{MAX_OUTPUT_ELEMENTS} elements. Lower max_tokens_per_rank, capacity_factor "
                "or the EP size."
            )


def _check_geometry(
    *,
    gated: bool,
    a: torch.Tensor,
    w: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
) -> tuple[int, int, int]:
    num_experts, shape_n, shape_k = w.shape
    n_out = shape_n // 2 if gated else shape_n
    if (
        a.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
    ):
        raise ValueError("grouped GEMM operands must be bf16")
    if a.shape[1] != shape_k or out.shape[1] != n_out or out.shape[0] != a.shape[0]:
        raise ValueError(
            f"grouped GEMM geometry mismatch: a={tuple(a.shape)} w={tuple(w.shape)} out={tuple(out.shape)}"
        )
    if a.shape[0] % BLOCK_M != 0:
        raise ValueError(
            f"activation row capacity must be a multiple of {BLOCK_M}, got {a.shape[0]}"
        )
    if shape_k % BLOCK_K != 0 or shape_n % BLOCK_N != 0:
        raise ValueError(
            f"grouped GEMM requires K % {BLOCK_K} == 0 and N % {BLOCK_N} == 0 "
            f"(the kernels tile N by {BLOCK_N} and K by {BLOCK_K} exactly), got N={shape_n}, K={shape_k}"
        )
    if a.shape[0] * n_out > MAX_OUTPUT_ELEMENTS:
        raise ValueError(
            f"grouped GEMM output {a.shape[0]} x {n_out} exceeds the kernels' 32-bit "
            f"row * N + col index range ({MAX_OUTPUT_ELEMENTS} elements)"
        )
    if offsets.dtype != torch.int64 or offsets.numel() != num_experts + 1:
        raise ValueError("offsets must be int64 [E + 1]")
    if not (
        a.is_contiguous()
        and w.is_contiguous()
        and out.is_contiguous()
        and offsets.is_contiguous()
    ):
        raise ValueError("grouped GEMM operands must be contiguous")
    return int(num_experts), int(shape_n), int(shape_k)


class ExportedGroupedGemm:
    """Shipped launcher: the generated, manifest-sealed FlashInfer JIT modules.

    ``run`` of each module encodes the two tensor maps on the host, opts the
    kernel into its dynamic shared memory once per device and launches on the
    caller's current stream; it owns no device state and never synchronises,
    so a captured round replays under CUDA graphs.
    """

    def __init__(self, *, clamp: float | None = None) -> None:
        from .cake_jit import gen_sm90_cake_bf16_grouped_gemm_module

        self._clamp = float(clamp) if clamp is not None else 0.0
        fc1_stage = "fc1_gated_clamp" if clamp is not None else "fc1_gated"
        self._fc1 = gen_sm90_cake_bf16_grouped_gemm_module(fc1_stage).build_and_load()
        self._fc2 = gen_sm90_cake_bf16_grouped_gemm_module("fc2").build_and_load()
        self._sm_count: dict[int, int] = {}
        self._destroyed = False

    def _grid_x(self, device: torch.device) -> int:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        sms = self._sm_count.get(index)
        if sms is None:
            sms = int(torch.cuda.get_device_properties(index).multi_processor_count)
            self._sm_count[index] = sms
        return sms

    def _launch(
        self,
        module: Any,
        *,
        gated: bool,
        a: torch.Tensor,
        w: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if self._destroyed:
            raise RuntimeError("ExportedGroupedGemm has been destroyed")
        num_experts, shape_n, shape_k = _check_geometry(
            gated=gated, a=a, w=w, offsets=offsets, out=out
        )
        module.run(
            num_experts,
            shape_n,
            shape_k,
            self._clamp,
            a,
            w,
            offsets,
            out,
            self._grid_x(a.device),
            1,
            1,
        )

    def fc1(
        self,
        a: torch.Tensor,
        w13: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._launch(self._fc1, gated=True, a=a, w=w13, offsets=offsets, out=out)

    def fc2(
        self,
        h: torch.Tensor,
        w2: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._launch(self._fc2, gated=False, a=h, w=w2, offsets=offsets, out=out)

    def destroy(self) -> None:
        self._destroyed = True


DEV_LAUNCHER_ENV = "FLASHINFER_SM90_CAKE_BF16_DEV_LAUNCHER"


def _dev_launcher_factory() -> Any | None:
    """Resolve the development launcher factory named by ``DEV_LAUNCHER_ENV``.

    The variable holds ``module`` or ``module:attribute`` (default attribute
    ``create_grouped_gemm``): a callable ``(*, clamp) -> GroupedGemm`` that
    builds the kernels from the generator in-process.  Unset, empty or ``0``
    selects the shipped launcher.
    """
    spec = os.environ.get(DEV_LAUNCHER_ENV, "").strip()
    if spec in ("", "0"):
        return None
    module_name, _, attribute = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute or "create_grouped_gemm")


def create_grouped_gemm(*, clamp: float | None = None) -> GroupedGemm:
    """Return the grouped GEMM launcher for this process.

    The shipped path is :class:`ExportedGroupedGemm` (generated sources, no
    generator at runtime).  ``FLASHINFER_SM90_CAKE_BF16_DEV_LAUNCHER`` may name
    a development launcher factory (see :func:`_dev_launcher_factory`) that
    compiles the kernels from the generator package in-process; it is the
    reference the exported modules are checked against and is never shipped.
    """
    factory = _dev_launcher_factory()
    if factory is not None:
        return factory(clamp=clamp)
    return ExportedGroupedGemm(clamp=clamp)
