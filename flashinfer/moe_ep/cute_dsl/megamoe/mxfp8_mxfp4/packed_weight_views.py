# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pointer-backed CuTe views over packed MXFP4 MegaMoE weights.

Weight preprocessing returns Torch tensors with physical shape
``(experts, K / 2, M)``.  Each Torch element is one byte containing two E2M1
values, and the K dimension is physically contiguous.  The swap-AB MMA sees
the same storage as a logical nibble tensor ``(M, K, experts)`` with stride
``(K, 1, M * K)``.

``cutlass.torch.from_dlpack`` cannot express that distinction reliably for
``torch.float4_e2m1fn_x2``: it expands Torch's trailing dimension, although
the packed dimension here is the middle (K) dimension.  This adapter therefore
passes typed raw pointers to CuTe and constructs the logical layout explicitly
while tracing the JIT entry point.

CUTLASS/CuTe imports are intentionally lazy so importing the backend remains
safe on CPU-only hosts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_TMA_MIN_ALIGNMENT = 32


def logical_nibble_offset(
    m: int,
    k: int,
    expert: int,
    *,
    rows: int,
    k_elements: int,
    num_experts: int,
) -> int:
    """Return the logical E2M1-element offset for ``(m, k, expert)``."""

    if rows <= 0 or k_elements <= 0 or num_experts <= 0:
        raise ValueError("packed weight dimensions must be positive")
    if k_elements % 2:
        raise ValueError("packed MXFP4 K must be even")
    if not 0 <= m < rows:
        raise IndexError(f"weight row {m} is outside [0, {rows})")
    if not 0 <= k < k_elements:
        raise IndexError(f"weight K coordinate {k} is outside [0, {k_elements})")
    if not 0 <= expert < num_experts:
        raise IndexError(f"weight expert {expert} is outside [0, {num_experts})")
    return expert * rows * k_elements + m * k_elements + k


@dataclass(frozen=True)
class CutePackedWeightArguments:
    """Raw Float4 pointers passed as runtime arguments to a CuTe JIT entry."""

    fc1_weight: Any
    fc2_weight: Any


@dataclass(frozen=True)
class CutePackedWeightViews:
    """Logical swap-AB A operands consumed by the persistent mainloop."""

    fc1_weight_a_logical: Any
    fc2_weight_a_logical: Any


class PackedWeightViews:
    """Validate transformed weights and expose zero-copy logical CuTe views.

    FC1's M coordinate is already gate8/up8 interleaved by preprocessing.
    This adapter deliberately preserves that row order.  FC2 uses the same
    K-major physical contract without the gate/up row transform.
    """

    def __init__(
        self,
        fc1_weight: Any,
        fc2_weight: Any,
        *,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError(
                "MXFP8 x MXFP4 hidden and intermediate dimensions must be "
                "multiples of 128"
            )

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.fc1_weight = fc1_weight
        self.fc2_weight = fc2_weight

        self.num_experts = self._validate_weight(
            fc1_weight,
            label="fc1_weight",
            rows=2 * self.intermediate_size,
            k_elements=self.hidden_size,
        )
        fc2_experts = self._validate_weight(
            fc2_weight,
            label="fc2_weight",
            rows=self.hidden_size,
            k_elements=self.intermediate_size,
        )
        if fc2_experts != self.num_experts:
            raise ValueError("FC1 and FC2 weights must have equal expert counts")
        if fc1_weight.device != fc2_weight.device:
            raise ValueError("FC1 and FC2 weights must be on the same device")

    @staticmethod
    def _validate_weight(
        tensor: Any,
        *,
        label: str,
        rows: int,
        k_elements: int,
    ) -> int:
        import torch

        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if tensor.dtype != torch.uint8 and (
            fp4_dtype is None or tensor.dtype != fp4_dtype
        ):
            raise TypeError(f"{label} must contain packed MXFP4 E2M1 values")

        if tensor.ndim != 3:
            raise ValueError(
                f"{label} must have packed-K-major shape "
                f"(experts, {k_elements // 2}, {rows}); got "
                f"{tuple(tensor.shape)}"
            )
        expected_shape = (int(tensor.shape[0]), k_elements // 2, rows)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{label} must have packed-K-major shape "
                f"(experts, {k_elements // 2}, {rows}); got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.shape[0] <= 0:
            raise ValueError(f"{label} must contain at least one expert")

        # This is the exact transpose view returned by preprocess_mega_weights:
        # contiguous (E,M,K/2) storage viewed as (E,K/2,M).  Requiring all
        # three strides prevents silently accepting expert padding or a copied
        # row-major tensor under the same public shape.
        expected_stride = (rows * (k_elements // 2), 1, k_elements // 2)
        if tuple(tensor.stride()) != expected_stride:
            raise ValueError(
                f"{label} must have packed-K-major strides {expected_stride}; "
                f"got {tuple(tensor.stride())}"
            )
        if tensor.storage_offset() != 0:
            raise ValueError(f"{label} must start at storage offset zero")
        if tensor.data_ptr() % _TMA_MIN_ALIGNMENT:
            raise ValueError(
                f"{label} must be {_TMA_MIN_ALIGNMENT}-byte aligned for TMA"
            )
        return int(tensor.shape[0])

    @property
    def fc1_logical_shape(self) -> tuple[int, int, int]:
        """FC1 swap-AB A domain: ``(gate_up_M, hidden_K, experts_L)``."""

        return (2 * self.intermediate_size, self.hidden_size, self.num_experts)

    @property
    def fc2_logical_shape(self) -> tuple[int, int, int]:
        """FC2 swap-AB A domain: ``(hidden_M, intermediate_K, experts_L)``."""

        return (self.hidden_size, self.intermediate_size, self.num_experts)

    @staticmethod
    def _logical_stride(shape: tuple[int, int, int]) -> tuple[int, int, int]:
        rows, k_elements, _ = shape
        return (k_elements, 1, rows * k_elements)

    def to_cute(self) -> CutePackedWeightArguments:
        """Create typed Float4 runtime pointers without using DLPack shapes."""

        if self.fc1_weight.device.type != "cuda":
            raise ValueError("CuTe packed weight views require CUDA tensors")
        try:
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import make_ptr
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("CUTLASS CuTe DSL is required for weight views") from exc

        def runtime_argument(tensor: Any) -> Any:
            return make_ptr(
                cutlass.Float4E2M1FN,
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=_TMA_MIN_ALIGNMENT,
            )

        return CutePackedWeightArguments(
            fc1_weight=runtime_argument(self.fc1_weight),
            fc2_weight=runtime_argument(self.fc2_weight),
        )

    def make_logical_views(
        self, fc1_weight: Any, fc2_weight: Any
    ) -> CutePackedWeightViews:
        """Build nibble-addressed ``(M,K,L)`` tensors while tracing CuTe JIT."""

        try:
            import cutlass.cute as cute
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("CUTLASS CuTe DSL is required for weight views") from exc

        def make_view(pointer: Any, shape: tuple[int, int, int]) -> Any:
            return cute.make_tensor(
                pointer,
                cute.make_layout(shape, stride=self._logical_stride(shape)),
            )

        return CutePackedWeightViews(
            fc1_weight_a_logical=make_view(fc1_weight, self.fc1_logical_shape),
            fc2_weight_a_logical=make_view(fc2_weight, self.fc2_logical_shape),
        )


__all__ = [
    "CutePackedWeightArguments",
    "CutePackedWeightViews",
    "PackedWeightViews",
    "logical_nibble_offset",
]
