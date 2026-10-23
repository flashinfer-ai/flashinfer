# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical CuTe views over transformed MXFP4 weight scale buffers.

Weight preprocessing stores each expert's E8M0 scale plane in the standard
Blackwell 32x4x4 scale-factor atom order.  The public Torch shape is therefore
``(experts, bytes_per_expert)`` and deliberately does *not* describe the
logical matrix.  TMA, on the other hand, needs the atom-tiled CuTe layout for
the swap-AB GEMM domain ``(M, K, L=experts)``.

This module is the single adapter between those representations.  CUTLASS and
CuTe imports remain inside :meth:`SwizzledWeightScaleViews.to_cute`, keeping
the backend importable on hosts that do not have CuTe DSL installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


SF_VECTOR_SIZE = 32
SF_ATOM_ROWS = 128
SF_ATOM_K_BLOCKS = 4
SF_ATOM_BYTES = SF_ATOM_ROWS * SF_ATOM_K_BLOCKS
_TMA_MIN_ALIGNMENT = 16


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_logical_shape(rows: int, k_elements: int) -> None:
    if rows <= 0 or k_elements <= 0:
        raise ValueError("weight scale matrix dimensions must be positive")
    if k_elements % SF_VECTOR_SIZE:
        raise ValueError("weight K dimension must be divisible by 32")


def swizzled_scale_numel(rows: int, k_elements: int) -> int:
    """Return the padded byte extent of one atom-swizzled scale plane."""

    _validate_logical_shape(rows, k_elements)
    k_blocks = k_elements // SF_VECTOR_SIZE
    return (
        _ceil_div(rows, SF_ATOM_ROWS)
        * _ceil_div(k_blocks, SF_ATOM_K_BLOCKS)
        * SF_ATOM_BYTES
    )


def atom_swizzled_offset(
    row: int,
    k_block: int,
    *,
    rows: int,
    k_elements: int,
) -> int:
    """Map logical ``(row, K/32 block)`` to a byte in one expert plane.

    This is the scalar form of CUTLASS' 32x4x4 SF atom layout.  Keeping the
    mapping explicit gives preprocessing and the CuTe view a round-trip
    contract independent of tensor reshapes.
    """

    _validate_logical_shape(rows, k_elements)
    k_blocks = k_elements // SF_VECTOR_SIZE
    if not 0 <= row < rows:
        raise IndexError(f"scale row {row} is outside [0, {rows})")
    if not 0 <= k_block < k_blocks:
        raise IndexError(f"scale K block {k_block} is outside [0, {k_blocks})")

    k_atoms = _ceil_div(k_blocks, SF_ATOM_K_BLOCKS)
    atom = (row // SF_ATOM_ROWS) * k_atoms + k_block // SF_ATOM_K_BLOCKS
    row_in_atom = row % SF_ATOM_ROWS
    return (
        atom * SF_ATOM_BYTES
        + (row_in_atom % 32) * 16
        + (row_in_atom // 32) * 4
        + k_block % SF_ATOM_K_BLOCKS
    )


def to_atom_swizzled(logical: torch.Tensor) -> torch.Tensor:
    """Apply the preprocessing 32x4x4 atom layout to one logical scale plane.

    ``logical`` has shape ``(M, K/32)``.  Padding bytes are zero, matching the
    existing W4A8 preprocessing contract.
    """

    if logical.ndim != 2:
        raise ValueError("logical scale plane must be two-dimensional")
    rows, k_blocks = (int(value) for value in logical.shape)
    _validate_logical_shape(rows, k_blocks * SF_VECTOR_SIZE)
    padded_rows = _ceil_div(rows, SF_ATOM_ROWS) * SF_ATOM_ROWS
    padded_k_blocks = _ceil_div(k_blocks, SF_ATOM_K_BLOCKS) * SF_ATOM_K_BLOCKS

    padded = logical.new_zeros((padded_rows, padded_k_blocks))
    padded[:rows, :k_blocks].copy_(logical)
    row_atoms = padded_rows // SF_ATOM_ROWS
    k_atoms = padded_k_blocks // SF_ATOM_K_BLOCKS
    atoms = padded.view(row_atoms, SF_ATOM_ROWS, k_atoms, SF_ATOM_K_BLOCKS).permute(
        0, 2, 1, 3
    )
    return (
        atoms.reshape(-1, 4, 32, SF_ATOM_K_BLOCKS)
        .transpose(1, 2)
        .reshape(-1)
        .contiguous()
    )


def from_atom_swizzled(
    flat: torch.Tensor, *, rows: int, k_elements: int
) -> torch.Tensor:
    """Invert :func:`to_atom_swizzled` and discard atom padding."""

    _validate_logical_shape(rows, k_elements)
    if flat.ndim != 1:
        raise ValueError("atom-swizzled scale plane must be flat")
    expected = swizzled_scale_numel(rows, k_elements)
    if flat.numel() != expected:
        raise ValueError(
            f"atom-swizzled plane has {flat.numel()} bytes; expected {expected}"
        )

    k_blocks = k_elements // SF_VECTOR_SIZE
    padded_rows = _ceil_div(rows, SF_ATOM_ROWS) * SF_ATOM_ROWS
    padded_k_blocks = _ceil_div(k_blocks, SF_ATOM_K_BLOCKS) * SF_ATOM_K_BLOCKS
    row_atoms = padded_rows // SF_ATOM_ROWS
    k_atoms = padded_k_blocks // SF_ATOM_K_BLOCKS
    atoms = flat.reshape(-1, 32, 16).reshape(-1, 32, 4, 4)
    blocks = atoms.transpose(1, 2).reshape(
        row_atoms, k_atoms, SF_ATOM_ROWS, SF_ATOM_K_BLOCKS
    )
    padded = blocks.permute(0, 2, 1, 3).reshape(padded_rows, padded_k_blocks)
    return padded[:rows, :k_blocks].contiguous()


@dataclass(frozen=True)
class CuteWeightScaleViews:
    """Named products consumed by ``DeviceAssemblyProtocol``."""

    fc1_weight_sfa_logical: Any
    fc2_weight_sfa_logical: Any


@dataclass(frozen=True)
class CuteWeightScaleArguments:
    """Flat runtime arguments passed into a CuTe JIT entry point."""

    fc1_weight_scales: Any
    fc2_weight_scales: Any


class SwizzledWeightScaleViews:
    """Validate transformed scale planes and expose zero-copy CuTe SF views.

    FC1 rows already have the preprocessing gate8/up8 interleave.  The
    adapter preserves that M ordering and only supplies the correct atom
    layout; it never transposes or reorders weight scale bytes.
    """

    def __init__(
        self,
        fc1_weight_scales: torch.Tensor,
        fc2_weight_scales: torch.Tensor,
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
        self.fc1_weight_scales = fc1_weight_scales
        self.fc2_weight_scales = fc2_weight_scales

        self.num_experts = self._validate_buffer(
            fc1_weight_scales,
            label="fc1_weight_scales",
            rows=2 * self.intermediate_size,
            k_elements=self.hidden_size,
        )
        fc2_experts = self._validate_buffer(
            fc2_weight_scales,
            label="fc2_weight_scales",
            rows=self.hidden_size,
            k_elements=self.intermediate_size,
        )
        if fc2_experts != self.num_experts:
            raise ValueError("FC1 and FC2 scale buffers must have equal expert counts")
        if fc1_weight_scales.device != fc2_weight_scales.device:
            raise ValueError("FC1 and FC2 scale buffers must be on the same device")

    @staticmethod
    def _validate_buffer(
        tensor: torch.Tensor,
        *,
        label: str,
        rows: int,
        k_elements: int,
    ) -> int:
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if tensor.dtype is not torch.uint8 and (
            e8m0 is None or tensor.dtype is not e8m0
        ):
            raise TypeError(f"{label} must contain raw uint8/E8M0 scale bytes")
        if tensor.ndim != 2 or not tensor.is_contiguous():
            raise ValueError(
                f"{label} must be a contiguous (experts, flat_bytes) tensor"
            )
        if tensor.shape[0] <= 0:
            raise ValueError(f"{label} must contain at least one expert")
        expected = swizzled_scale_numel(rows, k_elements)
        if tensor.shape[1] != expected:
            raise ValueError(
                f"{label} has {tensor.shape[1]} bytes per expert; expected {expected}"
            )
        if tensor.data_ptr() % _TMA_MIN_ALIGNMENT:
            raise ValueError(
                f"{label} must be {_TMA_MIN_ALIGNMENT}-byte aligned for TMA"
            )
        return int(tensor.shape[0])

    @property
    def fc1_logical_shape(self) -> tuple[int, int, int]:
        """Swap-AB A/SFA domain: ``(gate_up_M, hidden_K, experts_L)``."""

        return (2 * self.intermediate_size, self.hidden_size, self.num_experts)

    @property
    def fc2_logical_shape(self) -> tuple[int, int, int]:
        """Swap-AB A/SFA domain: ``(hidden_M, intermediate_K, experts_L)``."""

        return (self.hidden_size, self.intermediate_size, self.num_experts)

    @staticmethod
    def _e8m0_view(tensor: torch.Tensor) -> torch.Tensor:
        dtype = getattr(torch, "float8_e8m0fnu", None)
        if dtype is None:
            raise RuntimeError("Torch does not provide float8_e8m0fnu")
        return tensor if tensor.dtype is dtype else tensor.view(dtype)

    def to_cute(self) -> CuteWeightScaleArguments:
        """Create zero-copy flat CuTe runtime arguments.

        CuTe layouts can only be constructed while tracing a ``cute.jit``
        entry point.  The integrating kernel passes these arguments to
        :meth:`make_logical_views` inside that entry point.
        """

        if self.fc1_weight_scales.device.type != "cuda":
            raise ValueError("CuTe weight scale views require CUDA tensors")
        try:
            from cutlass.cute.runtime import from_dlpack
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("CUTLASS CuTe DSL is required for scale views") from exc

        def runtime_argument(tensor: torch.Tensor):
            return from_dlpack(
                self._e8m0_view(tensor), assumed_align=_TMA_MIN_ALIGNMENT
            ).mark_layout_dynamic()

        return CuteWeightScaleArguments(
            fc1_weight_scales=runtime_argument(self.fc1_weight_scales),
            fc2_weight_scales=runtime_argument(self.fc2_weight_scales),
        )

    def make_logical_views(
        self, fc1_weight_scales: Any, fc2_weight_scales: Any
    ) -> CuteWeightScaleViews:
        """Apply atom layouts while tracing the integrating ``cute.jit``.

        The inputs are the traced forms of the flat arguments returned by
        :meth:`to_cute`.  Returned tensors are directly compatible with
        :meth:`Mxfp8Mxfp4TmaStaging.make_tma_atoms`.
        """

        try:
            import cutlass.cute as cute
            import cutlass.utils.blockscaled_layout as blockscaled_utils
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("CUTLASS CuTe DSL is required for scale views") from exc

        def make_view(tensor: Any, shape: tuple[int, int, int]):
            layout = blockscaled_utils.tile_atom_to_shape_SF(shape, SF_VECTOR_SIZE)
            # The iterator points at already-swizzled storage.  The atom
            # layout is essential: reshaping ``tensor`` would assign row-major
            # strides and make TMA read the wrong scale for almost every row.
            return cute.make_tensor(tensor.iterator, layout)

        return CuteWeightScaleViews(
            fc1_weight_sfa_logical=make_view(fc1_weight_scales, self.fc1_logical_shape),
            fc2_weight_sfa_logical=make_view(fc2_weight_scales, self.fc2_logical_shape),
        )


__all__ = [
    "CuteWeightScaleViews",
    "CuteWeightScaleArguments",
    "SF_ATOM_BYTES",
    "SF_ATOM_K_BLOCKS",
    "SF_ATOM_ROWS",
    "SF_VECTOR_SIZE",
    "SwizzledWeightScaleViews",
    "atom_swizzled_offset",
    "from_atom_swizzled",
    "swizzled_scale_numel",
    "to_atom_swizzled",
]
