"""Prepared weight layout for the W8A8 MXFP8 MoE kernel.

Host-side, one-time preparation for the ``w8a8_mx`` quant recipe
(``source_format='mxfp8_e8m0_k32'``):

* MXFP8 E4M3 weight codes stay **source-native and byte-exact**:
  ``[E, N, K]`` with one E4M3 byte per element.  Nothing is requantized to
  FP4/FP6 and nothing is re-packed; the checkpoint bytes are the runtime
  operand bytes.
* Checkpoint UE8M0 K/32 block scales (``[E, N, K//32]`` bytes, unswizzled on
  disk) are validated by the canonical
  :func:`b12x._lib.intrinsics.validate_e8m0_scale_grid` rule (every finite
  UE8M0 byte preserved, ``0xFF`` NaN rejected, never clamped) and then swizzled
  by the canonical :func:`b12x._lib.intrinsics.swizzle_block_scale` into the
  MMA block-scale layout the kernel reads — the same two helpers the MX-FP6
  recipe uses, so there is exactly one scale layout in the tree.
* Per-expert f32 ``*_weight_scale_2`` globals combine with the reciprocal
  activation global scales into runtime alphas
  (``alpha = weight_scale_2 / a_gscale``).

Activations are quantized at runtime to FP8 E4M3 with UE8M0 K/32 block scales,
so both operands are genuine E4M3 and feed the ``MmaMXF8Op`` block-scaled
``m16n8k32`` MMA natively (no FP6 byte-container expansion anywhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from b12x._lib.intrinsics import (
    swizzle_block_scale,
    validate_e8m0_scale_grid,
)

_SF_BLOCK = 32  # K elements per UE8M0 block scale
_TILE_K = 128  # kernel K-tile width (one E4M3 byte per element along K)
_VALUE_DTYPES = (torch.uint8, torch.float8_e4m3fn)


@dataclass(frozen=True, kw_only=True)
class PreparedW8A8MXFP8Weights:
    """Runtime contract for prepared W8A8 MXFP8 MoE expert weights.

    ``w13_values``/``w2_values`` are the byte-exact source E4M3 codes
    (``[E, N, K]`` uint8, one code per byte, K == logical K); ``*_sf_swizzled``
    are UE8M0 block scales in the MMA swizzle
    (``[E, pad128(rows), pad4(K//32)]`` bytes); ``*_alpha`` are per-expert f32
    runtime dequant scales.  This container is the input contract of the
    ``quant_recipe="w8a8_mx"`` dynamic kernel.

    When ``intermediate_size`` is not a multiple of 128 the gate half of FC1
    starts inside a 128-row scale atom, so the kernel streams the up and gate
    halves through independent TMA descriptors.  ``w13_sf_swizzled`` then
    holds the two halves swizzled independently, half-major:
    ``[2, E, pad128(I), pad4(K//32)]`` (index 0 = up, 1 = gate).
    ``launch_views`` holds the dynamic-launch operand views and pointers that
    ``b12x.moe.fused_moe`` resolves once when it prepares the experts; the
    launch path reads them instead of re-deriving geometry per call.
    """

    w13_values: torch.Tensor
    w2_values: torch.Tensor
    w13_sf_swizzled: torch.Tensor
    w2_sf_swizzled: torch.Tensor
    w13_alpha: torch.Tensor
    w2_alpha: torch.Tensor
    num_experts: int
    hidden_size: int
    intermediate_size: int
    launch_views: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Coerce the geometry fields to ``int``."""
        object.__setattr__(self, "num_experts", int(self.num_experts))
        object.__setattr__(self, "hidden_size", int(self.hidden_size))
        object.__setattr__(self, "intermediate_size", int(self.intermediate_size))

    # Canonical-storage aliases consumed by the generic prepared-weight
    # plumbing in b12x.moe.fused_moe._impl (mirrors the w13/w2 attribute
    # lookups used for the W4A16 native containers).
    @property
    def w13(self) -> torch.Tensor:
        """FC1 E4M3 code bytes (canonical-storage alias)."""
        return self.w13_values

    @property
    def w2(self) -> torch.Tensor:
        """FC2 E4M3 code bytes (canonical-storage alias)."""
        return self.w2_values

    @property
    def w13_scale(self) -> torch.Tensor:
        """FC1 swizzled UE8M0 scales (canonical-storage alias)."""
        return self.w13_sf_swizzled

    @property
    def w2_scale(self) -> torch.Tensor:
        """FC2 swizzled UE8M0 scales (canonical-storage alias)."""
        return self.w2_sf_swizzled

    @property
    def w13_global_scale(self) -> torch.Tensor:
        """FC1 per-expert runtime alphas (canonical-storage alias)."""
        return self.w13_alpha

    @property
    def w2_global_scale(self) -> torch.Tensor:
        """FC2 per-expert runtime alphas (canonical-storage alias)."""
        return self.w2_alpha


def _validate_value_bytes(
    values: torch.Tensor,
    *,
    name: str,
    num_experts: int,
    rows: int,
    k: int,
) -> torch.Tensor:
    """Return the contiguous uint8 view of ``values`` without touching bytes."""
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.dtype not in _VALUE_DTYPES:
        raise TypeError(
            f"{name} must hold MXFP8 E4M3 codes (uint8/float8_e4m3fn), "
            f"got {values.dtype}"
        )
    expected = (num_experts, rows, k)
    if values.dim() != 3 or tuple(values.shape) != expected:
        raise ValueError(
            f"{name} must have one E4M3 byte per element with shape "
            f"{expected} (E, N, K), got {tuple(values.shape)}"
        )
    uint8 = values.view(torch.uint8)
    if not uint8.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return uint8


def _validate_e8m0_scale_grid(
    scale: torch.Tensor,
    *,
    name: str,
    num_experts: int,
    rows: int,
    k: int,
) -> torch.Tensor:
    """Apply the canonical UE8M0 grid rule (dtype, extent, finite bytes)."""
    return validate_e8m0_scale_grid(
        scale,
        name=name,
        num_experts=num_experts,
        rows=rows,
        k=k,
        sf_block=_SF_BLOCK,
    )


def _validate_expert_scalars(
    scale: torch.Tensor, *, name: str, num_experts: int
) -> torch.Tensor:
    """Return a contiguous ``[E]`` f32 view of a scalar or per-expert scale."""
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if scale.numel() == 1:
        return scale.reshape(()).expand(num_experts).to(torch.float32).contiguous()
    if scale.numel() != num_experts:
        raise ValueError(
            f"{name} must have 1 or {num_experts} elements, got {scale.numel()}"
        )
    return scale.reshape(num_experts).to(torch.float32).contiguous()


def prepare_w8a8_mxfp8_weights(
    *,
    w13_values: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2_values: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> PreparedW8A8MXFP8Weights:
    """Prepare W8A8 MXFP8 expert weights for the fused MoE kernel.

    Args:
        w13_values: FC1 (gated up+gate) E4M3 codes,
            ``[E, 2*intermediate, hidden]`` uint8 or float8_e4m3fn.
        w13_scale: FC1 UE8M0 K/32 grid, ``[E, 2*intermediate, hidden//32]``.
        w13_scale_2: per-expert f32 FC1 global weight scales (``[E]`` or scalar).
        w2_values: FC2 (down) E4M3 codes, ``[E, hidden, intermediate]``.
        w2_scale: FC2 UE8M0 K/32 grid, ``[E, hidden, intermediate//32]``.
        w2_scale_2: per-expert f32 FC2 global weight scales.
        a1_gscale / a2_gscale: reciprocal activation global scales for the
            FC1/FC2 inputs; default is unit scale.  Runtime alphas follow the
            established preparation math ``alpha = weight_scale_2 / a_gscale``.
    """
    num_experts = int(num_experts)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    for name, value in (
        ("num_experts", num_experts),
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if hidden_size % _TILE_K != 0:
        raise ValueError(
            f"W8A8 MXFP8 requires hidden_size % {_TILE_K} == 0, got "
            f"hidden_size={hidden_size}"
        )
    if intermediate_size % _SF_BLOCK != 0:
        raise ValueError(
            f"W8A8 MXFP8 requires intermediate_size % {_SF_BLOCK} == 0 (one "
            f"UE8M0 scale per FC2 K block), got intermediate_size="
            f"{intermediate_size}"
        )

    w13_rows = 2 * intermediate_size  # gated silu FC1: [up; gate] stacked rows
    w13_values = _validate_value_bytes(
        w13_values,
        name="w13_values",
        num_experts=num_experts,
        rows=w13_rows,
        k=hidden_size,
    )
    w2_values = _validate_value_bytes(
        w2_values,
        name="w2_values",
        num_experts=num_experts,
        rows=hidden_size,
        k=intermediate_size,
    )
    w13_grid = _validate_e8m0_scale_grid(
        w13_scale,
        name="w13_scale",
        num_experts=num_experts,
        rows=w13_rows,
        k=hidden_size,
    )
    w2_grid = _validate_e8m0_scale_grid(
        w2_scale,
        name="w2_scale",
        num_experts=num_experts,
        rows=hidden_size,
        k=intermediate_size,
    )

    w13_scale_2 = _validate_expert_scalars(
        w13_scale_2, name="w13_scale_2", num_experts=num_experts
    )
    w2_scale_2 = _validate_expert_scalars(
        w2_scale_2, name="w2_scale_2", num_experts=num_experts
    )
    if a1_gscale is None:
        w13_alpha = w13_scale_2.clone()
    else:
        a1 = _validate_expert_scalars(
            a1_gscale, name="a1_gscale", num_experts=num_experts
        )
        w13_alpha = torch.empty_like(w13_scale_2)
        torch.div(w13_scale_2, a1, out=w13_alpha)
    if a2_gscale is None:
        w2_alpha = w2_scale_2.clone()
    else:
        a2 = _validate_expert_scalars(
            a2_gscale, name="a2_gscale", num_experts=num_experts
        )
        w2_alpha = torch.empty_like(w2_scale_2)
        torch.div(w2_scale_2, a2, out=w2_alpha)
    for name, alpha in (("w13", w13_alpha), ("w2", w2_alpha)):
        if not bool(torch.isfinite(alpha).all().item()):
            raise ValueError(
                f"{name}_alpha = {name}_scale_2 / a_gscale is non-finite; a "
                "zero activation global scale is a preparation error"
            )
    # One canonical swizzle: swizzle_block_scale already pads rows to 128 and
    # scale columns to 4, so no second shape contract is asserted here.
    if w13_split_halves(intermediate_size):
        # The gate half begins mid-atom; swizzle each half on its own so both
        # FC1 TMA descriptors address 128-row atoms from row zero.
        w13_sf_swizzled = torch.stack(
            [
                swizzle_block_scale(half.contiguous().view(torch.float8_e8m0fnu))
                for half in (
                    w13_grid[:, :intermediate_size],
                    w13_grid[:, intermediate_size:],
                )
            ]
        )
    else:
        w13_sf_swizzled = swizzle_block_scale(w13_grid.view(torch.float8_e8m0fnu))
    w2_sf_swizzled = swizzle_block_scale(w2_grid.view(torch.float8_e8m0fnu))

    return PreparedW8A8MXFP8Weights(
        w13_values=w13_values,
        w2_values=w2_values,
        w13_sf_swizzled=w13_sf_swizzled.view(torch.uint8).contiguous(),
        w2_sf_swizzled=w2_sf_swizzled.view(torch.uint8).contiguous(),
        w13_alpha=w13_alpha,
        w2_alpha=w2_alpha,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def w13_split_halves(intermediate_size: int) -> bool:
    """Whether FC1 up/gate halves use independent descriptors and scale grids."""
    return int(intermediate_size) % _TILE_K != 0


__all__ = [
    "PreparedW8A8MXFP8Weights",
    "prepare_w8a8_mxfp8_weights",
    "w13_split_halves",
]
