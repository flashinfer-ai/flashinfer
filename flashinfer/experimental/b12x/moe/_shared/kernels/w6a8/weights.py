"""Prepared weight layout for the W6A8 MX-FP6 MoE kernel.

Host-side, one-time preparation for the ``w6a8_mx`` quant recipe
(``source_format='mxfp6_e2m3'``):

* Packed MX-FP6 E2M3 weight codes stay **source-native**: ``[E, N, 3*K//4]``
  uint8 with four 6-bit values packed into three bytes along K.
* Checkpoint UE8M0 K/32 block scales (``[E, N, K//32]`` bytes, unswizzled on
  disk) are swizzled into the MMA block-scale layout consumed by the kernel.
* Per-expert f32 ``*_weight_scale_2`` globals combine with the reciprocal
  activation global scales into runtime alphas
  (``alpha = weight_scale_2 / a_gscale``).

Activations are quantized at runtime to FP8 E4M3 with UE8M0 K/32 block
scales; both operands feed the same mxf8f6f4 MMA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.intrinsics import align_up, swizzle_block_scale

_SF_BLOCK = 32  # K elements per UE8M0 block scale
_TILE_K = 128  # kernel K-tile width (packed 3:4 along K)
_FP6_PACK_NUM = 4  # logical values per packed group
_FP6_PACK_BYTES = 3  # bytes per packed group


def _packed_k_bytes(k: int) -> int:
    """Bytes per row of ``k`` FP6 values packed 4-values-in-3-bytes."""
    return (k // _FP6_PACK_NUM) * _FP6_PACK_BYTES


@dataclass(frozen=True, kw_only=True)
class PreparedW6A8MXFP6Weights:
    """Runtime contract for prepared W6A8 MX-FP6 MoE expert weights.

    ``w13_packed``/``w2_packed`` are the unmodified source FP6 code bytes;
    ``*_sf_swizzled`` are UE8M0 block scales in the MMA swizzle
    (``[E, pad128(rows), pad4(K//32)]`` bytes); ``*_alpha`` are per-expert
    f32 runtime dequant scales.  This container is the input contract of the
    ``quant_recipe="w6a8_mx"`` dynamic-kernel port.
    """

    w13_packed: torch.Tensor
    w2_packed: torch.Tensor
    w13_sf_swizzled: torch.Tensor
    w2_sf_swizzled: torch.Tensor
    w13_alpha: torch.Tensor
    w2_alpha: torch.Tensor
    num_experts: int
    hidden_size: int
    intermediate_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_experts", int(self.num_experts))
        object.__setattr__(self, "hidden_size", int(self.hidden_size))
        object.__setattr__(
            self, "intermediate_size", int(self.intermediate_size)
        )

    # Canonical-storage aliases consumed by the generic prepared-weight
    # plumbing in b12x.moe.fused_moe._impl (mirrors w13/w2 attribute
    # lookups used for the W4A16 native containers).
    @property
    def w13(self) -> torch.Tensor:
        return self.w13_packed

    @property
    def w2(self) -> torch.Tensor:
        return self.w2_packed

    @property
    def w13_scale(self) -> torch.Tensor:
        return self.w13_sf_swizzled

    @property
    def w2_scale(self) -> torch.Tensor:
        return self.w2_sf_swizzled

    @property
    def w13_global_scale(self) -> torch.Tensor:
        return self.w13_alpha

    @property
    def w2_global_scale(self) -> torch.Tensor:
        return self.w2_alpha


def _validate_packed_codes(
    packed: torch.Tensor,
    *,
    name: str,
    num_experts: int,
    rows: int,
    k: int,
) -> torch.Tensor:
    if not isinstance(packed, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if packed.dtype != torch.uint8:
        raise TypeError(
            f"{name} must be packed MX-FP6 uint8 codes, got {packed.dtype}"
        )
    expected = (num_experts, rows, _packed_k_bytes(k))
    if packed.dim() != 3 or tuple(packed.shape) != expected:
        raise ValueError(
            f"{name} must have packed shape {expected} "
            f"(E, N, 3*K//4), got {tuple(packed.shape)}"
        )
    if not packed.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return packed


def _validate_e8m0_scale_grid(
    scale: torch.Tensor,
    *,
    name: str,
    num_experts: int,
    rows: int,
    k: int,
) -> torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError(
            f"{name} must hold UE8M0 bytes (uint8/float8_e8m0fnu), "
            f"got {scale.dtype}"
        )
    expected = (num_experts, rows, k // _SF_BLOCK)
    if scale.dim() != 3 or tuple(scale.shape) != expected:
        raise ValueError(
            f"{name} must be an unswizzled per-K/{_SF_BLOCK} grid with shape "
            f"{expected}, got {tuple(scale.shape)}"
        )
    return scale.view(torch.uint8).contiguous()


def _validate_expert_scalars(
    scale: torch.Tensor, *, name: str, num_experts: int
) -> torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if scale.numel() == 1:
        return scale.reshape(()).expand(num_experts).to(torch.float32).contiguous()
    if scale.numel() != num_experts:
        raise ValueError(
            f"{name} must have 1 or {num_experts} elements, got {scale.numel()}"
        )
    return scale.reshape(num_experts).to(torch.float32).contiguous()


def _swizzle_e8m0_grid(grid_u8: torch.Tensor) -> torch.Tensor:
    """Swizzle an unswizzled ``[E, rows, blocks]`` UE8M0 grid to MMA layout.

    Returns ``[E, pad128(rows), pad4(blocks)]`` uint8 (zero-padded), matching
    the cutlass block-scale swizzle the kernel reads.
    """
    swizzled = swizzle_block_scale(grid_u8.view(torch.float8_e8m0fnu))
    return swizzled.view(torch.uint8).contiguous()


def prepare_w6a8_mxfp6_weights(
    *,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> PreparedW6A8MXFP6Weights:
    """Prepare W6A8 MX-FP6 expert weights for the fused MoE kernel.

    Args:
        w13_packed: FC1 (gated up+gate) packed E2M3 codes,
            ``[E, 2*intermediate, 3*hidden//4]`` uint8.
        w13_scale: FC1 UE8M0 K/32 grid, ``[E, 2*intermediate, hidden//32]``.
        w13_scale_2: per-expert f32 FC1 global weight scales (``[E]`` or scalar).
        w2_packed: FC2 (down) packed codes, ``[E, hidden, 3*intermediate//4]``.
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
    if hidden_size % _TILE_K != 0 or intermediate_size % _TILE_K != 0:
        raise ValueError(
            f"W6A8 MX-FP6 requires hidden_size % {_TILE_K} == 0 and "
            f"intermediate_size % {_TILE_K} == 0, got hidden_size="
            f"{hidden_size}, intermediate_size={intermediate_size}"
        )

    w13_rows = 2 * intermediate_size  # gated silu FC1: [up; gate] stacked rows
    w13_packed = _validate_packed_codes(
        w13_packed,
        name="w13_packed",
        num_experts=num_experts,
        rows=w13_rows,
        k=hidden_size,
    )
    w2_packed = _validate_packed_codes(
        w2_packed,
        name="w2_packed",
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

    w13_sf_swizzled = _swizzle_e8m0_grid(w13_grid)
    w2_sf_swizzled = _swizzle_e8m0_grid(w2_grid)
    expected_w13_sf = (
        num_experts,
        align_up(w13_rows, 128),
        align_up(hidden_size // _SF_BLOCK, 4),
    )
    if tuple(w13_sf_swizzled.shape) != expected_w13_sf:
        raise AssertionError(
            f"swizzled w13 scale shape {tuple(w13_sf_swizzled.shape)} != "
            f"{expected_w13_sf}"
        )
    expected_w2_sf = (
        num_experts,
        align_up(hidden_size, 128),
        align_up(intermediate_size // _SF_BLOCK, 4),
    )
    if tuple(w2_sf_swizzled.shape) != expected_w2_sf:
        raise AssertionError(
            f"swizzled w2 scale shape {tuple(w2_sf_swizzled.shape)} != "
            f"{expected_w2_sf}"
        )

    return PreparedW6A8MXFP6Weights(
        w13_packed=w13_packed,
        w2_packed=w2_packed,
        w13_sf_swizzled=w13_sf_swizzled,
        w2_sf_swizzled=w2_sf_swizzled,
        w13_alpha=w13_alpha,
        w2_alpha=w2_alpha,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


__all__ = [
    "PreparedW6A8MXFP6Weights",
    "prepare_w6a8_mxfp6_weights",
]
