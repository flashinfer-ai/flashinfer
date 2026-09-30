"""Checkpoint-side fused-MoE weight representations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .._shared.trellis_codebooks import validate_codebook_bits
from .config import RateGranularity, TrellisConfig


class PackedSourceFormat(str, Enum):
    """Packed checkpoint encoding, including its scale-grid contract."""

    MXFP4_E8M0_K32 = "fp4_e8m0_k32"
    MODELOPT_NVFP4 = "modelopt_nvfp4"
    COMPRESSED_TENSORS_FP4 = "compressed_tensors"
    MXFP6_E8M0_K32 = "mxfp6_e2m3"
    IQ2_XS = "iq2_xs"
    IQ2_XXS = "iq2_xxs"
    Q8_0 = "q8_0"


class W13Layout(str, Enum):
    """Logical order of the two gated FC1 projections."""

    W13 = "w13"
    W31 = "w31"


@dataclass(frozen=True, kw_only=True)
class PackedSource:
    """Source encoding and packing of one ordinary MoE checkpoint."""

    format: PackedSourceFormat
    w13_layout: W13Layout = W13Layout.W13

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", PackedSourceFormat(self.format))
        object.__setattr__(self, "w13_layout", W13Layout(self.w13_layout))


@dataclass(frozen=True, kw_only=True)
class TrellisExtent:
    """Contiguous 32-channel slots in the checkpoint-global intermediate axis.

    Sign sequences are generated at ``global_intermediate_size`` and sliced,
    never regenerated using a tensor-parallel rank's shorter local length.
    """

    global_intermediate_size: int
    first_slot: int
    slot_count: int

    def __post_init__(self) -> None:
        for name in ("global_intermediate_size", "first_slot", "slot_count"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"TrellisExtent.{name} must be an integer")
        if self.global_intermediate_size <= 0 or self.global_intermediate_size % 32:
            raise ValueError(
                "global_intermediate_size must be a positive multiple of 32"
            )
        if self.first_slot < 0 or self.slot_count <= 0:
            raise ValueError(
                "a trellis extent requires first_slot >= 0 and slot_count > 0"
            )
        if 32 * (self.first_slot + self.slot_count) > self.global_intermediate_size:
            raise ValueError("trellis extent exceeds the global intermediate axis")

    @property
    def intermediate_size(self) -> int:
        return 32 * self.slot_count


@dataclass(frozen=True, kw_only=True)
class TrellisSource:
    """Container-independent trellis encoding and optional global coordinates.

    ``uniform_bits`` declares the scalar rate before weights are transferred;
    preparation verifies it against the rate tensor. Omitting it retains the
    native config's K3 LUT or K3/K4/K5 projection-tiered MCG contract.
    """

    config: TrellisConfig
    uniform_bits: int | None = None
    extent: TrellisExtent | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, TrellisConfig):
            raise TypeError("TrellisSource.config must be a TrellisConfig")
        if self.uniform_bits is not None:
            if type(self.uniform_bits) is not int or self.uniform_bits not in (
                2,
                3,
                4,
                5,
                6,
            ):
                raise ValueError("uniform_bits must be one of 2, 3, 4, 5, 6")
            if self.config.rate.granularity is not RateGranularity.UNIFORM:
                raise ValueError("uniform_bits requires uniform rate granularity")
            validate_codebook_bits(self.config.codebook.value, self.uniform_bits)
            if self.config.codebook.value == "mcg" and self.uniform_bits < 3:
                raise ValueError("MCG trellis execution requires at least K3")
        if self.extent is not None and not isinstance(self.extent, TrellisExtent):
            raise TypeError("TrellisSource.extent must be a TrellisExtent or None")


WeightSource: TypeAlias = PackedSource | TrellisConfig | TrellisSource


__all__ = [
    "TrellisExtent",
    "TrellisSource",
    "PackedSource",
    "PackedSourceFormat",
    "W13Layout",
    "WeightSource",
]
