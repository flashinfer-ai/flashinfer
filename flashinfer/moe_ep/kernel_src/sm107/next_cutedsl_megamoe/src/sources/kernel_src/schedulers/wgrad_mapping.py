"""Wgrad work-tile ABI and expert token-range mapping."""

import dataclasses
from enum import IntEnum
from typing import List, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values

from .base import SchedulerWorkTileBase


class WgradWorkTileState(IntEnum):
    """Sentinel values carried in the expert index field."""

    Done = -1


@dataclasses.dataclass(frozen=True)
class WgradWorkTileInfo(SchedulerWorkTileBase):
    """CTA-level output tile and reduction extent for one expert."""

    storage_field_count = 4

    expert_idx: Int32
    tile_m_idx: Int32
    tile_n_idx: Int32
    k_tile_count: Int32

    @property
    def is_valid_tile(self):
        return self.expert_idx >= Int32(0)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (self.expert_idx, self.tile_m_idx, self.tile_n_idx, self.k_tile_count):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "WgradWorkTileInfo":
        if len(values) != self.storage_field_count:
            raise ValueError(f"WgradWorkTileInfo expects {self.storage_field_count} MLIR values, got {len(values)}.")
        fields = (self.expert_idx, self.tile_m_idx, self.tile_n_idx, self.k_tile_count)
        rebuilt = [new_from_mlir_values(field, [value]) for field, value in zip(fields, values)]
        return type(self)(*rebuilt)

    def to_rmem(self) -> cute.Tensor:
        registers = cute.make_rmem_tensor((self.storage_field_count,), cutlass.Int32)
        registers[0] = self.expert_idx
        registers[1] = self.tile_m_idx
        registers[2] = self.tile_n_idx
        registers[3] = self.k_tile_count
        return registers

    @classmethod
    def from_rmem(cls, registers: cute.Tensor) -> "WgradWorkTileInfo":
        return cls(expert_idx=registers[0], tile_m_idx=registers[1], tile_n_idx=registers[2], k_tile_count=registers[3])


@cute.jit
def compute_expert_token_range(offs: cute.Tensor, expert_idx: Int32) -> Tuple[Int32, Int32]:
    """Return the token offset and count from cumulative expert ends."""
    token_start = Int32(0)
    if expert_idx > Int32(0):
        token_start = offs[expert_idx - Int32(1)]
    return token_start, offs[expert_idx] - token_start


@cute.jit
def make_wgrad_work_tile(
    expert_idx: Int32, tile_m_idx: Int32, tile_n_idx: Int32, offs: cute.Tensor, cta_tile_k: int
) -> WgradWorkTileInfo:
    """Build one CTA-level Wgrad tile, including an empty expert."""
    _, token_count = compute_expert_token_range(offs, expert_idx)
    k_tile_count = (token_count + Int32(cta_tile_k) - Int32(1)) // Int32(cta_tile_k)
    return WgradWorkTileInfo(
        expert_idx=expert_idx, tile_m_idx=tile_m_idx, tile_n_idx=tile_n_idx, k_tile_count=k_tile_count
    )


@cute.jit
def make_wgrad_done_tile() -> WgradWorkTileInfo:
    """Build the terminal scheduler tile."""
    return WgradWorkTileInfo(
        expert_idx=Int32(WgradWorkTileState.Done), tile_m_idx=Int32(0), tile_n_idx=Int32(0), k_tile_count=Int32(0)
    )


__all__ = [
    "WgradWorkTileInfo",
    "WgradWorkTileState",
    "compute_expert_token_range",
    "make_wgrad_done_tile",
    "make_wgrad_work_tile",
]
