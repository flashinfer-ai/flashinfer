"""Optional Rubin shared-expert geometry and format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Sm107SharedExpertConfig:
    """A replicated MXFP8 shared FFN added to the routed output on each rank."""

    intermediate_size: int
    kind: Literal["mxfp8_e4m3", "mxfp8_e5m2"] = "mxfp8_e4m3"
    l2_position: Literal["early", "tail"] | None = None
    work_id_mode: Literal["grid_stride", "atomic_counter"] | None = None
    fc1_swizzle: int | None = None

    def __post_init__(self):
        if (
            type(self.intermediate_size) is not int
            or self.intermediate_size <= 0
            or self.intermediate_size % 128
        ):
            raise ValueError(
                "shared intermediate_size must be a positive multiple of 128"
            )
        if self.kind not in ("mxfp8_e4m3", "mxfp8_e5m2"):
            raise ValueError("shared experts require MXFP8 E4M3 or E5M2")
        if self.l2_position not in (None, "early", "tail"):
            raise ValueError("shared l2_position must be early, tail, or None")
        if self.work_id_mode not in (None, "grid_stride", "atomic_counter"):
            raise ValueError("unsupported shared work_id_mode")
        if self.fc1_swizzle is not None and (
            type(self.fc1_swizzle) is not int or self.fc1_swizzle < 0
        ):
            raise ValueError("shared fc1_swizzle must be a nonnegative integer or None")


def validate_feature_config(config):
    shared = getattr(config, "shared_expert", None)
    if shared is not None and not isinstance(shared, Sm107SharedExpertConfig):
        raise ValueError("shared_expert must be Sm107SharedExpertConfig or None")
    if type(config.combine_dedup) is not bool:
        raise ValueError("combine_dedup must be a bool")
    if getattr(config, "batch_invariant", False) and (
        shared is not None or config.combine_dedup
    ):
        raise ValueError(
            "shared fusion and dedup do not implement the batch-invariant contract"
        )
