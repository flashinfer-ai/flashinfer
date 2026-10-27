# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-only NVFP4 policy: FC1 K256 schedule selection and conversion layout.

Stdlib only and loadable as a standalone module: no package-relative imports
or CUDA initialization.
"""

import dataclasses
from typing import Any, Dict, Mapping, Optional, Tuple


NVFP4_FC1_SCHEDULE_VERSION = "nvfp4_fc1_k256_auto_overlap_v1"

# K-permuted weight layouts (the permutation itself lives with the weight
# preprocessing): 0 keeps natural K order; 32/64 permute K inside each K32/K64
# chunk so that a lane's eight A-fragment values in one row share one K16
# scale (32), and one scale serves a K32 fragment pair (64).
NVFP4_KPERM_CHOICES = (0, 32, 64)
# Identical replicas of the 1 KiB conversion table; they only spread lanes
# over shared-memory banks. 32 replicas use 256-byte rows whose addresses are
# built with one PRMT per scale byte.
NVFP4_LUT_REPLICA_CHOICES = (8, 16, 32)
# Lane -> replica map: lane quad (default), lane pair, or lane.
NVFP4_LUT_MAPS = ("quad", "pair", "lane")
# NVFP4-only tactic fields and the values a legacy (MXFP4-schema) tactic
# implies. They depend on the tile: a 32-replica table costs 24 KiB more
# shared memory, which N32/N64 tiles recover with the compact pull buffer.
NVFP4_CONVERSION_TACTIC_DEFAULTS: Dict[str, Any] = {
    "compact_pull_buffer": False,
    "nvfp4_lut_replicas": 8,
    "nvfp4_lut_map": "quad",
}


def validate_nvfp4_kperm(kperm: Any) -> int:
    if type(kperm) is not int or kperm not in NVFP4_KPERM_CHOICES:
        raise ValueError(f"NVFP4 kperm must be 0, 32 or 64; got {kperm!r}")
    return int(kperm)


@dataclasses.dataclass(frozen=True)
class Nvfp4ConversionLayout:
    """Compile-time layout of the in-register NVFP4 promotion.

    ``kperm`` must match the weight preprocessing and activation staging.
    The LUT fields change only shared-memory banking and size, never values.
    """

    kperm: int = 64
    lut_replicas: int = 8
    lut_map: str = "quad"

    def __post_init__(self) -> None:
        validate_nvfp4_kperm(self.kperm)
        if (
            type(self.lut_replicas) is not int
            or self.lut_replicas not in NVFP4_LUT_REPLICA_CHOICES
        ):
            raise ValueError(
                f"nvfp4_lut_replicas must be 8, 16 or 32; got {self.lut_replicas!r}"
            )
        if self.lut_map not in NVFP4_LUT_MAPS:
            raise ValueError(
                f"nvfp4_lut_map must be 'quad', 'pair' or 'lane'; got {self.lut_map!r}"
            )

    @property
    def lut_bytes(self) -> int:
        # 127 finite scale codes plus a zero row, one u64 per replica.
        return 128 * 8 * self.lut_replicas


NVFP4_DEFAULT_CONVERSION = Nvfp4ConversionLayout()


def split_nvfp4_conversion_tactic(
    tactic: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (MXFP4-schema fields, validated NVFP4 conversion fields)."""
    nvfp4 = {
        name: tactic.get(name, default)
        for name, default in NVFP4_CONVERSION_TACTIC_DEFAULTS.items()
    }
    if type(nvfp4["compact_pull_buffer"]) is not bool:
        raise ValueError("compact_pull_buffer must be a bool")
    layout = Nvfp4ConversionLayout(
        lut_replicas=nvfp4["nvfp4_lut_replicas"], lut_map=nvfp4["nvfp4_lut_map"]
    )
    core = {
        name: value
        for name, value in tactic.items()
        if name not in NVFP4_CONVERSION_TACTIC_DEFAULTS
    }
    return core, {
        "compact_pull_buffer": nvfp4["compact_pull_buffer"],
        "nvfp4_lut_replicas": layout.lut_replicas,
        "nvfp4_lut_map": layout.lut_map,
    }


def validate_nvfp4_fc1_overlap_override(
    override: Optional[bool],
    *,
    mma_tiler_mnk: Tuple[int, int, int],
    fp8_scale_mode: str,
    pingpong: bool,
) -> None:
    """A forced schedule is allowed only on its implemented FC1 path."""
    if override is not None and not isinstance(override, bool):
        raise ValueError("nvfp4_fc1_k256_overlap must be None or a bool")
    if override is True and (
        mma_tiler_mnk[2] != 256 or fp8_scale_mode != "mxfp4_hybrid" or pingpong
    ):
        raise ValueError(
            "nvfp4_fc1_k256_overlap=True requires hybrid K256 without pingpong"
        )


def nvfp4_fc1_auto_shape_eligible(
    *,
    num_tokens: int,
    hidden: int,
    intermediate: int,
    num_total_experts: int,
    num_topk: int,
    world_size: int,
    mma_tiler_mnk: Tuple[int, int, int],
    cluster_shape_mnk: Tuple[int, int, int],
    fp8_scale_mode: str,
    pingpong: bool,
) -> bool:
    """Bound auto selection to the shape and token range measured on H200."""
    return (
        4096 <= num_tokens <= 32768
        and (hidden, intermediate, num_total_experts, num_topk, world_size)
        == (7168, 3072, 384, 6, 4)
        and tuple(mma_tiler_mnk) == (256, 64, 256)
        and tuple(cluster_shape_mnk) == (1, 2, 1)
        and fp8_scale_mode == "mxfp4_hybrid"
        and not pingpong
    )


def resolve_nvfp4_fc1_overlap(
    *,
    override: Optional[bool],
    device_name: Optional[str],
    sm_count: Optional[int],
    **shape,
) -> bool:
    """Resolve a host boolean; this value is part of compilation identity."""
    validate_nvfp4_fc1_overlap_override(
        override,
        mma_tiler_mnk=shape["mma_tiler_mnk"],
        fp8_scale_mode=shape["fp8_scale_mode"],
        pingpong=shape["pingpong"],
    )
    if override is not None:
        return override
    return (
        nvfp4_fc1_auto_shape_eligible(**shape)
        and device_name is not None
        and device_name == "NVIDIA H200"
        and sm_count == 132
    )
