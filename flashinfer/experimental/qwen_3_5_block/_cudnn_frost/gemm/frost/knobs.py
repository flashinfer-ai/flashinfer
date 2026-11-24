# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FROST GEMM engine's knobs: one :class:`~cudnn.gemm.frost.tile_config.TileConfig` spelled in the shared ``cudnn.knob_type`` vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

from .tile_config import _CONFIG_NAME_RE, TileConfig, by_name


@dataclass(frozen=True)
class GemmKnobs:
    pipeline_arch: int  # kernel template family as its SM number: 100 -> "sm100"
    cta_tile_m: int
    cta_tile_n: int
    cta_tile_k_bytes: int
    mma_tile_m: int
    mma_tile_n: int
    mma_tile_k_bytes: int
    cga_size_m: int  # cluster shape (CTAs along M / N)
    cga_size_n: int
    cta_group: Optional[int] = None  # CTAs cooperating on one MMA tile; CTA-pair families only
    warps_m: Optional[int] = None  # warp grid over the CTA tile; warp-scoped families (sm120) only
    warps_n: Optional[int] = None
    split_k_slices: int = 1
    swap_ab: bool = False
    cluster_k_reduce: bool = False

    # (field, cudnn.knob_type member, required)
    _PUBLIC_KNOBS = (
        ("pipeline_arch", "PIPELINE_ARCH", True),
        ("cta_tile_m", "TILE_M", True),
        ("cta_tile_n", "TILE_N", True),
        ("cta_tile_k_bytes", "TILEK", True),
        ("mma_tile_m", "MMA_TILE_M", True),
        ("mma_tile_n", "MMA_TILE_N", True),
        ("mma_tile_k_bytes", "MMA_TILE_K", True),
        ("cga_size_m", "TILE_CGA_M", True),
        ("cga_size_n", "TILE_CGA_N", True),
        ("cta_group", "CTA_GROUP", False),
        ("warps_m", "WARPS_M", False),
        ("warps_n", "WARPS_N", False),
        ("split_k_slices", "SPLIT_K_SLC", False),
        ("swap_ab", "SWAP_AB", False),
        ("cluster_k_reduce", "CLUSTER_K_REDUCE", False),
    )

    @classmethod
    def from_config(cls, cfg: TileConfig) -> "GemmKnobs":
        """The knobs that name ``cfg``: read off its canonical name, the one
        spelling ``by_name`` accepts."""
        m = _CONFIG_NAME_RE.match(cfg.name)
        if m is None:
            raise ValueError(f"tile config {cfg.name!r} does not spell a canonical CONFIG_<pipeline>_... name")
        pipeline = m.group("pipeline")
        if not pipeline.startswith("sm") or not pipeline[2:].isdigit():
            raise ValueError(f"tile config pipeline {pipeline!r} is not an smNNN family")
        return cls(
            pipeline_arch=int(pipeline[2:]),
            cta_tile_m=int(m.group("cta_m")),
            cta_tile_n=int(m.group("cta_n")),
            cta_tile_k_bytes=int(m.group("k_bytes")),
            mma_tile_m=int(m.group("mma_m")),
            mma_tile_n=int(m.group("mma_n")),
            mma_tile_k_bytes=int(m.group("mma_k_bytes")),
            cga_size_m=int(m.group("cga_m")),
            cga_size_n=int(m.group("cga_n")),
            cta_group=int(m.group("cta_group")) if m.group("cta_group") else None,
            warps_m=int(m.group("warps_m")) if m.group("warps_m") else None,
            warps_n=int(m.group("warps_n")) if m.group("warps_n") else None,
            split_k_slices=int(m.group("split_k") or 1),
            swap_ab=bool(m.group("swap_ab")),
            cluster_k_reduce=bool(m.group("cluster_k_reduce")),
        )

    @property
    def config_name(self) -> str:
        """The canonical TileConfig name these knobs spell."""
        name = (
            f"CONFIG_sm{self.pipeline_arch}_"
            f"{self.cta_tile_m}x{self.cta_tile_n}x{self.cta_tile_k_bytes}_"
            f"{self.mma_tile_m}x{self.mma_tile_n}x{self.mma_tile_k_bytes}_"
            f"cluster{self.cga_size_m}x{self.cga_size_n}"
        )
        if self.cta_group is not None:
            name += f"_{self.cta_group}ctamma"
        if self.warps_m is not None or self.warps_n is not None:
            if self.warps_m is None or self.warps_n is None:
                raise ValueError("WARPS_M and WARPS_N must be given together")
            name += f"_warps{self.warps_m}x{self.warps_n}"
        if self.split_k_slices > 1:
            name += f"_splitK{self.split_k_slices}"
        if self.cluster_k_reduce:
            name += "_clusterKReduce"
        if self.swap_ab:
            name += "_swapAB"
        return name

    def to_config(self) -> TileConfig:
        """The TileConfig these knobs name."""
        return by_name(self.config_name)

    def to_public(self) -> dict:
        """``{cudnn.knob_type: int}``: every axis this record carries (bools as 0/1)."""
        kt = cudnn.knob_type
        out = {}
        for field, member, _required in self._PUBLIC_KNOBS:
            value = getattr(self, field)
            if value is None:
                continue
            if field == "split_k_slices" and value == 1:
                out[getattr(kt, member)] = 1  # spelled explicitly: 1 = one slice, a real value
            else:
                out[getattr(kt, member)] = int(value)
        return out

    @classmethod
    def from_public(cls, public: dict) -> "GemmKnobs":
        """Inverse of :meth:`to_public`. Rejects knob types that are not GEMM tile
        axes, non-int values, and records missing a required axis."""
        kt = cudnn.knob_type
        by_type = {getattr(kt, member): (field, required) for field, member, required in cls._PUBLIC_KNOBS}
        kwargs = {}
        for knob, value in public.items():
            knob = kt(int(knob)) if not isinstance(knob, kt) else knob
            entry = by_type.get(knob)
            if entry is None:
                raise ValueError(f"knob {knob.name} is not a tile axis of the FROST GEMM engine")
            field, _ = entry
            if isinstance(value, bool):
                value = int(value)
            if not isinstance(value, int):
                raise ValueError(f"knob {knob.name} value must be an int, got {value!r}")
            if field in ("swap_ab", "cluster_k_reduce"):
                if value not in (0, 1):
                    raise ValueError(f"knob {knob.name} must be 0 or 1, got {value}")
                kwargs[field] = bool(value)
            elif field == "split_k_slices" and value < 1:
                # 0 would spell the same name as one slice and replay as it
                raise ValueError(f"knob SPLIT_K_SLC must be >= 1, got {value}")
            else:
                kwargs[field] = value
        missing = [member for field, member, required in cls._PUBLIC_KNOBS if required and field not in kwargs]
        if missing:
            raise ValueError(f"FROST GEMM knob record is missing required axes: {', '.join(missing)}")
        return cls(**kwargs)
