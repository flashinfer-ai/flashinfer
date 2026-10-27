# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile config catalog — PURE GEOMETRY for the fused GEMM kernels.

A ``TileConfig`` describes ONLY dtype-independent tile geometry (cta tile,
MMA-inst tile, cluster shape, pipeline). It does NOT carry ``cta_group`` /
``ab_stages`` — those are execution strategy chosen by the
kernel template. K is stored in *bytes*, so one config covers every dtype.
Name: ``CONFIG_<pipeline>_<CTA_M>x<CTA_N>x<K_BYTES>_<MMA_M>x<MMA_N>x<MMA_K_BYTES>_cluster<cgrp_m>x<cgrp_n>[_swapAB]``.
A warp-scoped family (sm120) additionally ALWAYS names its compute-warp grid
via a ``_warpsMxN`` suffix — no grid is implied by an unsuffixed spelling.
See ``kernel_registry`` for the template registry and the support funnel.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, replace
from typing import ClassVar

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.occupancy import MAX_CLUSTER_SIZE


@functools.lru_cache(maxsize=None)
def _sm_smem_budget_bytes_of(device: int) -> int:
    """Largest per-CTA SMEM the device gives a CTA — the oversized carveout where the
    part has one, else the opt-in limit."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import device_name, is_available, oversized_shared_memory_per_block, shared_memory_per_block_optin

    if not is_available():
        raise NotImplementedError("cannot size the SMEM pipeline: no CUDA device is visible to query MaxSharedMemoryPerBlockOptin")
    optin = shared_memory_per_block_optin(device)
    if not optin:
        raise NotImplementedError(f"the driver did not report MaxSharedMemoryPerBlockOptin for device {device_name(device)!r}; cannot size the SMEM pipeline")
    return max(int(optin), oversized_shared_memory_per_block(device))


def _sm_smem_budget_bytes(device=None) -> int:
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import resolve_device

    return _sm_smem_budget_bytes_of(resolve_device(device))


def smem_ab_budget_bytes(smem_fixed_reserve: int, device=None) -> int:
    """SMEM left for the ab pipeline once the template's fixed reserve is off
    the top. That reserve -- the scheduler ring, every smem barrier, the TMEM
    base address and, on MoE, the per-CTA TMA tensormap scratch -- is the
    TEMPLATE's, so it is passed in rather than looked up from the geometry."""
    return _sm_smem_budget_bytes(device) - smem_fixed_reserve


@functools.lru_cache(maxsize=None)
def _l2_swizzle_budget_bytes_of(device: int) -> int:
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import device_name, is_available, l2_cache_bytes

    # A third of L2: the streaming operand and the C output take the rest.
    retention_divisor = 3
    if not is_available():
        raise NotImplementedError("cannot size the L2 tile-rasterization budget: no CUDA device is visible to query L2CacheSize")
    l2 = l2_cache_bytes(device)
    if not l2:
        raise NotImplementedError(f"the driver did not report L2CacheSize for device {device_name(device)!r}; cannot size the L2 tile-rasterization budget")
    return int(l2) // retention_divisor


def l2_swizzle_budget_bytes(device=None) -> int:
    """Operand bytes the N-super-block rasterization may assume stay resident in L2."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import resolve_device

    return _l2_swizzle_budget_bytes_of(resolve_device(device))


def _issuable_mma_tile_k_bytes(cls: "type[TileConfig]") -> tuple[int, ...]:
    """The K widths ``cls``'s MMA can issue. A hardware fact, so each family
    DECLARES it -- the base has no such attribute, and a family that forgot is
    an error rather than a silent inheritance of somebody else's width."""
    ok = getattr(cls, "MMA_TILE_K_BYTES", None)
    if not ok:
        raise NotImplementedError(f"MMA-inst K width not known for pipeline {cls.__name__} -- declare MMA_TILE_K_BYTES on its config class")
    return ok


def smem_ab_stages(per_stage_bytes: int, *, smem_fixed_reserve: int, extra_smem_bytes: int = 0, device=None) -> int:
    """How many ab-pipeline stages of ``per_stage_bytes`` fit, capped.

    The caller owns the stage LAYOUT -- a plain K-tile, a packed tile plus its
    scale factors, or one 128-byte K chunk -- so it passes the size in; this
    owns the budget and the ceiling. 0 when not even one stage fits, so each
    caller can complain in its own terms.
    """
    cap = 16  # nothing in the templates assumes it: mbar arrays, ab_iter % ab_stages and the tail drain are all parameterized
    avail = smem_ab_budget_bytes(smem_fixed_reserve, device) - extra_smem_bytes
    return min(avail // per_stage_bytes, cap) if avail >= per_stage_bytes else 0


@dataclass(frozen=True)
class TileConfig:
    """One pure-geometry tile config. Dtype- AND execution-independent.

    Four nested tiles, outermost first -- CGA (a cluster of CTAs), CTA, warp,
    MMA instruction -- plus the K-split and logical A/B-orientation axes. Each
    tile level must divide the one above it. K is stored in BYTES throughout,
    so one config serves every dtype.

    NOT here, because they are not choices: `ab_stages` (the device SMEM budget
    decides), the epilogue subtile N (the output dtype and drain width decide),
    the acc-stage count (the TMEM budget decides) and the L2 rasterization width
    (the kernel picks it per launch from the runtime shape). Anything the
    compiler can derive is derived at render time, not stored here.

    Family facts live on the family CLASS, not in a table keyed by pipeline
    name. ``cta_group`` is the clearest case: it does NOT exist here -- the
    families whose MMA spans a CTA pair declare it, and a family without the
    pair has no such attribute at all, so reading it is an AttributeError rather
    than a silent 1.
    """

    pipeline: str  # family; pairs with the template whose filename leads with it

    cta_tile_m: int
    cta_tile_n: int
    cta_tile_k_bytes: int

    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k_bytes: int

    mma_tile_m: int
    mma_tile_n: int
    mma_tile_k_bytes: int

    mma_size_m: int
    mma_size_n: int
    mma_size_k: int

    cga_size_m: int
    cga_size_n: int
    cga_size_k: int

    warps_per_cta: int

    split_k_slices: int

    swap_ab: bool

    def __post_init__(self) -> None:
        name = self.name

        if not isinstance(self.swap_ab, bool):
            raise ValueError(f"TileConfig {name!r}: swap_ab must be bool; got {self.swap_ab!r}")

        for label, v in (
            ("cta_tile_m", self.cta_tile_m),
            ("cta_tile_n", self.cta_tile_n),
            ("cta_tile_k_bytes", self.cta_tile_k_bytes),
            ("warp_tile_m", self.warp_tile_m),
            ("warp_tile_n", self.warp_tile_n),
            ("warp_tile_k_bytes", self.warp_tile_k_bytes),
            ("mma_tile_m", self.mma_tile_m),
            ("mma_tile_n", self.mma_tile_n),
            ("mma_tile_k_bytes", self.mma_tile_k_bytes),
            ("cga_size_m", self.cga_size_m),
            ("cga_size_n", self.cga_size_n),
            ("cga_size_k", self.cga_size_k),
            ("warps_per_cta", self.warps_per_cta),
            ("split_k_slices", self.split_k_slices),
        ):
            if v <= 0:
                raise NotImplementedError(f"TileConfig {name!r}: {label}={v} must be positive")

        # Each tile level divides the one above it.
        for label, outer, inner in (
            ("m", self.cta_tile_m, self.warp_tile_m),
            ("n", self.cta_tile_n, self.warp_tile_n),
            ("k_bytes", self.cta_tile_k_bytes, self.warp_tile_k_bytes),
        ):
            if outer % inner:
                raise NotImplementedError(f"TileConfig {name!r}: warp_tile_{label}={inner} must divide cta_tile_{label}={outer}")
        for label, outer, inner in (
            ("m", self.warp_tile_m, self.mma_tile_m),
            ("n", self.warp_tile_n, self.mma_tile_n),
            ("k_bytes", self.warp_tile_k_bytes, self.mma_tile_k_bytes),
        ):
            if outer % inner:
                raise NotImplementedError(f"TileConfig {name!r}: mma_tile_{label}={inner} must divide warp_tile_{label}={outer}")

        for axis, warp, mma in (
            ("m", self.warp_tile_m, self.mma_tile_m),
            ("n", self.warp_tile_n, self.mma_tile_n),
            ("k", self.warp_tile_k_bytes, self.mma_tile_k_bytes),
        ):
            want = warp // mma
            got = getattr(self, f"mma_size_{axis}")
            if got != want:
                raise NotImplementedError(f"TileConfig {name!r}: mma_size_{axis}={got} but the tiles say {want}")

        # K is never split across CTAs of a cluster -- an axis reserved for later.
        if self.cga_size_k != 1:
            raise NotImplementedError(f"TileConfig {name!r}: cga_size_k={self.cga_size_k} — cross-CTA K reduction is not implemented")

        cga = self.cga_size_m * self.cga_size_n * self.cga_size_k
        if cga > MAX_CLUSTER_SIZE:
            raise NotImplementedError(
                f"TileConfig {name!r}: cluster {self.cga_size_m}x{self.cga_size_n} is {cga} CTAs — the architecture CGA limit is {MAX_CLUSTER_SIZE}"
            )

    # -- derived -------------------------------------------------------------

    @property
    def threads_per_cta(self) -> int:
        # Imported here, not at module scope: the whole gemm.frost package is
        # importable without the [cutedsl] extra, and this is the only thing in
        # the geometry that would have pulled cutlass in.
        from cutlass.cute.arch import WARP_SIZE

        return self.warps_per_cta * WARP_SIZE

    @property
    def ctas_per_mma(self) -> int:
        """CTAs ONE MMA instruction spans. Everything a multi-CTA MMA changes
        follows from this number, so a family without one answers 1 truthfully
        rather than being asked for an axis it does not have."""
        return 1

    @property
    def is_cta_pair_mma(self) -> bool:
        return self.ctas_per_mma == 2

    @property
    def cta_smem_tile_n(self) -> int:
        """B's SMEM N per CTA -- an MMA spanning several CTAs splits it."""
        return self.cta_tile_n // self.ctas_per_mma

    @property
    def mma_tile_m_hw(self) -> int:
        """M the MMA INSTRUCTION covers, across however many CTAs it spans."""
        return self.mma_tile_m * self.ctas_per_mma

    @property
    def epi_tile_m(self) -> int:
        """Rows the epilogue drains per pass -- one MMA-M block."""
        return self.mma_tile_m

    @property
    def fallback_cga_size_mnk(self) -> tuple[int, int, int]:
        """The cluster a mixed-CGA launch falls back to on the SMs a wider one
        cannot fill: the smallest this geometry is legal in. An MMA that spans
        several CTAs must keep them inside one cluster, so it can never fall
        back below its own span."""
        return (self.ctas_per_mma, 1, 1)

    @property
    def geometry_name(self) -> str:
        """Canonical geometry token used in the TileConfig name."""
        return self._geometry_base + (f"_splitK{self.split_k_slices}" if self.split_k_slices > 1 else "") + ("_swapAB" if self.swap_ab else "")

    @property
    def _geometry_base(self) -> str:
        """``geometry_name`` without the split-K suffix; a family with extra axes extends it."""
        return (
            f"{self.cta_tile_m}x{self.cta_tile_n}x{self.cta_tile_k_bytes}"
            f"_{self.mma_tile_m}x{self.mma_tile_n}x{self.mma_tile_k_bytes}"
            f"_cluster{self.cga_size_m}x{self.cga_size_n}"
            # Named only where it is an AXIS -- a pipeline without the CTA pair
            # declares no such field, so there is nothing to spell.
            + (f"_{self.cta_group}ctamma" if isinstance(self, CtaPairTileConfig) else "")
        )

    @property
    def name(self) -> str:
        """Canonical identifier ``CONFIG_<pipeline>_<geometry_name>`` — pure
        geometry (``ab_stages`` is derived from the device budget, not named)."""
        return f"CONFIG_{self.pipeline}_{self.geometry_name}"

    @property
    def cta_tile_mn(self) -> tuple[int, int]:
        """Per-CTA logical output tile (M, N) in elements."""
        return (self.cta_tile_m, self.cta_tile_n)

    @property
    def cga_size_mn(self) -> tuple[int, int]:
        """CTAs per cluster along (M, N)."""
        return (self.cga_size_m, self.cga_size_n)

    @property
    def cga_size_mnk(self) -> tuple[int, int, int]:
        """cga_size as a cluster_shape-style triple."""
        return (self.cga_size_m, self.cga_size_n, 1)

    @property
    def cga_tile_mn(self) -> tuple[int, int]:
        """Cluster aggregate output tile (M, N) in elements."""
        return (self.cta_tile_m * self.cga_size_m, self.cta_tile_n * self.cga_size_n)

    @property
    def cluster_shape(self) -> tuple[int, int, int]:
        """Alias for cga_size_mnk (cluster-launch terminology)."""
        return self.cga_size_mnk

    # -- dtype-dependent shape views (require elem_bytes) -------------------

    def cta_tile_k(self, elem_bytes: int) -> int:
        """K tile in *elements*, given the input dtype's byte width."""
        if self.cta_tile_k_bytes % elem_bytes != 0:
            raise ValueError(f"TileConfig {self.name!r}: cta_tile_k_bytes " f"({self.cta_tile_k_bytes}) is not divisible by " f"elem_bytes={elem_bytes}")
        return self.cta_tile_k_bytes // elem_bytes

    def cta_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Per-CTA logical tile in elements (M, N, K)."""
        return (self.cta_tile_m, self.cta_tile_n, self.cta_tile_k(elem_bytes))

    def cga_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Cluster aggregate tile in elements (M, N, K)."""
        m, n = self.cga_tile_mn
        return (m, n, self.cta_tile_k(elem_bytes))

    def cta_smem_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Per-CTA SMEM tile in elements."""
        return (self.cta_tile_m, self.cta_smem_tile_n, self.cta_tile_k(elem_bytes))

    def mma_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Hardware MMA-instruction shape in elements; K is ``mma_tile_k_bytes``."""
        return (self.mma_tile_m_hw, self.mma_tile_n, self.mma_tile_k_bytes // elem_bytes)

    def max_ab_stages(self, *, smem_fixed_reserve: int, extra_smem_bytes: int = 0, extra_per_stage_bytes: int = 0) -> int:
        """Largest ab-pipeline depth this geometry fits, once the TEMPLATE's
        fixed reserve is off the top (which is why the caller supplies it)."""
        per_stage = (self.cta_tile_m + self.cta_smem_tile_n) * self.cta_tile_k_bytes + extra_per_stage_bytes
        stages = smem_ab_stages(per_stage, smem_fixed_reserve=smem_fixed_reserve, extra_smem_bytes=extra_smem_bytes)
        if stages == 0:
            raise ValueError(
                f"TileConfig {self.name!r}: per-stage SMEM {per_stage} bytes exceeds the budget "
                f"(reserve={smem_fixed_reserve}, extra_smem_bytes={extra_smem_bytes}) — cannot fit even 1 stage"
            )
        return stages

    # -- multicast model -----------------------------------------------------

    @property
    def multicast_a_factor(self) -> int:
        """# CTAs sharing the same M slice of A. Independent of cta_group."""
        return self.cga_size_n

    @property
    def multicast_b_factor(self) -> int:
        """# CTAs sharing the same N slice of B. The CTAs an MMA spans are
        consumed by the instruction, not by B-multicast."""
        return self.cga_size_m // self.ctas_per_mma

    @property
    def multicast_a(self) -> bool:
        return self.multicast_a_factor > 1

    @property
    def multicast_b(self) -> bool:
        return self.multicast_b_factor > 1


# ---------------------------------------------------------------------------
# Per-pipeline config families. A kernel template pairs with the config class
# matching the ``sm<NNN>`` token in its filename (see kernel_registry). Each
# family may fix axes that its pipeline's MMA instruction dictates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CtaPairTileConfig(TileConfig):
    """Base for the tcgen05 families, whose MMA may span a PAIR of CTAs.

    Their tensor core is CTA-scoped, so a warp tile IS the CTA tile -- the axis
    exists in the base for families that really have one (sm120), and here it is
    pinned rather than validated. ``cta_group`` is a real geometry axis, and
    everything the pair changes is an override so the base never needs it: the
    pair splits B's N across the two CTAs, one instruction covers twice the M,
    the leading factor of 2 in ``cga_size_m`` is consumed by the pair rather
    than by B-multicast, and the pair must stay inside one cluster."""

    # The tcgen05 instruction's own M/N bounds: M is the two TMEM row layouts
    # (128 full lanes / 64 packed half-lanes), N is the idesc n_dim encoded with
    # its 3 LSBs dropped. These were never the CTA tile's -- they read that way
    # only while one instruction covered the whole tile.
    _MMA_TILE_M_VALUES: ClassVar[tuple[int, ...]] = (64, 128)
    _MMA_TILE_N_MAX: ClassVar[int] = 256
    # MMA instructions one warp tile may span along M. K is not an axis (the
    # mainloop already walks cta_tile_k / mma_tile_k) and N is deliberately not
    # one at all: splitting it measured within noise of a single instruction,
    # cuBLAS does not do it either, and neither the CTA pair nor the block-scale
    # pipeline can express it.
    MMA_SIZE_M_MAX: ClassVar[int] = 4

    cta_group: int

    def __post_init__(self) -> None:
        # Their tensor core is CTA-scoped, so a warp tile IS the CTA tile: pin
        # it rather than ask the caller to spell an axis they do not have.
        object.__setattr__(self, "warp_tile_m", self.cta_tile_m)
        object.__setattr__(self, "warp_tile_n", self.cta_tile_n)
        object.__setattr__(self, "warp_tile_k_bytes", self.cta_tile_k_bytes)
        if self.cta_group not in (1, 2):
            raise NotImplementedError(f"TileConfig {self.name!r}: cta_group must be 1 or 2, got {self.cta_group}")
        # CTA pairs occupy adjacent M positions within a cluster.
        if self.cta_group == 2 and self.cga_size_m % 2:
            raise NotImplementedError(f"TileConfig {self.name!r}: 2-CTA MMA needs cga_size_m % 2 == 0; got {self.cga_size_m}")
        # The M/N hardware bounds of the tcgen05 instruction: M is the two TMEM
        # row layouts (128 full lanes / 64 packed half-lanes); N is the idesc
        # n_dim, encoded with its 3 LSBs dropped. Checked BEFORE the generic
        # instruction-count rule so an illegal M is named as such, not as a count.
        if self.mma_tile_m not in self._MMA_TILE_M_VALUES:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_tile_m={self.mma_tile_m} — supported values are {list(self._MMA_TILE_M_VALUES)}")
        if self.mma_tile_n < 8 or self.mma_tile_n > self._MMA_TILE_N_MAX or self.mma_tile_n % 8:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: mma_tile_n={self.mma_tile_n} — supported range is 8 ≤ N ≤ {self._MMA_TILE_N_MAX}, multiple of 8"
            )
        # One tcgen05 instruction covers the tile's whole N; only M is an
        # instruction-count axis here.
        if self.mma_tile_n != self.warp_tile_n:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: N is not split across MMA instructions — mma_tile_n must equal warp_tile_n={self.warp_tile_n}; got {self.mma_tile_n}"
            )
        issuable = _issuable_mma_tile_k_bytes(type(self))
        if self.mma_tile_k_bytes not in issuable:
            raise NotImplementedError(f"TileConfig {self.name!r}: {self.pipeline} issues mma_tile_k_bytes {sorted(issuable)}; got {self.mma_tile_k_bytes}")
        if self.mma_size_m > self.MMA_SIZE_M_MAX:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: {self.pipeline} spans at most {self.MMA_SIZE_M_MAX} MMA instruction(s) along M per warp tile; got mma_size_m={self.mma_size_m}"
            )
        super().__post_init__()

    @property
    def ctas_per_mma(self) -> int:
        return self.cta_group


@dataclass(frozen=True)
class ConfigSm100(CtaPairTileConfig):
    """sm100 geometry — every axis is free, including ``mma_tile_k_bytes``
    ∈ {32, 64} and the 2-CTA MMA pair. The 64-byte dense FP8 / block-scale MMA
    depends on the active GPU and operand types, checked by the registry's
    config gates rather than the geometry family."""

    MMA_TILE_K_BYTES: ClassVar[tuple[int, ...]] = (32, 64)
    # The widest SMEM row swizzle.
    CTA_TILE_K_BYTES_MAX: ClassVar[int] = 128

    def __post_init__(self) -> None:
        if self.cta_tile_k_bytes > self.CTA_TILE_K_BYTES_MAX:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: cta_tile_k_bytes={self.cta_tile_k_bytes} — at most {self.CTA_TILE_K_BYTES_MAX} for pipeline {self.pipeline}"
            )
        super().__post_init__()


# 4 epilogue + MMA + TMA + CLC scheduler + a donor. The mainloop variant of the
# same geometry needs 4 more; that is the TEMPLATE's override, not a config.
_TCGEN05_WARPS_PER_CTA = 8

# The fp4 K=48B UTCOMMA fixes both: the K-tile is lcm(128, 48).
_SM103_CTA_TILE_K_BYTES = 384
_SM103_MMA_TILE_K_BYTES = 48


@dataclass(frozen=True)
class ConfigSm103(CtaPairTileConfig):
    """sm103 geometry — ``cta_tile_k_bytes`` (384) and ``mma_tile_k_bytes``
    (48) are fixed by the fp4 K=48B UTCOMMA (K-tile = lcm(128, 48)); the other
    axes match :class:`ConfigSm100`."""

    MMA_TILE_K_BYTES: ClassVar[tuple[int, ...]] = (_SM103_MMA_TILE_K_BYTES,)
    # The chunk pipeline miscomputes at a split M tile (A reads unwritten SMEM
    # in K) and its ab_stages budget under-counts; both are silent-wrong.
    MMA_SIZE_M_MAX: ClassVar[int] = 1
    CTA_TILE_K_BYTES_FIXED: ClassVar[int] = _SM103_CTA_TILE_K_BYTES

    def __post_init__(self) -> None:
        if self.cta_tile_k_bytes != self.CTA_TILE_K_BYTES_FIXED:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: {self.pipeline} fixes cta_tile_k_bytes={self.CTA_TILE_K_BYTES_FIXED}; got {self.cta_tile_k_bytes}"
            )
        super().__post_init__()


@dataclass(frozen=True)
class ConfigSm120(TileConfig):
    """sm120 geometry — warp-scoped MMA on consumer Blackwell (CC 12.x).

    ``mma.sync`` is m16n8; the template issues two of them back to back along N
    and treats the pair as ONE 16x16 instruction, which is what ``mma_tile_n``
    counts. There is no CTA pair (CC 12.x has no 2-CTA MMA), and the cluster is
    1x1 by design: CC 12.x does launch thread-block clusters (2-16 CTAs), but
    its TMA multicast is not usable (emulated, ~3e4x slower), so a wider
    cluster buys these kernels nothing. The 12-warp block is 8 compute + TMA +
    CLC scheduler + 2 donors."""

    MMA_TILE_K_BYTES: ClassVar[tuple[int, ...]] = (32,)
    # Family facts live HERE (one home, per the TileConfig docstring),
    # consulted by `_geom_sm120`, `__post_init__` and `as_pipeline`. The warp
    # grid has NO default: every sm120 name spells it (`_warpsMxN`); the
    # CROSSING grid below is only what `as_pipeline` builds when a config
    # crosses in from a family that has no warp-grid axis at all.
    FIXED_MMA_TILE_MN: ClassVar[tuple[int, int]] = (16, 16)
    FIXED_CGA_MN: ClassVar[tuple[int, int]] = (1, 1)
    CROSSING_WARP_GRID_MN: ClassVar[tuple[int, int]] = (4, 2)
    DEFAULT_WARPS_PER_CTA: ClassVar[int] = 12  # 8 compute + TMA + CLC scheduler + 2 donors

    def __post_init__(self) -> None:
        # VALIDATED, not pinned: the geometry axes (CTA tile, warp grid, K
        # width) are real choices here; only the facts of the family are
        # enforced -- the 16x16x32B warp-MMA pair, the 1x1 cluster (see the
        # class docstring), no warp-level K split, and a block wide enough for
        # the compute grid + TMA producer + CLC scheduler.
        super().__post_init__()
        name = self.name
        mma = (self.mma_tile_m, self.mma_tile_n, self.mma_tile_k_bytes)
        fixed_mma = (*self.FIXED_MMA_TILE_MN, self.MMA_TILE_K_BYTES[0])
        if mma != fixed_mma:
            raise NotImplementedError(f"ConfigSm120 {name!r}: the warp MMA pair is fixed at " f"{fixed_mma}; got mma_tile={mma}")
        if (self.cga_size_m, self.cga_size_n) != (1, 1):
            raise NotImplementedError(
                f"ConfigSm120 {name!r}: the sm120 family uses no CGA; cluster must be 1x1 (CC 12.x launches "
                f"clusters, but its TMA multicast is not usable), got {self.cga_size_m}x{self.cga_size_n}"
            )
        if self.warp_tile_k_bytes != self.cta_tile_k_bytes:
            raise NotImplementedError(
                f"ConfigSm120 {name!r}: no warp-level K split; warp_tile_k_bytes="
                f"{self.warp_tile_k_bytes} must equal cta_tile_k_bytes={self.cta_tile_k_bytes}"
            )
        compute_warps = (self.cta_tile_m // self.warp_tile_m) * (self.cta_tile_n // self.warp_tile_n)
        if compute_warps + 2 > self.warps_per_cta:
            raise NotImplementedError(
                f"ConfigSm120 {name!r}: {compute_warps} compute warps + the TMA producer " f"+ the CLC scheduler exceed warps_per_cta={self.warps_per_cta}"
            )

    @property
    def _geometry_base(self) -> str:
        """The warp grid is an AXIS here (unlike the CTA-scoped families), so it
        is ALWAYS named via a ``_warpsMxN`` suffix -- no grid is implied by an
        unsuffixed spelling."""
        base = super()._geometry_base
        if self.warp_tile_m and self.warp_tile_n:
            base += f"_warps{self.cta_tile_m // self.warp_tile_m}x{self.cta_tile_n // self.warp_tile_n}"
        return base

    @property
    def epi_tile_m(self) -> int:
        """The accumulators are already in registers -- the epilogue drains the
        whole CTA tile in one pass, not an MMA-M block at a time."""
        return self.cta_tile_m


_CONFIG_CLASS_BY_PIPELINE: dict[str, type[TileConfig]] = {
    "sm100": ConfigSm100,
    "sm103": ConfigSm103,
    "sm120": ConfigSm120,
}


def config_class_for_pipeline(pipeline: str) -> type[TileConfig]:
    """The config family a template of pipeline-family ``pipeline`` pairs with."""
    try:
        return _CONFIG_CLASS_BY_PIPELINE[pipeline]
    except KeyError:
        raise KeyError(f"no config family for pipeline {pipeline!r}; known: " f"{sorted(_CONFIG_CLASS_BY_PIPELINE)}") from None


# Catalog geometry axes; the registry filters graph/template support.
# CTA M = mma_tile_m * mma_size_m; cluster M * N <= MAX_CLUSTER_SIZE.
# Enumerate mma_size_m=1 first to preserve first-match config lookups.
_MMA_TILE_M_VALUES: tuple[int, ...] = (128, 64)
_MMA_SIZE_M_VALUES: tuple[int, ...] = (1, 2, 4)
_M_AXES: tuple[tuple[int, int], ...] = tuple((mma_tile_m, mma_size_m) for mma_size_m in _MMA_SIZE_M_VALUES for mma_tile_m in _MMA_TILE_M_VALUES)

_CGA_SIZE_M_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16)
_CGA_SIZE_N_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16)
_CLUSTERS: tuple[tuple[int, int], ...] = tuple(
    (cga_m, cga_n) for cga_m in _CGA_SIZE_M_VALUES for cga_n in _CGA_SIZE_N_VALUES if cga_m * cga_n <= MAX_CLUSTER_SIZE
)


def _geom_sm100(
    cta_tile_m: int,
    cta_tile_n: int,
    cta_tile_k_bytes: int,
    mma_tile_m: int,
    mma_tile_n: int,
    mma_tile_k_bytes: int,
    cga_size_m: int,
    cga_size_n: int,
    cta_group: int,
) -> ConfigSm100:
    """One sm100 geometry. The warp tile is the CTA tile (a CTA-scoped MMA), so
    the family pins it and it is not spelled here."""
    return ConfigSm100(
        pipeline="sm100",
        cta_tile_m=cta_tile_m,
        cta_tile_n=cta_tile_n,
        cta_tile_k_bytes=cta_tile_k_bytes,
        warp_tile_m=cta_tile_m,
        warp_tile_n=cta_tile_n,
        warp_tile_k_bytes=cta_tile_k_bytes,
        mma_tile_m=mma_tile_m,
        mma_tile_n=mma_tile_n,
        mma_tile_k_bytes=mma_tile_k_bytes,
        mma_size_m=cta_tile_m // mma_tile_m,
        mma_size_n=cta_tile_n // mma_tile_n,
        mma_size_k=cta_tile_k_bytes // mma_tile_k_bytes,
        cga_size_m=cga_size_m,
        cga_size_n=cga_size_n,
        cga_size_k=1,
        warps_per_cta=_TCGEN05_WARPS_PER_CTA,
        split_k_slices=1,
        swap_ab=False,
        cta_group=cta_group,
    )


def _geom_sm103(cta_tile_m: int, cta_tile_n: int, cga_size_m: int, cga_size_n: int, cta_group: int) -> ConfigSm103:
    """One sm103 geometry. The K axes are the FAMILY's (the fp4 K=48B UTCOMMA
    fixes both), and one instruction covers the whole tile."""
    return ConfigSm103(
        pipeline="sm103",
        cta_tile_m=cta_tile_m,
        cta_tile_n=cta_tile_n,
        cta_tile_k_bytes=_SM103_CTA_TILE_K_BYTES,
        warp_tile_m=cta_tile_m,
        warp_tile_n=cta_tile_n,
        warp_tile_k_bytes=_SM103_CTA_TILE_K_BYTES,
        mma_tile_m=cta_tile_m,
        mma_tile_n=cta_tile_n,
        mma_tile_k_bytes=_SM103_MMA_TILE_K_BYTES,
        mma_size_m=1,
        mma_size_n=1,
        mma_size_k=_SM103_CTA_TILE_K_BYTES // _SM103_MMA_TILE_K_BYTES,
        cga_size_m=cga_size_m,
        cga_size_n=cga_size_n,
        cga_size_k=1,
        warps_per_cta=_TCGEN05_WARPS_PER_CTA,
        split_k_slices=1,
        swap_ab=False,
        cta_group=cta_group,
    )


def _geom_sm120(
    cta_tile_m: int,
    cta_tile_n: int,
    cta_tile_k_bytes: int,
    warps_size_m: int,
    warps_size_n: int,
    cga_size_m: int,
    cga_size_n: int,
) -> ConfigSm120:
    """One sm120 geometry. The CTA tile is split over a ``warps_size_m`` x
    ``warps_size_n`` grid of compute warps; the MMA tile is the family's."""
    mma_tile_m, mma_tile_n = ConfigSm120.FIXED_MMA_TILE_MN
    mma_tile_k_bytes = ConfigSm120.MMA_TILE_K_BYTES[0]
    return ConfigSm120(
        pipeline="sm120",
        cta_tile_m=cta_tile_m,
        cta_tile_n=cta_tile_n,
        cta_tile_k_bytes=cta_tile_k_bytes,
        warp_tile_m=cta_tile_m // warps_size_m,
        warp_tile_n=cta_tile_n // warps_size_n,
        warp_tile_k_bytes=cta_tile_k_bytes,
        mma_tile_m=mma_tile_m,
        mma_tile_n=mma_tile_n,
        mma_tile_k_bytes=mma_tile_k_bytes,
        mma_size_m=cta_tile_m // warps_size_m // mma_tile_m,
        mma_size_n=cta_tile_n // warps_size_n // mma_tile_n,
        mma_size_k=cta_tile_k_bytes // mma_tile_k_bytes,
        cga_size_m=cga_size_m,
        cga_size_n=cga_size_n,
        cga_size_k=1,
        warps_per_cta=ConfigSm120.DEFAULT_WARPS_PER_CTA,
        split_k_slices=1,
        swap_ab=False,
    )


_SM120_SMEM_CAP_FALLBACK_BYTES = 99 * 1024


def _sm120_smem_cap_bytes() -> int:
    try:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import compute_capability, is_available, resolve_device

        if not is_available():
            return _SM120_SMEM_CAP_FALLBACK_BYTES
        dev = resolve_device(None)
        if compute_capability(dev)[0] != 12:
            return _SM120_SMEM_CAP_FALLBACK_BYTES
        return _sm_smem_budget_bytes_of(dev)
    except Exception:
        return _SM120_SMEM_CAP_FALLBACK_BYTES


_SM120_SMEM_CAP_BYTES = _sm120_smem_cap_bytes()
_SM120_SMEM_FIXED_RESERVE = 2048  # kernel_registry's default smem_fixed_reserve
_SM120_STG_STAGE_ELEMS = 528  # f32 staging elems per compute warp (template _STG_EPI_WARP_ELEMS)


def _sm120_smem_feasible(cta_m: int, cta_n: int, kb: int, num_compute_warps: int) -> bool:
    per_stage = (cta_m + cta_n) * kb + 16
    staging = 4 * _SM120_STG_STAGE_ELEMS * num_compute_warps
    stages = min((_SM120_SMEM_CAP_BYTES - _SM120_SMEM_FIXED_RESERVE) // per_stage, 16)
    return stages - -(-staging // per_stage) >= 1


def _build_catalog() -> tuple[TileConfig, ...]:
    cfgs: list[TileConfig] = []
    for mma_m, mma_size_m in _M_AXES:
        for cta_n in range(256, 0, -8):
            for k_bytes in (128, 64):
                for mma_k_bytes in (32, 64):
                    if k_bytes % mma_k_bytes:
                        continue
                    for cga_m, cga_n in _CLUSTERS:
                        for cta_group in (1, 2):
                            if cga_m % cta_group:
                                continue
                            cfgs.append(_geom_sm100(mma_m * mma_size_m, cta_n, k_bytes, mma_m, cta_n, mma_k_bytes, cga_m, cga_n, cta_group))
    # sm103 block-scale geometries: M pinned to 128 by the 1-CTA MMA atom;
    # clusters share the sm100 enumeration (the templates use the same generic
    # rank-decomposition multicast masks for data + SF).
    for cta_n in (256, 128):
        for cga_m, cga_n in _CLUSTERS:
            for cta_group in (1, 2):
                if cga_m % cta_group:
                    continue
                cfgs.append(_geom_sm103(128, cta_n, cga_m, cga_n, cta_group))

    # sm120: sweep the warp-MMA geometries over CTA tile x K width x warp grid.
    # Only combos the template can render are cataloged: the warp grid must cut
    # the CTA tile into whole 16x16 warp-MMA pairs, and SMEM must fit at least
    # one AB stage on top of the transposed-STG staging.
    _SM120_WARP_SIZE_CONFIGS: tuple[tuple[int, int], ...] = (
        (2, 4),
        (4, 2),
        (1, 8),
        (8, 1),
    )
    for cta_tile_m in (512, 256, 128, 64, 32):
        for cta_tile_n in range(512, 0, -16):
            for cta_tile_k_bytes in (128, 64, 32):
                for warps_size_m, warps_size_n in _SM120_WARP_SIZE_CONFIGS:
                    if cta_tile_m % (warps_size_m * 16) or cta_tile_n % (warps_size_n * 16):
                        continue
                    if not _sm120_smem_feasible(cta_tile_m, cta_tile_n, cta_tile_k_bytes, warps_size_m * warps_size_n):
                        continue
                    cfgs.append(_geom_sm120(cta_tile_m, cta_tile_n, cta_tile_k_bytes, warps_size_m, warps_size_n, 1, 1))
    return tuple(cfgs)


CATALOG: tuple[TileConfig, ...] = _build_catalog()


# Expose each catalog entry as a module-level variable matching its canonical
# name — preserves `from .tile_config import CONFIG_sm100_...`.
for _cfg in CATALOG:
    globals()[_cfg.name] = _cfg
del _cfg


_CATALOG_BY_NAME: dict[str, TileConfig] = {c.name: c for c in CATALOG}

_CONFIG_NAME_RE = re.compile(
    r"^CONFIG_(?P<pipeline>sm\d+)_"
    r"(?P<cta_m>\d+)x(?P<cta_n>\d+)x(?P<k_bytes>\d+)_"
    r"(?P<mma_m>\d+)x(?P<mma_n>\d+)x(?P<mma_k_bytes>\d+)_"
    r"cluster(?P<cga_m>\d+)x(?P<cga_n>\d+)(?:_(?P<cta_group>\d+)ctamma)?"
    r"(?:_warps(?P<warps_m>\d+)x(?P<warps_n>\d+))?(?:_splitK(?P<split_k>\d+))?(?P<swap_ab>_swapAB)?$"
)


def _synthesize_config(name: str) -> TileConfig:
    """Build a TileConfig from a well-formed canonical name that isn't in the
    catalog (e.g. an MMA-inst tile smaller than the CTA tile). Geometry bounds
    are still enforced by ``TileConfig.__post_init__``."""
    m = _CONFIG_NAME_RE.match(name)
    if m is None:
        raise KeyError(f"unknown tile config {name!r} (not in the catalog and not a " f"canonical CONFIG_<pipeline>_MxNxKB_MxNxKB_clusterMxN name)")
    pipeline = m.group("pipeline")
    cls = config_class_for_pipeline(pipeline)  # KeyError for unknown pipelines
    cta_m = int(m.group("cta_m"))
    mma_m = int(m.group("mma_m"))
    # The warp tile: the CTA tile split over the `_warpsMxN` token when the
    # name spells one (the warp-scoped families' axis); without a token the
    # grid is 1x1, i.e. the CTA tile itself. Non-canonical spellings -- an
    # sm120 name missing its token, or a token on a CTA-scoped family -- die
    # in the round-trip check below either way.
    warp_m = cta_m // int(m.group("warps_m") or 1)
    warp_n = int(m.group("cta_n")) // int(m.group("warps_n") or 1)
    cfg = cls(
        pipeline=pipeline,
        cta_tile_m=cta_m,
        cta_tile_n=int(m.group("cta_n")),
        cta_tile_k_bytes=int(m.group("k_bytes")),
        warp_tile_m=warp_m,
        warp_tile_n=warp_n,
        warp_tile_k_bytes=int(m.group("k_bytes")),
        mma_tile_m=mma_m,
        mma_tile_n=int(m.group("mma_n")),
        mma_tile_k_bytes=int(m.group("mma_k_bytes")),
        mma_size_m=warp_m // mma_m,
        mma_size_n=warp_n // int(m.group("mma_n")),
        mma_size_k=int(m.group("k_bytes")) // int(m.group("mma_k_bytes")),
        cga_size_m=int(m.group("cga_m")),
        cga_size_n=int(m.group("cga_n")),
        cga_size_k=1,
        # Not spelled by the name; a family with its own block size declares it.
        warps_per_cta=getattr(cls, "DEFAULT_WARPS_PER_CTA", _TCGEN05_WARPS_PER_CTA),
        split_k_slices=int(m.group("split_k") or 1),
        swap_ab=bool(m.group("swap_ab")),
        # Only where the family HAS the axis; a name for one that does not
        # carries no such token either.
        **({"cta_group": int(m.group("cta_group") or 1)} if "cta_group" in cls.__dataclass_fields__ else {}),
    )
    # Leading zeros etc. must still round-trip; a name that merely omitted the
    # cta_group suffix is accepted as the 1-CTA spelling.
    if cfg.name != name and cfg.name != f"{name}_{getattr(cfg, 'cta_group', 1)}ctamma":
        raise KeyError(f"tile config name {name!r} is not canonical (round-trips to {cfg.name!r})")
    return cfg


def by_name(name: str) -> TileConfig:
    cfg = _CATALOG_BY_NAME.get(name) or _CATALOG_BY_NAME.get(f"{name}_1ctamma")
    if cfg is not None:
        return cfg
    return _synthesize_config(name)


DEFAULT_CONFIG: TileConfig = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma")


def _floor_pow2(v: int) -> int:
    """Largest power of two <= v (v >= 1)."""
    p = 1
    while p * 2 <= v:
        p *= 2
    return p


_DEFAULT_SM_COUNT = 148


def _sm_count() -> int:
    """SM count of the active device, with a B200-shaped fallback for CPU-only use.

    Queried through frost's device layer (not torch) so it honours a
    ``build_device()`` scope — i.e. follows the handle's GPU during a build, like
    every other device-derived constant."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import is_available, multiprocessor_count, resolve_device

    try:
        if is_available():
            return multiprocessor_count(resolve_device(None))
    except Exception:
        pass
    return _DEFAULT_SM_COUNT


def _mma_inst_k64(sm_count: int) -> bool:
    """Whether the TARGET silicon issues the 64-byte-K dense-FP8/block-scale MMA
    (``kernel_registry.MMA_INST_K64_ARCH_RANGES``). When a device is active the
    answer is the arch itself, so the gate is exactly the parts that have the
    instruction and nothing else. Only when no device is present (the sweep's
    CPU-only replay, which must stay a pure function of the arguments the
    caller supplies) does it fall back to an ``sm_count`` proxy: every k64 part
    measured so far has >= 190 SMs, every part without it has <= 188."""
    from . import compiler as C
    from .kernel_registry import MMA_INST_K64_ARCH_RANGES

    arch = C._current_arch()
    if arch is not None:
        return any(lo <= arch < hi for lo, hi in MMA_INST_K64_ARCH_RANGES)
    return sm_count >= 190


# Cluster shapes worth considering. 2-D shapes are included but, with the operand-reuse
# term switched off (see _cluster_score), the scorer has so far never selected one.
_CLUSTERS_1D = ((1, 1), (1, 2), (1, 4), (2, 1), (4, 1), (8, 1))
_CLUSTERS_2D = ((2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (8, 2))


def _hang_prone(cta_m: int, cta_n: int, cgrp_m: int, cgrp_n: int) -> bool:
    """64x256 with both cluster dims >= 2 can deadlock; never emit it."""
    return (cta_m, cta_n) == (64, 256) and cgrp_m >= 2 and cgrp_n >= 2


def _tile_score(rep_m: int, N: int, K: int, cta_m: int, cta_n: int, sm_count: int) -> float:
    """Wave-quantisation tile score (higher is better).

    Geometry only: ceil-divisions plus one sqrt over (M, N, K, tile, SM count). No
    measured timings and no fitted coefficients. Inspired by the analytic tile scorer
    in flashinfer PR #2940, with its aspect-ratio bias removed -- that term penalised
    exactly the wide 128x256 tiles this workload wants.
    """
    small_k = 1.0 if K is None else (0.50 if K <= 1024 else 0.80 if K <= 2048 else 1.0)
    max_tile_area = 256 * 256
    n_tiles = -(-N // cta_n)
    n_eff = N / (n_tiles * cta_n)
    score_n = n_eff * ((cta_m * cta_n) / max_tile_area) ** 0.5
    if cta_m * cta_n > 128 * 128:
        score_n *= small_k
    m_tiles = -(-rep_m // cta_m)
    total_ctas = m_tiles * n_tiles
    waves = -(-total_ctas // sm_count)
    return rep_m * total_ctas * score_n / (cta_m * waves * sm_count)


def _cluster_score(M: int, N: int, cta_m: int, cta_n: int, cta_group: int, cgrp_m: int, cgrp_n: int, sm_count: int) -> float:
    """Grid quantisation x last-wave occupancy (higher is better)."""
    eff_m = cta_m * cta_group
    m_tiles = -(-M // eff_m)
    n_tiles = -(-N // cta_n)
    launched = (-(-m_tiles // cgrp_m)) * cgrp_m * (-(-n_tiles // cgrp_n)) * cgrp_n
    if launched <= 0:
        return -1.0
    quant = (m_tiles * n_tiles) / launched
    waves = -(-launched // sm_count)
    return quant * (launched / (waves * sm_count))


def select_config(
    M: int,
    N: int,
    num_gemms: int,
    *,
    K: int | None = None,
    block_scale: bool = False,
    b_n_major: bool = False,
    b_elem_bytes: int = 2,
    sm_count: int | None = None,
    force_cta_group: int | None = None,
    m_is_group_average: bool = False,
    _pipeline: str | None = None,
    _batch: int = 1,
) -> TileConfig:
    """Pick a TileConfig from problem geometry.

    One selection path for every graph type frost supports. The tile and the cluster
    are each chosen by an analytic score over (M, N, K, tile geometry,
    SM count) -- integer ceil-divisions plus one sqrt, no measured timings and no fitted
    coefficients. The support constraints that used to sit around the old N-bucket rule
    (multi-GEMM N-tile budget, block-scale 128-multiples, N-major swizzle groups) are
    applied to the scored result, so they hold exactly as before.

    ``K`` is optional only for callers that do not have it to hand; without it the
    small-K bias is neutral and everything else is unchanged. The pick is an sm100
    geometry; a caller building for another template family passes it through
    :func:`as_pipeline` — except sm120, which has selection axes of its own (warp
    grid, CTA K-tile width): a caller that knows it is building for that family
    passes ``_pipeline="sm120"`` (and ``_batch``) so they are chosen HERE, where
    M/N/K and the SM count are in hand, instead of being pinned by the crossing. ``force_cta_group`` overrides only the 1-CTA/2-CTA
    strategy heuristic; tile and cluster geometry are still selected and scored
    normally for that final strategy.
    """
    sm = sm_count if sm_count is not None else _sm_count()
    x = max(1, num_gemms)
    # Multi-GEMM shares the 256-wide N budget across the parallel GEMMs.
    cta_n_max = max(32, min(256, _floor_pow2(256 // x)))

    if force_cta_group not in (None, 1, 2) or isinstance(force_cta_group, bool):
        raise ValueError(f"force_cta_group must be None, 1, or 2; got {force_cta_group!r}")

    # --- tile ---------------------------------------------------------------
    rep_m = min(max(1, M), 4096)
    # M-starved problems (few 128-tall tile rows) are occupancy-bound, and the sweep
    # shows 64-tall tiles winning them well past one tile row, so the 64-tall family
    # joins the scan up to M = 512 — 128 first, so a scoring tie keeps the taller
    # tile (fewer CTAs, same waves). The intermediate N widths are likewise measured
    # wins, and both extensions are restricted to the plain single-GEMM row-major-B
    # path: block-scale needs cta_n % 128 == 0 (SF swizzle), multi-GEMM squeezes the
    # shared budget to powers of two, and N-major B re-rounds to swizzle groups.
    plain = x == 1 and not block_scale and not b_n_major
    cta_m_choices = (128, 64) if (plain and M <= 512) else (128,)
    n_axis = (32, 64, 96, 128, 160, 192, 224, 256) if plain else (32, 64, 128, 256)
    tiles = []
    for tm in cta_m_choices:
        choices = [c for c in n_axis if c <= cta_n_max]
        if block_scale or (N >= 512 and M > 128):
            # Widening beat every narrower tile family across the measured range —
            # where M supplies more than one 128-tall tile row. M-starved problems
            # keep the full tile list rather than extrapolating.
            wide = [c for c in choices if c >= 128]
            if wide:
                choices = wide
        tiles += [(tm, c) for c in choices]
    cta_m, cta_n = max(tiles, key=lambda t: _tile_score(rep_m, N, K, t[0], t[1], sm))
    # On k64 silicon the 256-tall CTA pair (mma_tile_m 128, one pair per 512 M
    # rows) wins the machine-filling dense layers in the sweep; the wave scorer
    # cannot see that (its rep_m/cta_m prior always favors the shorter tile), so
    # upgrade the picked geometry when the taller grid still fills the machine.
    if plain and not m_is_group_average and b_elem_bytes == 1 and _mma_inst_k64(sm) and cta_m == 128:
        pairs = -(-M // 512)
        # Fill the machine and keep the pair-row padding small: a ragged M
        # (e.g. 1031 -> 1536 padded rows) hands the taller tile a third of its
        # work back as waste, and the sweep shows those layers regressing.
        if 2 * pairs * (-(-N // cta_n)) >= sm and pairs * 512 <= M + M // 16:
            cta_m = 256

    # 2-CTA needs a second M-tile to be worth it. Multi-GEMM is only implemented by the
    # 1ctamma template (see compiler._check_multi_gemm), so it stays at 1.
    if x > 1:
        if force_cta_group == 2:
            raise NotImplementedError("multi-GEMM is 1ctamma-only; cannot force cta_group=2")
        cta_group = 1
    elif force_cta_group is not None:
        cta_group = force_cta_group
    else:
        # Block-scale prefers the CTA pair even without a second M tile (halved per-CTA
        # B footprint buys pipeline depth) — but only while the pair's padded M row still
        # fits the machine in one wave; past that the padding costs a whole extra wave.
        # NOT when M is a per-group average (MoE): the pairing win was measured on dense
        # block-scale GEMMs where M is exact, and a one-wave bound computed against the
        # average of ragged runtime groups bounds nothing — grouped M-starved problems
        # keep 1-CTA through the ordinary rule, as the MoE planner tests pin.
        early_pair = block_scale and not m_is_group_average and 2 * (-(-N // cta_n)) <= sm
        cta_group = 2 if (M > cta_m or early_pair) else 1

    # The tcgen05 SF swizzle needs whole 128-wide blocks per MMA. The sm120 warp
    # MMA reads its scales from registers, so a CTA tile there only has to be a
    # multiple or a divisor of the 128-row block (validate_block_scale_config_sm120)
    # -- every n_axis width is -- and 3+ parallel GEMMs keep their sub-128 N tile.
    if block_scale and _pipeline != "sm120":
        cta_m = max(cta_m, 128)
        cta_n = max(cta_n, 128)
        if cta_n > cta_n_max:
            raise NotImplementedError(
                f"block-scaled matmul needs cta_tile_n % 128 == 0, but {x} parallel "
                f"GEMM(s) cap the N tile at {cta_n_max}; pick a geometry explicitly "
                "via jit_from_cudnn_graph"
            )

    if b_n_major:
        # N-major B is TMA-loaded one swizzle group of columns at a time, so the
        # PER-CTA N extent must be a whole number of groups (K_BYTES=128 here).
        group_elems = 128 // b_elem_bytes
        group = group_elems * cta_group
        cta_n = -(-max(cta_n, group) // group) * group
        if cta_n > cta_n_max:
            raise NotImplementedError(
                f"N-major B needs cta_tile_n % {group} == 0 (a {group_elems}-element "
                f"swizzle group per CTA under cta_group={cta_group}), but {x} parallel "
                f"GEMM(s) cap the N tile at {cta_n_max}; pick a geometry explicitly "
                "via jit_from_cudnn_graph"
            )

    # --- cluster ------------------------------------------------------------
    pool = [g for g in _CLUSTERS_1D + _CLUSTERS_2D if not _hang_prone(cta_m, cta_n, g[0], g[1])]
    if cta_group == 2:
        pool = [g for g in pool if g[0] % 2 == 0]  # 2-CTA needs cga_size_m % 2 == 0
    # _cluster_score depends only on the wave count, so shapes tie in whole groups —
    # and under a single wave the quantisation and occupancy terms cancel, so a
    # padded launch (cluster wider than the grid) ties with an exact fit. Break
    # ties by fewest launched CTAs first (never pad), then by the sweep's measured
    # preference: A-multicast x4 — cluster (cta_group, 4) — wins the remaining tie
    # group in every cohort (fp4/fp8, 1- and 2-CTA, block-scale or not).
    m_tiles = -(-M // (cta_m * cta_group)) * cta_group
    n_tiles = -(-N // cta_n)

    def _launched(g: tuple[int, int]) -> int:
        return (-(-m_tiles // g[0])) * g[0] * (-(-n_tiles // g[1])) * g[1]

    prefer = ((2, 1), (2, 4)) if cta_m == 256 else ((cta_group, 4),)
    rank = {g: len(prefer) - i for i, g in enumerate(prefer)}
    cgrp_m, cgrp_n = max(pool, key=lambda g: (_cluster_score(M, N, cta_m, cta_n, cta_group, g[0], g[1], sm), -_launched(g), rank.get(g, 0)))

    # --- sm120 ---------------------------------------------------------------
    # The warp-scoped family has axes this scorer never sees (warp grid, CTA
    # K-tile width), and crossing the sm100 pick onto the family's fixed 4x2
    # grid at 128 K-bytes leaves half the measured performance behind on the
    # machine-filling layers (weekly sweep, sm120: fp8 0.73 / fp4 0.72
    # geomean efficiency for the crossing). Choose them here instead.
    if _pipeline == "sm120":
        if block_scale:
            # A 256-wide N tile measures at roughly half the 128-wide rate on
            # the sm120 block-scale template (every machine-filling fp4 layer:
            # 128x256 best 0.61 vs 128x128 best 0.99 of the per-layer optimum).
            cta_n = min(cta_n, 128)
        out_tiles = max(1, _batch) * (-(-M // cta_m)) * (-(-N // cta_n))
        grid_big = out_tiles >= sm
        # Both re-mappings below pay for themselves only when the mainloop is
        # long enough to amortize them; at a shallow K they measure as losses
        # (fp16 K<=64: -0.10..-0.16, fp8 K=1024 at 64-tall tiles: -0.12..-0.17).
        # A-operand bytes: fp4 packs two elements per byte.
        deep_k = (K or 0) * (0.5 if block_scale else b_elem_bytes) >= 2048
        # Dense 1-byte operands on a machine-filling grid with deep K: the
        # 64-byte K tile (fp8: 64 elements) halves the per-stage SMEM and
        # measures 0.95 of the per-layer optimum where the 128-byte tile
        # measures 0.58-0.70. fp16 (128 B = 64 elements already), block-scale
        # (SF swizzle needs the 128-B row) and 64-tall tiles keep 128.
        kb120 = 64 if (grid_big and deep_k and not block_scale and b_elem_bytes == 1 and cta_m >= 128) else 128
        # Warp grid: grids at >= 1/4 machine fill over a deep K want the
        # squarer 2x4 cut (64-wide warp N at cta_n=256, 64x64 at 128x128);
        # latency-bound small grids and shallow K keep the crossing's 4x2.
        # First grid that cuts the CTA tile into whole 16x16 warp MMAs wins.
        order = ((2, 4), (4, 2), (8, 1), (1, 8)) if (deep_k and 4 * out_tiles >= sm) else ((4, 2), (2, 4), (8, 1), (1, 8))
        for wm, wn in order:
            if cta_m % (wm * 16) == 0 and cta_n % (wn * 16) == 0:
                cfg120 = _CATALOG_BY_NAME.get(f"CONFIG_sm120_{cta_m}x{cta_n}x{kb120}_16x16x32_cluster1x1_warps{wm}x{wn}")
                if cfg120 is not None:
                    return cfg120
        # No cataloged sm120 geometry for this tile: fall through to the sm100
        # spelling and let as_pipeline's crossing handle it as before.

    # The 64-byte-K MMA halves the mainloop instruction count and measures
    # faster at the same DENSE-fp8 geometry on every part that issues it, so it
    # is not scored -- it is taken whenever measured-safe: 1-byte
    # operands, a 128-tall CTA in a pair (the swept k64 envelope). Block-scale
    # keeps its width decision in kernel_registry.preferred_mma_tile_k_bytes --
    # at the geometry picked here the sweep shows k64 HURTING a third of the
    # block-scale layers, so nothing is forced from this side.
    mma_kb = 64 if (not block_scale and b_elem_bytes == 1 and cta_m >= 128 and cta_group == 2 and _mma_inst_k64(sm)) else 32
    mma_m = min(cta_m, 128)
    name = f"CONFIG_sm100_{cta_m}x{cta_n}x128_{mma_m}x{cta_n}x{mma_kb}_cluster{cgrp_m}x{cgrp_n}_{cta_group}ctamma"
    return by_name(name)


def as_mma_tile_k_bytes(k_bytes: int, pipeline: str) -> int:
    """The MMA-inst K width ``pipeline`` issues closest to ``k_bytes``: itself
    when the family can issue it, else the family's narrowest. A family that
    fixes the axis (sm103's 48-byte UTCOMMA) therefore always wins."""
    ok = _issuable_mma_tile_k_bytes(config_class_for_pipeline(pipeline))
    return k_bytes if k_bytes in ok else min(ok)


def as_mma_tile_k(cfg: TileConfig, k_bytes: int) -> TileConfig:
    """The same geometry at a different MMA-inst K width. A no-op when the width
    already matches or the config's family cannot issue the requested one, so
    callers need no arch or family knowledge."""
    ok = _issuable_mma_tile_k_bytes(type(cfg))
    if cfg.mma_tile_k_bytes == k_bytes or k_bytes not in ok or cfg.cta_tile_k_bytes % k_bytes:
        return cfg
    # mma_size_k counts instructions per warp tile, so it moves with the width.
    return replace(cfg, mma_tile_k_bytes=k_bytes, mma_size_k=cfg.warp_tile_k_bytes // k_bytes)


def as_pipeline(cfg: TileConfig, pipeline: str) -> TileConfig:
    """The same geometry as a ``pipeline``-family config: the CTA tile is
    carried over, family-fixed axes snap to the target's ClassVar facts
    (``FIXED_MMA_TILE_MN`` / ``FIXED_CGA_MN`` / ``CROSSING_WARP_GRID_MN`` /
    ``DEFAULT_WARPS_PER_CTA``; a CTA-scoped family declares none and keeps the
    source values). A geometry the target family cannot hold raises from its
    ``__post_init__``, so the invariant stays in one place -- the only caller
    is the auto path, whose ``select_config`` picks are all crossable."""
    if cfg.pipeline == pipeline:
        return cfg
    cls = config_class_for_pipeline(pipeline)
    mma_kb = as_mma_tile_k_bytes(cfg.mma_tile_k_bytes, pipeline)
    mma_m, mma_n = getattr(cls, "FIXED_MMA_TILE_MN", (cfg.mma_tile_m, cfg.mma_tile_n))
    cga_m, cga_n = getattr(cls, "FIXED_CGA_MN", (cfg.cga_size_m, cfg.cga_size_n))
    warps_m, warps_n = getattr(cls, "CROSSING_WARP_GRID_MN", (1, 1))
    warp_m = cfg.cta_tile_m // warps_m
    warp_n = cfg.cta_tile_n // warps_n
    return cls(
        pipeline=pipeline,
        cta_tile_m=cfg.cta_tile_m,
        cta_tile_n=cfg.cta_tile_n,
        cta_tile_k_bytes=cfg.cta_tile_k_bytes,
        warp_tile_m=warp_m,
        warp_tile_n=warp_n,
        warp_tile_k_bytes=cfg.cta_tile_k_bytes,
        mma_tile_m=mma_m,
        mma_tile_n=mma_n,
        mma_tile_k_bytes=mma_kb,
        mma_size_m=warp_m // mma_m,
        mma_size_n=warp_n // mma_n,
        mma_size_k=cfg.cta_tile_k_bytes // mma_kb,
        cga_size_m=cga_m,
        cga_size_n=cga_n,
        cga_size_k=cfg.cga_size_k,
        warps_per_cta=getattr(cls, "DEFAULT_WARPS_PER_CTA", cfg.warps_per_cta),
        split_k_slices=cfg.split_k_slices,
        swap_ab=cfg.swap_ab,
        **({"cta_group": getattr(cfg, "cta_group", 1)} if "cta_group" in cls.__dataclass_fields__ else {}),
    )


# ---------------------------------------------------------------------------
# Block-scaled matmul config validation (geometry-only; cta_group lives on the
# template), one function per template family plus a dispatcher. tcgen05: the
# F8_128x4 SF swizzle + 32x128b.warpx4 utccp atom impose mma_tile_m/n % 128 == 0
# and cta_tile_k (elements) % (4*block_size) == 0. sm120: the warp MMA reads its
# scales from registers, so the 128-row block is a TMA-box (CTA tile) fact.
# ---------------------------------------------------------------------------


def validate_block_scale_config_sm100(cfg: TileConfig, block_size: int, cta_tile_k_elems: int) -> None:
    """Raise if ``cfg``'s GEOMETRY cannot run a tcgen05 block-scaled matmul (the
    sm100 family; sm103 differs only in its 384-byte K-tile). ``cta_tile_k_elems``
    is the K-tile in *elements* (FP4: 256 on sm100 / 768 on sm103; FP8: 128)."""
    # The SF 128x4 swizzle gives one scale-factor word per 128-row / 128-column
    # block, and the rule is that EACH MMA INSTRUCTION must land on whole blocks —
    # so it applies to the instruction tile, not the CTA tile. They coincide at
    # num_mma == 1, which is why this used to read cta_tile_m/n.
    if cfg.mma_tile_m % 128 != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires mma_tile_m % 128 == 0 (SF 128x4 swizzle: "
            f"each MMA instruction must cover whole SF blocks); config {cfg.name!r} "
            f"has cta_tile_m={cfg.cta_tile_m} / mma_size_m={cfg.mma_size_m} = {cfg.mma_tile_m}"
        )
    if cfg.mma_tile_n % 128 != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires mma_tile_n % 128 == 0 (SF 128x4 swizzle: "
            f"each MMA instruction must cover whole SF blocks); config {cfg.name!r} "
            f"has mma_tile_n={cfg.mma_tile_n}"
        )
    # The 64-byte MMA-inst K is SM 10.7+ SILICON, not a pipeline property, so it
    # is gated on the ACTIVE GPU here rather than by a config family. No GPU
    # visible (render-only / CI) leaves it open, like every other arch gate.
    if cfg.mma_tile_k_bytes == 64:
        from .kernel_registry import MMA_INST_K64_ARCH_RANGES
        from . import compiler as _C

        arch = _C._current_arch()
        if arch is not None and not any(lo <= arch < hi for lo, hi in MMA_INST_K64_ARCH_RANGES):
            spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in MMA_INST_K64_ARCH_RANGES)
            raise NotImplementedError(f"block-scaled matmul at mma_tile_k_bytes=64 needs {spans}, but the " f"active GPU is sm_{arch}; config {cfg.name!r}")
    # K-tile bytes: sm100 = one 128-B swizzled SMEM row; sm103 = 384 B
    kb_want = 384 if cfg.pipeline == "sm103" else 128
    if cfg.cta_tile_k_bytes != kb_want:
        raise NotImplementedError(
            f"block-scaled matmul on {cfg.pipeline} requires cta_tile_k_bytes == " f"{kb_want}; config {cfg.name!r} has {cfg.cta_tile_k_bytes}"
        )
    if cta_tile_k_elems % (4 * block_size) != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires cta_tile_k (elements) % (4*block_size) " f"== 0; got cta_tile_k={cta_tile_k_elems}, block_size={block_size}"
        )


def validate_block_scale_config_sm120(cfg: TileConfig, block_size: int, cta_tile_k_elems: int) -> None:
    """Raise if ``cfg``'s GEOMETRY cannot run the sm120 warp-MMA block-scaled
    matmul. ``cta_tile_k_elems`` is the K-tile in *elements* (FP4: 256; FP8: 128)."""
    # The warp MMA reads its scales from registers, so the 128-row F8_128x4
    # block is a TMA-box fact here, not an instruction one: the SF box covers
    # whole blocks, so a CTA tile is a multiple of 128 (whole blocks) or a
    # divisor of it (inside one block) -- never astride a block boundary.
    for label, ext in (("cta_tile_m", cfg.cta_tile_m), ("cta_tile_n", cfg.cta_tile_n)):
        if ext % 128 != 0 and 128 % ext != 0:
            raise NotImplementedError(
                f"block-scaled matmul on sm120 requires {label} to be a multiple or a divisor of 128 "
                f"(one F8_128x4 SF block per TMA box); config {cfg.name!r} has {label}={ext}"
            )
    # One K-tile = one 128-byte swizzled SMEM row = one 4-scale SF word per
    # row at block 32 fp8 (and whole words at every other combo).
    if cfg.cta_tile_k_bytes != 128:
        raise NotImplementedError(f"block-scaled matmul on sm120 requires cta_tile_k_bytes == 128; " f"config {cfg.name!r} has {cfg.cta_tile_k_bytes}")
    if cta_tile_k_elems % (4 * block_size) != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires cta_tile_k (elements) % (4*block_size) " f"== 0; got cta_tile_k={cta_tile_k_elems}, block_size={block_size}"
        )
    # Whether the SF boxes still leave an AB stage next to the template's
    # epilogue staging is the renderer's call (the catalog's SMEM sweep
    # counts data only).


def validate_block_scale_config(cfg: TileConfig, block_size: int, cta_tile_k_elems: int) -> None:
    """Raise if ``cfg``'s GEOMETRY cannot run a block-scaled matmul. Dispatches
    on the config's family: the sm120 warp-MMA rules or the tcgen05 ones."""
    if cfg.pipeline == "sm120":
        return validate_block_scale_config_sm120(cfg, block_size, cta_tile_k_elems)
    return validate_block_scale_config_sm100(cfg, block_size, cta_tile_k_elems)
