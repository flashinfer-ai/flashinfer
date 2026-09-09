"""Immutable SM120 SVDQuant linear route manifest.

Routes only. This module used to carry 51 measured tactics as a source literal
that bypassed the autotuner, and an A/B ladder for hand-driving them. Both are
gone: a tactic is a measurement, and a measurement belongs in something the
tuner produced -- `AutoTuner.save_configs` / `load_configs` -- not in a file
somebody edits. What remains here is what a tactic cannot be derived from: which
runners a shape may offer, which prefix carries which producer route, and the ABI
versions that name a persisted record.

`tests/gemm/test_svdquant_sm120_routes.py` holds a guard that fails if a
shape-keyed tactic table reappears in this package.
"""

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, TypeAlias


LinearRouteKey: TypeAlias = tuple[int, int, int]

SM120_TACTIC_ABI_VERSION: Final = 3

# v4 is the packed (consumer row, producer variant) grammar. It is deliberately
# NOT a global bump. Where the producer axis is degenerate -- every shape off
# the nine-entry ladder -- a packed tactic is the same integer as the bare row
# it replaces and selects the same thing, so a v3 record still resolves
# correctly and throwing it away would cost a re-profile for nothing.
#
# The cost of getting that wrong is recorded in
# test_sm120_linear_op_name_keeps_v5_for_every_other_shape: an earlier global
# bump made every unrelated shape miss its record and re-profile, and one such
# re-tune picked a slower tactic and regressed that shape.
SM120_LINEAR_ROUTE_ABI_VERSION: Final = 5

# A route version changes only when the runner set for that exact (M, K, rank)
# changes. Unlisted shapes retain the v5 namespace and its persisted winner.
SM120_LINEAR_ROUTE_ABI_OVERRIDES: Final[Mapping[LinearRouteKey, int]] = (
    MappingProxyType(
        {
            # v6: screened MiniMax-H3 rows 50, 52, 59, 61, 68, and 70.
            (73984, 14336, 32): 6,
            (73984, 7168, 32): 6,
            (61056, 7168, 32): 6,
            (61056, 14336, 32): 6,
            (82752, 14336, 32): 6,
            (82752, 7168, 32): 6,
            # v7: the remaining Batch-4 routes with an exact fused prefix.
            (537, 5120, 32): 7,
            (537, 7168, 32): 7,
            (537, 5376, 32): 7,
            (1935, 5120, 32): 7,
            (1935, 7168, 32): 7,
            (1935, 14336, 32): 7,
            (6913, 5120, 32): 7,
            (6913, 7168, 32): 7,
            (6913, 14336, 32): 7,
            # v8: the previously unfused K5376 producer families.
            (1935, 5376, 32): 8,
            (6913, 5376, 32): 8,
            (73984, 5376, 32): 8,
            (61056, 5376, 32): 8,
            (82752, 5376, 32): 8,
            # v9: Batch-4 case 4 uses the independently screened cuBLASLt prefix.
            (537, 14336, 32): 9,
        }
    )
)


def _route_keys_for_version(version: int) -> frozenset[LinearRouteKey]:
    return frozenset(
        key
        for key, route_version in SM120_LINEAR_ROUTE_ABI_OVERRIDES.items()
        if route_version == version
    )


SM120_LINEAR_ROUTE_V6_MKR: Final = _route_keys_for_version(6)
SM120_LINEAR_ROUTE_V7_MKR: Final = _route_keys_for_version(7)
SM120_LINEAR_ROUTE_V8_MKR: Final = _route_keys_for_version(8)
SM120_LINEAR_ROUTE_V9_MKR: Final = _route_keys_for_version(9)


def sm120_linear_route_abi_version(m: int, k: int, rank: int) -> int:
    """Return the persistent route namespace for one exact linear shape."""
    return SM120_LINEAR_ROUTE_ABI_OVERRIDES.get(
        (m, k, rank), SM120_LINEAR_ROUTE_ABI_VERSION
    )


def sm120_tactic_abi_version(m: int, k: int) -> int:
    """Return the tactic namespace for one exact shape.

    One namespace for every shape now that a tactic is just a K3 consumer row.
    It stayed shape-keyed while a second, producer-geometry axis was packed into
    the same integer, because only the shapes that carried that axis had a
    candidate space that changed meaning.
    """
    return SM120_TACTIC_ABI_VERSION


# The producer geometries the kernel is built for. A property of the kernel, not
# of any shape: every one is instantiated, and which of them a given (M, K) may
# use is decided by the predicate below rather than by a table.
SM120_PRODUCER_GEOMETRY_LADDER: Final[tuple[tuple[int, int, int], ...]] = (
    (192, 16, 512),
    (192, 32, 128),
    (192, 48, 128),
    (224, 48, 128),
    (256, 32, 256),
    (256, 80, 128),
    (288, 80, 128),
    (384, 32, 256),
)

# How the runtime row offset is formed. Both are built; which one is faster is a
# property of the geometry -- accumulating wins on some and loses on others --
# so the tuner picks, on the device it is running
# on.
SM120_ADDRESS_POLICIES: Final[tuple[int, ...]] = (0, 1)  # 0 recompute, 1 accumulate

_GEOMETRY_RANK: Final = 32
_GEOMETRY_SF_VEC: Final = 16
_GEOMETRY_ELEM_BYTES: Final = 2
_GEOMETRY_DOWN_WARPS: Final = 4
_GEOMETRY_SHARED_LIMIT: Final = 48 * 1024
_GEOMETRY_MMA_N: Final = 8


def sm120_producer_geometry_is_valid(
    m: int, k: int, block_threads: int, tile_m: int, tile_k: int
) -> bool:
    """Mirror of the launcher's compile-time guard, as a predicate.

    Enumeration has to be able to skip a combination this shape cannot
    instantiate, and the C++ side refuses the same set, so the two must agree.
    Nothing here reads a shape table: it is arithmetic on (M, K) and the
    geometry.
    """
    if tile_k <= 0 or k % tile_k:
        return False
    if tile_k % (4 * _GEOMETRY_SF_VEC):
        return False
    if not (0 < tile_m <= 80 and tile_m % 16 == 0):
        return False
    quant_threads = block_threads - _GEOMETRY_DOWN_WARPS * 32
    if block_threads % 32 or quant_threads <= 0:
        return False
    sf_cols_per_tile = tile_k // _GEOMETRY_SF_VEC
    if (tile_m * sf_cols_per_tile) % quant_threads:
        return False
    long_k = k >= 8192
    padded_m = (m + 127) // 128 * 128
    grid_blocks = (padded_m + tile_m - 1) // tile_m
    if long_k:
        prefetch_lines = k * _GEOMETRY_RANK * _GEOMETRY_ELEM_BYTES // 128
        if grid_blocks * block_threads < prefetch_lines:
            return False
    shared_bytes = (
        2 * tile_m * (tile_k + 8) * _GEOMETRY_ELEM_BYTES
        + 2 * tile_k * _GEOMETRY_ELEM_BYTES
        + 2 * ((tile_m * sf_cols_per_tile) if long_k else 1)
    )
    return shared_bytes <= _GEOMETRY_SHARED_LIMIT


# The small-M producer is a second kernel with its own tiling, and its own
# constraints. Both families are built for every tiling in their ladder; which
# family and tiling a shape may use is arithmetic, and which is fastest is the
# tuner's to find.
SM120_SMALL_M_TILING_LADDER: Final[tuple[tuple[int, int, int], ...]] = (
    (256, 16, 4),
    (768, 8, 8),
    (768, 16, 16),
    (768, 32, 16),
    (896, 16, 16),
    (1024, 8, 4),
    (1024, 8, 16),
    (1024, 16, 16),
)

SM120_FAMILY_LARGE_M: Final = 0
SM120_FAMILY_SMALL_M: Final = 1
SM120_FAMILY_M537: Final = 2
# The cuBLASLt prefix is a producer like any other, not a route decided by a
# table. It used to be selected by an 8-shape "measured faster" list mirrored in
# three places; those shapes are where it was *measured*, not where it *runs* --
# it is a library GEMM and takes any shape the quantizer does. Enumerating it
# here hands the choice to the autotuner, which is the only thing that can know
# whether it wins on a card nobody measured.
SM120_FAMILY_CUBLASLT: Final = 3

# The M537 producer reads L2T in a prepacked layout while every other family
# reads it row-major. That is a property of the family, not of the shape: hand a
# packed matrix to a row-major reader and it misreads rather than fails, which is
# how a tactic that selected one producer while the caller packed for another
# turns into a wrong answer instead of an error.
SM120_FAMILY_PACKS_L2T: Final[frozenset[int]] = frozenset({SM120_FAMILY_M537})

# The M537 producer is a family of kernels for one exact M, parameterised on K
# and its own role split. It cannot generalise -- M is baked into how the roles
# divide -- so it is enumerated for the M it was written for and nothing else.
SM120_M537_M: Final = 537
SM120_M537_TILING: Final[tuple[int, int, int]] = (1024, 24, 16)


def sm120_m537_is_valid(m: int, k: int) -> bool:
    """Whether the M537 producer family exists for this shape."""
    return m == SM120_M537_M and k in (5120, 5376, 7168)


def sm120_small_m_tiling_is_valid(
    m: int, k: int, block_threads: int, down_tile_cols: int, rows_per_quant_block: int
) -> bool:
    """Mirror of SmallMLaunchGeometry's asserts, as a predicate."""
    if m <= 0 or rows_per_quant_block <= 0 or block_threads % 32:
        return False
    down_warps = block_threads // 32
    if down_warps <= 0:
        return False
    if k % down_warps or (k // down_warps) % 16:
        return False
    if k % (4 * _GEOMETRY_SF_VEC):
        return False
    if down_tile_cols % _GEOMETRY_MMA_N or down_tile_cols // _GEOMETRY_MMA_N <= 0:
        return False
    down_n_tiles = _GEOMETRY_RANK // down_tile_cols
    if down_n_tiles * down_tile_cols != _GEOMETRY_RANK:
        return False
    padded_m = (m + 127) // 128 * 128
    quant_blocks = (m + rows_per_quant_block - 1) // rows_per_quant_block
    return quant_blocks * rows_per_quant_block <= padded_m


@functools.cache
def sm120_producer_variants(
    m: int, k: int
) -> tuple[tuple[int, tuple[int, int, int], int], ...]:
    """Every (family, tiling, address policy) this exact shape can run.

    Both producer families are enumerated from their own constraints, so a shape
    nobody measured still reaches the fused prefix and a card nobody measured on
    ranks the candidates itself. The address policy is a large-M axis only; the
    small-M kernel forms its addresses differently and takes 0.
    """
    large = tuple(
        (SM120_FAMILY_LARGE_M, geometry, policy)
        for geometry in SM120_PRODUCER_GEOMETRY_LADDER
        if sm120_producer_geometry_is_valid(m, k, *geometry)
        for policy in SM120_ADDRESS_POLICIES
    )
    small = tuple(
        (SM120_FAMILY_SMALL_M, tiling, 0)
        for tiling in SM120_SMALL_M_TILING_LADDER
        if sm120_small_m_tiling_is_valid(m, k, *tiling)
    )
    m537 = (
        ((SM120_FAMILY_M537, SM120_M537_TILING, 0),)
        if sm120_m537_is_valid(m, k)
        else ()
    )
    # Offered wherever a native producer is, and last, so the variant indices of
    # every shape that had one before keep their meaning.
    native = large + small + m537
    cublaslt = ((SM120_FAMILY_CUBLASLT, (0, 0, 0), 0),) if native else ()
    return native + cublaslt


def sm120_variant_packs_l2t(m: int, k: int, variant: int) -> bool:
    """Whether the producer this variant selects wants the packed L2T layout.

    Read from the chosen family rather than from the shape: once the tuner picks
    the producer, the shape no longer determines which layout is correct.
    """
    family, _, _ = sm120_decode_producer_variant(m, k, variant)
    return family in SM120_FAMILY_PACKS_L2T


def sm120_producer_variant_count(m: int, k: int) -> int:
    return len(sm120_producer_variants(m, k))


SM120_PRODUCER_SHIFT: Final = 8
SM120_K3_ROW_MASK: Final = (1 << SM120_PRODUCER_SHIFT) - 1


def sm120_pack_tactic(k3_row: int, producer_variant: int = 0) -> int:
    """Pack a (consumer row, producer variant) pair into one tactic integer."""
    if not 0 <= k3_row <= SM120_K3_ROW_MASK:
        raise ValueError(f"k3 row outside the packable range: {k3_row}")
    if producer_variant < 0:
        raise ValueError(f"negative producer variant: {producer_variant}")
    return k3_row | (producer_variant << SM120_PRODUCER_SHIFT)


def sm120_unpack_tactic(tactic: int) -> tuple[int, int]:
    """Split a packed tactic back into (consumer row, producer variant)."""
    if tactic < 0:
        return tactic, 0
    return tactic & SM120_K3_ROW_MASK, tactic >> SM120_PRODUCER_SHIFT


def sm120_decode_producer_variant(
    m: int, k: int, variant: int
) -> tuple[int, tuple[int, int, int], int]:
    """Resolve a variant index to the geometry and policy the launcher needs.

    Out of range falls back to the first admissible pair rather than failing: a
    persisted tactic must still run something correct after the ladder moves.
    """
    variants = sm120_producer_variants(m, k)
    if not 0 <= variant < len(variants):
        return variants[0]
    return variants[variant]


PrefixRoute: TypeAlias = str
PrefixRouteKey: TypeAlias = tuple[int, int]

# Mirror of the exact-shape ladder in
# include/flashinfer/gemm/svdquant_sm120_prefix_route.h: (default, admitted).
# The two sides must agree because the M537 native producer consumes a prepacked
# LoRA-down matrix while the cuBLASLt prefix consumes the row-major one; the
# packed layout below is chosen from the effective route, not from the shape.
#
# Every alternate rung this ladder once carried was measured and refuted, so each shape now admits only its measured default. The
# C++ header records the per-rung medians. The table is retained for the
# gate/layout reconciliation, not as a live A/B instrument.

# Shapes whose native producer is the M537 mixed kernel, which indexes L2T with
# the prepacked layout. Row-major L2T on those routes would be misread.


# --- the producer axis of a packed tactic ------------------------------------
#
# A tactic is a pair, packed the way nvfp4_svdquant_gemm_tactic_row already
# packs its own fields (kernel_id | splits << 8 | raster << 16 | swizzle << 24):
#
#     tactic = k3_row | producer_variant << SM120_PRODUCER_SHIFT
#
# k3_row is a row of kRuntimeTacticTableSm120 -- 0..82, so eight bits is ample --
# and producer_variant indexes sm120_producer_variants(m, k).
#
# Variant 0 is always the ladder default. That is what makes the change inert
# where the axis is degenerate: a packed tactic whose producer half is zero is
# numerically the same integer as the bare row it replaces, and selects the same
# producer that row selected before this axis existed.


# --- producer geometry axis -------------------------------------------------
#
# The large-M producer takes its block size and both tile extents as template
# arguments, so a geometry is a whole kernel instantiation rather than a runtime
# argument. Everything below mirrors
# include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh; the mirror
# exists because Python has to know how many candidates a shape offers in order
# to build the tactic list, and C++ has to know which one an index names in
# order to launch it. Tests hold the two against each other.

# ORDER IS ABI -- see the same note on kGeometryLadder in the header.

# Budget, not a judgement: every admitted geometry is another instantiation in a
# single translation unit and another candidate the tuner profiles. Raising this
# from 1 to 4 measurably lengthens that unit's compile and grows its object.

# The geometry each exact large-M shape is pinned to today, which is always
# variant 0 so an unselected shape runs what it has always run.


# The (M, K) projection of the frozen benchmark matrix -- the shapes this
# backend has actually been measured on. It no longer decides anything: fused
# admission is computed from the launch-geometry ladder
# (_sm120_fused_linear_supported), so a shape outside this list is offered to
# the tuner too. Kept as data because the tests and the route ladder need a
# name for "the shapes we benchmark", and pinned against the registry by
# test_fused_linear_mk_is_the_benchmark_projection so the two cannot drift.
SM120_FUSED_LINEAR_MK: Final[frozenset[tuple[int, int]]] = frozenset(
    {
        (64, 3072),
        (512, 3072),
        (512, 5120),
        (4096, 12288),
        (6889, 12288),
        (7800, 5120),
        (7800, 8960),
        (7800, 13824),
        (27280, 3072),
        (27280, 14336),
        # 32760x1536x1536 and 32760x8960x1536 reached the fused runner only
        # because they carried a pinned tactic, whose prefixes were unioned in
        # separately. Admission is route data, not a measured preference, so it
        # stays behind when the pins go. Without it the tuner sees only
        # Sm120CutlassLinearRunner for these two, which measured slower on
        # 32760x1536x1536.
        (32760, 1536),
        (32760, 5120),
        (32760, 8960),
        (32760, 13824),
        (73984, 7168),
        (61056, 7168),
        (61056, 14336),
        (73984, 14336),
        (82752, 14336),
        (82752, 7168),
        (537, 5120),
        (537, 7168),
        (537, 5376),
        (537, 14336),
        (1935, 5120),
        (1935, 5376),
        (1935, 7168),
        (1935, 14336),
        (6913, 5120),
        (6913, 5376),
        (73984, 5376),
        (61056, 5376),
        (82752, 5376),
        (6913, 7168),
        (6913, 14336),
        (75600, 5120),
        (75600, 13824),
        (9216, 12288),
        (16384, 12288),
        # Prefixes whose native producer was added after a full sweep of the
        # legal launch geometries showed the fused producer beating the unfused
        # prefix -- a quantize kernel plus a separate rank-32 torch.mm, which
        # streams the activation twice. The six further
        # prefixes swept alongside these are deliberately absent: fusing them
        # measured slower, so admitting them would cost time. The dispatcher
        # entries and the full measured table live in
        # include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh.
        (512, 1536),
        (7800, 1536),
        (256, 3072),
        (6889, 3072),
        (9216, 3072),
        (16384, 3072),
        # Prefixes whose producer measured SLOWER than the prefix they pay
        # today. They are admitted anyway, so the fused route exists for every
        # production shape and the tuner is the thing that rejects it. Leaving
        # them out would mean this file had decided, once, on one card, that a
        # candidate was not worth measuring -- and nothing would ever re-check.
        # The measured numbers are in the dispatcher comment beside each one.
        (1024, 3072),
        (4096, 3072),
        (64, 12288),
        (256, 12288),
        (512, 12288),
        (1024, 12288),
    }
)
