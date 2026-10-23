"""Immutable SM120 SVDQuant linear route manifest.

Routes only. This module used to carry 51 measured tactics as a source literal
that bypassed the autotuner, and an A/B ladder for hand-driving them. Both are
gone: a tactic is a measurement, and a measurement belongs in something the
tuner produced -- `AutoTuner.save_configs` / `load_configs` -- not in a file
somebody edits. What remains here is what a tactic cannot be derived from: which
runners a shape may offer, which prefix carries which producer route, and the ABI
versions that name a persisted record.

`tests/gemm/test_nvfp4_svdquant_host.py` holds a guard that fails if a
shape-keyed tactic table reappears in this package.
"""

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, TypeAlias


LinearRouteKey: TypeAlias = tuple[int, int, int]

SM120_TACTIC_ABI_VERSION: Final = 3

# Preserve cached winners wherever the linear candidate set is unchanged.
SM120_LINEAR_ROUTE_ABI_VERSION: Final = 5
SM120_PACKED_SMALL_M_ROUTE_ABI_VERSION: Final = 10
SM120_RANK64_FUSED_ROUTE_ABI_VERSION: Final = 11
SM120_RANK128_FUSED_ROUTE_ABI_VERSION: Final = 15

# Historical route versions remain the fallback for shapes without the new
# packed small-M family. Candidate admission below determines the v10 scope.
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
    if rank == 64:
        return SM120_RANK64_FUSED_ROUTE_ABI_VERSION
    if rank == 128:
        return SM120_RANK128_FUSED_ROUTE_ABI_VERSION
    if rank == 32 and any(
        family == SM120_FAMILY_SMALL_M_PACKED
        for family, _, _ in sm120_producer_variants(m, k)
    ):
        return SM120_PACKED_SMALL_M_ROUTE_ABI_VERSION
    return SM120_LINEAR_ROUTE_ABI_OVERRIDES.get(
        (m, k, rank), SM120_LINEAR_ROUTE_ABI_VERSION
    )


def sm120_tactic_abi_version(m: int, k: int) -> int:
    """Return the shared namespace for the unchanged consumer-row encoding."""
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

# Eight LoRA-down warps leave the remaining threads to quantize each tile.
SM120_RANK64_PRODUCER_GEOMETRY_LADDER: Final[tuple[tuple[int, int, int], ...]] = (
    (320, 16, 128),
    (320, 32, 128),
    (352, 48, 128),
    (384, 32, 256),
    (384, 64, 128),
    (384, 80, 128),
    (512, 32, 256),
)

# Rank 128 additionally offers K64 geometries that stage B (L2T) in shared
# memory with cp.async; they mirror FI_SVDQ_RANK128_STAGED_B_GEOMETRIES and are
# appended after the shared ladder so existing producer indices keep their meaning.
SM120_RANK128_STAGED_B_GEOMETRY_LADDER: Final[tuple[tuple[int, int, int], ...]] = (
    (320, 64, 64),
    (352, 48, 64),
    (384, 64, 64),
    (384, 80, 64),
    (416, 80, 64),
    (768, 160, 64),
    (832, 160, 64),
)
# One down warp's staged slice: 16 columns x (64 + 8) bf16.
_STAGED_B_SLICE_BYTES: Final = 16 * (64 + 8) * 2


def _stages_b(rank: int, tile_k: int) -> bool:
    """Mirrors large_m_stages_b() in the producer header."""
    return rank > 64 and tile_k == 64


# How the runtime row offset is formed. Both are built; which one is faster is a
# property of the geometry -- accumulating wins on some and loses on others --
# so the tuner picks, on the device it is running
# on.
SM120_ADDRESS_POLICIES: Final[tuple[int, ...]] = (0, 1)  # 0 recompute, 1 accumulate

_GEOMETRY_SF_VEC: Final = 16
_GEOMETRY_ELEM_BYTES: Final = 2
_GEOMETRY_SHARED_LIMIT: Final = 48 * 1024
_STAGED_B_SHARED_LIMIT: Final = 99 * 1024
_GEOMETRY_MMA_N: Final = 8


def _staged_b_row_split(tile_m: int) -> int:
    """Mirrors staged_b_row_split(): above 80 rows two down-warp sets share B."""
    return 2 if tile_m > 80 else 1


def _staged_b_shared_bytes(tile_m: int, tile_k: int, long_k: bool) -> int:
    """Mirrors staged_b_shared_bytes() (dynamic shared memory of the staged kernel)."""
    sf = 2 * (tile_m * (tile_k // _GEOMETRY_SF_VEC) if long_k else 1)
    return (
        2 * tile_m * (tile_k + 8) * _GEOMETRY_ELEM_BYTES
        + 2 * tile_k * _GEOMETRY_ELEM_BYTES
        + (sf + 15) // 16 * 16
        + 8 * _STAGED_B_SLICE_BYTES
    )


def sm120_producer_geometry_is_valid(
    m: int, k: int, block_threads: int, tile_m: int, tile_k: int, rank: int = 32
) -> bool:
    """Python counterpart of the runtime large-M launcher guards.

    Enumeration has to be able to skip a combination this shape cannot
    instantiate, and the C++ side refuses the same set, so the two must agree.
    Nothing here reads a shape table: it is arithmetic on (M, K) and the
    geometry.
    """
    if m <= 0 or tile_k <= 0 or k % tile_k:
        return False
    if tile_k % (4 * _GEOMETRY_SF_VEC):
        return False
    staged = _stages_b(rank, tile_k)
    max_tile_m = 160 if staged else 80
    if not (0 < tile_m <= max_tile_m and tile_m % 16 == 0):
        return False
    # Mirrors large_m_down_warps(): one n8 fragment per down warp up to rank 64,
    # eight wide down warps above it.
    down_warps = min(rank // _GEOMETRY_MMA_N, 8)
    if staged:
        if tile_m % (16 * _staged_b_row_split(tile_m)):
            return False
        down_warps *= _staged_b_row_split(tile_m)
    quant_threads = block_threads - down_warps * 32
    if block_threads % 32 or quant_threads <= 0:
        return False
    sf_cols_per_tile = tile_k // _GEOMETRY_SF_VEC
    # The staged-B kernel's quantize loop tolerates a remainder.
    if (tile_m * sf_cols_per_tile) % quant_threads and not staged:
        return False
    long_k = k >= 8192
    padded_m = (m + 127) // 128 * 128
    grid_blocks = (padded_m + tile_m - 1) // tile_m
    if long_k:
        prefetch_lines = k * rank * _GEOMETRY_ELEM_BYTES // 128
        if grid_blocks * block_threads < prefetch_lines:
            return False
    if staged:
        return _staged_b_shared_bytes(tile_m, tile_k, long_k) <= _STAGED_B_SHARED_LIMIT
    shared_bytes = (
        2 * tile_m * (tile_k + 8) * _GEOMETRY_ELEM_BYTES
        + 2 * tile_k * _GEOMETRY_ELEM_BYTES
        + 2 * ((tile_m * sf_cols_per_tile) if long_k else 1)
    )
    return shared_bytes <= _GEOMETRY_SHARED_LIMIT


# The row-major and packed small-M families share these tilings and launch
# constraints. The packed family additionally requires a supported L2T pack.
# The autotuner measures every admitted family and tiling.
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

# LoRA ranks the native CUTLASS kernels can stage. Other multiples of 32 compile
# but leave no tactic, so callers exclude them before loading a module.
SM120_CUTLASS_LORA_RANKS: Final[tuple[int, ...]] = (32, 64, 128)

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
SM120_FAMILY_SMALL_M_PACKED: Final = 4
# Warp-specialized rank-128 large-M producer (persistent grid, producer warp +
# mbarrier stage ring). Tiling = (tile M, tile K, stages * 100 + quantize
# warps); the address-policy slot carries the grid mode.
SM120_FAMILY_WS_RANK128: Final = 5

# M537 and packed small-M producers share the lane-packed L2T layout. Other
# families read row-major L2T, so the selected family determines preprocessing.
SM120_FAMILY_PACKS_L2T: Final[frozenset[int]] = frozenset(
    {SM120_FAMILY_M537, SM120_FAMILY_SMALL_M_PACKED}
)

# The M537 producer is a family of kernels for one exact M, parameterised on K
# and its own role split. It cannot generalise -- M is baked into how the roles
# divide -- so it is enumerated for the M it was written for and nothing else.
SM120_M537_M: Final = 537
SM120_M537_TILING: Final[tuple[int, int, int]] = (1024, 24, 16)


def sm120_m537_is_valid(m: int, k: int) -> bool:
    """Whether the M537 producer family exists for this shape."""
    return m == SM120_M537_M and k in (5120, 5376, 7168)


# Mirrors FI_SVDQ_RANK128_WS_TILINGS in nvfp4_smooth_quantize_lora_down_ws_sm120.cuh:
# (tile M, tile K, down layout * 1000 + stages * 100 + quantize warps), down
# layout 0 = 2x4 down warps (rows x columns), 1 = 2x8, 2 = 1x8.
SM120_RANK128_WS_TILING_LADDER: Final[tuple[tuple[int, int, int], ...]] = (
    (160, 64, 1205),
    (160, 64, 205),
    (160, 64, 1210),
    (144, 64, 2209),
    (128, 64, 204),
    (96, 64, 306),
    (96, 64, 1303),
    (80, 64, 2305),
    (64, 64, 304),
    (64, 64, 1304),
    (48, 64, 2403),
    (32, 64, 402),
    (32, 64, 404),
)
# Grid mode (the address-policy slot): 0 = one CTA per SM.
SM120_WS_GRID_MODES: Final[tuple[int, ...]] = (0,)
_WS_RANK: Final = 128


def _ws_shared_bytes(tile_m: int, tile_k: int, stages: int) -> int:
    """Mirrors WsConfig::kSmemBytes (128-byte-swizzled TMA tiles, 1 KB aligned)."""
    stage = (
        (tile_m * tile_k * 2 + _WS_RANK * tile_k * 2 + tile_k * 2 + 1023) // 1024 * 1024
    )
    return 1024 + stages * stage + 2 * stages * 8


def sm120_ws_tiling_is_valid(
    m: int, k: int, tile_m: int, tile_k: int, packed: int
) -> bool:
    """Python counterpart of launch_ws_kernel's guards."""
    stages = packed // 100 % 10
    if tile_k % 64 or m <= 0 or k <= 0 or k % tile_k or m > (1 << 30) or stages <= 0:
        return False
    return _ws_shared_bytes(tile_m, tile_k, stages) <= _STAGED_B_SHARED_LIMIT


def sm120_small_m_tiling_is_valid(
    m: int,
    k: int,
    block_threads: int,
    down_tile_cols: int,
    rows_per_quant_block: int,
    rank: int = 32,
) -> bool:
    """Python counterpart of the runtime small-M launcher guards."""
    if m <= 0 or rows_per_quant_block <= 0 or block_threads % 32:
        return False
    down_warps = block_threads // 32
    if down_warps <= 0:
        return False
    if k % down_warps or (k // down_warps) % 16:
        return False
    # The rank-128 down loop consumes K in 32-element chunks.
    if rank > 64 and (k // down_warps) % 32:
        return False
    if k % (4 * _GEOMETRY_SF_VEC):
        return False
    if down_tile_cols % _GEOMETRY_MMA_N or down_tile_cols // _GEOMETRY_MMA_N <= 0:
        return False
    down_n_tiles = rank // down_tile_cols
    if down_n_tiles * down_tile_cols != rank:
        return False
    if rank > 64:
        # Rank 128 splits quantize work per item, not per RowsPerQuantBlock rows.
        return True
    padded_m = (m + 127) // 128 * 128
    quant_blocks = (m + rows_per_quant_block - 1) // rows_per_quant_block
    return quant_blocks * rows_per_quant_block <= padded_m


@functools.cache
def sm120_producer_variants(
    m: int, k: int, rank: int = 32
) -> tuple[tuple[int, tuple[int, int, int], int], ...]:
    """Every (family, tiling, address policy) this exact shape can run.

    Producer families are enumerated from their own constraints, so a shape
    nobody measured still reaches the fused prefix and a card nobody measured on
    ranks the candidates itself. The address policy is a large-M axis; the
    small-M kernel forms its addresses differently and takes 0, except that at
    rank 128 policy 1 repeats each small-M tiling with the PDL launch of the
    GEMM held until the quantize half is done, so that the GEMM mainloop runs
    under LoRA-down instead of after it.
    """
    if rank not in SM120_CUTLASS_LORA_RANKS:
        return ()
    geometries = (
        SM120_RANK64_PRODUCER_GEOMETRY_LADDER
        if rank >= 64
        else SM120_PRODUCER_GEOMETRY_LADDER
    )
    if rank == 128:
        geometries = geometries + SM120_RANK128_STAGED_B_GEOMETRY_LADDER
    large = tuple(
        (SM120_FAMILY_LARGE_M, geometry, policy)
        for geometry in geometries
        if sm120_producer_geometry_is_valid(m, k, *geometry, rank)
        for policy in SM120_ADDRESS_POLICIES
    )
    small = tuple(
        (SM120_FAMILY_SMALL_M, tiling, 0)
        for tiling in SM120_SMALL_M_TILING_LADDER
        if sm120_small_m_tiling_is_valid(m, k, *tiling, rank)
    )
    if rank == 64:
        return large + small
    if rank == 128:
        # Only the large-M producers stage wide fragments. The warp-specialized
        # and small-M families follow the shared ladder in that order so
        # existing indices keep their meaning.
        ws = tuple(
            (SM120_FAMILY_WS_RANK128, tiling, grid_mode)
            for tiling in SM120_RANK128_WS_TILING_LADDER
            if sm120_ws_tiling_is_valid(m, k, *tiling)
            for grid_mode in SM120_WS_GRID_MODES
        )
        deferred = tuple((family, tiling, 1) for family, tiling, _ in small)
        return large + ws + small + deferred
    m537 = (
        ((SM120_FAMILY_M537, SM120_M537_TILING, 0),)
        if sm120_m537_is_valid(m, k)
        else ()
    )
    # Retain the original native and cuBLASLt ordering. New packed candidates
    # follow them so every legacy producer index keeps its meaning.
    native = large + small + m537
    cublaslt = ((SM120_FAMILY_CUBLASLT, (0, 0, 0), 0),) if native else ()
    packed_small = (
        tuple((SM120_FAMILY_SMALL_M_PACKED, tiling, 0) for _, tiling, _ in small)
        if k in (3072, 5120, 5376, 7168)
        else ()
    )
    return native + cublaslt + packed_small


def sm120_variant_packs_l2t(m: int, k: int, variant: int, rank: int = 32) -> bool:
    """Whether the producer this variant selects wants the packed L2T layout.

    Read from the chosen family rather than from the shape: once the tuner picks
    the producer, the shape no longer determines which layout is correct.
    """
    family, _, _ = sm120_decode_producer_variant(m, k, variant, rank)
    return family in SM120_FAMILY_PACKS_L2T


def sm120_variant_reads_kmajor_l2t(
    m: int, k: int, variant: int, rank: int = 32
) -> bool:
    """Whether the chosen producer reads L2T K-major (every rank-128 family).

    Mirrors large_m_kmajor_l2() and the Rank > 64 small-M path in
    nvfp4_smooth_quantize_lora_down_sm120.cuh.
    """
    family, _, _ = sm120_decode_producer_variant(m, k, variant, rank)
    return (
        family in (SM120_FAMILY_LARGE_M, SM120_FAMILY_WS_RANK128, SM120_FAMILY_SMALL_M)
        and rank > 64
    )


def sm120_producer_variant_count(m: int, k: int, rank: int = 32) -> int:
    return len(sm120_producer_variants(m, k, rank))


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
    m: int, k: int, variant: int, rank: int = 32
) -> tuple[int, tuple[int, int, int], int]:
    """Resolve a variant index to the geometry and policy the launcher needs.

    Out of range falls back to the first admissible pair rather than failing: a
    persisted tactic must still run something correct after the ladder moves.
    """
    variants = sm120_producer_variants(m, k, rank)
    if not 0 <= variant < len(variants):
        return variants[0]
    return variants[variant]


PrefixRoute: TypeAlias = str
PrefixRouteKey: TypeAlias = tuple[int, int]

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
