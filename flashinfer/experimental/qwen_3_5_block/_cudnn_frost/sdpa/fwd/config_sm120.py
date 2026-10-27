# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile-time configuration for the FROST SM120 SDPA prefill templates."""

from __future__ import annotations

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from dataclasses import dataclass
from typing import Optional

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, DTYPE_FP16  # noqa: F401  (DTYPE_E4M3 re-exported for the FP8 template)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.sf_layout import SF_ATOM_BYTES, SF_ATOM_ROWS

SEQ_Q_TILES = (128, 64)
SEQ_KV_TILES = (128, 64, 32)
# Q tiles of the DECODE tile (sm120/decode_f16.py): 16 or 32 packed rows per CTA,
# the consumer warps spread over the KEY axis instead of the rows.
DECODE_Q_TILES = (16, 32)
HEAD_TILE_GRANULE = 16
SUPPORTED_HEAD_TILE_MIN = 16
GENERAL_HEAD_TILE_MAX = 256
SUPPORTED_HEAD_TILE_MAX = 512
GENERAL_HEAD_TILES = tuple(range(SUPPORTED_HEAD_TILE_MIN, GENERAL_HEAD_TILE_MAX + 1, HEAD_TILE_GRANULE))
FP8_HEAD_TILE_GRANULE = 32
FP8_GENERAL_HEAD_TILE_MAX = 256
FP8_GENERAL_HEAD_TILES = tuple(range(FP8_HEAD_TILE_GRANULE, FP8_GENERAL_HEAD_TILE_MAX + 1, FP8_HEAD_TILE_GRANULE))


def mxfp8_sf_tile_bytes(head_dim: int) -> int:
    """Bytes of one 128-row scale-factor tile covering every head-dim plane.

    The same count serves Q/K (rows = tokens, blocks along D) and V (rows = D
    columns, blocks along a 128-key span; a 64-key KV tile reads half of it).
    """
    return SF_ATOM_BYTES * ((head_dim + SF_ATOM_ROWS - 1) // SF_ATOM_ROWS)


D256_FLAVOR = (256, 256)
D512_FLAVOR = (512, 512)
D512_TILE = (64, 32)
FP8_D512_TILE = (64, 64)
GENERAL_KV_TILES = (128, 64)
F16_FLAVORS: frozenset[tuple[int, int]] = frozenset({D256_FLAVOR, D512_FLAVOR})
FP8_FLAVORS: frozenset[tuple[int, int]] = frozenset({D512_FLAVOR})


def head_tile(d: int, granule: int) -> int:
    """A head dim rounded up to a whole ``granule``-column head tile."""
    return -(-d // granule) * granule


def head_tiles(d_qk: int, d_v: int, granule: int) -> tuple[int, int]:
    """Both head dims rounded up to whole ``granule``-column head tiles."""
    return head_tile(d_qk, granule), head_tile(d_v, granule)


def pick_flavor(d_qk: int, d_v: int, fp8: bool) -> Optional[tuple[int, int]]:
    """Select the fixed-tile flavor for the head dimensions, or the general template."""
    if GENERAL_HEAD_TILE_MAX < d_qk <= D512_FLAVOR[0] and GENERAL_HEAD_TILE_MAX < d_v <= D512_FLAVOR[1]:
        return D512_FLAVOR
    tiles = head_tiles(d_qk, d_v, FP8_HEAD_TILE_GRANULE if fp8 else HEAD_TILE_GRANULE)
    return tiles if tiles in (FP8_FLAVORS if fp8 else F16_FLAVORS) else None


def tile_domain(d_qk: int, d_v: int, fp8: bool) -> frozenset[tuple[int, int]]:
    """The (q_tile, kv_tile) pairs the kernel serving these head dims is built for.

    The d512 flavor has one tile; the general and d256 templates take any Q tile
    with a 128- or 64-key KV tile. The half decode tile (any head tile up to
    256) adds its 16- and 32-row Q tiles, whose KV tile is the stage width
    ``decode_kv_stage`` fixes for the head tiles. The ranking and the adapter
    intersect their SMEM fit with this set, so no tile is proposed that
    declines at build.
    """
    if pick_flavor(d_qk, d_v, fp8) == D512_FLAVOR:
        return frozenset({FP8_D512_TILE if fp8 else D512_TILE})
    domain = frozenset((m, n) for m in SEQ_Q_TILES for n in GENERAL_KV_TILES)
    tiles = head_tiles(d_qk, d_v, HEAD_TILE_GRANULE)
    if not fp8 and max(tiles) <= GENERAL_HEAD_TILE_MAX:
        domain |= frozenset((m, decode_kv_stage(*tiles)) for m in DECODE_Q_TILES)
    return domain


def is_decode_tile(q_tile: int) -> bool:
    """Whether a Q tile names the decode tile (sm120/decode_f16.py) rather than a prefill template."""
    return q_tile in DECODE_Q_TILES


# The decode tile streams K/V in stages of at most DECODE_STAGE_BYTES (the K and
# V halves of ``decode_kv_stage`` keys) wherever the 16-key floor allows: head
# tiles summing past 256 overrun it, up to twice at 256 + 256. Every key group
# owns its own slots of the ring (``decode_slots_per_group`` each, at most
# DECODE_MAX_SLOTS_PER_GROUP), so several stages of every group are in flight
# while its warps compute and every slot has one waiter, which consumes its uses
# in order (an mbarrier parity wait cannot tell a phase from the one two phases
# later).
DECODE_STAGE_BYTES = 8192
DECODE_MAX_SLOTS_PER_GROUP = 6
# Consumer warps of the decode tile: ``q_tile // 16`` M-groups x the key groups
# that share them. One producer warp and three idle warps complete the second
# warpgroup, so setmaxregister moves registers per whole warpgroup.
DECODE_CONSUMER_WARPS = 4
# Each K and V region of a stage starts on a 1024-byte boundary: the period of
# the widest (128-byte) TMA swizzle, which the ldmatrix addressing assumes.
DECODE_REGION_ALIGN = 1024


def decode_kv_groups(q_tile: int) -> int:
    """Key groups of the decode tile: the consumer warps sharing one M-group."""
    return DECODE_CONSUMER_WARPS // (q_tile // 16)


def decode_kv_stage(head_tile_qk: int, head_tile_v: int) -> int:
    """Keys per K/V stage of the decode tile: the largest multiple of 16 (one
    P@V k-step) whose K and V half tiles fit DECODE_STAGE_BYTES, clamped to
    16..64 -- at equal Q/K and V head tiles 64 up to 32, 32 at 48 and 64, 16
    from 80. The 16-key floor overruns DECODE_STAGE_BYTES once the head tiles
    sum past 256 (a 256 + 256 stage is 16 KiB); :func:`decode_stage_bytes` is
    the stage's actual size."""
    keys = DECODE_STAGE_BYTES // (2 * (head_tile_qk + head_tile_v))
    return max(16, min(64, keys // 16 * 16))


def decode_region_bytes(keys: int, head_tile: int) -> int:
    """SMEM bytes of one K or V region of a decode stage: ``keys`` half rows of
    ``head_tile`` columns, rounded up to DECODE_REGION_ALIGN."""
    return -(-keys * head_tile * 2 // DECODE_REGION_ALIGN) * DECODE_REGION_ALIGN


def decode_stage_bytes(head_tile_qk: int, head_tile_v: int) -> int:
    """SMEM bytes of one decode K/V stage: the K region, then the V region."""
    keys = decode_kv_stage(head_tile_qk, head_tile_v)
    return decode_region_bytes(keys, head_tile_qk) + decode_region_bytes(keys, head_tile_v)


def decode_merge_bytes(head_tile_v: int, q_tile: int) -> int:
    """SMEM bytes the decode epilogue's key-group merge stages in the (drained)
    ring: every consumer warp but the first of its M-group spills its fp32 O
    accumulator plus its two rows' (max, sum) per lane."""
    return (decode_kv_groups(q_tile) - 1) * (q_tile // 16) * 32 * (head_tile_v // 2 + 4) * 4


MXFP8_KV64_MIN_HEAD_TILE_QK = 224


def kv_tile_preference(d_qk: int, mxfp8: bool = False) -> tuple[int, ...]:
    """KV tiles from fastest to slowest; the ranking and the adapter take the
    first one in ``tile_domain`` that fits SMEM. Larger is faster, except MXFP8
    Q/K head tiles of ``MXFP8_KV64_MIN_HEAD_TILE_QK`` or more, where the
    128-key tile crowds the compute warps' registers (it spills at 256)."""
    if mxfp8 and head_tile(d_qk, FP8_HEAD_TILE_GRANULE) >= MXFP8_KV64_MIN_HEAD_TILE_QK:
        return (64, 128, 32)
    return SEQ_KV_TILES


# SMEM the SM120 parts expose to a kernel. The adapter asks cutlass for the
# authoritative number at build time; this constant lets the ranking answer
# "would this tile even fit" without importing the DSL.
SMEM_CAPACITY_BYTES = 101376


def decode_slots_per_group(head_tile_qk: int, head_tile_v: int, q_tile: int) -> int:
    """Ring slots each key group of the decode tile owns: as many as fit
    SMEM_CAPACITY_BYTES with their full/empty mbarriers, at most
    DECODE_MAX_SLOTS_PER_GROUP. 0 when not even one slot per group fits, or the
    ring cannot hold the epilogue's key-group merge."""
    stage = decode_stage_bytes(head_tile_qk, head_tile_v)
    kv_groups = decode_kv_groups(q_tile)
    slots = min(DECODE_MAX_SLOTS_PER_GROUP, SMEM_CAPACITY_BYTES // ((stage + 16) * kv_groups))
    if slots < 1 or slots * kv_groups * stage < decode_merge_bytes(head_tile_v, q_tile):
        return 0
    return slots


def decode_num_stages(head_tile_qk: int, head_tile_v: int, q_tile: int) -> int:
    """Ring depth of the decode tile: every key group's slots (0 = no ring fits)."""
    return decode_kv_groups(q_tile) * decode_slots_per_group(head_tile_qk, head_tile_v, q_tile)


def register_budgets(q_tile: int) -> tuple[int, int]:
    max_regs_per_sm = 2048
    max_regs_per_thread = 256
    load_regs_per_thread = 24
    num_compute_warps = 8 if q_tile == 128 else 4
    num_load_warps = 4
    free_slots = max_regs_per_sm - num_load_warps * load_regs_per_thread
    compute_regs_per_thread = min(max_regs_per_thread, free_slots // num_compute_warps // 8 * 8)
    return load_regs_per_thread, compute_regs_per_thread


# The MXFP8 D512 kernel stages, per K tile and warp half, the first local K scale
# plane (64 keys: 8 bytes of each of a plane's 32 atom lines) and the second
# plane's first 32 keys (4 bytes per line).
MXFP8_D512_K_SF_STAGE_BYTES = 2 * 32 * (8 + 4)


def smem_bytes(d_qk: int, d_v: int, q_tile: int, kv_tile: int, itemsize: int = 2, out_itemsize: Optional[int] = None, mxfp8: bool = False) -> int:
    """SMEM the SM120 prefill kernel needs for one specialization.

    The general and D=256 kernels alias one K tile (D_QK wide) plus one V
    tile (D_V wide) with the q_tile x D_V output staging tile. The two terms size independently:
    ``itemsize`` is the QKV element, ``out_itemsize`` the staged output's, and
    FP8 differs on exactly that (1-byte KV, half-precision O). The f16 D=256
    kernel also keeps half of its Q tile resident in shared memory. The f16
    D=512 kernel keeps a quarter of Q there and adds the fp32 partial-S
    exchange between the two warps that split each Q slab's head dim. Its O
    epilogue reuses that Q/exchange storage in two rounds, while the K/V slab
    receives the next Q tile. It needs a third TMA barrier for those Q loads.
    Every d512 envelope shape uses the full 512-wide storage on both sides.
    The FP8 D512 flavor keeps Q in registers and uses a 64-key tile, so its
    extra storage is the score exchange plus the three TMA barriers.
    MXFP8 (``mxfp8``) adds one 128-row K scale-factor tile and one V
    scale-factor tile beside the K/V slab; they are read during the KV loop
    and do not alias the O staging tile. The MXFP8 D512 flavor adds the V
    tile and ``MXFP8_D512_K_SF_STAGE_BYTES`` of staged K scale words: there is
    no room beside the exchange for a K tile.

    The half decode tile (``is_decode_tile(q_tile)``) is its K/V stage ring
    (``decode_num_stages`` stages of ``decode_stage_bytes``) plus one full and
    one empty mbarrier per stage; its epilogue stages through the drained ring.
    A head tile it cannot ring reports more than any SM120 part provides.

    Lives here rather than in the adapter because the ranking must not propose
    a tile the kernel cannot fit, and two answers to that question is how a plan
    list fills with entries that decline at build.
    """
    if is_decode_tile(q_tile):
        hq, hv = head_tiles(d_qk, d_v, HEAD_TILE_GRANULE)
        stages = decode_num_stages(hq, hv, q_tile) if itemsize == 2 and not mxfp8 and max(hq, hv) <= GENERAL_HEAD_TILE_MAX else 0
        if stages == 0:
            return SMEM_CAPACITY_BYTES + 1
        return stages * (decode_stage_bytes(hq, hv) + 16)
    flavor = pick_flavor(d_qk, d_v, fp8=itemsize == 1)
    if flavor == D512_FLAVOR:
        d_qk, d_v = D512_FLAVOR
    kv_or_o = max(kv_tile * (d_qk + d_v) * itemsize, q_tile * d_v * (itemsize if out_itemsize is None else out_itemsize))
    q_resident = 0
    exchange = 0
    if flavor == D256_FLAVOR:
        q_resident = q_tile * (flavor[0] // 2) * itemsize
    elif flavor == D512_FLAVOR:
        q_resident = q_tile * (flavor[0] // 4) * itemsize if itemsize == 2 else 0
        exchange = q_tile * kv_tile * 4 * 2
    barrier_bytes = 24 if flavor == D512_FLAVOR else 16
    scale_factors = 0
    if mxfp8 and flavor == D512_FLAVOR:
        # The d512 kernel keeps the V scale tile and the staged K scale words.
        scale_factors = mxfp8_sf_tile_bytes(d_v) + MXFP8_D512_K_SF_STAGE_BYTES
    elif mxfp8:
        scale_factors = mxfp8_sf_tile_bytes(d_qk) + mxfp8_sf_tile_bytes(d_v)
    return kv_or_o + q_resident + exchange + barrier_bytes + scale_factors


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph parameters that change the traced SM120 kernel.

    Tensor geometry deliberately stays out of this record. Scalar dimensions
    are inputs to the template module's per-shape ``compile()`` cache, while
    strides follow the fixed compact-BSHD kernel contract. This frozen record
    identifies the import-time specialization shared by all compatible shapes.
    """

    dtype_qkv: int = DTYPE_FP16
    # O dtype (frost tile_dsl codes, same table as dtype_qkv). The f16
    # template stores O at the input dtype and ignores this; the FP8 template
    # selects its quantizing-store epilogue from it (E4M3/E5M2/BF16/FP16).
    dtype_o: int = DTYPE_FP16
    # The mask is ONE diagonal band (same model as config_sm100 / the analyzer
    # facts): per-side offsets from the diagonal, None = unbounded on that
    # side. window_right = 0 is plain causal; window_right > 0 widens the
    # diagonal right by R columns (cuDNN's diagonal_band_right_bound).
    # bottom_right anchors the band's diagonal at the bottom-right corner.
    window_left: int | None = None
    window_right: int | None = None
    bottom_right: bool = False
    seq_q_lens_present: bool = False
    seq_kv_lens_present: bool = False
    has_sink: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)).
    stats_log2: bool = False
    thd_varlen: bool = False
    sched_policy: int = SCHED_NATURAL
    q_tile: int = SEQ_Q_TILES[0]
    kv_tile: int = SEQ_KV_TILES[0]
    pack_gqa: bool = False
    # KV split: each Q tile's KV-tile range [min_kv_tile, num_kv_tiles) is cut
    # into ``split_kv`` contiguous chunks, each run by its own CTA writing a
    # partial (O, LSE) that kernels/sm100/split_combine.py reduces.  1 = off.
    # Opt-in only -- the graph front door never selects it.
    split_kv: int = 1
    # attn_scale < 0: the kernel negates Q as it loads it and runs at |attn_scale|, so the raw-score row max still
    # bounds P (#1435). APPEND-ONLY, default False.
    negate_scores: bool = False


def validate_params(
    params: TemplateParams,
    allowed_dtypes: tuple[int, ...] = (DTYPE_BF16, DTYPE_FP16),
    allowed_o_dtypes: tuple[int, ...] = (DTYPE_BF16, DTYPE_FP16),
    allow_right_band: bool = True,
    decode_tile: bool = False,
) -> None:
    """Validate the SM120 template specialization.

    Reachable failures should already have been rejected by the engine
    capabilities or adapter support checks; this validation is a backstop for
    direct template use. ``allowed_dtypes``/``allowed_o_dtypes`` default to the
    FP16/BF16 template's sets (which stores O at the input dtype and plumbs no
    quantizing epilogue); the FP8 template passes its own — FP8 in, any of its
    four quantizing-store epilogues out. ``allow_right_band=False`` rejects a
    widened right band for templates that do not plumb one. ``decode_tile``
    names the decode tile (sm120/decode_f16.py), which takes the
    ``DECODE_Q_TILES`` at its stage-width KV tile; every other template is
    built for the ``SEQ_Q_TILES``.
    """

    if params.dtype_qkv not in allowed_dtypes:
        raise ValueError(f"SM120 SDPA: dtype_qkv must be one of {allowed_dtypes}; got {params.dtype_qkv}")
    if params.dtype_o not in allowed_o_dtypes:
        raise ValueError(f"SM120 SDPA: dtype_o must be one of {allowed_o_dtypes}; got {params.dtype_o}")
    if params.split_kv < 1:
        raise ValueError(f"SM120 SDPA: split_kv must be >= 1 (1 = KV-split off); got {params.split_kv}")
    if params.split_kv > 1:
        # Each of these would need extra machinery in the combine pass.
        if params.thd_varlen:
            raise ValueError("SM120 SDPA: split_kv > 1 is dense-only (THD packs its own flat grid)")
        if params.has_sink and params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
            # The half templates write sink-free partials and the combine folds
            # the sink once; the FP8 templates fold it in their epilogue, so
            # every split would add its own copy of it.
            raise ValueError("SM120 SDPA: split_kv > 1 with an attention sink is served by the FP16/BF16 templates only")
        if params.sched_policy != SCHED_NATURAL:
            # The split index rides the NATURAL grid's batch axis (y = batch +
            # split*B); LPT / LPT_L2 flatten the grid to 1-D and derive the batch
            # from the linear tile id, so there is no axis left to fold it into.
            raise ValueError(f"SM120 SDPA: split_kv > 1 currently requires SCHED_NATURAL; got sched_policy={params.sched_policy}")

    if params.window_right is not None and params.window_right < 0:
        raise ValueError(f"SM120 SDPA: window_right must be None (unbounded) or >= 0 (0 = plain causal); got {params.window_right}")
    if not allow_right_band and params.window_right not in (None, 0):
        raise ValueError(f"SM120 SDPA: right-band widening is not plumbed for this template; window_right must be None or 0, got {params.window_right}")
    if params.bottom_right and params.window_right is None:
        raise ValueError("SM120 SDPA: bottom_right anchors the band's diagonal and requires a right bound (window_right)")
    if params.window_left is not None and params.window_left < 0:
        raise ValueError(f"SM120 SDPA: window_left must be non-negative; got {params.window_left}")
    q_tiles = DECODE_Q_TILES if decode_tile else SEQ_Q_TILES
    if params.q_tile not in q_tiles:
        raise ValueError(f"SM120 SDPA: this template's q_tile must be one of {q_tiles}; got {params.q_tile}")
    if decode_tile:
        # The decode tile's KV tile is its stage width (decode_kv_stage), fixed by the head tiles.
        if params.kv_tile not in (16, 32, 64):
            raise ValueError(f"SM120 SDPA: the decode tile's kv_tile is its stage width (16, 32 or 64); got {params.kv_tile}")
        if params.thd_varlen:
            raise ValueError("SM120 SDPA: the decode tile is dense-only (THD keeps the prefill templates)")
        if params.sched_policy != SCHED_NATURAL:
            raise ValueError(f"SM120 SDPA: the decode tile runs SCHED_NATURAL only; got sched_policy={params.sched_policy}")
    elif params.kv_tile not in SEQ_KV_TILES:
        raise ValueError(f"SM120 SDPA: kv_tile must be one of {SEQ_KV_TILES}; got {params.kv_tile}")
    if params.thd_varlen:
        if not params.seq_kv_lens_present:
            raise ValueError("SM120 SDPA: thd_varlen requires seq_kv_lens_present (the THD metadata tensor)")
        if params.seq_q_lens_present:
            raise ValueError("SM120 SDPA: seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
    if params.pack_gqa and params.thd_varlen:
        raise ValueError("SM120 SDPA: pack_gqa is not supported with thd_varlen")
