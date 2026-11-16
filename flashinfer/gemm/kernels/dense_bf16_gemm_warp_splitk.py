# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""cp.async-pipelined BF16 GEMM whose four compute warps hold split-K partials (warp split-K)."""

from __future__ import annotations

import dataclasses
import functools
import math

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
import torch as _torch
from cutlass import const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.distributed import multimem_st_4xb32

_MMA_SHAPE = (16, 8, 16)
_COMPUTE_WARPS = 4
_A_LOADER_WARPS = 2
_A_LOADER_THREADS = _A_LOADER_WARPS * 32
_MAX_M = 32
# Ring depth cap: cold-read bandwidth saturates around 12 stages (96 KB in
# flight), so deeper rings only enlarge the tactic space.
_MAX_STAGES = 16
# The consumer's K loop is fully unrolled; bound the tile count so JIT time and
# code size stay proportionate (64 tiles of 256 elements = K 16384).
_MAX_K_TILES = 64
_SUPPORTED_OUTPUT_TILES = (16, 32)
_SUPPORTED_TOKEN_TILES = (8, 16, 32)
_SUPPORTED_K_TILES = (128, 256)
_SUPPORTED_B_LOADER_WARPS = (1, 2)
_SMEM_CAPACITY = cutlass.utils.get_smem_capacity_in_bytes("sm_100")
# Per-SM shared memory is the per-CTA maximum plus the 1 KB the driver reserves
# for every resident CTA; residency estimates must include that reservation.
_CTA_RESERVED_SMEM = 1024
_SM_SMEM_BYTES = _SMEM_CAPACITY + _CTA_RESERVED_SMEM
_MAILBOX_GROUP_VALUES = 32
_MAILBOX_PADDED_GROUP_VALUES = 34
# Peer-store kernels keyed without addresses; functools.cache would key on the launch arguments.
_PEER_STORE_KERNELS: dict = {}


# Pad each mailbox group by two FP32 words to reduce bank conflicts.
def _partial_offset(feature, token, token_tile: int):
    group_rows = _MAILBOX_GROUP_VALUES // token_tile
    return (
        (feature % group_rows) * token_tile
        + (feature // group_rows) * _MAILBOX_PADDED_GROUP_VALUES
        + token
    )


def _a_fragment_offset(lane, compute_warp, m_iter, phase, k_tile):
    """Element offset of one lane's A ``ldmatrix.x4`` address within a stage."""
    linear = lane % 16 + (lane // 16) * 128 + phase * 256
    row = linear % _MMA_SHAPE[0] + m_iter * _MMA_SHAPE[0]
    col = linear // _MMA_SHAPE[0] + compute_warp * (k_tile // _COMPUTE_WARPS)
    return row * k_tile + (col ^ ((row % 8) * 8))


def _b_pair_offset(lane, compute_warp, n_iter, pair, token_tile, k_tile):
    """Element offset of one lane's B ``ldmatrix.x4`` address within a stage."""
    b_atom = lane % 8 + ((lane % 32) // 8) * (token_tile * 8)
    linear = b_atom + pair * 2 * token_tile * 16 + n_iter * 8
    row = linear % token_tile
    col = linear // token_tile + compute_warp * (k_tile // _COMPUTE_WARPS)
    return row * k_tile + (col ^ ((row % 8) * 8))


@dsl_user_op
def _ldmatrix_x4(smem, offset, *, loc=None, ip=None):
    """``ldmatrix.x4`` of the 16-bit fragment ``offset`` elements into ``smem``."""
    return prims.ldmatrix(
        smem.iterator.raw_ptr() + offset,
        num=4,
        layout=prims.MMALayout.ROW,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _l2_policy_evict_first(*, loc=None, ip=None):
    """64-bit L2 evict_first policy (fraction 1.0).

    Must stay a runtime ``createpolicy``: with the equivalent constant, ptxas
    13.0 emits the invalid ``LDGSTS ... [R+UR], desc[UR1]`` encoding.
    """
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cp_async_16_with_policy(dst, src, policy, *, loc=None, ip=None):
    """16-byte cp.async.cg global->shared carrying an L2 cache-policy hint."""
    llvm.inline_asm(
        None,
        [
            dst.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            policy.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _issue_weight_tile(
    smem,
    weight,
    tidx,
    output_tile_idx,
    k_tile,
    stage,
    policy,
    output_tile,
    k_tile_size,
    with_policy,
    *,
    loc=None,
    ip=None,
):
    """Issue one A-loader thread's cp.async copies for weight K tile ``k_tile``.

    Plain Python control flow only: this helper is not preprocessed by the DSL.
    ``with_policy`` selects the ``.L2::cache_hint`` form; the pre-barrier
    prologue tile must use the plain form because ptxas 13.0 mis-encodes the
    hinted copy in that position (``LDGSTS ... [R+UR], desc[UR<odd>]``).
    """
    copy_count = output_tile * k_tile_size // (8 * _A_LOADER_THREADS)
    for copy_idx in range(copy_count):
        linear = tidx * 8 + copy_idx * _A_LOADER_THREADS * 8
        row = linear // k_tile_size
        col = linear % k_tile_size
        dst_offset = (
            stage * output_tile * k_tile_size
            + row * k_tile_size
            + (col ^ ((row % 8) * 8))
        )
        src_offset = weight.layout(
            (
                output_tile_idx * output_tile + row,
                k_tile * k_tile_size + col,
            )
        )
        if with_policy:
            # CuTe pointers: ``toint()`` yields the 32-bit shared address for
            # SMEM and the 64-bit global address for GMEM, as the PTX expects.
            _cp_async_16_with_policy(
                smem.iterator + dst_offset,
                weight.iterator + src_offset,
                policy,
                loc=loc,
                ip=ip,
            )
        else:
            prims.cp_async_shared_global(
                smem.iterator.raw_ptr() + dst_offset,
                weight.iterator.raw_ptr() + src_offset,
                16,
                "cg",
                loc=loc,
                ip=ip,
            )


@dataclasses.dataclass(frozen=True)
class WarpSplitKTactic:
    output_tile: int
    token_tile: int
    k_tile: int
    stages: int
    b_loader_warps: int


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _smem_bytes(tactic: WarpSplitKTactic) -> int:
    cursor = tactic.output_tile * tactic.k_tile * 2 * tactic.stages
    cursor = _align_up(cursor, 1024)
    cursor += tactic.token_tile * tactic.k_tile * 2 * tactic.stages
    cursor = _align_up(cursor, 16)
    partial_stride = (
        tactic.output_tile * tactic.token_tile // _MAILBOX_GROUP_VALUES
    ) * _MAILBOX_PADDED_GROUP_VALUES
    cursor += _COMPUTE_WARPS * partial_stride * 4
    cursor = _align_up(cursor, 8)
    cursor += tactic.stages * 8
    cursor = _align_up(cursor, 8)
    return cursor + tactic.stages * 8


def _k_tile_count_supported(k_tile: int, k_tile_count: int) -> bool:
    # The two-tile 256-wide schedule can generate an invalid hinted LDGSTS.
    return k_tile != 256 or k_tile_count != 2


def _legal_stages(
    output_tile: int,
    token_tile: int,
    k_tile: int,
    k: int,
    b_loader_warps: int,
) -> tuple[int, ...]:
    # A single-stage ring only makes sense when K is one tile; with ring reuse it
    # cannot overlap anything, and ptxas mis-encodes its hinted LDGSTS.
    k_tile_count = k // k_tile
    if k_tile_count > _MAX_K_TILES or not _k_tile_count_supported(k_tile, k_tile_count):
        return ()
    return tuple(
        stages
        for stages in range(_min_stages(k, k_tile), min(_MAX_STAGES, k_tile_count) + 1)
        if _smem_bytes(
            WarpSplitKTactic(output_tile, token_tile, k_tile, stages, b_loader_warps)
        )
        <= _SMEM_CAPACITY
    )


def _min_stages(k: int, k_tile: int) -> int:
    return 1 if k // k_tile <= 1 else 2


def _k_supported(k: int) -> bool:
    return k > 0 and any(
        k % k_tile == 0 and k // k_tile <= _MAX_K_TILES for k_tile in _SUPPORTED_K_TILES
    )


def validate_tactic(tactic: WarpSplitKTactic, m: int, n: int, k: int) -> None:
    if tactic.output_tile not in _SUPPORTED_OUTPUT_TILES:
        raise ValueError(f"unsupported output_tile={tactic.output_tile}")
    if tactic.token_tile not in _SUPPORTED_TOKEN_TILES:
        raise ValueError(f"unsupported token_tile={tactic.token_tile}")
    if tactic.k_tile not in _SUPPORTED_K_TILES:
        raise ValueError(f"unsupported k_tile={tactic.k_tile}")
    if tactic.b_loader_warps not in _SUPPORTED_B_LOADER_WARPS:
        raise ValueError(f"unsupported b_loader_warps={tactic.b_loader_warps}")
    if not 1 <= m <= _MAX_M or n <= 0 or n % tactic.output_tile:
        raise ValueError(f"unsupported shape {(m, n, k)}")
    if k <= 0 or k % tactic.k_tile:
        raise ValueError(f"K={k} must be divisible by k_tile={tactic.k_tile}")
    k_tile_count = k // tactic.k_tile
    if k_tile_count > _MAX_K_TILES:
        raise ValueError(
            f"K={k} spans more than {_MAX_K_TILES} tiles of {tactic.k_tile}"
        )
    if not _k_tile_count_supported(tactic.k_tile, k_tile_count):
        raise ValueError(f"unsupported k_tile={tactic.k_tile} for K={k}")
    if not (
        _min_stages(k, tactic.k_tile) <= tactic.stages <= min(_MAX_STAGES, k_tile_count)
    ):
        raise ValueError(f"invalid stages={tactic.stages}")
    required_smem = _smem_bytes(tactic)
    if required_smem > _SMEM_CAPACITY:
        raise ValueError(f"tactic requires {required_smem} bytes of shared memory")


def validate_inputs(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    bias: _torch.Tensor | None = None,
) -> tuple[int, int, int]:
    tensors = (a, b, out) + ((bias,) if bias is not None else ())
    if any(not isinstance(tensor, _torch.Tensor) for tensor in tensors):
        raise ValueError("a, b, out, and bias must be torch tensors")
    if any(tensor.ndim != 2 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 2D tensors")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors):
        raise ValueError("a, b, out, and bias must be CUDA tensors on the same device")
    if any(tensor.dtype != _torch.bfloat16 for tensor in tensors):
        raise ValueError("a, b, out, and bias must have bfloat16 dtype")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if bias is not None and (bias.shape != (n,) or not bias.is_contiguous()):
        raise ValueError(f"bias must be a contiguous ({n},) vector")
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"M must be in [1, {_MAX_M}], got {m}")
    if n <= 0 or n % 16:
        raise ValueError(f"N={n} must be a positive multiple of 16")
    if not _k_supported(k):
        raise ValueError(
            f"K={k} must be a positive multiple of 128 spanning at most "
            f"{_MAX_K_TILES} tiles of 256 (or 128) elements"
        )

    if a.stride() != (k, 1):
        raise ValueError(f"a must be packed row-major, got stride {a.stride()}")
    if b.stride() != (1, k):
        raise ValueError(f"b must be packed column-major, got stride {b.stride()}")
    if out.stride() != (n, 1):
        raise ValueError(f"out must be packed row-major, got stride {out.stride()}")
    if any(tensor.data_ptr() % 32 for tensor in tensors):
        raise ValueError("a, b, and out must be 32-byte aligned")
    return m, n, k


def default_tactic(m: int, n: int, k: int) -> WarpSplitKTactic:
    k_tile = 128 if k <= 512 or k % 256 else 256
    k_tile_count = k // k_tile
    sm_count = _torch.cuda.get_device_properties(
        _torch.cuda.current_device()
    ).multi_processor_count
    output_tiles = tuple(tile for tile in _SUPPORTED_OUTPUT_TILES if n % tile == 0)
    if not output_tiles:
        raise ValueError(f"N={n} must be divisible by a supported output tile")
    # Minimize scheduled wave work; prefer the wider tile on ties.
    output_tile = min(
        output_tiles,
        key=lambda tile: (
            ((n // tile + sm_count - 1) // sm_count) * tile,
            -tile,
        ),
    )
    # Splitting M over two half-size token tiles doubles the CTA count at the
    # cost of a second pass over each weight tile; that only pays while the
    # doubled grid still fits in one wave.
    token_tile = 8 if m <= 8 else 16 if m <= 16 else 32
    if token_tile > 8 and 2 * (n // output_tile) <= sm_count:
        token_tile //= 2
    # Two B-loader warps pay off once the ring is reused (long K); for a handful
    # of K tiles the extra loader threads only add barrier arrivals.
    b_loader_warps = 2 if token_tile >= 16 or k_tile_count >= 8 else 1
    legal_stages = _legal_stages(output_tile, token_tile, k_tile, k, b_loader_warps)
    if not legal_stages:
        raise ValueError(f"no legal tactic for shape {(m, n, k)}")
    cta_count = (n // output_tile) * ((m + token_tile - 1) // token_tile)
    target_residency = 2 if cta_count > sm_count else 1
    resident_stages = tuple(
        stages
        for stages in legal_stages
        if target_residency
        * (
            _smem_bytes(
                WarpSplitKTactic(
                    output_tile, token_tile, k_tile, stages, b_loader_warps
                )
            )
            + _CTA_RESERVED_SMEM
        )
        <= _SM_SMEM_BYTES
    )
    stages = max(resident_stages or legal_stages)
    tactic = WarpSplitKTactic(output_tile, token_tile, k_tile, stages, b_loader_warps)
    validate_tactic(tactic, m, n, k)
    return tactic


def autotune_tactics(m: int, n: int, k: int) -> list[WarpSplitKTactic]:
    if not 1 <= m <= _MAX_M or n <= 0 or n % 16 or not _k_supported(k):
        return []
    tactics: list[WarpSplitKTactic] = []
    for output_tile in _SUPPORTED_OUTPUT_TILES:
        if n % output_tile:
            continue
        for token_tile in _SUPPORTED_TOKEN_TILES:
            if token_tile > 16 and m <= 16:
                continue  # A 32-token tile only pays for M > 16.
            for k_tile in _SUPPORTED_K_TILES:
                if k % k_tile:
                    continue
                # Prune same-grid token padding for multi-tile K.
                # Wider tiles can win when K fits one tile.
                if k // k_tile > 1 and any(
                    smaller_token_tile < token_tile
                    and (m + smaller_token_tile - 1) // smaller_token_tile
                    == (m + token_tile - 1) // token_tile
                    for smaller_token_tile in _SUPPORTED_TOKEN_TILES
                ):
                    continue
                for b_loader_warps in _SUPPORTED_B_LOADER_WARPS:
                    tactics.extend(
                        WarpSplitKTactic(
                            output_tile,
                            token_tile,
                            k_tile,
                            stages,
                            b_loader_warps,
                        )
                        for stages in _legal_stages(
                            output_tile,
                            token_tile,
                            k_tile,
                            k,
                            b_loader_warps,
                        )
                    )
    default = default_tactic(m, n, k)
    sm_count = _torch.cuda.get_device_properties(
        _torch.cuda.current_device()
    ).multi_processor_count
    return [
        default,
        *(
            tactic
            for tactic in tactics
            if tactic != default
            and _keep_autotune_tactic(tactic, default, tactics, m, n, k, sm_count)
        ),
    ]


# Below ~96 KB of A + B in flight per SM a cold read is latency-bound (see
# _MAX_STAGES); a shallower ring is only worth profiling above it.
_MIN_BYTES_IN_FLIGHT = 96 * 1024


def _resident_ring(output_tile, token_tile, k_tile, stages, ctas_per_sm):
    """Deepest legal ring that lets ``ctas_per_sm`` CTAs co-reside on one SM."""
    fit = [
        s
        for s in stages
        if ctas_per_sm
        * (
            _smem_bytes(WarpSplitKTactic(output_tile, token_tile, k_tile, s, 1))
            + _CTA_RESERVED_SMEM
        )
        <= _SM_SMEM_BYTES
    ]
    return max(fit) if fit else min(stages)


# Stages below the co-resident ring that stay in the autotune space.  The best
# ring depth moves with SM count and bandwidth, so keep a window instead of one.
_RING_WINDOW = 1

# The other output tile is profiled while its work per SM wave is within this
# factor of the default's.  The default picks the smaller wave work, but whole
# waves are a coarse model and the other width can win away from exact ties.
_OUTPUT_TILE_WAVE_WORK_RATIO = 1.5


def _keep_autotune_tactic(t, default, tactics, m, n, k, sm_count) -> bool:
    """Keep the tactics that can beat the default; see the rules below.

    The rules are the same on every GPU and depend only on the shape, the SM
    count and shared memory.  The kernel is bound by the weight stream, so what
    matters is how many CTAs co-reside (wave fill) and how many bytes each SM
    keeps in flight; K tile and B-loader count are both profiled because their
    best value moves with the SM count.  The kept tactics contain the narrower
    set fitted on B200 / B300 (cold-L2 CUPTI sweep of every legal tactic,
    M=1..32, N=128..8192, K=768..8192), which kept the best tactic within 2.6%.
    """
    ctas = (n // t.output_tile) * ((m + t.token_tile - 1) // t.token_tile)
    default_ctas = (n // default.output_tile) * (
        (m + default.token_tile - 1) // default.token_tile
    )
    same_grid = ctas == default_ctas
    # Token tile: the default or any finer one (more CTAs).  A coarser tile
    # re-streams fewer weights but halves the CTAs; it only won on tiny
    # short-K grids.
    if t.token_tile > default.token_tile and not (
        t.token_tile <= 16 and n // 16 <= 16 and k <= 2048
    ):
        return False

    # Output tile: the default, or the other width when its work per SM wave is
    # close to the default's, on a tiny short-K grid, or when it keeps the
    # default CTA count.
    def wave_work(tile):
        return (n // tile + sm_count - 1) // sm_count * tile

    if t.output_tile != default.output_tile:
        if not (
            (
                n % 32 == 0
                and wave_work(t.output_tile)
                <= _OUTPUT_TILE_WAVE_WORK_RATIO * wave_work(default.output_tile)
                and t.token_tile <= 16
            )
            or (n // 16 <= 16 and k <= 1024)
            or same_grid
        ):
            return False
        if t.token_tile != default.token_tile and not same_grid:
            return False
    stages = [
        x.stages
        for x in tactics
        if (x.output_tile, x.token_tile, x.k_tile, x.b_loader_warps)
        == (t.output_tile, t.token_tile, t.k_tile, t.b_loader_warps)
    ]
    waves = (ctas + sm_count - 1) // sm_count
    ring = _resident_ring(t.output_tile, t.token_tile, t.k_tile, stages, waves)
    # Stages: a window ending at the deepest ring that keeps the grid in one
    # wave, and on a half-filled default-token-tile grid one ring shallower
    # (one more CTA per SM) if it still keeps enough bytes in flight.
    if ring - _RING_WINDOW <= t.stages <= ring:
        return True
    shallower = _resident_ring(t.output_tile, t.token_tile, t.k_tile, stages, waves + 1)
    return (
        t.stages == shallower
        and t.token_tile == default.token_tile
        and 2 * ctas >= waves * sm_count
        and waves * shallower * (t.output_tile + t.token_tile) * t.k_tile * 2
        >= _MIN_BYTES_IN_FLIGHT
        and t.b_loader_warps == 1
    )


def _make_smem_layout(dtype, rows: int, cols: int, stages: int):
    copy_elems = 128 // dtype.width
    major_size = min(cols, 128 * 8 // dtype.width)
    outer = cute.tile_to_shape(
        cute.make_layout((8, major_size), stride=(major_size, 1)),
        (rows, cols, stages),
        (0, 1, 2),
    )
    return cute.make_composed_layout(
        cute.make_swizzle(
            min(int(math.log2(major_size // copy_elems)), 3),
            4,
            int(math.log2(copy_elems)),
        ),
        0,
        outer,
    )


@dataclasses.dataclass(frozen=True)
class LamportPeerStoreConfig:
    """Destinations of a Lamport peer-store epilogue. The addresses are kernel
    arguments, the rest is compiled in (see ``split_peer_store_config``).

    Lamport protocol: consumers pre-fill the slots with zero, which the producer
    never stores, and poll until every 32-bit word is nonzero; no flag or barrier.
    The tile goes into every ``peer_ptrs`` buffer (one per rank, in rank order)
    or once through ``multicast_ptr``. A buffer holds two halves of
    ``world_size * slot_bytes`` bytes and the producer writes half
    ``h = epoch & 1``, reading the int32 at ``epoch_ptr`` after the
    grid-dependency wait, and releases its dependents only after that wait, so a
    consumer that reads the epoch at launch sees every earlier advance; the
    consumer must not advance the epoch before this grid completes (a
    grid-dependency wait before the advance). Element
    ``(m, n)`` of this rank's ``[M, N]`` tile lands at byte offset

    * ``"broadcast"`` (all-reduce input, full tile per rank):
      ``(h * world_size + rank) * slot_bytes + (m * N + n) * 2``
    * ``"gather"`` (all-gather input, this rank's column slice):
      ``h * world_size * slot_bytes + ((rank * M + m) * N + n) * 2``

    An all-zero 32-bit word is stored as ``0x00008000`` (its first element as ``-0.0``).
    """

    layout: str
    world_size: int
    rank: int
    slot_bytes: int
    epoch_ptr: int
    peer_ptrs: tuple[int, ...] | None
    multicast_ptr: int | None


def make_lamport_peer_store_config(
    *, peer_ptrs, multicast_ptr, epoch_ptr, slot_bytes, rank, world_size, layout, m, n
) -> LamportPeerStoreConfig:
    """Validate the destinations of an ``m`` x ``n`` peer-store GEMM; see ``LamportPeerStoreConfig``."""
    if (peer_ptrs is None) == (multicast_ptr is None):
        raise ValueError("pass exactly one of peer_ptrs and multicast_ptr")
    if layout not in ("broadcast", "gather"):
        raise ValueError(f"unsupported layout {layout!r}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} must be in [0, {world_size})")
    if peer_ptrs is not None:
        peer_ptrs = tuple(int(ptr) for ptr in peer_ptrs)
        if len(peer_ptrs) != world_size:
            raise ValueError(f"expected {world_size} peer_ptrs, got {len(peer_ptrs)}")
    else:
        multicast_ptr = int(multicast_ptr)
    if any(ptr <= 0 or ptr % 16 for ptr in peer_ptrs or (multicast_ptr,)):
        raise ValueError("destinations must be non-null and 16-byte aligned")
    if int(epoch_ptr) <= 0 or int(epoch_ptr) % 4:
        raise ValueError("epoch_ptr must be non-null and 4-byte aligned")
    if slot_bytes % 16 or m * n * 2 > slot_bytes:
        raise ValueError(f"slot_bytes must be 16-aligned and >= {m * n * 2}")
    return LamportPeerStoreConfig(
        layout, world_size, rank, slot_bytes, int(epoch_ptr), peer_ptrs, multicast_ptr
    )


def split_peer_store_config(config: LamportPeerStoreConfig):
    """Split ``config`` into its compiled part (addresses zeroed; the compile key) and
    the kernel arguments: ``epoch_ptr``, then the destination addresses."""
    unicast = config.multicast_ptr is None
    ptrs = config.peer_ptrs if unicast else (config.multicast_ptr,)
    compiled = dataclasses.replace(
        config,
        epoch_ptr=0,
        peer_ptrs=(0,) * len(ptrs) if unicast else None,
        multicast_ptr=None if unicast else 0,
    )
    # Int64: plain Python ints would be passed as Int32.
    return compiled, tuple(cutlass.Int64(ptr) for ptr in (config.epoch_ptr, *ptrs))


def bind_peer_store_args(config: LamportPeerStoreConfig, args):
    """In the kernel: ``config`` with the arguments of ``split_peer_store_config``."""
    if config.multicast_ptr is not None:
        return dataclasses.replace(config, epoch_ptr=args[0], multicast_ptr=args[1])
    return dataclasses.replace(config, epoch_ptr=args[0], peer_ptrs=args[1:])


def load_peer_epoch_parity(config: LamportPeerStoreConfig):
    """``epoch & 1``; a plain load, issued after the grid-dependency wait."""
    epoch = cute.make_tensor(
        cute.make_ptr(
            cutlass.Int32,
            cutlass.Int64(config.epoch_ptr),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        cute.make_layout(1),
    )
    return epoch[0] & 1


def replace_zero_sentinel(chunk):
    """Store an all-zero word of ``chunk`` as ``0x00008000``, the Lamport sentinel rule."""
    one = cutlass.Uint32(1)
    for i in range(4):
        word = chunk[i]
        # word | -word has its top bit set exactly when word != 0.
        chunk[i] = word | ((((word | (cutlass.Uint32(0) - word)) >> 31) ^ one) << 15)
    return chunk


def store_peer_chunk(config: LamportPeerStoreConfig, parity, row, feature, m, n, chunk):
    """Store the 16 B chunk of output (``row``, ``feature``..``feature + 7``) of an
    ``m`` x ``n`` GEMM to every destination buffer, in half ``parity``."""
    half = cutlass.Int64(parity) * config.world_size
    if config.layout == "gather":
        offset = half * config.slot_bytes + (
            (cutlass.Int64(config.rank * m + row) * n + feature) * 2
        )
    else:
        offset = (half + config.rank) * config.slot_bytes + (
            (cutlass.Int64(row) * n + feature) * 2
        )
    if config.multicast_ptr is not None:
        multimem_st_4xb32(
            cute.make_ptr(
                cutlass.Uint32,
                cutlass.Int64(config.multicast_ptr) + offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            *(chunk[i].ir_value() for i in range(4)),
        )
    else:
        for ptr in config.peer_ptrs:
            dst = cute.make_ptr(
                cutlass.Uint32,
                cutlass.Int64(ptr) + offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            cute.autovec_copy(chunk, cute.make_tensor(dst, cute.make_layout(4)))


@cute.experimental.jit
def store_tile_to_peers(
    config: LamportPeerStoreConfig,
    rD,
    tiled_copy_t2r,
    epi_tid,
    d_layout,
    acc_dtype,
    sA,
    mC,
    m_idx,
    n_idx,
    cta_m: cutlass.Constexpr,
    cta_n: cutlass.Constexpr,
):
    """Peer-store epilogue of the TMEM GEMMs (128 epilogue threads, named barrier 1): the
    ``cta_m`` x ``cta_n`` tile ``rD`` of the (N, M) output ``mC``, staged in A's first stage ``sA``."""
    c_dtype = rD.element_type
    # (1) stmatrix the tile into A's stage 0 (idle after the MMAs), token-major
    #     with the 16 B chunks of each token XOR-swizzled by token % 8.
    stage = cute.make_tensor(
        cute.recast_ptr(sA.iterator, dtype=c_dtype),
        cute.make_composed_layout(
            cute.make_swizzle(3, 3, cta_m.bit_length() - 4),
            0,
            cute.make_layout((cta_m, cta_n), stride=(1, cta_m)),
        ),
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(
        sm100_utils.get_smem_store_op(d_layout, c_dtype, acc_dtype, tiled_copy_t2r),
        tiled_copy_t2r,
    )
    cute_ext.partition_and_copy(
        tiled_copy_r2s.get_slice(epi_tid), tiled_copy_r2s.retile(rD), stage
    )
    cute.arch.barrier(barrier_id=1, number_of_threads=128)
    # Ordered after the B loader's grid-dependency wait by the mainloop barriers.
    parity = load_peer_epoch_parity(config)
    n = cute.size(mC, mode=[0])
    m = cute.size(mC, mode=[1])
    feat_chunks = cta_m // 8
    chunks = feat_chunks * cta_n
    # The stage as 32-bit words: (word, 16 B chunk, token).
    stage_words = cute.make_tensor(
        cute.recast_ptr(sA.iterator, dtype=cutlass.Uint32),
        cute.make_layout((4, feat_chunks, cta_n)),
    )
    for c_iter in cutlass.range_constexpr((chunks + 127) // 128):
        # Consecutive threads take consecutive chunks of one token, so each
        # warp stores whole rows to the peers.
        chunk = c_iter * 128 + epi_tid
        feature_block = chunk % feat_chunks
        token_in_tile = chunk // feat_chunks
        token = n_idx * cta_n + token_in_tile
        feature = m_idx * cta_m + feature_block * 8
        if chunk < chunks:
            # A 2-CTA cluster can round the grid up past N: predicate features as well as tokens.
            if token < m:
                if feature < n:
                    # (2) Load the chunk from SMEM.
                    values = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
                    cute.autovec_copy(
                        stage_words[
                            None, feature_block ^ token_in_tile % 8, token_in_tile
                        ],
                        values,
                    )
                    # (3) Replace zeros with the sentinel.
                    values = replace_zero_sentinel(values)
                    # (4) Store to every peer.
                    store_peer_chunk(config, parity, token, feature, m, n, values)


class CpAsyncWarpSplitKKernel:
    def __init__(
        self,
        tactic: WarpSplitKTactic,
        use_pdl: bool,
        has_bias: bool,
        peer_store: LamportPeerStoreConfig | None = None,
    ) -> None:
        self.peer_store = peer_store
        self.output_tile = tactic.output_tile
        self.token_tile = tactic.token_tile
        self.k_tile = tactic.k_tile
        self.stages = tactic.stages
        self.b_loader_warps = tactic.b_loader_warps
        self.compute_warp_base = _A_LOADER_WARPS + self.b_loader_warps
        self.threads = (self.compute_warp_base + _COMPUTE_WARPS) * 32
        self.load_threads = self.compute_warp_base * 32
        self.use_pdl = use_pdl
        self.has_bias = has_bias
        self.smem_bytes = _smem_bytes(tactic)

    @cute.experimental.jit
    def __call__(
        self, weight, activation, bias, output, stream: _cuda.CUstream, peer_args=None
    ):
        sA_layout = _make_smem_layout(
            weight.element_type, self.output_tile, self.k_tile, self.stages
        )
        sB_layout = _make_smem_layout(
            activation.element_type, self.token_tile, self.k_tile, self.stages
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(weight.element_type, cutlass.Float32, _MMA_SHAPE),
            cute.make_layout((1, 1, _COMPUTE_WARPS)),
            permutation_mnk=(self.output_tile, self.token_tile, self.k_tile),
        )
        self.kernel(
            weight, activation, bias, output, sA_layout, sB_layout, tiled_mma, peer_args
        ).launch(
            grid=(
                cute.ceil_div(weight.shape[0], self.output_tile),
                cute.ceil_div(activation.shape[0], self.token_tile),
                1,
            ),
            block=(self.threads, 1, 1),
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        weight: cute.Tensor,
        activation: cute.Tensor,
        bias: cute.Tensor,  # (N,) broadcast bias; unread when has_bias is False
        output: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        peer_args,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        output_tile_idx, token_tile_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE
        reuse_stages = cute.size(weight, mode=[1]) > self.stages * self.k_tile

        sA = cute_ext.allocate(
            weight.element_type,
            cute.AddressSpace.smem,
            sA_layout,
            alignment=1024,
        )
        sB = cute_ext.allocate(
            activation.element_type,
            cute.AddressSpace.smem,
            sB_layout,
            alignment=1024,
        )
        values_per_lane = self.output_tile * self.token_tile // cute.arch.WARP_SIZE
        partial_stride = (
            self.output_tile * self.token_tile // _MAILBOX_GROUP_VALUES
        ) * _MAILBOX_PADDED_GROUP_VALUES
        partials = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(_COMPUTE_WARPS * partial_stride),
            alignment=16,
        )
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator

        k_tile_count = cute.size(weight, mode=[1]) // self.k_tile

        if warp_idx < _A_LOADER_WARPS:
            # Tile 0 has no mbarrier dependency: issue it before the init handshake
            # so its DRAM latency overlaps init and the CTA barrier.
            _issue_weight_tile(
                sA,
                weight,
                tidx,
                output_tile_idx,
                0,
                0,
                cutlass.Int64(0),
                self.output_tile,
                self.k_tile,
                False,
            )
        bias_value = cutlass.Float32(0.0)
        if warp_idx == self.compute_warp_base:
            # The first compute warp is idle here; it initializes the mbarriers so
            # the loader warps are never held behind the init.
            if lane_idx < self.stages:
                cute.arch.mbarrier_init(bar_full + lane_idx, self.load_threads)
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_init(bar_empty + lane_idx, _COMPUTE_WARPS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx < _A_LOADER_WARPS:
            # Weights stream through L2 evict_first so the activation rows every
            # CTA re-reads stay resident. Only tile 0's arrive may follow the init.
            weight_policy = _l2_policy_evict_first()
            empty_phase = cutlass.Int32(1)
            prims.cp_async_mbarrier_arrive(bar_full.llvm_ptr, noinc=True)
            for k_tile in cutlass.range(1, k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                _issue_weight_tile(
                    sA,
                    weight,
                    tidx,
                    output_tile_idx,
                    k_tile,
                    stage,
                    weight_policy,
                    self.output_tile,
                    self.k_tile,
                    True,
                )
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
        elif warp_idx < self.compute_warp_base:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            b_loader_threads = self.b_loader_warps * 32
            local_tid = tidx - _A_LOADER_WARPS * 32
            copy_count = self.token_tile * self.k_tile // (8 * b_loader_threads)
            empty_phase = cutlass.Int32(1)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                for copy_idx in cutlass.range_constexpr(copy_count):
                    linear = local_tid * 8 + copy_idx * b_loader_threads * 8
                    row = linear // self.k_tile
                    col = linear % self.k_tile
                    global_row = token_tile_idx * self.token_tile + row
                    # Out-of-range token rows are discarded by the output predicate.
                    if global_row < cute.size(activation, mode=[0]):
                        dst = (
                            sB.iterator.raw_ptr()
                            + stage * self.token_tile * self.k_tile
                            + row * self.k_tile
                            + (col ^ ((row % 8) * 8))
                        )
                        src = activation.iterator.raw_ptr() + activation.layout(
                            (
                                global_row,
                                k_tile * self.k_tile + col,
                            )
                        )
                        prims.cp_async_shared_global(dst, src, 16, "cg")
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
            if const_expr(self.use_pdl):
                # This grid-level hint is idempotent across B-loader warps.
                cute.arch.griddepcontrol_launch_dependents()
        else:
            if const_expr(self.has_bias):
                # Each reduce lane owns one output feature (32 % output_tile == 0).
                # Issue the load here, after the CTA barrier: waiting on the first
                # full stage hides its latency, whereas a load before the barrier
                # would hold the loader warps behind a cold DRAM read.
                bias_value = bias[
                    output_tile_idx * self.output_tile + lane_idx % self.output_tile
                ].to(cutlass.Float32)
            mma_tid = tidx - self.compute_warp_base * cute.arch.WARP_SIZE
            compute_warp = warp_idx - self.compute_warp_base
            thr_mma = tiled_mma.get_slice(mma_tid)
            acc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self.output_tile, self.token_tile)),
                cutlass.Float32,
            )
            acc.fill(0.0)
            k_phases = self.k_tile // (_COMPUTE_WARPS * _MMA_SHAPE[2])
            m_iters = self.output_tile // _MMA_SHAPE[0]
            n_iters = self.token_tile // _MMA_SHAPE[1]
            # Fragment addresses are loop-invariant per lane apart from the stage
            # base; hoist them so the K loop issues ldmatrix from one add each.
            a_offsets = tuple(
                tuple(
                    _a_fragment_offset(
                        lane_idx, compute_warp, m_iter, phase, self.k_tile
                    )
                    for phase in range(k_phases)
                )
                for m_iter in range(m_iters)
            )
            b_offsets = tuple(
                tuple(
                    _b_pair_offset(
                        lane_idx,
                        compute_warp,
                        n_iter,
                        pair,
                        self.token_tile,
                        self.k_tile,
                    )
                    for pair in range(k_phases // 2)
                )
                for n_iter in range(n_iters)
            )
            a_stage_elems = self.output_tile * self.k_tile
            b_stage_elems = self.token_tile * self.k_tile
            full_phase = cutlass.Int32(0)
            # The K-tile count is static: unrolling fully folds `stage`, barrier
            # addresses and phases to constants, so every ldmatrix is
            # `[R_lane + imm]` (a rolled loop costs 3-16% on K=7168).
            for k_tile in cutlass.range(k_tile_count, unroll_full=True):
                stage = k_tile % self.stages
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
                a_base = stage * a_stage_elems
                b_base = stage * b_stage_elems
                a_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sA, a_base + a_offsets[m_iter][phase])
                        for phase in range(k_phases)
                    )
                    for m_iter in range(m_iters)
                )
                b_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sB, b_base + b_offsets[n_iter][pair])
                        for pair in range(k_phases // 2)
                    )
                    for n_iter in range(n_iters)
                )
                for phase in cutlass.range_constexpr(k_phases):
                    for n_iter in cutlass.range_constexpr(n_iters):
                        packed_b = b_registers[n_iter][phase // 2]
                        b_offset = (phase % 2) * 2
                        for m_iter in cutlass.range_constexpr(m_iters):
                            acc_base = (m_iter * n_iters + n_iter) * 4
                            result = prims.mma_sync(
                                llvm.StructType.get_literal(
                                    [cutlass.Float32.mlir_type] * 4
                                ),
                                shape=_MMA_SHAPE,
                                layout_a=prims.MMALayout.ROW,
                                layout_b=prims.MMALayout.COL,
                                operand_a=[
                                    a_registers[m_iter][phase][i].ir_value()
                                    for i in range(4)
                                ],
                                operand_b=[
                                    packed_b[b_offset + i].ir_value() for i in range(2)
                                ],
                                operand_c=[
                                    acc[acc_base + i].ir_value() for i in range(4)
                                ],
                                multiplicand_a_ptx_type=prims.MMAType.BF16,
                                multiplicand_b_ptx_type=prims.MMAType.BF16,
                            )
                            for i in cutlass.range_constexpr(4):
                                acc[acc_base + i] = cutlass.Float32(
                                    llvm.extractvalue(
                                        cutlass.Float32.mlir_type,
                                        result,
                                        [i],
                                    )
                                )
                if const_expr(reuse_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_empty + stage)
                if stage == self.stages - 1:
                    full_phase = full_phase ^ 1
            for value in cutlass.range_constexpr(values_per_lane):
                atom = value % 4
                fragment = value // 4
                m_iter = fragment // n_iters
                n_iter = fragment % n_iters
                feature = lane_idx // 4 + 8 * (atom // 2) + m_iter * _MMA_SHAPE[0]
                token = 8 * n_iter + 2 * (lane_idx % 4) + atom % 2
                partials[
                    compute_warp * partial_stride
                    + _partial_offset(feature, token, self.token_tile)
                ] = acc[value]
            prims.barrier_cta_sync(1, thread_count=_COMPUTE_WARPS * 32)

        if warp_idx == self.compute_warp_base:
            # The peer store reduces and stores its own 16 B chunks below instead.
            reduced_values = values_per_lane if self.peer_store is None else 0
            final_acc = cute.make_rmem_tensor(
                cute.make_layout(values_per_lane), cutlass.Float32
            )
            for value in cutlass.range_constexpr(reduced_values):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                total = cutlass.Float32(0)
                for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                    total = (
                        total
                        + partials[
                            peer * partial_stride
                            + _partial_offset(feature, token, self.token_tile)
                        ]
                    )
                if const_expr(self.has_bias):
                    total = total + bias_value
                final_acc[value] = total

            output_base = output_tile_idx * self.output_tile
            token_base = token_tile_idx * self.token_tile
            for value in cutlass.range_constexpr(reduced_values):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                if token_base + token < cute.size(output, mode=[0]):
                    output[token_base + token, output_base + feature] = final_acc[
                        value
                    ].to(output.element_type)
            if const_expr(self.peer_store is not None):
                peer_store = bind_peer_store_args(self.peer_store, peer_args)
                # Ordered after the B loader's grid-dependency wait by the mainloop barriers.
                parity = load_peer_epoch_parity(peer_store)
                m = cute.size(activation, mode=[0])
                n = cute.size(weight, mode=[0])
                feat_chunks = self.output_tile // 8
                chunks = feat_chunks * self.token_tile
                for c_iter in cutlass.range_constexpr(
                    (chunks + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
                ):
                    # Each lane owns one 16 B chunk (8 features of one token).
                    chunk = c_iter * cute.arch.WARP_SIZE + lane_idx
                    feature = chunk % feat_chunks * 8
                    row = token_base + chunk // feat_chunks
                    if chunk < chunks:
                        if row < m:
                            # (1) Reduce the warps' partials, in the same order as above.
                            values = cute.make_rmem_tensor(
                                cute.make_layout(8), cutlass.BFloat16
                            )
                            for j in cutlass.range_constexpr(8):
                                chunk_total = cutlass.Float32(0)
                                for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                                    chunk_total = (
                                        chunk_total
                                        + partials[
                                            peer * partial_stride
                                            + _partial_offset(
                                                feature + j,
                                                chunk // feat_chunks,
                                                self.token_tile,
                                            )
                                        ]
                                    )
                                values[j] = chunk_total.to(cutlass.BFloat16)
                            # (2) Replace zeros with the sentinel.
                            words = replace_zero_sentinel(
                                cute.recast_tensor(values, cutlass.Uint32)
                            )
                            # (3) Store to every peer.
                            store_peer_chunk(
                                peer_store,
                                parity,
                                row,
                                output_base + feature,
                                m,
                                n,
                                words,
                            )


def _from_dlpack(tensor: _torch.Tensor, *, dynamic_m: bool = False):
    tensor = from_dlpack(tensor.detach(), assumed_align=32)
    if dynamic_m:
        # Only M varies; the row stride and N/K extents stay static.
        tensor = tensor.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
    return tensor


@functools.cache
def _compile(
    device_index: int,
    dtype,
    n: int,
    k: int,
    tactic: WarpSplitKTactic,
    use_pdl: bool,
    has_bias: bool,
):
    with _torch.cuda.device(device_index):
        return cute_ext.compile(
            CpAsyncWarpSplitKKernel(tactic, use_pdl, has_bias),
            *(
                _from_dlpack(
                    _torch.empty(shape, device="cuda", dtype=dtype),
                    dynamic_m=dynamic_m,
                )
                for shape, dynamic_m in (
                    ((n, k), False),
                    ((_MAX_M, k), True),
                    ((n,), False),
                    ((_MAX_M, n), True),
                )
            ),
            _cuda.CUstream(_torch.cuda.current_stream().cuda_stream),
        )


def run_warp_splitk_dense(a, b, out, pdl: bool, tactic: WarpSplitKTactic, bias=None):
    m, n, k = validate_inputs(a, b, out, bias)
    validate_tactic(tactic, m, n, k)
    device_index = a.device.index
    assert device_index is not None
    with _torch.cuda.device(a.device):
        compiled = _compile(device_index, a.dtype, n, k, tactic, pdl, bias is not None)
        # Without bias the kernel never reads its bias argument; pass a row of
        # ``out`` so the launch signature stays fixed.
        compiled(
            _from_dlpack(b.T),
            _from_dlpack(a, dynamic_m=True),
            _from_dlpack(bias if bias is not None else out[0]),
            _from_dlpack(out, dynamic_m=True),
            _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream),
        )
    return out


def run_warp_splitk_dense_peer_store(
    a,
    b,
    out,
    tactic: WarpSplitKTactic,
    *,
    peer_ptrs=None,
    multicast_ptr=None,
    epoch_ptr: int,
    slot_bytes: int,
    rank: int,
    world_size: int,
    layout: str = "broadcast",
    pdl: bool = True,
) -> None:
    """``run_warp_splitk_dense`` (no bias) storing into peer buffers (``LamportPeerStoreConfig``); ``out`` is not written."""
    m, n, k = validate_inputs(a, b, out)
    validate_tactic(tactic, m, n, k)
    peer_store = make_lamport_peer_store_config(
        peer_ptrs=peer_ptrs,
        multicast_ptr=multicast_ptr,
        epoch_ptr=epoch_ptr,
        slot_bytes=slot_bytes,
        rank=rank,
        world_size=world_size,
        layout=layout,
        m=m,
        n=n,
    )
    peer_store, peer_args = split_peer_store_config(peer_store)
    device_index = a.device.index
    assert device_index is not None
    key = (device_index, a.dtype, n, k, tactic, pdl, peer_store)
    with _torch.cuda.device(a.device):
        args = (
            _from_dlpack(b.T),
            _from_dlpack(a, dynamic_m=True),
            _from_dlpack(out[0]),
            _from_dlpack(out, dynamic_m=True),
            _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream),
            peer_args,
        )
        if key not in _PEER_STORE_KERNELS:
            kernel = CpAsyncWarpSplitKKernel(tactic, pdl, False, peer_store)
            _PEER_STORE_KERNELS[key] = cute_ext.compile(kernel, *args)
        _PEER_STORE_KERNELS[key](*args)


__all__ = [
    "LamportPeerStoreConfig",
    "WarpSplitKTactic",
    "autotune_tactics",
    "bind_peer_store_args",
    "default_tactic",
    "make_lamport_peer_store_config",
    "run_warp_splitk_dense",
    "run_warp_splitk_dense_peer_store",
    "split_peer_store_config",
    "store_tile_to_peers",
    "validate_inputs",
]
