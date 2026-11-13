"""Select SM120 tile shapes, launch budgets, and role residency."""
from dataclasses import dataclass
from functools import lru_cache
from math import isqrt

from ...impls.sm120_fp8_fp4_gc_moe import (
    MMA_SF_STAGE_BYTES, MMA_TILE_K, MMA_TILE_M, RELAY_SMEM_BYTES,
    shared_gemm_stage_bytes,
)
from ...layout.sm120_gc_moe import BATCHED_MIN_ROWS_PER_EXPERT


def _role2_smem(tile_n, stages, slots, compact_w, gmem_handoff, shared_chunks):
    per_w = MMA_TILE_M * MMA_TILE_K // (2 if compact_w else 1) + MMA_SF_STAGE_BYTES
    per_x = tile_n * MMA_TILE_K + MMA_SF_STAGE_BYTES
    # BF16 return tiles, destination addresses, and route metadata. The
    # utility warp overlays its FP32 FC1 handoff on the same storage.
    return_bytes = tile_n * (20 + 128 * slots)
    utility_bytes = 32 + 24 * slots + 16 * slots * tile_n
    barrier_bytes = 16 * (4 * stages + 4)
    if gmem_handoff:
        # Global source tiles, optionally replaced by shared 32-row chunks.
        return_bytes += 16 - 128 * slots * tile_n
        if shared_chunks:
            return_bytes += 128 * slots * 32 - 16
        if tile_n == 32:
            return_bytes -= 32  # Preserve the 33,024 B rounded allocation.
        barrier_bytes -= (16 * (4 * stages + 3 + 3 * slots)
                          - 8 * (4 * stages + 2 + 4 * slots))
    return (stages * (per_w + per_x) + tile_n * 16 + 16 + barrier_bytes
            + return_bytes + utility_bytes + 512)


def _select_tile_n(rows: int, bf16_handoff: bool) -> int:
    """Choose the swap-AB token tile from the expected rows per expert."""
    if bf16_handoff:
        if 32 <= rows < 48:
            return 48
        if 16 <= rows < 32:
            return 32
        if 64 <= rows < 96:
            # Allow for uneven routing, as in the SM100 token tile selector.
            return 80 if rows + isqrt(rows) <= 80 else 96
    return 16 if rows < 32 else 64


@dataclass(frozen=True)
class SM120RoleConfig:
    grids: tuple[int, int, int]
    threads: tuple[int, int, int]
    smem: tuple[int, int, int]
    tile_n: int
    mma_stages: int
    reg_cap: int
    l2_dedicated: int
    combine_slots: int
    gmem_handoff: bool
    compact_w: bool
    shared_chunks: bool
    bf16_handoff: bool
    batched_returns: bool
    relay_ctas: int
    mirror_metadata: bool
    pull_engine: bool
    dispatch_cache: bool
    shared_tile_n: int


@lru_cache(maxsize=256)
def _get_sm120_role_config(config, layout, dispatch_sms: int, compute_sms: int,
                           smem_per_sm: int, reserved_smem_per_block: int = 0):
    direct_scan = 0 < config.bound <= 2
    combine_bound = config.bound or config.capacity
    # All finalize CTAs can wait for block 0 to publish peer completion.
    combine_grid = min(compute_sms * 2,
                       max(1, (combine_bound * config.hidden + 2047) // 2048))
    # The return schedule depends only on the common bound and workspace geometry.
    batched_returns = (not direct_scan and 'batched_routes_in' in layout.views
                   and config.bound * config.topk >= BATCHED_MIN_ROWS_PER_EXPERT * layout.local)
    mirror_metadata = batched_returns and 'batched_scales_in' in layout.views
    relay = (batched_returns and 'relay_output' in layout.views
             and (not mirror_metadata or config.shared_experts > 0))
    tokens = config.bound if batched_returns else config.tokens
    rows = (tokens * config.topk + layout.local - 1) // layout.local
    bf16_handoff = batched_returns or (config.ranks > 1 and config.tokens >= 32)
    tile_n = _select_tile_n(rows, bf16_handoff)
    stages = 4 if tile_n == 16 else 2
    gmem_handoff = bf16_handoff and tile_n >= 32
    compact_w = tile_n in (48, 80, 96)
    shared_chunks = gmem_handoff and tile_n >= 48
    slots = 2 if gmem_handoff else 1
    if batched_returns:
        combine_grid = min(config.capacity, compute_sms * 4)
    role2_smem = _role2_smem(tile_n, stages, slots, compact_w, gmem_handoff, shared_chunks)
    shared_tile_n = 0
    if config.shared_experts and config.tokens:
        shared_tile_n = (8 if config.tokens <= 8 else 16 if config.tokens <= 16
                         else 32 if config.tokens <= 32 else 64)
        stage_bytes = shared_gemm_stage_bytes(shared_tile_n)
        role2_smem = max(role2_smem, 2 * stage_bytes + 16)
    if relay:
        role2_smem = max(role2_smem, RELAY_SMEM_BYTES)

    # Every dispatch CTA must be resident for its grid-wide rendezvous.
    # Streaming pull gives each warp two activation row slots and barriers.
    pull_smem = 512 + 16 * config.hidden + 64
    pull_allocation = (pull_smem + 255) // 256 * 256 + reserved_smem_per_block
    pull_engine = (not direct_scan and config.hidden % 512 == 0
                   and pull_allocation <= smem_per_sm)
    role1_smem = pull_smem if pull_engine else 512
    role1_allocation = (role1_smem + 255) // 256 * 256 + reserved_smem_per_block
    dispatch_ctas = dispatch_sms * max(1, min(4, smem_per_sm // role1_allocation))
    if direct_scan:
        dispatch_ctas = min(config.experts // config.ranks, dispatch_ctas)

    rounded_smem = (role2_smem + 255) // 256 * 256
    ctas = max(1, min(4, smem_per_sm // (rounded_smem + reserved_smem_per_block)))
    threads = 192  # Four math warps, one TMA producer, and one utility warp.
    # Match the register allocation granularity when enforcing co-residency.
    reg_cap = ((65536 // (threads * ctas)) & ~15) if ctas >= 2 else 0
    l2_dedicated = 1
    if tile_n >= 64 or gmem_handoff:
        l2_dedicated = 6 if tile_n == 80 and not batched_returns else 4
    if batched_returns and not relay and config.shared_experts == 0:
        l2_dedicated = 3
    compute_grid = compute_sms * ctas
    if gmem_handoff and not shared_chunks:
        compute_grid = min(compute_grid, layout.views['handoff'].nbytes // (16 * 1024))
    # Leave a full FC1/FC2 scheduling period even when copiers must serve
    # multiple peers on a small compute partition.
    if relay and compute_grid <= l2_dedicated:
        raise ValueError('SM120 compute partition cannot accommodate return copiers')
    relay_ctas = min(config.ranks, compute_grid - l2_dedicated) if relay else 0
    cache_bytes = config.ranks * config.capacity * 8
    # Shared handoff leaves the global pool free for immutable input rows.
    dispatch_cache = (pull_engine and shared_chunks and config.topk >= config.ranks
                      and (tile_n == 80 or (batched_returns and tile_n == 48))
                      and cache_bytes <= layout.views['handoff'].nbytes)
    return SM120RoleConfig(
        grids=(dispatch_ctas, compute_grid, combine_grid),
        threads=(256, threads, 256 if batched_returns else 384),
        smem=(role1_smem, role2_smem, max(64, 4 * config.topk) if batched_returns else 0),
        tile_n=tile_n, mma_stages=stages, reg_cap=reg_cap,
        l2_dedicated=l2_dedicated, combine_slots=slots,
        gmem_handoff=gmem_handoff, compact_w=compact_w, shared_chunks=shared_chunks,
        bf16_handoff=bf16_handoff, batched_returns=batched_returns,
        relay_ctas=relay_ctas, mirror_metadata=mirror_metadata,
        pull_engine=pull_engine, dispatch_cache=dispatch_cache,
        shared_tile_n=shared_tile_n,
    )


def get_sm120_role_config(config, context, layout):
    return _get_sm120_role_config(config, layout, context.dispatch_sms, context.compute_sms,
                                  context.smem_per_sm,
                                  context.reserved_smem_per_block)
