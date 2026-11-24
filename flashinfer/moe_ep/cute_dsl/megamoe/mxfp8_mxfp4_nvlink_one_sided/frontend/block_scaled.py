# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Workspace, descriptor construction, compilation and launch for Blackwell GenPhase.

The Blackwell frontend supplies native device checks and compiler targets.
The private source package is imported only inside this shim; DSL imports stay lazy.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal, Optional, Tuple, Union

import torch

from flashinfer.moe_ep._compat import deprecated_aliases

from . import comm
from .kernel_helpers import (
    Mxfp8BlockSize,
    Nvfp4BlockSize,
    default_weight_layout,
    round_up,
    swizzled_flat_sf_size,
)

BlockScaledQuantKind = Literal["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"]
BlockScaledTokenBackMode = Literal[
    "epi_warps", "standalone_warps", "reuse_dispatch_warps"
]
BlockScaledWorkIdMode = Literal["grid_stride", "atomic_counter"]
BlockScaledScheduleMode = Literal["grouped", "phase_interleave"]

# (act data dtype, act SF dtype, sf_vec_size, 2x-mode instruction K) per kind.
# The mxf8f6f4 UMMA path has instruction K 32 (64 in 2x mode); the fp4 pair
# kinds have 64 (128 in 2x mode) — see quant_def.QuantKind.instruction_k.
_KIND_TABLE: dict = {
    "nvfp4": (None, torch.float8_e4m3fn, Nvfp4BlockSize, 128),
    "mxfp8_e4m3": (torch.float8_e4m3fn, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
    "mxfp8_e5m2": (torch.float8_e5m2, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
    "mxfp4_mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
}

TransformedBlockScaledWeights = Tuple[torch.Tensor, torch.Tensor]


def _device_has_two_locality_domains() -> bool:
    """Whether the current CUDA device's memory forms two locality domains (one per die)."""
    if not torch.cuda.is_available():
        return False
    from .die_local import locality_domain_count

    return locality_domain_count() >= 2


def _fp4_storage_dtype() -> torch.dtype:
    return getattr(torch, "float4_e2m1fn_x2", torch.uint8)


def _to_cute(
    tensor: torch.Tensor, assumed_align: int = 16, dynamic_leading: bool = True
):
    """torch -> cute tensor via dlpack; leading dim marked dynamic by default."""
    import cutlass.torch as cutlass_torch

    cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    if not dynamic_leading:
        return cute_tensor
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 128):
    """Raw cute ``Uint8`` gmem pointer at ``tensor``'s base (opaque workspaces)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import AddressSpace

    return cute.runtime.make_ptr(
        cutlass.Uint8, tensor.data_ptr(), AddressSpace.gmem, assumed_align=assumed_align
    )


def _cu_stream(stream: torch.cuda.Stream):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(stream.cuda_stream)


_MAX_ACTIVE_CLUSTERS_CACHE: dict = {}

# Padded capacity (tokens per rank) up to which the Sentinel dispatch compiles
# just-in-time work claims by default. Every launch of such a kernel picks eager
# or just-in-time claims from its own tile counts; past this capacity the kernel
# keeps eager claims and no runtime choice.
JUST_IN_TIME_CLAIMS_MAX_TOKENS = 16

# Under the counter combine the end-of-round reduce drops every combine slot line
# it has consumed from L2 by default once a round's combine slots take at least
# this many bytes, about two thirds of a B300's 126.5 MiB L2. Larger slots no
# longer fit in L2 beside the reduce's own reads, so dirty slot lines would be
# written back while the reduce runs; smaller ones mostly stay in L2 through it,
# and the reduce would only pay for the drops.
DISCARD_COMBINE_SLOTS_MIN_BYTES = 80 * 1024 * 1024

# FC1 weight bytes the main kernel requests into L2 at entry, while the dispatched
# payload lands (``entry_prefetch_bytes``). Under the counter dispatch in one round
# every FC1 tile waits for the whole payload, which at up to
# WHOLE_PAYLOAD_ENTRY_PREFETCH_MAX_TOKENS tokens per rank keeps HBM mostly idle for
# about 25 us after entry on eight GPUs; there the prefetch takes about the L2's
# capacity (126.5 MiB on B300). Its allocations also evict the dirty lines earlier
# kernels left in L2 while HBM has time for their write-back, which during the
# grouped GEMMs slows the weight stream by about 6%: V4.1-Flash at 256 tokens per
# rank gains about 4 us on eight GPUs (nothing with a clean L2). The Sentinel
# dispatch releases FC1 per expert and the pull dispatch per token tile, after 10
# to 20 us, where a larger prefetch gains nothing or loses, except under the
# Sentinel dispatch past SENTINEL_DISPATCH_ENTRY_PREFETCH_MIN_TOKENS tokens per rank:
# its payload then lands long enough for the larger prefetch to pay as well
# (V4.1-Flash at 256 tokens per rank on eight GPUs: 2.6 us; at 64 and 128 about
# 0.5 us).
ENTRY_PREFETCH_BYTES = 64 * 1024 * 1024
WHOLE_PAYLOAD_ENTRY_PREFETCH_BYTES = 128 * 1024 * 1024
WHOLE_PAYLOAD_ENTRY_PREFETCH_MAX_TOKENS = 256
SENTINEL_DISPATCH_ENTRY_PREFETCH_MIN_TOKENS = 129

# Expected routes per local expert from which die-local weights are on by
# default (``die_local_weights``). From there each weight M tile feeds a token
# tile of 64 rows or more, the grouped GEMMs wait on the weight stream's latency
# rather than on HBM's rate, and on one GPU at large loads they run at the power
# cap; die-matched reads shorten both: V4.1-Flash gains 4-12 us per layer at
# 64-512 routes on one GPU and 2-6.5 us on eight, where the incoming NVLink
# bounds the largest loads. Below it the weight stream runs at about 6.3 TB/s
# with either placement, short of both placements' streaming ceilings, and the
# layer gains at most about 2 us.
DIE_LOCAL_WEIGHTS_MIN_EXPECTED_LOAD = 64

# Token tile N and expected routes per local expert from which the main kernel
# issues FC1's Gather4 token loads from eight Gather-B warps instead of four by
# default. A warp issues its four-row groups one at a time, and from N192 each
# MMA CTA gathers 88 to 120 rows per AB stage, about one Gather4 per 16
# tensor-pipe cycles: four warps then issue a stage's rows more slowly than the
# MMA consumes them. Below an expected load of 256 routes an expert has one token
# tile, and eight warps measured 0.3 to 0.8 us slower end to end (EP8, 1024
# tokens per rank).
EIGHT_GATHER_B_WARPS_MIN_TILE_N = 192
EIGHT_GATHER_B_WARPS_MIN_EXPECTED_LOAD = 256

# Expected routes per local expert from which the pull dispatch stores rows into
# the token pool and FC1 loads its tokens by tiled TMA (``fc1_token_pool``)
# instead of gathering them. From there an expert spans two or more token tiles,
# whose MMAs re-read its weights from L2 and consume FC1's token operand faster
# than Gather4 delivers it even from eight Gather-B warps: V4.1-Flash on eight
# GPUs gains 6.7 / 6.3 us at 2048 / 4096 tokens per rank. With one token tile
# per expert four Gather-B warps keep up (-0.5 / +0.6 us at 512 / 1024), and the
# pool, one row per route against one landing row per source token, would only
# cost memory.
FC1_TOKEN_POOL_MIN_EXPECTED_LOAD = 256

# Rows of the token pool by default, as a multiple of the routes a rank receives
# under balanced routing (tokens per rank times top-k). The pool holds one row
# per route in processing order, and the token tiles past it fall back to
# Gather4 from the landing slots. At eight ranks and 2048 or 4096 tokens per
# rank the most loaded rank of V4.1-Flash receives 1.0 times the balanced routes
# under perfect routing, at most 1.02 under random routing and 1.18 under skewed
# routing (expert popularity falling off geometrically, the same order on every
# rank).
# Twice that keeps every token tile of these in the pool, with room for the
# imbalance of real routing, at about a sixth of the pool sized for any routing.
FC1_TOKEN_POOL_CAPACITY_FACTOR = 2

# Token stages of the main kernel's split AB ring, per token tile N, where the
# largest expected expert (``largest_expected_expert_load``) fits one token tile;
# the weight ring takes the rest of the SMEM budget (9 weight stages at N128, 8 at
# N192, against 8 and 7 in one shared ring). With one token tile per expert every
# weight byte is read once from HBM, and the weight stream needs the deeper ring;
# once an expert has several token tiles they re-read its weights from L2 and the
# token ring needs its depth back, so the main kernel keeps the shared ring there
# (V4.1-Flash on eight GPUs runs N192 at 2048 and 4096 tokens per rank on the
# shared 7-stage ring). One token stage fewer still pays at these tiles, two do
# not.
SPLIT_RING_TOKEN_STAGES = {128: 7, 192: 6}


def largest_expected_expert_load(expected_load: float) -> float:
    """The token tile policies' estimate of the largest local expert's routes at an expected load."""
    return expected_load + 1.6 * math.sqrt(expected_load)


# Token tile N of the main kernel's alternate configuration under a strided
# token tile (N192, N240); see ``alternate_token_tile_n``.
ALTERNATE_TOKEN_TILE_N = 128


def _strided_token_tile(tile_n: int) -> bool:
    """Whether token tile N lays its tiles out in slots of whole 128-row scale-factor atoms (N192, N240)."""
    return tile_n % 128 != 0 and 128 % tile_n != 0


def balanced_token_tile_rows(load: int, tile_n: int) -> int:
    """Routes in each but the last of an expert's token tiles under a strided tile N, host side.

    The main kernel's ``balanced_tile_rows``: ceil(load / tile_n) tiles, each but
    the last the even share rounded up to 16 routes unless that leaves the last
    tile empty; 0 for an expert without routes.
    """
    tiles = max(-(-load // tile_n), 1)
    even_share = -(-load // tiles)
    rounded_share = -(-even_share // 16) * 16
    return rounded_share if (tiles - 1) * rounded_share < load else even_share


def chosen_token_tile_n(expert_loads, tile_n: int, alternate_tile_n: int) -> int:
    """The token tile N a launch runs given its local experts' routes: the main kernel's choice rule, host side.

    Both configurations run the plan's token tiles of ``tile_n``; the launch takes
    the alternate exactly when every one of them fits it, that is when no expert's
    first (largest) tile has more than ``alternate_tile_n`` routes.
    """
    if not alternate_tile_n:
        return tile_n
    largest = max(
        (balanced_token_tile_rows(int(load), tile_n) for load in expert_loads),
        default=0,
    )
    return alternate_tile_n if largest <= alternate_tile_n else tile_n


def _max_active_clusters(cluster_size: int) -> int:
    """Occupancy probe, cached per device and cluster size."""
    key = (torch.cuda.current_device(), cluster_size)
    cached = _MAX_ACTIVE_CLUSTERS_CACHE.get(key)
    if cached is None:
        import cutlass.utils as cutlass_utils

        cached = int(cutlass_utils.HardwareInfo().get_max_active_clusters(cluster_size))
        _MAX_ACTIVE_CLUSTERS_CACHE[key] = cached
    return cached


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
@dataclasses.dataclass(frozen=True)
class BlockScaledMoeConfig:
    """Validated construction parameters for block-scaled inference and GenPhase."""

    num_total_experts: int
    max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate: int  # width after activation; the FC1 GEMM N is 2*intermediate
    rank: int
    world_size: int
    quant_kind: BlockScaledQuantKind = "mxfp8_e4m3"
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None  # None -> (256, 128, 2*ik)
    cluster_shape_mn: Tuple[int, int] = (2, 1)
    # Mixed-CGA launch: alongside the preferred clusters, fill leftover SMs with
    # smaller fallback clusters (upstream commit a5b4d33). None = uniform launch.
    fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None
    # ("grouped" | "phase_interleave", optional hint); typed loosely so backend
    # configs can carry it as a plain (str, int|None) tuple.
    schedule_policy: Tuple[str, Optional[int]] = ("grouped", None)
    # Grouped scheduling: claim each schedule group's FC2 tiles together with the
    # next group's FC1 tiles instead of right after its own, spreading the combine
    # pushes over the FC1 work.
    pipelined_groups: bool = False
    work_id_mode: BlockScaledWorkIdMode = "grid_stride"
    fc2_use_bulk: bool = False
    fc2_tma_stages: Optional[int] = None
    # GenPhase: seed the gather index plane so padded tile rows gather distinct
    # dense rows instead of all aliasing row 0.
    gather_padding_spread: bool = True
    # L2 eviction priority of weight loads: "normal", "first" or "adaptive"
    # (evict-first only for an expert's last, partial token tile).
    weight_evict: Literal["normal", "first", "adaptive"] = "adaptive"
    # Combine completion: "sentinel" (data-as-flag slots the reduce resets) or
    # "counter" (bulk pushes published by one all-rank barrier at the tail).
    combine_protocol: Literal["sentinel", "counter"] = "sentinel"
    # Dispatch completion: "sentinel" (0xFF-armed landing rows verified byte by
    # byte), "counter" (per-peer payload counters after a system fence) or
    # "pull" (each target rank fetches its rows from the source ranks' staged
    # activations in its own processing order; needs the counter combine).
    dispatch_protocol: Literal["sentinel", "counter", "pull"] = "sentinel"
    # Pull dispatch only: data-row ring slots per walking warp, routes per work
    # claim, and with walking CTAs routes per claim of the experts that start
    # within the first wave of claims, one claim per walking warp (None: the
    # kernel defaults).
    comm_pull_depth: Optional[int] = None
    comm_pull_chunk_rows: Optional[int] = None
    comm_pull_lead_rows: Optional[int] = None
    # Counter and pull dispatches only: communication CTAs that each sort one
    # chunk of this rank's routes into per-destination lists (None: the kernel
    # default, at least one per rank).
    comm_exchange_ctas: Optional[int] = None
    # Pull dispatch only: warps of every main-kernel CTA that fetch the routed
    # rows (1 to 4), or 0 to fetch them in walking CTAs of the communication grid,
    # which hold their SMs until the fetch ends (None: the kernel default).
    pull_warps: Optional[int] = None
    # Warps of every main-kernel CTA that issue FC1's Gather4 token loads, 4 or 8
    # (None: eight from token tile EIGHT_GATHER_B_WARPS_MIN_TILE_N and an expected
    # expert load of EIGHT_GATHER_B_WARPS_MIN_EXPECTED_LOAD routes at cluster
    # (2, 1) without pull warps, otherwise the kernel's four, or two at N64 on
    # cluster (4, 1)). Eight take the fourth warpgroup, which the pull warps also
    # use.
    gather_b_warps: Optional[int] = None
    # Pull dispatch with walking CTAs: the index-copy CTAs acquire the source
    # ranks' expert-id releases before they count into the index lists'
    # completion, which the walking CTAs then wait for alone (None: the kernel
    # default, on; False: every walking CTA acquires them itself after its
    # index copy). Other dispatches ignore it.
    pull_acquire_in_copy: Optional[bool] = None
    # Pull dispatch with walking CTAs: claim each route's landing slot and fetch
    # a row past a claim's first few only if its slot was won, so a source token
    # travels about once per target rank rather than once per route (None or
    # False: every route is fetched). Other dispatches ignore it.
    pull_walk_dedup: Optional[bool] = None
    # Pull dispatch with walking CTAs: count landed routes per token tile and gate
    # each FC1 tile on its own tile rather than on its whole expert (None: the
    # kernel default). Ignored at token tile N64, whose scale-factor atoms span
    # two tiles, and by other dispatches.
    pull_tile_ready: Optional[bool] = None
    # Pull dispatch with walking CTAs: the walk stores each route's row at its
    # expert-major pool row instead of its landing slot, and the main kernel loads
    # FC1's tokens from there with tiled TMA loads, as it loads FC2's, instead of
    # gathering them from the landing slots by index (TMA Gather4); the landing
    # slots are then not allocated, the pool holds one row per route (None: from
    # FC1_TOKEN_POOL_MIN_EXPECTED_LOAD expected routes per local expert, see
    # ``resolved_fc1_token_pool``). Needs the walking CTAs (no pull_warps), every
    # route fetched (no pull_walk_dedup) and the default Gather-B warps.
    fc1_token_pool: Optional[bool] = None
    # Rows of the token pool, one per route of a launch in processing order (None:
    # FC1_TOKEN_POOL_CAPACITY_FACTOR times the tokens per rank times top-k). The
    # token tiles past it fall back to Gather4 from the landing slots, which then
    # stay allocated; per tile, or with ``fc1_token_pool_fallback`` "launch" for
    # the whole launch whenever its routes do not all fit. 0, or a capacity that
    # reaches the data rows of the worst case, sizes the pool for any routing
    # without landing slots. Applies under a strided token tile (N192, N240);
    # other tiles keep the pool sized for any routing.
    fc1_token_pool_capacity: Optional[int] = None
    fc1_token_pool_fallback: Literal["tile", "launch"] = "tile"
    # Pusher CTAs of the communication grid (None: the kernel default, which
    # depends on the dispatch protocol; under the pull dispatch they fetch, or
    # with pull warps only copy index lists). Each one holds an SM that a
    # main-kernel cluster waits for until it exits.
    comm_pusher_ctas: Optional[int] = None
    # Counter dispatch and combine only: split every rank's local experts into
    # this many rounds (1, 2, 4 or 8). Tokens travel and are published round by
    # round and each expert's FC1 waits for its own round instead of the whole
    # payload.
    dispatch_rounds: int = 1
    # Per-warp payload staging slots of the pushers (None: the kernel default).
    comm_push_depth: Optional[int] = None
    # Rounds only: exchange CTAs that join the scale-factor refine after their
    # sort (None: all of them).
    refine_exchange_ctas: Optional[int] = None
    # Rounds diagnostics: "order" lays experts out round-major only, "push" also
    # pushes by round behind the whole-payload wait, "gate" is the protocol.
    dispatch_round_mode: Literal["order", "push", "gate"] = "gate"
    # Rounds only: bulk groups a pusher warp commits after a round closes before
    # publishing it (None: the kernel default).
    round_publish_lag: Optional[int] = None
    # Rounds fault injection: pushers hold rounds >= 1 back by this long.
    round_publish_delay_ns: int = 0
    # Rounds negative control: gate every expert on round 0 only (incorrect).
    refine_gate_round0_only: bool = False
    # Diagnostic %globaltimer stamps in a local workspace region.
    comm_stamps: bool = False
    # Physical weight layout the kernel addresses; it must match the layout the
    # weights were prepared in (``weight_layout`` of the preprocess functions).
    # "blockmajor" stores every (128 rows x 128 K) block contiguously, so each
    # TMA stage box is one contiguous run. None: ``default_weight_layout``.
    weight_layout: Optional[Literal["kmajor", "blockmajor"]] = None
    # FC1 weight bytes prefetched into L2, in processing order, while the
    # dispatched payload is still arriving (0 disables; None:
    # ``resolved_entry_prefetch_bytes``).
    entry_prefetch_bytes: Optional[int] = None
    # GenPhase: split the main kernel's AB stage ring into a weight ring and a
    # token ring, the token ring this many stages deep and the weight ring the
    # rest of the SMEM budget; 0 keeps one shared ring (None: the
    # SPLIT_RING_TOKEN_STAGES of the token tile where the largest expected
    # expert fits one token tile, else one shared ring).
    token_ab_stages: Optional[int] = None
    # GenPhase under a strided token tile (N192, N240): a second configuration of
    # the main kernel's mainloop and epilogue at this token tile N, which runs the
    # same plan's token tiles while each of them fits it; every launch picks one
    # once its expert sizes are known (``token_tile_choice``). 0 compiles none
    # (None: ALTERNATE_TOKEN_TILE_N under a strided tile at cluster (2, 1)).
    alternate_token_tile_n: Optional[int] = None
    # Token stages of the alternate tile's split AB ring, or 0 for one shared ring
    # (None: SPLIT_RING_TOKEN_STAGES of the alternate tile where the largest
    # expected expert fits one token tile of the plan, else one shared ring).
    alternate_token_ab_stages: Optional[int] = None
    # L2 eviction priority of the alternate tile's weight loads (None: weight_evict).
    alternate_weight_evict: Optional[Literal["normal", "first", "adaptive"]] = None
    # How a launch with an alternate token tile picks its configuration: "rule"
    # takes the alternate exactly when every token tile of its plan fits it,
    # "primary" keeps the plan's own tile N in the kernel that carries both.
    token_tile_choice: Literal["rule", "primary"] = "rule"
    # GenPhase on a device with two locality domains (B300's two dies): the first
    # and second half of every weight matrix's M tiles live in the HBM of one die
    # each, and a cluster claims the tiles of its own die's half from one work
    # counter per die (``die_local.py``). The launch paths convert weights
    # prepared without it once and keep the copy with them. None: on from
    # DIE_LOCAL_WEIGHTS_MIN_EXPECTED_LOAD expected routes per local expert where
    # the configuration admits it (``_admits_die_local_weights``) and the current
    # device has two locality domains.
    die_local_weights: Optional[bool] = None
    # Compile just-in-time work claims, under which a launch whose tile counts
    # favour them claims a cluster's next tile only once the current one has passed
    # its readiness gate (None: the Sentinel dispatch up to
    # JUST_IN_TIME_CLAIMS_MAX_TOKENS padded tokens per rank).
    just_in_time_claims: Optional[bool] = None
    # Just-in-time claims only: weight K tiles a work tile still has to issue when
    # its cluster claims the next one (None: the kernel default). A lead shorter
    # than a claim, about 1.3 us at the median while the SM streams weights, idles
    # every SM's weight stream between tiles.
    claim_lead_k_tiles: Optional[int] = None
    # Counter combine: the end-of-round reduce drops every combine slot line it has
    # consumed from L2 instead of leaving it to be written back (None: when the
    # round's combine slots take at least DISCARD_COMBINE_SLOTS_MIN_BYTES).
    discard_combine_slots: Optional[bool] = None
    # Counter combine (GenPhase): the FC2 epilogue pushes its rows to the combine
    # slots with warp-wide 16-byte stores instead of one bulk copy per row, keeping
    # the pushes off the TMA unit the mainloop loads through (None: under the pull
    # dispatch with pipelined schedule groups).
    combine_warp_stores: Optional[bool] = None
    epi_flag_batches: Tuple[int, int] = (4, 2)
    token_in_flag_batch: int = 1
    token_padding_block: int = 64
    sf_padding_block: int = 128
    gate_up_clamp: Optional[float] = None
    # Top-k reduce in the kernel: each route's partial adds into its source
    # token's output row (BF16, in arrival order) instead of landing in a combine
    # slot that a separate reduce sums. The output then lives on the symmetric
    # heap and is zeroed at every launch (by GenPhase's main kernel itself).
    # GenPhase supports it under the counter combine, whose tail barrier
    # publishes the adds; there it drops the combine slots and the reduce, and
    # changes the output's rounding (one BF16 rounding per add instead of one
    # FP32 sum per token), so it is never a default.
    reduce_topk_in_kernel: bool = False
    token_back_mode: BlockScaledTokenBackMode = "epi_warps"
    # Routing weights multiply each route's SwiGLU output before its FC2-input
    # requantization (True), or its FC2 partial in FP32 after FC2 (False; under
    # GenPhase in the top-k reduce, or in the FC2 epilogue with the in-kernel
    # reduce).
    apply_routing_weights_before_fc2: bool = True
    max_sm_count: Optional[int] = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: Optional[float] = None
    situ_linear_beta: Optional[float] = None

    kernel_variant: Literal["inference", "genphase"] = "inference"
    combine_dtype: Literal["bf16", "nvfp4", "mxfp8"] = "bf16"
    # GenPhase with MXFP4 weights: one MXFP8 E4M3 shared expert, run in the
    # kernel over every token of the rank ("fp8_in_kernel"), with the routed
    # experts' intermediate width; its output joins the routed experts' sum.
    shared_expert: Optional[Literal["fp8_in_kernel"]] = None
    # The shared expert's SwiGLU clamp; None clamps nothing, whatever
    # gate_up_clamp is.
    shared_gate_up_clamp: Optional[float] = None
    # Round the shared expert's gate, up and SwiGLU product to BF16, and the
    # routes' FP32 sum to BF16 before the shared output joins it (the recipe of
    # a BF16 shared expert added to a BF16 routed output). Off, every rounding
    # but the MXFP8 requantization and the BF16 partials waits for the one of
    # the final sum. It changes the output, so it is never a default.
    shared_bf16_rounding: bool = False
    # Claim the shared expert's FC1 tiles (and its early FC2 tiles) from the main
    # kernel's entry, from a counter of their own, while the dispatch is in flight
    # (None: on with a shared expert); off, they lead every claim stream once the
    # plan exists. Placement only: the output is the same bit for bit.
    shared_expert_entry: Optional[bool] = None
    # Where the shared expert's FC2 tiles go: behind the routed tiles ("tail"),
    # where they overlap the last combine pushes and the reduce's polls; right
    # behind its FC1 tiles ("early"), where they hold their clusters at their
    # FC1 gate while the routed FC1 tiles become ready; or early exactly where
    # all of them fit one wave of clusters ("auto", DeepGEMM's rule).
    shared_fc2_placement: Literal["auto", "early", "tail"] = "tail"
    # Where the kernel carries an alternate token tile (``alternate_token_tile_n``),
    # an entry head whose shared token tiles hold more rows than it takes it away.
    # "balanced" sizes the shared token tiles to fit it where a balanced plan, every
    # local expert at the expected load, would take it; "always" wherever the kernel
    # carries one; "never" keeps the plan's tile N. Fitting holds the head until the
    # launch has chosen its configuration from the plan's expert sizes, and costs
    # more shared weight reads per token. Placement only: the output is the same
    # bit for bit.
    shared_alternate_token_tile: Literal["never", "balanced", "always"] = "never"
    # Token tiles whose shared FC1 tiles the entry head holds; the others' join
    # the claim streams, spread over the routed tiles. None: all of them; "window":
    # as many as finish before the dispatch's first routed FC1 tile
    # (``genphase.shared_head_token_tiles``). Placement only: the output is the
    # same bit for bit.
    shared_head_token_tiles: Optional[Union[int, Literal["window"]]] = None

    def __post_init__(self) -> None:
        if self.quant_kind not in _KIND_TABLE:
            raise ValueError(f"unsupported quant_kind {self.quant_kind!r}.")
        if self.kernel_variant not in ("inference", "genphase"):
            raise ValueError(f"unsupported kernel_variant {self.kernel_variant!r}.")
        if self.combine_dtype not in ("bf16", "nvfp4", "mxfp8"):
            raise ValueError(f"unsupported combine_dtype {self.combine_dtype!r}.")
        if self.kernel_variant == "genphase" and self.combine_dtype != "bf16":
            raise ValueError("GenPhase requires BF16 combine.")
        if self.combine_dtype != "bf16" and self.reduce_topk_in_kernel:
            raise ValueError("Quantized combine requires separate top-k reduction.")
        if (
            self.combine_dtype == "nvfp4"
            and self.fc2_use_bulk
            and self.token_back_mode == "epi_warps"
        ):
            raise ValueError(
                "NVFP4 combine bulk stores require a token-back data path."
            )
        if self.activation not in ("swiglu", "situ"):
            raise ValueError("activation must be 'swiglu' or 'situ'.")
        if self.activation == "situ":
            for name in ("situ_beta", "situ_linear_beta"):
                value = getattr(self, name)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"SiTU requires positive, finite {name}.")
            if self.gate_up_clamp is not None:
                raise ValueError("SiTU does not support gate_up_clamp.")
        elif self.situ_beta is not None or self.situ_linear_beta is not None:
            raise ValueError("SiTU beta parameters require activation='situ'.")
        for name in (
            "num_total_experts",
            "max_tokens_per_rank",
            "num_topk",
            "hidden",
            "intermediate",
            "world_size",
            "token_padding_block",
            "sf_padding_block",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer.")
        if type(self.rank) is not int or not (0 <= self.rank < self.world_size):
            raise ValueError(f"invalid rank/world_size {self.rank}/{self.world_size}.")
        for name in (
            "fc2_use_bulk",
            "reduce_topk_in_kernel",
            "apply_routing_weights_before_fc2",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a Python bool.")
        if self.num_total_experts > 16384:
            raise ValueError(
                "block-scaled routing supports at most 16384 global experts."
            )
        if self.num_topk > self.num_total_experts:
            raise ValueError("num_topk must not exceed num_total_experts.")
        if self.padded_tokens_per_rank * self.num_topk > 2097152:
            raise ValueError(
                "block-scaled routing supports at most 2097152 routes per rank."
            )
        if self.num_total_experts % self.world_size != 0:
            raise ValueError(
                f"num_total_experts ({self.num_total_experts}) must divide evenly "
                f"across world_size ({self.world_size})."
            )
        if (
            self.just_in_time_claims is not None
            and type(self.just_in_time_claims) is not bool
        ):
            raise ValueError(
                f"just_in_time_claims must be None or a Python bool, got {self.just_in_time_claims!r}."
            )
        if self.claim_lead_k_tiles is not None and (
            type(self.claim_lead_k_tiles) is not int or self.claim_lead_k_tiles < 1
        ):
            raise ValueError(
                f"claim_lead_k_tiles must be None or a positive Python integer, got {self.claim_lead_k_tiles!r}."
            )
        if (
            self.claim_lead_k_tiles is not None
            and not self.resolved_just_in_time_claims
        ):
            raise ValueError(
                "claim_lead_k_tiles applies only to just-in-time claims: the Sentinel dispatch at "
                f"up to {JUST_IN_TIME_CLAIMS_MAX_TOKENS} padded tokens per rank or just_in_time_claims=True."
            )
        if (
            self.discard_combine_slots is not None
            and type(self.discard_combine_slots) is not bool
        ):
            raise ValueError(
                f"discard_combine_slots must be None or a Python bool, got {self.discard_combine_slots!r}."
            )
        if self.entry_prefetch_bytes is not None and (
            type(self.entry_prefetch_bytes) is not int or self.entry_prefetch_bytes < 0
        ):
            raise ValueError(
                f"entry_prefetch_bytes must be None or a non-negative int, got {self.entry_prefetch_bytes!r}."
            )
        if self.discard_combine_slots and self.combine_protocol != "counter":
            raise ValueError(
                "discard_combine_slots requires combine_protocol='counter'."
            )
        if self.kernel_variant == "genphase" and self.reduce_topk_in_kernel:
            # The counter combine's tail barrier publishes the adds.
            if self.combine_protocol != "counter":
                raise ValueError(
                    "GenPhase reduces top-k in the kernel only under combine_protocol='counter'."
                )
            if self.discard_combine_slots:
                raise ValueError(
                    "The in-kernel top-k reduce has no combine slots to discard."
                )
        if self.token_ab_stages is not None and (
            type(self.token_ab_stages) is not int
            or self.token_ab_stages == 1
            or self.token_ab_stages < 0
            or (self.token_ab_stages and self.kernel_variant != "genphase")
        ):
            raise ValueError(
                "token_ab_stages must be None, 0 or an int of at least 2 (GenPhase), got "
                f"{self.token_ab_stages!r}."
            )
        if self.alternate_token_tile_n is not None and (
            type(self.alternate_token_tile_n) is not int
            or self.alternate_token_tile_n not in (0, ALTERNATE_TOKEN_TILE_N)
            or (self.alternate_token_tile_n and not self._admits_alternate_token_tile)
        ):
            raise ValueError(
                f"alternate_token_tile_n must be None, 0 or {ALTERNATE_TOKEN_TILE_N}, the latter under a strided "
                "GenPhase token tile (192, 240) at cluster (2, 1), got "
                f"{self.alternate_token_tile_n!r}."
            )
        if self.alternate_token_ab_stages is not None and (
            type(self.alternate_token_ab_stages) is not int
            or self.alternate_token_ab_stages == 1
            or self.alternate_token_ab_stages < 0
            or not self.resolved_alternate_token_tile_n
        ):
            raise ValueError(
                "alternate_token_ab_stages must be None, 0 or an int of at least 2 with an alternate token tile, got "
                f"{self.alternate_token_ab_stages!r}."
            )
        if self.alternate_weight_evict not in (None, "normal", "first", "adaptive") or (
            self.alternate_weight_evict is not None
            and not self.resolved_alternate_token_tile_n
        ):
            raise ValueError(
                "alternate_weight_evict must be None, 'normal', 'first' or 'adaptive' with an alternate token tile, "
                f"got {self.alternate_weight_evict!r}."
            )
        if self.token_tile_choice not in ("rule", "primary") or (
            self.token_tile_choice != "rule"
            and not self.resolved_alternate_token_tile_n
        ):
            raise ValueError(
                "token_tile_choice must be 'rule', or 'primary' with an alternate token tile, got "
                f"{self.token_tile_choice!r}."
            )
        if self.combine_warp_stores is not None and (
            type(self.combine_warp_stores) is not bool
            or (
                self.combine_warp_stores
                and (
                    self.combine_protocol != "counter"
                    or self.kernel_variant != "genphase"
                )
            )
        ):
            raise ValueError(
                "combine_warp_stores must be None or a Python bool, and True requires GenPhase with "
                f"combine_protocol='counter', got {self.combine_warp_stores!r}."
            )
        if self.dispatch_rounds not in (1, 2, 4, 8):
            raise ValueError(
                f"dispatch_rounds must be 1, 2, 4 or 8, got {self.dispatch_rounds!r}."
            )
        if self.dispatch_rounds > 1 and self.dispatch_protocol != "counter":
            raise ValueError(
                "dispatch_rounds > 1 requires dispatch_protocol='counter'."
            )
        # A rank acquires only the rounds its own experts need, so the clear of
        # the round counters relies on the counter combine's all-rank barrier to
        # order every peer's publications before it.
        if self.dispatch_rounds > 1 and self.combine_protocol != "counter":
            raise ValueError("dispatch_rounds > 1 requires combine_protocol='counter'.")
        # Peers read a rank's staged activations until their communication grids
        # complete, and only the counter combine's all-rank tail barrier orders
        # that before the rank's next staging.
        if self.dispatch_protocol == "pull" and self.combine_protocol != "counter":
            raise ValueError(
                "dispatch_protocol='pull' requires combine_protocol='counter'."
            )
        for name in ("comm_pull_depth", "comm_pull_chunk_rows", "comm_pull_lead_rows"):
            value = getattr(self, name)
            if value is not None and (
                self.dispatch_protocol != "pull" or type(value) is not int or value < 1
            ):
                raise ValueError(
                    f"{name} must be None or a positive Python integer under dispatch_protocol='pull', "
                    f"got {value!r}."
                )
        if self.comm_exchange_ctas is not None and (
            self.dispatch_protocol == "sentinel"
            or type(self.comm_exchange_ctas) is not int
            or self.comm_exchange_ctas < 1
        ):
            raise ValueError(
                "comm_exchange_ctas must be None or a positive Python integer under the counter or pull dispatch, "
                f"got {self.comm_exchange_ctas!r}."
            )
        if self.pull_warps is not None and (
            self.dispatch_protocol != "pull"
            or type(self.pull_warps) is not int
            or not 0 <= self.pull_warps <= 4
        ):
            raise ValueError(
                f"pull_warps must be None or a Python integer in [0, 4] under dispatch_protocol='pull', "
                f"got {self.pull_warps!r}."
            )
        if self.gather_b_warps is not None:
            counts = (4, 8)
            if (
                type(self.gather_b_warps) is not int
                or self.gather_b_warps not in counts
            ):
                raise ValueError(
                    f"gather_b_warps must be None, 4 or 8, got {self.gather_b_warps!r}."
                )
            if self.gather_b_warps == 8 and (
                self.pull_warps
                or tuple(self.cluster_shape_mn) != (2, 1)
                or self.resolved_mma_tiler_mnk[1] < 128
            ):
                raise ValueError(
                    "gather_b_warps=8 needs cluster (2, 1), token tile N >= 128 and no pull_warps."
                )
            if self.gather_b_warps == 4 and (
                self.resolved_mma_tiler_mnk[1] == 64
                and tuple(self.cluster_shape_mn) == (4, 1)
            ):
                raise ValueError(
                    "gather_b_warps at N64 on cluster (4, 1) is two; leave it None."
                )
        for name in (
            "pull_acquire_in_copy",
            "pull_walk_dedup",
            "pull_tile_ready",
            "fc1_token_pool",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(
                    f"{name} must be None or a Python bool, got {value!r}."
                )
        if self.fc1_token_pool_capacity is not None and (
            type(self.fc1_token_pool_capacity) is not int
            or self.fc1_token_pool_capacity < 0
        ):
            raise ValueError(
                "fc1_token_pool_capacity must be None or a nonnegative int, "
                f"got {self.fc1_token_pool_capacity!r}."
            )
        if self.fc1_token_pool_fallback not in ("tile", "launch"):
            raise ValueError(
                "fc1_token_pool_fallback must be 'tile' or 'launch', "
                f"got {self.fc1_token_pool_fallback!r}."
            )
        if self.fc1_token_pool and (
            self.kernel_variant != "genphase"
            or self.dispatch_protocol != "pull"
            or self.pull_warps
            or self.pull_walk_dedup
            or self.gather_b_warps == 8
        ):
            raise ValueError(
                "fc1_token_pool=True needs GenPhase under dispatch_protocol='pull' with walking CTAs "
                "(no pull_warps), no pull_walk_dedup and the default Gather-B warps."
            )
        vec = self.sf_vec_size
        if self.hidden % (4 * vec) != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a multiple of {4 * vec} "
                f"for {self.quant_kind}."
            )
        if self.intermediate % (2 * vec) != 0 or self.intermediate % 32 != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a multiple of "
                f"{max(2 * vec, 32)} (gate/up interleave + SF blocks) for "
                f"{self.quant_kind}."
            )
        if self.quant_kind == "mxfp4_mxfp8" and self.intermediate % 128:
            raise ValueError("mxfp4_mxfp8 intermediate must be a multiple of 128.")
        tiler = self.resolved_mma_tiler_mnk
        instruction_k = self.instruction_k
        if len(tiler) != 3:
            raise ValueError("mma_tiler_mnk must be a 3-tuple.")
        if any(type(dim) is not int or dim <= 0 for dim in tiler):
            raise ValueError(
                "mma_tiler_mnk dimensions must be positive Python integers."
            )
        if tiler[0] not in (128, 256):
            raise ValueError(
                f"block-scaled instruction M must be 128 or 256, got {tiler[0]}."
            )
        if tiler[1] not in (64, 128, 192, 240, 256):
            raise ValueError(
                f"block-scaled instruction N must be 64, 128, 192, 240 or 256, got {tiler[1]}."
            )
        # N192 and N240 lay token tiles out in 256-row slots of both pools rather
        # than in token padding blocks; only the GenPhase kernel implements that.
        strided_token_tile = tiler[1] in (192, 240)
        if strided_token_tile and self.kernel_variant != "genphase":
            raise ValueError(f"tile N {tiler[1]} is a GenPhase-only geometry.")
        if tiler[2] not in (2 * instruction_k, 4 * instruction_k):
            raise ValueError(
                f"block-scaled {self.quant_kind} tile K must be {2 * instruction_k} or "
                f"{4 * instruction_k}, got {tiler[2]}."
            )
        if (
            self.quant_kind == "nvfp4"
            and tiler[1] == 256
            and tiler[2] == 4 * instruction_k
        ):
            raise ValueError("nvfp4 does not support (N=256, K=4x-instruction) tiles.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must be a 2-tuple.")
        cm, cn = self.cluster_shape_mn
        if type(cm) is not int or cm <= 0 or cm > 16 or cm & (cm - 1):
            raise ValueError("cluster M must be a power of two no greater than 16.")
        if type(cn) is not int or cn != 1:
            raise ValueError("the swap-AB FC12 path requires cluster N=1.")
        if tiler[0] == 256 and self.cluster_shape_mn[0] % 2 != 0:
            raise ValueError("instruction M 256 requires an even cluster M.")
        if self.fallback_cluster_shape_mn is not None:
            fallback = tuple(self.fallback_cluster_shape_mn)
            if fallback == tuple(self.cluster_shape_mn):
                # Kernel-side NonClcMixedCgaConfig collapses this to a uniform
                # launch; normalize here so downstream logic sees one spelling.
                object.__setattr__(self, "fallback_cluster_shape_mn", None)
            else:
                if len(fallback) != 2:
                    raise ValueError("fallback_cluster_shape_mn must be a 2-tuple.")
                if any(type(dim) is not int or dim <= 0 for dim in fallback):
                    raise ValueError(
                        "fallback_cluster_shape_mn dimensions must be positive."
                    )
                if fallback[1] != 1:
                    raise ValueError(
                        "the swap-AB FC12 fallback path requires cluster N=1."
                    )
                if any(
                    preferred % fb != 0
                    for preferred, fb in zip(
                        self.cluster_shape_mn, fallback, strict=True
                    )
                ):
                    raise ValueError(
                        "every preferred cluster dimension must be divisible by "
                        "its fallback dimension."
                    )
                if tiler[0] == 256 and fallback[0] % 2 != 0:
                    raise ValueError(
                        "instruction M 256 requires an even fallback cluster M."
                    )
                if self.max_sm_count is not None:
                    raise ValueError(
                        "max_sm_count is not supported together with "
                        "fallback_cluster_shape_mn (mixed-CGA occupancy is "
                        "resolved from the hardware probe)."
                    )
        if self.token_padding_block % 64 != 0:
            raise ValueError("token_padding_block must be a multiple of 64.")
        if self.sf_padding_block % 128 != 0:
            raise ValueError("sf_padding_block must be a multiple of 128.")
        if tiler[1] % self.token_padding_block != 0 and not strided_token_tile:
            raise ValueError(
                f"mma tiler N ({tiler[1]}) must be a whole number of token "
                f"padding blocks ({self.token_padding_block})."
            )
        if len(self.schedule_policy) != 2:
            raise ValueError("schedule_policy requires (mode, hint).")
        mode, hint = self.schedule_policy
        if mode not in ("grouped", "phase_interleave"):
            raise ValueError(f"unknown schedule mode {mode!r}.")
        if hint is not None and (type(hint) is not int or hint <= 0):
            raise ValueError("schedule hint must be a positive Python integer or None.")
        if type(self.pipelined_groups) is not bool:
            raise ValueError("pipelined_groups must be a Python bool.")
        if self.pipelined_groups and mode != "grouped":
            raise ValueError("pipelined_groups requires grouped scheduling.")
        if self.work_id_mode not in ("grid_stride", "atomic_counter"):
            raise ValueError(f"unknown work_id_mode {self.work_id_mode!r}.")
        if self.token_back_mode not in (
            "epi_warps",
            "standalone_warps",
            "reuse_dispatch_warps",
        ):
            raise ValueError(f"unknown token_back_mode {self.token_back_mode!r}.")
        if mode == "phase_interleave" and self.work_id_mode != "atomic_counter":
            raise ValueError(
                "schedule_policy 'phase_interleave' requires "
                "work_id_mode='atomic_counter'."
            )
        if len(self.epi_flag_batches) != 2:
            raise ValueError("epi_flag_batches requires (FC1, FC2) values.")
        if any(
            type(batch) is not int or not 1 <= batch <= 4
            for batch in self.epi_flag_batches
        ):
            raise ValueError("block-scaled epi_flag_batches values must be in [1, 4].")
        if (
            type(self.token_in_flag_batch) is not int
            or not 1 <= self.token_in_flag_batch <= 32
        ):
            raise ValueError("token_in_flag_batch must be in [1, 32].")
        if self.max_sm_count is not None and (
            type(self.max_sm_count) is not int or self.max_sm_count < cm
        ):
            raise ValueError("max_sm_count must accommodate at least one cluster.")
        if self.gate_up_clamp is not None and (
            not math.isfinite(self.gate_up_clamp) or self.gate_up_clamp < 0
        ):
            raise ValueError("gate_up_clamp must be finite and nonnegative.")
        if self.fc2_tma_stages is not None and not (
            type(self.fc2_tma_stages) is int
            and 1 <= self.fc2_tma_stages <= -(-tiler[1] // 64)
        ):
            raise ValueError(
                f"fc2_tma_stages must be in [1, {-(-tiler[1] // 64)}] for tiler N "
                f"{tiler[1]}."
            )
        # GenPhase's FC2 epilogue weights each route's partial before the in-kernel
        # reduce adds it; the inference kernel's adds need already-weighted terms.
        if (
            self.reduce_topk_in_kernel
            and not self.apply_routing_weights_before_fc2
            and self.kernel_variant != "genphase"
        ):
            raise ValueError(
                "reduce_topk_in_kernel requires "
                "apply_routing_weights_before_fc2=True (the in-kernel reduce "
                "red-adds already-weighted terms)."
            )

        if self.kernel_variant == "genphase":
            if self.padded_tokens_per_rank > 8192:
                raise ValueError("GenPhase supports at most 8192 tokens per rank.")
            n64_decode = (
                self.cluster_shape_mn in ((2, 1), (4, 1))
                and tiler == (256, 64, 128)
                and self.quant_kind in ("mxfp8_e4m3", "mxfp4_mxfp8")
            )
            cga2_geometry = (
                self.cluster_shape_mn == (2, 1)
                and tiler
                in ((256, 64, 128), (256, 128, 128), (256, 192, 128), (256, 240, 128))
                and self.quant_kind in ("mxfp8_e4m3", "mxfp4_mxfp8")
            )
            if (
                self.cluster_shape_mn != (4, 1) and not cga2_geometry
            ) or self.fallback_cluster_shape_mn is not None:
                raise ValueError(
                    "GenPhase requires uniform cluster (4, 1), or an admitted N64 decode geometry at (2, 1) or (4, 1)."
                )
            if not self.fc2_use_bulk:
                raise ValueError("GenPhase requires fc2_use_bulk=True.")
            if (
                tiler[1] not in (128, 256)
                and not n64_decode
                and not (strided_token_tile and cga2_geometry)
            ):
                raise ValueError(
                    "GenPhase requires tile N 128 or 256, N192/N240 at cluster (2, 1), "
                    "or an admitted N64 decode geometry."
                )
            if n64_decode and self.world_size > 1 and mode == "phase_interleave":
                # EP1 phase-interleave coverage does not establish multirank
                # liveness: an EP2 dense/replay case stalls with this geometry.
                # Reject before allocation or compilation; autotune applies
                # this same config validation before compilation.
                raise ValueError(
                    "GenPhase N64 with world_size > 1 requires grouped scheduling; "
                    "multirank phase_interleave is not qualified."
                )
            gather_tile_k = 2 * _KIND_TABLE[self.quant_kind][3]
            if tiler[2] != gather_tile_k:
                raise ValueError(
                    f"GenPhase {self.quant_kind} requires tile K {gather_tile_k} "
                    "so each gathered row occupies one 128-byte swizzle atom."
                )
            if self.sf_padding_block != 128 or 128 % self.token_padding_block:
                raise ValueError(
                    "GenPhase requires SF padding 128 and token padding 64 or 128."
                )
            if self.token_back_mode != "epi_warps" or self.token_in_flag_batch != 1:
                raise ValueError(
                    "GenPhase does not use token-back or token-in flag knobs."
                )
        if (
            self.die_local_weights is not None
            and type(self.die_local_weights) is not bool
        ):
            raise ValueError("die_local_weights must be None or a Python bool.")
        if self.shared_expert not in (None, "fp8_in_kernel"):
            raise ValueError(
                f"shared_expert must be None or 'fp8_in_kernel', got {self.shared_expert!r}."
            )
        if (
            self.shared_expert_entry is not None
            and type(self.shared_expert_entry) is not bool
        ):
            raise ValueError("shared_expert_entry must be None or a Python bool.")
        if self.shared_fc2_placement not in ("auto", "early", "tail"):
            raise ValueError(
                f"shared_fc2_placement must be 'auto', 'early' or 'tail', got {self.shared_fc2_placement!r}."
            )
        if self.shared_alternate_token_tile not in ("never", "balanced", "always"):
            raise ValueError(
                "shared_alternate_token_tile must be 'never', 'balanced' or 'always', got "
                f"{self.shared_alternate_token_tile!r}."
            )
        if self.shared_head_token_tiles not in (None, "window") and (
            type(self.shared_head_token_tiles) is not int
            or self.shared_head_token_tiles < 1
        ):
            raise ValueError(
                f"shared_head_token_tiles must be None, 'window' or a positive int, got "
                f"{self.shared_head_token_tiles!r}."
            )
        if self.shared_expert is None:
            if (
                self.shared_gate_up_clamp is not None
                or self.shared_bf16_rounding
                or self.shared_expert_entry
                or self.shared_fc2_placement != "tail"
                or self.shared_alternate_token_tile != "never"
                or self.shared_head_token_tiles is not None
            ):
                raise ValueError(
                    "shared_gate_up_clamp, shared_bf16_rounding, shared_expert_entry, shared_fc2_placement, "
                    "shared_alternate_token_tile and shared_head_token_tiles need shared_expert."
                )
        elif self.shared_head_token_tiles is not None and (
            self.shared_expert_entry is False or self.shared_fc2_placement != "tail"
        ):
            raise ValueError(
                "shared_head_token_tiles caps the entry head, with the shared FC2 tiles at the tail."
            )
        else:
            if (
                self.kernel_variant != "genphase"
                or self.quant_kind != "mxfp4_mxfp8"
                or tuple(self.cluster_shape_mn) != (2, 1)
                or tiler[0] != 256
            ):
                raise ValueError(
                    "shared_expert needs GenPhase with MXFP4 x MXFP8 weights, tile M 256 and cluster (2, 1)."
                )
            if self.shared_gate_up_clamp is not None and (
                not math.isfinite(self.shared_gate_up_clamp)
                or self.shared_gate_up_clamp < 0
            ):
                raise ValueError("shared_gate_up_clamp must be finite and nonnegative.")
            if type(self.shared_bf16_rounding) is not bool:
                raise ValueError("shared_bf16_rounding must be a Python bool.")
            if self.shared_bf16_rounding and self.reduce_topk_in_kernel:
                raise ValueError(
                    "shared_bf16_rounding rounds the routes' sum before the shared output joins it, which the "
                    "in-kernel top-k reduce, adding every partial in arrival order, cannot do."
                )
        if self.die_local_weights and not self._admits_die_local_weights:
            raise ValueError(
                "die_local_weights requires GenPhase MXFP4 block-major weights, cluster "
                "(2, 1) with tile M 256, grouped atomic-counter scheduling over the full "
                "launch, and 2 * intermediate and hidden that split into whole M256 "
                "tiles per die (multiples of 512)."
            )

    @property
    def experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def resolved_weight_layout(self) -> str:
        return self.weight_layout or default_weight_layout(self.quant_kind)

    @property
    def _admits_die_local_weights(self) -> bool:
        """Whether each die's half of every weight matrix is whole M256 tiles the claim streams can split."""
        tiler = self.resolved_mma_tiler_mnk
        return (
            self.kernel_variant == "genphase"
            and self.quant_kind == "mxfp4_mxfp8"
            and self.resolved_weight_layout == "blockmajor"
            and tuple(self.cluster_shape_mn) == (2, 1)
            and tiler[0] == 256
            and self.schedule_policy[0] == "grouped"
            and self.work_id_mode == "atomic_counter"
            and self.max_sm_count is None
            and (2 * self.intermediate) % 512 == 0
            and self.hidden % 512 == 0
        )

    @property
    def resolved_die_local_weights(self) -> bool:
        """Whether the weights are die-local; the default also asks the current device for two locality domains."""
        if self.die_local_weights is not None:
            return self.die_local_weights
        expected_load = (
            self.max_tokens_per_rank
            * self.world_size
            * self.num_topk
            / self.num_total_experts
        )
        return (
            expected_load >= DIE_LOCAL_WEIGHTS_MIN_EXPECTED_LOAD
            and self._admits_die_local_weights
            and _device_has_two_locality_domains()
        )

    @property
    def resolved_entry_prefetch_bytes(self) -> int:
        """FC1 weight bytes the main kernel prefetches into L2 at entry (``ENTRY_PREFETCH_BYTES`` notes).

        A shared expert claimed from kernel entry streams its FC1 weights and
        scale factors through the same window, so the prefetch leaves them out.
        """
        if self.entry_prefetch_bytes is not None:
            return self.entry_prefetch_bytes
        prefetch = ENTRY_PREFETCH_BYTES
        whole_payload = (
            self.dispatch_protocol == "counter" and self.dispatch_rounds == 1
        )
        long_sentinel_dispatch = (
            self.dispatch_protocol == "sentinel"
            and self.max_tokens_per_rank >= SENTINEL_DISPATCH_ENTRY_PREFETCH_MIN_TOKENS
        )
        if (
            self.kernel_variant == "genphase"
            and (whole_payload or long_sentinel_dispatch)
            and self.max_tokens_per_rank <= WHOLE_PAYLOAD_ENTRY_PREFETCH_MAX_TOKENS
        ):
            prefetch = WHOLE_PAYLOAD_ENTRY_PREFETCH_BYTES
        if self.resolved_shared_expert_entry:
            shared_fc1_bytes = (
                2 * self.intermediate * self.hidden * (1 + 1 / self.sf_vec_size)
            )
            prefetch = max(0, prefetch - int(shared_fc1_bytes))
        return prefetch

    @property
    def resolved_shared_expert_entry(self) -> bool:
        """Whether the shared expert's FC1 tiles are claimed from the main kernel's entry (``shared_expert_entry``)."""
        if self.shared_expert is None:
            return False
        return True if self.shared_expert_entry is None else self.shared_expert_entry

    @property
    def resolved_shared_alternate_token_tile(self) -> bool:
        """Whether the shared expert's token tiles fit the alternate token tile (``shared_alternate_token_tile``).

        Only an entry head that does not fit takes the alternate away. Under
        "balanced" an uneven plan only grows the largest tile, so where a
        balanced plan does not fit the alternate it rarely runs.
        """
        alternate_tile_n = self.resolved_alternate_token_tile_n
        if (
            self.shared_alternate_token_tile == "never"
            or not self.resolved_shared_expert_entry
            or not alternate_tile_n
        ):
            return False
        if self.shared_alternate_token_tile == "always":
            return True
        expected_load = (
            self.max_tokens_per_rank
            * self.world_size
            * self.num_topk
            / self.num_total_experts
        )
        tile_n = self.resolved_mma_tiler_mnk[1]
        return (
            balanced_token_tile_rows(math.ceil(expected_load), tile_n)
            <= alternate_tile_n
        )

    @property
    def shared_expert_token_tiling(self) -> Tuple[int, int]:
        """(token tiles, rows per token tile but the last) of the shared expert, as the main kernel lays them out.

        Rows of at most the plan's tile N, or of the alternate's where the
        shared tiles fit it, spread evenly over the fewest tiles under a strided
        tile N (``balanced_token_tile_rows``).
        """
        tile_n = self.resolved_mma_tiler_mnk[1]
        bound = (
            self.resolved_alternate_token_tile_n
            if self.resolved_shared_alternate_token_tile
            else tile_n
        )
        tokens = self.padded_tokens_per_rank
        tiles = -(-tokens // bound)
        if not _strided_token_tile(tile_n):
            return tiles, bound
        return tiles, balanced_token_tile_rows(tokens, bound)

    @property
    def resolved_just_in_time_claims(self) -> bool:
        if self.just_in_time_claims is not None:
            return self.just_in_time_claims
        return (
            self.dispatch_protocol == "sentinel"
            and self.padded_tokens_per_rank <= JUST_IN_TIME_CLAIMS_MAX_TOKENS
        )

    @property
    def resolved_fc1_token_pool(self) -> bool:
        """Whether the pull dispatch's walk stores rows into the token pool that FC1 loads by tiled TMA."""
        if self.fc1_token_pool is not None:
            return self.fc1_token_pool
        expected_load = (
            self.max_tokens_per_rank
            * self.world_size
            * self.num_topk
            / self.num_total_experts
        )
        return (
            self.kernel_variant == "genphase"
            and self.dispatch_protocol == "pull"
            and expected_load >= FC1_TOKEN_POOL_MIN_EXPECTED_LOAD
            and not self.pull_warps
            and not self.pull_walk_dedup
            and self.gather_b_warps != 8
        )

    @property
    def resolved_fc1_token_pool_capacity(self) -> int:
        """Rows of the token pool the launch asks for (``fc1_token_pool_capacity``), or 0 for a pool sized for any routing.

        0 also where the pool is off. The kernel sizes the pool for any routing
        as well where the capacity reaches the data rows of the worst case or
        the token tile is not strided; ``kernel.token_pool_capacity`` is then 0.
        """
        if not self.resolved_fc1_token_pool:
            return 0
        if self.fc1_token_pool_capacity is not None:
            return self.fc1_token_pool_capacity
        return (
            FC1_TOKEN_POOL_CAPACITY_FACTOR * self.padded_tokens_per_rank * self.num_topk
        )

    @property
    def resolved_gather_b_warps(self) -> Optional[int]:
        """Gather-B warps of the main kernel, or None for the kernel's default (which idle under the token pool)."""
        if self.gather_b_warps is not None:
            return self.gather_b_warps
        if self.resolved_fc1_token_pool:
            return None
        expected_load = (
            self.max_tokens_per_rank
            * self.world_size
            * self.num_topk
            / self.num_total_experts
        )
        if (
            self.resolved_mma_tiler_mnk[1] >= EIGHT_GATHER_B_WARPS_MIN_TILE_N
            and expected_load >= EIGHT_GATHER_B_WARPS_MIN_EXPECTED_LOAD
            and tuple(self.cluster_shape_mn) == (2, 1)
            and not self.pull_warps
        ):
            return 8
        return None

    @property
    def resolved_token_ab_stages(self) -> int:
        """Token stages of the main kernel's split AB ring, or 0 for one shared ring."""
        if self.token_ab_stages is not None:
            return self.token_ab_stages
        tile_n = self.resolved_mma_tiler_mnk[1]
        if self.kernel_variant != "genphase" or tile_n not in SPLIT_RING_TOKEN_STAGES:
            return 0
        expected_load = (
            self.max_tokens_per_rank
            * self.world_size
            * self.num_topk
            / self.num_total_experts
        )
        fits_one_tile = largest_expected_expert_load(expected_load) <= tile_n
        return SPLIT_RING_TOKEN_STAGES[tile_n] if fits_one_tile else 0

    @property
    def _admits_alternate_token_tile(self) -> bool:
        """A strided GenPhase token tile with two-CTA mxf8f6f4 MMAs at cluster (2, 1), whose plan N128 can run."""
        tiler = self.resolved_mma_tiler_mnk
        return (
            self.kernel_variant == "genphase"
            and _strided_token_tile(tiler[1])
            and tiler[0] == 256
            and tuple(self.cluster_shape_mn) == (2, 1)
            and self.quant_kind in ("mxfp8_e4m3", "mxfp4_mxfp8")
        )

    @property
    def resolved_alternate_token_tile_n(self) -> int:
        """Token tile N of the main kernel's alternate configuration, or 0 for none."""
        if self.alternate_token_tile_n is not None:
            return self.alternate_token_tile_n
        return ALTERNATE_TOKEN_TILE_N if self._admits_alternate_token_tile else 0

    @property
    def resolved_alternate_token_ab_stages(self) -> int:
        """Token stages of the alternate tile's split AB ring, or 0 for one shared ring (or no alternate)."""
        alternate_tile_n = self.resolved_alternate_token_tile_n
        if not alternate_tile_n:
            return 0
        if self.alternate_token_ab_stages is not None:
            return self.alternate_token_ab_stages
        expected_load = (
            self.max_tokens_per_rank
            * self.world_size
            * self.num_topk
            / self.num_total_experts
        )
        fits_one_tile = (
            largest_expected_expert_load(expected_load)
            <= self.resolved_mma_tiler_mnk[1]
        )
        return SPLIT_RING_TOKEN_STAGES[alternate_tile_n] if fits_one_tile else 0

    @property
    def resolved_combine_warp_stores(self) -> bool:
        if self.combine_warp_stores is not None:
            return self.combine_warp_stores
        return (
            self.kernel_variant == "genphase"
            and self.combine_protocol == "counter"
            and self.dispatch_protocol == "pull"
            and self.pipelined_groups
        )

    @property
    def resolved_discard_combine_slots(self) -> bool:
        if self.discard_combine_slots is not None:
            return self.discard_combine_slots
        if self.reduce_topk_in_kernel:
            return False
        # BF16 slots, one per (token, top-k route).
        slot_bytes = self.padded_tokens_per_rank * self.num_topk * self.hidden * 2
        return (
            self.combine_protocol == "counter"
            and slot_bytes >= DISCARD_COMBINE_SLOTS_MIN_BYTES
        )

    @property
    def padded_tokens_per_rank(self) -> int:
        """Physical capacity keeps the router's four-int vector loads in bounds."""
        multiple = 4 // math.gcd(self.num_topk, 4)
        return (self.max_tokens_per_rank + multiple - 1) // multiple * multiple

    @property
    def sf_vec_size(self) -> int:
        return _KIND_TABLE[self.quant_kind][2]

    @property
    def instruction_k(self) -> int:
        return _KIND_TABLE[self.quant_kind][3]

    @property
    def resolved_mma_tiler_mnk(self) -> Tuple[int, int, int]:
        if self.mma_tiler_mnk is not None:
            return self.mma_tiler_mnk
        return (256, 128, 2 * self.instruction_k)

    @property
    def torch_act_data_dtype(self) -> torch.dtype:
        dtype = _KIND_TABLE[self.quant_kind][0]
        return _fp4_storage_dtype() if dtype is None else dtype

    @property
    def torch_weight_data_dtype(self) -> torch.dtype:
        if self.quant_kind in ("nvfp4", "mxfp4_mxfp8"):
            return _fp4_storage_dtype()
        return self.torch_act_data_dtype

    @property
    def torch_act_sf_dtype(self) -> torch.dtype:
        return _KIND_TABLE[self.quant_kind][1]


class BlockScaledSymmBuffer:
    """Shared session workspace for the block-scaled inference mega kernel.

    Owns the staging tensors the backend fills (``x``, ``x_sf``, ``topk_idx``,
    ``topk_weights``), the kernel + device workspaces, and the compile cache.
    Construct through an architecture-specific subclass.
    """

    _architecture: str

    @staticmethod
    def _require_device() -> None:
        raise NotImplementedError("An architecture adapter must validate the device.")

    @staticmethod
    def _configure_dsl() -> None:
        raise NotImplementedError(
            "An architecture adapter must select the compiler target."
        )

    def __init__(self, config: BlockScaledMoeConfig, process_group=None) -> None:
        """Allocate staging, output and kernel storage collectively before capture.

        Symmetric allocations and the arming barrier run over ``process_group``
        (default: the world group), whose rank and size must be the config's.
        """
        comm.ensure_not_capturing("block-scaled mega workspace allocation")
        self._require_device()
        self._configure_dsl()
        self.config = config
        self.process_group = process_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        cfg = config
        if not comm._no_dist():
            import torch.distributed as dist

            group = process_group or dist.group.WORLD
            if (dist.get_rank(group), dist.get_world_size(group)) != (
                cfg.rank,
                cfg.world_size,
            ):
                raise ValueError(
                    f"config rank/world_size {cfg.rank}/{cfg.world_size} do not match "
                    f"the process group's {dist.get_rank(group)}/"
                    f"{dist.get_world_size(group)}."
                )

        kernel = self._build_kernel(cfg)
        self.kernel = kernel

        tokens = cfg.padded_tokens_per_rank

        # Staging tensors. Activation / SF / scores live on the symmetric heap;
        # the routing indices are a local (16B-aligned) int32 tensor read by this
        # rank's router only. The activation and its SF share one allocation, so
        # one peer offset table reaches both: under the pull dispatch peers read
        # their rows straight out of them.
        data_columns = cfg.hidden // 2 if cfg.quant_kind == "nvfp4" else cfg.hidden
        self.sf_cols = int(
            kernel.token_comm.plain_activation_sf_row_elements
            if cfg.kernel_variant == "genphase"
            else kernel.token_comm.activation_sf_hidden_padded
        )
        # GenPhase's in-kernel top-k reduce adds into the output from every peer, so
        # the output joins that allocation and its offset table.
        genphase_output = cfg.reduce_topk_in_kernel and cfg.kernel_variant == "genphase"
        staging = comm.sym_zeros_views(
            (
                ((tokens, data_columns), cfg.torch_act_data_dtype),
                ((tokens, self.sf_cols), cfg.torch_act_sf_dtype),
            )
            + ((((tokens, cfg.hidden), torch.bfloat16),) if genphase_output else ()),
            process_group,
        )
        self.x, self.x_sf = staging[:2]
        self.topk_weights = comm.sym_zeros(
            (tokens, cfg.num_topk), torch.float32, process_group
        )
        self.topk_idx = torch.full(
            (tokens, cfg.num_topk), -1, dtype=torch.int32, device="cuda"
        )
        # Stable local-expert pointers allow per-call scaling updates in graphs.
        # Defaults and overrides are copied on every backend staging call.
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            setattr(
                self,
                name,
                torch.ones(
                    cfg.experts_per_rank, dtype=torch.float32, device=self.device
                )
                if cfg.quant_kind == "nvfp4"
                else None,
            )
        if self.topk_idx.data_ptr() % 16 != 0:
            raise RuntimeError("routing index tensor must be 16-byte aligned.")

        # In-kernel reduce: the combine peer-writes (red.add) each topk term
        # straight into the output, so it must live on the SYMMETRIC heap and
        # start zeroed; the separate-reduce path only sees local stores.
        if genphase_output:
            self.output_activation = staging[2]
        elif cfg.reduce_topk_in_kernel:
            self.output_activation = comm.sym_zeros(
                (tokens, cfg.hidden), torch.bfloat16, process_group
            )
        else:
            self.output_activation = torch.zeros(
                (tokens, cfg.hidden), dtype=torch.bfloat16, device="cuda"
            )

        self._allocate_kernel_workspace(kernel)

        self._compiled: Optional[Any] = None
        self._launch_key: Optional[tuple] = None
        self._launch_kwargs: Optional[dict] = None
        self._launch_weights: Optional[
            Tuple[
                TransformedBlockScaledWeights,
                TransformedBlockScaledWeights,
                Optional[
                    Tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights]
                ],
            ]
        ] = None
        self._staged_tokens: Optional[int] = None
        self._destroyed = False

    def _allocate_kernel_workspace(self, kernel) -> None:
        """Allocate kernel-owned storage; staging and output pointers stay stable."""
        cfg = self.config
        tokens = cfg.padded_tokens_per_rank
        workspace = kernel._device_workspace
        local_bytes = int(workspace.total_bytes("local"))
        # Allocated fully zeroed, which covers require_zero_workspace_leading_bytes.
        self.local_workspace = torch.zeros(
            (max(local_bytes, 1),), dtype=torch.uint8, device="cuda"
        )
        # Two landing sets alternate by round. Every landing region starts armed with
        # 0xFF bytes in both sets, and no peer may push before every rank has armed.
        set_bytes = int(kernel.shared_set_bytes)
        self.shared_workspace = comm.sym_zeros(
            (2 * set_bytes,), torch.uint8, self.process_group
        )
        for landing_set in range(2):
            for region in kernel.sentinel_landing_regions:
                start = landing_set * set_bytes + workspace.offset(region)
                self.shared_workspace[start : start + workspace.nbytes(region)].fill_(
                    0xFF
                )
        if not comm._no_dist():
            import torch.distributed as dist

            torch.cuda.synchronize()
            dist.barrier(group=self.process_group)
        self._pre_reduced_activation = None
        self._pre_reduced_activation_sf = None
        if not cfg.reduce_topk_in_kernel:
            region = (
                kernel.pre_reduced_activation_region
                if cfg.kernel_variant == "genphase"
                else kernel.token_comm.pre_reduced_activation_region
            )
            start = workspace.offset(region)
            end = start + workspace.nbytes(region)
            # Byte views also cover packed FP4/FP8 payloads, which do not
            # support Torch masked_fill directly.
            self._pre_reduced_activation = self.shared_workspace[start:end].view(
                tokens, getattr(kernel, "combine_slots_per_token", cfg.num_topk), -1
            )
            if cfg.combine_dtype != "bf16":
                region = kernel.token_comm.pre_reduced_activation_sf_region
                start = workspace.offset(region)
                end = start + workspace.nbytes(region)
                self._pre_reduced_activation_sf = self.shared_workspace[start:end].view(
                    tokens, cfg.num_topk, -1
                )
        if cfg.kernel_variant == "genphase" and cfg.gather_padding_spread:
            # The router writes gather indices only for routed rows; the padding
            # rows of every token tile keep whatever the plane holds. Zero-filled,
            # they all gather dense row 0, so every CTA's padded Gather4 rows hit
            # the same 128-byte lines each K step: an L2 hot spot whose cost
            # depends on where that row is homed. Seed the plane with distinct
            # in-bounds rows; later launches only overwrite routed rows with
            # other in-bounds rows.
            region = kernel.token_comm.gather_index_region
            start = workspace.offset(region)
            gather_index = self.local_workspace[
                start : start + workspace.nbytes(region)
            ].view(torch.int32)
            dense_rows = cfg.world_size * cfg.padded_tokens_per_rank
            torch.remainder(
                torch.arange(
                    gather_index.numel(), dtype=torch.int32, device=gather_index.device
                ),
                dense_rows,
                out=gather_index,
            )
        if (
            self.local_workspace.data_ptr() % 128 != 0
            or self.shared_workspace.data_ptr() % 128 != 0
        ):
            raise RuntimeError("block-scaled mega workspaces must be 128-byte aligned.")

        if cfg.shared_expert:
            self._initialize_shared_expert_rows(kernel)
        if cfg.resolved_die_local_weights:
            from .die_local import sm_die_map

            region = kernel.scheduler.sm_die_table_region
            start = workspace.offset(region)
            table = sm_die_map(self.device).table()
            self.local_workspace[start : start + workspace.nbytes(region)].view(
                torch.int32
            ).copy_(table)
        base, peer_offsets = comm.compute_peer_offsets(
            self.shared_workspace, cfg.world_size
        )
        self._symmetric_base = base
        self._peer_offsets = peer_offsets
        self._staging_base, self._staging_peer_offsets = comm.compute_peer_offsets(
            self.x, cfg.world_size
        )

    def _initialize_shared_expert_rows(self, kernel) -> None:
        """Write the shared expert's routing records and top-k scores, and prepare its scale-factor staging.

        Its rows of the data pool follow the routed worst case (``SharedExpertTiles``):
        every token's record sends its FC2 row to this rank's own combine slot
        ``topk``, and its top-k score is one. They never change, so they are
        written once with the workspace. Its token scale factors (the canonical
        pool's rows past the routed ones) follow the staged input on every
        launch (``stage_shared_expert_scales``).
        """
        cfg = self.config
        comm_regions = kernel.token_comm
        workspace = kernel._device_workspace
        tiles = kernel.shared_expert_tiles
        tokens = torch.arange(tiles.tokens, dtype=torch.int64, device=self.device)
        slot_rows = (
            tokens // tiles.tile_rows * tiles.slot_rows + tokens % tiles.tile_rows
        )
        start = workspace.offset(comm_regions.token_src_metadata_region)
        records = self.local_workspace[
            start : start + workspace.nbytes(comm_regions.token_src_metadata_region)
        ].view(torch.int64)
        # TokenSrcMetadata: rank (16 bits), top-k slot (16 bits), token (32 bits).
        records[tiles.data_base + slot_rows] = (
            ((cfg.rank << 16) | cfg.num_topk) << 32
        ) | tokens
        start = workspace.offset(comm_regions.fc1_topk_scores_region)
        scores = self.local_workspace[
            start : start + workspace.nbytes(comm_regions.fc1_topk_scores_region)
        ].view(torch.float32)
        scores[
            tiles.data_base : tiles.data_base + tiles.token_tiles * tiles.slot_rows
        ] = 1.0
        hidden_sf = cfg.hidden // cfg.sf_vec_size
        start = workspace.offset(comm_regions.canonical_activation_sf_region)
        sf_rows = round_up(tiles.token_tiles * tiles.slot_rows, kernel.sf_padding_block)
        begin = start + tiles.sf_base * hidden_sf
        self._shared_expert_scales = self.local_workspace[
            begin : begin + sf_rows * hidden_sf
        ]
        self._shared_expert_scale_rows = slot_rows
        self._shared_expert_scale_scratch = torch.zeros(
            (sf_rows, hidden_sf), dtype=torch.uint8, device=self.device
        )

    def shared_expert_scale_layout(self) -> Tuple[torch.Tensor, int, int]:
        """(scale plane, token tile rows, slot rows) of the shared expert's token scale factors.

        The plane is the workspace's uint8 view of them: token ``t``'s scale
        column ``c`` (UE8M0, one per 32 values) belongs at the 32x4x4 atom
        swizzle of row ``t // tile_rows * slot_rows + t % tile_rows``, byte
        ``((row // 128) * (hidden // 128) + c // 4) * 512 + (row % 32) * 16 +
        (row // 32 % 4) * 4 + c % 4``. ``stage_shared_expert_scales`` writes it
        from ``x_sf``; a caller with its own staging kernel may write it there.
        """
        tiles = self.kernel.shared_expert_tiles
        return self._shared_expert_scales, tiles.tile_rows, tiles.slot_rows

    def stage_shared_expert_scales(self) -> None:
        """Copy the staged token scales (``x_sf``) into the shared expert's scale plane, every row of the capacity.

        The shared expert runs every row, so rows past a launch's token count
        keep whatever ``x_sf`` holds; their outputs are never read.
        """
        hidden_sf = self.config.hidden // self.config.sf_vec_size
        scratch = self._shared_expert_scale_scratch
        scratch.index_copy_(
            0,
            self._shared_expert_scale_rows,
            self.x_sf[:, :hidden_sf].view(torch.uint8),
        )
        rows = scratch.shape[0]
        self._shared_expert_scales.view(rows // 128, hidden_sf // 4, 32, 4, 4).copy_(
            scratch.view(rows // 128, 4, 32, hidden_sf // 4, 4).permute(0, 3, 2, 1, 4)
        )

    @classmethod
    def _build_kernel(cls, cfg: BlockScaledMoeConfig):
        import cutlass
        from ..engine import BlackwellInferenceGenphaseMegaMoE
        from ..engine.api import ImplDesc, ProblemDesc
        from ..engine.quant_def import CombineFormat
        from cutlass.cute.nvgpu import OperandMajorMode

        tiler = cfg.resolved_mma_tiler_mnk
        cluster_size = cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1]
        preferred_count: Optional[int] = None
        fallback_count: Optional[int] = None
        if cfg.fallback_cluster_shape_mn is None:
            launch_clusters = _max_active_clusters(cluster_size)
            if cfg.max_sm_count is not None:
                requested = cfg.max_sm_count // cluster_size
                if requested <= 0:
                    raise ValueError(
                        "max_sm_count must cover at least one full cluster."
                    )
                launch_clusters = min(launch_clusters, requested)
        else:
            # Mirror the drop's launch_cluster_configuration()/max_active_clusters()
            # mixed recipe (tester/host_utils.py): pack preferred clusters at their
            # occupancy limit, then convert the leftover fallback CTA capacity into
            # whole preferred-sized groups of fallback clusters.
            fallback_size = (
                cfg.fallback_cluster_shape_mn[0] * cfg.fallback_cluster_shape_mn[1]
            )
            if cluster_size % fallback_size != 0:
                raise ValueError(
                    "preferred cluster size must be a multiple of the fallback "
                    "cluster size."
                )
            preferred_count = _max_active_clusters(cluster_size)
            fallback_occupancy = _max_active_clusters(fallback_size)
            fallback_capacity = fallback_occupancy * fallback_size
            preferred_capacity = preferred_count * cluster_size
            if fallback_capacity < preferred_capacity:
                raise ValueError(
                    "fallback cluster CTA capacity must not be smaller than "
                    "preferred cluster CTA capacity."
                )
            split_factor = cluster_size // fallback_size
            remaining = (fallback_capacity - preferred_capacity) // fallback_size
            fallback_count = remaining // split_factor * split_factor
            launch_clusters = preferred_count + fallback_count // split_factor

        problem_desc = ProblemDesc(
            {
                "expert_count": cfg.num_total_experts,
                "intermediate_gateup_size": 2 * cfg.intermediate,
                "hidden_size": cfg.hidden,
                "quant_kind": cfg.quant_kind,
                "a_major_mode": OperandMajorMode.K,
                "b_major_mode": OperandMajorMode.K,
                "combine_format": CombineFormat.parse(
                    {
                        "bf16": "bf16",
                        "nvfp4": "16e2m1xbf16",
                        "mxfp8": "32e4m3xe8m0",
                    }[cfg.combine_dtype]
                ),
                "gate_up_clamp": cfg.gate_up_clamp,
                "situ_beta": cfg.situ_beta,
                "situ_linear_beta": cfg.situ_linear_beta,
                "world_size": cfg.world_size,
                "topk": cfg.num_topk,
                "topk_index_dtype": cutlass.Int32,
                "max_tokens_per_rank": cfg.padded_tokens_per_rank,
                "apply_topk_at_fc1": cfg.apply_routing_weights_before_fc2,
                **(
                    {
                        "shared_expert": cfg.shared_expert,
                        "shared_gate_up_clamp": cfg.shared_gate_up_clamp,
                        "shared_bf16_rounding": cfg.shared_bf16_rounding,
                    }
                    if cfg.shared_expert
                    else {}
                ),
            }
        )
        impl_fields = {
            "architecture": cls._architecture,
            "mma_instruction_mnk": (tiler[0], tiler[1], cfg.instruction_k),
            "mma_tiler_mnk": tiler,
            "mma_k_mode": "1x",
            "cluster_shape_mn": tuple(cfg.cluster_shape_mn),
            "use_2cta_instrs": tiler[0] == 256,
            "schedule_policy": tuple(cfg.schedule_policy),
            "token_padding_block": cfg.token_padding_block,
            "sf_padding_block": cfg.sf_padding_block,
            "work_id_mode": cfg.work_id_mode,
            "fc2_use_bulk": cfg.fc2_use_bulk,
            "epi_flag_batches": tuple(cfg.epi_flag_batches),
            "launch_cluster_count": launch_clusters,
            "token_in_flag_batch": cfg.token_in_flag_batch,
            "token_back_mode": cfg.token_back_mode,
            "reduce_topk_in_kernel": cfg.reduce_topk_in_kernel,
            "weight_evict": cfg.weight_evict,
            "combine_protocol": cfg.combine_protocol,
            "dispatch_protocol": cfg.dispatch_protocol,
            "weight_layout": cfg.resolved_weight_layout,
            "entry_prefetch_bytes": cfg.resolved_entry_prefetch_bytes,
        }
        if cfg.resolved_die_local_weights:
            impl_fields["die_local_weights"] = True
        if cfg.comm_pusher_ctas is not None:
            impl_fields["comm_pusher_ctas"] = cfg.comm_pusher_ctas
        if cfg.dispatch_rounds != 1:
            impl_fields["dispatch_rounds"] = cfg.dispatch_rounds
            impl_fields["dispatch_round_mode"] = cfg.dispatch_round_mode
        if cfg.comm_push_depth is not None:
            impl_fields["comm_push_depth"] = cfg.comm_push_depth
        if cfg.comm_pull_depth is not None:
            impl_fields["comm_pull_depth"] = cfg.comm_pull_depth
        if cfg.comm_pull_chunk_rows is not None:
            impl_fields["comm_pull_chunk_rows"] = cfg.comm_pull_chunk_rows
        if cfg.comm_pull_lead_rows is not None:
            impl_fields["comm_pull_lead_rows"] = cfg.comm_pull_lead_rows
        if cfg.comm_exchange_ctas is not None:
            impl_fields["comm_exchange_ctas"] = cfg.comm_exchange_ctas
        if cfg.pull_warps is not None:
            impl_fields["pull_warps"] = cfg.pull_warps
        if cfg.resolved_gather_b_warps is not None:
            impl_fields["gather_b_warps"] = cfg.resolved_gather_b_warps
        if cfg.pull_acquire_in_copy is not None:
            impl_fields["pull_acquire_in_copy"] = cfg.pull_acquire_in_copy
        if cfg.pull_walk_dedup:
            impl_fields["pull_walk_dedup"] = True
        if cfg.pull_tile_ready is not None:
            impl_fields["pull_tile_ready"] = cfg.pull_tile_ready
        if cfg.resolved_fc1_token_pool:
            impl_fields["fc1_token_pool"] = True
            if cfg.resolved_fc1_token_pool_capacity:
                impl_fields["fc1_token_pool_capacity"] = (
                    cfg.resolved_fc1_token_pool_capacity
                )
                impl_fields["fc1_token_pool_fallback"] = cfg.fc1_token_pool_fallback
        if cfg.refine_exchange_ctas is not None:
            impl_fields["refine_exchange_ctas"] = cfg.refine_exchange_ctas
        if cfg.round_publish_lag is not None:
            impl_fields["round_publish_lag"] = cfg.round_publish_lag
        if cfg.round_publish_delay_ns:
            impl_fields["round_publish_delay_ns"] = cfg.round_publish_delay_ns
        if cfg.refine_gate_round0_only:
            impl_fields["refine_gate_round0_only"] = True
        if cfg.comm_stamps:
            impl_fields["stamps"] = True
        if cfg.resolved_just_in_time_claims:
            impl_fields["just_in_time_claims"] = True
        if cfg.claim_lead_k_tiles is not None:
            impl_fields["claim_lead_k_tiles"] = cfg.claim_lead_k_tiles
        if cfg.resolved_discard_combine_slots:
            impl_fields["discard_combine_slots"] = True
        if cfg.resolved_combine_warp_stores:
            impl_fields["combine_warp_stores"] = True
        if cfg.resolved_token_ab_stages:
            impl_fields["token_ab_stages"] = cfg.resolved_token_ab_stages
        if cfg.resolved_alternate_token_tile_n:
            impl_fields["alternate_tile_n"] = cfg.resolved_alternate_token_tile_n
            impl_fields["alternate_token_ab_stages"] = (
                cfg.resolved_alternate_token_ab_stages
            )
            impl_fields["alternate_weight_evict"] = (
                cfg.alternate_weight_evict or cfg.weight_evict
            )
            impl_fields["token_tile_choice"] = cfg.token_tile_choice
        if cfg.fc2_tma_stages is not None:
            impl_fields["fc2_tma_stages"] = cfg.fc2_tma_stages
        if cfg.pipelined_groups:
            impl_fields["pipelined_groups"] = True
        if cfg.resolved_shared_expert_entry:
            impl_fields["shared_expert_entry"] = True
        if cfg.shared_expert:
            impl_fields["shared_fc2_placement"] = cfg.shared_fc2_placement
        if cfg.resolved_shared_alternate_token_tile:
            impl_fields["shared_token_tile_rows"] = cfg.resolved_alternate_token_tile_n
        head_token_tiles = cls._shared_head_token_tiles(cfg, launch_clusters)
        if head_token_tiles is not None:
            impl_fields["shared_head_token_tiles"] = head_token_tiles
        if cfg.fallback_cluster_shape_mn is not None:
            impl_fields["fallback_cluster_shape_mn"] = tuple(
                cfg.fallback_cluster_shape_mn
            )
            impl_fields["preferred_cluster_count"] = preferred_count
            impl_fields["fallback_cluster_count"] = fallback_count
        return BlackwellInferenceGenphaseMegaMoE(problem_desc, ImplDesc(impl_fields))

    @classmethod
    def _shared_head_token_tiles(
        cls, cfg: BlockScaledMoeConfig, launch_clusters: int
    ) -> Optional[int]:
        """Token tiles of the shared expert's entry head (``shared_head_token_tiles``), or None for all of them.

        Under "window" the head holds the shared FC1 tiles that finish within
        the dispatch's window before its first routed FC1 tile
        (``genphase.dispatch_window_us``, ``genphase.shared_head_token_tiles``)
        on the clusters the communication grid's CTAs leave the main kernel
        there. A cap needs the FC2 tiles at the tail.
        """
        if (
            cfg.shared_head_token_tiles is None
            or not cfg.resolved_shared_expert_entry
            or cfg.shared_fc2_placement != "tail"
        ):
            return None
        token_tiles, tile_rows = cfg.shared_expert_token_tiling
        head = cfg.shared_head_token_tiles
        if head == "window":
            from . import genphase
            from ..engine import BlackwellInferenceGenphaseMegaMoE as kernel_class

            window_us = genphase.dispatch_window_us(
                cfg.dispatch_protocol,
                cfg.max_tokens_per_rank,
                cfg.world_size,
                cfg.num_total_experts,
                cfg.num_topk,
                cfg.hidden,
            )
            comm_ctas = cfg.comm_pusher_ctas
            if comm_ctas is None:
                comm_ctas = (
                    kernel_class.pull_pusher_cta_count
                    if cfg.dispatch_protocol == "pull"
                    else kernel_class.pusher_cta_count
                )
            cluster_size = cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1]
            head = genphase.shared_head_token_tiles(
                window_us,
                hidden=cfg.hidden,
                shared_intermediate=cfg.intermediate,
                tile_rows=tile_rows,
                token_tiles=token_tiles,
                clusters=max(1, launch_clusters - -(-(comm_ctas + 1) // cluster_size)),
                after_choice=cfg.resolved_shared_alternate_token_tile,
            )
        return head if head < token_tiles else None

    def recorded_token_tile_n(self) -> Optional[int]:
        """The token tile N the latest launch ran, or None without an alternate token tile.

        Reads the main kernel's record (one int32 in the local workspace) after a
        device synchronization.
        """
        region = getattr(self.kernel, "token_tile_record_region", None)
        if not getattr(self.kernel, "alternate_tile_n", 0) or region is None:
            return None
        torch.cuda.synchronize()
        start = self.kernel._device_workspace.offset(region)
        return int(self.local_workspace[start : start + 4].view(torch.int32).item())

    def recorded_token_pool(self) -> Optional[Tuple[int, int]]:
        """The latest launch's routes and token pool capacity, or None without a pool capacity.

        Token tiles whose routes end within that capacity in processing order
        loaded from the pool and the rest gathered from the landing slots, so a
        capacity of 0 means every tile fell back. Reads the main kernel's record
        (two int32 in the local workspace) after a device synchronization.
        """
        region = getattr(self.kernel, "token_pool_record_region", None)
        if not getattr(self.kernel, "token_pool_capacity", 0) or region is None:
            return None
        torch.cuda.synchronize()
        start = self.kernel._device_workspace.offset(region)
        words = self.local_workspace[start : start + 8].view(torch.int32)
        return int(words[0].item()), int(words[1].item())

    def note_staged_tokens(self, num_tokens: int) -> None:
        self._staged_tokens = int(num_tokens)

    def staged_tokens(self) -> Optional[int]:
        return self._staged_tokens

    def _peer_mapper(self, staging: bool = False):
        """Peer offsets of the shared workspace, or of the staging allocation."""
        from ..engine.communication.nvlink_domain.symmetric_buffer import (
            SymmetricBufferHost,
        )

        base, offsets = (
            (self._staging_base, self._staging_peer_offsets)
            if staging
            else (self._symmetric_base, self._peer_offsets)
        )
        return SymmetricBufferHost(
            base_address=base,
            offsets=tuple(offsets),
            rank=self.config.rank,
            max_ranks=self.config.world_size,
        )

    def _runtime_kwargs(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
        shared_legs: Optional[
            Tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights]
        ] = None,
    ) -> dict:
        stream = torch.cuda.current_stream()
        shared = {}
        if shared_legs is not None:
            shared = {
                "shared_fc1_weight": _to_cute(shared_legs[0][0]),
                "shared_fc1_weight_sf": _to_cute(shared_legs[0][1]),
                "shared_fc2_weight": _to_cute(shared_legs[1][0]),
                "shared_fc2_weight_sf": _to_cute(shared_legs[1][1]),
            }
        return {
            **shared,
            "activation": _to_cute(self.x),
            "activation_sf": _to_cute(self.x_sf),
            "topk_indices": _to_cute(self.topk_idx),
            "topk_scores": _to_cute(self.topk_weights, assumed_align=4),
            "fc1_weight": _to_cute(transformed_l1[0]),
            "fc1_weight_sf": _to_cute(transformed_l1[1]),
            "fc2_weight": _to_cute(transformed_l2[0]),
            "fc2_weight_sf": _to_cute(transformed_l2[1]),
            "output_activation": _to_cute(self.output_activation),
            "local_workspace": _to_cute_ptr(self.local_workspace),
            "shared_workspace": _to_cute_ptr(self.shared_workspace),
            "peer_rank_ptr_mapper_host": self._peer_mapper(),
            "staging_peer_rank_ptr_mapper_host": self._peer_mapper(staging=True),
            "stream": _cu_stream(stream),
            **{
                name: _to_cute(getattr(self, name), assumed_align=4)
                for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
                if self.config.quant_kind == "nvfp4"
            },
        }

    def launch(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
        shared_legs: Optional[
            Tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights]
        ] = None,
    ) -> None:
        """Compile on first use, then launch the fused mega kernel.

        ``shared_legs`` are the shared expert's (FC1, FC2) legs, which a
        workspace with a shared expert requires.
        """
        import cutlass.cute as cute

        if self._destroyed:
            raise RuntimeError("cannot launch a destroyed block-scaled workspace.")
        if (
            self.device.type == "cuda"
            and self.device.index != torch.cuda.current_device()
        ):
            raise RuntimeError(
                "block-scaled workspace must launch on the CUDA device that owns it."
            )
        if (shared_legs is None) != (self.config.shared_expert is None):
            raise ValueError(
                "a workspace with a shared expert launches with its weights, and one without with none."
            )
        key = (
            transformed_l1[0].data_ptr(),
            transformed_l1[1].data_ptr(),
            transformed_l2[0].data_ptr(),
            transformed_l2[1].data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if shared_legs is not None:
            key += tuple(t.data_ptr() for leg in shared_legs for t in leg)
        if self._compiled is None or self._launch_key != key:
            if self._compiled is None:
                comm.ensure_not_capturing("block-scaled mega kernel compile")
            # DLPack metadata binding and stream conversion are host-only.
            # A warmed workspace may bind another layer or stream in capture.
            kwargs = self._runtime_kwargs(transformed_l1, transformed_l2, shared_legs)
            if self._compiled is None:
                self._compiled = cute.compile(self.kernel, **kwargs)
            self._launch_key = key
            self._launch_kwargs = kwargs
            # Keep pointers alive until the cached bindings are replaced.
            self._launch_weights = (transformed_l1, transformed_l2, shared_legs)
        if (
            self.config.reduce_topk_in_kernel
            and self.config.kernel_variant != "genphase"
        ):
            # red.add accumulation base: zero the output every launch. GenPhase
            # zeroes it inside the kernel.
            self.output_activation.zero_()
        elif not getattr(self.kernel, "masks_invalid_combine_slots", False):
            # Invalid routes do not write a combine slot. Clear those slots on
            # every launch so a masked route cannot reuse an earlier result.
            invalid = (self.topk_idx < 0) | (
                self.topk_idx >= self.config.num_total_experts
            )
            self._pre_reduced_activation.masked_fill_(invalid.unsqueeze(-1), 0)
            if self.config.combine_dtype != "bf16":
                # Clear BF16 amax for NVFP4; E8M0 byte 127 is unit scale for MXFP8.
                sf_byte = 0 if self.config.combine_dtype == "nvfp4" else 127
                self._pre_reduced_activation_sf.masked_fill_(
                    invalid.unsqueeze(-1), sf_byte
                )
        self._compiled(**self._launch_kwargs)

    def destroy(self) -> None:
        if self._destroyed:
            return
        comm.ensure_not_capturing("block-scaled mega workspace free")
        # Releasing symmetric memory does not wait for in-flight work: drain this
        # rank's device first. Peers' launches may also write here, so callers
        # destroy only after every rank has synchronized.
        torch.cuda.synchronize()
        self._compiled = None
        self._launch_kwargs = None
        self._launch_weights = None
        self._pre_reduced_activation = None
        self._pre_reduced_activation_sf = None
        for name in ("x", "x_sf", "topk_weights", "shared_workspace"):
            comm.free_sym_tensor(getattr(self, name, None))
            setattr(self, name, None)
        if self.config.reduce_topk_in_kernel:
            comm.free_sym_tensor(self.output_activation)
        self.output_activation = None
        self.local_workspace = None
        self.topk_idx = None
        self.fc1_alpha = self.fc2_alpha = self.fc1_norm_const = None
        self._destroyed = True


def _expected_weight_shapes(
    cfg: BlockScaledMoeConfig,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
    """(fc1 shape, fc2 shape, fc1 SF numel/expert, fc2 SF numel/expert).

    Die-local weights hold each expert's two row halves as two entries of half
    the rows (``die_local.die_local_weight_leg``).
    """
    experts = cfg.experts_per_rank
    fc1_out = 2 * cfg.intermediate
    fc2_out = cfg.hidden
    vec = cfg.sf_vec_size
    if cfg.resolved_die_local_weights:
        experts, fc1_out, fc2_out = 2 * experts, fc1_out // 2, fc2_out // 2
    if cfg.quant_kind in ("nvfp4", "mxfp4_mxfp8"):
        fc1_shape = (experts, cfg.hidden // 2, fc1_out)
        fc2_shape = (experts, cfg.intermediate // 2, fc2_out)
    else:
        fc1_shape = (experts, cfg.hidden, fc1_out)
        fc2_shape = (experts, cfg.intermediate, fc2_out)
    fc1_sf = swizzled_flat_sf_size(fc1_out, cfg.hidden // vec)
    fc2_sf = swizzled_flat_sf_size(fc2_out, cfg.intermediate // vec)
    return fc1_shape, fc2_shape, fc1_sf, fc2_sf


def _expected_shared_weight_shapes(
    cfg: BlockScaledMoeConfig,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
    """(fc1 shape, fc2 shape, fc1 SF numel, fc2 SF numel) of the shared expert's K-major E4M3 legs."""
    vec = cfg.sf_vec_size
    fc1_shape = (1, cfg.hidden, 2 * cfg.intermediate)
    fc2_shape = (1, cfg.intermediate, cfg.hidden)
    fc1_sf = swizzled_flat_sf_size(2 * cfg.intermediate, cfg.hidden // vec)
    fc2_sf = swizzled_flat_sf_size(cfg.hidden, cfg.intermediate // vec)
    return fc1_shape, fc2_shape, fc1_sf, fc2_sf


def _validate_shared_weight_legs(
    shared_legs, cfg: BlockScaledMoeConfig, device: torch.device
) -> None:
    """Check a shared expert's (FC1, FC2) legs, as ``preprocess_*_weights(..., quant_kind="mxfp8_e4m3")`` returns them."""
    if (shared_legs is None) != (cfg.shared_expert is None):
        raise ValueError(
            "shared expert weights are required with shared_expert and rejected without it."
        )
    if shared_legs is None:
        return
    if not isinstance(shared_legs, tuple) or len(shared_legs) != 2:
        raise ValueError("shared expert weights must be an (fc1, fc2) pair of legs.")
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_shared_weight_shapes(cfg)
    for name, leg, shape, sf in (
        ("shared fc1", shared_legs[0], fc1_shape, fc1_sf),
        ("shared fc2", shared_legs[1], fc2_shape, fc2_sf),
    ):
        if (
            not isinstance(leg, tuple)
            or len(leg) != 2
            or not all(isinstance(t, torch.Tensor) for t in leg)
        ):
            raise ValueError(f"{name} must be a (weight, scale) tensor pair.")
        weight, scale = leg
        if tuple(weight.shape) != shape or tuple(scale.shape) != (1, sf):
            raise ValueError(
                f"{name} must have weight shape {shape} and scale shape {(1, sf)}, got "
                f"{tuple(weight.shape)} and {tuple(scale.shape)}."
            )
        if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
            raise ValueError(
                f"{name} requires float8_e4m3fn weights and float8_e8m0fnu scales."
            )
        if not weight.permute(0, 2, 1).is_contiguous() or not scale.is_contiguous():
            raise ValueError(
                f"{name} requires K-major weights and contiguous scale planes."
            )
        if getattr(weight, "_mega_weight_layout", "kmajor") != "kmajor":
            raise ValueError(f"{name} weights must be K-major.")
        if any(t.device != device or t.data_ptr() % 16 for t in leg):
            raise ValueError(
                f"{name} must be on {device} with 16-byte aligned storage."
            )


def _launch_weights(
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    cfg: BlockScaledMoeConfig,
) -> Tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights]:
    """The legs the kernel reads: under ``die_local_weights`` their die-local copies."""
    if not cfg.resolved_die_local_weights:
        return transformed_l1, transformed_l2
    from .die_local import die_local_weight_legs

    if not all(
        getattr(leg[0], "_mega_die_local", False)
        or hasattr(leg[0], "_mega_die_local_leg")
        for leg in (transformed_l1, transformed_l2)
    ):
        comm.ensure_not_capturing("die-local weight allocation")
    return die_local_weight_legs(transformed_l1, transformed_l2)


def _validate_weight_leg(
    name: str,
    leg: TransformedBlockScaledWeights,
    expected_weight_shape: Tuple[int, ...],
    expected_sf_numel: int,
    cfg: BlockScaledMoeConfig,
    device: torch.device,
) -> None:
    if (
        not isinstance(leg, tuple)
        or len(leg) != 2
        or not all(isinstance(t, torch.Tensor) for t in leg)
    ):
        raise ValueError(f"{name} must be a (weight, scale) tensor pair.")
    weight, scale = leg
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"{name} weight shape {tuple(weight.shape)} != expected "
            f"{expected_weight_shape}."
        )
    if tuple(scale.shape) != (expected_weight_shape[0], expected_sf_numel):
        raise ValueError(
            f"{name} scale numel {scale.numel()} != expected "
            f"{expected_sf_numel * expected_weight_shape[0]}."
        )
    if (
        weight.dtype != cfg.torch_weight_data_dtype
        or scale.dtype != cfg.torch_act_sf_dtype
    ):
        raise ValueError(
            f"{name} requires {cfg.torch_weight_data_dtype} weights and {cfg.torch_act_sf_dtype} scales."
        )
    if not weight.permute(0, 2, 1).is_contiguous() or not scale.is_contiguous():
        raise ValueError(
            f"{name} requires K-major weights and contiguous scale planes."
        )
    die_local = cfg.resolved_die_local_weights
    if getattr(weight, "_mega_die_local", False) != die_local:
        raise ValueError(
            f"{name} weights are {'' if die_local else 'not '}expected die-local."
        )
    layout = getattr(weight, "_mega_weight_layout", "kmajor")
    if layout != cfg.resolved_weight_layout:
        raise ValueError(
            f"{name} weights were prepared {layout} but the config expects "
            f"{cfg.resolved_weight_layout}."
        )
    if any(t.device != device or t.data_ptr() % 16 for t in leg):
        raise ValueError(f"{name} must be on {device} with 16-byte aligned storage.")


def block_scaled_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    symm_buffer: BlockScaledSymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    fast_math: bool = True,  # accepted for mega API parity; the kernel has no toggle
    sync: bool = False,
    shared_weights: Optional[
        Tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights]
    ] = None,
) -> Optional[torch.Tensor]:
    """Fused dispatch, FC1, SwiGLU/SiTU, FC2, and combine; writes ``y[:num_tokens]``.

    The caller must have staged ``symm_buffer.x`` / ``.x_sf`` and the routing
    slices first, and with a shared expert its scale factors
    (``stage_shared_expert_scales``). With ``y=None`` returns a workspace view
    (valid under stream ordering until the next launch). ``shared_weights``
    are the shared expert's (FC1, FC2) legs, required with ``shared_expert``:
    the output is then the routed experts' sum plus the shared expert's.
    """
    if not fast_math:
        import warnings

        warnings.warn(
            "fast_math=False is a no-op for the block-scaled block-scaled kernel.",
            stacklevel=2,
        )
    cfg = symm_buffer.config
    if y is not None and (
        y.ndim != 2
        or y.shape[1] != cfg.hidden
        or y.dtype != torch.bfloat16
        or y.device != symm_buffer.device
    ):
        raise ValueError(
            f"y must be a BF16 matrix with {cfg.hidden} columns on {symm_buffer.device}."
        )
    if num_tokens is None:
        num_tokens = int(y.shape[0]) if y is not None else symm_buffer.staged_tokens()
    if num_tokens is None:
        raise ValueError(
            "num_tokens unset and no tokens were staged; call the backend's "
            "stage_inputs (or note_staged_tokens) first."
        )
    if not (0 <= num_tokens <= cfg.max_tokens_per_rank):
        raise ValueError(
            f"num_tokens ({num_tokens}) out of range [0, {cfg.max_tokens_per_rank}]."
        )
    if y is not None and y.shape[0] < num_tokens:
        raise ValueError("y does not have enough rows for num_tokens.")

    transformed_l1, transformed_l2 = _launch_weights(
        transformed_l1, transformed_l2, cfg
    )
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(cfg)
    _validate_weight_leg(
        "fc1", transformed_l1, fc1_shape, fc1_sf, cfg, symm_buffer.device
    )
    _validate_weight_leg(
        "fc2", transformed_l2, fc2_shape, fc2_sf, cfg, symm_buffer.device
    )
    _validate_shared_weight_legs(shared_weights, cfg, symm_buffer.device)

    symm_buffer.launch(transformed_l1, transformed_l2, shared_weights)
    out = symm_buffer.output_activation[:num_tokens]
    if y is None:
        if sync:
            torch.cuda.synchronize()
        return out
    y[:num_tokens].copy_(out)
    if sync:
        torch.cuda.synchronize()
    return None


def block_scaled_mega_launch_thunk(
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    symm_buffer: BlockScaledSymmBuffer,
    shared_weights: Optional[
        Tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights]
    ] = None,
):
    """Zero-arg relauncher over pre-staged inputs, for steady-state timing loops."""
    cfg = symm_buffer.config
    transformed_l1, transformed_l2 = _launch_weights(
        transformed_l1, transformed_l2, cfg
    )
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(cfg)
    _validate_weight_leg(
        "fc1", transformed_l1, fc1_shape, fc1_sf, cfg, symm_buffer.device
    )
    _validate_weight_leg(
        "fc2", transformed_l2, fc2_shape, fc2_sf, cfg, symm_buffer.device
    )
    _validate_shared_weight_legs(shared_weights, cfg, symm_buffer.device)

    def _thunk() -> None:
        symm_buffer.launch(transformed_l1, transformed_l2, shared_weights)

    return _thunk


__all__ = [
    "ALTERNATE_TOKEN_TILE_N",
    "BlockScaledMoeConfig",
    "BlockScaledSymmBuffer",
    "BlockScaledQuantKind",
    "TransformedBlockScaledWeights",
    "block_scaled_mega_launch_thunk",
    "block_scaled_mega_moe",
    "chosen_token_tile_n",
]
