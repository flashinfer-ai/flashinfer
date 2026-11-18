"""Compile-time schema for lightweight MegaMoE pipeline timing."""

from __future__ import annotations

# The timing probes are enabled per kernel instance through
# ``MegaMoEDiagnostics.pipeline_timing`` / ``fc_tile_timeline``
# (pre_unification/comm/megamoe_config.py); this module holds only the schema.

# The fused kernel currently launches one resident CTA per SM (152 on GB300).
# Keep explicit headroom and fail compilation if a future launch exceeds it.
PIPELINE_TIMING_MAX_CTAS = 512
PIPELINE_TIMING_MAX_WARPS = 16

# These are overlapping wall-clock lifecycles, not additive work estimates.
# FC1 and FC2 stay fused into one expert-stage envelope so profiling adds only
# one start/end pair per functional worker rather than per-tile instrumentation.
PIPELINE_STAGE_DISPATCH = 0
PIPELINE_STAGE_FC1_FC2 = 1
PIPELINE_STAGE_COMBINE = 2
# GIN-only sub-envelope nested inside the dispatch lifecycle. `push` is the
# relay-warp span from the end of the inter-node metadata exchange to the end
# of Stage 1 `internode_push` (pack, range put, flush, credit); the relay
# warps' Stage 2 fan-out is therefore `[push end, dispatch end]`. It stays
# zero, and the host reporters skip it, when NCCL GIN is disabled. Slot 4
# (`relay_copy`) has had no device writer since the staged Stage 3 path was
# removed on 2026-09-21; the slot is kept so older dumps still decode.
PIPELINE_STAGE_PUSH = 3
PIPELINE_STAGE_RELAY_COPY = 4
# `meta` is the relay-warp span from the end of dispatch_barrier to the push
# start (inter_meta_exchange); `fanout` is the relay-warp span from the push
# end to the end of Stage 2 relay_fanout. With `push` they partition the relay
# warps' dispatch lifecycle after dispatch_prep and dispatch_barrier.
PIPELINE_STAGE_META = 5
PIPELINE_STAGE_FANOUT = 6
# SM0-only sub-envelopes (one writer each) that partition the coalesced push
# and the fan-out prologue: `push_sync` is pack plus the relay grid sync,
# `push_issue` the range puts, `push_flush` the flush plus strong credits;
# `fanout_wait` is SM0's wait for the remote credits and `fanout_sync` the
# relay grid sync that releases every CTA's fan-out.
PIPELINE_STAGE_PUSH_SYNC = 7
PIPELINE_STAGE_PUSH_ISSUE = 8
PIPELINE_STAGE_PUSH_FLUSH = 9
PIPELINE_STAGE_FANOUT_WAIT = 10
PIPELINE_STAGE_FANOUT_SYNC = 11
# SM0 dispatch-warp-0 sub-envelopes of inter_meta_exchange: Step 3 metadata
# puts, the Step 5a wait for the peers' metadata, Step 5b redistribution up to
# its closing NVLink barrier, Step 5c layout, and the Step 6 publication
# barrier. `fanout_route/copy/publish` are worker-local sums over the messages
# each relay warp fanned out (route decode plus slot reservation, row copies,
# fence plus release counts), anchored at that warp's loop start; the reduced
# envelope therefore approximates the busiest warp's summed phase time and is
# an attribution aid, not a lifecycle envelope. `tail` brackets kernel_tail on
# the dispatch warps.
PIPELINE_STAGE_META_STEP3 = 12
PIPELINE_STAGE_META_WAIT = 13
PIPELINE_STAGE_META_REDIST = 14
PIPELINE_STAGE_META_LAYOUT = 15
PIPELINE_STAGE_META_PUBLISH = 16
PIPELINE_STAGE_FANOUT_ROUTE = 17
PIPELINE_STAGE_FANOUT_COPY = 18
PIPELINE_STAGE_FANOUT_PUBLISH = 19
PIPELINE_STAGE_TAIL = 20
# Slots 21 and 22 (`relay_wait`, `relay_work`) were the per-dispatch-warp
# sums inside the staged Stage 3 relay_copy; they have had no device writer
# since 2026-09-21 and are kept so older dumps still decode.
# `fanout_wait_slot0..2` are SM0 sub-envelopes from the
# fan-out wait start to the end of the sequential credit wait of source slot
# 0, 1, and 2; slots that do not exist on a shape are never written.
PIPELINE_STAGE_RELAY_WAIT = 21
PIPELINE_STAGE_RELAY_WORK = 22
PIPELINE_STAGE_FANOUT_WAIT_SLOT0 = 23
PIPELINE_STAGE_FANOUT_WAIT_SLOT1 = 24
PIPELINE_STAGE_FANOUT_WAIT_SLOT2 = 25
# SM0 dispatch-warp-0 split of `meta_redist`: `meta_redist_issue` is the Step
# 5b redistribution itself (CTA barrier, the peer atomics and the closing
# system fence), `meta_redist_sync` the node-local NVLink barrier that follows
# it (with its grid syncs when they are enabled).
PIPELINE_STAGE_META_REDIST_ISSUE = 26
PIPELINE_STAGE_META_REDIST_SYNC = 27
# Relay-warp probes inside Stage 1 internode_push, one writer per relay warp:
# `push_prologue` is the span from function entry to the first destination
# slot's loop head (handle/window setup); `push_pack`, `push_pack_route` and
# `push_pack_copy` are worker-local sums over the messages the warp packed
# (whole message; tokens_to_node load through the route/weight stores; DATA,
# SF and padding stores), anchored at the loop head like the fan-out sums, so
# the reduced envelope approximates the busiest warp (attribution only).
PIPELINE_STAGE_PUSH_PROLOGUE = 28
PIPELINE_STAGE_PUSH_PACK = 29
PIPELINE_STAGE_PUSH_PACK_ROUTE = 30
PIPELINE_STAGE_PUSH_PACK_COPY = 31
# `push_prologue_setup` is the first part of `push_prologue`: function entry
# to the end of the GIN handle, team, window and base-pointer setup (the rest
# of the prologue is the per-slot smem loads and the first ballot).
PIPELINE_STAGE_PUSH_PROLOGUE_SETUP = 32
# Combine diagnostics: each relay worker stamps the gather-reduce interval of
# its LAST message into exactly one of these two stages, by the path the
# message ended on (scratch = one 2 KB round in flight per warp; ring = the
# deep-prefetch ring in the freed AB stage buffers). The stage envelope end
# therefore shows how late the slow-path reductions finish relative to the
# FC end, and the worker count how many last messages took each path.
PIPELINE_STAGE_COMBINE_REDUCE_SCRATCH = 33
PIPELINE_STAGE_COMBINE_REDUCE_RING = 34
# Epilogue warp 0 of each CTA: from the return of the epilogue tile loop
# (last accumulator consumed, tail flag fire done) to the end of the
# early ring-release hook; and in kernel_tail from entry to the ring flag
# store. Together they place the ring availability against the last FC2
# epilogue and the GEMM warp role exit.
PIPELINE_STAGE_FC_EPI_EXIT = 35
PIPELINE_STAGE_FC_RING_RELEASE = 36
# Token-back warps: from the (possibly deferred, field 58) start of a warp's
# intra-node FC2 row pushes to its last push. Places the NVLink push window
# against the FC stragglers and the combine critical path.
PIPELINE_STAGE_TOKEN_BACK = 37
# Dispatch warps, field 61 (staged direct fan-out): from the start of the
# owner-side scale-factor materialization walk to its last readiness release
# (per warp; the rank envelope end is when the last relay row became ready).
PIPELINE_STAGE_SF_MATERIALIZE = 38
PIPELINE_STAGE_NAMES = (
    "dispatch", "fc1_fc2", "combine", "push", "relay_copy", "meta", "fanout",
    "push_sync", "push_issue", "push_flush", "fanout_wait", "fanout_sync",
    "meta_step3", "meta_wait", "meta_redist", "meta_layout", "meta_publish",
    "fanout_route", "fanout_copy", "fanout_publish", "tail",
    "relay_wait", "relay_work", "fanout_wait_slot0", "fanout_wait_slot1",
    "fanout_wait_slot2", "meta_redist_issue", "meta_redist_sync",
    "push_prologue", "push_pack", "push_pack_route", "push_pack_copy",
    "push_prologue_setup", "combine_reduce_scratch", "combine_reduce_ring",
    "fc_epi_exit", "fc_ring_release", "token_back", "sf_materialize",
)
PIPELINE_TIMING_EDGES = 2
PIPELINE_TIMING_SHAPE = (
    PIPELINE_TIMING_MAX_CTAS,
    PIPELINE_TIMING_MAX_WARPS,
    len(PIPELINE_STAGE_NAMES),
    PIPELINE_TIMING_EDGES,
)

# Per-tile FC1/FC2 timeline (diagnostic; MegaMoEDiagnostics.fc_tile_timeline,
# which requires pipeline_timing).
# One record per work tile, indexed by the order in which the CTA's consumer
# warps see tiles (they share one scheduler ring, so the k-th tile is the same
# tile in every warp): the TMA-B warp writes identity, the claim edge and the
# readiness edge (after the dispatch-release or fc1-done spin), the MMA warp
# the mainloop end, epilogue warp 0 the epilogue end. Slots past
# FC_TILE_TIMELINE_MAX_TILES are dropped and the host reports the truncation.
FC_TILE_TIMELINE_MAX_TILES = 128
FC_TILE_FIELD_NAMES = (
    "phase", "expert", "tile_m", "tile_n",
    "claim_ns", "ready_ns", "mma_end_ns", "epi_end_ns",
    # %clock64 cycles inside the tile's k-loop: TMA-B time blocked in
    # acquire_and_advance (no free smem stage) and its whole loop; MMA time
    # blocked in wait_and_advance (data not yet in smem) and its whole loop.
    "tmab_wait_cyc", "tmab_loop_cyc", "mma_wait_cyc", "mma_loop_cyc",
    # Finer split of the MMA warp's loop: cycles in the accumulator-stage
    # acquire before the k-loop, and, summed over the k-loop, in the two SF
    # smem-to-TMEM copies, in the MMA descriptor/issue code, and in the
    # stage release.
    "mma_acquire_cyc", "mma_s2t_cyc", "mma_issue_cyc", "mma_release_cyc",
)
FC_TILE_PHASE_FC1 = 1
FC_TILE_PHASE_FC2 = 2
FC_TILE_TIMELINE_SHAPE = (
    PIPELINE_TIMING_MAX_CTAS,
    FC_TILE_TIMELINE_MAX_TILES,
    len(FC_TILE_FIELD_NAMES),
)

# Fine combine profiling keeps the lifecycle landmarks and adds worker-local
# accumulated work. Each relay worker writes once after the final grid sync;
# instrumentation performs no global-memory writes inside the message loop.
COMBINE_TIMING_FC2_WAIT_START = 0
COMBINE_TIMING_ROUTE_DONE = 1
COMBINE_TIMING_REDUCE_DONE = 2
COMBINE_TIMING_ISSUE_DONE = 3
COMBINE_TIMING_FLUSH_CREDIT_DONE = 4
COMBINE_TIMING_ARRIVAL_DONE = 5
COMBINE_TIMING_SYNC_DONE = 6
COMBINE_TIMING_FC2_WAIT_NS = 7
COMBINE_TIMING_ROUTE_INCLUSIVE_NS = 8
COMBINE_TIMING_REDUCE_NS = 9
COMBINE_TIMING_PUBLISH_ISSUE_NS = 10
COMBINE_TIMING_MESSAGE_COUNT = 11
COMBINE_TIMING_WAITED_HIT_COUNT = 12
COMBINE_TIMING_MAX_FC2_WAIT_NS = 13
COMBINE_TIMING_LANDMARK_NAMES = (
    "fc2_wait_start",
    "route_done",
    "reduce_done",
    "issue_done",
    "flush_credit_done",
    "arrival_done",
    "sync_done",
    "fc2_wait_ns",
    "route_inclusive_ns",
    "reduce_ns",
    "publish_issue_ns",
    "message_count",
    "waited_hit_count",
    "max_fc2_wait_ns",
)
COMBINE_TIMING_LANDMARK_SHAPE = (
    PIPELINE_TIMING_MAX_CTAS,
    PIPELINE_TIMING_MAX_WARPS,
    len(COMBINE_TIMING_LANDMARK_NAMES),
)
