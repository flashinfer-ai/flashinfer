# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Genphase MegaMoE token communication.

Every dispatch sends this rank's routing at kernel entry, with no routing
dependency. The Sentinel dispatch sends the payload at entry as well, so a
single cross-rank transfer sits on its critical path; under the pull dispatch
the target ranks fetch the payload once the routing has arrived:

  exchange CTA c   writes this rank's whole top-k planes into peer c's inbox
                   (Sentinel), or sends every peer its count row and its routes
                   among chunk c of this rank's top-k plane (pull; see
                   "Expert-id lists")
  pusher CTA p     sends a slice of this rank's tokens to every peer that wants
                   them (Sentinel only)

The rest runs on the target rank:

  Sentinel     the exchange CTAs turn into planners that each count every
               source's plane, rank the experts and scatter their own source's
               routes into the index lists, then leave ("Sentinel top-k
               inbox"); the helper and the pushers refine the landed rows,
               verifying their bytes and rewriting their scale factors into the
               layout the block-scaled MMA wants
  pull         one helper CTA folds the received count rows into per-expert
               sizes, the processing order and the row-space prefixes; the
               exchange and pusher CTAs copy slices of the received lists into
               the index lists, and the exchange CTAs leave; the helper and the
               pushers then walk this rank's routes in processing order and
               fetch each row from its source rank's staging, or with
               ``pull_warps`` the main kernel fetches and this grid ends with
               the plan ("Pull dispatch")

The exchange CTAs leave instead of joining that last phase because the dependent
main kernel sizes its grid to fill the device: whatever this component holds comes
out of the main kernel's resident cluster count, so a role releases its SM as soon
as nothing depends on it.

Contrast with ``communication/nvlink_domain/token_comm.py``, which pays three
round trips (size broadcast, metadata push, activation pull) and cannot start the
bulk activation transfer until the second one lands.

Row spaces
----------
Two per-expert row spaces coexist and must not be merged:

  data rows   padded to ``token_padding_block`` (64), indexes gather_index,
              token metadata, the FC1 output pool and therefore FC2's B operand
  sf rows     padded to ``sf_padding_block``, indexes the canonical scale factor
              pool; it is pinned to 128, the row count of one block-scaled SF
              atom, because the refine block and its readiness counter are one
              padding block tall

A token tile N that is not a whole number of SF atoms and does not divide one
(192, 240) gives each token tile a slot of ``token_slot_rows(N)`` rows in both
spaces instead: an expert keeps ceil(count / N) token tiles and spreads its
routes evenly over them (``balanced_tile_rows``), the expert's ``r``-th route
lands on row ``slot_physical_row(r)``, the slot's trailing rows stay unrouted
padding, and an expert spans one whole slot per token tile. Both spaces then
share one in-expert row numbering, so a refine block is a half slot in either of
them.

The main kernel derives both cumulative bases from ``expert_sizes`` alone, so the
bases this component writes must reproduce the scheduler's formula bit for bit.

Payload representations
-----------------------
Activation scale factors exist twice: the linear per-token plane the Sentinel
dispatch pushes over NVLink, and the canonical atom-major pool the MMA reads,
whose readiness ``fc1_sf_ready[expert]`` counts so that one expert opens
without the others.

Expert-id lists
---------------
Under the pull dispatch the expert-id slots carry lists rather than planes.
Exchange CTA ``c`` of a source rank stages routes ``[c * exchange_chunk_routes,
(c + 1) * exchange_chunk_routes)`` of its top-k plane (route ``token * topk +
slot``) and counting-sorts them in SMEM by
(destination rank, local expert), so every route is sorted once, on the source.
Destination ``d`` receives under (source, chunk) its count row, one count per
local expert (``topk_count_inbox``), and its routes and their scores in list
order (``topk_index_inbox``, ``topk_score_inbox``, one chunk's room per exchange
CTA in a source's slab), padded to whole 16-byte vectors with sentinel words.

The rows and the list words announce their own arrival, as the Sentinel inbox
does: every word of the three inboxes holds the 0xFFFFFFFF init sentinel in
both ping-pong buffers from allocation on, and no word sent into them takes that
value (a score whose bits do travels as 0x7FFFFFFF). The helper and every
index-copy CTA poll the count rows with relaxed system loads and run the same
fold and ranking on them, which break ties the same way in every CTA (by expert
index, except that the CTA-wide ranking of 33 to 64 local experts alternates the
two halves of the local experts, 0, half, 1, half + 1, ...; see
``_rank_by_counting_in_cta``), so they agree on the processing order without a
hand-off. An index-copy CTA then derives where each (source, chunk, expert)
segment starts in the lists and among the expert's routes, takes
an equal slice of all list positions, polls each position's route and score
words, writes the gather index, the routing record and the score, resets the
words it read to the sentinel, and adds one to ``metadata_ready``. The helper
resets the count rows on its way out, after its refine or walk, which opened
only once ``metadata_ready`` showed every slice copied, or, when the main
kernel's pull warps fetch the rows, once it has seen ``metadata_ready``
complete itself.

Under the pull dispatch each exchange CTA also adds one to its destination's
``topk_ready[source]`` with a system-scope release once its writes have
completed. With walking CTAs the index-copy CTAs acquire those releases, each
source by one of them, before they count into ``metadata_ready``, so the walking
CTAs' wait for the index lists carries them (``pull_acquire_in_copy``, on by
default; off, every walking CTA acquires all W slots before its first fetch);
every lane of a main-kernel pull warp acquires all W before its first fetch.
That edge orders every source's staged activations before the fetches (see
"Pull dispatch"). A rank's resets precede the writes two rounds later into the
same ping-pong buffer: they complete with the rank's communication grid, and
every peer acquires the rank's ``topk_ready`` release of the next round before
the peer's grid of the round after that starts.

Sentinel top-k inbox
-------------------
Under the Sentinel dispatch the top-k planes carry their own arrival too. Both
inbox planes start with the 0xFFFFFFFF init sentinel in both ping-pong buffers,
and each exchange CTA writes this rank's whole id and score planes into its
peer's slab with volatile 16-byte vector stores. The exchange CTAs then turn
into W replicated planners: each one polls every source's id plane, counts the
routes per (source, expert) in SMEM and runs the same deterministic fold,
processing order and data-row scan as every other planner, which give it each
expert's data-row base and its own source's segment base. Planner 0 publishes
the sizes, the processing order and ``sizes_ready``, then the data-row, SF-row
and refine-block bases; above 32 local experts it alone runs the SF-row and
refine-block prefixes (``_plan_bases``). Every planner scatters the routes of
the source it exchanged with and adds one to ``metadata_ready``, whose target is
then W. No hand-off between CTAs precedes either counter, and the helper goes
straight to the refine. The end-of-round reduce resets every inbox word once the
round's planners are done with it.

Every route this rank receives is thus counted W times, once by each planner,
on purpose. The processing order and the bases need every source's counts: an
expert's load sums them, and source c's segment inside an expert starts after
the segments of the sources before c in ring order. Planners that each counted
one source would have to pass their count rows to one another through global
memory, a release and an acquire among the W CTAs, before any of them could
rank; counting every plane removes that hand-off from the plan's critical
path. The duplicate work is cheap: its loads read this rank's own inbox within
the poll that waits for the planes, its atomics are CTA-scope SMEM atomics, a
thread counts its vectors as soon as they have landed, and the planners hold
SMs that have no other work yet. The pull dispatch, which serves larger
batches, counts each route once, on its source, and sends count rows instead
(see "Expert-id lists").

Only the counts agree across planners. The value a route's SMEM atomic
returns, its offset inside its (source, expert) segment, depends on the order
in which the CTA's atomics run and differs between planners, so planner c
keeps the offsets of source c's routes alone and is the only planner that
scatters them. The offsets of one (c, expert) segment then all come from one
CTA and are 0 to the segment's count minus one, each once; which route takes
which offset varies from launch to launch and changes no output.

``topk_ready`` keeps its system-scope release, a one issued after the inbox
stores, which the planner acquires after it has published. Planes observed
through volatile loads order nothing, and this edge is what orders a rank's
reduce, which resets the inbox its peers write two rounds later, before
those writes. It also tells a late peer from a lost word: the planners' polls
are the round's first wait on the peers, so they wait for a peer that has not
launched for as long as it takes, and trap only once every source's
``topk_ready`` is set and a word is still missing.

Pull dispatch
-------------
Under the pull dispatch no source rank sends payload. The expert-id lists, the
helper and the index copy run as "Expert-id lists" describes, with the pusher
CTAs copying slices of the lists as well; the helper and the pusher CTAs then
walk this rank's routes in processing order and fetch the rows themselves. A
warp claims up to ``pull_chunk_rows`` consecutive routes of one token tile of one
expert at a time, and each route's landing slot names its source rank and token.
The warp streams the token's data row out of the source rank's staged
activations through a ring of SMEM rows into the landing slot, and the row's
lane loads the token's scale factors out of the source rank's staged scale
factors and stores each word straight into its canonical atom. Once the stores
have completed the warp adds the chunk's rows to their token tile's
``fc1_tile_ready`` (``pull_tile_ready``, on by default; a claim never crosses a
tile), which certifies the tile's data rows as well as its scale factors, and
each FC1 tile waits only for its own tile's routes. Token tiles become ready one
after another in the order the main kernel processes them, from the end of the
plan on: the rows claimed but not yet landed span only a few experts at a time.
An N64 token tile shares its scale-factor atom with the next tile, so there, or
with ``pull_tile_ready`` off, the rows count into the expert's ``fc1_sf_ready``
instead, which then counts routes rather than refine blocks and gates every FC1
tile of the expert. The first wave of claims, one per walking warp, otherwise
lands together: every warp keeps its claim's first rows in flight and the link
serves them side by side. The experts starting within that wave are claimed
``pull_lead_chunk_rows`` routes at a time instead, few enough for a claim's rows
to be in flight at once, so the first tiles count in about one fetch round.
A source token routed to several local experts is fetched once per route, into the same
landing slot with the same bytes, unless the walk claims landing slots
(``pull_walk_dedup``): then a warp fetches a route's row beyond the first few of
its claim only if it won the slot, and a source token travels about once per
target rank.

With ``fc1_token_pool`` a warp stores each fetched row at its route's data row
of an expert-major token pool instead of at its landing slot, the row space of
the FC1 output pool: a token tile's rows then sit on consecutive rows in the
order the MMA consumes them, and the main kernel loads FC1's token operand with
one tiled TMA load per AB stage (or, in a partly filled tile, one per 16 live
rows), as it loads FC2's, instead of gathering it from the landing slots four
rows per Gather4. A route's row is fetched once per route either way, so the pool costs
no extra fetch; it holds a row per route where the landing slots hold one per
source token, and the landing slots are not allocated.

Sized for any routing, that pool holds every data row a plan can reach. With
``fc1_token_pool_capacity`` C it holds C rows instead, one per route in
processing order with no padding: expert e's routes start at row R_e, the
routes of the experts before it, and its token tile t at R_e + t * tile_rows,
the tile's routes then sitting on consecutive rows as before. Both this grid
and the main kernel derive R_e from the plan's sizes in processing order
(``_stage_pull_chunk_table``, ``_plan_token_pool`` there). A token tile whose
routes end within the pool's capacity for the launch loads from the pool; any
other falls back to the landing slots, which stay allocated, and to Gather4.
That capacity is C, or with ``fc1_token_pool_fallback`` "launch" C only when
all of the launch's routes fit and zero otherwise, so a launch then runs one
path throughout.

The staged activations precede the source rank's communication grid in stream
order, so its system-scope ``topk_ready`` releases cover them. Under
``pull_acquire_in_copy`` index-copy CTA i of C acquires the ``topk_ready`` of
sources i, i + C, ... at system scope before its ``metadata_ready`` add, so the
copiers between them acquire every source before ``metadata_ready`` completes,
and every walking CTA acquires ``metadata_ready`` complete before its first
fetch (without it, every walking CTA acquires every source's ``topk_ready``
itself), so every source's staging happens before the first fetch; a global
proxy fence then orders that generic acquire before the bulk loads of the data
rows, which read through the async proxy. The data rows land through the async
proxy as well, so the warp fences again between their completion and its
release of the count; the scale-factor stores precede that release in program
order and through the warp barrier. A source rank restages only after its
counter combine's tail barrier, which every rank enters only once its own
communication grid, and with it every fetch it issued, has completed.

With ``pull_warps`` the main kernel fetches instead (``run_pull_warp``) and this
grid ends with the plan: the exchange CTAs and the pusher CTAs copy the lists,
and they and the helper leave once the index lists are out, so their SMs go to
the main kernel. Up to four warps of every main-kernel CTA pull through
registers, one row at a time: each lane loads its 16-byte vectors of the row
out of the source rank's staging and stores them into the landing slot with
generic accesses. A landing slot depends only on the
source rank and token, so the warps start before the plan: once the sources'
lists have arrived they walk them and fetch the source tokens they name, until
``metadata_ready`` is complete, and then take the claims of the processing
order from the same work counter as the walking CTAs, copying each route's
scale factors into its canonical atoms and fetching its row unless it has
landed. A state word per landing slot, tagged with the round, lets exactly one
warp claim each row and announces it once landed, so a source token travels
once whatever its routes to this rank. Every fetch is ordered behind the
fetching lane's acquire of every source's ``topk_ready``, and every load of the
index lists behind its acquire of ``metadata_ready``. FC1 reads the landing
rows and atoms through the async proxy (Gather4 and the scale-factor TMA), so
every lane fences its stores toward the async proxy before the warp barrier and
the first lane's release, of a landed row or of an expert's count; TMA-B fences
again behind its acquire and Gather-B behind its rendezvous with TMA-B. A
warp's claimed rows land with no wait in between, and the warp waits only for
the plan, for the lists and for rows others claimed, so every claim is finished
by the warp that took it, also while main-kernel clusters still wait for this
grid's SMs. The main kernel's CTAs enter the tail barrier only after their pull
warps' last claims, so the restaging argument holds unchanged.
"""

import dataclasses
from typing import ClassVar, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, Uint32
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from .....api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc
from .....communication.nvlink_domain.symmetric_buffer import SymmetricBufferDevice
from .....communication.token_protocol import TokenSrcMetadata
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.dsl_helpers import smem_exclusive_prefix, spin_wait
from .....helpers.iket_compat import iket
from .....helpers.ptx_helpers import (
    atoms_add_u32_if,
    cp_async_bulk_s2g,
    exit,
    ldg_volatile_b32_if,
    lds128_v4_b32,
    nanosleep,
    red_async_add_release_gpu_s32,
    red_async_add_release_sys_u32,
    stg_b32,
    stg_b64,
    sts128_v4_b32,
    sts_b32_if,
    tma_load_1d,
)
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import (
    balanced_tile_rows,
    ceil_div,
    expert_pool_rows,
    round_up,
    slot_physical_row,
    tile_rows_reciprocal,
    token_slot_rows,
)
from .....quant_def import QuantKind
from ......comm import ptx as sentinel_ptx


@cute.jit
def _first_matching_lane(predicate: cutlass.Boolean) -> Int32:
    """Lowest lane whose predicate holds, or -1 when none does.

    A local copy of the scheduler's primitive; it belongs beside the other warp
    helpers rather than in either caller, but moving it is a separate change.
    """
    ballot = Int32(cute.arch.vote_ballot_sync(predicate))
    first_lane = Int32(-1)
    if ballot != Int32(0):
        lowest_set_bit = ballot & (-ballot)
        first_lane = Int32(cute.arch.popc(lowest_set_bit - Int32(1)))
    return first_lane


@cute.jit
def _warp_inclusive_sum(value: Int32, lane_idx: Int32) -> Int32:
    """Inclusive prefix sum of one value per lane, across a whole warp.

    The out-of-range shuffle on the first steps is harmless: a lane below the
    step does not add what it read. Another local copy of a scheduler primitive,
    for the same reason as ``_first_matching_lane``.
    """
    inclusive = value
    for step_log in cutlass.range_constexpr(5):
        step = Int32(1 << step_log)
        previous = Int32(cute.arch.shuffle_sync(inclusive, lane_idx - step))
        if lane_idx >= step:
            inclusive = inclusive + previous
    return inclusive


@cute.jit
def _flat_exclusive_scan(
    values: cute.Tensor,
    prefix: cute.Tensor,
    warp_totals: cute.Tensor,
    count: int,
    threads: int,
    thread_idx: Int32,
    lane_idx: Int32,
    warp_idx: Int32,
) -> Int32:
    """CTA-wide exclusive prefix of ``count`` SMEM words, in one pass and behind one barrier.

    Thread ``t`` takes a contiguous run of whole 16-byte vectors, so the run's
    own prefix needs no exchange; the runs' totals are scanned by shuffles
    within each warp and through one SMEM word per warp across warps. Both
    tensors are 16-byte aligned and ``count`` is a multiple of four. Every
    thread returns the total; ``warp_totals`` is free again after the next
    barrier.
    """
    vectors_per_thread = ceil_div(count, threads * 4)
    run = vectors_per_thread * 4
    words = cute.make_rmem_tensor((run,), cutlass.Int32)
    first = thread_idx * Int32(run)
    for vector in cutlass.range_constexpr(vectors_per_thread):
        element = first + Int32(vector * 4)
        loaded = lds128_v4_b32(values.iterator + cutlass.min(element, Int32(count - 4)))
        for word in cutlass.range_constexpr(4):
            words[vector * 4 + word] = cutlass.select_(
                element < Int32(count), loaded[word], Int32(0)
            )
    run_total = Int32(0)
    for word in cutlass.range_constexpr(run):
        exclusive = run_total
        run_total = run_total + words[word]
        words[word] = exclusive
    run_inclusive = _warp_inclusive_sum(run_total, lane_idx)
    if lane_idx == Int32(31):
        warp_totals[warp_idx] = run_inclusive
    cute.arch.sync_threads()
    run_base = run_inclusive - run_total
    total = Int32(0)
    for warp in cutlass.range_constexpr(threads // 32):
        warp_total = warp_totals[warp]
        run_base = run_base + cutlass.select_(
            Int32(warp) < warp_idx, warp_total, Int32(0)
        )
        total = total + warp_total
    for vector in cutlass.range_constexpr(vectors_per_thread):
        element = first + Int32(vector * 4)
        if element < Int32(count):
            sts128_v4_b32(
                prefix.iterator + element,
                run_base + words[vector * 4],
                run_base + words[vector * 4 + 1],
                run_base + words[vector * 4 + 2],
                run_base + words[vector * 4 + 3],
            )
    return total


@dataclasses.dataclass(frozen=True)
class GenphaseTokenCommArgs:
    """Caller-owned tensors the communication kernel reads.

    These are inputs only. Everything the component produces lives in the
    ``DeviceWorkspace`` so that the dependent main kernel can find it by region
    name without threading extra pointers through its launch.

    ``activation`` and ``activation_sf`` must already carry the padded row stride
    the component advertises (``plain_activation_row_elements`` and
    ``plain_activation_sf_row_elements``). Bulk copies move whole padded rows, so
    a tightly packed caller tensor would be read past its last row.
    """

    topk_indices: cute.Tensor  # (max_tokens_per_rank, topk), Int32 or Int64
    topk_scores: Optional[cute.Tensor]  # (max_tokens_per_rank, topk), Float32
    activation: cute.Tensor  # (max_tokens_per_rank, hidden), padded row stride
    activation_sf: (
        cute.Tensor
    )  # (max_tokens_per_rank, hidden / sf_vec_size), padded row stride


class GenphaseTokenComm(KernelComponent):
    """One-round-trip token dispatch for generation-phase MegaMoE.

    Owns a standalone kernel launch. ``griddepcontrol_launch_dependents`` fires at
    entry so the dependent main kernel becomes eligible immediately and streams
    weights while the payload is still in flight; the two kernels are expected to
    be co-resident, which the host must budget SMs for.
    """

    threads_per_cta: ClassVar[int] = 256
    # CTA-cluster width of the launch. No role here needs cluster-scoped anything;
    # this only changes placement, so the grid is padded up to a whole cluster and
    # the surplus CTAs leave at entry.
    launch_cluster_width: ClassVar[int] = 2
    refine_warps_per_group: ClassVar[int] = 4
    # Landing rows one warp verifies concurrently; each holds nine 16-byte vectors
    # per lane in registers until it is tested. A warp owns up to 32 rows of a
    # block. Three per round take the largest block of a random-routing 128-token
    # batch (about nine rows per warp) in three rounds and a full block in eleven;
    # two would take five rounds there, and every round waits out a load latency.
    verify_rows_in_flight: ClassVar[int] = 3
    gather_index_tail_slack: ClassVar[int] = 256
    default_smem_limit_bytes: ClassVar[int] = 227 * 1024
    # highest bit 1 -> invalid
    load_bin_count: ClassVar[int] = 32
    # Default per-warp payload staging depth (``comm_push_depth`` overrides it).
    # Two slots keep a second row in flight while the first is being read out of
    # SMEM, which matters because the wire is the long pole.
    pusher_pipeline_depth: ClassVar[int] = 2
    # Ring slots of each walking warp under the pull dispatch, one of which
    # drains a store while the others hold fetches: the default when the rows
    # fit the SMEM budget (``comm_pull_depth`` overrides it). A fetch crosses
    # NVLink, so the ring has to cover its latency: three fetches per warp of 14
    # walking CTAs keep about 1.7 MB in flight per rank at 5 KB rows. More rows
    # in flight pull no faster once the links saturate, and they slow the main
    # kernel's combine traffic on the same links.
    pull_pipeline_depth: ClassVar[int] = 4
    # SMEM the walk needs besides its rings: the expert table and claim table.
    pull_smem_reserve_bytes: ClassVar[int] = 4096
    # Default routes of one pull claim (``comm_pull_chunk_rows`` overrides it).
    # Every walking warp holds one claim, so this sets the rows claimed but not
    # landed and with it how far ahead of the first incomplete expert the walk
    # runs; each claim also pays a work-counter atomic and a drain.
    pull_chunk_rows: ClassVar[int] = 8
    # The same when the main kernel's pull warps fetch the rows. Every pull warp
    # of the grid holds one claim and the next, and fetches a claim's routes one
    # after another, so the first claims of all of them land together: the
    # claim size multiplies how much lands before the first expert is complete.
    main_pull_chunk_rows: ClassVar[int] = 1
    # Walking CTAs: the expected expert load (routes) up to which the experts of
    # the first wave take lead claims by default (``comm_pull_lead_rows``). The
    # first wave of lead claims (3 routes for each of 112 walking warps by
    # default) then holds the first experts' first token tiles, which count in
    # one fetch round. An expert expected to hold half a wave more than that
    # stretches its smaller claims well past the first wave, where they only
    # slow it down.
    pull_lead_max_expected_load: ClassVar[int] = 384
    # List positions one thread of the pull dispatch's index copy resolves per
    # batch; the batch's loads all go out before its stores.
    index_copy_batch: ClassVar[int] = 8
    # Route vectors (four routes each) one thread of a pull exchange CTA loads
    # per step of its count and list passes, before it uses any of them.
    exchange_vector_batch: ClassVar[int] = 4
    # Pull dispatch: the CTAs that split this rank's top-k plane, at least one
    # per peer by default (``comm_exchange_ctas`` overrides it). Each sorts
    # its chunk of routes once for every destination, so more of them shorten
    # the sort; every destination then folds one count row per (source, chunk).
    default_exchange_ctas: ClassVar[int] = 8
    # The default exchange CTA count shrinks until a rank's (source, chunk,
    # expert) list segments stay within this many, unless an exchange CTA's
    # chunk would then outgrow SMEM: every index-copy CTA holds three words per
    # segment in SMEM and searches them per list position.
    max_list_segments: ClassVar[int] = 4096
    # Diagnostic timestamps (``stamps``): one Int64 %globaltimer value per
    # (site, column). The column is the comm or main CTA, or as noted per row.
    stamp_columns: ClassVar[int] = 256
    stamp_comm_entry: ClassVar[int] = 0
    stamp_comm_exit: ClassVar[int] = 1
    # metadata_ready contribution, per contributing CTA (``index_copy_cta_count``
    # of them under the pull dispatch, one per planner under the Sentinel one)
    stamp_sort_done: ClassVar[int] = 2
    # Column 0: sizes_ready; column 1: prefix_ready, or under the Sentinel dispatch
    # planner 0's row-space bases published.
    stamp_helper: ClassVar[int] = 3
    stamp_push_done: ClassVar[int] = 4  # last bulk push issued, per pusher CTA
    stamp_published: ClassVar[int] = 5  # pushes drained, per pusher CTA
    stamp_refine_open: ClassVar[int] = 6  # refine entry wait passed, per CTA
    stamp_expert_ready: ClassVar[int] = 7  # last fc1_sf_ready release, per local expert
    stamp_main_entry: ClassVar[int] = 8
    stamp_main_gather_open: ClassVar[int] = (
        9  # Gather-B passed its entry waits, per main CTA
    )
    stamp_main_first_fc1: ClassVar[int] = 10  # TMA-B passed its first fc1_sf_ready wait
    stamp_main_epilogue_done: ClassVar[int] = 11
    stamp_main_exit: ClassVar[int] = 12
    # Pull dispatch: the plan's steps, one column range per step
    # (``plan_detail_columns``).
    stamp_plan_detail: ClassVar[int] = 13
    # Main kernel, per main CTA: its scheduler left a shared expert's entry
    # head, passed sizes_ready, and claimed past its last tile.
    stamp_main_head_done: ClassVar[int] = 14
    stamp_main_plan_seen: ClassVar[int] = 15
    stamp_main_last_claim: ClassVar[int] = 16
    stamp_rows: ClassVar[int] = 17
    # Named barrier ids the refine groups synchronize on. Barrier 0 belongs to the
    # block-wide `sync_threads`, so the groups start at 1.
    refine_barrier_base: ClassVar[int] = 1
    # Row-stride padding for the two linear payload planes and the top-k inbox
    # slabs. Sixteen bytes is `cp.async.bulk`'s address and length granularity,
    # which is the only reason this padding exists, so it is a constant rather
    # than a knob: nothing here benefits from a wider stride, and at sixteen the
    # padding is a no-op for any hidden extent that already fills whole atoms.
    # It must be known at construction because `register_device_workspace` sizes
    # the symmetric payload regions from it, before any caller tensor exists --
    # which is why it cannot simply be read off `args.activation`. If a caller
    # ever needs a wider stride, relax the equality check in `launch` to "a
    # multiple of sixteen, no smaller than the logical extent" rather than
    # reintroducing a descriptor field the two sides have to agree on.
    plain_row_byte_alignment: ClassVar[int] = 16
    # Sentinel top-k inbox: the 16-byte id vectors one planner thread holds per
    # polling pass (larger planes take several passes), and the most vectors of
    # its own source's score plane it keeps in registers from entry to scatter.
    planner_id_vectors_per_thread: ClassVar[int] = 8
    planner_score_vectors_per_thread: ClassVar[int] = 2
    # A planner poll that has run out of sweeps while some source has not
    # released `topk_ready` yet sleeps this long before each further sweep.
    planner_backoff_ns: ClassVar[int] = 100_000

    # Symmetric regions: peers write into these through the peer pointer mapper.
    topk_index_inbox_region: ClassVar[str] = "genphase.token_comm.topk_index_inbox"
    # Local round state: word 0 is the round epoch.
    sentinel_control_region: ClassVar[str] = "sentinel.control"
    # Main-kernel pull warps only: per landing slot, the round tag of its claim
    # (odd) or of its landed row (even); see `run_pull_warp`.
    landing_state_region: ClassVar[str] = "genphase.token_comm.landing_state"
    sentinel_epoch_word: ClassVar[int] = 0
    # The end-of-round reduce's entry ticket, spread over this many 128-byte lines
    # (word 0 of each) so that its CTAs' adds do not queue on one address.
    sentinel_ticket_region: ClassVar[str] = "sentinel.entry_ticket"
    sentinel_ticket_lines: ClassVar[int] = 32
    topk_score_inbox_region: ClassVar[str] = "genphase.token_comm.topk_score_inbox"
    topk_ready_region: ClassVar[str] = "genphase.token_comm.topk_ready"
    # Pull dispatch only: per (source rank, exchange CTA of that source), its
    # routes to this rank per local expert, which say where each expert's
    # segment of that CTA's list starts in the index and score inbox slabs.
    topk_count_inbox_region: ClassVar[str] = "genphase.token_comm.topk_count_inbox"
    plain_activation_region: ClassVar[str] = "genphase.token_comm.plain_activation"
    plain_activation_sf_region: ClassVar[str] = (
        "genphase.token_comm.plain_activation_sf"
    )
    # Words no role reads or writes, at the head of the symmetric reset span; see
    # ``_register_symmetric_regions``.
    shared_layout_pad_region: ClassVar[str] = "genphase.token_comm.shared_layout_pad"

    # Local regions: produced and consumed on this rank only.
    expert_sizes_region: ClassVar[str] = "genphase.token_comm.expert_sizes"
    sizes_ready_region: ClassVar[str] = "genphase.token_comm.sizes_ready"
    # The processing order: per position, the local expert the main kernel
    # processes there, and that expert's routed size.
    expert_order_region: ClassVar[str] = "genphase.token_comm.expert_order"
    expert_sizes_in_order_region: ClassVar[str] = (
        "genphase.token_comm.expert_sizes_in_order"
    )
    data_expert_base_region: ClassVar[str] = "genphase.token_comm.data_expert_base"
    sf_expert_base_region: ClassVar[str] = "genphase.token_comm.sf_expert_base"
    refine_block_base_region: ClassVar[str] = "genphase.token_comm.refine_block_base"
    refine_block_total_region: ClassVar[str] = "genphase.token_comm.refine_block_total"
    prefix_ready_region: ClassVar[str] = "genphase.token_comm.prefix_ready"
    token_src_metadata_region: ClassVar[str] = "genphase.token_comm.token_src_metadata"
    gather_index_region: ClassVar[str] = "genphase.token_comm.gather_index"
    fc1_topk_scores_region: ClassVar[str] = "genphase.token_comm.fc1_topk_scores"
    metadata_ready_region: ClassVar[str] = "genphase.token_comm.metadata_ready"
    canonical_activation_sf_region: ClassVar[str] = (
        "genphase.token_comm.canonical_activation_sf"
    )
    fc1_sf_ready_region: ClassVar[str] = "genphase.token_comm.fc1_sf_ready"
    # Pull dispatch with ``pull_tile_ready`` only: the routes landed per token
    # tile of the processing order, which the main kernel's FC1 gate reads.
    fc1_tile_ready_region: ClassVar[str] = "genphase.token_comm.fc1_tile_ready"
    # Pull dispatch with ``fc1_token_pool`` only: each route's data row at its
    # data row of the pool, or with ``fc1_token_pool_capacity`` at its route's
    # row in processing order (see "Pull dispatch"), FC1's token operand.
    fc1_token_pool_region: ClassVar[str] = "genphase.token_comm.fc1_token_pool"
    refine_work_counter_region: ClassVar[str] = (
        "genphase.token_comm.refine_work_counter"
    )
    stamps_region: ClassVar[str] = "genphase.stamps"
    # Sentinel dispatch only: unused words that fix where its local regions and its
    # exchange CTAs' staged plane sit. Its small-batch latency moves by 1-2 us with
    # these offsets, through which counters share a 128-byte line and through the
    # SMEM accesses the planners' loads of the plane compile to (see
    # ``_register_local_regions`` and ``_build_smem_workspace``).
    sentinel_layout_pad_regions: ClassVar[tuple] = (
        "genphase.token_comm.sentinel_layout_pad0",
        "genphase.token_comm.sentinel_layout_pad1",
        "genphase.token_comm.sentinel_layout_pad2",
    )
    sentinel_exchange_pad_region: ClassVar[str] = (
        "genphase.token_comm.smem.sentinel_layout_pad"
    )

    # SMEM regions, grouped into mutually exclusive role lifetimes.
    # Pull dispatch: per (destination, local expert) bucket its routes and
    # where its list segment starts; per destination its list's shift from the
    # unpadded bucket offsets, start and length; the scan's warp totals; and the
    # routes and scores in list order.
    exchange_histogram_region: ClassVar[str] = (
        "genphase.token_comm.smem.exchange_histogram"
    )
    exchange_list_base_region: ClassVar[str] = (
        "genphase.token_comm.smem.exchange_list_base"
    )
    exchange_destination_region: ClassVar[str] = (
        "genphase.token_comm.smem.exchange_destination"
    )
    exchange_warp_totals_region: ClassVar[str] = (
        "genphase.token_comm.smem.exchange_warp_totals"
    )
    exchange_route_list_region: ClassVar[str] = (
        "genphase.token_comm.smem.exchange_route_list"
    )
    exchange_score_list_region: ClassVar[str] = (
        "genphase.token_comm.smem.exchange_score_list"
    )
    exchange_mbarrier_region: ClassVar[str] = (
        "genphase.token_comm.smem.exchange_mbarrier"
    )
    pusher_mbarrier_region: ClassVar[str] = "genphase.token_comm.smem.pusher_mbarrier"
    exchange_topk_stage_region: ClassVar[str] = "genphase.token_comm.smem.exchange_topk"
    exchange_score_stage_region: ClassVar[str] = (
        "genphase.token_comm.smem.exchange_score"
    )
    helper_count_matrix_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_count_matrix"
    )
    helper_totals_region: ClassVar[str] = "genphase.token_comm.smem.helper_totals"
    helper_scan_input_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_scan_input"
    )
    helper_prefix_region: ClassVar[str] = "genphase.token_comm.smem.helper_prefix"
    helper_warp_totals_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_warp_totals"
    )
    # The ranking's processing order and the expert sizes in that order, in SMEM
    # (`_place_expert_at_position`).
    helper_expert_order_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_expert_order"
    )
    helper_expert_sizes_in_order_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_expert_sizes_in_order"
    )
    helper_load_bin_region: ClassVar[str] = "genphase.token_comm.smem.helper_load_bin"
    # Pull dispatch only: every (source, chunk) count row.
    helper_chunk_counts_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_chunk_counts"
    )
    # Pull dispatch with up to ``cta_rank_experts`` local expert buckets: per
    # share of the buckets and per expert, how many experts of that share outrank
    # it (``_rank_by_counting_in_cta``).
    helper_rank_shares_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_rank_shares"
    )
    cta_rank_experts: ClassVar[int] = 64
    # Sentinel dispatch only, in the helper lifetime the planners reuse. Row 0 is
    # each expert's data-row base, row 1 the planner's own source's segment base.
    helper_plan_bases_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_plan_bases"
    )
    # Sentinel dispatch with strided token slots: row 0 is each expert's routes per
    # token tile (``balanced_tile_rows``), row 1 their ``tile_rows_reciprocal``.
    helper_tile_rows_region: ClassVar[str] = "genphase.token_comm.smem.helper_tile_rows"
    # Per route of the planner's own source: offset in its segment << 16 | local
    # expert, or -1 for a route this rank does not own.
    helper_route_records_region: ClassVar[str] = (
        "genphase.token_comm.smem.helper_route_records"
    )
    # Per warp and load bin, the warp's experts in that bin during one round of
    # ``_bin_experts_in_index_order``.
    helper_warp_bins_region: ClassVar[str] = "genphase.token_comm.smem.helper_warp_bins"
    pusher_data_stage_region: ClassVar[str] = "genphase.token_comm.smem.pusher_data"
    pusher_sf_stage_region: ClassVar[str] = "genphase.token_comm.smem.pusher_sf"
    refine_mbarrier_region: ClassVar[str] = "genphase.token_comm.smem.refine_mbarrier"
    refine_claim_region: ClassVar[str] = "genphase.token_comm.smem.refine_claim"
    refine_expert_table_region: ClassVar[str] = (
        "genphase.token_comm.smem.refine_expert_table"
    )
    # The refine block total, when the refine derives its table itself.
    refine_block_count_region: ClassVar[str] = (
        "genphase.token_comm.smem.refine_block_count"
    )
    refine_output_region: ClassVar[str] = "genphase.token_comm.smem.refine_output"
    # Pull dispatch only: per walking warp, a ring of data rows and its barriers;
    # per CTA, the claim table (rows below).
    refine_pull_mbarrier_region: ClassVar[str] = (
        "genphase.token_comm.smem.refine_pull_mbarrier"
    )
    refine_pull_ring_region: ClassVar[str] = "genphase.token_comm.smem.refine_pull_ring"
    refine_pull_chunk_base_region: ClassVar[str] = (
        "genphase.token_comm.smem.refine_pull_chunk_base"
    )
    # Rows of `refine_pull_chunk_base`, one column per local expert bucket: the
    # expert's first claim in processing order (the column past the last bucket
    # holds the claim total), its claim count, and its routes per token tile;
    # with ``pull_tile_ready`` or lead claims also its first token tile among all
    # experts' tiles in processing order and its routes per claim; with
    # ``fc1_token_pool_capacity`` also its first route in processing order (the
    # column past the last bucket holds the route total).
    pull_table_chunk_base: ClassVar[int] = 0
    pull_table_chunks: ClassVar[int] = 1
    pull_table_tile_rows: ClassVar[int] = 2
    pull_table_tile_base: ClassVar[int] = 3
    pull_table_claim_rows: ClassVar[int] = 4
    pull_table_route_base: ClassVar[int] = 5
    # Pull dispatch only, in the helper lifetime of every index-copy CTA:
    # where each (source, chunk) list starts among all received lists laid end
    # to end, plus their total past the last list. Then one entry per (source,
    # chunk, local expert) segment in that row-major order
    # (``list_segment_count``): where it starts among the lists, plus the total
    # past the last segment; and the rank of its first route among its expert's
    # routes. Then per (ring position, expert) one source's routes, and per
    # expert its data-row base. Under strided token slots, last, per expert its
    # routes per token tile (``balanced_tile_rows``) in row 0 and their
    # ``tile_rows_reciprocal`` in row 1.
    copy_list_base_region: ClassVar[str] = "genphase.token_comm.smem.copy_list_base"
    copy_segment_base_region: ClassVar[str] = (
        "genphase.token_comm.smem.copy_segment_base"
    )
    copy_row_base_region: ClassVar[str] = "genphase.token_comm.smem.copy_row_base"
    copy_ring_totals_region: ClassVar[str] = "genphase.token_comm.smem.copy_ring_totals"
    copy_data_base_region: ClassVar[str] = "genphase.token_comm.smem.copy_data_base"
    copy_tile_rows_region: ClassVar[str] = "genphase.token_comm.smem.copy_tile_rows"
    # Fields of the main kernel's pull warps' register claim table, per position
    # round (see `_build_pull_lane_table`).
    pull_lane_expert: ClassVar[int] = 0
    pull_lane_first_claim: ClassVar[int] = 1
    pull_lane_end_claim: ClassVar[int] = 2
    pull_lane_size: ClassVar[int] = 3
    pull_lane_fields: ClassVar[int] = 4

    # Fields of `refine_claim`. A claim carries everything its block implies, so
    # the block body never re-derives which expert it landed in.
    refine_claim_block: ClassVar[int] = 0
    refine_claim_data_row: ClassVar[int] = 1
    refine_claim_sf_row: ClassVar[int] = 2
    refine_claim_live_rows: ClassVar[int] = 3
    refine_claim_fields: ClassVar[int] = 4

    # Rows of `refine_expert_table`, staged once per CTA before the claim loop.
    refine_table_block_base: ClassVar[int] = 0
    refine_table_expert_size: ClassVar[int] = 1
    refine_table_data_base: ClassVar[int] = 2
    refine_table_sf_base: ClassVar[int] = 3
    refine_table_rows: ClassVar[int] = 4

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "world_size": int,
            "expert_count": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "hidden_size": int,
            "quant_kind": str,
            "topk_index_dtype": type,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "sf_padding_block": int,
            "pusher_cta_count": int,
            "refine_participant_groups": int,
            "smem_limit_bytes": OptionalRequirement(int),
            "dispatch_protocol": OptionalRequirement(str),
            "comm_push_depth": OptionalRequirement(int),
            "comm_pull_depth": OptionalRequirement(int),
            "comm_pull_chunk_rows": OptionalRequirement(int),
            "comm_pull_lead_rows": OptionalRequirement(int),
            "comm_exchange_ctas": OptionalRequirement(int),
            "pull_warps": OptionalRequirement(int),
            "pull_acquire_in_copy": OptionalRequirement(bool),
            "pull_walk_dedup": OptionalRequirement(bool),
            "pull_tile_ready": OptionalRequirement(bool),
            "fc1_token_pool": OptionalRequirement(bool),
            "fc1_token_pool_capacity": OptionalRequirement(int),
            "fc1_token_pool_fallback": OptionalRequirement(str),
            "stamps": OptionalRequirement(bool),
            "shared_expert_rows": OptionalRequirement(tuple),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.topk_index_dtype = problem_desc["topk_index_dtype"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.pusher_cta_count = impl_desc["pusher_cta_count"]
        self.refine_participant_groups = impl_desc["refine_participant_groups"]
        # "sentinel": landing rows are armed with 0xFF and the refine verifies every
        # byte before publishing an expert; "pull": this rank fetches every row
        # from its source rank (see "Pull dispatch").
        self.dispatch_protocol = impl_desc.get("dispatch_protocol", "sentinel")
        if self.dispatch_protocol not in ("sentinel", "pull"):
            raise ValueError(
                f"dispatch_protocol must be sentinel or pull, got {self.dispatch_protocol!r}."
            )
        # Under the pull dispatch, the warps of every main-kernel CTA that fetch the
        # routed rows (``run_pull_warp``); with none, the helper and the pusher
        # CTAs of this grid walk them (``_run_pull``).
        self.main_pull_warps = impl_desc.get("pull_warps", 0)
        self.pull_chunk_rows = impl_desc.get(
            "comm_pull_chunk_rows",
            type(self).main_pull_chunk_rows
            if self.main_pull_warps
            else type(self).pull_chunk_rows,
        )
        # Under the walking CTAs' pull: the index-copy CTAs acquire the sources'
        # ``topk_ready`` before they count into ``metadata_ready``, and the walking
        # CTAs take that edge from the index lists' completion instead of acquiring
        # it themselves (see "Pull dispatch").
        self.pull_acquire_in_copy = bool(
            impl_desc.get("pull_acquire_in_copy", True)
        ) and (self.dispatch_protocol == "pull" and self.main_pull_warps == 0)
        # Under the walking CTAs' pull: a walking warp claims each route's
        # landing slot and fetches a row past its claim's first few only if it
        # won the slot, so a source token travels about once per target rank
        # whatever its routes there.
        self.pull_walk_dedup = bool(impl_desc.get("pull_walk_dedup", False)) and (
            self.dispatch_protocol == "pull" and self.main_pull_warps == 0
        )
        # Under the walking CTAs' pull: a walking warp stores each route's row at
        # the route's data row of an expert-major token pool rather than at its
        # landing slot, so FC1 loads its token tiles with tiled TMA loads instead
        # of gathering them by index; the landing slots then go unused.
        self.fc1_token_pool = bool(impl_desc.get("fc1_token_pool", False))
        # The Sentinel dispatch keeps one exchange CTA per peer; see "Expert-id
        # lists" for the pull dispatch, whose default may grow once the SMEM plan
        # can be priced (``_fit_exchange_ctas``).
        self.exchange_cta_count = self.default_exchange_cta_count = self.world_size
        if self.dispatch_protocol == "pull":
            self.default_exchange_cta_count = self._default_exchange_ctas()
            self.exchange_cta_count = impl_desc.get(
                "comm_exchange_ctas", self.default_exchange_cta_count
            )
        self.pusher_pipeline_depth = impl_desc.get(
            "comm_push_depth", type(self).pusher_pipeline_depth
        )
        self.stamps = impl_desc.get("stamps", False)
        self.smem_limit_bytes = impl_desc.get(
            "smem_limit_bytes", self.default_smem_limit_bytes
        )
        ring_row_bytes = (
            self.pull_warp_count
            * self.plain_activation_row_elements
            * int(self.activation_dtype.width)
            // 8
        )
        ring_budget = self.smem_limit_bytes - self.pull_smem_reserve_bytes
        self.pull_pipeline_depth = impl_desc.get(
            "comm_pull_depth",
            min(type(self).pull_pipeline_depth, ring_budget // ring_row_bytes),
        )
        # Under the walking CTAs' pull: routes per claim of the experts that start
        # within the first wave, one such claim per walking warp
        # (``pull_lead_routes``). A claim no larger than the fetches a warp keeps
        # in flight lands in one round, so the first token tiles are ready before
        # the rest of the first wave rather than with it. By default the experts
        # of an expected load up to ``pull_lead_max_expected_load`` take such
        # claims.
        expected_load = (
            self.max_tokens_per_rank * self.world_size * self.topk / self.expert_count
        )
        lead_default = self.pull_chunk_rows
        if expected_load <= type(self).pull_lead_max_expected_load:
            lead_default = min(self.pull_pipeline_depth - 1, self.pull_chunk_rows)
        self.pull_lead_chunk_rows = impl_desc.get("comm_pull_lead_rows", lead_default)
        # The main kernel's token tile fixes how routes are laid out inside an
        # expert; see "Row spaces" above.
        self.token_tile_n = tuple(
            impl_desc.get("mma_tiler_mnk", (0, self.sf_padding_block, 0))
        )[1]
        self.token_slot_rows = token_slot_rows(self.token_tile_n)
        self.token_tile_strided = self.token_slot_rows != self.token_tile_n
        # With the token pool and a strided token tile: the pool's rows, one
        # route each in processing order, with the landing slots kept for the
        # token tiles past them (see "Pull dispatch"); 0 for a pool of every data
        # row, which needs no landing slots. A capacity that reaches the data rows
        # of the worst case takes that pool.
        requested_capacity = impl_desc.get("fc1_token_pool_capacity", 0)
        if (
            isinstance(requested_capacity, bool)
            or not isinstance(requested_capacity, int)
            or requested_capacity < 0
        ):
            raise ValueError(
                f"fc1_token_pool_capacity must be a nonnegative int, got {requested_capacity!r}."
            )
        self.fc1_token_pool_capacity = (
            requested_capacity
            if self.fc1_token_pool
            and self.token_tile_strided
            and requested_capacity < self.worst_case_data_rows
            else 0
        )
        # "tile": each token tile whose routes end within the capacity loads from
        # the pool; "launch": a launch uses the pool only if all of its routes fit.
        self.fc1_token_pool_fallback = impl_desc.get("fc1_token_pool_fallback", "tile")
        if self.fc1_token_pool_fallback not in ("tile", "launch"):
            raise ValueError(
                f"fc1_token_pool_fallback must be 'tile' or 'launch', got {self.fc1_token_pool_fallback!r}."
            )
        # Under the walking CTAs' pull: a claim's routes count into their token
        # tile's ``fc1_tile_ready`` instead of the expert's ``fc1_sf_ready``, and
        # the main kernel gates each FC1 tile on its own tile. A tile has to own
        # its scale-factor atoms for that, which every token tile but N64 does
        # (an N64 tile shares its atom with the next tile).
        self.pull_tile_ready = (
            bool(impl_desc.get("pull_tile_ready", True))
            and self.dispatch_protocol == "pull"
            and self.main_pull_warps == 0
            and self.token_tile_n >= 128
        )

        # Data and scale-factor pool rows of the main kernel's shared expert past
        # the routed worst case: its routing records and top-k scores (which the
        # main kernel's owner writes once) and its token scale factors (which
        # staging writes) live in this component's pools, where the main kernel
        # reads the routed ones. This kernel never touches them.
        self.shared_expert_data_rows, self.shared_expert_sf_rows = impl_desc.get(
            "shared_expert_rows", (0, 0)
        )

        self._validate_configuration()
        if self.dispatch_protocol == "pull":
            self.default_exchange_cta_count = self._fit_exchange_ctas(
                self.default_exchange_cta_count
            )
            self.exchange_cta_count = impl_desc.get(
                "comm_exchange_ctas", self.default_exchange_cta_count
            )
        self._smem_workspace = self._build_smem_workspace()
        # Byte stride between the ping and pong buffers of the symmetric workspace.
        # The owning kernel sets it once its workspace is finalized.
        self.pingpong_buffer_bytes: Optional[int] = None

    def _validate_configuration(self) -> None:
        positive_fields = (
            "world_size",
            "expert_count",
            "topk",
            "max_tokens_per_rank",
            "hidden_size",
            "token_padding_block",
            "sf_padding_block",
            "refine_participant_groups",
            "smem_limit_bytes",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(
                    f"{field_name} must be positive, got {getattr(self, field_name)}."
                )

        # The pusher CTAs walk the routes under the pull dispatch; when the main
        # kernel's pull warps fetch them instead, the pusher CTAs only copy index
        # lists, and the grid may have none.
        if self.dispatch_protocol != "pull" and self.main_pull_warps:
            raise ValueError("pull_warps applies only to the pull dispatch.")
        if self.main_pull_warps < 0:
            raise ValueError(
                f"pull_warps must be nonnegative, got {self.main_pull_warps}."
            )
        if self.fc1_token_pool and (not self.pull_walks or self.pull_walk_dedup):
            raise ValueError(
                "fc1_token_pool needs the pull dispatch with walking CTAs (no pull_warps) and no pull_walk_dedup."
            )
        if self.dispatch_protocol == "pull" and not self.pull_walks:
            if self.pusher_cta_count < 0:
                raise ValueError(
                    f"pusher_cta_count must be nonnegative, got {self.pusher_cta_count}."
                )
        elif self.pusher_cta_count <= 0:
            raise ValueError(
                f"pusher_cta_count must be positive, got {self.pusher_cta_count}."
            )
        if self.expert_count % self.world_size != 0:
            raise ValueError(
                f"expert_count must be divisible by world_size, got {self.expert_count} and {self.world_size}."
            )
        if self.expert_count > 16384:
            raise NotImplementedError(
                "Genphase token comm supports at most 16384 global experts."
            )
        if self.topk > self.expert_count:
            raise ValueError(
                f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}."
            )
        if self.max_tokens_per_rank > 8192:
            raise ValueError(
                f"Genphase token comm requires max_tokens_per_rank <= 8192, got {self.max_tokens_per_rank}."
            )
        if self.topk_index_dtype not in (cutlass.Int32, cutlass.Int64):
            raise ValueError(
                f"topk_index_dtype must be Int32 or Int64, got {self.topk_index_dtype}."
            )

        if self.sf_padding_block != 128:
            raise ValueError(
                f"sf_padding_block must be the 128-row SF atom, got {self.sf_padding_block}."
            )
        if self.sf_padding_block % self.token_padding_block != 0:
            raise ValueError(
                "sf_padding_block must be divisible by token_padding_block."
            )
        if self.token_tile_strided and (
            self.token_slot_rows % self.sf_padding_block != 0
            or self.token_tile_n % 8 != 0
        ):
            raise ValueError(
                f"token tile {self.token_tile_n} needs a slot of whole SF atoms and eight-row gather runs."
            )

        # One SF value covers sf_vec_size activation elements and one atom spans
        # four of them along K, so the hidden extent has to fill whole atoms.
        elements_per_sf_atom = self.quant_kind.sf_vec_size * 4
        if self.hidden_size % elements_per_sf_atom != 0:
            raise ValueError(
                f"{self.quant_kind} requires hidden_size divisible by {elements_per_sf_atom}."
            )

        # The caller's top-k plane is exactly `routes_per_rank` elements with no
        # slack, so the bulk copy that stages it has to land on that boundary
        # rather than rounding up into memory we do not own.
        topk_plane_bytes = self.routes_per_rank * int(self.topk_index_dtype.width) // 8
        if topk_plane_bytes % 16 != 0:
            raise ValueError(
                f"max_tokens_per_rank * topk * sizeof({self.topk_index_dtype}) must be a multiple of 16 bytes, "
                f"got {topk_plane_bytes}."
            )
        if self.apply_topk_at_fc1 and (self.routes_per_rank * 4) % 16 != 0:
            raise ValueError(
                f"max_tokens_per_rank * topk * 4 must be a multiple of 16 bytes for the score plane, "
                f"got {self.routes_per_rank * 4}."
            )
        if (
            self.dispatch_protocol == "pull"
            and (self.exchange_chunk_routes << self.exchange_record_bucket_bits)
            >= 1 << 31
        ):
            raise ValueError(
                f"a pull exchange record of {self.exchange_chunk_routes} routes and "
                f"{self.exchange_bucket_count} buckets does not fit 31 bits; raise comm_exchange_ctas."
            )
        if self.dispatch_protocol == "sentinel":
            # The inbox moves whole 16-byte vectors of 32-bit words, and a route
            # record keeps its offset inside a segment in 15 bits.
            if self.routes_per_rank % 4 != 0:
                raise ValueError(
                    f"the Sentinel top-k inbox needs max_tokens_per_rank * topk to be a multiple of 4, "
                    f"got {self.routes_per_rank}."
                )
            if self.routes_per_rank > 1 << 15:
                raise ValueError(
                    f"the Sentinel top-k planners support at most {1 << 15} routes per rank, got {self.routes_per_rank}."
                )
        # Each refine group claims and finishes blocks on its own, so it must own
        # whole warps: the group's named barrier id is warp-uniform only if no warp
        # straddles two groups, and `bar.sync` is implicitly aligned.
        refine_threads = (
            self.refine_participant_groups * self.refine_warps_per_group * 32
        )
        if refine_threads != self.threads_per_cta:
            raise ValueError(
                f"{self.refine_participant_groups} refine groups of "
                f"{self.refine_warps_per_group} warps cover {refine_threads} threads, "
                f"which must equal the {self.threads_per_cta}-thread block."
            )
        # One thread stages one row of a 128-row block.
        if self.refine_warps_per_group * 32 != 128:
            raise ValueError(
                "A refine group must be exactly four warps so one thread owns one row."
            )
        if self.refine_barrier_base + self.refine_participant_groups > 16:
            raise ValueError(
                f"{self.refine_participant_groups} refine groups starting at barrier "
                f"{self.refine_barrier_base} exceed the 16 CTA barriers."
            )
        self._validate_pipelines()

    def _validate_pipelines(self) -> None:
        if not 1 <= self.pusher_pipeline_depth <= 4:
            raise ValueError(
                f"comm_push_depth must be in [1, 4], got {self.pusher_pipeline_depth}."
            )
        # One ring slot drains a store while the others hold fetches.
        if self.pull_walks and not 2 <= self.pull_pipeline_depth <= 8:
            raise ValueError(
                f"comm_pull_depth must be in [2, 8], got {self.pull_pipeline_depth}."
            )
        if self.dispatch_protocol == "pull":
            # A claim's routes sit one per lane, on consecutive pool rows of one
            # token tile.
            if self.pull_chunk_rows not in (1, 2, 4, 8, 16, 32):
                raise ValueError(
                    f"comm_pull_chunk_rows must be a power of two up to 32, got {self.pull_chunk_rows}."
                )
            if not 1 <= self.pull_lead_chunk_rows <= 32:
                raise ValueError(
                    f"comm_pull_lead_rows must be in [1, 32], got {self.pull_lead_chunk_rows}."
                )

    # ------------------------------------------------------------------
    # Static geometry
    # ------------------------------------------------------------------

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    @property
    def local_expert_bucket_count(self) -> int:
        """Histogram bucket count, padded because the CTA-wide scan reads four at a time."""
        return round_up(self.experts_per_rank, 4)

    @property
    def invalid_expert_marker(self) -> int:
        """Wire value for a route no rank owns.

        ``expert_count`` is outside every rank's local range, so a "is this mine"
        test rejects it without a second rule.
        """
        return self.expert_count

    def _default_exchange_ctas(self) -> int:
        exchange_ctas = max(self.world_size, self.default_exchange_ctas)
        while (
            exchange_ctas > 1
            and self.world_size * exchange_ctas * self.local_expert_bucket_count
            > self.max_list_segments
        ):
            exchange_ctas -= 1
        return exchange_ctas

    def _fit_exchange_ctas(self, exchange_ctas: int) -> int:
        """``exchange_ctas``, or more where their chunks of the top-k plane outgrow SMEM.

        An exchange CTA holds its chunk's ids and scores and the lists it sorts
        them into, about 16 bytes per route, so many routes per rank among many
        local experts (Kimi-K3's top-16 of 112 per rank at 4096 tokens per rank)
        outgrow SMEM at the segment bound's count. The plane then splits into one
        more chunk at a time, for as long as each shrinks the plan.
        """
        plan_bytes = self._smem_plan_bytes(exchange_ctas)
        while plan_bytes > self.smem_limit_bytes:
            split_bytes = self._smem_plan_bytes(exchange_ctas + 1)
            if split_bytes >= plan_bytes:
                break
            exchange_ctas, plan_bytes = exchange_ctas + 1, split_bytes
        return exchange_ctas

    def _smem_plan_bytes(self, exchange_ctas: int) -> int:
        """Bytes of this kernel's SMEM plan with ``exchange_ctas`` pull exchange CTAs."""
        resolved = self.exchange_cta_count
        self.exchange_cta_count = exchange_ctas
        try:
            return self._plan_smem_workspace().estimate_total_bytes()
        finally:
            self.exchange_cta_count = resolved

    @property
    def list_segment_count(self) -> int:
        """(source rank, source chunk, local expert) segments of the pull dispatch's lists, padded buckets included."""
        return (
            self.world_size * self.exchange_cta_count * self.local_expert_bucket_count
        )

    @property
    def exchange_chunk_routes(self) -> int:
        """Routes of the top-k plane one pull exchange CTA sorts, whole 16-byte vectors of ids."""
        return int(round_up(ceil_div(self.routes_per_rank, self.exchange_cta_count), 4))

    @property
    def exchange_bucket_count(self) -> int:
        """(destination rank, local expert) buckets of a pull exchange CTA's sort."""
        return self.world_size * self.local_expert_bucket_count

    @property
    def exchange_record_bucket_bits(self) -> int:
        """Low bits of a pull exchange record that name its bucket; the bits above count within it."""
        return max(1, (self.exchange_bucket_count - 1).bit_length())

    @property
    def exchange_list_capacity(self) -> int:
        """Entries of a pull exchange CTA's lists: its routes, each destination's padded to 16 bytes."""
        return int(round_up(self.exchange_chunk_routes + 3 * self.world_size, 4))

    @property
    def fence_inbox_slab_elements(self) -> int:
        """Per-source index and score inbox slab under the pull dispatch: one chunk's room per exchange CTA."""
        return self.exchange_cta_count * self.exchange_chunk_routes

    @property
    def ranks_experts_in_warp(self) -> bool:
        """Whether the load ranking fits one warp at one lane per expert."""
        return self.local_expert_bucket_count <= 32

    @property
    def ranks_experts_in_cta(self) -> bool:
        """Whether the pull dispatch's planning CTAs rank the experts with every thread (``_rank_by_counting_in_cta``)."""
        return (
            self.dispatch_protocol == "pull"
            and self.local_expert_bucket_count <= self.cta_rank_experts
        )

    @property
    def routes_per_rank(self) -> int:
        return self.max_tokens_per_rank * self.topk

    @property
    def topk_slab_elements(self) -> int:
        """Per-source-rank inbox slab, padded so every slab base is bulk-copy aligned."""
        return int(round_up(self.routes_per_rank, self.plain_row_byte_alignment // 4))

    @property
    def planner_id_slots(self) -> int:
        """Inbox id vectors one planner thread holds per polling pass."""
        id_vectors = self.world_size * (self.routes_per_rank // 4)
        return min(
            ceil_div(id_vectors, self.threads_per_cta),
            self.planner_id_vectors_per_thread,
        )

    @property
    def planner_score_slots(self) -> int:
        """Own-source score vectors a planner thread polls with its ids and holds until the scatter."""
        if not self.apply_topk_at_fc1:
            return 0
        return min(
            ceil_div(self.routes_per_rank // 4, self.threads_per_cta),
            self.planner_score_vectors_per_thread,
        )

    @property
    def activation_dtype(self) -> type:
        return self.quant_kind.activation_dtype

    @property
    def activation_sf_dtype(self) -> type:
        return self.quant_kind.sf_dtype

    @property
    def hidden_sf(self) -> int:
        return self.hidden_size // self.quant_kind.sf_vec_size

    def _padded_row_elements(self, elements: int, dtype: type) -> int:
        elements_per_alignment = self.plain_row_byte_alignment * 8 // int(dtype.width)
        return int(round_up(elements, elements_per_alignment))

    @property
    def plain_activation_row_elements(self) -> int:
        return self._padded_row_elements(self.hidden_size, self.activation_dtype)

    @property
    def plain_activation_sf_row_elements(self) -> int:
        """``Pg``: the linear SF plane's row stride."""
        return self._padded_row_elements(self.hidden_sf, self.activation_sf_dtype)

    @property
    def pusher_warp_count(self) -> int:
        return self.threads_per_cta // 32

    @property
    def pull_warp_count(self) -> int:
        return self.threads_per_cta // 32

    @property
    def pull_table_row_count(self) -> int:
        """Rows of the walk's claim table (``pull_table_*``)."""
        if self.fc1_token_pool_capacity:
            return self.pull_table_route_base + 1
        if self.pull_lead_chunk_rows != self.pull_chunk_rows:
            return self.pull_table_claim_rows + 1
        return (
            self.pull_table_tile_base + 1
            if self.pull_tile_ready
            else self.pull_table_tile_base
        )

    @property
    def pull_lead_routes(self) -> int:
        """Routes of the processing order the lead claims cover: one lead claim per walking warp."""
        return (
            self.pull_lead_chunk_rows
            * self.pull_warp_count
            * (self.pusher_cta_count + 1)
        )

    @property
    def pull_walks(self) -> bool:
        """Whether this grid fetches the pull dispatch's rows (``_run_pull``) rather than the main kernel."""
        return self.dispatch_protocol == "pull" and self.main_pull_warps == 0

    @property
    def lands_rows(self) -> bool:
        """Whether routed rows may land in the landing slots: all but the pull dispatch into a pool of every data row."""
        return not self.fc1_token_pool or self.fc1_token_pool_capacity > 0

    @property
    def fc1_token_pool_rows(self) -> int:
        """Rows of the token pool: its capacity, or every data row of the worst case."""
        return self.fc1_token_pool_capacity or self.worst_case_data_rows

    @property
    def pusher_global_warp_count(self) -> int:
        return self.pusher_cta_count * self.pusher_warp_count

    @property
    def pusher_tokens_per_warp(self) -> int:
        """Tokens one warp is responsible for pushing, across both payload passes."""
        return int(ceil_div(self.max_tokens_per_rank, self.pusher_global_warp_count))

    @property
    def destination_mask_words(self) -> int:
        """Words one token's destination bitmask occupies, at one bit per peer."""
        return int(ceil_div(self.world_size, 32))

    @property
    def topk_lane_rounds(self) -> int:
        """Passes a warp needs for one token's top-k slots, at one slot per lane."""
        return int(ceil_div(self.topk, 32))

    @property
    def refine_atoms_per_stage(self) -> int:
        """Atoms flushed per bulk store; four keeps the 16-byte value mode whole."""
        return 4

    @property
    def refine_output_stage_elements(self) -> int:
        return self.refine_atoms_per_stage * 512

    @property
    def refine_output_slots(self) -> int:
        """Run slots of a refine group's output stage.

        The refine writes every run of a block before it stores any, so it holds
        one slot per run.
        """
        return self.refine_run_count

    @property
    def refine_run_columns(self) -> int:
        """Scale-factor columns one run covers: four per atom, four atoms."""
        return 4 * self.refine_atoms_per_stage

    @property
    def refine_total_atoms(self) -> int:
        """Atoms along one row. One atom spans four columns, and the hidden extent
        is required to fill whole atoms, so this is always exact."""
        return self.hidden_sf // 4

    @property
    def refine_run_count(self) -> int:
        """Runs per row, rounded up.

        The atom count need not be a multiple of the four atoms a run holds, so
        the last run is partial, and only its live atoms are stored.
        """
        return int(ceil_div(self.refine_total_atoms, self.refine_atoms_per_stage))

    @property
    def refine_static_blocks(self) -> int:
        """Leading refine blocks that are taken without a claim.

        Under the Sentinel dispatch the refine runs in the helper and the pushers
        only, so every refine group has a fixed index among all of them, and its
        first block is the block of that index: it needs no atomic round trip.
        The helper, which pushes nothing and reaches the refine first, gives its
        first group index 0, the first expert in processing order, and its other
        groups the last indices, which small batches never reach, so no other
        group's verification shares block 0's SM. The pushers' groups take the
        indices between, in CTA order. The work counter hands out the blocks past
        these.
        """
        return (1 + self.pusher_cta_count) * self.refine_participant_groups

    @property
    def refine_plans_table(self) -> bool:
        """Whether the refine derives its expert table from the processing order.

        Under the Sentinel dispatch planner 0 publishes the processing order and
        the expert sizes in that order with ``sizes_ready``, and the row-space
        bases only ahead of its ``metadata_ready`` contribution. With at most 32
        buckets the refine runs planner 0's one-warp scan on the published loads
        itself, so its table and its first block are settled while the planners
        still scatter, and ``metadata_ready`` gates the block's rows only.
        """
        return self.dispatch_protocol == "sentinel" and self.ranks_experts_in_warp

    def worst_case_padded_rows(self, block: int) -> int:
        """Tight bound on ``sum_e round_up(size_e, block)`` for this rank's experts.

        Two bounds apply and the smaller wins: every expert wastes at most one
        block beyond its own routes, and no expert can hold more routes than the
        whole source token population.
        """
        source_token_capacity = self.world_size * self.max_tokens_per_rank
        routes_per_source_token = min(self.topk, self.experts_per_rank)
        route_capacity = source_token_capacity * routes_per_source_token
        active_expert_capacity = min(self.experts_per_rank, route_capacity)
        route_budget_blocks = (
            active_expert_capacity + (route_capacity - active_expert_capacity) // block
        )
        expert_bound_blocks = active_expert_capacity * int(
            ceil_div(source_token_capacity, block)
        )
        return min(route_budget_blocks, expert_bound_blocks) * block

    def _worst_case_pool_rows(self, padding_block: int) -> int:
        if self.token_tile_strided:
            # Every token tile, partial or not, spans one whole slot.
            tile_n = self.token_tile_n
            return self.worst_case_padded_rows(tile_n) // tile_n * self.token_slot_rows
        return self.worst_case_padded_rows(padding_block)

    @property
    def worst_case_token_tiles(self) -> int:
        """Bound on the token tiles of all local experts, ``sum_e ceil(size_e / N)``."""
        return self.worst_case_padded_rows(self.token_tile_n) // self.token_tile_n

    @property
    def worst_case_data_rows(self) -> int:
        return self._worst_case_pool_rows(self.token_padding_block)

    @property
    def worst_case_sf_rows(self) -> int:
        return self._worst_case_pool_rows(self.sf_padding_block)

    @property
    def worst_case_refine_blocks(self) -> int:
        return self.worst_case_sf_rows // self.sf_padding_block

    # ------------------------------------------------------------------
    # Grid layout
    # ------------------------------------------------------------------

    @property
    def helper_cta_index(self) -> int:
        return self.exchange_cta_count

    @property
    def pusher_cta_begin(self) -> int:
        return self.exchange_cta_count + 1

    @property
    def grid_cta_count(self) -> int:
        return self.exchange_cta_count + 1 + self.pusher_cta_count

    @property
    def launch_cluster_count(self) -> int:
        return ceil_div(self.grid_cta_count, self.launch_cluster_width)

    @property
    def topk_ready_target(self) -> int:
        """One producer per source rank, so the wait target is exact."""
        return 1

    @property
    def sentinel_inbox_regions(self) -> tuple:
        """Symmetric regions whose words announce their own arrival, holding the 0xFF init sentinel from allocation.

        Both dispatches: the index and score inboxes; the pull dispatch also the
        count inbox. No word sent into them is 0xFFFFFFFF.
        """
        regions = [self.topk_index_inbox_region]
        if self.apply_topk_at_fc1:
            regions.append(self.topk_score_inbox_region)
        if self.dispatch_protocol == "pull":
            regions.append(self.topk_count_inbox_region)
        return tuple(regions)

    @property
    def index_copy_cta_count(self) -> int:
        """CTAs that copy the pull dispatch's route lists into the index lists, one slice each.

        The exchange CTAs and the pusher CTAs, which push nothing under the pull
        dispatch and would otherwise wait for the lists, or, when the main
        kernel's pull warps fetch the rows, copy and leave. The helper, whose
        ranking and prefixes the other roles wait for, copies nothing.
        """
        return self.exchange_cta_count + self.pusher_cta_count

    @property
    def metadata_ready_target(self) -> int:
        if self.dispatch_protocol == "sentinel":
            return self.world_size
        return self.index_copy_cta_count

    @property
    def plan_detail_columns(self) -> dict:
        """Columns of ``stamp_plan_detail`` per plan step, as (first column, count).

        Per exchange CTA: its chunk landed, counted and listed, and its lists
        published to every destination. Then when the helper had every count row
        folded and ranked, and per index-copy CTA when it had, when its tables
        were ready and when its last batch of list words had arrived.
        """
        exchange = self.exchange_cta_count
        copiers = self.index_copy_cta_count
        steps = (
            ("x_loaded", exchange),
            ("x_counted", exchange),
            ("x_listed", exchange),
            ("x_released", exchange),
            ("h_planned", 1),
            ("c_planned", copiers),
            ("c_staged", copiers),
            ("c_loaded", copiers),
        )
        columns = {}
        first = 0
        for name, count in steps:
            columns[name] = (first, count)
            first += count
        return columns

    def variant_suffix(self) -> str:
        """Name fragment for every non-default dispatch knob; empty for the default build."""
        parts = []
        if self.pusher_pipeline_depth != type(self).pusher_pipeline_depth:
            parts.append(f"depth{self.pusher_pipeline_depth}")
        if self.pull_walks:
            parts.append(f"pull{self.pull_chunk_rows}x{self.pull_pipeline_depth}")
            if self.pull_acquire_in_copy:
                parts.append("copyacq")
            if self.pull_walk_dedup:
                parts.append("dedup")
            if self.pull_tile_ready:
                parts.append("tileready")
            if self.pull_lead_chunk_rows != self.pull_chunk_rows:
                parts.append(f"lead{self.pull_lead_chunk_rows}")
            if self.fc1_token_pool:
                parts.append("tokenpool")
                if self.fc1_token_pool_capacity:
                    parts.append(
                        f"cap{self.fc1_token_pool_capacity}{self.fc1_token_pool_fallback}"
                    )
        elif self.dispatch_protocol == "pull":
            parts.append(f"pull{self.pull_chunk_rows}w{self.main_pull_warps}")
        if self.exchange_cta_count != self.default_exchange_cta_count:
            parts.append(f"exchange{self.exchange_cta_count}")
        if self.stamps:
            parts.append("stamps")
        return "".join(f"_{part}" for part in parts)

    # ------------------------------------------------------------------
    # GMEM workspace
    # ------------------------------------------------------------------

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        """Declare every region this component reads or writes.

        Ready counters are ``tail_reset`` so exactly the leading span can be
        zeroed between launches. This component never resets from inside: it exits
        while the dependent main kernel is still consuming ``metadata_ready``,
        so clearing counters here would strand any CTA that had not yet observed
        them. The reset therefore belongs to whoever outlives it -- normally the
        main kernel's tail, which zeroes the span between two NVLink barriers. A
        launch without that main kernel (a standalone test) has to zero on the
        host instead.
        """
        self._register_symmetric_regions(workspace)
        self._register_local_regions(workspace)

    def _register_symmetric_regions(self, workspace: DeviceWorkspace) -> None:
        # The Sentinel inbox holds each source's whole top-k plane; the pull one
        # holds, per source, one chunk's room per exchange CTA of that source,
        # filled with its routes to this rank in list order.
        inboxes = [(self.topk_index_inbox_region, cutlass.Int32)]
        if self.apply_topk_at_fc1:
            inboxes.append((self.topk_score_inbox_region, cutlass.Float32))
        for region_name, dtype in inboxes:
            if self.dispatch_protocol == "sentinel":
                workspace.register(
                    region_name,
                    dtype,
                    (self.world_size, self.max_tokens_per_rank, self.topk),
                    buffer_space="shared",
                    stride=(self.topk_slab_elements, self.topk, 1),
                    byte_alignment=16,
                )
            else:
                workspace.register(
                    region_name,
                    dtype,
                    (self.world_size, self.fence_inbox_slab_elements),
                    buffer_space="shared",
                    stride=(self.fence_inbox_slab_elements, 1),
                    byte_alignment=16,
                )
        # Not a tail counter: each slot is cleared by the exchange of the round
        # after the one that published it, just before that exchange's own
        # release to the slot's writer (``_clear_previous_topk_ready``).
        workspace.register(
            self.topk_ready_region,
            cutlass.Int32,
            (self.world_size,),
            buffer_space="shared",
        )
        if self.dispatch_protocol == "pull":
            # Rewritten whole by every source each round, so it needs no reset.
            buckets = self.local_expert_bucket_count
            workspace.register(
                self.topk_count_inbox_region,
                cutlass.Int32,
                (self.world_size, self.exchange_cta_count, buckets),
                buffer_space="shared",
                stride=(self.exchange_cta_count * buckets, buckets, 1),
                byte_alignment=16,
            )
        # The shape is the logical column extent and the row padding is a stride.
        # Bulk copies move whole padded rows, so the trailing row's padding has to
        # be inside the region even though no index reaches it; the workspace sizes
        # a region by what its strides occupy for exactly this reason. The pull
        # dispatch into a token pool of every data row lands nothing here and
        # registers neither.
        landing_planes = (
            (
                self.plain_activation_region,
                self.activation_dtype,
                self.hidden_size,
                self.plain_activation_row_elements,
            ),
            (
                self.plain_activation_sf_region,
                self.activation_sf_dtype,
                self.hidden_sf,
                self.plain_activation_sf_row_elements,
            ),
        )
        for region_name, dtype, columns, row_elements in (
            landing_planes if self.lands_rows else ()
        ):
            workspace.register(
                region_name,
                dtype,
                (self.world_size, self.max_tokens_per_rank, columns),
                buffer_space="shared",
                stride=(self.max_tokens_per_rank * row_elements, row_elements, 1),
                byte_alignment=128,
            )
        # Thirty-two bytes at the head of the symmetric reset span that no role
        # reads or writes. They hold every later symmetric region (the inboxes,
        # the landing planes, the combine slots) at the offset the engine was
        # measured with; the small-batch latency moves with such offsets (see
        # ``sentinel_layout_pad_regions``).
        workspace.register(
            self.shared_layout_pad_region,
            cutlass.Int32,
            (8,),
            buffer_space="shared",
            reset="tail_reset",
        )

    def _register_local_regions(self, workspace: DeviceWorkspace) -> None:
        # Both start at zero: every rank begins in the ping buffer, and the reduce's
        # round advance waits for the ticket to count exactly its grid.
        workspace.register(
            self.sentinel_control_region,
            cutlass.Int32,
            (4,),
            buffer_space="local",
            byte_alignment=128,
            reset="zero_on_first_allocate",
        )
        workspace.register(
            self.sentinel_ticket_region,
            cutlass.Int32,
            (self.sentinel_ticket_lines, 32),
            buffer_space="local",
            stride=(32, 1),
            byte_alignment=128,
            reset="zero_on_first_allocate",
        )
        if self.dispatch_protocol == "pull" and (
            not self.pull_walks or self.pull_walk_dedup
        ):
            # Tags carry the round, which only grows, so no round clears them.
            workspace.register(
                self.landing_state_region,
                cutlass.Int32,
                (self.world_size * self.max_tokens_per_rank,),
                buffer_space="local",
                byte_alignment=16,
                reset="zero_on_first_allocate",
            )
        # Under the Sentinel dispatch two unused (rank, expert) tables pad the data
        # regions below, and a spare word leads the readiness counters.
        sentinel_layout = self.dispatch_protocol == "sentinel"
        if sentinel_layout:
            workspace.register(
                self.sentinel_layout_pad_regions[0],
                cutlass.Int32,
                (self.world_size, self.local_expert_bucket_count),
                buffer_space="local",
                stride=(self.local_expert_bucket_count, 1),
                byte_alignment=16,
            )
        workspace.register(
            self.expert_sizes_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            buffer_space="local",
            reset="tail_reset",
        )
        if sentinel_layout:
            workspace.register(
                self.sentinel_layout_pad_regions[1],
                cutlass.Int32,
                (self.world_size, self.local_expert_bucket_count),
                buffer_space="local",
                stride=(self.local_expert_bucket_count, 1),
                byte_alignment=16,
            )
        workspace.register(
            self.expert_order_region,
            cutlass.Int32,
            (self.local_expert_bucket_count,),
            buffer_space="local",
            byte_alignment=16,
        )
        workspace.register(
            self.expert_sizes_in_order_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            buffer_space="local",
            byte_alignment=16,
        )
        # Sized to the padded bucket count rather than the live expert count: the
        # scan that fills them runs over padded buckets, and letting the tail
        # entries exist removes every boundary special case downstream.
        for region_name in (
            self.data_expert_base_region,
            self.sf_expert_base_region,
            self.refine_block_base_region,
        ):
            workspace.register(
                region_name,
                cutlass.Int32,
                (self.local_expert_bucket_count,),
                buffer_space="local",
                byte_alignment=16,
            )
        workspace.register(
            self.token_src_metadata_region,
            cutlass.Int64,
            (self.worst_case_data_rows + self.shared_expert_data_rows,),
            buffer_space="local",
            byte_alignment=16,
        )
        # The FC1 gather window reaches up to a whole token tile past a tile base,
        # so the index array needs readable slack beyond the last expert -- unlike a
        # descriptor coordinate, an index is read before any bounds check. The
        # values there are irrelevant: a wild index is zero-filled by the TMA
        # descriptor bounds, and a valid-but-wrong one only lands on an N column the
        # dynamic instruction extent already excludes.
        #
        # 32-byte alignment, not 16: the gather warps read eight identifiers per
        # 256-bit access, and every run they address sits on an eight-row boundary,
        # so the base has to carry that alignment for the access to be legal.
        workspace.register(
            self.gather_index_region,
            cutlass.Int32,
            (self.worst_case_data_rows + self.gather_index_tail_slack,),
            buffer_space="local",
            byte_alignment=32,
        )
        if self.apply_topk_at_fc1:
            workspace.register(
                self.fc1_topk_scores_region,
                cutlass.Float32,
                (self.worst_case_data_rows + self.shared_expert_data_rows,),
                buffer_space="local",
                byte_alignment=16,
            )
        workspace.register(
            self.canonical_activation_sf_region,
            self.activation_sf_dtype,
            ((self.worst_case_sf_rows + self.shared_expert_sf_rows) * self.hidden_sf,),
            buffer_space="local",
            byte_alignment=128,
        )
        if self.fc1_token_pool:
            # Whole padded rows, as the landing planes hold them: the walk's bulk
            # stores move them, and FC1's tiled loads read the leading hidden columns.
            row_elements = self.plain_activation_row_elements
            workspace.register(
                self.fc1_token_pool_region,
                self.activation_dtype,
                (self.fc1_token_pool_rows, row_elements),
                buffer_space="local",
                stride=(row_elements, 1),
                byte_alignment=128,
            )
        # The readiness counters and the first experts' `fc1_sf_ready` share
        # 128-byte lines, which they are polled and released through. Under the
        # pull dispatch `metadata_ready` starts a line that the refine counters
        # and the first experts' `fc1_sf_ready` share; under the Sentinel dispatch
        # a spare word leads the counters, and all of them, 16 bytes apart, share
        # one line with the first four experts' `fc1_sf_ready`. Placements with
        # these counters spread differently measured 1-2 us slower end to end at
        # small batches, in a direction that depends on the dispatch.
        counters = (
            self.sizes_ready_region,
            self.prefix_ready_region,
            self.metadata_ready_region,
            self.refine_block_total_region,
            self.refine_work_counter_region,
        )
        if sentinel_layout:
            counters = (self.sentinel_layout_pad_regions[2],) + counters
        for region_name in counters:
            workspace.register(
                region_name,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                byte_alignment=128
                if region_name == self.metadata_ready_region and not sentinel_layout
                else 16,
                reset="tail_reset",
            )
        # One slot per local expert
        workspace.register(
            self.fc1_sf_ready_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            buffer_space="local",
            reset="tail_reset",
        )
        if self.pull_tile_ready:
            # One slot per token tile of the processing order, the space the main
            # kernel's `cumulative_token_block_count + tile_n_idx` indexes.
            workspace.register(
                self.fc1_tile_ready_region,
                cutlass.Int32,
                (self.worst_case_token_tiles,),
                buffer_space="local",
                byte_alignment=128,
                reset="tail_reset",
            )
        if self.stamps:
            # Rewritten every launch; the readout takes whichever launch ran last.
            workspace.register(
                self.stamps_region,
                cutlass.Int64,
                (self.stamp_rows, self.stamp_columns),
                buffer_space="local",
                stride=(self.stamp_columns, 1),
                byte_alignment=16,
            )

    # ------------------------------------------------------------------
    # SMEM plan
    # ------------------------------------------------------------------

    def _build_smem_workspace(self) -> SmemWorkspace:
        """The finalized SMEM plan (``_plan_smem_workspace``) within ``smem_limit_bytes``."""
        workspace = self._plan_smem_workspace()
        workspace.finalize(max_bytes=self.smem_limit_bytes)
        return workspace

    def _plan_smem_workspace(self) -> SmemWorkspace:
        """Place the four role lifetimes on one overlay, not yet finalized.

        Every CTA walks a strict subset of the roles in time order (exchange then
        index copy then refine, exchange then planner, helper then refine or
        walk, pusher then refine, or under the pull dispatch pusher then index
        copy then walk), and no two roles are live at once, so the physical
        footprint is the largest single role rather than their sum. The index
        copy runs the helper's fold itself, so its tables sit in the helper
        lifetime beside the fold's regions; the walk's claim table sits in the
        refine lifetime, staged after the copy's closing barrier.
        """
        warp_count = self.threads_per_cta // 32
        workspace = SmemWorkspace()
        workspace.register_mbarrier(self.exchange_mbarrier_region, 1)
        workspace.register_mbarrier(
            self.pusher_mbarrier_region, warp_count * self.pusher_pipeline_depth
        )
        workspace.register_mbarrier(
            self.refine_mbarrier_region, self.refine_participant_groups
        )
        if self.pull_walks:
            workspace.register_mbarrier(
                self.refine_pull_mbarrier_region, warp_count * self.pull_pipeline_depth
            )
        overlay = workspace.create_overlay("genphase.token_comm.smem.role")

        # Staged at slab rather than route granularity: the wire transfer moves a
        # whole padded slab so that its byte count is bulk-copy aligned even when
        # `max_tokens_per_rank * topk` is not, and the staging buffer has to cover
        # what the transfer reads.
        # The Sentinel exchange stages the whole plane; a pull exchange CTA
        # stages its chunk, sorts it into lists and keeps those until they leave.
        exchange = overlay.add_lifetime("exchange")
        if self.dispatch_protocol == "sentinel":
            # One spare word per warp ahead of the staged plane (see
            # ``sentinel_layout_pad_regions``).
            exchange.register_tensor(
                self.sentinel_exchange_pad_region, cutlass.Int32, (warp_count,)
            )
        staged_routes = self.topk_slab_elements
        if self.dispatch_protocol == "pull":
            staged_routes = self.exchange_chunk_routes
        exchange.register_tensor(
            self.exchange_topk_stage_region,
            self.topk_index_dtype,
            (staged_routes,),
            byte_alignment=16,
        )
        if self.apply_topk_at_fc1:
            exchange.register_tensor(
                self.exchange_score_stage_region,
                cutlass.Float32,
                (staged_routes,),
                byte_alignment=16,
            )
        if self.dispatch_protocol == "pull":
            for region_name in (
                self.exchange_histogram_region,
                self.exchange_list_base_region,
            ):
                exchange.register_tensor(
                    region_name,
                    cutlass.Int32,
                    (self.exchange_bucket_count,),
                    byte_alignment=16,
                )
            world_columns = round_up(self.world_size, 4)
            exchange.register_tensor(
                self.exchange_destination_region,
                cutlass.Int32,
                (3, world_columns),
                stride=(world_columns, 1),
                byte_alignment=16,
            )
            exchange.register_tensor(
                self.exchange_warp_totals_region,
                cutlass.Int32,
                (warp_count,),
                byte_alignment=16,
            )
            exchange.register_tensor(
                self.exchange_route_list_region,
                cutlass.Int32,
                (self.exchange_list_capacity,),
                byte_alignment=16,
            )
            if self.apply_topk_at_fc1:
                exchange.register_tensor(
                    self.exchange_score_list_region,
                    cutlass.Int32,
                    (self.exchange_list_capacity,),
                    byte_alignment=16,
                )

        helper = overlay.add_lifetime("helper")
        helper.register_tensor(
            self.helper_count_matrix_region,
            cutlass.Int32,
            (self.world_size, self.local_expert_bucket_count),
            stride=(self.local_expert_bucket_count, 1),
            byte_alignment=16,
        )
        for region_name in (
            self.helper_totals_region,
            self.helper_scan_input_region,
            self.helper_prefix_region,
        ):
            helper.register_tensor(
                region_name,
                cutlass.Int32,
                (self.local_expert_bucket_count,),
                byte_alignment=16,
            )
        helper.register_tensor(
            self.helper_warp_totals_region,
            cutlass.Int32,
            (self.threads_per_cta // 32,),
            byte_alignment=16,
        )
        for region_name in (
            self.helper_expert_order_region,
            self.helper_expert_sizes_in_order_region,
        ):
            helper.register_tensor(
                region_name,
                cutlass.Int32,
                (self.local_expert_bucket_count,),
                byte_alignment=16,
            )
        if not self.ranks_experts_in_warp:
            helper.register_tensor(
                self.helper_load_bin_region,
                cutlass.Int32,
                (self.load_bin_count,),
                byte_alignment=16,
            )
        if self.dispatch_protocol == "pull":
            # The helper and every index-copy CTA plan from the same rows in
            # this lifetime; the copy's own tables sit beside the plan's.
            segments = self.list_segment_count
            buckets = self.local_expert_bucket_count
            helper.register_tensor(
                self.helper_chunk_counts_region,
                cutlass.Int32,
                (segments,),
                byte_alignment=16,
            )
            helper.register_tensor(
                self.copy_list_base_region,
                cutlass.Int32,
                (round_up(self.world_size * self.exchange_cta_count + 1, 4),),
                byte_alignment=16,
            )
            # The segment bases carry the total as one more entry.
            helper.register_tensor(
                self.copy_segment_base_region,
                cutlass.Int32,
                (round_up(segments + 1, 4),),
                byte_alignment=16,
            )
            helper.register_tensor(
                self.copy_row_base_region, cutlass.Int32, (segments,), byte_alignment=16
            )
            helper.register_tensor(
                self.copy_ring_totals_region,
                cutlass.Int32,
                (self.world_size, buckets),
                stride=(buckets, 1),
                byte_alignment=16,
            )
            helper.register_tensor(
                self.copy_data_base_region, cutlass.Int32, (buckets,), byte_alignment=16
            )
            if self.token_tile_strided:
                helper.register_tensor(
                    self.copy_tile_rows_region,
                    cutlass.Int32,
                    (2, self.experts_per_rank),
                    stride=(self.experts_per_rank, 1),
                    byte_alignment=16,
                )
        if self.ranks_experts_in_cta:
            shares = self.threads_per_cta // self.cta_rank_experts
            helper.register_tensor(
                self.helper_rank_shares_region,
                cutlass.Int32,
                (shares, self.cta_rank_experts),
                stride=(self.cta_rank_experts, 1),
                byte_alignment=16,
            )
        if self.dispatch_protocol == "pull" and not self.ranks_experts_in_warp:
            helper.register_tensor(
                self.helper_warp_bins_region,
                cutlass.Int32,
                (warp_count, self.load_bin_count),
                stride=(self.load_bin_count, 1),
                byte_alignment=16,
            )
        if self.dispatch_protocol == "sentinel":
            # The planners follow the exchange in the same CTAs, so they reuse the
            # helper lifetime and its tables, with their own bases and records.
            helper.register_tensor(
                self.helper_plan_bases_region,
                cutlass.Int32,
                (2, self.local_expert_bucket_count),
                stride=(self.local_expert_bucket_count, 1),
                byte_alignment=16,
            )
            helper.register_tensor(
                self.helper_route_records_region,
                cutlass.Int32,
                (self.routes_per_rank,),
                byte_alignment=16,
            )
            if self.token_tile_strided:
                helper.register_tensor(
                    self.helper_tile_rows_region,
                    cutlass.Int32,
                    (2, self.experts_per_rank),
                    stride=(self.experts_per_rank, 1),
                    byte_alignment=16,
                )
            if not self.ranks_experts_in_warp:
                helper.register_tensor(
                    self.helper_warp_bins_region,
                    cutlass.Int32,
                    (warp_count, self.load_bin_count),
                    stride=(self.load_bin_count, 1),
                    byte_alignment=16,
                )

        pusher = overlay.add_lifetime("pusher")
        for region_name, dtype, row_elements in (
            (
                self.pusher_data_stage_region,
                self.activation_dtype,
                self.plain_activation_row_elements,
            ),
            (
                self.pusher_sf_stage_region,
                self.activation_sf_dtype,
                self.plain_activation_sf_row_elements,
            ),
        ):
            pusher.register_tensor(
                region_name,
                dtype,
                (warp_count, self.pusher_pipeline_depth, row_elements),
                stride=(self.pusher_pipeline_depth * row_elements, row_elements, 1),
                byte_alignment=16,
            )

        # The pull walk takes this lifetime too, with the expert table, its claim
        # bases and its rings only: its warps claim on their own and write
        # canonical atoms directly.
        refine = overlay.add_lifetime("refine")
        if self.dispatch_protocol == "sentinel":
            refine.register_tensor(
                self.refine_claim_region,
                cutlass.Int32,
                (self.refine_participant_groups, self.refine_claim_fields),
                stride=(self.refine_claim_fields, 1),
                byte_alignment=16,
            )
        refine.register_tensor(
            self.refine_expert_table_region,
            cutlass.Int32,
            (self.refine_table_rows, self.local_expert_bucket_count),
            stride=(self.local_expert_bucket_count, 1),
            byte_alignment=16,
        )
        if self.refine_plans_table:
            refine.register_tensor(
                self.refine_block_count_region, cutlass.Int32, (1,), byte_alignment=16
            )
        if self.dispatch_protocol == "sentinel":
            refine.register_tensor(
                self.refine_output_region,
                self.activation_sf_dtype,
                (
                    self.refine_participant_groups,
                    self.refine_output_slots,
                    self.refine_output_stage_elements,
                ),
                stride=(
                    self.refine_output_slots * self.refine_output_stage_elements,
                    self.refine_output_stage_elements,
                    1,
                ),
                byte_alignment=16,
            )
        if self.pull_walks:
            refine.register_tensor(
                self.refine_pull_chunk_base_region,
                cutlass.Int32,
                (self.pull_table_row_count, self.local_expert_bucket_count + 1),
                stride=(self.local_expert_bucket_count + 1, 1),
                byte_alignment=16,
            )
            refine.register_tensor(
                self.refine_pull_ring_region,
                self.activation_dtype,
                (
                    warp_count,
                    self.pull_pipeline_depth,
                    self.plain_activation_row_elements,
                ),
                stride=(
                    self.pull_pipeline_depth * self.plain_activation_row_elements,
                    self.plain_activation_row_elements,
                    1,
                ),
                byte_alignment=128,
            )

        return workspace

    @property
    def smem_workspace(self) -> SmemWorkspace:
        return self._smem_workspace

    @property
    def smem_bytes(self) -> int:
        return self._smem_workspace.total_bytes

    # ------------------------------------------------------------------
    # Typed views on workspace regions
    # ------------------------------------------------------------------

    @cute.jit
    def topk_index_inbox_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.topk_index_inbox_region)

    @cute.jit
    def topk_score_inbox_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.apply_topk_at_fc1):
            return None
        return device_workspace.tensor(self.topk_score_inbox_region)

    @cute.jit
    def _payload_plane_by_dense_row(
        self, device_workspace: DeviceWorkspace, region_name: str
    ) -> cute.Tensor:
        """``(world * tokens, columns)`` view of a payload plane.

        The registered shape keeps the source rank as its own mode, which is how
        the push side addresses a row. The gather side wants the opposite: one flat
        row per (source rank, source token) pair, because that is what
        ``gather_index`` stores and what the FC1 gather4 descriptor indexes. Those
        two leading modes are contiguous, so merging them is a reshape rather than
        a reinterpretation, and both the column extent and the padded row stride
        are read back off the region so they stay stated only at its registration.
        """
        _, _, columns = device_workspace.region(region_name).shape
        row_elements = device_workspace.stride(region_name)[1]
        return cute.make_tensor(
            device_workspace.ptr(region_name),
            cute.make_layout(
                (self.world_size * self.max_tokens_per_rank, columns),
                stride=(row_elements, 1),
            ),
        )

    @cute.jit
    def plain_activation_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.plain_activation_region)

    @cute.jit
    def plain_activation_sf_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> cute.Tensor:
        return device_workspace.tensor(self.plain_activation_sf_region)

    @cute.jit
    def dense_activation_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return self._payload_plane_by_dense_row(
            device_workspace, self.plain_activation_region
        )

    @cute.jit
    def dense_activation_sf_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> cute.Tensor:
        return self._payload_plane_by_dense_row(
            device_workspace, self.plain_activation_sf_region
        )

    @cute.jit
    def fc1_token_pool_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        """``(pool rows, hidden)`` view of the token pool (``fc1_token_pool``), rows at the padded row stride."""
        row_elements = self.plain_activation_row_elements
        return cute.make_tensor(
            device_workspace.ptr(self.fc1_token_pool_region),
            cute.make_layout(
                (self.fc1_token_pool_rows, self.hidden_size), stride=(row_elements, 1)
            ),
        )

    @cute.jit
    def canonical_activation_sf_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> cute.Tensor:
        """The SF pool in the layout the block-scaled MMA reads.

        ``tile_atom_to_shape_SF`` takes the *data* shape, not the SF shape: the
        broadcast mode it inserts already absorbs ``sf_vec_size``.
        """
        layout = tile_atom_to_shape_SF(
            (self.worst_case_sf_rows + self.shared_expert_sf_rows, self.hidden_size, 1),
            self.quant_kind.sf_vec_size,
        )
        return cute.make_tensor(
            device_workspace.ptr(self.canonical_activation_sf_region),
            cute.select(layout, mode=[0, 1]),
        )

    @cute.jit
    def expert_sizes_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.expert_sizes_region)

    @cute.jit
    def expert_order_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        """The processing order: the local expert at each position the main kernel walks."""
        return device_workspace.tensor(self.expert_order_region)

    @cute.jit
    def scheduler_expert_sizes_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> cute.Tensor:
        """Per-expert sizes in the order the FC12 scheduler must walk them.

        The scheduler rebuilds every pool base by scanning this array in index
        order, so handing it the sizes already permuted is what makes the
        processing order invisible to it: its running prefix reproduces the
        layout this component wrote, unchanged. ``expert_sizes`` stays in expert
        order for the readers that address an expert -- the refine table and the
        main kernel's per-expert scale-factor gate.
        """
        return device_workspace.tensor(self.expert_sizes_in_order_region)

    @cute.jit
    def token_src_metadata_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> cute.Tensor:
        return device_workspace.tensor(self.token_src_metadata_region)

    @cute.jit
    def gather_index_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.gather_index_region)

    @cute.jit
    def fc1_topk_scores_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.apply_topk_at_fc1):
            return None
        return device_workspace.tensor(self.fc1_topk_scores_region)

    @cute.jit
    def stamp(self, device_workspace: DeviceWorkspace, row: int, column: Int32) -> None:
        """Record the current %globaltimer at one stamp site; a no-op without stamps."""
        if cutlass.const_expr(self.stamps):
            device_workspace.tensor(self.stamps_region)[row, column] = (
                sentinel_ptx.globaltimer_ns()
            )

    @cute.jit
    def stamp_latest(
        self, device_workspace: DeviceWorkspace, row: int, column: Int32
    ) -> None:
        """Keep the latest %globaltimer seen at a stamp site several threads reach."""
        if cutlass.const_expr(self.stamps):
            address = device_workspace.tensor(self.stamps_region)[
                row, None
            ].iterator.toint() + Int64(column) * Int64(8)
            sentinel_ptx.red_max_relaxed_gpu_u64(address, sentinel_ptx.globaltimer_ns())

    @cute.jit
    def stamp_plan_step(
        self, device_workspace: DeviceWorkspace, step: str, index: Int32
    ) -> None:
        """Stamp one plan step (``plan_detail_columns``); a no-op without stamps or past the last column."""
        if cutlass.const_expr(self.stamps):
            first, count = self.plan_detail_columns[step]
            if cutlass.const_expr(first + count <= self.stamp_columns):
                self.stamp(
                    device_workspace, self.stamp_plan_detail, Int32(first) + index
                )

    @cute.jit
    def select_pingpong_buffer(
        self,
        device_workspace: DeviceWorkspace,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
    ) -> Int32:
        """Point device_workspace at this round's ping-pong buffer and return its index.

        The symmetric workspace holds the ping and pong buffers
        pingpong_buffer_bytes apart; round epoch uses buffer epoch & 1 (0 ping,
        1 pong). Every kernel of a round reads the same epoch:
        only the end-of-round reduce advances it, after the round's other kernels
        have finished.
        """
        device_workspace.assign_device_members(local_workspace, shared_workspace)
        epoch = sentinel_ptx.ld_volatile_u32(
            device_workspace.ptr(self.sentinel_control_region).toint()
            + Int64(self.sentinel_epoch_word * 4)
        )
        pingpong_buffer = Int32(epoch & Uint32(1))
        buffer_base = shared_workspace.toint() + Int64(pingpong_buffer) * Int64(
            self.pingpong_buffer_bytes
        )
        device_workspace.assign_device_members(
            local_workspace,
            cute.make_ptr(
                cutlass.Uint8, buffer_base, AddressSpace.gmem, assumed_align=128
            ),
        )
        return pingpong_buffer

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    @cute.jit
    def launch(
        self,
        args: GenphaseTokenCommArgs,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
        staging_peer_rank_ptr_mapper_host=None,
    ) -> None:
        """Launch the communication kernel ahead of the dependent main kernel.

        ``staging_peer_rank_ptr_mapper_host`` carries the peer offsets of the
        allocation that holds ``args.activation`` and ``args.activation_sf``; the
        walking CTAs of the pull dispatch read peers' rows through it, and no
        other role uses it.
        """
        if cutlass.const_expr(self.apply_topk_at_fc1 and args.topk_scores is None):
            raise ValueError("apply_topk_at_fc1 requires topk_scores.")
        if cutlass.const_expr(
            self.pull_walks and staging_peer_rank_ptr_mapper_host is None
        ):
            raise ValueError(
                "The pull dispatch requires the staging allocation's peer offsets."
            )
        if cutlass.const_expr(
            args.topk_indices.element_type is not self.topk_index_dtype
        ):
            raise TypeError(
                f"topk_indices must be the declared {self.topk_index_dtype}, got {args.topk_indices.element_type}."
            )
        if cutlass.const_expr(
            args.activation.element_type is not self.activation_dtype
        ):
            raise TypeError(
                f"activation must be {self.activation_dtype}, got {args.activation.element_type}."
            )
        if cutlass.const_expr(
            args.activation_sf.element_type is not self.activation_sf_dtype
        ):
            raise TypeError(
                f"activation_sf must be {self.activation_sf_dtype}, got {args.activation_sf.element_type}."
            )
        # Row addressing uses each plane's own layout, so the padded stride is a
        # requirement rather than something this component can paper over: a whole
        # padded row is bulk-copied, and a tighter caller stride would be read past
        # its last row. `from_dlpack` unpacks sub-byte tensors, so both strides are
        # already in logical elements and comparable.
        for plane_name, plane, row_elements in (
            ("activation", args.activation, self.plain_activation_row_elements),
            (
                "activation_sf",
                args.activation_sf,
                self.plain_activation_sf_row_elements,
            ),
        ):
            if cutlass.const_expr(isinstance(plane.stride[0], int)):
                if cutlass.const_expr(plane.stride[0] != row_elements):
                    raise ValueError(
                        f"{plane_name} needs the padded row stride {row_elements}, got {plane.stride[0]}."
                    )

        staging_peer_rank_ptr_mapper = None
        if cutlass.const_expr(self.pull_walks):
            staging_peer_rank_ptr_mapper = (
                staging_peer_rank_ptr_mapper_host.make_device_object()
            )
        self._kernel(
            args.topk_indices,
            args.topk_scores,
            args.activation,
            args.activation_sf,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host.make_device_object(),
            device_workspace,
            staging_peer_rank_ptr_mapper,
        ).launch(
            grid=[self.launch_cluster_width, 1, self.launch_cluster_count],
            cluster=(self.launch_cluster_width, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def _kernel(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        device_workspace: DeviceWorkspace,
        staging_peer_rank_ptr_mapper: Optional[SymmetricBufferDevice],
    ) -> None:
        """Dispatch CTA roles.

        The dependent main kernel is released before anything else so its weight
        streaming overlaps the whole payload transfer.

        Every role takes the launch context explicitly rather than through
        instance attributes: the component is shared by all CTAs, and stashing
        per-CTA values on it makes the data flow invisible at the call site.
        """
        cute.arch.griddepcontrol_launch_dependents()

        cta_idx_in_cluster, _, cluster_idx = cute.arch.block_idx()
        linear_cta_idx = (
            cluster_idx * Int32(self.launch_cluster_width) + cta_idx_in_cluster
        )

        # Cluster padding: these CTAs hold no role and are counted by no readiness
        # target, so they leave before allocating anything rather than falling
        # through the whole body. The predicate is CTA-uniform and no barrier here
        # is cluster-scoped, so the cluster peer is unaffected; the dependent grid
        # was released above.
        if linear_cta_idx >= Int32(self.grid_cta_count):
            exit()

        thread_idx, _, _ = cute.arch.thread_idx()

        storage = cutlass.utils.SmemAllocator().allocate(
            self._smem_workspace.storage_class()
        )
        smem_base = storage.buffer.data_ptr()

        if cutlass.const_expr(self.dispatch_protocol == "sentinel"):
            # The top-k planes do not depend on the round's ping-pong buffer, so their
            # load leaves before the epoch read and the two latencies overlap.
            if linear_cta_idx < Int32(self.world_size):
                self._issue_topk_load(topk_indices, topk_scores, smem_base, thread_idx)
        pingpong_buffer = self.select_pingpong_buffer(
            device_workspace, local_workspace, shared_workspace
        )
        if thread_idx == Int32(0):
            self.stamp(device_workspace, self.stamp_comm_entry, linear_cta_idx)

        if linear_cta_idx < Int32(self.exchange_cta_count):
            if cutlass.const_expr(self.dispatch_protocol == "sentinel"):
                # Dual identity: this CTA writes into peer `linear_cta_idx`'s inbox,
                # then plans this rank's dispatch and scatters what that same peer
                # sent.
                self._send_topk_inbox(
                    device_workspace,
                    peer_rank_ptr_mapper,
                    local_rank,
                    smem_base,
                    linear_cta_idx,
                    pingpong_buffer,
                    thread_idx,
                )
                self._run_planner(
                    device_workspace, local_rank, smem_base, linear_cta_idx, thread_idx
                )
                if thread_idx == Int32(0):
                    self.stamp(device_workspace, self.stamp_comm_exit, linear_cta_idx)
                exit()
            else:
                # This CTA sends every peer its routes among chunk `linear_cta_idx`
                # of the plane.
                self._run_exchange(
                    device_workspace,
                    peer_rank_ptr_mapper,
                    local_rank,
                    topk_indices,
                    topk_scores,
                    smem_base,
                    linear_cta_idx,
                    pingpong_buffer,
                    thread_idx,
                )
        elif linear_cta_idx == Int32(self.helper_cta_index):
            # The planners publish everything the helper would under the Sentinel
            # dispatch, so there it goes straight to the refine.
            if cutlass.const_expr(self.dispatch_protocol == "pull"):
                self._run_helper(device_workspace, local_rank, smem_base, thread_idx)
        else:
            # Under the pull dispatch the target ranks fetch the payload, so the
            # pusher CTAs push nothing.
            if cutlass.const_expr(self.dispatch_protocol == "sentinel"):
                self._run_pusher(
                    device_workspace,
                    peer_rank_ptr_mapper,
                    local_rank,
                    topk_indices,
                    activation,
                    activation_sf,
                    smem_base,
                    linear_cta_idx - Int32(self.pusher_cta_begin),
                    thread_idx,
                )

        if cutlass.const_expr(self.dispatch_protocol == "pull"):
            # The exchange CTAs and the pushers each copy one slice of every
            # source's lists into the index lists, from this one call site.
            copier = linear_cta_idx
            if linear_cta_idx > Int32(self.helper_cta_index):
                copier = linear_cta_idx - Int32(1)
            if (linear_cta_idx != Int32(self.helper_cta_index)) & (
                copier < Int32(self.index_copy_cta_count)
            ):
                self._run_index_copy(
                    device_workspace, smem_base, local_rank, copier, thread_idx
                )
            # The exchange CTAs, [0, exchange_cta_count), leave once their slice is
            # copied.
            if (linear_cta_idx >= Int32(0)) & (
                linear_cta_idx < Int32(self.exchange_cta_count)
            ):
                if thread_idx == Int32(0):
                    self.stamp(device_workspace, self.stamp_comm_exit, linear_cta_idx)
                exit()

        if cutlass.const_expr(self.pull_walks):
            self._run_pull(
                device_workspace,
                smem_base,
                linear_cta_idx,
                thread_idx,
                activation,
                activation_sf,
                staging_peer_rank_ptr_mapper,
            )
        elif cutlass.const_expr(self.dispatch_protocol == "sentinel"):
            self._run_refine(device_workspace, smem_base, linear_cta_idx, thread_idx)
        if cutlass.const_expr(self.dispatch_protocol == "pull"):
            # Every index-copy CTA had read the count rows before the walk
            # opened; the helper resets them on its way out. When the main
            # kernel's pull warps fetch the rows the helper does not walk, so it
            # waits for the index lists itself.
            if linear_cta_idx == Int32(self.helper_cta_index):
                if cutlass.const_expr(not self.pull_walks):
                    if thread_idx == Int32(0):
                        spin_wait(
                            device_workspace.ptr(self.metadata_ready_region),
                            lambda value: value == Int32(self.metadata_ready_target),
                            scope="gpu",
                        )
                    cute.arch.sync_threads()
                self._reset_count_rows(device_workspace, thread_idx)
        if cutlass.const_expr(self.stamps):
            cute.arch.sync_threads()
            if thread_idx == Int32(0):
                self.stamp(device_workspace, self.stamp_comm_exit, linear_cta_idx)

        device_workspace.remove_device_members()

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

    @cute.jit
    def _run_exchange(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        smem_base: cute.Pointer,
        chunk: Int32,
        pingpong_buffer: Int32,
        thread_idx: Int32,
    ) -> None:
        """Sort one chunk of this rank's routes into a list per destination rank and send them (pull dispatch).

        This is the only cross-rank round trip on the critical path, and it has no
        routing dependency, so it leaves at kernel entry. See "Expert-id lists"
        above for what the CTA sends. The sort is a counting sort in SMEM over
        (destination, local expert) buckets: each staged id is replaced in place
        by its route's record (its index inside its bucket, shifted above the
        ``exchange_record_bucket_bits`` bits that name the bucket); the bucket
        sizes are scanned into offsets, with every destination's list padded to
        whole 16-byte vectors; and each route id and its score are placed at its
        bucket's offset plus its index. A thread takes ``4 *
        exchange_vector_batch`` routes per step, one at a time in the count pass
        and in whole 16-byte vectors in the list pass, loads a step's routes
        before it uses any, and keeps its atomics and stores free of branches,
        so their latencies overlap.
        """
        iket.range_push("exchange")
        threads = self.threads_per_cta
        lane_idx = thread_idx % Int32(32)
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        world = self.world_size
        buckets = self.local_expert_bucket_count
        bucket_count = self.exchange_bucket_count
        bucket_bits = self.exchange_record_bucket_bits
        chunk_capacity = self.exchange_chunk_routes
        id_stage = self._smem_workspace.tensor(
            self.exchange_topk_stage_region, smem_base
        )
        histogram = self._smem_workspace.tensor(
            self.exchange_histogram_region, smem_base
        )
        bucket_base = self._smem_workspace.tensor(
            self.exchange_list_base_region, smem_base
        )
        destinations = self._smem_workspace.tensor(
            self.exchange_destination_region, smem_base
        )
        warp_totals = self._smem_workspace.tensor(
            self.exchange_warp_totals_region, smem_base
        )
        route_list = self._smem_workspace.tensor(
            self.exchange_route_list_region, smem_base
        )
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_stage = self._smem_workspace.tensor(
                self.exchange_score_stage_region, smem_base
            )
            score_list = self._smem_workspace.tensor(
                self.exchange_score_list_region, smem_base
            )
        load_mbarrier = self._smem_workspace.ptr(
            self.exchange_mbarrier_region, smem_base
        )

        chunk_begin = chunk * Int32(chunk_capacity)
        chunk_routes = cutlass.max(
            cutlass.min(
                Int32(self.routes_per_rank) - chunk_begin, Int32(chunk_capacity)
            ),
            Int32(0),
        )
        id_bytes = int(self.topk_index_dtype.width) // 8
        route_bytes = id_bytes
        if cutlass.const_expr(self.apply_topk_at_fc1):
            route_bytes = id_bytes + 4

        if thread_idx == Int32(0):
            cute.arch.mbarrier_init(load_mbarrier, 1)
        cute.arch.mbarrier_init_fence()
        for bucket_round in cutlass.range_constexpr(ceil_div(bucket_count, threads)):
            bucket = Int32(bucket_round * threads) + thread_idx
            if bucket < Int32(bucket_count):
                histogram[bucket] = Int32(0)
        cute.arch.sync_threads()

        if thread_idx == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(
                load_mbarrier, chunk_routes * Int32(route_bytes)
            )
            if chunk_routes > Int32(0):
                tma_load_1d(
                    id_stage.iterator,
                    topk_indices.iterator.toint()
                    + Int64(chunk_begin) * Int64(id_bytes),
                    load_mbarrier,
                    chunk_routes * Int32(id_bytes),
                )
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    tma_load_1d(
                        score_stage.iterator,
                        topk_scores.iterator.toint() + Int64(chunk_begin) * Int64(4),
                        load_mbarrier,
                        chunk_routes * Int32(4),
                    )
        cute.arch.mbarrier_wait(load_mbarrier, 0)
        if thread_idx == Int32(0):
            self.stamp_plan_step(device_workspace, "x_loaded", chunk)

        iket.range_push("exchange.count")
        if cutlass.const_expr(self.topk_index_dtype is cutlass.Int64):
            self._canonicalize_wire_indices(id_stage, thread_idx)
        records = self._wire_index_view(id_stage)
        capacity_vectors = chunk_capacity // 4
        vectors = chunk_routes // Int32(4)
        batch = self.exchange_vector_batch
        steps = ceil_div(ceil_div(capacity_vectors, threads), batch)
        words = cute.make_rmem_tensor((batch * 4,), cutlass.Int32)
        # The count pass takes one route per thread per slot, consecutive threads
        # on consecutive routes, so a bucket's indices follow the routes' order
        # within each warp: an expert's rows keep their tokens' order.
        route_steps = ceil_div(ceil_div(chunk_capacity, threads), batch * 4)
        for step in cutlass.range(route_steps, unroll=1):
            for slot in cutlass.range_constexpr(batch * 4):
                route = (step * Int32(batch * 4) + Int32(slot)) * Int32(
                    threads
                ) + thread_idx
                words[slot] = records[cutlass.min(route, Int32(chunk_capacity - 1))]
            for slot in cutlass.range_constexpr(batch * 4):
                route = (step * Int32(batch * 4) + Int32(slot)) * Int32(
                    threads
                ) + thread_idx
                # One unsigned compare rejects a negative or out-of-range id.
                routed = (route < chunk_routes) & (
                    cutlass.Uint32(words[slot]) < cutlass.Uint32(self.expert_count)
                )
                expert = cutlass.select_(routed, words[slot], Int32(0))
                bucket = expert
                if cutlass.const_expr(buckets != self.experts_per_rank):
                    destination = expert // Int32(self.experts_per_rank)
                    bucket = (
                        destination * Int32(buckets)
                        + expert
                        - destination * Int32(self.experts_per_rank)
                    )
                index_in_bucket = atoms_add_u32_if(
                    histogram.iterator + bucket,
                    Int32(1),
                    cutlass.select_(routed, Int32(1), Int32(0)),
                )
                words[slot] = cutlass.select_(
                    routed, (index_in_bucket << Int32(bucket_bits)) | bucket, Int32(-1)
                )
            for slot in cutlass.range_constexpr(batch * 4):
                route = (step * Int32(batch * 4) + Int32(slot)) * Int32(
                    threads
                ) + thread_idx
                if route < chunk_routes:
                    records[route] = words[slot]
        cute.arch.sync_threads()

        listed = _flat_exclusive_scan(
            histogram,
            bucket_base,
            warp_totals,
            bucket_count,
            threads,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        cute.arch.sync_threads()
        # Each destination's list starts on a 16-byte boundary, so its bulk store
        # does too; row 0 is the shift from the unpadded offsets, which every
        # bucket offset then takes on, row 1 the list start and row 2 its length.
        if warp_idx == Int32(0):
            carried = Int32(0)
            for destination_round in cutlass.range_constexpr(ceil_div(world, 32)):
                destination = Int32(destination_round * 32) + lane_idx
                first = bucket_base[
                    cutlass.min(destination, Int32(world - 1)) * Int32(buckets)
                ]
                end = cutlass.select_(
                    destination + Int32(1) < Int32(world),
                    bucket_base[
                        cutlass.min(destination + Int32(1), Int32(world - 1))
                        * Int32(buckets)
                    ],
                    listed,
                )
                length = cutlass.select_(
                    destination < Int32(world), end - first, Int32(0)
                )
                padded = (length + Int32(3)) // Int32(4) * Int32(4)
                inclusive = carried + _warp_inclusive_sum(padded, lane_idx)
                if destination < Int32(world):
                    destinations[0, destination] = inclusive - padded - first
                    destinations[1, destination] = inclusive - padded
                    destinations[2, destination] = length
                    # The padding travels with the list and keeps the sentinel, so
                    # the destination never takes it for a route.
                    for pad in cutlass.range_constexpr(3):
                        if length + Int32(pad) < padded:
                            route_list[inclusive - padded + length + Int32(pad)] = (
                                Int32(-1)
                            )
                            if cutlass.const_expr(self.apply_topk_at_fc1):
                                score_list[inclusive - padded + length + Int32(pad)] = (
                                    Int32(-1)
                                )
                carried = Int32(cute.arch.shuffle_sync(inclusive, Int32(31)))
        cute.arch.sync_threads()
        for bucket_round in cutlass.range_constexpr(ceil_div(bucket_count, threads)):
            bucket = Int32(bucket_round * threads) + thread_idx
            if bucket < Int32(bucket_count):
                bucket_base[bucket] = (
                    bucket_base[bucket] + destinations[0, bucket // Int32(buckets)]
                )
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            self.stamp_plan_step(device_workspace, "x_counted", chunk)
        iket.range_pop()

        iket.range_push("exchange.list")
        positions = cute.make_rmem_tensor((batch * 4,), cutlass.Int32)
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_words = cute.make_rmem_tensor((batch * 4,), cutlass.Int32)
        for step in cutlass.range(steps, unroll=1):
            for vector_slot in cutlass.range_constexpr(batch):
                vector = (step * Int32(batch) + Int32(vector_slot)) * Int32(
                    threads
                ) + thread_idx
                staged_vector = cutlass.min(
                    vector, Int32(capacity_vectors - 1)
                ) * Int32(4)
                loaded = lds128_v4_b32(records.iterator + staged_vector)
                for word in cutlass.range_constexpr(4):
                    words[vector_slot * 4 + word] = cutlass.select_(
                        vector < vectors, loaded[word], Int32(-1)
                    )
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    # A score whose bits are the arrival sentinel (a NaN) travels
                    # as 0x7FFFFFFF.
                    loaded_scores = lds128_v4_b32(score_stage.iterator + staged_vector)
                    for word in cutlass.range_constexpr(4):
                        score_words[vector_slot * 4 + word] = cutlass.select_(
                            loaded_scores[word] == Int32(-1),
                            Int32(0x7FFFFFFF),
                            loaded_scores[word],
                        )
            for slot in cutlass.range_constexpr(batch * 4):
                record = cutlass.max(words[slot], Int32(0))
                positions[slot] = bucket_base[
                    record & Int32((1 << bucket_bits) - 1)
                ] + (record >> Int32(bucket_bits))
            for vector_slot in cutlass.range_constexpr(batch):
                vector = (step * Int32(batch) + Int32(vector_slot)) * Int32(
                    threads
                ) + thread_idx
                for word in cutlass.range_constexpr(4):
                    live = cutlass.select_(
                        words[vector_slot * 4 + word] >= Int32(0), Int32(1), Int32(0)
                    )
                    sts_b32_if(
                        route_list.iterator + positions[vector_slot * 4 + word],
                        chunk_begin + vector * Int32(4) + Int32(word),
                        live,
                    )
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        sts_b32_if(
                            score_list.iterator + positions[vector_slot * 4 + word],
                            score_words[vector_slot * 4 + word],
                            live,
                        )
        # The counts and the lists were written by generic stores; the bulk stores
        # read them through the async proxy.
        cute.arch.fence_proxy(kind="async.shared", space="cta")
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            self.stamp_plan_step(device_workspace, "x_listed", chunk)
        iket.range_pop()

        # The first lane of warp ``d % warps`` sends destination d its count row,
        # its lists and the release, so every warp-uniform operand stays uniform
        # and the destinations' sends overlap. The peer's addresses are this
        # rank's addresses shifted by the symmetric-heap delta, so the layout
        # supplies the intra-region offset and the mapper supplies the inter-rank
        # one.
        slab_offset = Int64(
            (local_rank * Int32(self.exchange_cta_count) + chunk)
            * Int32(chunk_capacity)
        ) * Int64(4)
        warps = threads // 32
        for destination_round in cutlass.range_constexpr(ceil_div(world, warps)):
            destination = Int32(destination_round * warps) + warp_idx
            if (lane_idx == Int32(0)) & (destination < Int32(world)):
                peer_offset = peer_rank_ptr_mapper.map(Int64(0), destination, Int64(0))
                own_counts = device_workspace.tensor(self.topk_count_inbox_region)[
                    local_rank, chunk, None
                ]
                cp_async_bulk_s2g(
                    cute.make_ptr(
                        cutlass.Int32,
                        own_counts.iterator.toint() + peer_offset,
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    histogram.iterator + destination * Int32(buckets),
                    Int32(buckets * 4),
                )
                length = destinations[2, destination]
                if length > Int32(0):
                    list_bytes = (length + Int32(3)) // Int32(4) * Int32(16)
                    list_begin = destinations[1, destination]
                    cp_async_bulk_s2g(
                        cute.make_ptr(
                            cutlass.Int32,
                            device_workspace.ptr(self.topk_index_inbox_region).toint()
                            + slab_offset
                            + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        route_list.iterator + list_begin,
                        list_bytes,
                    )
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        cp_async_bulk_s2g(
                            cute.make_ptr(
                                cutlass.Int32,
                                device_workspace.ptr(
                                    self.topk_score_inbox_region
                                ).toint()
                                + slab_offset
                                + peer_offset,
                                AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            score_list.iterator + list_begin,
                            list_bytes,
                        )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                self._clear_previous_topk_ready(
                    device_workspace, destination, pingpong_buffer
                )
                # One add per exchange CTA of this rank; the destination's
                # walking CTAs wait for all of them.
                own_ready_slot = (
                    device_workspace.ptr(self.topk_ready_region) + local_rank
                )
                red_async_add_release_sys_u32(
                    own_ready_slot.toint() + peer_offset, Int32(1)
                )
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            self.stamp_plan_step(device_workspace, "x_released", chunk)
        iket.range_pop()

    @cute.jit
    def _issue_topk_load(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        smem_base: cute.Pointer,
        thread_idx: Int32,
    ) -> None:
        """Start the bulk load of this rank's top-k planes into the exchange stage.

        The first thread initializes the stage's barrier and issues the copies
        itself, so no CTA barrier precedes them; ``_send_topk_inbox`` publishes
        the initialization to the other threads before any of them waits.
        """
        if thread_idx == Int32(0):
            load_mbarrier = self._smem_workspace.ptr(
                self.exchange_mbarrier_region, smem_base
            )
            index_plane_bytes = (
                self.routes_per_rank * int(self.topk_index_dtype.width) // 8
            )
            score_plane_bytes = self.routes_per_rank * 4
            staged_bytes = index_plane_bytes
            if cutlass.const_expr(self.apply_topk_at_fc1):
                staged_bytes = index_plane_bytes + score_plane_bytes
            cute.arch.mbarrier_init(load_mbarrier, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(load_mbarrier, Int32(staged_bytes))
            index_stage = self._smem_workspace.tensor(
                self.exchange_topk_stage_region, smem_base
            )
            tma_load_1d(
                index_stage.iterator,
                topk_indices.iterator,
                load_mbarrier,
                Int32(index_plane_bytes),
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                score_stage = self._smem_workspace.tensor(
                    self.exchange_score_stage_region, smem_base
                )
                tma_load_1d(
                    score_stage.iterator,
                    topk_scores.iterator,
                    load_mbarrier,
                    Int32(score_plane_bytes),
                )

    @cute.jit
    def _send_topk_inbox(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        smem_base: cute.Pointer,
        destination_rank: Int32,
        pingpong_buffer: Int32,
        thread_idx: Int32,
    ) -> None:
        """Write this rank's top-k planes into one peer's Sentinel inbox.

        Slab ``[source = this rank]`` of the peer's inbox is armed with
        0xFFFFFFFF words, which no word sent here takes: an index is canonicalized
        to ``[0, expert_count]``, with ``expert_count`` for a route no rank owns,
        and a score whose bits are 0xFFFFFFFF (a NaN) is sent as 0x7FFFFFFF. The
        whole ``max_tokens_per_rank * topk`` plane goes, so the receiver knows
        every word it waits for. A thread stores whole 16-byte vectors of 32-bit
        elements, each single-copy atomic, and a warp covers 512 contiguous bytes
        per plane.

        The ``topk_ready`` release that follows covers every thread's stores
        through the CTA barrier before it. The receiver reads the planes without
        it and acquires it only after planning; see "Sentinel top-k inbox" above.
        """
        iket.range_push("exchange.inbox")
        load_mbarrier = self._smem_workspace.ptr(
            self.exchange_mbarrier_region, smem_base
        )
        index_stage = self._smem_workspace.tensor(
            self.exchange_topk_stage_region, smem_base
        )
        # Publishes `_issue_topk_load`'s barrier initialization to every waiter.
        cute.arch.sync_threads()
        cute.arch.mbarrier_wait(load_mbarrier, 0)

        peer_offset = peer_rank_ptr_mapper.map(Int64(0), destination_rank, Int64(0))
        slab_bytes = self.topk_slab_elements * 4
        own_slab_offset = Int64(local_rank) * Int64(slab_bytes) + peer_offset
        index_slab = (
            device_workspace.ptr(self.topk_index_inbox_region).toint() + own_slab_offset
        )
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_slab = (
                device_workspace.ptr(self.topk_score_inbox_region).toint()
                + own_slab_offset
            )
            score_stage = self._smem_workspace.tensor(
                self.exchange_score_stage_region, smem_base
            )
            score_bits = cute.make_tensor(
                cute.recast_ptr(score_stage.iterator, dtype=Uint32),
                cute.make_layout((self.topk_slab_elements,)),
            )

        vectors = self.routes_per_rank // 4
        for vector_round in cutlass.range_constexpr(
            ceil_div(vectors, self.threads_per_cta)
        ):
            vector = Int32(vector_round * self.threads_per_cta) + thread_idx
            if vector < Int32(vectors):
                words = cute.make_rmem_tensor((4,), Uint32)
                for element in cutlass.range_constexpr(4):
                    raw_index = index_stage[vector * Int32(4) + Int32(element)]
                    words[element] = Uint32(self.invalid_expert_marker)
                    if not self._is_unowned_expert(raw_index):
                        words[element] = Uint32(raw_index)
                sentinel_ptx.st_volatile_v4_words(
                    index_slab + Int64(vector) * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    for element in cutlass.range_constexpr(4):
                        words[element] = score_bits[vector * Int32(4) + Int32(element)]
                        if words[element] == Uint32(sentinel_ptx.WORD_SENTINEL):
                            words[element] = Uint32(0x7FFFFFFF)
                    sentinel_ptx.st_volatile_v4_words(
                        score_slab + Int64(vector) * Int64(16),
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            self._clear_previous_topk_ready(
                device_workspace, destination_rank, pingpong_buffer
            )
            own_ready_slot = device_workspace.ptr(self.topk_ready_region) + local_rank
            red_async_add_release_sys_u32(
                own_ready_slot.toint() + peer_offset, Int32(1)
            )
        iket.range_pop()

    @cute.jit
    def _clear_previous_topk_ready(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank: Int32,
        pingpong_buffer: Int32,
    ) -> None:
        """Zero this rank's ``topk_ready[peer]`` in the other ping-pong buffer.

        ``peer``'s exchange CTAs are the slot's only writers. Their last writes
        there came in the previous round, and this rank acquired them in a launch
        that completed before this one began. Their next writes there come in the
        next round, from a launch that follows the peer's acquire of the
        ``topk_ready`` release this thread issues right after the zero, so that
        release orders the zero before the writes.
        """
        other_buffer_delta = Int64(Int32(1) - Int32(2) * pingpong_buffer) * Int64(
            self.pingpong_buffer_bytes
        )
        previous_slot = (
            device_workspace.ptr(self.topk_ready_region) + peer_rank
        ).toint() + other_buffer_delta
        sentinel_ptx.st_volatile_u32(previous_slot, Uint32(0))

    @cute.jit
    def _wire_index_view(self, index_stage: cute.Tensor) -> cute.Tensor:
        """Int32 view of a pull exchange CTA's staged chunk.

        For Int64 input this aliases the front half of the same bytes, which is
        where ``_canonicalize_wire_indices`` compacted the narrowed values.
        """
        if cutlass.const_expr(self.topk_index_dtype is cutlass.Int32):
            return index_stage
        return cute.make_tensor(
            cute.recast_ptr(index_stage.iterator, dtype=cutlass.Int32),
            cute.make_layout((self.exchange_chunk_routes,)),
        )

    @cute.jit
    def _canonicalize_wire_indices(
        self, index_stage: cute.Tensor, thread_idx: Int32
    ) -> None:
        """Rewrite a pull exchange CTA's staged Int64 chunk into Int32 ids in place."""
        wire_indices = self._wire_index_view(index_stage)
        routes = self.exchange_chunk_routes
        narrow_rounds = ceil_div(routes, self.threads_per_cta)
        narrowed = cute.make_rmem_tensor((narrow_rounds,), cutlass.Int32)

        for narrow_round in cutlass.range_constexpr(narrow_rounds):
            route = Int32(narrow_round * self.threads_per_cta) + thread_idx
            narrowed[narrow_round] = Int32(self.invalid_expert_marker)
            if route < Int32(routes):
                raw_index = index_stage[route]
                if not self._is_unowned_expert(raw_index):
                    narrowed[narrow_round] = Int32(raw_index)

        cute.arch.sync_threads()

        for narrow_round in cutlass.range_constexpr(narrow_rounds):
            route = Int32(narrow_round * self.threads_per_cta) + thread_idx
            if route < Int32(routes):
                wire_indices[route] = narrowed[narrow_round]

    @cute.jit
    def _is_unowned_expert(self, raw_index) -> cutlass.Boolean:
        """Whether a caller-supplied index falls outside the global expert range."""
        if cutlass.const_expr(self.topk_index_dtype is cutlass.Int32):
            return cutlass.Uint32(raw_index) >= cutlass.Uint32(self.expert_count)
        return cutlass.Uint64(raw_index) >= cutlass.Uint64(self.expert_count)

    @cute.jit
    def _classify_tokens(
        self, topk_indices: cute.Tensor, global_warp_idx: Int32, lane_idx: Int32
    ) -> cute.Tensor:
        """One bit per (token, peer): does this warp owe that peer this token?

        Both payload passes need the same answer from the same input, so deriving
        it twice is not just wasted work, it is a silent invariant that the two
        derivations agree. Derived once, what the passes are left with is a bit
        test against a warp-uniform word, which keeps their store predicate a
        uniform branch just as the ballot it replaces did.

        A lane owns one top-k slot and contributes the bit of that slot's owning
        rank; the warp-wide OR of those contributions is the token's mask. That
        turns one ballot per peer into one reduction per mask word, and it is what
        lets the slot count exceed a warp: further rounds only OR more bits into
        the same accumulator.

        The loads all issue before any reduction. A reduction is warp-synchronous,
        so a fused loop would put one in front of every load but the first, and the
        loads would stop overlapping.
        """
        tokens_per_warp = self.pusher_tokens_per_warp
        mask_words = self.destination_mask_words
        # Holds each lane's own contributions until the second loop reduces them
        # in place, so the two phases share one set of registers.
        masks = cute.make_rmem_tensor((tokens_per_warp, mask_words), cutlass.Uint32)

        for token_round in cutlass.range_constexpr(tokens_per_warp):
            token = global_warp_idx + Int32(token_round * self.pusher_global_warp_count)
            for word in cutlass.range_constexpr(mask_words):
                masks[token_round, word] = cutlass.Uint32(0)

            for slot_round in cutlass.range_constexpr(self.topk_lane_rounds):
                slot = Int32(slot_round * 32) + lane_idx
                lane_owns_slot = token < Int32(self.max_tokens_per_rank)
                # Only a trailing partial round can run off the top-k width, and
                # only when the width is not a whole number of warps.
                if cutlass.const_expr((slot_round + 1) * 32 > self.topk):
                    lane_owns_slot = lane_owns_slot & (slot < Int32(self.topk))

                if lane_owns_slot:
                    raw_index = topk_indices[token, slot]
                    if not self._is_unowned_expert(raw_index):
                        lane_rank = cutlass.Uint32(raw_index) // cutlass.Uint32(
                            self.experts_per_rank
                        )
                        if cutlass.const_expr(mask_words == 1):
                            masks[token_round, 0] = masks[token_round, 0] | (
                                cutlass.Uint32(1) << lane_rank
                            )
                        else:
                            lane_word = lane_rank // cutlass.Uint32(32)
                            lane_bit = cutlass.Uint32(1) << (
                                lane_rank % cutlass.Uint32(32)
                            )
                            for word in cutlass.range_constexpr(mask_words):
                                if lane_word == cutlass.Uint32(word):
                                    masks[token_round, word] = (
                                        masks[token_round, word] | lane_bit
                                    )

        for token_round in cutlass.range_constexpr(tokens_per_warp):
            for word in cutlass.range_constexpr(mask_words):
                masks[token_round, word] = cutlass.Uint32(
                    cute.arch.warp_redux_sync(masks[token_round, word], "or")
                )
        return masks

    @cute.jit
    def _run_pusher(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        topk_indices: cute.Tensor,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        smem_base: cute.Pointer,
        pusher_idx: Int32,
        thread_idx: Int32,
    ) -> None:
        """Push this rank's tokens to every peer that routed to them (Sentinel dispatch).

        Both planes go out back to back through one staging pipeline; receivers
        detect each landing byte leaving the 0xFF sentinel, so no flag follows
        either plane.
        """
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane_idx = thread_idx % Int32(32)
        mbarriers = self._smem_workspace.ptr(self.pusher_mbarrier_region, smem_base)
        global_warp_idx = pusher_idx * Int32(self.pusher_warp_count) + warp_idx
        sf_stage = self._smem_workspace.tensor(self.pusher_sf_stage_region, smem_base)
        data_stage = self._smem_workspace.tensor(
            self.pusher_data_stage_region, smem_base
        )

        # Initialized once for both passes; the state below carries the phase
        # across both.
        if thread_idx < Int32(self.pusher_warp_count * self.pusher_pipeline_depth):
            cute.arch.mbarrier_init(mbarriers + thread_idx, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        iket.range_push("pusher.classify")
        destination_masks = self._classify_tokens(
            topk_indices, global_warp_idx, lane_idx
        )
        iket.range_pop()
        stage_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.pusher_pipeline_depth
        )

        iket.range_push("pusher.sf")
        stage_state = self._push_one_plane(
            peer_rank_ptr_mapper,
            local_rank,
            destination_masks,
            stage_state,
            activation_sf,
            self.plain_activation_sf_tensor(device_workspace),
            sf_stage,
            mbarriers,
            self.plain_activation_sf_row_elements
            * int(self.activation_sf_dtype.width)
            // 8,
            global_warp_idx,
            warp_idx,
            lane_idx,
        )
        iket.range_pop()

        iket.range_push("pusher.data")
        stage_state = self._push_one_plane(
            peer_rank_ptr_mapper,
            local_rank,
            destination_masks,
            stage_state,
            activation,
            self.plain_activation_tensor(device_workspace),
            data_stage,
            mbarriers,
            self.plain_activation_row_elements * int(self.activation_dtype.width) // 8,
            global_warp_idx,
            warp_idx,
            lane_idx,
        )
        iket.range_pop()
        if cutlass.const_expr(self.stamps):
            if lane_idx == Int32(0):
                self.stamp_latest(
                    device_workspace,
                    self.stamp_push_done,
                    pusher_idx + Int32(self.pusher_cta_begin),
                )
        # Draining here only keeps the staging slots valid until exit.
        cute.arch.cp_async_bulk_wait_group(0)
        if cutlass.const_expr(self.stamps):
            if lane_idx == Int32(0):
                self.stamp_latest(
                    device_workspace,
                    self.stamp_published,
                    pusher_idx + Int32(self.pusher_cta_begin),
                )

    @cute.jit
    def _push_one_plane(
        self,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        destination_masks: cute.Tensor,
        stage_state: pipeline.PipelineState,
        source_plane: cute.Tensor,
        destination_plane: cute.Tensor,
        stage: cute.Tensor,
        mbarriers: cute.Pointer,
        row_bytes: int,
        global_warp_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
    ) -> pipeline.PipelineState:
        """Send one warp's tokens of one payload plane to their destination ranks.

        A token nobody asked for is skipped outright rather than staged, which
        covers both a round past the token count and a token whose whole top-k is
        invalid. The staging slot and barrier phase come from ``stage_state``,
        which advances only on the rounds that used a slot, so they stay in step
        with the barriers whichever rounds were skipped. It is returned rather than
        advanced in place: the state is a value, so the caller's copy would not see
        an advance made here, and the other pass shares these barriers.

        Every lane sees the same mask, so the skip and the advance are both
        warp-uniform and the state cannot diverge within a warp.
        """
        depth = self.pusher_pipeline_depth

        for token_round in cutlass.range_constexpr(self.pusher_tokens_per_warp):
            token = global_warp_idx + Int32(token_round * self.pusher_global_warp_count)
            any_destination = destination_masks[token_round, 0]
            for word in cutlass.range_constexpr(1, self.destination_mask_words):
                any_destination = any_destination | destination_masks[token_round, word]

            if any_destination != cutlass.Uint32(0):
                slot_mbarrier = mbarriers + (
                    warp_idx * Int32(depth) + stage_state.index
                )
                slot_stage = stage[warp_idx, stage_state.index, None]

                iket.range_push("pusher.into_smem")
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        slot_mbarrier, Int32(row_bytes)
                    )
                    tma_load_1d(
                        slot_stage.iterator,
                        source_plane[token, None].iterator,
                        slot_mbarrier,
                        Int32(row_bytes),
                    )
                cute.arch.mbarrier_wait(slot_mbarrier, stage_state.phase)
                iket.range_pop()

                iket.range_push("pusher.out_of_smem")
                for destination_rank in cutlass.range_constexpr(self.world_size):
                    mask_word = destination_rank // 32
                    peer_bit = cutlass.Uint32(1 << (destination_rank % 32))
                    peer_offset = peer_rank_ptr_mapper.map(
                        Int64(0), Int32(destination_rank), Int64(0)
                    )
                    destination_row = destination_plane[local_rank, token, None]
                    if (
                        destination_masks[token_round, mask_word] & peer_bit
                    ) != cutlass.Uint32(0):
                        with cute.arch.elect_one():
                            cp_async_bulk_s2g(
                                cute.make_ptr(
                                    destination_plane.element_type,
                                    destination_row.iterator.toint() + peer_offset,
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                slot_stage.iterator,
                                Int32(row_bytes),
                            )
                cute.arch.cp_async_bulk_commit_group()
                stage_state.advance()
                iket.range_pop()
            cute.arch.sync_warp()
            cute.arch.cp_async_bulk_wait_group(depth - 1, read=True)
        return stage_state

    @cute.jit
    def _run_index_copy(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        local_rank: Int32,
        copier: Int32,
        thread_idx: Int32,
    ) -> None:
        """Copy one slice of the received route lists into the index lists (pull dispatch).

        The lists of every (source, chunk), laid end to end in that order, are cut
        into ``index_copy_cta_count`` equal slices, and ``copier`` takes its own.
        Each copying CTA reads every count row itself. The rows give each list's
        length, so the CTA knows its slice and sends the loads of its first
        batch of route and score words out at once, finding each position's
        list by binary search over the list bases. While they are in flight it
        runs the helper's fold and ranking and derives its tables: each expert's
        data-row base, where each (source, chunk, expert) segment starts among
        the lists (one scan over the rows), and the rank of the segment's first
        route among the expert's routes (sources in ring order beginning at this
        rank, each source's chunks in order). A list position in a segment is
        then that rank plus its offset in the segment, and its pool row is the
        expert's data-row base plus that rank, or under strided token slots that
        rank's row among the expert's balanced token tiles (``_route_pool_row``).
        Consecutive positions of a segment sit side by side in the slabs and
        land on consecutive rows of a token tile, so a warp's loads and stores
        coalesce.

        The rows and the list words announce their own arrival (see "Expert-id
        lists"); each list word is reset to the sentinel once read. The CTA adds one to
        ``metadata_ready`` once its slice is written; under ``pull_acquire_in_copy``
        it first acquires the ``topk_ready`` of sources ``copier``, ``copier +
        index_copy_cta_count``, ..., so ``metadata_ready`` complete covers every
        source's release.
        """
        threads = self.threads_per_cta
        lane_idx = thread_idx % Int32(32)
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        world = self.world_size
        segments = self.list_segment_count
        buckets = self.local_expert_bucket_count
        chunks = self.exchange_cta_count
        lists = world * chunks
        counts = self._smem_workspace.tensor(self.helper_chunk_counts_region, smem_base)
        list_base = self._smem_workspace.tensor(self.copy_list_base_region, smem_base)
        segment_base = self._smem_workspace.tensor(
            self.copy_segment_base_region, smem_base
        )
        row_base = self._smem_workspace.tensor(self.copy_row_base_region, smem_base)
        ring_totals = self._smem_workspace.tensor(
            self.copy_ring_totals_region, smem_base
        )
        data_base = self._smem_workspace.tensor(self.copy_data_base_region, smem_base)
        loads = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        warp_totals = self._smem_workspace.tensor(
            self.helper_warp_totals_region, smem_base
        )

        iket.range_push("index_copy.lists")
        self._poll_count_rows(device_workspace, counts, thread_idx)
        cute.arch.sync_threads()
        # One thread per list sums its row; then the first warp scans the lengths.
        for list_round in cutlass.range_constexpr(ceil_div(lists, threads)):
            source_list = Int32(list_round * threads) + thread_idx
            if source_list < Int32(lists):
                length = Int32(0)
                for vector in cutlass.range_constexpr(buckets // 4):
                    loaded = lds128_v4_b32(
                        counts.iterator
                        + source_list * Int32(buckets)
                        + Int32(vector * 4)
                    )
                    length = length + loaded[0] + loaded[1] + loaded[2] + loaded[3]
                list_base[source_list] = length
        cute.arch.sync_threads()
        if warp_idx == Int32(0):
            carried = Int32(0)
            for list_round in cutlass.range_constexpr(ceil_div(lists, 32)):
                source_list = Int32(list_round * 32) + lane_idx
                length = cutlass.select_(
                    source_list < Int32(lists),
                    list_base[cutlass.min(source_list, Int32(lists - 1))],
                    Int32(0),
                )
                inclusive = carried + _warp_inclusive_sum(length, lane_idx)
                if source_list < Int32(lists):
                    list_base[source_list] = inclusive - length
                carried = Int32(cute.arch.shuffle_sync(inclusive, Int32(31)))
            if lane_idx == Int32(0):
                list_base[lists] = carried
        cute.arch.sync_threads()
        total = list_base[lists]
        copiers = Int32(self.index_copy_cta_count)
        slice_end = (total * (copier + Int32(1))) // copiers
        batch_begin = (total * copier) // copiers

        index_address = device_workspace.ptr(self.topk_index_inbox_region).toint()
        score_inbox_address = Int64(0)
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_inbox_address = device_workspace.ptr(
                self.topk_score_inbox_region
            ).toint()
        batch = self.index_copy_batch
        found = cute.make_rmem_tensor((batch,), cutlass.Int32)
        slab_offsets = cute.make_rmem_tensor((batch,), Int64)
        routes = cute.make_rmem_tensor((batch,), Uint32)
        scores = cute.make_rmem_tensor((batch,), Uint32)
        pool_rows = cute.make_rmem_tensor((batch,), cutlass.Int32)
        positions = cute.make_rmem_tensor((batch,), cutlass.Int32)
        firsts = cute.make_rmem_tensor((batch,), cutlass.Int32)
        if batch_begin < slice_end:
            self._load_list_words(
                list_base,
                index_address,
                score_inbox_address,
                batch_begin,
                slice_end,
                found,
                slab_offsets,
                routes,
                scores,
                thread_idx,
            )
        iket.range_pop()

        iket.range_push("index_copy.tables")
        self._plan_from_count_rows(
            device_workspace,
            smem_base,
            False,
            thread_idx,
            lane_idx,
            warp_idx,
            poll=False,
        )
        if thread_idx == Int32(0):
            self.stamp_plan_step(device_workspace, "c_planned", copier)
        # The data-row bases: the helper's padded scan in processing order, over
        # the same loads in the same order.
        expert_sizes_in_order = self._smem_workspace.tensor(
            self.helper_expert_sizes_in_order_region, smem_base
        )
        expert_order = self._smem_workspace.tensor(
            self.helper_expert_order_region, smem_base
        )
        prefix = self._smem_workspace.tensor(self.helper_prefix_region, smem_base)
        self._scan_padded_rows(
            expert_sizes_in_order,
            self._smem_workspace.tensor(self.helper_scan_input_region, smem_base),
            prefix,
            warp_totals,
            self.token_padding_block,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        for bucket_round in cutlass.range_constexpr(ceil_div(buckets, threads)):
            position = Int32(bucket_round * threads) + thread_idx
            if position < Int32(buckets):
                data_base[expert_order[position]] = prefix[position]
        if cutlass.const_expr(self.token_tile_strided):
            tile_layout = self._smem_workspace.tensor(
                self.copy_tile_rows_region, smem_base
            )
            for expert_round in cutlass.range_constexpr(
                ceil_div(self.experts_per_rank, threads)
            ):
                local_expert = Int32(expert_round * threads) + thread_idx
                if local_expert < Int32(self.experts_per_rank):
                    tile_rows = balanced_tile_rows(
                        loads[local_expert], self.token_tile_n
                    )
                    tile_layout[0, local_expert] = tile_rows
                    tile_layout[1, local_expert] = tile_rows_reciprocal(tile_rows)
        _flat_exclusive_scan(
            counts,
            segment_base,
            warp_totals,
            segments,
            threads,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        # One thread per (ring position, expert): first the source's routes over
        # its chunks, then the sources ahead of it and its own chunks in turn.
        pairs = world * buckets
        pair_rounds = ceil_div(pairs, threads)
        for pair_round in cutlass.range_constexpr(pair_rounds):
            pair = Int32(pair_round * threads) + thread_idx
            if pair < Int32(pairs):
                ring_position = pair // Int32(buckets)
                bucket = pair - ring_position * Int32(buckets)
                source_rank = (local_rank + ring_position) % Int32(world)
                source_total = Int32(0)
                for source_chunk in cutlass.range_constexpr(chunks):
                    source_total = (
                        source_total
                        + counts[
                            (source_rank * Int32(chunks) + Int32(source_chunk))
                            * Int32(buckets)
                            + bucket
                        ]
                    )
                ring_totals[ring_position, bucket] = source_total
        cute.arch.sync_threads()
        for pair_round in cutlass.range_constexpr(pair_rounds):
            pair = Int32(pair_round * threads) + thread_idx
            if pair < Int32(pairs):
                ring_position = pair // Int32(buckets)
                bucket = pair - ring_position * Int32(buckets)
                source_rank = (local_rank + ring_position) % Int32(world)
                rank_begin = Int32(0)
                for earlier in cutlass.range_constexpr(world - 1):
                    if Int32(earlier) < ring_position:
                        rank_begin = rank_begin + ring_totals[Int32(earlier), bucket]
                for source_chunk in cutlass.range_constexpr(chunks):
                    segment = (
                        source_rank * Int32(chunks) + Int32(source_chunk)
                    ) * Int32(buckets) + bucket
                    row_base[segment] = rank_begin
                    rank_begin = rank_begin + counts[segment]
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            self.stamp_plan_step(device_workspace, "c_staged", copier)
        iket.range_pop()

        iket.range_push("index_copy.copy")
        gather_address = device_workspace.tensor(
            self.gather_index_region
        ).iterator.toint()
        metadata_address = device_workspace.tensor(
            self.token_src_metadata_region
        ).iterator.toint()
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_address = self.fc1_topk_scores_tensor(
                device_workspace
            ).iterator.toint()
        sentinel = Uint32(sentinel_ptx.WORD_SENTINEL)
        while batch_begin < slice_end:
            # The words announce their own arrival; one still holding the sentinel
            # is polled until it lands. A position past the slice may already be
            # reset by its owner, so it is never polled.
            for entry in cutlass.range_constexpr(batch):
                if batch_begin + Int32(entry * threads) + thread_idx < slice_end:
                    while routes[entry] == sentinel:
                        sentinel_ptx.nanosleep(Int32(32))
                        routes[entry] = sentinel_ptx.ld_volatile_u32(
                            index_address + slab_offsets[entry]
                        )
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        while scores[entry] == sentinel:
                            sentinel_ptx.nanosleep(Int32(32))
                            scores[entry] = sentinel_ptx.ld_volatile_u32(
                                score_inbox_address + slab_offsets[entry]
                            )
            if thread_idx == Int32(0):
                self.stamp_plan_step(device_workspace, "c_loaded", copier)
            # Every row before the first store, for the same reason as the loads.
            for entry in cutlass.range_constexpr(batch):
                positions[entry] = cutlass.min(
                    batch_begin + Int32(entry * threads) + thread_idx,
                    slice_end - Int32(1),
                )
                firsts[entry] = found[entry] * Int32(buckets)
            self._count_bases_at_most_batch(
                segment_base, firsts, buckets, positions, pool_rows
            )
            for entry in cutlass.range_constexpr(batch):
                segment = firsts[entry] + pool_rows[entry] - Int32(1)
                expert = segment - firsts[entry]
                pool_rows[entry] = self._route_pool_row(
                    smem_base,
                    expert,
                    row_base[segment] + positions[entry] - segment_base[segment],
                )
            for entry in cutlass.range_constexpr(batch):
                live = cutlass.select_(
                    batch_begin + Int32(entry * threads) + thread_idx < slice_end,
                    Int32(1),
                    Int32(0),
                )
                source = found[entry] // Int32(chunks)
                pool_row = pool_rows[entry]
                route = Int32(routes[entry])
                source_token = route // Int32(self.topk)
                stg_b32(
                    gather_address + Int64(pool_row) * Int64(4),
                    source * Int32(self.max_tokens_per_rank) + source_token,
                    live,
                )
                stg_b64(
                    metadata_address + Int64(pool_row) * Int64(8),
                    TokenSrcMetadata(
                        src_rank=source,
                        src_token=source_token,
                        src_topk=route - source_token * Int32(self.topk),
                    ).pack(),
                    live,
                )
                # Reset for the round that reuses this ping-pong buffer; see
                # "Expert-id lists".
                stg_b32(index_address + slab_offsets[entry], Int32(-1), live)
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    # The score travels as bits, so it lands exactly as it was sent.
                    stg_b32(
                        score_address + Int64(pool_row) * Int64(4),
                        Int32(scores[entry]),
                        live,
                    )
                    stg_b32(score_inbox_address + slab_offsets[entry], Int32(-1), live)
            batch_begin = batch_begin + Int32(batch * threads)
            if batch_begin < slice_end:
                self._load_list_words(
                    list_base,
                    index_address,
                    score_inbox_address,
                    batch_begin,
                    slice_end,
                    found,
                    slab_offsets,
                    routes,
                    scores,
                    thread_idx,
                )

        if cutlass.const_expr(self.pull_acquire_in_copy):
            # Thread k acquires source copier + k * copiers, so the copiers between
            # them acquire every source before metadata_ready can complete.
            for source_round in cutlass.range_constexpr(
                ceil_div(world, self.index_copy_cta_count)
            ):
                source_rank = copier + Int32(source_round * self.index_copy_cta_count)
                if (thread_idx == Int32(source_round)) & (source_rank < Int32(world)):
                    self._acquire_source_ready(device_workspace, source_rank)
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.metadata_ready_region), Int32(1)
            )
            self.stamp(device_workspace, self.stamp_sort_done, copier)
        iket.range_pop()

    @cute.jit
    def _route_pool_row(
        self, smem_base: cute.Pointer, expert: Int32, route_rank: Int32
    ) -> Int32:
        """Data-pool row of an expert's ``route_rank``-th route.

        The one place the index copy maps a route's rank inside its expert to a
        row: the expert's data-row base plus the rank under dense token slots.
        Under strided ones the expert's routes spread evenly over its token
        tiles, and ``slot_physical_row`` places the rank in its tile's slot from
        the tile rows and reciprocal staged in ``copy_tile_rows``.
        """
        expert_base = self._smem_workspace.tensor(
            self.copy_data_base_region, smem_base
        )[expert]
        pool_row = expert_base + route_rank
        if cutlass.const_expr(self.token_tile_strided):
            tile_layout = self._smem_workspace.tensor(
                self.copy_tile_rows_region, smem_base
            )
            pool_row = expert_base + slot_physical_row(
                route_rank,
                tile_layout[0, expert],
                tile_layout[1, expert],
                self.token_slot_rows,
            )
        return pool_row

    @cute.jit
    def _load_list_words(
        self,
        list_base: cute.Tensor,
        index_address: Int64,
        score_inbox_address: Int64,
        batch_begin: Int32,
        slice_end: Int32,
        found: cute.Tensor,
        slab_offsets: cute.Tensor,
        routes: cute.Tensor,
        scores: cute.Tensor,
        thread_idx: Int32,
    ) -> None:
        """Send out the route and score loads of one batch of a copy slice, before any is used.

        Positions past the slice stand in for the slice's last one and load
        nothing; their words are never polled or stored. Leaves each position's
        (source, chunk) list in ``found`` and its byte offset in the slabs in
        ``slab_offsets``.
        """
        lists = self.world_size * self.exchange_cta_count
        batch = self.index_copy_batch
        # Every search before the first load: the loads are volatile, and nothing
        # moves across them.
        positions = cute.make_rmem_tensor((batch,), cutlass.Int32)
        firsts = cute.make_rmem_tensor((batch,), cutlass.Int32)
        for entry in cutlass.range_constexpr(batch):
            positions[entry] = cutlass.min(
                batch_begin + Int32(entry * self.threads_per_cta) + thread_idx,
                slice_end - Int32(1),
            )
            firsts[entry] = Int32(0)
        self._count_bases_at_most_batch(list_base, firsts, lists, positions, found)
        for entry in cutlass.range_constexpr(batch):
            source_list = found[entry] - Int32(1)
            found[entry] = source_list
            slab_offsets[entry] = Int64(
                source_list * Int32(self.exchange_chunk_routes)
                + positions[entry]
                - list_base[source_list]
            ) * Int64(4)
        for entry in cutlass.range_constexpr(self.index_copy_batch):
            live = cutlass.select_(
                batch_begin + Int32(entry * self.threads_per_cta) + thread_idx
                < slice_end,
                Int32(1),
                Int32(0),
            )
            routes[entry] = Uint32(
                ldg_volatile_b32_if(index_address + slab_offsets[entry], live)
            )
            scores[entry] = Uint32(0)
            if cutlass.const_expr(self.apply_topk_at_fc1):
                scores[entry] = Uint32(
                    ldg_volatile_b32_if(score_inbox_address + slab_offsets[entry], live)
                )

    @cute.jit
    def _count_bases_at_most_batch(
        self,
        bases: cute.Tensor,
        firsts: cute.Tensor,
        count: int,
        positions: cute.Tensor,
        below_counts: cute.Tensor,
    ) -> None:
        """How many of the ``count`` nondecreasing ``bases`` from ``firsts[i]`` on are at most ``positions[i]``.

        A branch-free binary search per batch entry. The last base at most a
        position names the list or segment holding it: an empty one shares its
        base with the next and is skipped. The searches are interleaved step by
        step: every step issues all of the batch's probes before it uses any, so
        the batch pays one SMEM latency per step rather than one per probe.
        """
        batch = self.index_copy_batch
        for entry in cutlass.range_constexpr(batch):
            below_counts[entry] = Int32(0)
        step = 1
        while step * 2 <= count:
            step *= 2
        while step >= 1:
            for entry in cutlass.range_constexpr(batch):
                probe = below_counts[entry] + Int32(step)
                below = (probe <= Int32(count)) & (
                    bases[firsts[entry] + cutlass.min(probe, Int32(count)) - Int32(1)]
                    <= positions[entry]
                )
                below_counts[entry] = cutlass.select_(below, probe, below_counts[entry])
            step //= 2

    @cute.jit
    def _run_helper(
        self,
        device_workspace: DeviceWorkspace,
        local_rank: Int32,
        smem_base: cute.Pointer,
        thread_idx: Int32,
    ) -> None:
        """Fold the count rows into per-expert sizes and both row-space prefixes, and publish them.

        Every source sends one count row per exchange CTA into this rank's count
        inbox; the rows announce their own arrival (see "Expert-id lists"). Each
        index-copy CTA runs the same fold and ranking on the same rows itself
        (``_plan_from_count_rows``), so nothing here gates the index lists.

        ``expert_sizes`` and its flag go out before the prefixes: the main kernel's
        scheduler only needs the sizes to lay out work and start streaming weights,
        so it should not wait for tables that only the refine and the walk read.

        The processing order is settled here as well, between the fold and that
        flag, because it is the earliest point at which every expert's load is
        known and the latest at which the sizes in processing order, which the
        flag releases, can still be written. Everything the phase produces is
        covered by the flag's release, and the only reader of the order -- the
        main kernel's scheduler warp -- is also the only waiter on the flag, so
        the order needs no counter of its own.
        """
        lane_idx = thread_idx % Int32(32)
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        scan_input = self._smem_workspace.tensor(
            self.helper_scan_input_region, smem_base
        )
        prefix = self._smem_workspace.tensor(self.helper_prefix_region, smem_base)
        warp_totals = self._smem_workspace.tensor(
            self.helper_warp_totals_region, smem_base
        )

        iket.range_push("helper.fold")
        self._plan_from_count_rows(
            device_workspace, smem_base, True, thread_idx, lane_idx, warp_idx
        )
        if thread_idx == Int32(0):
            self.stamp_plan_step(device_workspace, "h_planned", Int32(0))

        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        expert_sizes = device_workspace.tensor(self.expert_sizes_region)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.experts_per_rank):
                expert_sizes[bucket] = totals[bucket]
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.sizes_ready_region), Int32(1)
            )
            self.stamp(device_workspace, self.stamp_helper, Int32(0))
        iket.range_pop()

        iket.range_push("helper.prefix")
        # Every prefix below walks the order the pool is physically laid out in,
        # which is the processing order rather than the expert order. The bases
        # themselves stay keyed by expert, because every reader of them -- the
        # refine expert table and the main kernel's gather -- addresses an
        # expert rather than a position.
        expert_sizes_in_order = self._smem_workspace.tensor(
            self.helper_expert_sizes_in_order_region, smem_base
        )
        expert_order = self._smem_workspace.tensor(
            self.helper_expert_order_region, smem_base
        )

        for padding_block, region_name in (
            (self.token_padding_block, self.data_expert_base_region),
            (self.sf_padding_block, self.sf_expert_base_region),
        ):
            self._publish_padded_prefix(
                device_workspace,
                expert_sizes_in_order,
                expert_order,
                scan_input,
                prefix,
                warp_totals,
                padding_block,
                region_name,
                thread_idx,
                lane_idx,
                warp_idx,
            )
        self._publish_refine_block_prefix(
            device_workspace,
            expert_sizes_in_order,
            expert_order,
            scan_input,
            prefix,
            warp_totals,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.prefix_ready_region), Int32(1)
            )
            self.stamp(device_workspace, self.stamp_helper, Int32(1))
        iket.range_pop()

    @cute.jit
    def _plan_from_count_rows(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        publish: bool,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
        poll: bool = True,
    ) -> None:
        """Bring every count row into SMEM, fold it into per-expert loads and rank them (pull dispatch).

        The helper (``publish``) and every index-copy CTA run this on the same
        rows. Ties resolve the same way in all of them
        (``_rank_experts_by_load``), so all of them settle on the processing
        order the helper publishes. Leaves the rows in ``helper_chunk_counts``,
        the loads in ``helper_totals`` and the order in ``helper_expert_order``
        and ``helper_expert_sizes_in_order``. Without ``poll`` the rows are
        already in SMEM.
        """
        chunk_counts = self._smem_workspace.tensor(
            self.helper_chunk_counts_region, smem_base
        )
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        if cutlass.const_expr(poll):
            self._poll_count_rows(device_workspace, chunk_counts, thread_idx)
            cute.arch.sync_threads()

        # One warp per column of four buckets: each lane sums its rows' 16-byte
        # vectors, then the warp reduces them by shuffles.
        buckets = self.local_expert_bucket_count
        rows = self.world_size * self.exchange_cta_count
        columns = buckets // 4
        warps = self.threads_per_cta // 32
        for column_round in cutlass.range_constexpr(ceil_div(columns, warps)):
            column = Int32(column_round * warps) + warp_idx
            if column < Int32(columns):
                sums = cute.make_rmem_tensor((4,), cutlass.Int32)
                for word in cutlass.range_constexpr(4):
                    sums[word] = Int32(0)
                for row_round in cutlass.range_constexpr(ceil_div(rows, 32)):
                    row = Int32(row_round * 32) + lane_idx
                    if row < Int32(rows):
                        loaded = lds128_v4_b32(
                            chunk_counts.iterator
                            + row * Int32(buckets)
                            + column * Int32(4)
                        )
                        for word in cutlass.range_constexpr(4):
                            sums[word] = sums[word] + loaded[word]
                for word in cutlass.range_constexpr(4):
                    for step_log in cutlass.range_constexpr(5):
                        sums[word] = sums[word] + Int32(
                            cute.arch.shuffle_sync_bfly(
                                sums[word], offset=1 << step_log
                            )
                        )
                if lane_idx == Int32(0):
                    for word in cutlass.range_constexpr(4):
                        totals[column * Int32(4) + Int32(word)] = sums[word]
        cute.arch.sync_threads()
        self._rank_experts_by_load(
            device_workspace,
            smem_base,
            thread_idx,
            lane_idx,
            warp_idx,
            not publish,
            ordered_ties=True,
        )

    @cute.jit
    def _rank_experts_by_load(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
        replicated: bool = False,
        ordered_ties: bool = False,
    ) -> None:
        """Settle the order the main kernel processes this rank's experts in.

        Descending routed load, because the FC2 tiles of a token block cannot
        start until every FC1 tile of that block has finished. Leaving the biggest
        expert for last therefore ends the kernel on a long FC1 stretch followed
        by an FC2 stretch with nothing left to overlap it; taking it first moves
        that pair into the pipeline fill, and the tail becomes a run of small
        experts whose FC1-to-FC2 waits are short and cover each other.

        Three implementations: the pull dispatch ranks up to 64 experts with the
        whole CTA, the Sentinel planners up to 32 in one warp, and either
        dispatch more by load histogram. All write the same four outputs through
        ``_place_expert_at_position``, so nothing downstream can tell which one
        ran.

        ``replicated`` is for CTAs that each rank the same loads (the Sentinel
        planners, the pull index copy): only the two SMEM outputs are written,
        and every tie resolves by a fixed key (the expert index, or the
        alternating halves of ``_rank_by_counting_in_cta``), so all of them
        settle on the same order. ``ordered_ties`` resolves ties by expert index
        while publishing, for the CTA whose order replicated CTAs must
        reproduce.
        """
        if cutlass.const_expr(self.ranks_experts_in_cta):
            self._rank_by_counting_in_cta(
                device_workspace, smem_base, thread_idx, not replicated
            )
        elif cutlass.const_expr(self.ranks_experts_in_warp):
            self._rank_by_counting_in_warp(
                device_workspace, smem_base, lane_idx, warp_idx, not replicated
            )
        else:
            self._rank_by_load_histogram(
                device_workspace,
                smem_base,
                thread_idx,
                lane_idx,
                warp_idx,
                not replicated,
                replicated or ordered_ties,
            )

    @cute.jit
    def _place_expert_at_position(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        position: Int32,
        expert: Int32,
        routed_count: Int32,
        publish: bool = True,
    ) -> None:
        """Record one expert's position in the processing order.

        Four destinations, all built from values the caller already holds, which
        is why neither ranking implementation ever reads a load back through the
        permutation it just produced.

        A position below ``experts_per_rank`` can hold a padded bucket, and that
        is harmless rather than merely unlikely: descending order puts every
        expert holding at least one route ahead of every empty one, so such a
        position always carries a zero size, the scheduler emits no tile for it,
        and the main kernel never translates it.
        """
        self._smem_workspace.tensor(self.helper_expert_order_region, smem_base)[
            position
        ] = expert
        self._smem_workspace.tensor(
            self.helper_expert_sizes_in_order_region, smem_base
        )[position] = routed_count
        if cutlass.const_expr(publish):
            device_workspace.tensor(self.expert_order_region)[position] = expert
            if position < Int32(self.experts_per_rank):
                device_workspace.tensor(self.expert_sizes_in_order_region)[position] = (
                    routed_count
                )

    @cute.jit
    def _rank_by_counting_in_warp(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        lane_idx: Int32,
        warp_idx: Int32,
        publish: bool = True,
    ) -> None:
        """Rank the experts inside one warp by counting how many outrank each (Sentinel planners).

        Lane ``e`` carries expert ``e``'s load and each round broadcasts one
        lane's load to all of them, so a lane accumulates its own position. The
        rounds share no state and pipeline; a selection loop over the same values
        would instead serialize on masking out each round's winner, which costs a
        full cross-lane latency per round.

        The comparison is on the exact load rather than on its binade because
        exactness is free here -- the round count is the expert count either way
        -- so this branch resolves orderings the histogram branch leaves tied.
        """
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        if warp_idx == Int32(0):
            own_load = Int32(0)
            if lane_idx < Int32(self.local_expert_bucket_count):
                own_load = totals[lane_idx]
            position = Int32(0)
            for other in cutlass.range_constexpr(self.local_expert_bucket_count):
                other_load = Int32(cute.arch.shuffle_sync(own_load, Int32(other)))
                outranks = (other_load > own_load) | (
                    (other_load == own_load) & (Int32(other) < lane_idx)
                )
                if outranks:
                    position = position + Int32(1)
            if lane_idx < Int32(self.local_expert_bucket_count):
                self._place_expert_at_position(
                    device_workspace, smem_base, position, lane_idx, own_load, publish
                )
        cute.arch.sync_threads()

    @cute.jit
    def _rank_by_counting_in_cta(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
        publish: bool = True,
    ) -> None:
        """Rank up to 64 experts with every thread of the CTA, counting how many outrank each (pull dispatch).

        Thread ``t`` counts, for expert ``t % 64``, the experts of share ``t //
        64`` of the buckets that outrank it, and the shares' counts meet in SMEM:
        a few independent SMEM loads per thread and two barriers, where one warp
        would take one dependent shuffle round per expert. The comparison is on
        the exact load: a higher load outranks. Equal loads keep expert order up
        to 32 buckets. Above, they
        alternate between the two halves of the local experts (0, half, 1, half
        + 1, ...), so experts of equal load processed back to back come from
        opposite halves; that measured faster than plain index order with every
        expert equally loaded.
        """
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        rank_shares = self._smem_workspace.tensor(
            self.helper_rank_shares_region, smem_base
        )
        buckets = self.local_expert_bucket_count
        experts = self.cta_rank_experts
        shares = self.threads_per_cta // experts
        per_share = ceil_div(buckets, shares)
        half = buckets // 2
        index_ties = self.ranks_experts_in_warp

        expert = thread_idx % Int32(experts)
        share = thread_idx // Int32(experts)
        if expert < Int32(buckets):
            own_load = totals[expert]
            own_key = expert
            if cutlass.const_expr(not index_ties):
                own_key = cutlass.select_(
                    expert < Int32(half),
                    expert * Int32(2),
                    (expert - Int32(half)) * Int32(2) + Int32(1),
                )
            outranked = Int32(0)
            for step in cutlass.range_constexpr(per_share):
                other = share * Int32(per_share) + Int32(step)
                other_load = totals[cutlass.min(other, Int32(buckets - 1))]
                other_key = other
                if cutlass.const_expr(not index_ties):
                    other_key = cutlass.select_(
                        other < Int32(half),
                        other * Int32(2),
                        (other - Int32(half)) * Int32(2) + Int32(1),
                    )
                outranks = (other_load > own_load) | (
                    (other_load == own_load) & (other_key < own_key)
                )
                outranked = outranked + cutlass.select_(
                    (other < Int32(buckets)) & outranks, Int32(1), Int32(0)
                )
            rank_shares[share, expert] = outranked
        cute.arch.sync_threads()
        if thread_idx < Int32(buckets):
            position = Int32(0)
            for counted in cutlass.range_constexpr(shares):
                position = position + rank_shares[Int32(counted), thread_idx]
            self._place_expert_at_position(
                device_workspace,
                smem_base,
                position,
                thread_idx,
                totals[thread_idx],
                publish,
            )
        cute.arch.sync_threads()

    @cute.jit
    def _rank_by_load_histogram(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
        publish: bool = True,
        ordered_ties: bool = False,
    ) -> None:
        """Bucket-sort the experts by descending load binade, at one thread per expert.

        Used once the expert count outgrows a warp, where counting how many
        experts outrank each would be quadratic. Cost here is instead linear in
        the expert count plus the bin count, so it barely moves as the expert
        count grows.

        A single pass, because the atomic does double duty: what it returns is
        this expert's index inside its own bin and what it leaves behind is that
        bin's population. The pull exchange's count pass carries the same trick;
        only the bucket definition differs. The atomic orders a bin's experts by arrival;
        ``ordered_ties`` orders them by expert index instead
        (``_bin_experts_in_index_order``).
        """
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        load_bins = self._smem_workspace.tensor(self.helper_load_bin_region, smem_base)

        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        keys = cute.make_rmem_tensor((bucket_rounds,), cutlass.Int32)
        offsets_in_bin = cute.make_rmem_tensor((bucket_rounds,), cutlass.Int32)

        if warp_idx == Int32(0):
            load_bins[lane_idx] = Int32(0)
        cute.arch.sync_threads()

        # Key and within-bin index stay in registers across the scan rather than
        # being recomputed after it. Rounds past the bucket count still get a
        # defined key so that no register is read on a path that never wrote it.
        if cutlass.const_expr(ordered_ties):
            self._bin_experts_in_index_order(
                smem_base, keys, offsets_in_bin, thread_idx, lane_idx, warp_idx
            )
        else:
            for bucket_round in cutlass.range_constexpr(bucket_rounds):
                bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
                keys[bucket_round] = Int32(0)
                offsets_in_bin[bucket_round] = Int32(0)
                if bucket < Int32(self.local_expert_bucket_count):
                    # No histogram spans the exact counts, so the bin is the binade;
                    # inverting it is what makes the prefix below descend.
                    binade = Int32(32) - Int32(cute.arch.clz(totals[bucket]))
                    keys[bucket_round] = Int32(self.load_bin_count - 1) - binade
                    offsets_in_bin[bucket_round] = Int32(
                        cute.arch.atomic_add(
                            load_bins.iterator + keys[bucket_round],
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                    )
            cute.arch.sync_threads()

        # The bin count is exactly a warp, so the populations become the bins'
        # first positions in five shuffles instead of through the CTA-wide
        # shared-memory scan: no warp-total round trip and no barrier inside. In
        # place, because every lane holds its own population in a register before
        # any lane stores.
        if warp_idx == Int32(0):
            population = load_bins[lane_idx]
            load_bins[lane_idx] = _warp_inclusive_sum(population, lane_idx) - population
        cute.arch.sync_threads()

        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                position = load_bins[keys[bucket_round]] + offsets_in_bin[bucket_round]
                self._place_expert_at_position(
                    device_workspace,
                    smem_base,
                    position,
                    bucket,
                    totals[bucket],
                    publish,
                )
        cute.arch.sync_threads()

    @cute.jit
    def _bin_experts_in_index_order(
        self,
        smem_base: cute.Pointer,
        keys: cute.Tensor,
        offsets_in_bin: cute.Tensor,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
    ) -> None:
        """Each expert's load bin and its index among that bin's experts in expert order.

        The same bins and populations as the atomic of ``_rank_by_load_histogram``
        leaves, but an expert's index inside its bin counts only the experts
        before it, so CTAs that bin the same loads agree on it. A warp finds its
        lanes' bin peers with one ballot per bin; the warps before it, in this
        round and earlier ones, come from per-warp bin counts. Leaves each bin's
        population in ``helper_load_bin``.
        """
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        load_bins = self._smem_workspace.tensor(self.helper_load_bin_region, smem_base)
        warp_bins = self._smem_workspace.tensor(self.helper_warp_bins_region, smem_base)
        warps = self.threads_per_cta // 32
        lanes_below = Int32((Uint32(1) << Uint32(lane_idx)) - Uint32(1))
        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            # Past the bucket count the key names no bin.
            key = Int32(self.load_bin_count)
            if bucket < Int32(self.local_expert_bucket_count):
                binade = Int32(32) - Int32(cute.arch.clz(totals[bucket]))
                key = Int32(self.load_bin_count - 1) - binade
            keys[bucket_round] = key
            same_bin_lanes = Int32(0)
            lane_bin_population = Int32(0)
            for bin_idx in cutlass.range_constexpr(self.load_bin_count):
                bin_lanes = Int32(cute.arch.vote_ballot_sync(key == Int32(bin_idx)))
                if key == Int32(bin_idx):
                    same_bin_lanes = bin_lanes
                if lane_idx == Int32(bin_idx):
                    lane_bin_population = Int32(cute.arch.popc(bin_lanes))
            warp_bins[warp_idx, lane_idx] = lane_bin_population
            cute.arch.sync_threads()

            offsets_in_bin[bucket_round] = Int32(0)
            if bucket < Int32(self.local_expert_bucket_count):
                earlier = load_bins[key]
                for warp in cutlass.range_constexpr(warps):
                    if Int32(warp) < warp_idx:
                        earlier = earlier + warp_bins[Int32(warp), key]
                offsets_in_bin[bucket_round] = earlier + Int32(
                    cute.arch.popc(same_bin_lanes & lanes_below)
                )
            cute.arch.sync_threads()
            if warp_idx == Int32(0):
                round_population = Int32(0)
                for warp in cutlass.range_constexpr(warps):
                    round_population = (
                        round_population + warp_bins[Int32(warp), lane_idx]
                    )
                load_bins[lane_idx] = load_bins[lane_idx] + round_population
            cute.arch.sync_threads()

    @cute.jit
    def _scan_padded_rows(
        self,
        expert_sizes_in_order: cute.Tensor,
        scan_input: cute.Tensor,
        prefix: cute.Tensor,
        warp_totals: cute.Tensor,
        padding_block: int,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
    ) -> None:
        """Exclusive prefix of the padded row counts into ``prefix``, in processing order."""
        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            scan_entry = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if scan_entry < Int32(self.local_expert_bucket_count):
                scan_input[scan_entry] = expert_pool_rows(
                    expert_sizes_in_order[scan_entry],
                    padding_block,
                    self.token_tile_n,
                    self.token_slot_rows,
                )
        cute.arch.sync_threads()

        smem_exclusive_prefix(
            scan_input,
            prefix,
            warp_totals,
            self.threads_per_cta,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        cute.arch.sync_threads()

    @cute.jit
    def _publish_padded_prefix(
        self,
        device_workspace: DeviceWorkspace,
        expert_sizes_in_order: cute.Tensor,
        expert_order: cute.Tensor,
        scan_input: cute.Tensor,
        prefix: cute.Tensor,
        warp_totals: cute.Tensor,
        padding_block: int,
        region_name: str,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
    ) -> None:
        """Exclusive prefix of per-expert padded row counts, into one row space.

        The loads are read twice, once per row space, so the padded values go to a
        separate scan input instead of overwriting them.

        The padding call is the shared one the scheduler uses, because the main
        kernel rebuilds these same bases from the sizes it is handed and the two
        have to agree exactly.

        ``expert_order`` is the processing order, so the scan accumulates along
        its positions and only the store side translates back to an expert.
        """
        self._scan_padded_rows(
            expert_sizes_in_order,
            scan_input,
            prefix,
            warp_totals,
            padding_block,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        destination = device_workspace.tensor(region_name)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            scan_entry = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if scan_entry < Int32(self.local_expert_bucket_count):
                destination[expert_order[scan_entry]] = prefix[scan_entry]
        cute.arch.sync_threads()

    @cute.jit
    def _publish_refine_block_prefix(
        self,
        device_workspace: DeviceWorkspace,
        expert_sizes_in_order: cute.Tensor,
        expert_order: cute.Tensor,
        scan_input: cute.Tensor,
        prefix: cute.Tensor,
        warp_totals: cute.Tensor,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
    ) -> None:
        """Per-expert refine block counts and their prefix.

        A block is one SF padding block of scale-factor rows, the row count of one
        canonical atom. Only blocks holding at least one live route are enumerated:
        padding rows carry scale factors the MMA's dynamic instruction extent never
        reads.

        This count is also what the main kernel waits for on ``fc1_sf_ready``, so
        the divisor here and the one it compares against are the same field.

        Scanning in the same order as the row-space bases is not just for
        symmetry: it makes the linear block index the refine workers claim run
        through the experts in the order the main kernel processes them, so the
        scale factors of the expert scheduled first are also refined first.
        """
        # The caller has just published the SF-row prefix in this same processing
        # order. Each padded row count is a multiple of sf_padding_block, so
        # converting that prefix to blocks replaces another CTA-wide scan.
        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        destination = device_workspace.tensor(self.refine_block_base_region)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            scan_entry = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if scan_entry < Int32(self.local_expert_bucket_count):
                destination[expert_order[scan_entry]] = prefix[scan_entry] // Int32(
                    self.sf_padding_block
                )
        if thread_idx == Int32(0):
            last = Int32(self.local_expert_bucket_count - 1)
            total_rows = prefix[last] + expert_pool_rows(
                expert_sizes_in_order[last],
                Int32(self.sf_padding_block),
                self.token_tile_n,
                self.token_slot_rows,
            )
            device_workspace.tensor(self.refine_block_total_region)[0] = (
                total_rows // Int32(self.sf_padding_block)
            )
        cute.arch.sync_threads()

    @cute.jit
    def _poll_count_rows(
        self,
        device_workspace: DeviceWorkspace,
        chunk_counts: cute.Tensor,
        thread_idx: Int32,
    ) -> None:
        """Wait for every (source, chunk) count row and bring it into SMEM.

        Every word is a count, never the 0xFFFFFFFF sentinel the rows hold until
        it lands, so a word has arrived exactly when it differs from the
        sentinel. A thread polls whole 16-byte vectors with relaxed system loads
        and reloads only the vectors that still hold a sentinel word. The helper
        resets the rows once every reader is done with them
        (``_reset_count_rows``).
        """
        vectors = self.list_segment_count // 4
        vector_rounds = ceil_div(vectors, self.threads_per_cta)
        row_base = device_workspace.ptr(self.topk_count_inbox_region).toint()
        sentinel = Uint32(sentinel_ptx.WORD_SENTINEL)
        words = cute.make_rmem_tensor((vector_rounds * 4,), Uint32)
        pending = Int32(0)
        for vector_round in cutlass.range_constexpr(vector_rounds):
            vector = Int32(vector_round * self.threads_per_cta) + thread_idx
            for word in cutlass.range_constexpr(4):
                words[vector_round * 4 + word] = Uint32(0)
            if vector < Int32(vectors):
                loaded = sentinel_ptx.ld_volatile_v4(
                    row_base + Int64(vector) * Int64(16)
                )
                for word in cutlass.range_constexpr(4):
                    words[vector_round * 4 + word] = loaded[word]
                    if loaded[word] == sentinel:
                        pending = Int32(1)
        while pending != Int32(0):
            sentinel_ptx.nanosleep(Int32(32))
            pending = Int32(0)
            for vector_round in cutlass.range_constexpr(vector_rounds):
                vector = Int32(vector_round * self.threads_per_cta) + thread_idx
                missing = cutlass.Boolean(False)
                for word in cutlass.range_constexpr(4):
                    if words[vector_round * 4 + word] == sentinel:
                        missing = cutlass.Boolean(True)
                if missing:
                    loaded = sentinel_ptx.ld_volatile_v4(
                        row_base + Int64(vector) * Int64(16)
                    )
                    for word in cutlass.range_constexpr(4):
                        words[vector_round * 4 + word] = loaded[word]
                        if loaded[word] == sentinel:
                            pending = Int32(1)
        for vector_round in cutlass.range_constexpr(vector_rounds):
            vector = Int32(vector_round * self.threads_per_cta) + thread_idx
            if vector < Int32(vectors):
                sts128_v4_b32(
                    chunk_counts.iterator + vector * Int32(4),
                    Int32(words[vector_round * 4]),
                    Int32(words[vector_round * 4 + 1]),
                    Int32(words[vector_round * 4 + 2]),
                    Int32(words[vector_round * 4 + 3]),
                )

    @cute.jit
    def _reset_count_rows(
        self, device_workspace: DeviceWorkspace, thread_idx: Int32
    ) -> None:
        """Reset every count row of this ping-pong buffer to the sentinel once the helper and every index-copy CTA read it.

        The buffer is reused two rounds later; the system-scope ``topk_ready``
        releases of the round in between order this reset before those writes
        (see "Expert-id lists").
        """
        vectors = self.list_segment_count // 4
        row_base = device_workspace.ptr(self.topk_count_inbox_region).toint()
        sentinel = Uint32(sentinel_ptx.WORD_SENTINEL)
        for vector_round in cutlass.range_constexpr(
            ceil_div(vectors, self.threads_per_cta)
        ):
            vector = Int32(vector_round * self.threads_per_cta) + thread_idx
            if vector < Int32(vectors):
                sentinel_ptx.st_volatile_v4_words(
                    row_base + Int64(vector) * Int64(16),
                    sentinel,
                    sentinel,
                    sentinel,
                    sentinel,
                )

    @cute.jit
    def _run_planner(
        self,
        device_workspace: DeviceWorkspace,
        local_rank: Int32,
        smem_base: cute.Pointer,
        source_rank: Int32,
        thread_idx: Int32,
    ) -> None:
        """Plan this rank's dispatch from every source's top-k plane (Sentinel dispatch).

        Every exchange CTA runs the same deterministic plan over the W landed
        planes: per-(source, expert) route counts, per-expert loads, the
        processing order and the data-row bases. Planner 0 alone publishes the
        sizes and the processing order with ``sizes_ready``, then the bases the
        refine reads; every planner scatters ``source_rank``'s routes and adds one
        to ``metadata_ready``. Each planner counts every source itself, so no
        planner waits for another (see "Sentinel top-k inbox").

        Placement inside a (source, expert) segment comes from SMEM atomics and
        varies from launch to launch. No output depends on it: a route's FC1 and
        FC2 results do not depend on which pool row it occupies.
        """
        lane_idx = thread_idx % Int32(32)
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        publishes = source_rank == Int32(0)
        buckets = self.local_expert_bucket_count
        count_matrix = self._smem_workspace.tensor(
            self.helper_count_matrix_region, smem_base
        )
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)

        iket.range_push("planner.count")
        # The counts overlay the exchange stage, whose last reads precede the
        # final barrier of `_send_topk_inbox`.
        cells = self.world_size * buckets
        for cell_round in cutlass.range_constexpr(
            ceil_div(cells, self.threads_per_cta)
        ):
            cell = Int32(cell_round * self.threads_per_cta) + thread_idx
            if cell < Int32(cells):
                count_matrix[cell // Int32(buckets), cell % Int32(buckets)] = Int32(0)
        cute.arch.sync_threads()
        words = self._count_inbox_routes(
            device_workspace, local_rank, source_rank, smem_base, thread_idx
        )
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("planner.order")
        for bucket_round in cutlass.range_constexpr(
            ceil_div(buckets, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(buckets):
                expert_total = Int32(0)
                for source in cutlass.range_constexpr(self.world_size):
                    expert_total = expert_total + count_matrix[Int32(source), bucket]
                totals[bucket] = expert_total
                if cutlass.const_expr(self.token_tile_strided):
                    if bucket < Int32(self.experts_per_rank):
                        tile_layout = self._smem_workspace.tensor(
                            self.helper_tile_rows_region, smem_base
                        )
                        tile_rows = balanced_tile_rows(expert_total, self.token_tile_n)
                        tile_layout[0, bucket] = tile_rows
                        tile_layout[1, bucket] = tile_rows_reciprocal(tile_rows)
        cute.arch.sync_threads()
        self._rank_experts_by_load(
            device_workspace, smem_base, thread_idx, lane_idx, warp_idx, replicated=True
        )
        iket.range_pop()

        if publishes:
            iket.range_push("planner.publish_sizes")
            self._publish_plan_sizes(device_workspace, smem_base, thread_idx)
            iket.range_pop()

        iket.range_push("planner.bases")
        self._plan_bases(
            device_workspace,
            smem_base,
            local_rank,
            source_rank,
            publishes,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        if publishes:
            if thread_idx == Int32(0):
                self.stamp(device_workspace, self.stamp_helper, Int32(1))
        iket.range_pop()

        iket.range_push("planner.scatter")
        self._scatter_planned_routes(
            device_workspace, smem_base, source_rank, words, thread_idx
        )
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.metadata_ready_region), Int32(1)
            )
            self.stamp(device_workspace, self.stamp_sort_done, source_rank)
            # Nothing above waited for this release; see "Sentinel top-k inbox".
            # The slot held zero before the source's one add to it this round
            # (`_clear_previous_topk_ready`), so any other nonzero value is a count
            # left from an earlier round.
            ready_slot = device_workspace.ptr(self.topk_ready_region) + source_rank
            ready = cute.arch.load(ready_slot, Int32, sem="acquire", scope="sys")
            while ready == Int32(0):
                nanosleep(150)
                ready = cute.arch.load(ready_slot, Int32, sem="acquire", scope="sys")
            if ready != Int32(self.topk_ready_target):
                cute.printf(
                    "sentinel dispatch: topk_ready of source %d holds %d, not %d\n",
                    source_rank,
                    ready,
                    Int32(self.topk_ready_target),
                )
                sentinel_ptx.trap()
        iket.range_pop()

    @cute.jit
    def _count_inbox_routes(
        self,
        device_workspace: DeviceWorkspace,
        local_rank: Int32,
        source_rank: Int32,
        smem_base: cute.Pointer,
        thread_idx: Int32,
    ) -> cute.Tensor:
        """Wait for every source's id plane and count this rank's routes in it.

        A thread takes whole 16-byte vectors of the W id planes, issues every load
        before testing any and reloads only the vectors still holding a sentinel
        word; planes beyond one pass of ``planner_id_slots`` vectors per thread
        take several passes. Its first vectors of ``source_rank``'s score plane are
        loaded before the ids and polled in the same sweeps, so they have landed by
        the scatter, which finds them in the returned words after the id slots.

        Every planner counts all W planes, so that each derives every source's
        counts without a hand-off (see "Sentinel top-k inbox"). A local route
        counts itself with a shared-memory atomic on its (source, expert)
        segment. Only ``source_rank``'s routes use the value it returns, their
        offset inside the segment: that value depends on the order of this CTA's
        atomics and differs between planners, so a segment's offsets must all
        come from the one planner that scatters it. The route's record keeps the
        offset and the expert for the scatter, and a route this rank does not
        own is recorded as -1. All atomics of a pass leave before any of their
        results is used, so a thread waits for them once rather than once per
        route.
        """
        threads = self.threads_per_cta
        buckets = self.local_expert_bucket_count
        plane_vectors = self.routes_per_rank // 4
        id_vectors = self.world_size * plane_vectors
        id_slots = self.planner_id_slots
        score_slots = self.planner_score_slots
        slots = id_slots + score_slots
        slab_bytes = self.topk_slab_elements * 4
        index_inbox = device_workspace.ptr(self.topk_index_inbox_region).toint()
        count_matrix = self._smem_workspace.tensor(
            self.helper_count_matrix_region, smem_base
        )
        records = self._smem_workspace.tensor(
            self.helper_route_records_region, smem_base
        )
        local_expert_begin = local_rank * Int32(self.experts_per_rank)

        # A slot without an address holds zeros, which never read as a sentinel.
        words = cute.make_rmem_tensor((slots * 4,), Uint32)
        addresses = cute.make_rmem_tensor((slots,), Int64)
        if cutlass.const_expr(score_slots > 0):
            score_plane = device_workspace.ptr(
                self.topk_score_inbox_region
            ).toint() + Int64(source_rank) * Int64(slab_bytes)
            for score_slot in cutlass.range_constexpr(score_slots):
                slot = id_slots + score_slot
                vector = Int32(score_slot * threads) + thread_idx
                addresses[slot] = Int64(0)
                for element in cutlass.range_constexpr(4):
                    words[slot * 4 + element] = Uint32(0)
                if vector < Int32(plane_vectors):
                    addresses[slot] = score_plane + Int64(vector) * Int64(16)
                    loaded = sentinel_ptx.ld_volatile_v4(addresses[slot])
                    for element in cutlass.range_constexpr(4):
                        words[slot * 4 + element] = loaded[element]

        for pass_idx in cutlass.range(
            ceil_div(id_vectors, threads * id_slots), unroll=1
        ):
            first_vector = pass_idx * Int32(threads * id_slots) + thread_idx
            for slot in cutlass.range_constexpr(id_slots):
                vector = first_vector + Int32(slot * threads)
                addresses[slot] = Int64(0)
                for element in cutlass.range_constexpr(4):
                    words[slot * 4 + element] = Uint32(0)
                if vector < Int32(id_vectors):
                    source = vector // Int32(plane_vectors)
                    addresses[slot] = (
                        index_inbox
                        + Int64(source) * Int64(slab_bytes)
                        + Int64(vector - source * Int32(plane_vectors)) * Int64(16)
                    )
                    loaded = sentinel_ptx.ld_volatile_v4(addresses[slot])
                    for element in cutlass.range_constexpr(4):
                        words[slot * 4 + element] = loaded[element]
            self._poll_inbox_vectors(
                device_workspace, words, addresses, slots, source_rank
            )

            offsets = cute.make_rmem_tensor((id_slots * 4,), Int32)
            for slot in cutlass.range_constexpr(id_slots):
                vector = first_vector + Int32(slot * threads)
                for element in cutlass.range_constexpr(4):
                    offsets[slot * 4 + element] = Int32(-1)
                if vector < Int32(id_vectors):
                    segments = count_matrix.iterator + (
                        vector // Int32(plane_vectors)
                    ) * Int32(buckets)
                    for element in cutlass.range_constexpr(4):
                        local_expert = (
                            Int32(words[slot * 4 + element]) - local_expert_begin
                        )
                        if Uint32(local_expert) < Uint32(self.experts_per_rank):
                            offsets[slot * 4 + element] = Int32(
                                cute.arch.atomic_add(
                                    segments + local_expert,
                                    Int32(1),
                                    sem="relaxed",
                                    scope="cta",
                                )
                            )
            for slot in cutlass.range_constexpr(id_slots):
                vector = first_vector + Int32(slot * threads)
                if vector < Int32(id_vectors):
                    source = vector // Int32(plane_vectors)
                    if source == source_rank:
                        first_route = (vector - source * Int32(plane_vectors)) * Int32(
                            4
                        )
                        for element in cutlass.range_constexpr(4):
                            offset = offsets[slot * 4 + element]
                            local_expert = (
                                Int32(words[slot * 4 + element]) - local_expert_begin
                            )
                            record = Int32(-1)
                            if offset >= Int32(0):
                                record = (offset << Int32(16)) | local_expert
                            records[first_route + Int32(element)] = record
        return words

    @cute.jit
    def _poll_inbox_vectors(
        self,
        device_workspace: DeviceWorkspace,
        words: cute.Tensor,
        addresses: cute.Tensor,
        slots: cutlass.Constexpr[int],
        source_rank: Int32,
    ) -> None:
        """Reload every inbox vector that still holds a sentinel word until none does.

        Every word leaves the sentinel once per round, for its final value, so a
        vector none of whose words is the sentinel any more is final.

        This is the round's first wait on the peers, and a peer may launch any
        time later, so a sweep budget running out is no failure by itself. Once
        ``POLL_SWEEP_LIMIT`` sweeps have passed, the poll acquires every source's
        ``topk_ready`` before each further sweep and sleeps
        ``planner_backoff_ns`` while one of them is still zero, since that
        source may not have sent yet. Once all are set every sent word is
        visible, and a sentinel still there a whole budget later traps.
        """
        topk_ready = device_workspace.ptr(self.topk_ready_region)
        pending = Int32(1)
        sweeps = Int32(0)
        all_sent = Int32(0)
        while pending != Int32(0):
            pending = Int32(0)
            for slot in cutlass.range_constexpr(slots):
                missing = (
                    (words[slot * 4] == Uint32(sentinel_ptx.WORD_SENTINEL))
                    | (words[slot * 4 + 1] == Uint32(sentinel_ptx.WORD_SENTINEL))
                    | (words[slot * 4 + 2] == Uint32(sentinel_ptx.WORD_SENTINEL))
                    | (words[slot * 4 + 3] == Uint32(sentinel_ptx.WORD_SENTINEL))
                )
                if missing:
                    loaded = sentinel_ptx.ld_volatile_v4(addresses[slot])
                    for word in cutlass.range_constexpr(4):
                        words[slot * 4 + word] = loaded[word]
                    pending = Int32(1)
            sweeps += Int32(1)
            if sweeps == Int32(sentinel_ptx.POLL_SWEEP_LIMIT):
                if pending != Int32(0):
                    if all_sent != Int32(0):
                        cute.printf(
                            "sentinel dispatch: planner %d never received every top-k plane\n",
                            source_rank,
                        )
                        sentinel_ptx.trap()
                    all_sent = Int32(1)
                    for source in cutlass.range_constexpr(self.world_size):
                        if cute.arch.load(
                            topk_ready + Int32(source),
                            Int32,
                            sem="acquire",
                            scope="sys",
                        ) == Int32(0):
                            all_sent = Int32(0)
                    sweeps = Int32(0)
                    if all_sent == Int32(0):
                        sentinel_ptx.nanosleep(Int32(self.planner_backoff_ns))
                        sweeps = Int32(sentinel_ptx.POLL_SWEEP_LIMIT - 1)

    @cute.jit
    def _publish_plan_sizes(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
    ) -> None:
        """Planner 0: publish the sizes and the processing order, then release ``sizes_ready``.

        The same outputs the helper's fold and ``_place_expert_at_position`` write,
        copied out of the SMEM tables the ranking left.
        """
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        helper_expert_order = self._smem_workspace.tensor(
            self.helper_expert_order_region, smem_base
        )
        helper_expert_sizes_in_order = self._smem_workspace.tensor(
            self.helper_expert_sizes_in_order_region, smem_base
        )
        expert_sizes = device_workspace.tensor(self.expert_sizes_region)
        expert_sizes_in_order = device_workspace.tensor(
            self.expert_sizes_in_order_region
        )
        expert_order = device_workspace.tensor(self.expert_order_region)
        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                expert_order[bucket] = helper_expert_order[bucket]
                if bucket < Int32(self.experts_per_rank):
                    expert_sizes_in_order[bucket] = helper_expert_sizes_in_order[bucket]
                    expert_sizes[bucket] = totals[bucket]
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.sizes_ready_region), Int32(1)
            )
            self.stamp(device_workspace, self.stamp_helper, Int32(0))

    @cute.jit
    def _plan_bases(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        local_rank: Int32,
        source_rank: Int32,
        publishes: cutlass.Boolean,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
    ) -> None:
        """Each expert's data-row base and ``source_rank``'s segment base in it, into SMEM.

        Sources are laid out in ring order from this rank
        (``_store_plan_bases``), the order the pull index copy gives the
        segments as well. With ``publishes`` the data-row, SF-row and
        refine-block bases and the refine block total go out as well, for the
        refine, which reads them once ``metadata_ready`` is complete. Up to 32
        buckets one warp scans both row spaces with shuffles in every planner;
        above, every planner runs the helper's CTA-wide scan of the data rows,
        and only the publishing one the SF-row and refine-block prefixes.
        """
        buckets = self.local_expert_bucket_count
        expert_sizes_in_order = self._smem_workspace.tensor(
            self.helper_expert_sizes_in_order_region, smem_base
        )
        expert_order = self._smem_workspace.tensor(
            self.helper_expert_order_region, smem_base
        )
        data_expert_base = device_workspace.tensor(self.data_expert_base_region)
        ring_position = (source_rank - local_rank + Int32(self.world_size)) % Int32(
            self.world_size
        )
        if cutlass.const_expr(self.ranks_experts_in_warp):
            if warp_idx == Int32(0):
                load = Int32(0)
                expert = Int32(0)
                if lane_idx < Int32(buckets):
                    load = expert_sizes_in_order[lane_idx]
                    expert = expert_order[lane_idx]
                data_rows = expert_pool_rows(
                    load,
                    self.token_padding_block,
                    self.token_tile_n,
                    self.token_slot_rows,
                )
                sf_rows = expert_pool_rows(
                    load, self.sf_padding_block, self.token_tile_n, self.token_slot_rows
                )
                data_base = _warp_inclusive_sum(data_rows, lane_idx) - data_rows
                sf_end = _warp_inclusive_sum(sf_rows, lane_idx)
                sf_base = sf_end - sf_rows
                if lane_idx < Int32(buckets):
                    self._store_plan_bases(
                        smem_base, expert, data_base, local_rank, ring_position
                    )
                    if publishes:
                        data_expert_base[expert] = data_base
                        device_workspace.tensor(self.sf_expert_base_region)[expert] = (
                            sf_base
                        )
                        device_workspace.tensor(self.refine_block_base_region)[
                            expert
                        ] = sf_base // Int32(self.sf_padding_block)
                # Lanes past the bucket count add nothing, so the last lane's sum
                # is the whole SF-row extent.
                if publishes:
                    if lane_idx == Int32(31):
                        device_workspace.tensor(self.refine_block_total_region)[0] = (
                            sf_end // Int32(self.sf_padding_block)
                        )
            cute.arch.sync_threads()
        else:
            scan_input = self._smem_workspace.tensor(
                self.helper_scan_input_region, smem_base
            )
            prefix = self._smem_workspace.tensor(self.helper_prefix_region, smem_base)
            warp_totals = self._smem_workspace.tensor(
                self.helper_warp_totals_region, smem_base
            )
            self._scan_padded_rows(
                expert_sizes_in_order,
                scan_input,
                prefix,
                warp_totals,
                self.token_padding_block,
                thread_idx,
                lane_idx,
                warp_idx,
            )
            for bucket_round in cutlass.range_constexpr(
                ceil_div(buckets, self.threads_per_cta)
            ):
                scan_entry = Int32(bucket_round * self.threads_per_cta) + thread_idx
                if scan_entry < Int32(buckets):
                    expert = expert_order[scan_entry]
                    self._store_plan_bases(
                        smem_base, expert, prefix[scan_entry], local_rank, ring_position
                    )
                    if publishes:
                        data_expert_base[expert] = prefix[scan_entry]
            cute.arch.sync_threads()
            if publishes:
                self._publish_padded_prefix(
                    device_workspace,
                    expert_sizes_in_order,
                    expert_order,
                    scan_input,
                    prefix,
                    warp_totals,
                    self.sf_padding_block,
                    self.sf_expert_base_region,
                    thread_idx,
                    lane_idx,
                    warp_idx,
                )
                self._publish_refine_block_prefix(
                    device_workspace,
                    expert_sizes_in_order,
                    expert_order,
                    scan_input,
                    prefix,
                    warp_totals,
                    thread_idx,
                    lane_idx,
                    warp_idx,
                )

    @cute.jit
    def _store_plan_bases(
        self,
        smem_base: cute.Pointer,
        expert: Int32,
        data_base: Int32,
        local_rank: Int32,
        ring_position: Int32,
    ) -> None:
        """Record one expert's data-row base and the planner's segment base in it.

        The planner's source comes ``ring_position`` sources after this rank, and
        the segments of the sources before it come first.
        """
        count_matrix = self._smem_workspace.tensor(
            self.helper_count_matrix_region, smem_base
        )
        bases = self._smem_workspace.tensor(self.helper_plan_bases_region, smem_base)
        segment_base = data_base
        for ring_step in cutlass.range_constexpr(self.world_size - 1):
            if Int32(ring_step) < ring_position:
                source = (local_rank + Int32(ring_step)) % Int32(self.world_size)
                segment_base = segment_base + count_matrix[source, expert]
        bases[0, expert] = data_base
        bases[1, expert] = segment_base

    @cute.jit
    def _scatter_planned_routes(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        source_rank: Int32,
        words: cute.Tensor,
        thread_idx: Int32,
    ) -> None:
        """Write the gather index, routing record and score of ``source_rank``'s local routes.

        A thread takes the score vectors ``_count_inbox_routes`` polled for it,
        whose words follow the id slots in ``words``. Planes with more vectors than
        ``planner_score_slots`` passes of the CTA load and poll the rest here.
        """
        threads = self.threads_per_cta
        plane_vectors = self.routes_per_rank // 4
        vector_rounds = ceil_div(plane_vectors, threads)
        held_rounds = self.planner_score_slots
        id_slots = self.planner_id_slots
        for vector_round in cutlass.range_constexpr(held_rounds):
            vector = Int32(vector_round * threads) + thread_idx
            if vector < Int32(plane_vectors):
                scores = cute.make_rmem_tensor((4,), Uint32)
                for element in cutlass.range_constexpr(4):
                    scores[element] = words[(id_slots + vector_round) * 4 + element]
                self._scatter_route_vector(
                    device_workspace, smem_base, source_rank, vector, scores
                )
        if cutlass.const_expr(vector_rounds > held_rounds):
            for vector_round in cutlass.range(held_rounds, vector_rounds, unroll=1):
                vector = vector_round * Int32(threads) + thread_idx
                if vector < Int32(plane_vectors):
                    scores = cute.make_rmem_tensor((4,), Uint32)
                    for element in cutlass.range_constexpr(4):
                        scores[element] = Uint32(0)
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        address = cute.make_rmem_tensor((1,), Int64)
                        address[0] = (
                            device_workspace.ptr(self.topk_score_inbox_region).toint()
                            + Int64(source_rank) * Int64(self.topk_slab_elements * 4)
                            + Int64(vector) * Int64(16)
                        )
                        loaded = sentinel_ptx.ld_volatile_v4(address[0])
                        for element in cutlass.range_constexpr(4):
                            scores[element] = loaded[element]
                        self._poll_inbox_vectors(
                            device_workspace, scores, address, 1, source_rank
                        )
                    self._scatter_route_vector(
                        device_workspace, smem_base, source_rank, vector, scores
                    )

    @cute.jit
    def _scatter_route_vector(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        source_rank: Int32,
        vector: Int32,
        scores: cute.Tensor,
    ) -> None:
        """Place the local routes among one 16-byte vector of ``source_rank``'s plane."""
        records = self._smem_workspace.tensor(
            self.helper_route_records_region, smem_base
        )
        bases = self._smem_workspace.tensor(self.helper_plan_bases_region, smem_base)
        gather_index = device_workspace.tensor(self.gather_index_region)
        token_metadata = device_workspace.tensor(self.token_src_metadata_region)
        dense_row_begin = source_rank * Int32(self.max_tokens_per_rank)
        if cutlass.const_expr(self.apply_topk_at_fc1):
            routed_scores = self.fc1_topk_scores_tensor(device_workspace)
            # The score travels as bits, so it lands exactly as it was sent.
            routed_score_bits = cute.make_tensor(
                cute.recast_ptr(routed_scores.iterator, dtype=Uint32),
                routed_scores.layout,
            )
        for element in cutlass.range_constexpr(4):
            route = vector * Int32(4) + Int32(element)
            record = records[route]
            if record >= Int32(0):
                local_expert = record & Int32(0xFFFF)
                pool_row = bases[1, local_expert] + (record >> Int32(16))
                if cutlass.const_expr(self.token_tile_strided):
                    # The segment bases count routes, not pool rows; the slot
                    # layout is applied to the route's rank within its expert.
                    expert_base = bases[0, local_expert]
                    tile_layout = self._smem_workspace.tensor(
                        self.helper_tile_rows_region, smem_base
                    )
                    pool_row = expert_base + slot_physical_row(
                        pool_row - expert_base,
                        tile_layout[0, local_expert],
                        tile_layout[1, local_expert],
                        self.token_slot_rows,
                    )
                source_token = route // Int32(self.topk)
                gather_index[pool_row] = dense_row_begin + source_token
                token_metadata[pool_row] = TokenSrcMetadata(
                    src_rank=source_rank,
                    src_token=source_token,
                    src_topk=route - source_token * Int32(self.topk),
                ).pack()
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    routed_score_bits[pool_row] = scores[element]

    @cute.jit
    def _run_refine(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        linear_cta_idx: Int32,
        thread_idx: Int32,
    ) -> None:
        """Verify the landed rows and rewrite their scale factors into the layout the MMA reads (Sentinel dispatch).

        The table and the first block may be settled from ``sizes_ready`` alone
        (``refine_plans_table``), so ``metadata_ready`` then gates the first
        block's rows only.
        """
        group_threads = self.refine_warps_per_group * 32
        group_idx = thread_idx // Int32(group_threads)
        thread_in_group = thread_idx % Int32(group_threads)
        # Warp-uniform because a group owns whole warps, which is what makes an
        # implicitly aligned `bar.sync` on a runtime id legal.
        group_barrier = Int32(self.refine_barrier_base) + group_idx

        if cutlass.const_expr(not self.refine_plans_table):
            self._wait_refine_open(device_workspace, linear_cta_idx, thread_idx)

        # One mbarrier per refine group sits in the SMEM plan and is initialized
        # here, though no refine step waits on it: the plan and this entry keep
        # the layout and instruction stream the refine was measured with.
        group_mbarriers = self._smem_workspace.ptr(
            self.refine_mbarrier_region, smem_base
        )
        if thread_idx < Int32(self.refine_participant_groups):
            cute.arch.mbarrier_init(group_mbarriers + thread_idx, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        iket.range_push("refine.table")
        if cutlass.const_expr(self.refine_plans_table):
            expert_table = self._plan_refine_expert_table(
                device_workspace, smem_base, thread_idx
            )
            block_total = self._smem_workspace.tensor(
                self.refine_block_count_region, smem_base
            )[0]
        else:
            expert_table = self._stage_refine_expert_table(
                device_workspace, smem_base, thread_idx
            )
        iket.range_pop()
        claim = self._smem_workspace.tensor(self.refine_claim_region, smem_base)
        work_counter = device_workspace.ptr(self.refine_work_counter_region)
        if cutlass.const_expr(not self.refine_plans_table):
            block_total = device_workspace.tensor(self.refine_block_total_region)[0]

        claim_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
        )
        claimed = cute.make_rmem_tensor((self.refine_claim_fields,), cutlass.Int32)

        fc1_sf_ready = device_workspace.ptr(self.fc1_sf_ready_region)
        first_block = None
        if cutlass.const_expr(self.refine_static_blocks > 0):
            # This group's index among all refine groups; see `refine_static_blocks`.
            refine_cta = linear_cta_idx - Int32(self.helper_cta_index)
            first_block = (
                (refine_cta - Int32(1)) * Int32(self.refine_participant_groups)
                + Int32(1)
                + group_idx
            )
            if refine_cta == Int32(0):
                first_block = Int32(0)
                if group_idx != Int32(0):
                    first_block = (
                        Int32(
                            self.refine_static_blocks - self.refine_participant_groups
                        )
                        + group_idx
                    )
        owning_expert = self._claim_next_block(
            claim,
            expert_table,
            work_counter,
            block_total,
            group_idx,
            thread_in_group,
            group_barrier,
            group_threads,
            static_block=first_block,
        )
        cute.copy(claim_atom, claim[group_idx, None], claimed)
        if cutlass.const_expr(self.refine_plans_table):
            # The first block is settled; its rows are what the scatter still owes.
            self._wait_refine_open(device_workspace, linear_cta_idx, thread_idx)

        while claimed[self.refine_claim_block] < block_total:
            self._refine_one_block(
                device_workspace,
                smem_base,
                claimed[self.refine_claim_data_row],
                claimed[self.refine_claim_sf_row],
                claimed[self.refine_claim_live_rows],
                group_idx,
                thread_in_group,
                group_barrier,
                group_threads,
            )
            iket.range_push("refine.publish_block")
            cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)
            if thread_in_group == Int32(0):
                red_async_add_release_gpu_s32(fc1_sf_ready + owning_expert, Int32(1))
                if cutlass.const_expr(self.stamps):
                    self.stamp_latest(
                        device_workspace, self.stamp_expert_ready, owning_expert
                    )
            iket.range_pop()
            owning_expert = self._claim_next_block(
                claim,
                expert_table,
                work_counter,
                block_total,
                group_idx,
                thread_in_group,
                group_barrier,
                group_threads,
            )
            cute.copy(claim_atom, claim[group_idx, None], claimed)

    @cute.jit
    def _wait_refine_open(
        self,
        device_workspace: DeviceWorkspace,
        linear_cta_idx: Int32,
        thread_idx: Int32,
    ) -> None:
        """Wait until the gather index is complete (``metadata_ready``).

        Under the pull dispatch the index-copy CTAs acquired every source's
        ``topk_ready`` before their ``metadata_ready`` adds
        (``pull_acquire_in_copy``), or else this CTA acquires them as well
        (``_acquire_topk_ready``); that orders every source rank's staging before
        this CTA's fetches (see "Pull dispatch" above). Each thread then fences
        its later bulk loads of the sources' staged rows behind these generic
        acquires.
        """
        iket.range_push("refine.wait")
        if cutlass.const_expr(
            self.dispatch_protocol == "pull" and not self.pull_acquire_in_copy
        ):
            # The second warp acquires the expert-id releases while the first waits
            # for the index lists; both are needed before the barrier below.
            if thread_idx == Int32(32):
                self._acquire_topk_ready(device_workspace)
        if thread_idx == Int32(0):
            spin_wait(
                device_workspace.ptr(self.metadata_ready_region),
                lambda value: value == Int32(self.metadata_ready_target),
                scope="gpu",
            )
            self.stamp(device_workspace, self.stamp_refine_open, linear_cta_idx)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.dispatch_protocol == "pull"):
            # Orders the generic acquires above before this thread's async-proxy loads of the rows they publish.
            sentinel_ptx.fence_proxy_async_global()
        iket.range_pop()

    @cute.jit
    def _acquire_topk_ready(self, device_workspace: DeviceWorkspace) -> None:
        """Acquire every source's ``topk_ready`` once all its exchange CTAs have released it (pull dispatch).

        The lists announce their own arrival, so the plan never waits for these
        releases; this is the edge that orders every source's staged activations
        and earlier writes before this rank's fetches, reads and resets (see
        "Expert-id lists").
        Relaxed system polls of all W words go out together, and once every one
        shows its target a system-scope acquire fence makes the observed
        releases synchronize with this thread.
        """
        target = Uint32(self.exchange_cta_count)
        pending = Int32(1)
        while pending != Int32(0):
            pending = Int32(0)
            for source_rank in cutlass.range_constexpr(self.world_size):
                ready_address = (
                    device_workspace.ptr(self.topk_ready_region) + Int32(source_rank)
                ).toint()
                if sentinel_ptx.ld_volatile_u32(ready_address) != target:
                    pending = Int32(1)
            if pending != Int32(0):
                sentinel_ptx.nanosleep(Int32(32))
        sentinel_ptx.fence_acq_rel_sys()

    @cute.jit
    def _acquire_source_ready(
        self, device_workspace: DeviceWorkspace, source_rank: Int32
    ) -> None:
        """Acquire one source's ``topk_ready`` at system scope once all its exchange CTAs have released it.

        Relaxed polls wait for the target, then one system-scope acquire load of
        the word synchronizes with every release in its sequence of adds. An
        acquire load costs no system-scope fence, which waits for every write
        this SM still has in flight.
        """
        ready_address = (
            device_workspace.ptr(self.topk_ready_region) + source_rank
        ).toint()
        target = Uint32(self.exchange_cta_count)
        while sentinel_ptx.ld_volatile_u32(ready_address) != target:
            sentinel_ptx.nanosleep(Int32(32))
        sentinel_ptx.ld_acquire_sys_u32(ready_address)

    @cute.jit
    def _plan_refine_expert_table(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
    ) -> cute.Tensor:
        """Derive the four per-expert columns the claim loop needs, and the block total.

        See ``refine_plans_table``. The first warp takes one position of the
        processing order per lane and runs the scan of ``_plan_bases`` on the loads
        planner 0 published, which is the same scan over the same values, so the
        bases agree with the ones the planner publishes bit for bit. Positions past
        the live experts carry no load. The block total lands in
        ``refine_block_count``.
        """
        expert_table = self._smem_workspace.tensor(
            self.refine_expert_table_region, smem_base
        )
        block_count = self._smem_workspace.tensor(
            self.refine_block_count_region, smem_base
        )
        expert_sizes_in_order = device_workspace.tensor(
            self.expert_sizes_in_order_region
        )
        expert_order = device_workspace.tensor(self.expert_order_region)
        buckets = self.local_expert_bucket_count

        if thread_idx == Int32(0):
            spin_wait(
                device_workspace.ptr(self.sizes_ready_region),
                lambda value: value == Int32(1),
                scope="gpu",
            )
        cute.arch.sync_threads()
        if thread_idx < Int32(32):
            lane_idx = thread_idx
            load = Int32(0)
            expert = Int32(0)
            if lane_idx < Int32(buckets):
                expert = expert_order[lane_idx]
                if lane_idx < Int32(self.experts_per_rank):
                    load = expert_sizes_in_order[lane_idx]
            data_rows = expert_pool_rows(
                load, self.token_padding_block, self.token_tile_n, self.token_slot_rows
            )
            sf_rows = expert_pool_rows(
                load, self.sf_padding_block, self.token_tile_n, self.token_slot_rows
            )
            data_base = _warp_inclusive_sum(data_rows, lane_idx) - data_rows
            sf_end = _warp_inclusive_sum(sf_rows, lane_idx)
            sf_base = sf_end - sf_rows
            if lane_idx < Int32(buckets):
                expert_table[self.refine_table_block_base, expert] = sf_base // Int32(
                    self.sf_padding_block
                )
                expert_table[self.refine_table_expert_size, expert] = load
                expert_table[self.refine_table_data_base, expert] = data_base
                expert_table[self.refine_table_sf_base, expert] = sf_base
            # Lanes past the bucket count add nothing, so the last lane's sum is
            # the whole SF-row extent.
            if lane_idx == Int32(31):
                block_count[0] = sf_end // Int32(self.sf_padding_block)
        cute.arch.sync_threads()
        return expert_table

    @cute.jit
    def _claim_next_block(
        self,
        claim: cute.Tensor,
        expert_table: cute.Tensor,
        work_counter: cute.Pointer,
        block_total: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
        group_barrier: Int32,
        group_threads: int,
        static_block: Optional[Int32] = None,
    ) -> Int32:
        """Take the next block and publish everything it implies.

        ``static_block``, when given, is the block to take; otherwise the block
        comes from the work counter, offset past the ``refine_static_blocks``
        blocks that are never claimed.

        Only the group's first warp resolves the block, and the barrier that has
        to be here anyway hands the result to the rest. Resolving inside the block
        body instead would have every warp repeat the same ballot and every thread
        the same table lookups to reach one group-uniform answer.

        The fields are assembled in registers and published as one 16-byte store,
        so a reader either sees the whole claim or none of it.

        The owning expert is returned rather than stored with them: only the
        thread that publishes this block's readiness needs it, and that thread is
        in the warp that just resolved it. Widening the claim would cost the
        single-store property for a value no other thread reads.
        """
        owning_expert = Int32(-1)
        if thread_in_group < Int32(32):
            claimed_block = Int32(0)
            if cutlass.const_expr(static_block is not None):
                claimed_block = static_block
            else:
                if thread_in_group == Int32(0):
                    claimed_block = Int32(self.refine_static_blocks) + Int32(
                        cute.arch.atomic_add(
                            work_counter, Int32(1), sem="relaxed", scope="gpu"
                        )
                    )
                # Broadcast rather than staged through shared memory: the ballot
                # below needs the whole warp to agree on the block, and a shuffle
                # gets it there without a second barrier.
                claimed_block = Int32(cute.arch.shuffle_sync(claimed_block, Int32(0)))

            resolved = cute.make_rmem_tensor((self.refine_claim_fields,), cutlass.Int32)
            resolved[self.refine_claim_block] = claimed_block
            resolved[self.refine_claim_data_row] = Int32(0)
            resolved[self.refine_claim_sf_row] = Int32(0)
            resolved[self.refine_claim_live_rows] = Int32(0)
            if claimed_block < block_total:
                block_rows = Int32(self.sf_padding_block)
                owning_expert = self._resolve_block_expert(
                    expert_table, claimed_block, thread_in_group
                )
                block_in_expert = (
                    claimed_block
                    - expert_table[self.refine_table_block_base, owning_expert]
                )
                rows_before = block_in_expert * block_rows
                expert_size = expert_table[self.refine_table_expert_size, owning_expert]
                data_base = expert_table[self.refine_table_data_base, owning_expert]
                sf_base = expert_table[self.refine_table_sf_base, owning_expert]
                resolved[self.refine_claim_data_row] = data_base + rows_before
                resolved[self.refine_claim_sf_row] = sf_base + rows_before
                if cutlass.const_expr(self.token_tile_strided):
                    # A block is one atom of a slot. Its routes are the tile's,
                    # offset by the atom, and the slot's padding atom has none.
                    blocks_per_slot = self.token_slot_rows // self.sf_padding_block
                    block_in_slot = block_in_expert % Int32(blocks_per_slot)
                    rows_in_slot_before = block_in_slot * block_rows
                    tile_rows = balanced_tile_rows(expert_size, self.token_tile_n)
                    routes_before = (
                        block_in_expert // Int32(blocks_per_slot)
                    ) * tile_rows + rows_in_slot_before
                    live_rows = cutlass.min(
                        expert_size - routes_before,
                        cutlass.min(block_rows, tile_rows - rows_in_slot_before),
                    )
                    resolved[self.refine_claim_live_rows] = cutlass.max(
                        live_rows, Int32(0)
                    )
                else:
                    resolved[self.refine_claim_live_rows] = cutlass.min(
                        expert_size - rows_before, block_rows
                    )

            if thread_in_group == Int32(0):
                claim_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
                )
                cute.copy(claim_atom, resolved, claim[group_idx, None])
        cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)
        return owning_expert

    @cute.jit
    def _stage_refine_expert_table(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
    ) -> cute.Tensor:
        """Bring the four per-expert columns the claim loop needs into SMEM.

        Every thread resolves its block's owning expert by scanning all buckets,
        so these belong in SMEM rather than being re-read from GMEM per block.
        """
        expert_table = self._smem_workspace.tensor(
            self.refine_expert_table_region, smem_base
        )
        block_base = device_workspace.tensor(self.refine_block_base_region)
        data_base = device_workspace.tensor(self.data_expert_base_region)
        sf_base = device_workspace.tensor(self.sf_expert_base_region)
        expert_sizes = device_workspace.tensor(self.expert_sizes_region)

        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                expert_table[self.refine_table_block_base, bucket] = block_base[bucket]
                expert_table[self.refine_table_data_base, bucket] = data_base[bucket]
                expert_table[self.refine_table_sf_base, bucket] = sf_base[bucket]
                # `expert_sizes` only covers live experts; padded buckets read as
                # empty so the ownership test rejects them.
                expert_table[self.refine_table_expert_size, bucket] = Int32(0)
                if bucket < Int32(self.experts_per_rank):
                    expert_table[self.refine_table_expert_size, bucket] = expert_sizes[
                        bucket
                    ]
        cute.arch.sync_threads()
        return expert_table

    @cute.jit
    def _refine_one_block(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        data_row_begin: Int32,
        sf_row_begin: Int32,
        live_rows: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
        group_barrier: Int32,
        group_threads: int,
    ) -> None:
        """Turn one 128-row block of landed scale factors into canonical atoms.

        Takes the rows to move rather than the block to resolve: which expert the
        block landed in was settled when it was claimed.
        """
        iket.range_push("refine.into_smem")
        self._stage_block_rows(
            device_workspace,
            smem_base,
            data_row_begin,
            live_rows,
            group_idx,
            thread_in_group,
            group_barrier,
            group_threads,
        )
        iket.range_pop()

        iket.range_push("refine.out_of_smem")
        self._scatter_block_atoms(
            device_workspace,
            smem_base,
            sf_row_begin,
            group_idx,
            thread_in_group,
        )
        iket.range_pop()

    @cute.jit
    def _resolve_block_expert(
        self, expert_table: cute.Tensor, block_index: Int32, lane_idx: Int32
    ) -> Int32:
        """Which local expert owns a linear refine block.

        One lane per bucket, so the answer comes out of a ballot rather than a scan
        every thread repeats. The whole group works on one block, so a scan would
        have all `group_threads` threads walk every bucket to reach the same
        answer, and would read one bucket broadcast to all lanes instead of one
        bucket per lane.

        Exactly one bucket satisfies the half-open test. Testing the upper bound
        too, rather than just taking the last bucket whose base fits, is what keeps
        empty experts from claiming a block that starts where they end.
        """
        resolved = Int32(0)
        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, 32)
        ):
            bucket = Int32(bucket_round * 32) + lane_idx
            bucket_owns_block = cutlass.Boolean(False)
            if bucket < Int32(self.local_expert_bucket_count):
                bucket_base = expert_table[self.refine_table_block_base, bucket]
                bucket_blocks = expert_pool_rows(
                    expert_table[self.refine_table_expert_size, bucket],
                    Int32(self.sf_padding_block),
                    self.token_tile_n,
                    self.token_slot_rows,
                ) // Int32(self.sf_padding_block)
                bucket_owns_block = (bucket_base <= block_index) & (
                    block_index < bucket_base + bucket_blocks
                )
            owning_lane = _first_matching_lane(bucket_owns_block)
            if owning_lane >= Int32(0):
                resolved = Int32(bucket_round * 32) + owning_lane
        return resolved

    @cute.jit
    def _stage_block_rows(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        data_row_begin: Int32,
        live_rows: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
        group_barrier: Int32,
        group_threads: int,
    ) -> None:
        """Verify a block's landing rows and write their scale factors into the group's output stage.

        The rows come out of the landing verification: the loads that prove a
        row has arrived already hold its scale factors, so the lanes holding
        them write them as atom words straight into the output stage, and the
        landed rows are never read a second time. Block row ``r`` takes
        canonical row position ``t = 4 * (r % 32) + r // 32`` of each atom, so
        the ``gather_index`` read stays sequential.

        Rows past the expert's live count are left holding whatever the previous
        block put there. They land in atoms the MMA does read, but only in N
        columns its dynamic instruction extent excludes.
        """
        gather_index = device_workspace.tensor(self.gather_index_region)
        output = self._smem_workspace.tensor(self.refine_output_region, smem_base)
        self._verify_block_rows(
            device_workspace,
            gather_index,
            data_row_begin,
            live_rows,
            output[group_idx, 0, None].iterator,
            thread_in_group,
        )
        # Publish every writer's generic SMEM stores before the async bulk reads
        # of `_scatter_block_atoms`; every live row is verified and its atoms
        # written once the whole group is here.
        cute.arch.fence_proxy(kind="async.shared", space="cta")
        cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)

    @cute.jit
    def _run_pull(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        linear_cta_idx: Int32,
        thread_idx: Int32,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        staging_peer_rank_ptr_mapper: SymmetricBufferDevice,
    ) -> None:
        """Fetch this rank's routed rows from their source ranks (pull dispatch).

        Every warp walks on its own. The work counter hands out claims in
        processing order, each ``pull_chunk_rows`` consecutive routes of one
        token tile of one expert (a tile's last claim may be short), which sit
        on consecutive rows of that tile's slot in both pools. Lane ``i`` holds
        the landing slot ``s * max_tokens_per_rank + t`` of the claim's ``i``-th
        route and loads that token's scale factors out of source ``s``'s staged
        scale factors.
        The data rows stream through the warp's ring of ``pull_pipeline_depth``
        slots: up to one less than that many bulk loads from the source ranks'
        staged activations in flight, each landed row leaving for its landing
        slot in this round's ping-pong buffer (with ``fc1_token_pool``, for its
        route's row of the token pool unless its token tile falls back to the
        landing slots) as one bulk store, and the slot of the store
        before it reloaded, so the warp never waits for the store it has just
        issued. The warp claims its next routes while the claim's stores drain,
        then each lane writes its route's scale-factor words into their
        canonical atoms (``_verify_block_rows`` has the atom layout). Once the
        data stores have completed, the claim's routes count into the expert's
        ``fc1_sf_ready``, or with ``pull_tile_ready`` into its token tile's
        ``fc1_tile_ready``. Each warp takes its first claim before the index
        lists are complete, as its number needs nothing the plan writes.

        With ``pull_walk_dedup`` every lane claims its route's landing slot
        (``landing_state``, tagged with the round as for the pull warps) as the
        claim opens, without waiting for the result: the claim's first routes,
        as many as the ring keeps in flight, are fetched either way, and a later
        route's row only if the warp won its slot (``_open_walk_claim``). Once its
        stores have completed the warp releases the slots it won, and a lane whose
        row it skipped waits for the winner's landed tag before the claim counts.
        """
        depth = self.pull_pipeline_depth
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane = thread_idx % Int32(32)
        data_row_bytes = (
            self.plain_activation_row_elements * int(self.activation_dtype.width) // 8
        )
        sf_vectors = self.hidden_sf * int(self.activation_sf_dtype.width) // 8 // 16
        if cutlass.const_expr(
            sf_vectors * 16 * 8 != self.hidden_sf * int(self.activation_sf_dtype.width)
        ):
            raise ValueError(
                "The pull dispatch moves scale-factor rows in whole 16-byte vectors."
            )
        atoms = self.refine_total_atoms

        ring_mbarriers = self._smem_workspace.ptr(
            self.refine_pull_mbarrier_region, smem_base
        )
        if thread_idx < Int32(self.threads_per_cta // 32 * depth):
            cute.arch.mbarrier_init(ring_mbarriers + thread_idx, 1)
        cute.arch.mbarrier_init_fence()
        # The tables below precede `prefix_ready`, so they are staged while other
        # CTAs may still be copying their slices of the index lists. The first
        # closing barrier also publishes the ring barriers.
        if thread_idx == Int32(0):
            spin_wait(
                device_workspace.ptr(self.prefix_ready_region),
                lambda value: value == Int32(1),
                scope="gpu",
            )
        cute.arch.sync_threads()
        expert_table = self._stage_refine_expert_table(
            device_workspace, smem_base, thread_idx
        )
        chunk_table = self._stage_pull_chunk_table(
            device_workspace, smem_base, expert_table, thread_idx
        )
        # A claim's number depends on nothing the plan writes, so every warp takes
        # its first one while the index lists may still be in progress.
        work_counter = device_workspace.ptr(self.refine_work_counter_region)
        claimed = Int32(0)
        if lane == Int32(0):
            claimed = Int32(
                cute.arch.atomic_add(work_counter, Int32(1), sem="relaxed", scope="gpu")
            )
        self._wait_refine_open(device_workspace, linear_cta_idx, thread_idx)
        chunk_total = chunk_table[
            self.pull_table_chunk_base, self.local_expert_bucket_count
        ]
        warp_mbarriers = ring_mbarriers + warp_idx * Int32(depth)
        ring = self._smem_workspace.tensor(self.refine_pull_ring_region, smem_base)

        gather_index = device_workspace.tensor(self.gather_index_region)
        # Rows leave for their landing slots, or for their routes' rows of the token pool.
        if cutlass.const_expr(self.fc1_token_pool):
            store_base = device_workspace.ptr(self.fc1_token_pool_region).toint()
        else:
            store_base = device_workspace.ptr(self.plain_activation_region).toint()
        # With a capacity, the rows of the token tiles past the launch's capacity
        # still leave for their landing slots (see "Pull dispatch").
        landing_base = store_base
        pool_capacity = None
        if cutlass.const_expr(self.fc1_token_pool_capacity):
            landing_base = device_workspace.ptr(self.plain_activation_region).toint()
            pool_capacity = Int32(self.fc1_token_pool_capacity)
            if cutlass.const_expr(self.fc1_token_pool_fallback == "launch"):
                route_total = chunk_table[
                    self.pull_table_route_base, self.local_expert_bucket_count
                ]
                pool_capacity = cutlass.select_(
                    route_total <= pool_capacity, pool_capacity, Int32(0)
                )
        data_base = activation.iterator.toint()
        sf_base = activation_sf.iterator.toint()
        canonical_words = cute.make_tensor(
            cute.recast_ptr(
                self.canonical_activation_sf_tensor(device_workspace).iterator,
                dtype=Uint32,
            ),
            cute.make_layout((self.worst_case_sf_rows * atoms,)),
        )
        # A claim's routes count into its expert, or with `pull_tile_ready` into
        # its token tile.
        ready_counts = device_workspace.ptr(self.fc1_sf_ready_region)
        if cutlass.const_expr(self.pull_tile_ready):
            ready_counts = device_workspace.ptr(self.fc1_tile_ready_region)
        landing_state = Int64(0)
        claimed_tag = Uint32(0)
        if cutlass.const_expr(self.pull_walk_dedup):
            landing_state = device_workspace.ptr(self.landing_state_region).toint()
            epoch = sentinel_ptx.ld_volatile_u32(
                device_workspace.ptr(self.sentinel_control_region).toint()
                + Int64(self.sentinel_epoch_word * 4)
            )
            claimed_tag = epoch * Uint32(2) + Uint32(1)
        landed_tag = claimed_tag + Uint32(1)

        sf_words = cute.make_rmem_tensor((sf_vectors * 4,), Uint32)
        chunk = Int32(cute.arch.shuffle_sync(claimed, Int32(0)))
        ring_position = Int32(0)
        expert = Int32(0)
        tile_slot = Int32(0)
        rows = Int32(0)
        sf_row = Int32(0)
        data_row = Int32(0)
        lane_landing = Int32(0)
        fetch_rows = Int32(0)
        fetch_landing = Int32(0)
        fetch_won = Int32(0)
        waits = cutlass.Boolean(False)
        if chunk < chunk_total:
            (
                expert,
                tile_slot,
                rows,
                sf_row,
                data_row,
                lane_landing,
                fetch_rows,
                fetch_landing,
                fetch_won,
                waits,
            ) = self._open_walk_claim(
                chunk,
                chunk_table,
                expert_table,
                gather_index,
                staging_peer_rank_ptr_mapper,
                sf_base,
                sf_words,
                landing_state,
                claimed_tag,
                data_base,
                ring,
                warp_idx,
                warp_mbarriers,
                ring_position,
                lane,
                pool_capacity,
            )
        while chunk < chunk_total:
            iket.range_push("pull.chunk")
            for row in cutlass.range(fetch_rows, unroll=1):
                position = ring_position + row
                slot = position % Int32(depth)
                cute.arch.mbarrier_wait(
                    warp_mbarriers + slot, (position // Int32(depth)) % Int32(2)
                )
                landing = Int32(cute.arch.shuffle_sync(fetch_landing, row))
                next_landing = Int32(
                    cute.arch.shuffle_sync(
                        fetch_landing, cutlass.min(row + Int32(depth - 1), Int32(31))
                    )
                )
                # A claim's routes sit on consecutive rows of the token pool, and
                # without ``pull_walk_dedup`` the warp fetches every one of them in
                # order; a claim whose token tile falls back to the landing slots
                # stores each row at its landing slot.
                store_row = landing
                row_base = store_base
                if cutlass.const_expr(self.fc1_token_pool):
                    store_row = data_row + row
                if cutlass.const_expr(self.fc1_token_pool_capacity):
                    in_pool = data_row >= Int32(0)
                    store_row = cutlass.select_(in_pool, data_row + row, landing)
                    row_base = cutlass.select_(in_pool, store_base, landing_base)
                if lane == Int32(0):
                    cp_async_bulk_s2g(
                        cute.make_ptr(
                            self.activation_dtype,
                            row_base + Int64(store_row) * Int64(data_row_bytes),
                            AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        ring[warp_idx, slot, None].iterator,
                        Int32(data_row_bytes),
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    if row + Int32(depth - 1) < fetch_rows:
                        # Every store but this one has read its slot, including the
                        # previous row's, whose slot the next fetch takes.
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                        next_slot = (position + Int32(depth - 1)) % Int32(depth)
                        self._issue_row_fetch(
                            staging_peer_rank_ptr_mapper,
                            data_base,
                            next_landing,
                            ring[warp_idx, next_slot, None].iterator,
                            warp_mbarriers + next_slot,
                            data_row_bytes,
                        )
            ring_position = ring_position + fetch_rows

            if lane == Int32(0):
                claimed = Int32(
                    cute.arch.atomic_add(
                        work_counter, Int32(1), sem="relaxed", scope="gpu"
                    )
                )
            if lane < rows:
                canonical_row = sf_row + lane
                block_row = canonical_row % Int32(self.sf_padding_block)
                atom_word = (
                    (canonical_row // Int32(self.sf_padding_block)) * Int32(atoms * 128)
                    + (block_row % Int32(32)) * Int32(4)
                    + block_row // Int32(32)
                )
                for atom in cutlass.range_constexpr(atoms):
                    canonical_words[atom_word + Int32(atom * 128)] = sf_words[atom]
            if lane == Int32(0):
                cute.arch.cp_async_bulk_wait_group(0)
                # The landing rows were written through the async proxy; the count
                # is a generic release the main kernel's TMA reads are ordered behind.
                sentinel_ptx.fence_proxy_async_global()
            # Orders every lane's scale-factor stores before the first lane's release.
            cute.arch.sync_warp()
            if cutlass.const_expr(self.pull_walk_dedup):
                # The first lane, whose wait and fence above cover every row this
                # warp stored, releases the slots it won before any lane waits for a
                # slot another warp won: a won slot holds exactly this round's
                # claimed tag, which one release add raises to the landed tag. A
                # waiting lane's acquire of the landed tag then makes the first
                # lane's release below cover the row whoever fetched it. A warp waits
                # only for slots claimed before its own claim, and the claimer of
                # such a slot releases it before waiting for anything, so every chain
                # of waits ends at a warp that waits for nothing.
                for row in cutlass.range(fetch_rows, unroll=1):
                    landing = Int32(cute.arch.shuffle_sync(fetch_landing, row))
                    won_row = Int32(cute.arch.shuffle_sync(fetch_won, row))
                    if (lane == Int32(0)) & (won_row != Int32(0)):
                        red_async_add_release_gpu_s32(
                            cute.make_ptr(
                                cutlass.Int32,
                                landing_state + Int64(landing) * Int64(4),
                                AddressSpace.gmem,
                            ),
                            Int32(1),
                        )
                if waits:
                    spin_wait(
                        cute.make_ptr(
                            Uint32,
                            landing_state + Int64(lane_landing) * Int64(4),
                            AddressSpace.gmem,
                        ),
                        lambda value: value >= landed_tag,
                        scope="gpu",
                    )
                cute.arch.sync_warp()
            if lane == Int32(0):
                ready_slot = expert
                if cutlass.const_expr(self.pull_tile_ready):
                    ready_slot = tile_slot
                red_async_add_release_gpu_s32(ready_counts + ready_slot, rows)
                if cutlass.const_expr(self.stamps):
                    self.stamp_latest(device_workspace, self.stamp_expert_ready, expert)
            chunk = Int32(cute.arch.shuffle_sync(claimed, Int32(0)))
            if chunk < chunk_total:
                (
                    expert,
                    tile_slot,
                    rows,
                    sf_row,
                    data_row,
                    lane_landing,
                    fetch_rows,
                    fetch_landing,
                    fetch_won,
                    waits,
                ) = self._open_walk_claim(
                    chunk,
                    chunk_table,
                    expert_table,
                    gather_index,
                    staging_peer_rank_ptr_mapper,
                    sf_base,
                    sf_words,
                    landing_state,
                    claimed_tag,
                    data_base,
                    ring,
                    warp_idx,
                    warp_mbarriers,
                    ring_position,
                    lane,
                    pool_capacity,
                )
            iket.range_pop()

    @cute.jit
    def _open_walk_claim(
        self,
        chunk: Int32,
        chunk_table: cute.Tensor,
        expert_table: cute.Tensor,
        gather_index: cute.Tensor,
        staging_peer_rank_ptr_mapper: SymmetricBufferDevice,
        sf_base: Int64,
        sf_words: cute.Tensor,
        landing_state: Int64,
        claimed_tag: Uint32,
        data_base: Int64,
        ring: cute.Tensor,
        warp_idx: Int32,
        warp_mbarriers: cute.Pointer,
        ring_position: Int32,
        lane: Int32,
        pool_capacity: Optional[Int32] = None,
    ):
        """Resolve a walk claim, issue its first fetches and pick the rest of the rows this warp fetches.

        Claims never cross a token tile: tile t of the expert holds its routes
        [t * tile_rows, (t + 1) * tile_rows) at the start of slot t. Lane ``i``
        loads the landing slot of the claim's ``i``-th route and that token's
        scale factors, and the claim's first routes, as many as the ring keeps
        in flight, are fetched at once. Under ``pull_walk_dedup`` every lane
        also claims its landing slot as the fetches leave; a later route's row is
        then fetched only if this warp won its slot, so the claim waits for no
        claim result before its first fetch, and a first route's row is stored
        even when another warp won the slot, with the same bytes.

        With ``fc1_token_pool_capacity`` the claim's tile loads from the token
        pool only if its routes end within ``pool_capacity``, the launch's
        capacity; the first row returned is then the claim's first route's row
        of the pool (see "Pull dispatch"), and -1 for a tile that falls back to
        the landing slots.

        Returns the claim's expert, its token tile among all experts' tiles in
        processing order (with ``pull_tile_ready``, else zero), its routes, its
        first scale-factor and first data row, this lane's landing slot, the
        rows to fetch with lane ``j`` holding the ``j``-th one's slot and whether
        this warp won it, and whether this lane waits for its row to land from
        another warp.
        """
        depth = self.pull_pipeline_depth
        tokens = Int32(self.max_tokens_per_rank)
        sf_row_bytes = (
            self.plain_activation_sf_row_elements
            * int(self.activation_sf_dtype.width)
            // 8
        )
        sf_vectors = self.hidden_sf * int(self.activation_sf_dtype.width) // 8 // 16
        expert = self._resolve_chunk_expert(chunk_table, chunk, lane)
        tile_rows = chunk_table[self.pull_table_tile_rows, expert]
        chunk_in_expert = chunk - chunk_table[self.pull_table_chunk_base, expert]
        chunk_rows = Int32(self.pull_chunk_rows)
        if cutlass.const_expr(self.pull_lead_chunk_rows != self.pull_chunk_rows):
            chunk_rows = chunk_table[self.pull_table_claim_rows, expert]
            chunks_per_tile = (tile_rows + chunk_rows - Int32(1)) // chunk_rows
        else:
            chunks_per_tile = (tile_rows + Int32(self.pull_chunk_rows - 1)) // Int32(
                self.pull_chunk_rows
            )
        tile = chunk_in_expert // chunks_per_tile
        first_in_tile = (chunk_in_expert - tile * chunks_per_tile) * chunk_rows
        tile_routes = cutlass.min(
            expert_table[self.refine_table_expert_size, expert] - tile * tile_rows,
            tile_rows,
        )
        rows = cutlass.min(tile_routes - first_in_tile, chunk_rows)
        tile_slot = Int32(0)
        if cutlass.const_expr(self.pull_tile_ready):
            tile_slot = chunk_table[self.pull_table_tile_base, expert] + tile
        pool_row = tile * Int32(self.token_slot_rows) + first_in_tile
        data_row = expert_table[self.refine_table_data_base, expert] + pool_row
        sf_row = expert_table[self.refine_table_sf_base, expert] + pool_row

        lane_landing = Int32(0)
        previous = Uint32(0)
        if lane < rows:
            lane_landing = gather_index[data_row + lane]
            if cutlass.const_expr(self.pull_walk_dedup):
                previous = sentinel_ptx.atom_max_relaxed_gpu_u32(
                    landing_state + Int64(lane_landing) * Int64(4), claimed_tag
                )
            source_rank = lane_landing // tokens
            sf_source = staging_peer_rank_ptr_mapper.map(
                sf_base
                + Int64(lane_landing - source_rank * tokens) * Int64(sf_row_bytes),
                source_rank,
            )
            for vector in cutlass.range_constexpr(sf_vectors):
                loaded = sentinel_ptx.ld_global_v4(sf_source + Int64(vector * 16))
                for word in cutlass.range_constexpr(4):
                    sf_words[vector * 4 + word] = loaded[word]
        first_rows = cutlass.min(rows, Int32(depth - 1))
        self._issue_walk_fetches(
            staging_peer_rank_ptr_mapper,
            data_base,
            ring,
            warp_idx,
            warp_mbarriers,
            lane_landing,
            first_rows,
            ring_position,
            lane,
        )
        fetch_rows = rows
        fetch_landing = lane_landing
        fetch_won = Int32(1)
        waits = cutlass.Boolean(False)
        if cutlass.const_expr(self.pull_walk_dedup):
            won = cutlass.Boolean(False)
            fetched = cutlass.Boolean(False)
            if lane < rows:
                won = previous < claimed_tag
                fetched = won | (lane < first_rows)
                waits = (previous >= claimed_tag) & (lane >= first_rows)
            fetched_lanes = Int32(cute.arch.vote_ballot_sync(fetched))
            fetch_rows = Int32(cute.arch.popc(fetched_lanes))
            # Lane j takes the j-th fetched lane's slot; the first rows keep their lanes.
            remaining = fetched_lanes
            for skipped in cutlass.range_constexpr(31):
                if Int32(skipped) < lane:
                    remaining = remaining & (remaining - Int32(1))
            source_lane = Int32(
                cute.arch.popc((remaining & (-remaining)) - Int32(1))
            ) & Int32(31)
            fetch_landing = Int32(cute.arch.shuffle_sync(lane_landing, source_lane))
            fetch_won = Int32(
                cute.arch.shuffle_sync(
                    cutlass.select_(won, Int32(1), Int32(0)), source_lane
                )
            )
        if cutlass.const_expr(self.fc1_token_pool_capacity):
            tile_route = (
                chunk_table[self.pull_table_route_base, expert] + tile * tile_rows
            )
            data_row = cutlass.select_(
                tile_route + tile_routes <= pool_capacity,
                tile_route + first_in_tile,
                Int32(-1),
            )
        return (
            expert,
            tile_slot,
            rows,
            sf_row,
            data_row,
            lane_landing,
            fetch_rows,
            fetch_landing,
            fetch_won,
            waits,
        )

    @cute.jit
    def _issue_walk_fetches(
        self,
        staging_peer_rank_ptr_mapper: SymmetricBufferDevice,
        data_base: Int64,
        ring: cute.Tensor,
        warp_idx: Int32,
        warp_mbarriers: cute.Pointer,
        fetch_landing: Int32,
        fetch_rows: Int32,
        ring_position: Int32,
        lane: Int32,
    ) -> None:
        """Issue a claim's first fetches, up to one less than the ring's slots, from ring position ``ring_position``."""
        depth = self.pull_pipeline_depth
        data_row_bytes = (
            self.plain_activation_row_elements * int(self.activation_dtype.width) // 8
        )
        for row in cutlass.range(cutlass.min(fetch_rows, Int32(depth - 1)), unroll=1):
            landing = Int32(cute.arch.shuffle_sync(fetch_landing, row))
            if lane == Int32(0):
                slot = (ring_position + row) % Int32(depth)
                self._issue_row_fetch(
                    staging_peer_rank_ptr_mapper,
                    data_base,
                    landing,
                    ring[warp_idx, slot, None].iterator,
                    warp_mbarriers + slot,
                    data_row_bytes,
                )

    @cute.jit
    def _stage_pull_chunk_table(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        expert_table: cute.Tensor,
        thread_idx: Int32,
    ) -> cute.Tensor:
        """Each expert's claims, routes per token tile and first claim in processing order, and the claim total.

        Claims follow the expert's token tiles: every tile but the last holds
        ``tile_rows`` routes (the tile N under dense slots, ``balanced_tile_rows``
        under strided ones) and takes ceil(tile_rows / ``pull_chunk_rows``)
        claims, the last tile ceil(rest / ``pull_chunk_rows``). The first warp
        scans the experts' claim counts in processing order, one position per lane
        and 32 positions a round, carrying the running total from round to round.
        Positions past the live experts carry no routes. An expert whose first
        route comes before ``pull_lead_routes`` in processing order claims
        ``pull_lead_chunk_rows`` routes at a time instead. With
        ``pull_tile_ready`` the same scan runs over the experts' token tiles,
        ceil(size / N) each as the main kernel's scheduler counts them, for each
        expert's first tile. With ``fc1_token_pool_capacity`` the routes before
        each expert in processing order are its first row of the token pool,
        and their total follows the last bucket.
        """
        chunk_table = self._smem_workspace.tensor(
            self.refine_pull_chunk_base_region, smem_base
        )
        expert_order = device_workspace.tensor(self.expert_order_region)
        buckets = self.local_expert_bucket_count
        chunk_rows = Int32(self.pull_chunk_rows)
        tile_n = Int32(self.token_tile_n)
        if thread_idx < Int32(32):
            lane_idx = thread_idx
            carried = Int32(0)
            carried_tiles = Int32(0)
            carried_routes = Int32(0)
            for position_round in cutlass.range_constexpr(ceil_div(buckets, 32)):
                position = Int32(position_round * 32) + lane_idx
                expert = Int32(0)
                chunks = Int32(0)
                tiles = Int32(0)
                tile_rows = tile_n
                expert_size = Int32(0)
                if position < Int32(buckets):
                    expert = expert_order[position]
                    expert_size = expert_table[self.refine_table_expert_size, expert]
                claim_rows = chunk_rows
                if cutlass.const_expr(
                    self.pull_lead_chunk_rows != self.pull_chunk_rows
                    or self.fc1_token_pool_capacity
                ):
                    route_end = carried_routes + _warp_inclusive_sum(
                        expert_size, lane_idx
                    )
                    carried_routes = Int32(cute.arch.shuffle_sync(route_end, Int32(31)))
                if cutlass.const_expr(
                    self.pull_lead_chunk_rows != self.pull_chunk_rows
                ):
                    claim_rows = cutlass.select_(
                        route_end - expert_size < Int32(self.pull_lead_routes),
                        Int32(self.pull_lead_chunk_rows),
                        chunk_rows,
                    )
                if position < Int32(buckets):
                    if cutlass.const_expr(self.token_tile_strided):
                        tile_rows = balanced_tile_rows(expert_size, self.token_tile_n)
                    full_tiles = cutlass.max(
                        (expert_size + tile_n - Int32(1)) // tile_n, Int32(1)
                    ) - Int32(1)
                    last_tile_rows = expert_size - full_tiles * tile_rows
                    chunks = (
                        full_tiles * ((tile_rows + claim_rows - Int32(1)) // claim_rows)
                        + (last_tile_rows + claim_rows - Int32(1)) // claim_rows
                    )
                    tiles = (expert_size + tile_n - Int32(1)) // tile_n
                chunk_end = carried + _warp_inclusive_sum(chunks, lane_idx)
                if position < Int32(buckets):
                    chunk_table[self.pull_table_chunk_base, expert] = chunk_end - chunks
                    chunk_table[self.pull_table_chunks, expert] = chunks
                    chunk_table[self.pull_table_tile_rows, expert] = tile_rows
                    if cutlass.const_expr(
                        self.pull_lead_chunk_rows != self.pull_chunk_rows
                    ):
                        chunk_table[self.pull_table_claim_rows, expert] = claim_rows
                    if cutlass.const_expr(self.fc1_token_pool_capacity):
                        chunk_table[self.pull_table_route_base, expert] = (
                            route_end - expert_size
                        )
                carried = Int32(cute.arch.shuffle_sync(chunk_end, Int32(31)))
                if cutlass.const_expr(self.pull_tile_ready):
                    tile_end = carried_tiles + _warp_inclusive_sum(tiles, lane_idx)
                    if position < Int32(buckets):
                        chunk_table[self.pull_table_tile_base, expert] = (
                            tile_end - tiles
                        )
                    carried_tiles = Int32(cute.arch.shuffle_sync(tile_end, Int32(31)))
            if lane_idx == Int32(0):
                chunk_table[self.pull_table_chunk_base, buckets] = carried
                if cutlass.const_expr(self.fc1_token_pool_capacity):
                    chunk_table[self.pull_table_route_base, buckets] = carried_routes
        cute.arch.sync_threads()
        return chunk_table

    @cute.jit
    def _resolve_chunk_expert(
        self, chunk_table: cute.Tensor, chunk: Int32, lane_idx: Int32
    ) -> Int32:
        """Which local expert owns a pull claim; see ``_resolve_block_expert``."""
        resolved = Int32(0)
        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, 32)
        ):
            bucket = Int32(bucket_round * 32) + lane_idx
            bucket_owns_chunk = cutlass.Boolean(False)
            if bucket < Int32(self.local_expert_bucket_count):
                bucket_base = chunk_table[self.pull_table_chunk_base, bucket]
                bucket_owns_chunk = (bucket_base <= chunk) & (
                    chunk < bucket_base + chunk_table[self.pull_table_chunks, bucket]
                )
            owning_lane = _first_matching_lane(bucket_owns_chunk)
            if owning_lane >= Int32(0):
                resolved = Int32(bucket_round * 32) + owning_lane
        return resolved

    @cute.jit
    def _issue_row_fetch(
        self,
        staging_peer_rank_ptr_mapper: SymmetricBufferDevice,
        data_base: Int64,
        landing: Int32,
        ring_slot: cute.Pointer,
        ring_mbarrier: cute.Pointer,
        data_row_bytes: int,
    ) -> None:
        """Bulk-load the staged data row behind landing slot ``landing`` from its source rank into a ring slot."""
        tokens = Int32(self.max_tokens_per_rank)
        source_rank = landing // tokens
        cute.arch.mbarrier_arrive_and_expect_tx(ring_mbarrier, Int32(data_row_bytes))
        tma_load_1d(
            ring_slot,
            staging_peer_rank_ptr_mapper.map(
                data_base
                + Int64(landing - source_rank * tokens) * Int64(data_row_bytes),
                source_rank,
            ),
            ring_mbarrier,
            Int32(data_row_bytes),
        )

    @cute.jit
    def run_pull_warp(
        self,
        device_workspace: DeviceWorkspace,
        lane: Int32,
        pull_warp: Int32,
        pull_warp_count: int,
        activation_address: Int64,
        activation_sf_address: Int64,
        staging_peer_rank_ptr_mapper: SymmetricBufferDevice,
    ) -> None:
        """Fetch routed rows from their source ranks as pull warp ``pull_warp`` of ``pull_warp_count`` in the grid.

        Every landing slot ``s * T + t`` of the round has a state word that
        claims and announces its row: a fetcher raises it to this round's claimed
        tag and, once the row's stores are issued and fenced, releases it to the
        landed tag (``_claim_landing``). Exactly one fetcher wins a slot, so a
        source token is fetched once whatever its routes to this rank, and the
        slot's location is known before the plan.

        Before the plan is complete (``_pull_before_plan``) the warp walks the
        received expert-id lists and fetches the rows they name.
        Then it takes claims of the work counter as ``_run_pull`` does: in
        processing order, at most ``pull_chunk_rows`` consecutive routes of one
        token tile of one expert each, resolved from a claim table in registers
        (``_build_pull_lane_table``). For each route the warp claims the
        landing slot and fetches the data row if no other warp has, and copies
        the route's scale factors into their canonical atoms either way
        (``_verify_block_rows`` has the atom layout). Rows go through registers,
        one at a time: every lane loads its 16-byte vectors out of the source
        rank's staging and stores them into the landing slot. The next claim is
        taken as a claim starts and resolved, with its gather-index load, while
        the claim's last route is in flight. Once a claim's routes have landed,
        whoever fetched them, every lane fences its stores toward the async
        proxy, which FC1 reads them through, and after the warp barrier the
        first lane adds the claim's routes to the expert's ``fc1_sf_ready``.

        A warp holds a claimed slot that has not landed only between its claim
        and its release, with no wait between them, and waits for slots others
        claimed only after that; so a slot's claimer never waits for the warp
        that waits for it, before the plan or after it.
        """
        tokens = Int32(self.max_tokens_per_rank)
        data_row_bytes = (
            self.plain_activation_row_elements * int(self.activation_dtype.width) // 8
        )
        sf_row_bytes = (
            self.plain_activation_sf_row_elements
            * int(self.activation_sf_dtype.width)
            // 8
        )
        row_vectors = data_row_bytes // 16
        vector_rounds = ceil_div(row_vectors, 32)
        atoms = self.refine_total_atoms
        word_rounds = ceil_div(atoms, 32)
        landing_base = device_workspace.ptr(self.plain_activation_region).toint()
        landing_state = device_workspace.ptr(self.landing_state_region).toint()
        epoch = sentinel_ptx.ld_volatile_u32(
            device_workspace.ptr(self.sentinel_control_region).toint()
            + Int64(self.sentinel_epoch_word * 4)
        )
        claimed_tag = epoch * Uint32(2) + Uint32(1)
        landed_tag = claimed_tag + Uint32(1)
        data = cute.make_rmem_tensor((vector_rounds * 4,), Uint32)
        words = cute.make_rmem_tensor((word_rounds,), Uint32)

        self._pull_before_plan(
            device_workspace,
            lane,
            pull_warp,
            pull_warp_count,
            activation_address,
            staging_peer_rank_ptr_mapper,
            landing_base,
            landing_state,
            claimed_tag,
            landed_tag,
            data,
        )

        # Every lane acquires the plan itself: its loads of the index lists and of
        # the helper's sizes, processing order and prefixes are ordered behind its
        # own acquires. The index-copy CTAs rank the experts themselves, so the
        # index lists may be complete before the helper has published.
        spin_wait(
            device_workspace.ptr(self.metadata_ready_region),
            lambda value: value == Int32(self.metadata_ready_target),
            scope="gpu",
        )
        spin_wait(
            device_workspace.ptr(self.prefix_ready_region),
            lambda value: value == Int32(1),
            scope="gpu",
        )
        table, chunk_total = self._build_pull_lane_table(device_workspace, lane)

        gather_index = device_workspace.tensor(self.gather_index_region)
        canonical_base = self.canonical_activation_sf_tensor(
            device_workspace
        ).iterator.toint()
        fc1_sf_ready = device_workspace.ptr(self.fc1_sf_ready_region)
        work_counter = device_workspace.ptr(self.refine_work_counter_region)

        claimed = Int32(0)
        if lane == Int32(0):
            claimed = Int32(
                cute.arch.atomic_add(work_counter, Int32(1), sem="relaxed", scope="gpu")
            )
        chunk = Int32(cute.arch.shuffle_sync(claimed, Int32(0)))
        expert = Int32(0)
        rows = Int32(0)
        sf_row = Int32(0)
        lane_landing = Int32(0)
        if chunk < chunk_total:
            expert, rows, sf_row, lane_landing = self._open_pull_claim(
                device_workspace, table, chunk, lane, gather_index
            )
        while chunk < chunk_total:
            iket.range_push("pull_warp.claim")
            if lane == Int32(0):
                claimed = Int32(
                    cute.arch.atomic_add(
                        work_counter, Int32(1), sem="relaxed", scope="gpu"
                    )
                )
            next_chunk = chunk_total
            next_expert = expert
            next_rows = rows
            next_sf_row = sf_row
            next_landing = lane_landing
            fetched = Int32(0)
            for route in cutlass.range(rows, unroll=1):
                landing = Int32(cute.arch.shuffle_sync(lane_landing, route))
                source_rank = landing // tokens
                token = landing - source_rank * tokens
                sf_source = staging_peer_rank_ptr_mapper.map(
                    activation_sf_address + Int64(token) * Int64(sf_row_bytes),
                    source_rank,
                )
                for word_round in cutlass.range_constexpr(word_rounds):
                    word = Int32(word_round * 32) + lane
                    if word < Int32(atoms):
                        words[word_round] = sentinel_ptx.ld_global_u32_no_allocate(
                            sf_source + Int64(word) * Int64(4)
                        )
                fetches = self._claim_landing(landing_state, landing, lane, claimed_tag)
                if fetches:
                    fetched = fetched | (Int32(1) << route)
                    self._load_staged_row(
                        staging_peer_rank_ptr_mapper,
                        activation_address,
                        source_rank,
                        token,
                        lane,
                        data,
                    )
                if route == rows - Int32(1):
                    next_chunk = Int32(cute.arch.shuffle_sync(claimed, Int32(0)))
                    if next_chunk < chunk_total:
                        next_expert, next_rows, next_sf_row, next_landing = (
                            self._open_pull_claim(
                                device_workspace, table, next_chunk, lane, gather_index
                            )
                        )
                canonical_row = sf_row + route
                block_row = canonical_row % Int32(self.sf_padding_block)
                atom_word = (
                    (canonical_row // Int32(self.sf_padding_block)) * Int32(atoms * 128)
                    + (block_row % Int32(32)) * Int32(4)
                    + block_row // Int32(32)
                )
                for word_round in cutlass.range_constexpr(word_rounds):
                    word = Int32(word_round * 32) + lane
                    if word < Int32(atoms):
                        sentinel_ptx.st_global_u32(
                            canonical_base
                            + Int64(atom_word + word * Int32(128)) * Int64(4),
                            words[word_round],
                        )
                if fetches:
                    self._store_landed_row(
                        landing_base, landing_state, landing, lane, landed_tag, data
                    )
            # Rows another warp claimed land without that warp waiting on anything:
            # a claimed slot is fetched, stored and released before its claimer
            # waits here. The first lane acquires each one's landed tag, so its
            # release of the claim below covers the row whoever fetched it.
            for route in cutlass.range(rows, unroll=1):
                landing = Int32(cute.arch.shuffle_sync(lane_landing, route))
                if (lane == Int32(0)) & ((fetched >> route) & Int32(1) == Int32(0)):
                    spin_wait(
                        cute.make_ptr(
                            Uint32,
                            landing_state + Int64(landing) * Int64(4),
                            AddressSpace.gmem,
                        ),
                        lambda value: value >= landed_tag,
                        scope="gpu",
                    )
            # The atoms were written through the generic proxy and FC1 reads them
            # through the async proxy; the warp barrier then orders every lane's
            # stores and fence before the first lane's release.
            sentinel_ptx.fence_proxy_async_global()
            cute.arch.sync_warp()
            if lane == Int32(0):
                red_async_add_release_gpu_s32(fc1_sf_ready + expert, rows)
                if cutlass.const_expr(self.stamps):
                    self.stamp_latest(device_workspace, self.stamp_expert_ready, expert)
            chunk = next_chunk
            expert = next_expert
            rows = next_rows
            sf_row = next_sf_row
            lane_landing = next_landing
            iket.range_pop()

    @cute.jit
    def _pull_before_plan(
        self,
        device_workspace: DeviceWorkspace,
        lane: Int32,
        pull_warp: Int32,
        pull_warp_count: int,
        activation_address: Int64,
        staging_peer_rank_ptr_mapper: SymmetricBufferDevice,
        landing_base: Int64,
        landing_state: Int64,
        claimed_tag: Uint32,
        landed_tag: Uint32,
        data: cute.Tensor,
    ) -> None:
        """Fetch rows routed to this rank while the plan is still being built.

        The links are otherwise idle until the index lists are out, and a row's
        landing slot depends only on its source rank and token. Every route in
        the received expert-id lists is routed here, so the warps walk the lists
        (see "Expert-id lists"): warp ``g`` takes list ``g`` modulo the W * X
        (source, chunk) lists and, with the other warps on that list, its
        positions in turn, 32 at a time. Every lane first acquires every
        source's ``topk_ready`` once all its exchange CTAs have released it
        (``_acquire_topk_ready``), which orders the lane's loads of count rows,
        list words and staged rows, here and after the plan, behind the
        sources' writes. The count row
        gives the list's length; each position's route names the source token,
        whose row the warp fetches unless its landing slot is claimed. The
        warp stops once ``metadata_ready`` is complete, so the claims of the
        processing order follow at once. Rows are fetched in list order (by
        local expert, in index order), not in the order experts become ready:
        this phase only uses time before the plan.

        The index-copy CTAs reset every list word they read, and the helper the
        count rows once the plan is complete, all while this phase may still
        read them. A word that reads as the sentinel after the acquire has been
        read by its copier and is skipped, and a count row with a sentinel word
        reads as empty: a skipped route is fetched by the claims after the plan.
        """
        world = self.world_size
        chunks = self.exchange_cta_count
        buckets = self.local_expert_bucket_count
        lists = world * chunks
        source_list = pull_warp % Int32(lists)
        source_rank = source_list // Int32(chunks)
        # This warp's rank among the warps on its list, and how many there are.
        walker = pull_warp // Int32(lists)
        walkers = (Int32(pull_warp_count - 1) - source_list) // Int32(lists) + Int32(1)
        sentinel = Uint32(sentinel_ptx.WORD_SENTINEL)

        self._acquire_topk_ready(device_workspace)

        count_row = device_workspace.ptr(self.topk_count_inbox_region).toint() + Int64(
            source_list * Int32(buckets)
        ) * Int64(4)
        length = Int32(0)
        reset = Int32(0)
        for bucket_round in cutlass.range_constexpr(ceil_div(buckets, 32)):
            bucket = Int32(bucket_round * 32) + lane
            if bucket < Int32(buckets):
                count = sentinel_ptx.ld_volatile_u32(
                    count_row + Int64(bucket) * Int64(4)
                )
                if count == sentinel:
                    reset = Int32(1)
                else:
                    length = length + Int32(count)
        for step_log in cutlass.range_constexpr(5):
            length = length + Int32(
                cute.arch.shuffle_sync_bfly(length, offset=1 << step_log)
            )
        if Int32(cute.arch.vote_ballot_sync(reset != Int32(0))) != Int32(0):
            length = Int32(0)

        list_words = device_workspace.ptr(self.topk_index_inbox_region).toint() + Int64(
            source_list * Int32(self.exchange_chunk_routes)
        ) * Int64(4)
        metadata_ready = device_workspace.ptr(self.metadata_ready_region).toint()
        landing_row_base = source_rank * Int32(self.max_tokens_per_rank)
        group = walker
        open_items = Int32(1)
        while (group * Int32(32) < length) & (open_items != Int32(0)):
            position = group * Int32(32) + lane
            route = sentinel
            if position < length:
                route = sentinel_ptx.ld_volatile_u32(
                    list_words + Int64(position) * Int64(4)
                )
            pending = Int32(cute.arch.vote_ballot_sync(route != sentinel))
            while (pending != Int32(0)) & (open_items != Int32(0)):
                lowest = pending & (-pending)
                owner = Int32(cute.arch.popc(lowest - Int32(1)))
                pending = pending - lowest
                token = Int32(cute.arch.shuffle_sync(Int32(route), owner)) // Int32(
                    self.topk
                )
                landing = landing_row_base + token
                if self._claim_landing(landing_state, landing, lane, claimed_tag):
                    self._load_staged_row(
                        staging_peer_rank_ptr_mapper,
                        activation_address,
                        source_rank,
                        token,
                        lane,
                        data,
                    )
                    self._store_landed_row(
                        landing_base, landing_state, landing, lane, landed_tag, data
                    )
                observed = Int32(0)
                if lane == Int32(0):
                    observed = Int32(sentinel_ptx.ld_volatile_u32(metadata_ready))
                open_items = Int32(
                    Int32(cute.arch.shuffle_sync(observed, Int32(0)))
                    != Int32(self.metadata_ready_target)
                )
            group = group + walkers

    @cute.jit
    def _claim_landing(
        self, landing_state: Int64, landing: Int32, lane: Int32, claimed_tag: Uint32
    ) -> cutlass.Boolean:
        """Whether this warp fetches landing slot ``landing``: true for the first claim of the round."""
        previous = Uint32(0)
        if lane == Int32(0):
            previous = sentinel_ptx.atom_max_relaxed_gpu_u32(
                landing_state + Int64(landing) * Int64(4), claimed_tag
            )
        return Uint32(cute.arch.shuffle_sync(Int32(previous), Int32(0))) < claimed_tag

    @cute.jit
    def _load_staged_row(
        self,
        staging_peer_rank_ptr_mapper: SymmetricBufferDevice,
        activation_address: Int64,
        source_rank: Int32,
        token: Int32,
        lane: Int32,
        data: cute.Tensor,
    ) -> None:
        """Load this lane's 16-byte vectors of a staged data row of ``source_rank`` into ``data``."""
        data_row_bytes = (
            self.plain_activation_row_elements * int(self.activation_dtype.width) // 8
        )
        row_vectors = data_row_bytes // 16
        data_source = staging_peer_rank_ptr_mapper.map(
            activation_address + Int64(token) * Int64(data_row_bytes), source_rank
        )
        for vector_round in cutlass.range_constexpr(ceil_div(row_vectors, 32)):
            vector = Int32(vector_round * 32) + lane
            if vector < Int32(row_vectors):
                loaded = sentinel_ptx.ld_global_v4_no_allocate(
                    data_source + Int64(vector) * Int64(16)
                )
                for word in cutlass.range_constexpr(4):
                    data[vector_round * 4 + word] = loaded[word]

    @cute.jit
    def _store_landed_row(
        self,
        landing_base: Int64,
        landing_state: Int64,
        landing: Int32,
        lane: Int32,
        landed_tag: Uint32,
        data: cute.Tensor,
    ) -> None:
        """Store ``data`` into landing slot ``landing`` and release the slot's landed tag.

        Every lane fences its stores toward the async proxy, which FC1 reads the
        row through, and the warp barrier orders them before the first lane's
        release.
        """
        data_row_bytes = (
            self.plain_activation_row_elements * int(self.activation_dtype.width) // 8
        )
        row_vectors = data_row_bytes // 16
        landing_row = landing_base + Int64(landing) * Int64(data_row_bytes)
        for vector_round in cutlass.range_constexpr(ceil_div(row_vectors, 32)):
            vector = Int32(vector_round * 32) + lane
            if vector < Int32(row_vectors):
                sentinel_ptx.st_global_v4_words(
                    landing_row + Int64(vector) * Int64(16),
                    data[vector_round * 4],
                    data[vector_round * 4 + 1],
                    data[vector_round * 4 + 2],
                    data[vector_round * 4 + 3],
                )
        sentinel_ptx.fence_proxy_async_global()
        cute.arch.sync_warp()
        if lane == Int32(0):
            sentinel_ptx.red_release_gpu_max_u32(
                landing_state + Int64(landing) * Int64(4), landed_tag
            )

    @cute.jit
    def _build_pull_lane_table(self, device_workspace: DeviceWorkspace, lane: Int32):
        """The claim table of ``_stage_pull_chunk_table`` in the registers of one warp, and the claim total.

        Lane ``l`` holds position ``32 r + l`` of the processing order in position
        round ``r``: its expert, its first and one-past-last claim in processing
        order and its routes. The claims are counted as in
        ``_stage_pull_chunk_table`` and scanned across the warp, carrying the
        running total from round to round.
        """
        expert_order = device_workspace.tensor(self.expert_order_region)
        expert_sizes = device_workspace.tensor(self.expert_sizes_region)
        buckets = self.local_expert_bucket_count
        fields = self.pull_lane_fields
        chunk_rows = Int32(self.pull_chunk_rows)
        tile_n = Int32(self.token_tile_n)
        table = cute.make_rmem_tensor((fields * ceil_div(buckets, 32),), Int32)
        carried = Int32(0)
        for position_round in cutlass.range_constexpr(ceil_div(buckets, 32)):
            position = Int32(position_round * 32) + lane
            expert = Int32(0)
            expert_size = Int32(0)
            chunks = Int32(0)
            if position < Int32(buckets):
                expert = expert_order[position]
                if expert < Int32(self.experts_per_rank):
                    expert_size = expert_sizes[expert]
                tile_rows = self._pull_tile_rows(expert_size)
                full_tiles = cutlass.max(
                    (expert_size + tile_n - Int32(1)) // tile_n, Int32(1)
                ) - Int32(1)
                last_tile_rows = expert_size - full_tiles * tile_rows
                chunks = (
                    full_tiles * ((tile_rows + chunk_rows - Int32(1)) // chunk_rows)
                    + (last_tile_rows + chunk_rows - Int32(1)) // chunk_rows
                )
            chunk_end = carried + _warp_inclusive_sum(chunks, lane)
            base = position_round * fields
            table[base + self.pull_lane_expert] = expert
            table[base + self.pull_lane_first_claim] = chunk_end - chunks
            table[base + self.pull_lane_end_claim] = chunk_end
            table[base + self.pull_lane_size] = expert_size
            carried = Int32(cute.arch.shuffle_sync(chunk_end, Int32(31)))
        return table, carried

    @cute.jit
    def _pull_tile_rows(self, expert_size: Int32) -> Int32:
        """Routes in each of an expert's token tiles but the last (see "Row spaces")."""
        tile_rows = Int32(self.token_tile_n)
        if cutlass.const_expr(self.token_tile_strided):
            tile_rows = balanced_tile_rows(expert_size, self.token_tile_n)
        return tile_rows

    @cute.jit
    def _open_pull_claim(
        self,
        device_workspace: DeviceWorkspace,
        table: cute.Tensor,
        chunk: Int32,
        lane: Int32,
        gather_index: cute.Tensor,
    ):
        """Resolve a pull claim from the register table: its expert, routes, first SF row and each lane's landing slot.

        Positions without routes own no claim, so exactly one lane of one round
        owns a claim below the total. The expert's row-space bases come from GMEM.
        Lane ``i`` below the claim's route count loads the landing slot of its
        ``i``-th route.
        """
        fields = self.pull_lane_fields
        chunk_rows = Int32(self.pull_chunk_rows)
        owner_lane = Int32(0)
        owner_round = Int32(0)
        for position_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, 32)
        ):
            base = position_round * fields
            owns = (table[base + self.pull_lane_first_claim] <= chunk) & (
                chunk < table[base + self.pull_lane_end_claim]
            )
            lane_owning = _first_matching_lane(owns)
            if lane_owning >= Int32(0):
                owner_lane = lane_owning
                owner_round = Int32(position_round)
        claim = cute.make_rmem_tensor((fields,), Int32)
        for position_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, 32)
        ):
            for field in cutlass.range_constexpr(fields):
                value = Int32(
                    cute.arch.shuffle_sync(
                        table[position_round * fields + field], owner_lane
                    )
                )
                if owner_round == Int32(position_round):
                    claim[field] = value
        expert = claim[self.pull_lane_expert]
        data_base = device_workspace.tensor(self.data_expert_base_region)[expert]
        sf_base = device_workspace.tensor(self.sf_expert_base_region)[expert]
        # Claims never cross a token tile: tile t of the expert holds its routes
        # [t * tile_rows, (t + 1) * tile_rows) at the start of slot t.
        tile_rows = self._pull_tile_rows(claim[self.pull_lane_size])
        chunks_per_tile = (tile_rows + chunk_rows - Int32(1)) // chunk_rows
        chunk_in_expert = chunk - claim[self.pull_lane_first_claim]
        tile = chunk_in_expert // chunks_per_tile
        first_in_tile = (chunk_in_expert - tile * chunks_per_tile) * chunk_rows
        tile_routes = cutlass.min(
            claim[self.pull_lane_size] - tile * tile_rows, tile_rows
        )
        rows = cutlass.min(tile_routes - first_in_tile, chunk_rows)
        pool_row = tile * Int32(self.token_slot_rows) + first_in_tile
        lane_landing = Int32(0)
        if lane < rows:
            lane_landing = gather_index[data_base + pool_row + lane]
        return expert, rows, sf_base + pool_row, lane_landing

    @cute.jit
    def _verify_block_rows(
        self,
        device_workspace: DeviceWorkspace,
        gather_index: cute.Tensor,
        data_row_begin: Int32,
        live_rows: Int32,
        atom_stage: cute.Pointer,
        thread_in_group: Int32,
    ) -> None:
        """Wait until every landing byte of the block's live rows has arrived, and
        write the rows' scale factors as canonical atoms.

        Landing rows are armed with 0xFF and the pushers never send a 0xFF byte, so
        a row has fully arrived once none of its data or scale bytes is 0xFF. A
        byte is written atomically by any writer, which is what lets the pushers
        use bulk copies. Each warp checks whole rows, ``verify_rows_in_flight`` at
        a time: every lane issues all of its vector loads before testing any of
        them, then reloads only the vectors still holding the sentinel. The
        warp's landing rows come out of ``gather_index`` once, one per lane,
        before the first round, so a round waits on its landing loads alone.

        Each landing byte leaves 0xFF exactly once, for its final value, so the
        scale-factor vectors held once a round's rows pass are the rows' final
        scale factors. A row's four scale factors of atom ``a`` are one word, word
        ``t`` (the row's canonical position, see ``_stage_block_rows``) of the
        atom's 512 bytes, so the lane holding a vector writes its four words into
        the four atoms the vector spans, at ``atom_stage``: this group's output
        slots, one 512-byte atom after another in row order. The caller's proxy
        fence and group barrier publish them.
        """
        rows_in_flight = self.verify_rows_in_flight
        warps = self.refine_warps_per_group
        data_row_bytes = (
            self.plain_activation_row_elements * int(self.activation_dtype.width) // 8
        )
        sf_row_bytes = (
            self.plain_activation_sf_row_elements
            * int(self.activation_sf_dtype.width)
            // 8
        )
        data_vectors = self.hidden_size * int(self.activation_dtype.width) // 8 // 16
        sf_vectors = self.hidden_sf * int(self.activation_sf_dtype.width) // 8 // 16
        if cutlass.const_expr(data_vectors % 32 != 0 or sf_vectors > 32):
            raise ValueError(
                "Landing verification needs whole warps of data vectors and at most 32 scale vectors."
            )
        data_per_lane = data_vectors // 32
        slots = data_per_lane + 1
        warp = thread_in_group // Int32(32)
        lane = thread_in_group % Int32(32)
        data_base = device_workspace.ptr(self.plain_activation_region).toint()
        sf_base = device_workspace.ptr(self.plain_activation_sf_region).toint()
        # One word per (atom, row): 128 words an atom, a 16-byte vector spans four atoms.
        atom_words = cute.make_tensor(
            cute.recast_ptr(atom_stage, dtype=Uint32),
            cute.make_layout((self.refine_total_atoms * 128,)),
        )

        if cutlass.const_expr(self.sf_padding_block > 32 * warps):
            raise ValueError(
                "Landing verification keeps a warp's landing rows of a block one per lane."
            )

        # Lane i holds the landing row of the warp's i-th row of the block.
        warp_landing_rows = Int32(0)
        lane_row = warp + lane * Int32(warps)
        if lane_row < live_rows:
            warp_landing_rows = gather_index[data_row_begin + lane_row]

        words = cute.make_rmem_tensor((rows_in_flight * slots * 4,), Uint32)
        # Per row slot, this lane's first data vector and its scale vector of the
        # slot's landing row; vector v of the data is 512 bytes past the first. A
        # slot or a lane without a vector holds 0 and loads nothing.
        data_addresses = cute.make_rmem_tensor((rows_in_flight,), Int64)
        sf_addresses = cute.make_rmem_tensor((rows_in_flight,), Int64)
        rows_for_warp = cutlass.max(
            live_rows - warp + Int32(warps - 1), Int32(0)
        ) // Int32(warps)
        rounds = (rows_for_warp + Int32(rows_in_flight - 1)) // Int32(rows_in_flight)
        for round_idx in cutlass.range(rounds, unroll=1):
            for row_slot in cutlass.range_constexpr(rows_in_flight):
                warp_row = round_idx * Int32(rows_in_flight) + Int32(row_slot)
                # Every lane takes part in the shuffle, whether or not the slot has a row.
                landing = Int64(
                    cute.arch.shuffle_sync(
                        warp_landing_rows, cutlass.min(warp_row, Int32(31))
                    )
                )
                data_addresses[row_slot] = Int64(0)
                sf_addresses[row_slot] = Int64(0)
                if warp + warp_row * Int32(warps) < live_rows:
                    data_addresses[row_slot] = (
                        data_base
                        + landing * Int64(data_row_bytes)
                        + Int64(lane) * Int64(16)
                    )
                    if lane < Int32(sf_vectors):
                        sf_addresses[row_slot] = (
                            sf_base
                            + landing * Int64(sf_row_bytes)
                            + Int64(lane) * Int64(16)
                        )
            # A slot that loads nothing holds zeros, which never read as a sentinel.
            for row_slot in cutlass.range_constexpr(rows_in_flight):
                for vector in cutlass.range_constexpr(slots):
                    slot = row_slot * slots + vector
                    for word in cutlass.range_constexpr(4):
                        words[slot * 4 + word] = Uint32(0)
                    row_vectors = sf_addresses[row_slot]
                    if cutlass.const_expr(vector < data_per_lane):
                        row_vectors = data_addresses[row_slot]
                    if row_vectors != Int64(0):
                        loaded = sentinel_ptx.ld_volatile_v4(
                            row_vectors + Int64(vector % data_per_lane * 512)
                        )
                        for word in cutlass.range_constexpr(4):
                            words[slot * 4 + word] = loaded[word]
            pending = Int32(1)
            sweeps = Int32(0)
            while pending != Int32(0):
                pending = Int32(0)
                for row_slot in cutlass.range_constexpr(rows_in_flight):
                    for vector in cutlass.range_constexpr(slots):
                        slot = row_slot * slots + vector
                        row_vectors = sf_addresses[row_slot]
                        if cutlass.const_expr(vector < data_per_lane):
                            row_vectors = data_addresses[row_slot]
                        missing = (
                            sentinel_ptx.has_ff_byte(words[slot * 4])
                            | sentinel_ptx.has_ff_byte(words[slot * 4 + 1])
                            | sentinel_ptx.has_ff_byte(words[slot * 4 + 2])
                            | sentinel_ptx.has_ff_byte(words[slot * 4 + 3])
                        )
                        if missing:
                            loaded = sentinel_ptx.ld_volatile_v4(
                                row_vectors + Int64(vector % data_per_lane * 512)
                            )
                            for word in cutlass.range_constexpr(4):
                                words[slot * 4 + word] = loaded[word]
                            pending = Int32(1)
                sweeps += Int32(1)
                if sweeps == Int32(sentinel_ptx.POLL_SWEEP_LIMIT):
                    cute.printf(
                        "sentinel dispatch: landing rows of block at pool row %d never arrived\n",
                        data_row_begin,
                    )
                    sentinel_ptx.trap()
            if lane < Int32(sf_vectors):
                for row_slot in cutlass.range_constexpr(rows_in_flight):
                    row = warp + (
                        round_idx * Int32(rows_in_flight) + Int32(row_slot)
                    ) * Int32(warps)
                    sf_slot = row_slot * slots + data_per_lane
                    if row < live_rows:
                        canonical_row = Int32(4) * (row % Int32(32)) + row // Int32(32)
                        for word in cutlass.range_constexpr(4):
                            atom = lane * Int32(4) + Int32(word)
                            atom_words[atom * Int32(128) + canonical_row] = words[
                                sf_slot * 4 + word
                            ]

    @cute.jit
    def _scatter_block_atoms(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        sf_row_begin: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
    ) -> None:
        """Store a block's canonical atoms from the group's output stage.

        The verification has already written every run into its own output
        slot and published them, so the bulk stores leave back to back in one
        bulk group, which the caller drains once before it releases the block.
        One elected thread of the group's first warp stores each slot whole, and
        only a run's live atoms: a partial last run's other atoms have no place
        in the pool.
        """
        output = self._smem_workspace.tensor(self.refine_output_region, smem_base)
        canonical = self.canonical_activation_sf_tensor(device_workspace)
        sf_vec_size = self.quant_kind.sf_vec_size
        columns_per_run = self.refine_run_columns

        if thread_in_group // Int32(32) == Int32(0):
            with cute.arch.elect_one():
                for run in cutlass.range_constexpr(self.refine_run_count):
                    live_atoms = min(
                        self.refine_atoms_per_stage,
                        self.refine_total_atoms - run * self.refine_atoms_per_stage,
                    )
                    destination = cute.domain_offset(
                        (sf_row_begin, Int32(run * columns_per_run * sf_vec_size)),
                        canonical,
                    )
                    cp_async_bulk_s2g(
                        destination.iterator,
                        output[group_idx, run, None].iterator,
                        Int32(live_atoms * 512),
                    )
        cute.arch.cp_async_bulk_commit_group()

    # ------------------------------------------------------------------
    # DSL plumbing
    # ------------------------------------------------------------------

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "GenphaseTokenComm":
        if values:
            raise ValueError("GenphaseTokenComm carries no MLIR values.")
        return self


__all__ = ["GenphaseTokenComm", "GenphaseTokenCommArgs"]
