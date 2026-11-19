# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Generation-phase MegaMoE kernel composition for Blackwell.

Specialised for few tokens per rank and large expert parallelism. The narrowing
buys a short communication path: ``GenphaseTokenComm`` is a separate launch that
fires ``griddepcontrol_launch_dependents`` at entry, so this kernel becomes
eligible immediately and streams weights while the payload is still on the wire.

Three consequences of that split shape the kernel.

Readiness is whole-rank. The communication component publishes four counters for
its entire grid rather than one per token tile, so each consumer warp waits once
before its work loop instead of per tile, and the FC12 extension runs with no FC1
ready counter at all.

There are no transfer warps into an expert pool: FC1 gathers straight from the
dense received plane, and the FC2 epilogue warps push each route's partial into
its owning rank's combine slot themselves. Twelve warps cover every role. Under
the pull dispatch into the token pool (``fc1_token_pool``) the communication
grid's walk itself stores each route's row at its expert-major pool row, and
FC1 loads its token tiles from there by tiled TMA, as FC2 does; the Gather-B
warps then idle. A pool of ``fc1_token_pool_capacity`` rows holds a launch's
token tiles only up to that many routes in processing order: each launch plans
the pool from its expert sizes once they are final (``_plan_token_pool``), and
every token tile past it falls back to the landing slots, which stay allocated,
and to the Gather-B warps. Under
the pull dispatch a fourth warpgroup can fetch the routed rows from their source
ranks into that plane instead of walking CTAs of the communication grid, which
then ends with the plan and holds no SM through the fetch.

The end-of-round reduce owns the workspace tail reset. This kernel triggers its
dependents at entry, so the reduce launches on SMs as CTAs here retire and its
poller CTAs poll their combine slots while FC2 is still running; its
housekeeping CTAs wait for this grid, which has itself drained the
communication grid, before they reset landing rows and clear the local counters
this round used. No cross-rank barrier is needed: every peer-written region is a
landing set whose bytes announce their own arrival.

Under the counter combine the top-k reduce can run in this kernel instead
(``reduce_topk_in_kernel``): the FC2 epilogue adds each route's BF16 partial
straight into its source token's output row with system-scope reductions (bulk
reduce-adds, or warp-wide vector reductions where the pushes are warp stores).
The output lives in the source rank's staging allocation, which peers reach
through its own offset table. The epilogue warps zero it at entry and publish it
to every rank, and every epilogue acquires all ranks' zeroed outputs before its
first tile (``_zero_output``). The tail barrier then publishes the sums, and the
end-of-round kernel only keeps house.

Under the Sentinel dispatch at small capacities work claims can be just in time.
The clusters whose SMs the communication grid holds start only when it exits,
shortly before the first FC1 tile can, and a scheduler warp that claims as soon
as its queue has room holds several tiles per cluster by then. When a launch has
more tiles than those early claims cover, the late clusters find every FC1 tile
taken and the last experts' FC2 tiles pile up on them. A just-in-time cluster
instead claims its next tile only once the current one has passed its readiness
gate and has ``claim_lead_k_tiles`` K tiles of weights left to issue: it holds
one tile until its first FC1 tile starts, and a tile it cannot start yet never
pulls another one behind it. Each launch picks its claim timing at sizes_ready
from its tile counts. Claims are just in time when the early claims cannot hold
every tile or when the late clusters save a round of FC1 tiles. Otherwise the
clusters that start with the kernel hold all the work at no more FC1 tiles each
than all clusters would, and late clusters taking FC1 tiles would only slow
every SM's weight stream, so claims stay eager.

The same claims serve the Fence dispatch at large capacities. With eager claims
every cluster holds several tiles beyond the one it runs: the last tiles are
bound to clusters before it is known which ones free up first, and an FC2 tile
can wait at the head of its cluster's queue for FC1 tiles that a later claim
would have found finished. Just-in-time claims hand each tile to a cluster that
is about to run it.

The token tile a launch runs follows its routing. A strided plan (N192, N240)
sizes its tiles for the largest expert the capacity allows, but an expert of at
most 128 routes per tile reaches only instruction N 128, and the wider tile's
token stages then cost the AB ring depth the MMA needs at that N. The kernel can
therefore carry a second configuration of its mainloop and epilogue at N128
(``alternate_tile_n``) that runs the same plan: the same tiles, rows and slots.
Each launch picks one once the plan's expert sizes are final, which the first
FC1 tile waits for anyway (``_choose_token_tile``), and every warp role then
runs the chosen configuration's body. The two SMEM plans overlay one allocation,
so each configuration initializes its own barriers and TMEM after the choice;
only the entry weight prefetch leaves before it.

A shared expert (``shared_expert="fp8_in_kernel"``: one MXFP8 E4M3 expert every
token of the rank runs, with the routed experts' intermediate width) is a third
weight stream through the same grouped GEMMs (``SharedExpertTiles``): its tiles
run on the same CTAs, stages, MMA warp and epilogue as the routed ones, with
their own weight descriptors, the E4M3 A format and an E4M3 stage's bytes. Its
FC1 tiles take the front of every claim stream and load their tokens by tiled
TMA from the rank's staged input, which is complete when this kernel starts, so
they wait on no dispatch; its FC2 tiles follow the routed tiles
(``shared_fc2_placement``). Its scale factors, FC1 outputs, FC2 gates,
routing records and top-k scores use pool rows past the routed worst case; the
records send every token's FC2 row to this rank's own combine slot ``topk`` with
weight one, which the top-k reduce adds after the routes (or, under the
in-kernel reduce, the FC2 epilogue adds into the output like a route's). The
shared expert runs over the whole capacity, since the kernel never learns the
staged token count; rows past it are never read.

With ``shared_expert_entry`` the shared tiles do not wait for the plan: its FC1
tiles (and its FC2 tiles if they are placed early) form a head of their own work
counter, which every cluster drains from kernel entry, one tile at a time
through the claim gate, before it claims a routed tile. Their operands are all
ready at entry: the staged input and its scale factors precede the
communication grid in stream order, which launches without a programmatic
dependency and so starts only once staging completed, and this grid launches
only once every communication CTA has started; the pools, gates, records and
combine slot they use are this rank's own, reset by the previous round's
reduce before that communication grid started. So the warps that wait for the
dispatch wait lazily, at their first routed tile: the scheduler for the plan's
sizes (and then plans the token pool), the epilogue for the index lists (after
it has allocated TMEM, which every MMA waits for), the Gather-B warps for the
index lists and the payload, and under the in-kernel reduce the epilogue for
every rank's zeroed output at its first FC2 tile. The shared FC1 tiles thus run
in the window in which the routed FC1 tiles wait for the dispatch.

The shared token tiles run in the launch's token tile configuration, so a head
whose tiles hold more rows than the alternate token tile takes the alternate
away. With ``shared_token_tile_rows`` at the alternate's N the shared tiles fit
it and the alternate stays; the head then starts once the launch has chosen
(``_choose_token_tile``, which counts the shared tiles' rows). With
``shared_head_token_tiles`` the head holds the FC1 tiles of the leading token
tiles only; the others' are spread over the claim streams among the routed
tiles, ahead of the shared FC2 tiles (``SharedExpertTiles``).

A route's routing weight multiplies its SwiGLU output in the FC1 epilogue, before
the MXFP8 requantization (``apply_topk_at_fc1``, the default). Without it the
weight applies after FC2, in FP32: the separate top-k reduce multiplies each
route's BF16 combine slot by the source rank's own routing weight as it sums the
slots, and under the in-kernel reduce the FC2 epilogue multiplies each route's
FP32 accumulator by the weight its routing record carries before rounding the
partial to BF16. The communication carries the routing weights to the target
ranks either way, so every protocol runs unchanged.
"""

import dataclasses
import functools
import math
from typing import ClassVar, Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
from .....helpers.ptx_helpers import barrier_sync_unaligned, red_add_relaxed_sys_s32
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Boolean, Int32, Int64
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .....api import (
    ImplDesc,
    KernelClass,
    OptionalRequirement,
    ProblemDesc,
    StaticOrRuntimeIntegerType,
)
from .....communication.nvlink_domain.symmetric_buffer import SymmetricBufferDevice
from .....helpers.cute_py_helpers import (
    Tcgen05MmaInstruction,
    make_tcgen05_tmem_plan,
    tcgen05_block_scaled_acc_dtype,
)
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.dsl_helpers import spin_wait
from ......comm import ptx as sentinel_ptx
from .....helpers.iket_compat import iket
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import balanced_tile_rows, ceil_div, round_up, token_slot_rows
from .....quant_def import CombineFormat, QuantKind
from ....schedulers.base import WorkIdAcquisitionMode
from ....schedulers.fc12_mapping import SharedExpertTiles
from ....schedulers.fc12_scheduler import (
    BlackwellFusedFc12Scheduler,
    PhaseInterleavedFc12Scheduler,
)
from .block_scaled_swap_ab_fc12_epilogue import (
    GatedActEpilogueArgs,
    SwapABGatedActEpilogue,
)
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .block_scaled_swap_ab_fc12_mainloop_gen_specialized import (
    BlockScaledSwapAbFc12MainloopGenSpecialized,
)
from .token_comm_gen_specialized import (
    GenphaseTokenComm,
    GenphaseTokenCommArgs,
    _warp_inclusive_sum,
)
from .combine_reduce import CombineTopkReduce, SetNvlinkBarrier


_aot_symbol_prefix = "blackwell_genphase_mega_moe_aot"


@dataclasses.dataclass(frozen=True)
class _TokenTileConfig:
    """The parts of the main kernel a token tile N sizes: mainloop, epilogue, and their SMEM plan.

    Every configuration of one kernel runs the same plan (the scheduler's tiles
    and the communication component's pools) and the same warp roles.
    """

    tile_n: int
    mainloop: BlockScaledSwapAbFc12MainloopGenSpecialized
    epilogue: SwapABGatedActEpilogue
    smem_workspace: SmemWorkspace

    # Static: the DSL passes it through runtime control flow as a value without IR.
    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "_TokenTileConfig":
        return self


class BlockScaledSwapAbGenphaseMoeKernel(KernelClass):
    """Compose generation-phase dispatch, persistent FC12, and the return path."""

    fc1_output_region: ClassVar[str] = "blackwell.genphase_moe.fc1_output"
    fc1_output_sf_region: ClassVar[str] = "blackwell.genphase_moe.fc1_output_sf"
    fc1_done_counter_region: ClassVar[str] = "blackwell.genphase_moe.fc1_done_counter"
    pre_reduced_activation_region: ClassVar[str] = (
        "blackwell.genphase_moe.pre_reduced_activation"
    )
    # In-kernel top-k reduce: the output's zeroing (see ``_zero_output``). Chunks
    # claimed and chunks zeroed on this rank, and the ranks whose output is zeroed
    # this round, which each rank adds to in every rank's set.
    output_zero_claim_region: ClassVar[str] = "blackwell.genphase_moe.output_zero_claim"
    output_zero_done_region: ClassVar[str] = "blackwell.genphase_moe.output_zero_done"
    output_zeroed_ranks_region: ClassVar[str] = (
        "blackwell.genphase_moe.output_zeroed_ranks"
    )
    # Bytes of output one epilogue warp zeroes per claim.
    output_zero_chunk_bytes: ClassVar[int] = 32 * 1024

    epilogue_warp_ids: ClassVar[Tuple[int, int, int, int]] = (0, 1, 2, 3)
    mma_warp_id: ClassVar[int] = 4
    tma_a_warp_id: ClassVar[int] = 5
    tma_b_warp_id: ClassVar[int] = 6
    scheduler_warp_id: ClassVar[int] = 7
    claim_gate_region: ClassVar[str] = "blackwell.genphase_moe.claim_gate_mbarrier"
    # The token tile configuration a launch runs (see ``_choose_token_tile``): an
    # SMEM word every configuration's plan places first, so it sits at offset 0 in
    # all of them, and a device word CTA 0 records the chosen tile N in.
    token_tile_choice_region: ClassVar[str] = "blackwell.genphase_moe.token_tile_choice"
    token_tile_record_region: ClassVar[str] = "blackwell.genphase_moe.token_tile_record"
    # Under a token pool with a capacity (see ``_plan_token_pool``): the launch's
    # capacity and, per local expert, its first route in processing order and its
    # routes per token tile, in SMEM words every configuration's plan places right
    # after the choice word; and a device word pair CTA 0 records the launch's
    # routes and capacity in.
    token_pool_plan_region: ClassVar[str] = "blackwell.genphase_moe.token_pool_plan"
    token_pool_record_region: ClassVar[str] = "blackwell.genphase_moe.token_pool_record"
    # Gather-B warps start at warp 8: the third warpgroup, and with eight of them
    # (``gather_b_warps``) the fourth as well.
    first_gather_b_warp_id: ClassVar[int] = 8
    # Under the pull dispatch with `pull_warps`, the fourth warpgroup; the first
    # `pull_warps` of its warps fetch the routed rows (GenphaseTokenComm.run_pull_warp).
    pull_warp_ids: ClassVar[Tuple[int, int, int, int]] = (12, 13, 14, 15)
    # Its registers: 512 threads launch at 128 registers each, and the epilogue
    # warpgroup's 256 leave 80 for each of the other three.
    pull_warp_register_count: ClassVar[int] = 80
    epilogue_register_count: ClassVar[int] = 256
    other_warp_register_count: ClassVar[int] = 80
    tail_barrier_id: ClassVar[int] = 8
    # Weight K tiles a work tile still has to issue when the scheduler claims the
    # next one under just-in-time claims, so the next tile goes to a cluster whose
    # stream is about to end. A load another warp issues while the SM streams
    # weights waits behind the weight loads, so claims map their tiles from
    # register copies of the token counts and the slot order; what remains is the
    # work-ID atomic, the cluster broadcast and, for an FC2 tile, its readiness
    # peek: about 1.3 us from claim to publication at the median, up to about 5 us.
    # Sixteen K tiles are about 3 us of an FC1 tile's stream, so the stream waits
    # between tiles only behind the slowest claims. An FC2 tile has sixteen K
    # tiles, so its successor is claimed as it starts.
    claim_lead_k_tiles: ClassVar[int] = 16
    # Claims a cluster that starts with the kernel holds at the first FC1 tile when
    # claims are eager: one per scheduler queue stage, one waiting for a stage, and
    # one more once the epilogue and Gather-B warps take their first tile at
    # metadata_ready.
    eager_claims_before_first_fc1: ClassVar[int] = 4

    # Every communication CTA holds an SM that a main-kernel cluster waits for
    # until it exits. The communication grid needs one helper and one exchange
    # CTA per peer in addition to these pushers, which also verify landing rows.
    pusher_cta_count: ClassVar[int] = 27
    # Under the pull dispatch these CTAs fetch instead of pushing, and with the
    # helper they walk this rank's routes. Fourteen walking CTAs keep the walk
    # ahead of the experts the main kernel consumes while 14 more SMs run it;
    # twice as many pull at the links' speed and hold those SMs instead.
    pull_pusher_cta_count: ClassVar[int] = 13
    # With the main kernel's pull warps these CTAs only copy slices of the index
    # lists beside the exchange CTAs and leave with the plan: the copy is
    # throughput bound at large capacities, and the SMs come back with the plan.
    pull_copy_cta_count: ClassVar[int] = 13
    # Default pull warps per CTA under the pull dispatch: none, so the walking CTAs
    # fetch. Four per SM (``pull_warps``) keep a row each in flight, about 3 MB per
    # rank over 148 SMs, and free the walking CTAs' SMs, but the 16-warp kernel
    # launches at 128 registers rather than 168 and spills a little; end to end
    # they measure 0.5 to 7 us slower than the walking CTAs at EP8.
    main_pull_warp_count: ClassVar[int] = 0
    refine_participant_groups: ClassVar[int] = 2
    refine_output_stages: ClassVar[int] = 4

    cluster_shape_mn: ClassVar[Tuple[int, int]] = (4, 1)
    mma_k_mode: ClassVar[str] = "1x"
    a_major_mode: ClassVar[OperandMajorMode] = OperandMajorMode.K
    b_major_mode: ClassVar[OperandMajorMode] = OperandMajorMode.K

    @classmethod
    def problem_desc_require(cls) -> dict[str, object]:
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
            "quant_kind": str,
            "combine_format": CombineFormat,
            "gate_up_clamp": Optional[float],
            "world_size": int,
            "topk": int,
            "topk_index_dtype": type,
            "max_tokens_per_rank": int,
            "apply_topk_at_fc1": bool,
            "shared_expert": OptionalRequirement(Optional[str]),
            "shared_gate_up_clamp": OptionalRequirement(Optional[float]),
            "shared_bf16_rounding": OptionalRequirement(bool),
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            "mma_instruction_mnk": tuple,
            "mma_tiler_mnk": tuple,
            "use_2cta_instrs": bool,
            "schedule_policy": tuple,
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "fc2_use_bulk": bool,
            "epi_flag_batches": tuple,
            "launch_cluster_count": int,
            "reduce_topk_in_kernel": bool,
            "comm_pusher_ctas": OptionalRequirement(int),
            "weight_layout": OptionalRequirement(str),
            "weight_block_rows": OptionalRequirement(int),
            "entry_prefetch_bytes": OptionalRequirement(int),
            "die_local_weights": OptionalRequirement(bool),
            # Dispatch knobs the communication component consumes (see GenphaseTokenComm).
            "dispatch_rounds": OptionalRequirement(int),
            "comm_push_depth": OptionalRequirement(int),
            "comm_pull_depth": OptionalRequirement(int),
            "comm_pull_chunk_rows": OptionalRequirement(int),
            "comm_pull_lead_rows": OptionalRequirement(int),
            "comm_exchange_ctas": OptionalRequirement(int),
            "pull_warps": OptionalRequirement(int),
            "gather_b_warps": OptionalRequirement(int),
            "token_ab_stages": OptionalRequirement(int),
            "pull_acquire_in_copy": OptionalRequirement(bool),
            "fc1_token_pool": OptionalRequirement(bool),
            "fc1_token_pool_capacity": OptionalRequirement(int),
            "fc1_token_pool_fallback": OptionalRequirement(str),
            "pull_walk_dedup": OptionalRequirement(bool),
            "pull_tile_ready": OptionalRequirement(bool),
            "refine_exchange_ctas": OptionalRequirement(int),
            "dispatch_round_mode": OptionalRequirement(str),
            "round_publish_delay_ns": OptionalRequirement(int),
            "round_publish_lag": OptionalRequirement(int),
            "refine_gate_round0_only": OptionalRequirement(bool),
            "stamps": OptionalRequirement(bool),
            "just_in_time_claims": OptionalRequirement(bool),
            "claim_lead_k_tiles": OptionalRequirement(int),
            "discard_combine_slots": OptionalRequirement(bool),
            "combine_warp_stores": OptionalRequirement(bool),
            "alternate_tile_n": OptionalRequirement(int),
            "alternate_token_ab_stages": OptionalRequirement(int),
            "alternate_weight_evict": OptionalRequirement(str),
            "token_tile_choice": OptionalRequirement(str),
            "shared_expert_entry": OptionalRequirement(bool),
            "shared_fc2_placement": OptionalRequirement(str),
            "shared_token_tile_rows": OptionalRequirement(Optional[int]),
            "shared_head_token_tiles": OptionalRequirement(Optional[int]),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)
        self.cluster_shape_mn = tuple(impl_desc.get("cluster_shape_mn", (4, 1)))
        self.architecture = "sm_100"
        self.mma_k_mode = "1x"

        for field_name, fixed_value, supplied_value in (
            (
                "cluster_shape_mn",
                self.cluster_shape_mn,
                impl_desc.get("cluster_shape_mn"),
            ),
            ("mma_k_mode", self.mma_k_mode, impl_desc.get("mma_k_mode")),
            ("a_major_mode", self.a_major_mode, problem_desc.get("a_major_mode")),
            ("b_major_mode", self.b_major_mode, problem_desc.get("b_major_mode")),
        ):
            if supplied_value is not None and supplied_value != fixed_value:
                raise ValueError(
                    f"The generation-phase kernel fixes {field_name} at {fixed_value}, got {supplied_value}."
                )

        fallback_cluster_shape_mn = impl_desc.get("fallback_cluster_shape_mn")
        if (
            fallback_cluster_shape_mn is not None
            and fallback_cluster_shape_mn != self.cluster_shape_mn
        ):
            raise ValueError(
                f"The generation-phase kernel launches a single {self.cluster_shape_mn} cluster shape, so "
                f"fallback_cluster_shape_mn must be unset or equal to it, got {fallback_cluster_shape_mn}."
            )

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.a_dtype = self.quant_kind.weight_dtype
        self.b_dtype = self.quant_kind.activation_dtype
        self.sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.combine_format = problem_desc["combine_format"]
        self.gate_up_clamp = problem_desc["gate_up_clamp"]
        self.world_size = problem_desc["world_size"]
        self.topk = problem_desc["topk"]
        self.topk_index_dtype = problem_desc["topk_index_dtype"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.schedule_policy = impl_desc["schedule_policy"]
        if len(self.schedule_policy) != 2:
            raise ValueError("schedule_policy must contain a mode and hint.")
        self.schedule_mode, self.hint = self.schedule_policy
        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.work_id_mode: WorkIdAcquisitionMode = impl_desc["work_id_mode"]
        self.fc2_use_bulk = impl_desc["fc2_use_bulk"]
        self.combine_protocol = impl_desc.get("combine_protocol", "sentinel")
        self.dispatch_protocol = impl_desc.get("dispatch_protocol", "sentinel")
        self.pull_warps = impl_desc.get(
            "pull_warps",
            type(self).main_pull_warp_count if self.dispatch_protocol == "pull" else 0,
        )
        if isinstance(self.pull_warps, bool) or not isinstance(self.pull_warps, int):
            raise ValueError(f"pull_warps must be an int, got {self.pull_warps!r}.")
        if not 0 <= self.pull_warps <= len(self.pull_warp_ids):
            raise ValueError(
                f"pull_warps must be in [0, {len(self.pull_warp_ids)}], got {self.pull_warps}."
            )
        if self.pull_warps and self.dispatch_protocol != "pull":
            raise ValueError("pull_warps requires the pull dispatch.")
        # Gather-B warps issue FC1's Gather4 token loads, one warp at a time per
        # four-row group: two at N64 on cluster (4, 1), where the multicast pair
        # splits a tile's rows, four otherwise unless `gather_b_warps` asks for
        # eight. Eight take the fourth warpgroup, so they exclude the pull warps.
        default_gather_b_warps = (
            2 if self.mma_tiler_mnk[1] == 64 and self.cluster_shape_mn == (4, 1) else 4
        )
        self.active_gather_b_warp_count = impl_desc.get(
            "gather_b_warps", default_gather_b_warps
        )
        if self.active_gather_b_warp_count not in (default_gather_b_warps, 8):
            raise ValueError(
                f"gather_b_warps must be {default_gather_b_warps} or 8 for this geometry, "
                f"got {self.active_gather_b_warp_count!r}."
            )
        if self.active_gather_b_warp_count == 8 and (
            self.pull_warps
            or self.cluster_shape_mn != (2, 1)
            or self.mma_tiler_mnk[1] < 128
        ):
            raise ValueError(
                "Eight Gather-B warps need cluster (2, 1), token tile N >= 128 and no pull warps."
            )
        self.gather_b_warp_ids = tuple(
            range(
                self.first_gather_b_warp_id,
                self.first_gather_b_warp_id + max(4, self.active_gather_b_warp_count),
            )
        )
        # Under the pull dispatch into the token pool TMA-B loads FC1's tokens and
        # the Gather-B warps take no work (see GenphaseTokenComm, "Pull dispatch").
        self.fc1_token_pool = impl_desc.get("fc1_token_pool", False)
        if self.fc1_token_pool and (
            self.dispatch_protocol != "pull"
            or self.pull_warps
            or impl_desc.get("pull_walk_dedup", False)
            or self.active_gather_b_warp_count != default_gather_b_warps
        ):
            raise ValueError(
                "fc1_token_pool needs the pull dispatch with walking CTAs, no pull_walk_dedup and the default "
                "Gather-B warps."
            )
        if self.pull_warps:
            # The pull warps fetch every row, so the pusher CTAs of the
            # communication grid only copy the index lists.
            default_pusher_ctas = type(self).pull_copy_cta_count
        elif self.dispatch_protocol == "pull":
            default_pusher_ctas = type(self).pull_pusher_cta_count
        else:
            default_pusher_ctas = type(self).pusher_cta_count
        self.pusher_cta_count = impl_desc.get("comm_pusher_ctas", default_pusher_ctas)
        # The end-of-round reduce drops consumed combine slot lines from L2; only the
        # counter combine leaves slots that are rewritten before they are next read.
        self.discard_combine_slots = impl_desc.get("discard_combine_slots", False)
        if not isinstance(self.discard_combine_slots, bool):
            raise ValueError(
                f"discard_combine_slots must be a bool, got {self.discard_combine_slots!r}."
            )
        if self.discard_combine_slots and self.combine_protocol != "counter":
            raise ValueError("discard_combine_slots requires the counter combine.")
        # The counter combine's FC2 epilogue pushes with warp-wide stores rather than
        # bulk copies (SwapABGatedActEpilogue.combine_warp_stores).
        self.combine_warp_stores = impl_desc.get("combine_warp_stores", False)
        if not isinstance(self.combine_warp_stores, bool):
            raise ValueError(
                f"combine_warp_stores must be a bool, got {self.combine_warp_stores!r}."
            )
        if self.combine_warp_stores and self.combine_protocol != "counter":
            raise ValueError("combine_warp_stores requires the counter combine.")
        if self.discard_combine_slots and impl_desc["reduce_topk_in_kernel"]:
            raise ValueError(
                "The in-kernel top-k reduce has no combine slots to discard."
            )
        # Compiled in, just-in-time claims still leave each launch to pick its claim
        # timing from its tile counts (``_claims_pay_just_in_time``).
        self.just_in_time_claims = impl_desc.get("just_in_time_claims", False)
        if not isinstance(self.just_in_time_claims, bool):
            raise ValueError(
                f"just_in_time_claims must be a bool, got {self.just_in_time_claims!r}."
            )
        claim_lead_k_tiles = impl_desc.get("claim_lead_k_tiles")
        if claim_lead_k_tiles is not None and not self.just_in_time_claims:
            raise ValueError("claim_lead_k_tiles applies only to just-in-time claims.")
        self.claim_lead_k_tiles = (
            type(self).claim_lead_k_tiles
            if claim_lead_k_tiles is None
            else claim_lead_k_tiles
        )
        if (
            isinstance(self.claim_lead_k_tiles, bool)
            or not isinstance(self.claim_lead_k_tiles, int)
            or self.claim_lead_k_tiles < 1
        ):
            raise ValueError(
                f"claim_lead_k_tiles must be a positive int, got {self.claim_lead_k_tiles!r}."
            )
        self.epi_flag_batches = impl_desc["epi_flag_batches"]
        self.launch_cluster_count = impl_desc["launch_cluster_count"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]
        # Physical weight layout. "kmajor": each weight row is contiguous in K.
        # "blockmajor": the host stores every (weight_block_rows x weight_block_k)
        # block contiguously, K blocks inner, so one TMA stage box is one contiguous
        # run of memory and a CTA's M tile is contiguous across all of K.
        # FC1 weight bytes, in the scheduler's expert order, that the epilogue warps
        # prefetch into L2 while the dispatched payload is still arriving.
        self.entry_prefetch_bytes = impl_desc.get("entry_prefetch_bytes", 0)
        self.weight_layout = impl_desc.get("weight_layout", "kmajor")
        self.weight_block_rows = impl_desc.get("weight_block_rows", 128)
        if self.weight_layout not in ("kmajor", "blockmajor"):
            raise ValueError(
                f"weight_layout must be kmajor or blockmajor, got {self.weight_layout!r}."
            )
        # Die-local weights: the first and second half of each weight matrix's M
        # tiles live in the HBM of one die each, as experts [h * E, (h + 1) * E)
        # of a tensor with half the rows, and the scheduler claims each die's half
        # from its own work counter (BlackwellFusedFc12Scheduler).
        self.die_local_weights = impl_desc.get("die_local_weights", False)
        if self.die_local_weights and (
            self.weight_layout != "blockmajor"
            or self.cluster_shape_mn != (2, 1)
            or self.work_id_mode != "atomic_counter"
        ):
            raise ValueError(
                "die_local_weights needs block-major weights, cluster (2, 1) and atomic-counter claims."
            )

        self.occupancy = 1
        if impl_desc.get("architecture", "sm_100") != "sm_100":
            raise ValueError(f"Unsupported GenPhase architecture {self.architecture}.")
        if (
            self.quant_kind not in (QuantKind.mxfp8_e4m3, QuantKind.mxfp4_mxfp8)
            or (self.mma_tiler_mnk, self.cluster_shape_mn)
            not in (
                ((256, 128, 128), (4, 1)),
                ((256, 128, 128), (2, 1)),
                ((256, 64, 128), (4, 1)),
                ((256, 64, 128), (2, 1)),
                ((256, 192, 128), (2, 1)),
                ((256, 240, 128), (2, 1)),
            )
            or self.schedule_mode != "grouped"
        ):
            raise ValueError(
                "Blackwell GenPhase requires MXFP8 E4M3 or MXFP4 weights x MXFP8, N64/N128 at CGA2/CGA4 or "
                "N192/N240 at CGA2 and grouped scheduling."
            )
        # In-kernel top-k reduce: every FC2 push adds its partial into the source
        # token's output row, which lives in the source rank's staging allocation,
        # so no combine slots exist and the end-of-round kernel only keeps house.
        # The tail barrier is what publishes the adds, so it needs the counter
        # combine; Sentinel slots announce their own arrival, which a sum cannot.
        if self.reduce_topk_in_kernel and self.combine_protocol != "counter":
            raise ValueError("The in-kernel top-k reduce requires the counter combine.")
        self.threads_per_cta = (
            16 if self.pull_warps or self.active_gather_b_warp_count > 4 else 12
        ) * 32
        self.local_expert_count = self.expert_count // self.world_size
        self.cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        self.promised_launchable_sm_count = (
            self.launch_cluster_count * self.cluster_size
        )

        self._validate_geometry()
        self._resolve_schedule_hint()
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=self.mma_instruction_mnk,
            participates=mma_cta_count,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        tmem_plan = make_tcgen05_tmem_plan(
            mma_instruction, self.architecture, self.mma_tiler_mnk
        )
        self.cta_tile_m = self.mma_tiler_mnk[0] // mma_cta_count
        self.cta_tile_n = self.mma_tiler_mnk[1]

        self._configure_shared_expert(problem_desc, impl_desc)
        # The claim gate (``claim_gate_region``) paces just-in-time claims and a
        # shared expert's entry head, which every cluster claims one tile at a time.
        self.shared_entry_head = bool(self.shared_expert and self.shared_expert_entry)
        self.claim_gate_compiled = self.just_in_time_claims or self.shared_entry_head
        token_comm_impl_desc = ImplDesc(
            {
                **impl_desc,
                "pusher_cta_count": self.pusher_cta_count,
                "refine_participant_groups": self.refine_participant_groups,
                "refine_output_stages": self.refine_output_stages,
                "dispatch_protocol": self.dispatch_protocol,
                "pull_warps": self.pull_warps,
                **(
                    {
                        "shared_expert_rows": (
                            self.shared_expert_data_rows,
                            self.shared_expert_sf_rows,
                        )
                    }
                    if self.shared_expert
                    else {}
                ),
            }
        )
        # The routing weights travel to the target ranks wherever they apply: the
        # FC1 epilogue, or the in-kernel reduce's FC2 epilogue, read them there.
        token_comm_problem_desc = (
            problem_desc
            if self.apply_topk_at_fc1
            else ProblemDesc({**problem_desc, "apply_topk_at_fc1": True})
        )
        self.token_comm = GenphaseTokenComm(
            token_comm_problem_desc, token_comm_impl_desc
        )
        self.max_routed_rows = self.token_comm.worst_case_data_rows
        # Rows of the FC1 output pool: the routed worst case, then the shared expert's.
        self.fc1_output_rows = self.max_routed_rows + (
            self.shared_expert_data_rows if self.shared_expert else 0
        )
        # Combine slots per token: the routes', then the shared expert's.
        self.combine_slots_per_token = self.topk + (1 if self.shared_expert else 0)
        # A token pool with a capacity keeps the landing slots and the Gather-B
        # warps for the token tiles past it; a pool of every data row needs neither.
        self.token_pool_capacity = self.token_comm.fc1_token_pool_capacity
        gather_consumer_warps = (
            0
            if self.fc1_token_pool and not self.token_pool_capacity
            else self.active_gather_b_warp_count
        )
        self.scheduler_consumer_thread_count = (7 + gather_consumer_warps) * 32
        self._validate_communication_geometry()
        # The counter combine publishes this rank's finished pushes with one
        # all-rank barrier after the grid's last FC2 store.
        self.tail_barrier = (
            SetNvlinkBarrier(
                world_size=self.world_size, barrier_id=self.tail_barrier_id
            )
            if self.combine_protocol == "counter"
            else None
        )
        self.shared_expert_tiles = (
            self._shared_expert_tiles() if self.shared_expert else None
        )
        # A rank's refine acquires only the dispatch rounds its experts need, so a
        # peer's publication of another round may still be in flight when this
        # rank's reduce clears the round counters. Only the tail barrier, which
        # every rank enters after its communication grid completed, orders all of
        # them before the clear.
        if self.token_comm.dispatch_rounds > 1 and self.tail_barrier is None:
            raise ValueError(
                "dispatch_rounds > 1 requires the counter combine protocol."
            )

        fc12_problem_desc = ProblemDesc(
            {**problem_desc, "expert_count": self.local_expert_count}
        )
        shared_impl_fields = {}
        if self.shared_expert:
            shared_impl_fields = {
                "shared_expert_tiles": self.shared_expert_tiles,
                "shared_expert_marker": self.local_expert_count,
                "shared_expert_token_tiles": self.shared_expert_tiles.token_tiles,
            }
        resolved_impl_desc = ImplDesc(
            {
                **impl_desc,
                **shared_impl_fields,
                # Fixed here, but the scheduler and the epilogue read them from the
                # descriptor, so they have to be forwarded rather than assumed.
                "cluster_shape_mn": self.cluster_shape_mn,
                "mma_k_mode": self.mma_k_mode,
                "hint": self.hint,
                "tmem_plan": tmem_plan,
                "is_swap_ab": True,
                "max_tokens": self.max_routed_rows,
                "num_scheduler_consumer_threads": self.scheduler_consumer_thread_count,
                "gather_b_warps": self.active_gather_b_warp_count,
                "fc1_token_pool_bounded": self.token_pool_capacity > 0,
                "num_accumulator_consumer_warps_per_cta": len(self.epilogue_warp_ids),
                "communication_enabled": True,
                # No token-back path exists in this component, so results never
                # take the local route and the FC2 store is always the peer-facing
                # bulk reduce-add.
                "token_back_push_data": False,
                "fc1_epi_flag_batch": self.epi_flag_batches[0],
                "fc2_epi_flag_batch": self.epi_flag_batches[1],
            }
        )
        scheduler_type = (
            BlackwellFusedFc12Scheduler
            if self.schedule_mode == "grouped"
            else PhaseInterleavedFc12Scheduler
        )
        self.scheduler = scheduler_type(fc12_problem_desc, resolved_impl_desc)
        self.epilogue = SwapABGatedActEpilogue(fc12_problem_desc, resolved_impl_desc)
        if self.epilogue.fc1_output_dtype is not self.b_dtype:
            raise ValueError(
                "Epilogue FC1 output dtype must match the Mainloop B dtype."
            )
        if self.epilogue.fc1_output_sf_dtype is not self.sf_dtype:
            raise ValueError(
                "Epilogue FC1 output scale dtype must match the Mainloop scale dtype."
            )
        if self.epilogue.sf_vec_size != self.sf_vec_size:
            raise ValueError("Epilogue and Mainloop scale vector sizes must match.")
        if not self.epilogue.fc2_use_ublk:
            raise ValueError(
                "The generation-phase FC2 store must be the bulk path: set fc2_use_bulk=True so the epilogue "
                "selects the peer-facing UBLK store."
            )
        if self.epilogue.token_back_enabled:
            raise ValueError(
                "The generation-phase communication component provides no token-back path."
            )
        self._configure_token_tile_choice(impl_desc)
        self._device_workspace = self._build_device_workspace()
        self._mainloop, self._smem_workspace = self._build_mainloop_and_smem(
            fc12_problem_desc, resolved_impl_desc, self.epilogue
        )
        self._primary = _TokenTileConfig(
            self.cta_tile_n, self._mainloop, self.epilogue, self._smem_workspace
        )
        self._alternate = (
            self._build_alternate_token_tile(fc12_problem_desc, resolved_impl_desc)
            if self.alternate_tile_n
            else None
        )
        # The reduce polls only valid routes' slots, so invalid slots never need clearing.
        self.masks_invalid_combine_slots = True
        self._topk_reduce = CombineTopkReduce(
            self.token_comm,
            self.hidden_size,
            self.topk,
            self.expert_count,
            self.world_size,
            self.max_tokens_per_rank,
            sm_count=self.promised_launchable_sm_count,
            tail_reset_bytes=max(
                self._device_workspace.tail_reset_bytes("local"),
                self._device_workspace.tail_reset_bytes("shared"),
            ),
            poll_slots=self.combine_protocol == "sentinel",
            reset_landing=self.dispatch_protocol == "sentinel",
            discard_slots=self.discard_combine_slots,
            reduce_slots=not self.reduce_topk_in_kernel,
            shared_slot=bool(self.shared_expert) and not self.reduce_topk_in_kernel,
            shared_bf16_rounding=self.shared_bf16_rounding,
            weight_routes=not self.apply_topk_at_fc1 and not self.reduce_topk_in_kernel,
        )

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def _configure_shared_expert(
        self, problem_desc: ProblemDesc, impl_desc: ImplDesc
    ) -> None:
        """Read the shared expert and fix the pool rows its token tiles take.

        ``shared_expert`` "fp8_in_kernel" runs one MXFP8 E4M3 shared expert of
        the routed experts' intermediate width over every token of the capacity;
        its token tiles follow the plan's token tile N and slot layout. Its
        numerics: ``shared_gate_up_clamp`` (None: no clamp, whatever the routed
        clamp) and ``shared_bf16_rounding`` (gate, up and the SwiGLU product
        rounded to BF16, and the routes' sum rounded to BF16 before the shared
        output joins it). Its placement: ``shared_expert_entry`` claims its FC1
        tiles from kernel entry (see the module notes), and
        ``shared_fc2_placement`` puts its FC2 tiles behind the routed tiles
        ("tail", the default), right behind its FC1 tiles ("early"), or early
        exactly where they fit one wave of clusters ("auto").
        ``shared_token_tile_rows`` (None: the plan's tile N) bounds the rows of
        its token tiles, which still take one slot of the plan each: a bound of
        the alternate token tile's N lets a launch take the alternate
        (``_configure_token_tile_choice``). ``shared_head_token_tiles`` (None:
        all) caps the entry head at that many token tiles' FC1 tiles; the rest
        join the claim streams among the routed tiles (``SharedExpertTiles``).
        """
        self.shared_expert = problem_desc.get("shared_expert")
        self.shared_gate_up_clamp = problem_desc.get("shared_gate_up_clamp")
        self.shared_bf16_rounding = problem_desc.get("shared_bf16_rounding", False)
        self.shared_expert_entry = impl_desc.get("shared_expert_entry", False)
        self.shared_fc2_placement = impl_desc.get("shared_fc2_placement", "tail")
        self.shared_token_tile_rows = impl_desc.get("shared_token_tile_rows")
        self.shared_head_token_tiles = impl_desc.get("shared_head_token_tiles")
        if self.shared_expert not in (None, "fp8_in_kernel"):
            raise ValueError(
                f"shared_expert must be None or 'fp8_in_kernel', got {self.shared_expert!r}."
            )
        if type(self.shared_expert_entry) is not bool:
            raise ValueError(
                f"shared_expert_entry must be a bool, got {self.shared_expert_entry!r}."
            )
        if self.shared_fc2_placement not in ("auto", "early", "tail"):
            raise ValueError(
                f"shared_fc2_placement must be 'auto', 'early' or 'tail', got {self.shared_fc2_placement!r}."
            )
        if not self.shared_expert:
            if (
                self.shared_gate_up_clamp is not None
                or self.shared_bf16_rounding
                or self.shared_expert_entry
                or "shared_fc2_placement" in impl_desc
                or self.shared_token_tile_rows is not None
                or self.shared_head_token_tiles is not None
            ):
                raise ValueError(
                    "shared_gate_up_clamp, shared_bf16_rounding, shared_expert_entry, shared_fc2_placement, "
                    "shared_token_tile_rows and shared_head_token_tiles need a shared expert."
                )
            return
        if (
            self.quant_kind != QuantKind.mxfp4_mxfp8
            or self.cluster_shape_mn != (2, 1)
            or not self.use_2cta_instrs
        ):
            raise ValueError(
                "A shared expert needs MXFP4 x MXFP8 weights, two-CTA MMAs and cluster (2, 1)."
            )
        if self.shared_bf16_rounding and self.reduce_topk_in_kernel:
            raise ValueError(
                "shared_bf16_rounding rounds the routes' sum before the shared output joins it, which the in-kernel "
                "top-k reduce, adding every partial in arrival order, cannot do."
            )
        tile_n = self.mma_tiler_mnk[1]
        tokens = self.max_tokens_per_rank
        slot_rows = token_slot_rows(tile_n)
        row_bound = tile_n
        if self.shared_token_tile_rows is not None:
            row_bound = self.shared_token_tile_rows
            if (
                isinstance(row_bound, bool)
                or not isinstance(row_bound, int)
                or row_bound <= 0
                or row_bound % 16
                or row_bound > tile_n
                or slot_rows == tile_n
            ):
                raise ValueError(
                    f"shared_token_tile_rows must be a positive multiple of 16 no larger than the strided tile N "
                    f"{tile_n}, got {row_bound!r}."
                )
        self.shared_expert_token_tiles = ceil_div(tokens, row_bound)
        self.shared_expert_tile_rows = row_bound
        if slot_rows != tile_n:
            # Strided slots spread a tile's routes evenly (``balanced_tile_rows``).
            even_share = ceil_div(tokens, self.shared_expert_token_tiles)
            rounded_share = round_up(even_share, 16)
            self.shared_expert_tile_rows = (
                rounded_share
                if (self.shared_expert_token_tiles - 1) * rounded_share < tokens
                else even_share
            )
        self.shared_expert_data_rows = self.shared_expert_token_tiles * slot_rows
        self.shared_expert_sf_rows = round_up(
            self.shared_expert_data_rows, self.sf_padding_block
        )
        head_token_tiles = self.shared_head_token_tiles
        if head_token_tiles is not None:
            if (
                isinstance(head_token_tiles, bool)
                or not isinstance(head_token_tiles, int)
                or head_token_tiles < 0
            ):
                raise ValueError(
                    f"shared_head_token_tiles must be a nonnegative int, got {head_token_tiles!r}."
                )
            if not self.shared_expert_entry or self.shared_fc2_placement != "tail":
                raise ValueError(
                    "shared_head_token_tiles caps the entry head, with the FC2 tiles at the tail."
                )
            if head_token_tiles >= self.shared_expert_token_tiles:
                head_token_tiles = None
        self.shared_head_token_tiles = head_token_tiles

    def _shared_expert_tiles(self) -> SharedExpertTiles:
        """The shared expert's tiles past the routed worst case of every pool, once the pools are sized.

        Its FC2 tiles follow the routed tiles ("tail"), where they overlap the
        last combine pushes and the reduce's polls, or its FC1 tiles ("early"),
        or the latter exactly where all of them fit one wave of clusters
        ("auto"). Early FC2 tiles hold their clusters at their FC1 gate while the
        routed FC1 tiles become ready, which costs more than they save.
        """
        tile_n = self.mma_tiler_mnk[1]
        fc2_cluster_tiles = self.shared_expert_token_tiles * ceil_div(
            self.hidden_size, self.mma_tiler_mnk[0] // 2 * self.cluster_shape_mn[0]
        )
        fc2_early = fc2_cluster_tiles <= self.launch_cluster_count
        if self.shared_fc2_placement != "auto":
            fc2_early = self.shared_fc2_placement == "early"
        return SharedExpertTiles(
            expert_marker=self.local_expert_count,
            tokens=self.max_tokens_per_rank,
            tile_rows=self.shared_expert_tile_rows,
            token_tiles=self.shared_expert_token_tiles,
            slot_rows=token_slot_rows(tile_n),
            data_base=self.token_comm.worst_case_data_rows,
            sf_base=self.token_comm.worst_case_sf_rows,
            block_base=self.token_comm.worst_case_padded_rows(tile_n) // tile_n,
            fc2_early=fc2_early,
            entry_head=self.shared_expert_entry,
            head_token_tiles=self.shared_head_token_tiles,
        )

    def _validate_geometry(self) -> None:
        static_expert_dimensions = (
            isinstance(self.expert_count, int),
            isinstance(self.intermediate_gateup_size, int),
            isinstance(self.hidden_size, int),
        )
        if any(static_expert_dimensions) and not all(static_expert_dimensions):
            raise ValueError(
                "Genphase MegaMoE expert dimensions must be either all static or all runtime."
            )
        if not all(static_expert_dimensions):
            raise NotImplementedError(
                "The Genphase MegaMoE kernel currently requires static expert dimensions."
            )
        if (
            self.expert_count <= 0
            or self.intermediate_gateup_size <= 0
            or self.hidden_size <= 0
        ):
            raise ValueError("Genphase MegaMoE expert dimensions must be positive.")
        if self.world_size <= 0 or self.expert_count % self.world_size != 0:
            raise ValueError("expert_count must be divisible by a positive world_size.")
        if self.topk <= 0 or self.topk > self.expert_count:
            raise ValueError("topk must be positive and no greater than expert_count.")
        if self.topk_index_dtype not in (cutlass.Int32, cutlass.Int64):
            raise ValueError(
                f"topk_index_dtype must be Int32 or Int64, got {self.topk_index_dtype}."
            )
        # Routing weights apply at FC1 (``apply_topk_at_fc1``) or after FC2: in
        # the top-k reduce, or under the in-kernel reduce in the FC2 epilogue (see
        # the module notes).
        # A quantized combine format would derive token_back_push_sf, and with it a
        # token-back path this component does not have: no FC2 done counter, no
        # FC2 output scale plane, no transfer warps.
        if self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError(
                "The generation-phase kernel requires a BF16 combine format: a quantized payload would need the "
                "token-back path, which the generation-phase communication component does not provide."
            )
        if self.intermediate_gateup_size % 2 != 0:
            raise ValueError("The SwiGLU intermediate dimension must be even.")
        if self.max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive.")
        if self.token_padding_block <= 0 or self.token_padding_block % 64 != 0:
            raise ValueError("token_padding_block must be a positive multiple of 64.")
        if self.launch_cluster_count <= 0:
            raise ValueError("launch_cluster_count must be positive.")
        if self.schedule_mode not in ("grouped", "phase_interleave"):
            raise ValueError(
                f"schedule_policy mode must be 'grouped' or 'phase_interleave', got {self.schedule_mode!r}."
            )
        if self.hint is not None and (
            isinstance(self.hint, bool)
            or not isinstance(self.hint, int)
            or self.hint <= 0
        ):
            raise ValueError(
                f"schedule_policy hint must be a positive Python int, got {self.hint!r}."
            )
        if (
            self.schedule_mode == "phase_interleave"
            and self.work_id_mode != "atomic_counter"
        ):
            raise ValueError(
                "phase_interleave currently requires work_id_mode='atomic_counter'."
            )
        if len(self.epi_flag_batches) != 2:
            raise ValueError("epi_flag_batches must contain FC1 and FC2 batch sizes.")
        if any(
            batch < 1 or batch > len(self.epilogue_warp_ids)
            for batch in self.epi_flag_batches
        ):
            raise ValueError(
                f"epi_flag_batches values must be in [1, {len(self.epilogue_warp_ids)}]."
            )
        if self.sf_vec_size <= 0:
            raise ValueError("sf_vec_size must be positive.")

        for field_name, dimensions, expected_rank in (
            ("mma_instruction_mnk", self.mma_instruction_mnk, 3),
            ("mma_tiler_mnk", self.mma_tiler_mnk, 3),
        ):
            if len(dimensions) != expected_rank:
                raise ValueError(
                    f"{field_name} must contain {expected_rank} dimensions."
                )
            if not all(
                isinstance(dimension, int) and not isinstance(dimension, bool)
                for dimension in dimensions
            ):
                raise TypeError(f"{field_name} dimensions must be Python integers.")
            if any(dimension <= 0 for dimension in dimensions):
                raise ValueError(f"{field_name} dimensions must be positive.")

        mma_m, mma_n, mma_k = self.mma_tiler_mnk
        instruction_m, instruction_n, instruction_k = self.mma_instruction_mnk
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        expected_instruction_m = 256 if self.use_2cta_instrs else 128
        if instruction_m != expected_instruction_m:
            raise ValueError(
                f"{'2CTA' if self.use_2cta_instrs else '1CTA'} Blackwell MMA requires "
                f"instruction M={expected_instruction_m}, got {instruction_m}."
            )
        expected_instruction_k = self.quant_kind.instruction_k(self.mma_k_mode)
        if instruction_k != expected_instruction_k:
            raise ValueError(
                f"{self.quant_kind} requires Blackwell 1x instruction K={expected_instruction_k}, got {instruction_k}."
            )
        if (instruction_m, instruction_n) != (mma_m, mma_n):
            raise NotImplementedError(
                "Blackwell Genphase MegaMoE does not implement M/N instruction repetition."
            )
        # The additional decode geometry is deliberately bounded to the
        # MXFP8 and mixed-FP4 paths with two-CTA MMA and two, four or eight gather warps.
        if self.cluster_shape_mn != (4, 1) or self.mma_tiler_mnk[1] in (64, 192, 240):
            if not (
                self.cluster_shape_mn in ((2, 1), (4, 1))
                and self.mma_tiler_mnk
                in ((256, 64, 128), (256, 128, 128), (256, 192, 128), (256, 240, 128))
                and (
                    self.mma_tiler_mnk[1] in (64, 128)
                    or self.cluster_shape_mn == (2, 1)
                )
                and self.use_2cta_instrs
                and self.quant_kind in (QuantKind.mxfp8_e4m3, QuantKind.mxfp4_mxfp8)
            ):
                raise ValueError(
                    "GenPhase CGA2/N64 requires MXFP8 E4M3 or mixed FP4, tile (256, 64|128|192|240, 128), "
                    "and cluster (2, 1) or (4, 1); N192 and N240 run at cluster (2, 1) only."
                )
        if mma_n not in (64, 128, 192, 240, 256):
            raise ValueError(
                f"mma_tiler N must be 64, 128, 192, 240 or 256, got {mma_n}."
            )
        if mma_k % instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
        if mma_k % (self.sf_vec_size * 4) != 0:
            raise ValueError(
                "mma_tiler K must be divisible by four scale-factor vectors."
            )
        if self.intermediate_gateup_size % (self.sf_vec_size * 4) != 0:
            raise ValueError(
                "The intermediate dimension must be divisible by four scale-factor vectors."
            )
        if mma_m // mma_cta_count != 128:
            raise ValueError(
                f"The epilogue fixes the per-CTA output tile at 128, got {mma_m // mma_cta_count}."
            )
        # The epilogue picks the TMA owner warp with `warp_idx == subtile_idx` and
        # does not check the bound itself, so a wider token tile would drop stores.
        subtile_count = ceil_div(mma_n, 64)
        if subtile_count > len(self.epilogue_warp_ids):
            raise ValueError(
                f"mma_tiler N {mma_n} needs {subtile_count} epilogue subtiles, more than the "
                f"{len(self.epilogue_warp_ids)} epilogue warps that own them."
            )

    def _validate_communication_geometry(self) -> None:
        """Check where the FC12 tiling has to agree with the communication pools.

        A gather row index is read from GMEM before it reaches the descriptor, so
        unlike a coordinate it is not bounds-checked: the read itself has to stay
        inside the index array. A gather window starts at a tile base inside the
        padded data pool and spans a whole token tile, so its last row can sit
        ``cta_tile_n - 1`` past the final expert, and the index array carries
        exactly that much readable slack for exactly this reason.
        """
        if self.cta_tile_n > self.token_comm.gather_index_tail_slack:
            raise ValueError(
                f"A token tile of {self.cta_tile_n} rows can read up to {self.cta_tile_n - 1} rows past the last "
                f"expert, exceeding the {self.token_comm.gather_index_tail_slack} rows of readable slack the gather "
                f"index array carries."
            )

    def _configure_token_tile_choice(self, impl_desc: ImplDesc) -> None:
        """Read the alternate token tile and how a launch chooses between it and the tile N of the plan.

        ``alternate_tile_n`` (0: none) adds a second configuration of the main
        kernel's mainloop and epilogue at that tile N, which runs the tiles of
        the same plan: N128 under a strided plan (N192, N240), whose 256-row
        slots hold two of its scale-factor atoms. It can run a launch only if no
        token tile of the plan has more rows than it holds. Its own split AB ring
        (``alternate_token_ab_stages`` token stages, 0 for one shared ring) and
        weight eviction (``alternate_weight_evict``) size it. Each launch picks
        one configuration once its expert sizes are known (``_choose_token_tile``):
        ``token_tile_choice`` "rule" applies the choice rule, "primary" keeps the
        plan's own tile N in a kernel that carries both.
        """
        self.alternate_tile_n = impl_desc.get("alternate_tile_n", 0)
        self.alternate_token_ab_stages = impl_desc.get("alternate_token_ab_stages", 0)
        self.alternate_weight_evict = impl_desc.get(
            "alternate_weight_evict", impl_desc.get("weight_evict", "normal")
        )
        self.token_tile_choice = impl_desc.get("token_tile_choice", "rule")
        if isinstance(self.alternate_tile_n, bool) or self.alternate_tile_n not in (
            0,
            128,
        ):
            raise ValueError(
                f"alternate_tile_n must be 0 or 128, got {self.alternate_tile_n!r}."
            )
        if self.token_tile_choice not in ("rule", "primary"):
            raise ValueError(
                f"token_tile_choice must be 'rule' or 'primary', got {self.token_tile_choice!r}."
            )
        if not self.alternate_tile_n:
            if (
                self.token_tile_choice != "rule"
                or "alternate_token_ab_stages" in impl_desc
            ):
                raise ValueError(
                    "token_tile_choice and alternate_token_ab_stages need an alternate_tile_n."
                )
            return
        if token_slot_rows(
            self.cta_tile_n
        ) == self.cta_tile_n or self.cluster_shape_mn != (2, 1):
            raise ValueError(
                f"An alternate token tile runs a strided plan (N192, N240) at cluster (2, 1), not tile N "
                f"{self.cta_tile_n} at cluster {self.cluster_shape_mn}."
            )
        if (
            self.shared_expert_entry
            and self.shared_expert_tile_rows > self.alternate_tile_n
        ):
            # The shared expert's token tiles run in the launch's configuration, so
            # no launch can take an alternate they do not fit; without it the shared
            # tiles need not wait for the choice.
            self.alternate_tile_n = 0

    def _build_alternate_token_tile(
        self, problem_desc: ProblemDesc, resolved_impl_desc: ImplDesc
    ) -> _TokenTileConfig:
        """The mainloop, epilogue and SMEM plan of the alternate token tile (``alternate_tile_n``)."""
        tile_n = self.alternate_tile_n
        instruction_mnk = (
            self.mma_instruction_mnk[0],
            tile_n,
            self.mma_instruction_mnk[2],
        )
        tiler_mnk = (self.mma_tiler_mnk[0], tile_n, self.mma_tiler_mnk[2])
        mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=instruction_mnk,
            participates=2 if self.use_2cta_instrs else 1,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        impl_desc = ImplDesc(
            {
                **resolved_impl_desc,
                "mma_instruction_mnk": instruction_mnk,
                "mma_tiler_mnk": tiler_mnk,
                "tmem_plan": make_tcgen05_tmem_plan(
                    mma_instruction, self.architecture, tiler_mnk
                ),
                "token_tile_plan_n": self.cta_tile_n,
                "token_ab_stages": self.alternate_token_ab_stages,
                "weight_evict": self.alternate_weight_evict,
            }
        )
        epilogue = SwapABGatedActEpilogue(problem_desc, impl_desc)
        mainloop, smem_workspace = self._build_mainloop_and_smem(
            problem_desc, impl_desc, epilogue
        )
        primary_regions = {region.name for region in self._smem_workspace.regions()}
        for region in (self.token_tile_choice_region, self.token_pool_plan_region):
            if region in primary_regions and smem_workspace.offset(
                region
            ) != self._smem_workspace.offset(region):
                raise ValueError(
                    f"SMEM region {region} must sit at one offset in every configuration."
                )
        return _TokenTileConfig(tile_n, mainloop, epilogue, smem_workspace)

    def _smem_storage_class(self) -> type:
        """The SMEM allocation of the kernel: every token tile configuration lays its plan out from its base."""
        configs = (
            (self._primary,)
            if self._alternate is None
            else (self._primary, self._alternate)
        )
        return max(
            configs, key=lambda config: config.smem_workspace.total_bytes
        ).smem_workspace.storage_class()

    def _resolve_schedule_hint(self) -> None:
        if self.schedule_mode != "phase_interleave":
            return

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        cluster_feature_tile = (
            self.mma_tiler_mnk[0] // mma_cta_count * self.cluster_shape_mn[0]
        )
        blocks_fc1 = ceil_div(self.intermediate_gateup_size, cluster_feature_tile)
        blocks_fc2 = ceil_div(self.hidden_size, cluster_feature_tile)
        # Cover the worst token-block alignment of one full persistent-cluster FC2 claim wave.
        max_dependent_token_blocks = ceil_div(
            self.launch_cluster_count + blocks_fc2 - 1, blocks_fc2
        )
        required_fc1_work = max_dependent_token_blocks * blocks_fc1
        raw_minimum_hint = max(
            1, ceil_div(required_fc1_work, self.launch_cluster_count)
        )
        minimum_safe_hint = round_up(raw_minimum_hint, self.epi_flag_batches[0])
        if self.hint is None:
            self.hint = minimum_safe_hint
        elif self.hint < minimum_safe_hint:
            raise ValueError(
                f"phase_interleave hint {self.hint} is unsafe with "
                f"fc1_epi_flag_batch={self.epi_flag_batches[0]}; minimum legal hint is {minimum_safe_hint}."
            )

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def _build_device_workspace(self) -> DeviceWorkspace:
        intermediate_downproj = self.intermediate_gateup_size // 2
        sf_column_count = round_up(ceil_div(intermediate_downproj, self.sf_vec_size), 4)
        max_sf_rows = self.token_comm.worst_case_sf_rows
        # One slot per token block across every local expert, which is the space
        # `cumulative_token_block_count + tile_n_idx` indexes.
        counter_slot_count = (
            self.token_comm.worst_case_padded_rows(self.cta_tile_n) // self.cta_tile_n
        )

        if self.shared_expert:
            # The shared expert's FC1 outputs, scale factors and FC2 gates follow
            # the routed worst case.
            max_sf_rows += self.shared_expert_sf_rows
            counter_slot_count += self.shared_expert_token_tiles

        device_workspace = DeviceWorkspace()
        device_workspace.register(
            self.fc1_output_region,
            self.epilogue.fc1_output_dtype,
            (self.fc1_output_rows, intermediate_downproj),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.fc1_output_sf_region,
            self.epilogue.fc1_output_sf_dtype,
            (max_sf_rows, sf_column_count),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.fc1_done_counter_region,
            cutlass.Int32,
            (counter_slot_count,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )
        if not self.reduce_topk_in_kernel:
            # Peers write their contributions straight into this plane, one row per
            # (token, top-k slot), so it has to be symmetric. The standalone
            # TopkReduce launch folds it into the caller's output afterwards. A
            # shared expert adds slot ``topk``, which only this rank writes.
            device_workspace.register(
                self.pre_reduced_activation_region,
                self.combine_format.act_dtype,
                (
                    self.max_tokens_per_rank,
                    self.combine_slots_per_token,
                    self.hidden_size,
                ),
                buffer_space="shared",
                mem_order=(2, 1, 0),
                byte_alignment=128,
            )
        if self.reduce_topk_in_kernel:
            for region in (self.output_zero_claim_region, self.output_zero_done_region):
                device_workspace.register(
                    region,
                    cutlass.Int32,
                    (1,),
                    buffer_space="local",
                    byte_alignment=16,
                    reset="tail_reset",
                )
            # Peers add into this rank's word of the round's set; the end-of-round
            # reset clears it, and peers next add into this set two rounds later.
            device_workspace.register(
                self.output_zeroed_ranks_region,
                cutlass.Int32,
                (1,),
                buffer_space="shared",
                byte_alignment=16,
                reset="tail_reset",
            )
        if self.alternate_tile_n:
            device_workspace.register(
                self.token_tile_record_region, cutlass.Int32, (1,), buffer_space="local"
            )
        if self.token_pool_capacity:
            device_workspace.register(
                self.token_pool_record_region, cutlass.Int32, (2,), buffer_space="local"
            )
        self.scheduler.register_device_workspace(device_workspace)
        if self.tail_barrier is not None:
            self.tail_barrier.register_device_workspace(device_workspace)
        self.token_comm.register_device_workspace(device_workspace)
        device_workspace.finalize()
        # Two landing sets alternate by round. Their stride is a whole number of
        # dense activation rows, so one Gather4 descriptor spanning both sets
        # reaches set 1's row r at row r + gather_set_row_offset.
        data_row_bytes = (
            self.token_comm.plain_activation_row_elements
            * int(self.token_comm.activation_dtype.width)
            // 8
        )
        self.shared_set_bytes = round_up(
            device_workspace.total_bytes("shared"), math.lcm(4096, data_row_bytes)
        )
        self.gather_set_row_offset = self.shared_set_bytes // data_row_bytes
        self.token_comm.shared_set_bytes = self.shared_set_bytes
        # Regions whose bytes announce their own arrival; armed with 0xFF at allocation.
        landing_regions = [
            (self.token_comm.plain_activation_region, self.dispatch_protocol),
            (self.token_comm.plain_activation_sf_region, self.dispatch_protocol),
            (self.pre_reduced_activation_region, self.combine_protocol),
        ]
        self.sentinel_landing_regions = (
            tuple(
                region for region, protocol in landing_regions if protocol == "sentinel"
            )
            + self.token_comm.sentinel_inbox_regions
        )
        return device_workspace

    def _build_mainloop_and_smem(
        self,
        problem_desc: ProblemDesc,
        resolved_impl_desc: ImplDesc,
        epilogue: SwapABGatedActEpilogue,
    ) -> Tuple[BlockScaledSwapAbFc12MainloopGenSpecialized, SmemWorkspace]:
        """The mainloop of one token tile configuration and the SMEM plan it fills with the fixed components."""
        # The communication component builds its own SMEM workspace for its own
        # launch, so nothing of it is charged against this kernel's budget.
        smem_api = getattr(cutlass, "memory", utils)
        smem_limit = (
            smem_api.get_smem_capacity_in_bytes(self.architecture) // self.occupancy
        )
        smem_workspace = SmemWorkspace()
        if self.alternate_tile_n or self.token_pool_capacity:
            # Placed before every other region (mbarriers come first), so every
            # configuration finds the choice at offset 0 before it lays out the rest.
            smem_workspace.register_mbarrier(self.token_tile_choice_region, 1)
        if self.token_pool_capacity:
            # Right after it, for the same reason: one word for the capacity and one
            # per local expert, two to each eight-byte slot.
            smem_workspace.register_mbarrier(
                self.token_pool_plan_region, ceil_div(1 + self.local_expert_count, 2)
            )
        self.scheduler.register_smem_regions(smem_workspace)
        epilogue.register_smem_regions(smem_workspace)
        if self.claim_gate_compiled:
            smem_workspace.register_mbarrier(self.claim_gate_region, 1)
        fixed_component_bytes = smem_workspace.estimate_total_bytes()
        alignment_shift_bytes = max(
            region.byte_alignment for region in smem_workspace.regions()
        )
        mainloop_smem_budget_bytes = (
            smem_limit - fixed_component_bytes - alignment_shift_bytes
        )
        if mainloop_smem_budget_bytes <= 0:
            raise ValueError(
                "Fixed kernel components leave no SMEM budget for the Mainloop."
            )

        mainloop_impl_desc = ImplDesc(
            {
                **resolved_impl_desc,
                "mainloop_smem_budget_bytes": mainloop_smem_budget_bytes,
            }
        )
        mainloop = BlockScaledSwapAbFc12MainloopGenSpecialized(
            problem_desc, mainloop_impl_desc
        )

        mainloop.register_smem_regions(smem_workspace)
        smem_workspace.finalize(max_bytes=smem_limit)
        return mainloop, smem_workspace

    def get_workspace_sizes(self) -> Tuple[int, int]:
        return self._device_workspace.local_and_shared_bytes

    @property
    def require_zero_workspace_leading_bytes(self) -> Tuple[int, int]:
        return self._device_workspace.require_zero_workspace_leading_bytes

    def name(self) -> str:
        """Canonical encoding of every constexpr: the compiled-kernel cache key.

        Two candidates that differ in any construction knob must produce different names, otherwise a tuning sweep
        silently collapses to whichever one compiled first.
        """

        def dtype_name(dtype: type) -> str:
            return getattr(dtype, "__name__", str(dtype)).lower()

        instruction = "x".join(str(dimension) for dimension in self.mma_instruction_mnk)
        tile = "x".join(str(dimension) for dimension in self.mma_tiler_mnk)
        cluster = "x".join(str(dimension) for dimension in self.cluster_shape_mn)
        epi_flags = "x".join(str(batch) for batch in self.epi_flag_batches)
        alternate = ""
        if self._alternate is not None:
            alternate_mainloop = self._alternate.mainloop
            alternate = (
                f"_alt{self._alternate.tile_n}x{alternate_mainloop.num_weight_ab_stages}x"
                f"{alternate_mainloop.num_token_ab_stages}{alternate_mainloop.weight_evict}_choice{self.token_tile_choice}"
            )
        return (
            f"{self.architecture.replace('_', '')}_block_scaled_swap_ab_gen_phase_specialized_moe_{self.quant_kind}_"
            f"{dtype_name(self.a_dtype)}_{dtype_name(self.b_dtype)}_{dtype_name(self.acc_dtype)}_"
            f"sfvec{self.sf_vec_size}_a{self.a_major_mode.name.lower()}_b{self.b_major_mode.name.lower()}_"
            f"w{self.world_size}_e{self.expert_count}_topk{self.topk}_"
            f"topkidx{dtype_name(self.topk_index_dtype)}_"
            f"h{self.hidden_size}_i{self.intermediate_gateup_size}_maxtoken{self.max_tokens_per_rank}_"
            f"inst{instruction}_{self.mma_k_mode}_tile{tile}_"
            f"cluster{cluster}_{'2cta' if self.use_2cta_instrs else '1cta'}_"
            f"sched{self.schedule_mode}{'pipelined' if self.scheduler.pipelined_groups else ''}_hint{self.hint}_"
            f"pad{self.token_padding_block}x{self.sf_padding_block}_"
            f"work{self.work_id_mode}_"
            f"fc2store{'ublk' if self.epilogue.fc2_use_ublk else 'stg'}_"
            f"{'tokenpool' if self.fc1_token_pool else f'gatherwarps{self.active_gather_b_warp_count}'}"
            f"{f'gatherwarps{self.active_gather_b_warp_count}' if self.token_pool_capacity else ''}_"
            f"fc2tmastages{self.epilogue.fc2_tma_stages}_"
            f"epiflag{epi_flags}_clusters{self.launch_cluster_count}_"
            f"pusher{self.pusher_cta_count}_refine{self.refine_participant_groups}x{self.refine_output_stages}_"
            f"combine{self.combine_format.name}_clamp{self.gate_up_clamp}_"
            f"{'apply_topk_fc1' if self.apply_topk_at_fc1 else 'apply_topk_fc2'}_"
            f"{'inkernel_reduce' if self.reduce_topk_in_kernel else 'separate_reduce'}_"
            f"abstages{self._mainloop.num_weight_ab_stages}x{self._mainloop.num_token_ab_stages}_"
            f"weightevict{self._mainloop.weight_evict}_weights{self.weight_layout}{self.weight_block_rows}_"
            f"entryprefetch{self.entry_prefetch_bytes}_dispatch{self.dispatch_protocol}_"
            f"{f'claimlead{self.claim_lead_k_tiles}' if self.just_in_time_claims else 'claimeager'}_"
            f"combine{self.combine_protocol}{'discard' if self.discard_combine_slots else ''}"
            f"{'warpstores' if self.combine_warp_stores else ''}"
            f"{self.token_comm.variant_suffix()}{alternate}"
            f"{'_dielocal' if self.die_local_weights else ''}"
            f"{self._shared_expert_name()}"
        )

    def _shared_expert_name(self) -> str:
        """The shared expert's part of ``name()``: empty without one."""
        if not self.shared_expert:
            return ""
        tiles = self.shared_expert_tiles
        return (
            f"_shared{self.shared_expert}_clamp{self.shared_gate_up_clamp}"
            f"{'_bf16rounding' if self.shared_bf16_rounding else ''}"
            f"_fc2{'early' if tiles.fc2_early else 'tail'}"
            f"{'_entry' if tiles.entry_head else ''}"
            f"{f'_rows{tiles.tile_rows}' if self.shared_token_tile_rows is not None else ''}"
            f"{f'_head{tiles.head_token_tiles}' if tiles.head_token_tiles is not None else ''}"
        )

    def aot_compile(self, out_path: Optional[str] = None, **_compile_kwargs):
        """Compile against fake (metadata-only) inputs; ``out_path=None`` returns the in-memory callable.

        The fake tensors define the runtime ABI, so each mirrors what the caller stages: the local expert tensors,
        the padded payload strides the communication component advertises, and ``to_cute``'s
        ``mark_layout_dynamic``. Occupancy is a construction-time knob (``launch_cluster_count``), so this entry
        accepts no occupancy argument and tolerates leftover caller kwargs.
        """
        from cutlass.cute.runtime import (
            make_fake_compact_tensor,
            make_fake_stream,
            make_ptr,
        )
        from cutlass.cute.typing import AddressSpace, sym_int64

        from .....communication.nvlink_domain.symmetric_buffer import (
            SymmetricBufferHost,
        )

        def fake_tensor(dtype, shape, stride_order, dynamic_axes, alignment):
            extents = tuple(
                sym_int64(divisibility=math.gcd(int(extent), 128))
                if axis in dynamic_axes
                else int(extent)
                for axis, extent in enumerate(shape)
            )
            return make_fake_compact_tensor(
                dtype, extents, stride_order=stride_order, assumed_align=alignment
            )

        tokens = self.max_tokens_per_rank
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        experts = self.local_expert_count
        sf_vec_size = self.sf_vec_size
        # Weight SF is atom-swizzled and opaque, so its extent is whatever the caller staged.
        fc1_weight_sf_columns = round_up(intermediate_gateup, 128) * round_up(
            hidden // sf_vec_size, 4
        )
        fc2_weight_sf_columns = round_up(hidden, 128) * round_up(
            intermediate_downproj // sf_vec_size, 4
        )

        fake_arguments = dict(
            activation=fake_tensor(
                self.token_comm.activation_dtype,
                (tokens, self.token_comm.plain_activation_row_elements),
                (1, 0),
                {},
                16,
            ),
            activation_sf=fake_tensor(
                self.token_comm.activation_sf_dtype,
                (tokens, self.token_comm.plain_activation_sf_row_elements),
                (1, 0),
                {},
                16,
            ),
            topk_indices=fake_tensor(
                self.topk_index_dtype, (tokens, self.topk), (1, 0), {}, 16
            ),
            topk_scores=fake_tensor(
                cutlass.Float32, (tokens, self.topk), (1, 0), {}, 4
            ),
            fc1_weight=fake_tensor(
                self.a_dtype,
                (experts, hidden, intermediate_gateup),
                (2, 0, 1),
                {1, 2},
                16,
            ),
            fc1_weight_sf=fake_tensor(
                self.sf_dtype, (experts, fc1_weight_sf_columns), (1, 0), {1}, 16
            ),
            fc2_weight=fake_tensor(
                self.a_dtype,
                (experts, intermediate_downproj, hidden),
                (2, 0, 1),
                {1, 2},
                16,
            ),
            fc2_weight_sf=fake_tensor(
                self.sf_dtype, (experts, fc2_weight_sf_columns), (1, 0), {1}, 16
            ),
            output_activation=fake_tensor(
                cutlass.BFloat16, (tokens, hidden), (1, 0), {}, 16
            ),
            local_workspace=make_ptr(
                cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128
            ),
            shared_workspace=make_ptr(
                cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128
            ),
            # Placeholder field values: they marshal as runtime scalars, so their widths
            # are part of the ABI and bare Python ints would narrow the offsets to i32.
            # Only max_ranks is constexpr.
            peer_rank_ptr_mapper_host=SymmetricBufferHost(
                base_address=Int64(0),
                offsets=tuple(Int64(0) for _ in range(self.world_size)),
                rank=Int32(0),
                max_ranks=self.world_size,
            ),
            staging_peer_rank_ptr_mapper_host=SymmetricBufferHost(
                base_address=Int64(0),
                offsets=tuple(Int64(0) for _ in range(self.world_size)),
                rank=Int32(0),
                max_ranks=self.world_size,
            ),
            stream=make_fake_stream(),
        )
        if self.quant_kind.uses_global_scale:
            fake_arguments.update(
                fc1_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
                fc2_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
                fc1_norm_const=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
            )
        if self.shared_expert:
            shared_fc1_sf_columns = round_up(intermediate_gateup, 128) * round_up(
                hidden // sf_vec_size, 4
            )
            shared_fc2_sf_columns = round_up(hidden, 128) * round_up(
                intermediate_downproj // sf_vec_size, 4
            )
            fake_arguments.update(
                shared_fc1_weight=fake_tensor(
                    cutlass.Float8E4M3FN,
                    (1, hidden, intermediate_gateup),
                    (2, 0, 1),
                    {1, 2},
                    16,
                ),
                shared_fc1_weight_sf=fake_tensor(
                    self.sf_dtype, (1, shared_fc1_sf_columns), (1, 0), {1}, 16
                ),
                shared_fc2_weight=fake_tensor(
                    cutlass.Float8E4M3FN,
                    (1, intermediate_downproj, hidden),
                    (2, 0, 1),
                    {1, 2},
                    16,
                ),
                shared_fc2_weight_sf=fake_tensor(
                    self.sf_dtype, (1, shared_fc2_sf_columns), (1, 0), {1}, 16
                ),
            )

        compiled = cute.compile[cute.EnableTVMFFI(True)](self, **fake_arguments)
        if out_path is None:
            return compiled
        compiled.export_to_c(
            out_path, function_name=_aot_symbol_prefix, export_only_tvm_ffi_symbols=True
        )
        return out_path

    @staticmethod
    def load_compiled(path: str):
        from cutlass.cute.runtime import load_module

        return load_module(path, enable_tvm_ffi=True)[_aot_symbol_prefix]

    # K elements of one block-major weight block: one K128 AB stage. Its row count
    # (WEIGHT_BLOCK_ROWS, default 128 = one CTA's M share of the two-CTA MMA tile)
    # is a multiple of 128, so every TMA box stays one contiguous run.
    weight_block_k: ClassVar[int] = 128

    def _block_major_weight_view(
        self, weight: cute.Tensor, rows: int, cols: int, experts: int
    ) -> cute.Tensor:
        """(rows, cols, experts) A view of block-major weights.

        Physical order is (expert, row block, K block, row in block, K in block), so
        mode 0 is (row in block, row block) and mode 1 is (K in block, K block). The
        TMA box of one AB stage covers 128 rows of the two inner modes, which the
        existing A atom (U4 unpack included) encodes as a 5D tensor map.
        """
        block_rows, block_k = self.weight_block_rows, self.weight_block_k
        if rows % block_rows or cols % block_k:
            raise ValueError(
                f"block-major weights need ({rows}, {cols}) divisible by ({block_rows}, {block_k})."
            )
        return cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                ((block_rows, rows // block_rows), (block_k, cols // block_k), experts),
                stride=(
                    (block_k, block_rows * cols),
                    (1, block_rows * block_k),
                    weight.stride[0],
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Host entry
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,  # (max_tokens_per_rank, padded hidden), padded row stride
        activation_sf: cute.Tensor,  # (max_tokens_per_rank, padded hidden / sf_vec_size)
        topk_indices: cute.Tensor,  # (max_tokens_per_rank, topk)
        topk_scores: cute.Tensor,  # (max_tokens_per_rank, topk)
        fc1_weight: cute.Tensor,  # (local_experts, hidden, intermediate_gateup)
        fc1_weight_sf: cute.Tensor,  # (local_experts, packed_fc1_scale_factors)
        fc2_weight: cute.Tensor,  # (local_experts, intermediate_downproj, hidden)
        fc2_weight_sf: cute.Tensor,  # (local_experts, packed_fc2_scale_factors)
        output_activation: cute.Tensor,  # (max_tokens_per_rank, hidden)
        local_workspace: cute.Pointer,  # local GMEM byte workspace
        shared_workspace: cute.Pointer,  # symmetric GMEM byte workspace
        peer_rank_ptr_mapper_host,
        stream: cuda.CUstream,
        fc1_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc2_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc1_norm_const: Optional[cute.Tensor] = None,  # (local_experts,)
        # Peer offsets of the allocation holding `activation` and `activation_sf`;
        # the pull dispatch reads peers' rows through it.
        staging_peer_rank_ptr_mapper_host=None,
        # With a shared expert: its E4M3 weights as K-major (1, hidden,
        # intermediate_gateup) and (1, intermediate_downproj, hidden) views, gate
        # and up rows interleaved as the routed experts', and their atom-swizzled
        # UE8M0 scales (1, packed scale factors).
        shared_fc1_weight: Optional[cute.Tensor] = None,
        shared_fc1_weight_sf: Optional[cute.Tensor] = None,
        shared_fc2_weight: Optional[cute.Tensor] = None,
        shared_fc2_weight_sf: Optional[cute.Tensor] = None,
    ) -> None:
        """Launch generation-phase dispatch, fused FC12, and the optional top-k reduce."""

        def rewrite_tensor_shape(tensor: cute.Tensor, shape: Tuple) -> cute.Tensor:
            return cute.make_tensor(
                tensor.iterator, cute.make_layout(shape, stride=tensor.stride)
            )

        if cutlass.const_expr(topk_indices.element_type is not self.topk_index_dtype):
            raise TypeError(
                f"topk_indices must be the declared {self.topk_index_dtype}, got {topk_indices.dtype}."
            )
        for operand_name, operand, expected_dtype in (
            ("activation", activation, self.b_dtype),
            ("activation_sf", activation_sf, self.sf_dtype),
            ("fc1_weight", fc1_weight, self.a_dtype),
            ("fc1_weight_sf", fc1_weight_sf, self.sf_dtype),
            ("fc2_weight", fc2_weight, self.a_dtype),
            ("fc2_weight_sf", fc2_weight_sf, self.sf_dtype),
        ):
            if cutlass.const_expr(operand.element_type is not expected_dtype):
                raise TypeError(
                    f"{self.quant_kind} requires {operand_name} to be {expected_dtype}, got {operand.element_type}."
                )
        if cutlass.const_expr(self.quant_kind.uses_global_scale):
            for scalar_name, scalar in (
                ("fc1_alpha", fc1_alpha),
                ("fc2_alpha", fc2_alpha),
                ("fc1_norm_const", fc1_norm_const),
            ):
                if cutlass.const_expr(scalar is None):
                    raise ValueError(f"{self.quant_kind} requires {scalar_name}.")

        local_rank = peer_rank_ptr_mapper_host.rank
        self.token_comm.launch(
            GenphaseTokenCommArgs(
                topk_indices=topk_indices,
                topk_scores=topk_scores,
                activation=activation,
                activation_sf=activation_sf,
            ),
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host,
            self._device_workspace,
            stream,
            staging_peer_rank_ptr_mapper_host,
        )

        self._device_workspace.assign_device_members(local_workspace, shared_workspace)
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_object()
        canonical_activation_sf = self.token_comm.canonical_activation_sf_tensor(
            self._device_workspace
        )
        scheduler_expert_sizes = self.token_comm.scheduler_expert_sizes_tensor(
            self._device_workspace
        )
        fc1_output = self._device_workspace.tensor(self.fc1_output_region)
        fc1_output_sf = self._device_workspace.tensor(self.fc1_output_sf_region)

        experts = self.local_expert_count
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        # Die-local weights hold expert e's row half h as entry h * E + e.
        weight_experts, fc1_weight_rows, fc2_weight_rows = (
            experts,
            intermediate_gateup,
            hidden,
        )
        if cutlass.const_expr(self.die_local_weights):
            weight_experts, fc1_weight_rows, fc2_weight_rows = (
                2 * experts,
                intermediate_gateup // 2,
                hidden // 2,
            )
        fc1_weight = rewrite_tensor_shape(
            fc1_weight, (weight_experts, hidden, fc1_weight_rows)
        )
        fc2_weight = rewrite_tensor_shape(
            fc2_weight, (weight_experts, intermediate_downproj, fc2_weight_rows)
        )

        singleton = cutlass.Int32(1)
        sf_vec_size = self.sf_vec_size

        if cutlass.const_expr(self.weight_layout == "blockmajor"):
            fc1_a = self._block_major_weight_view(
                fc1_weight, fc1_weight_rows, hidden, weight_experts
            )
        else:
            fc1_a = cute.make_tensor(
                fc1_weight.iterator,
                cute.make_layout(
                    (
                        cutlass.Int32(intermediate_gateup),
                        cutlass.Int32(hidden),
                        experts,
                    ),
                    stride=(
                        fc1_weight.stride[2],
                        fc1_weight.stride[1],
                        fc1_weight.stride[0],
                    ),
                ),
            )
        # FC1's B is the dense (source rank, source token) plane the communication
        # kernel filled, indexed by `gather_index` rather than by a pool row, so it
        # carries no per-expert offset; under a token pool of every data row it is
        # the pool, whose rows are the data rows of the FC1 output pool. Under a
        # pool with a capacity it stays the plane, and the pool, one row per route,
        # is a second B tensor.
        # Gather4 always issues a 2D TMA instruction. A static singleton row
        # mode is otherwise dropped from the descriptor at EP1 capacity one.
        # Keep that extent dynamic so the descriptor retains both dimensions.
        if cutlass.const_expr(self.fc1_token_pool and not self.token_pool_capacity):
            fc1_b_plane = self.token_comm.fc1_token_pool_tensor(self._device_workspace)
            fc1_b_rows = cutlass.Int32(fc1_b_plane.shape[0])
        else:
            fc1_b_plane = self.token_comm.dense_activation_tensor(
                self._device_workspace
            )
            fc1_b_rows = self.gather_set_row_offset + fc1_b_plane.shape[0]
        fc1_b = cute.make_tensor(
            fc1_b_plane.iterator,
            cute.make_layout(
                (fc1_b_rows, cutlass.Int32(hidden), singleton),
                stride=(fc1_b_plane.stride[0], fc1_b_plane.stride[1], 0),
            ),
        )
        fc1_b_pool = None
        if cutlass.const_expr(self.token_pool_capacity):
            pool_plane = self.token_comm.fc1_token_pool_tensor(self._device_workspace)
            fc1_b_pool = cute.make_tensor(
                pool_plane.iterator,
                cute.make_layout(
                    (
                        cutlass.Int32(pool_plane.shape[0]),
                        cutlass.Int32(hidden),
                        singleton,
                    ),
                    stride=(pool_plane.stride[0], pool_plane.stride[1], 0),
                ),
            )
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (self.fc1_output_rows, cutlass.Int32(intermediate_downproj), singleton),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )
        # The canonical scale-factor pool already carries the MMA's atom layout, so
        # FC1's SFB leg is an ordinary tiled TMA over it (with a shared expert, over
        # its rows past the routed ones too).
        fc1_sfb = cute.make_tensor(
            canonical_activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    self.token_comm.worst_case_sf_rows
                    + (self.shared_expert_sf_rows if self.shared_expert else 0),
                    cutlass.Int32(hidden),
                    singleton,
                ),
                sf_vec_size,
            ),
        )
        padded_intermediate_gateup = cute.round_up(fc1_weight_rows, sf_vec_size * 4)
        expected_fc1_weight_sf_columns = (
            padded_intermediate_gateup * hidden // sf_vec_size
        )
        if cutlass.const_expr(
            isinstance(fc1_weight_sf.shape[1], int)
            and isinstance(expected_fc1_weight_sf_columns, int)
        ):
            if cutlass.const_expr(
                fc1_weight_sf.shape[1] != expected_fc1_weight_sf_columns
            ):
                raise ValueError("fc1_weight_sf has an incompatible column count.")
        fc1_sfa = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    cutlass.Int32(padded_intermediate_gateup),
                    cutlass.Int32(hidden),
                    cutlass.Int32(weight_experts),
                ),
                sf_vec_size,
            ),
        )

        experts_fc2, intermediate_fc2, hidden_fc2 = fc2_weight.shape
        if cutlass.const_expr(self.weight_layout == "blockmajor"):
            fc2_a = self._block_major_weight_view(
                fc2_weight, fc2_weight_rows, intermediate_downproj, weight_experts
            )
        else:
            fc2_a = cute.make_tensor(
                fc2_weight.iterator,
                cute.make_layout(
                    (
                        cutlass.Int32(hidden_fc2),
                        cutlass.Int32(intermediate_fc2),
                        experts_fc2,
                    ),
                    stride=(
                        fc2_weight.stride[2],
                        fc2_weight.stride[1],
                        fc2_weight.stride[0],
                    ),
                ),
            )
        fc2_b = fc1_output_gemm
        padded_fc2_hidden = cute.round_up(hidden_fc2, 128)
        padded_fc2_intermediate = cute.round_up(intermediate_fc2, sf_vec_size * 4)
        expected_fc2_weight_sf_columns = (
            padded_fc2_hidden * padded_fc2_intermediate // sf_vec_size
        )
        if cutlass.const_expr(
            isinstance(fc2_weight_sf.shape[1], int)
            and isinstance(expected_fc2_weight_sf_columns, int)
        ):
            if cutlass.const_expr(
                fc2_weight_sf.shape[1] != expected_fc2_weight_sf_columns
            ):
                raise ValueError("fc2_weight_sf has an incompatible column count.")
        fc2_sfa = cute.make_tensor(
            fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    cutlass.Int32(padded_fc2_hidden),
                    cutlass.Int32(padded_fc2_intermediate),
                    cutlass.Int32(experts_fc2),
                ),
                sf_vec_size,
            ),
        )
        fc2_sfb = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    fc1_output_sf.shape[0],
                    cutlass.Int32(fc1_output_sf.shape[1] * sf_vec_size),
                    singleton,
                ),
                sf_vec_size,
            ),
        )

        if cutlass.const_expr(self.reduce_topk_in_kernel):
            # Every top-k contribution reduces into the topk=0 plane, so the caller's
            # output doubles as the (token, topk, hidden) destination. It has to live
            # in the symmetric heap: peers reach it through the per-rank offset of the
            # staging allocation that holds it.
            pre_reduced_activation = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (self.max_tokens_per_rank, 1, hidden),
                    stride=(
                        output_activation.stride[0],
                        output_activation.stride[0],
                        output_activation.stride[1],
                    ),
                ),
            )
        else:
            pre_reduced_activation = self._device_workspace.tensor(
                self.pre_reduced_activation_region
            )
        fc2_output = rewrite_tensor_shape(
            pre_reduced_activation,
            (pre_reduced_activation.shape[0], pre_reduced_activation.shape[1], hidden),
        )

        self._mainloop.materialize_codegen_members()
        (
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_atom,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom,
            fc2_tma_b_tensor,
            fc2_tma_b_atom,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom,
            fc2_tma_b_rows_atom,
            fc1_tma_b_rows_atom,
            fc1_tma_b_pool_atom,
        ) = self._mainloop.prepare_tma_load_params(
            fc1_a=fc1_a,
            fc1_b=fc1_b,
            fc1_sfa=fc1_sfa,
            fc1_sfb=fc1_sfb,
            fc2_a=fc2_a,
            fc2_b=fc2_b,
            fc2_sfa=fc2_sfa,
            fc2_sfb=fc2_sfb,
            fc1_b_pool=fc1_b_pool,
        )

        # FC2 stores through the peer-facing bulk path, so only the FC1 store atom
        # comes back populated.
        (fc1_output_tma_atom, fc1_output_tma_tensor, _, _) = (
            self.epilogue.prepare_tma_store_params(fc1_output_gemm, fc2_output)
        )
        # The alternate token tile's descriptors over the same tensors: its B and
        # SFB boxes follow its own tile N.
        alternate_tma = (None,) * 20
        if cutlass.const_expr(self._alternate is not None):
            self._alternate.mainloop.materialize_codegen_members()
            alternate_tma = (
                self._alternate.mainloop.prepare_tma_load_params(
                    fc1_a=fc1_a,
                    fc1_b=fc1_b,
                    fc1_sfa=fc1_sfa,
                    fc1_sfb=fc1_sfb,
                    fc2_a=fc2_a,
                    fc2_b=fc2_b,
                    fc2_sfa=fc2_sfa,
                    fc2_sfb=fc2_sfb,
                    fc1_b_pool=fc1_b_pool,
                )
                + self._alternate.epilogue.prepare_tma_store_params(
                    fc1_output_gemm, fc2_output
                )[:2]
            )

        # The shared expert's weight and token descriptors, per token tile configuration.
        shared_tma = (None,) * 10
        alternate_shared_tma = (None,) * 10
        if cutlass.const_expr(self.shared_expert):
            if cutlass.const_expr(
                any(
                    t is None
                    for t in (
                        shared_fc1_weight,
                        shared_fc1_weight_sf,
                        shared_fc2_weight,
                        shared_fc2_weight_sf,
                    )
                )
            ):
                raise ValueError(
                    "A shared expert needs its FC1 and FC2 weights and scale factors."
                )
            for operand_name, operand, expected_dtype in (
                ("shared_fc1_weight", shared_fc1_weight, cutlass.Float8E4M3FN),
                ("shared_fc1_weight_sf", shared_fc1_weight_sf, self.sf_dtype),
                ("shared_fc2_weight", shared_fc2_weight, cutlass.Float8E4M3FN),
                ("shared_fc2_weight_sf", shared_fc2_weight_sf, self.sf_dtype),
            ):
                if cutlass.const_expr(operand.element_type is not expected_dtype):
                    raise TypeError(
                        f"{operand_name} must be {expected_dtype}, got {operand.element_type}."
                    )
            shared_views = dict(
                fc1_a=cute.make_tensor(
                    shared_fc1_weight.iterator,
                    cute.make_layout(
                        (
                            cutlass.Int32(intermediate_gateup),
                            cutlass.Int32(hidden),
                            singleton,
                        ),
                        stride=(
                            shared_fc1_weight.stride[2],
                            shared_fc1_weight.stride[1],
                            shared_fc1_weight.stride[0],
                        ),
                    ),
                ),
                fc1_sfa=cute.make_tensor(
                    shared_fc1_weight_sf.iterator,
                    blockscaled_utils.tile_atom_to_shape_SF(
                        (
                            cutlass.Int32(
                                cute.round_up(intermediate_gateup, sf_vec_size * 4)
                            ),
                            cutlass.Int32(hidden),
                            singleton,
                        ),
                        sf_vec_size,
                    ),
                ),
                fc2_a=cute.make_tensor(
                    shared_fc2_weight.iterator,
                    cute.make_layout(
                        (
                            cutlass.Int32(hidden),
                            cutlass.Int32(intermediate_downproj),
                            singleton,
                        ),
                        stride=(
                            shared_fc2_weight.stride[2],
                            shared_fc2_weight.stride[1],
                            shared_fc2_weight.stride[0],
                        ),
                    ),
                ),
                fc2_sfa=cute.make_tensor(
                    shared_fc2_weight_sf.iterator,
                    blockscaled_utils.tile_atom_to_shape_SF(
                        (
                            cutlass.Int32(cute.round_up(hidden, 128)),
                            cutlass.Int32(
                                cute.round_up(intermediate_downproj, sf_vec_size * 4)
                            ),
                            singleton,
                        ),
                        sf_vec_size,
                    ),
                ),
                # The staged input, whose token t is the shared expert's token t.
                token_x=cute.make_tensor(
                    activation.iterator,
                    cute.make_layout(
                        (
                            cutlass.Int32(self.max_tokens_per_rank),
                            cutlass.Int32(hidden),
                        ),
                        stride=(activation.stride[0], 1),
                    ),
                ),
            )
            shared_tma = self._mainloop.prepare_shared_tma_load_params(**shared_views)
            if cutlass.const_expr(self._alternate is not None):
                alternate_shared_tma = (
                    self._alternate.mainloop.prepare_shared_tma_load_params(
                        **shared_views
                    )
                )

        # The pull warps read peers' staged rows through the staging allocation's offsets.
        pull_activation = None
        pull_activation_sf = None
        staging_peer_rank_ptr_mapper = None
        if cutlass.const_expr(self.pull_warps > 0 or self.reduce_topk_in_kernel):
            if cutlass.const_expr(staging_peer_rank_ptr_mapper_host is None):
                raise ValueError(
                    "The pull warps and the in-kernel top-k reduce require the staging allocation's peer offsets."
                )
            staging_peer_rank_ptr_mapper = (
                staging_peer_rank_ptr_mapper_host.make_device_object()
            )
        if cutlass.const_expr(self.pull_warps > 0):
            pull_activation = activation
            pull_activation_sf = activation_sf

        grid = self.scheduler.get_grid_shape(
            max_active_clusters=self.launch_cluster_count
        )
        self._device_workspace.remove_device_members()
        self._kernel(
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_atom,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom,
            fc2_tma_b_tensor,
            fc2_tma_b_atom,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom,
            fc2_tma_b_rows_atom,
            fc1_tma_b_rows_atom,
            fc1_tma_b_pool_atom,
            fc1_output_tma_atom,
            fc1_output_tma_tensor,
            fc2_output,
            (experts, intermediate_gateup, hidden),
            scheduler_expert_sizes,
            peer_rank_ptr_mapper,
            local_workspace,
            shared_workspace,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            fc1_weight.iterator.toint(),
            fc1_weight_sf.iterator.toint(),
            pull_activation,
            pull_activation_sf,
            staging_peer_rank_ptr_mapper,
            *alternate_tma,
            *shared_tma,
            *alternate_shared_tma,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
            use_pdl=True,
        )
        self._topk_reduce(
            pre_reduced_activation,
            output_activation,
            topk_indices,
            local_rank,
            local_workspace,
            shared_workspace,
            self._device_workspace,
            stream,
            topk_scores=topk_scores
            if cutlass.const_expr(self._topk_reduce.weight_routes)
            else None,
        )

    @cute.jit
    def _claims_pay_just_in_time(
        self, fc1_tile_count: Int32, fc2_tile_count: Int32
    ) -> Boolean:
        """Whether a launch with these FC1 and FC2 cluster tile counts claims just in time.

        The clusters whose SMs the communication grid holds start late; with eager
        claims each of the others holds ``eager_claims_before_first_fc1`` tiles by
        the first FC1 tile. Just-in-time claims pay when the launch has more tiles
        than that, which would reach the late clusters as FC2 tiles only, or when
        spreading the FC1 tiles over every cluster gives each fewer of them.
        """
        launch_clusters = self.launch_cluster_count
        late_clusters = min(
            ceil_div(self.token_comm.grid_cta_count, self.cluster_size),
            launch_clusters - 1,
        )
        early_clusters = launch_clusters - late_clusters
        exceeds_early_claims = fc1_tile_count + fc2_tile_count > Int32(
            self.eager_claims_before_first_fc1 * early_clusters
        )
        early_fc1_rounds = (fc1_tile_count + Int32(early_clusters - 1)) // Int32(
            early_clusters
        )
        all_fc1_rounds = (fc1_tile_count + Int32(launch_clusters - 1)) // Int32(
            launch_clusters
        )
        return exceeds_early_claims | (early_fc1_rounds > all_fc1_rounds)

    @cute.jit
    def _first_fc1_stamp_pointer(self, linear_cta_idx: Int32) -> Optional[cute.Pointer]:
        """Where TMA-B records its first FC1 gate pass, or None without stamps."""
        if cutlass.const_expr(not self.token_comm.stamps):
            return None
        stamps = self._device_workspace.tensor(self.token_comm.stamps_region)
        return (
            stamps[self.token_comm.stamp_main_first_fc1, None].iterator + linear_cta_idx
        )

    # Bytes of one entry-prefetch request; each lane of an epilogue warp issues one.
    entry_prefetch_chunk_bytes: ClassVar[int] = 32 * 1024

    @cute.jit
    def _prefetch_entry(
        self,
        fc1_weight_address: Int64,
        fc1_weight_sf_address: Int64,
        epilogue_warp: Int32,
        linear_cta_idx: Int32,
        lane: Int32,
    ) -> None:
        """One epilogue warp's share of the entry weight prefetch, per die under die-local weights."""
        if cutlass.const_expr(self.die_local_weights):
            self._prefetch_entry_die_local_weights(
                fc1_weight_address, fc1_weight_sf_address, epilogue_warp, lane
            )
        else:
            self._prefetch_entry_weights(
                fc1_weight_address,
                fc1_weight_sf_address,
                linear_cta_idx * Int32(len(self.epilogue_warp_ids)) + epilogue_warp,
                lane,
            )

    @cute.jit
    def _prefetch_entry_weights(
        self,
        fc1_weight_address: Int64,
        fc1_weight_sf_address: Int64,
        global_warp: Int32,
        lane: Int32,
    ) -> None:
        """Prefetch leading FC1 weights into L2 while the dispatched payload lands.

        FC1 cannot start before an expert's dispatched rows arrive, but its weights
        do not depend on them, so HBM would otherwise idle through the dispatch.
        The prefetch is issued at kernel entry, before the processing order is
        known (waiting for it leaves too little of the window), so it covers the
        local experts in index order: each expert's FC1 weights, then its scale
        factors, up to ``entry_prefetch_bytes``. Lines it brings in keep the normal
        eviction priority while the weight stream marks its own evict-first, so
        they survive until their expert's turn whatever that turn is. Every
        epilogue warp of the grid takes an interleaved share of 32 KiB chunks.
        Experts are stored contiguously, block-major or K-major alike.
        """
        weight_bytes = (
            self.hidden_size
            * self.intermediate_gateup_size
            * int(self.a_dtype.width)
            // 8
        )
        sf_atom_columns = self.sf_vec_size * 4
        padded_rows = (
            (self.intermediate_gateup_size + sf_atom_columns - 1)
            // sf_atom_columns
            * sf_atom_columns
        )
        sf_bytes = (
            padded_rows
            * self.hidden_size
            // self.sf_vec_size
            * int(self.sf_dtype.width)
            // 8
        )
        chunk = self.entry_prefetch_chunk_bytes
        if cutlass.const_expr(weight_bytes % chunk != 0):
            raise ValueError(
                "Entry prefetch needs whole chunks of an expert's FC1 weights."
            )
        weight_chunks = weight_bytes // chunk
        chunks_per_expert = weight_chunks + (sf_bytes + chunk - 1) // chunk
        total_chunks = min(
            self.entry_prefetch_bytes // chunk,
            chunks_per_expert * self.local_expert_count,
        )
        grid_warps = self.promised_launchable_sm_count * len(self.epilogue_warp_ids)
        for round_idx in cutlass.range_constexpr(
            (total_chunks + grid_warps * 32 - 1) // (grid_warps * 32)
        ):
            chunk_idx = (Int32(round_idx * 32) + lane) * Int32(grid_warps) + global_warp
            if chunk_idx < Int32(total_chunks):
                expert = chunk_idx // Int32(chunks_per_expert)
                within = chunk_idx - expert * Int32(chunks_per_expert)
                if within < Int32(weight_chunks):
                    sentinel_ptx.prefetch_l2_bulk(
                        fc1_weight_address
                        + Int64(expert) * Int64(weight_bytes)
                        + Int64(within) * Int64(chunk),
                        Int32(chunk),
                    )
                else:
                    offset = (within - Int32(weight_chunks)) * Int32(chunk)
                    sentinel_ptx.prefetch_l2_bulk(
                        fc1_weight_sf_address
                        + Int64(expert) * Int64(sf_bytes)
                        + Int64(offset),
                        cutlass.min(Int32(chunk), Int32(sf_bytes) - offset),
                    )

    @cute.jit
    def _prefetch_entry_die_local_weights(
        self,
        fc1_weight_address: Int64,
        fc1_weight_sf_address: Int64,
        epilogue_warp: Int32,
        lane: Int32,
    ) -> None:
        """``_prefetch_entry_weights`` over die-local weights: each die prefetches its own half.

        A die's epilogue warps take interleaved 32 KiB chunks of its half of each
        local expert's FC1 weights, then of that half's scale factors, experts in
        index order, up to half of ``entry_prefetch_bytes`` per die. A warp's share
        follows its SM's rank among the die's SMs (``sm_die_table``), so the lines
        enter the L2 of the die that homes them and later reads them.
        """
        weight_bytes = (
            self.hidden_size
            * self.intermediate_gateup_size
            // 2
            * int(self.a_dtype.width)
            // 8
        )
        sf_bytes = (
            self.intermediate_gateup_size
            // 2
            * self.hidden_size
            // self.sf_vec_size
            * int(self.sf_dtype.width)
            // 8
        )
        chunk = self.entry_prefetch_chunk_bytes
        if cutlass.const_expr(weight_bytes % chunk != 0):
            raise ValueError(
                "Entry prefetch needs whole chunks of an expert's FC1 weight half."
            )
        weight_chunks = weight_bytes // chunk
        chunks_per_expert = weight_chunks + (sf_bytes + chunk - 1) // chunk
        total_chunks = min(
            self.entry_prefetch_bytes // 2 // chunk,
            chunks_per_expert * self.local_expert_count,
        )
        table = self._device_workspace.tensor(self.scheduler.sm_die_table_region)
        slots = self.scheduler.sm_die_table_slots
        sm = sentinel_ptx.smid()
        die = table[sm]
        warps_per_sm = len(self.epilogue_warp_ids)
        die_warp = table[slots + sm] * Int32(warps_per_sm) + epilogue_warp
        die_warps = table[2 * slots + die] * Int32(warps_per_sm)
        weight_base = fc1_weight_address + Int64(die) * Int64(
            self.local_expert_count * weight_bytes
        )
        sf_base = fc1_weight_sf_address + Int64(die) * Int64(
            self.local_expert_count * sf_bytes
        )
        chunk_idx = lane * die_warps + die_warp
        while chunk_idx < Int32(total_chunks):
            expert = chunk_idx // Int32(chunks_per_expert)
            within = chunk_idx - expert * Int32(chunks_per_expert)
            if within < Int32(weight_chunks):
                sentinel_ptx.prefetch_l2_bulk(
                    weight_base
                    + Int64(expert) * Int64(weight_bytes)
                    + Int64(within) * Int64(chunk),
                    Int32(chunk),
                )
            else:
                offset = (within - Int32(weight_chunks)) * Int32(chunk)
                sentinel_ptx.prefetch_l2_bulk(
                    sf_base + Int64(expert) * Int64(sf_bytes) + Int64(offset),
                    cutlass.min(Int32(chunk), Int32(sf_bytes) - offset),
                )
            chunk_idx = chunk_idx + Int32(32) * die_warps

    # ------------------------------------------------------------------
    # Device entry
    # ------------------------------------------------------------------

    @cute.jit
    def _zero_output(
        self,
        output_address: Int64,
        lane: Int32,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
    ) -> None:
        """One epilogue warp's share of zeroing the output under the in-kernel top-k reduce.

        Every peer's FC2 epilogue adds into this rank's output, so the output has
        to be zero before any peer's first add, and its previous contents are the
        caller's until this launch. Epilogue warps claim 32 KiB chunks at entry, as
        they arrive, so CTAs the communication grid keeps off their SMs claim none;
        each zeroes its chunks with 16-byte stores and adds them to the done count
        with a release. The warp whose add completes the count publishes the zeroed
        output to every rank with a system-scope add into that rank's word of the
        round's set, which every epilogue thread acquires before its first tile
        (``_wait_outputs_zeroed``). The zeroing thus overlaps the plan.
        """
        chunk = self.output_zero_chunk_bytes
        total_bytes = self.max_tokens_per_rank * self.hidden_size * 2
        total_chunks = (total_bytes + chunk - 1) // chunk
        claim = self._device_workspace.ptr(self.output_zero_claim_region).toint()
        done = self._device_workspace.ptr(self.output_zero_done_region).toint()
        zeroed_chunks = Int32(0)
        index = Int32(0)
        if lane == Int32(0):
            index = Int32(
                sentinel_ptx.atom_add_acq_rel_gpu_u32(claim, cutlass.Uint32(1))
            )
        index = cute.arch.shuffle_sync(index, 0)
        while index < Int32(total_chunks):
            base = output_address + Int64(index) * Int64(chunk)
            valid = Int32(total_bytes) - index * Int32(chunk)
            for vector in cutlass.range_constexpr(chunk // 512):
                offset = Int32(vector * 512) + lane * Int32(16)
                if offset < valid:
                    sentinel_ptx.st_global_v4_words(
                        base + Int64(offset),
                        cutlass.Uint32(0),
                        cutlass.Uint32(0),
                        cutlass.Uint32(0),
                        cutlass.Uint32(0),
                    )
            zeroed_chunks += Int32(1)
            if lane == Int32(0):
                index = Int32(
                    sentinel_ptx.atom_add_acq_rel_gpu_u32(claim, cutlass.Uint32(1))
                )
            index = cute.arch.shuffle_sync(index, 0)
        if zeroed_chunks > Int32(0):
            # Each lane orders its stores before the count; the warp's last add of
            # the round then sees every warp's stores through the acquire.
            sentinel_ptx.fence_acq_rel_gpu()
            cute.arch.sync_warp()
            if lane == Int32(0):
                previous = Int32(
                    sentinel_ptx.atom_add_acq_rel_gpu_u32(
                        done, cutlass.Uint32(zeroed_chunks)
                    )
                )
                if previous + zeroed_chunks == Int32(total_chunks):
                    zeroed = self._device_workspace.ptr(
                        self.output_zeroed_ranks_region
                    ).toint()
                    sentinel_ptx.fence_acq_rel_sys()
                    for rank in cutlass.range_constexpr(self.world_size):
                        red_add_relaxed_sys_s32(
                            peer_rank_ptr_mapper.map(zeroed, Int32(rank)), Int32(1)
                        )

    @cute.jit
    def _wait_gather_input_ready(
        self, warp_idx: Int32, thread_idx: Int32, linear_cta_idx: Int32
    ) -> None:
        """A Gather-B warp's wait for the index lists, and under the whole-payload Fence dispatch for the payload.

        ``gather_index`` is local, published by this rank's index-copy CTAs (the
        planners under the Sentinel dispatch); the payload flag is written by
        peers, hence the wider scope. Per-expert and per-tile readiness is
        TMA-B's gate, which it passes before releasing the Gather4 issue barrier.
        """
        iket.range_push("gather_b1.wait_input_ready")
        if cutlass.const_expr(
            self.dispatch_protocol == "counter" and not self.token_comm.round_gated
        ):
            spin_wait(
                self._device_workspace.ptr(self.token_comm.token_data_ready_region),
                lambda value: value == Int32(self.token_comm.payload_ready_target),
                scope="sys",
            )
        spin_wait(
            self._device_workspace.ptr(self.token_comm.metadata_ready_region),
            lambda value: value == Int32(self.token_comm.metadata_ready_target),
            scope="gpu",
        )
        iket.range_pop()
        if cutlass.const_expr(self.token_comm.stamps):
            if (warp_idx == Int32(self.gather_b_warp_ids[0])) & (
                thread_idx % Int32(32) == Int32(0)
            ):
                self.token_comm.stamp(
                    self._device_workspace,
                    self.token_comm.stamp_main_gather_open,
                    linear_cta_idx,
                )

    @cute.jit
    def _wait_metadata_ready(self) -> None:
        """Acquire this rank's index lists, routing records and top-k scores (``metadata_ready``)."""
        iket.range_push("epilogue.wait_metadata_ready")
        spin_wait(
            self._device_workspace.ptr(self.token_comm.metadata_ready_region),
            lambda value: value == Int32(self.token_comm.metadata_ready_target),
            scope="gpu",
        )
        iket.range_pop()

    @cute.jit
    def _wait_outputs_zeroed(self) -> None:
        """Acquire every rank's zeroed output (``_zero_output``) before this thread's first tile adds into one."""
        spin_wait(
            self._device_workspace.ptr(self.output_zeroed_ranks_region),
            lambda value: value == Int32(self.world_size),
            scope="sys",
        )

    @cute.kernel
    def _kernel(
        self,
        fc1_tma_a_tensor: cute.Tensor,
        fc1_tma_a_atom: cute.CopyAtom,
        fc1_tma_sfa_tensor: cute.Tensor,
        fc1_tma_sfa_atom: cute.CopyAtom,
        fc2_tma_a_tensor: cute.Tensor,
        fc2_tma_a_atom: cute.CopyAtom,
        fc2_tma_sfa_tensor: cute.Tensor,
        fc2_tma_sfa_atom: cute.CopyAtom,
        fc1_tma_b_atom: cute.CopyAtom,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom: cute.CopyAtom,
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom: cute.CopyAtom,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom: cute.CopyAtom,
        # FC2's live-row token descriptor (the mainloop's ``fc2_b_live_rows``), else None.
        fc2_tma_b_rows_atom: Optional[cute.CopyAtom],
        # FC1's live-row token descriptor (the mainloop's ``fc1_b_live_rows``), else None.
        fc1_tma_b_rows_atom: Optional[cute.CopyAtom],
        # Under a token pool with a capacity, FC1's descriptor over the pool, whose
        # box is the CTA's half of the tile N (``fc1_tma_b_atom`` then gathers from
        # the landing slots), else None.
        fc1_tma_b_pool_atom: Optional[cute.CopyAtom],
        fc1_output_tma_atom: cute.CopyAtom,
        fc1_output_tma_tensor: cute.Tensor,
        fc2_output: cute.Tensor,
        actual_expert_shape: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        # Per-expert sizes in the order the scheduler walks them, which is the
        # slot order rather than the expert order.
        scheduler_expert_sizes: cute.Tensor,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        fc1_weight_address: Int64,
        fc1_weight_sf_address: Int64,
        # Pull warps only: this rank's staged activations and scale factors, and
        # the peer offsets of their allocation.
        pull_activation: Optional[cute.Tensor],
        pull_activation_sf: Optional[cute.Tensor],
        staging_peer_rank_ptr_mapper: Optional[SymmetricBufferDevice],
        # The alternate token tile's TMA atoms and tensors (``alternate_tile_n``),
        # each the counterpart of the primary's of the same name, else None.
        alternate_fc1_tma_a_tensor: Optional[cute.Tensor],
        alternate_fc1_tma_a_atom: Optional[cute.CopyAtom],
        alternate_fc1_tma_sfa_tensor: Optional[cute.Tensor],
        alternate_fc1_tma_sfa_atom: Optional[cute.CopyAtom],
        alternate_fc2_tma_a_tensor: Optional[cute.Tensor],
        alternate_fc2_tma_a_atom: Optional[cute.CopyAtom],
        alternate_fc2_tma_sfa_tensor: Optional[cute.Tensor],
        alternate_fc2_tma_sfa_atom: Optional[cute.CopyAtom],
        alternate_fc1_tma_b_atom: Optional[cute.CopyAtom],
        alternate_fc1_tma_sfb_tensor: Optional[cute.Tensor],
        alternate_fc1_tma_sfb_atom: Optional[cute.CopyAtom],
        alternate_fc2_tma_b_tensor: Optional[cute.Tensor],
        alternate_fc2_tma_b_atom: Optional[cute.CopyAtom],
        alternate_fc2_tma_sfb_tensor: Optional[cute.Tensor],
        alternate_fc2_tma_sfb_atom: Optional[cute.CopyAtom],
        alternate_fc2_tma_b_rows_atom: Optional[cute.CopyAtom],
        alternate_fc1_tma_b_rows_atom: Optional[cute.CopyAtom],
        alternate_fc1_tma_b_pool_atom: Optional[cute.CopyAtom],
        alternate_fc1_output_tma_atom: Optional[cute.CopyAtom],
        alternate_fc1_output_tma_tensor: Optional[cute.Tensor],
        # A shared expert's TMA atoms and tensors (``prepare_shared_tma_load_params``)
        # for the primary configuration and for the alternate one, else None.
        shared_fc1_tma_a_tensor: Optional[cute.Tensor],
        shared_fc1_tma_a_atom: Optional[cute.CopyAtom],
        shared_fc1_tma_sfa_tensor: Optional[cute.Tensor],
        shared_fc1_tma_sfa_atom: Optional[cute.CopyAtom],
        shared_fc2_tma_a_tensor: Optional[cute.Tensor],
        shared_fc2_tma_a_atom: Optional[cute.CopyAtom],
        shared_fc2_tma_sfa_tensor: Optional[cute.Tensor],
        shared_fc2_tma_sfa_atom: Optional[cute.CopyAtom],
        shared_tma_x_atom: Optional[cute.CopyAtom],
        shared_tma_x_rows_atom: Optional[cute.CopyAtom],
        alternate_shared_fc1_tma_a_tensor: Optional[cute.Tensor],
        alternate_shared_fc1_tma_a_atom: Optional[cute.CopyAtom],
        alternate_shared_fc1_tma_sfa_tensor: Optional[cute.Tensor],
        alternate_shared_fc1_tma_sfa_atom: Optional[cute.CopyAtom],
        alternate_shared_fc2_tma_a_tensor: Optional[cute.Tensor],
        alternate_shared_fc2_tma_a_atom: Optional[cute.CopyAtom],
        alternate_shared_fc2_tma_sfa_tensor: Optional[cute.Tensor],
        alternate_shared_fc2_tma_sfa_atom: Optional[cute.CopyAtom],
        alternate_shared_tma_x_atom: Optional[cute.CopyAtom],
        alternate_shared_tma_x_rows_atom: Optional[cute.CopyAtom],
    ):
        """Compose scheduler, mainloop, epilogue, and the workspace tail reset."""
        self._primary.mainloop.materialize_codegen_members()
        if cutlass.const_expr(self._alternate is not None):
            self._alternate.mainloop.materialize_codegen_members()
        storage_type = self._smem_storage_class()
        smem_allocator = utils.SmemAllocator()
        storage = smem_allocator.allocate(storage_type)
        smem_base = storage.buffer.data_ptr()
        # The end-of-round reduce may launch as soon as every CTA here has started.
        cute.arch.griddepcontrol_launch_dependents()
        round_set = self.token_comm.select_round_set(
            self._device_workspace, local_workspace, shared_workspace
        )
        # fc2_output addresses set 0; this round's peers expect their partials in this round's set.
        # Under the in-kernel top-k reduce it is the caller's output, which no set holds.
        if cutlass.const_expr(not self.reduce_topk_in_kernel):
            fc2_output = cute.make_tensor(
                cute.make_ptr(
                    fc2_output.element_type,
                    fc2_output.iterator.toint()
                    + Int64(round_set) * Int64(self.shared_set_bytes),
                    AddressSpace.gmem,
                    assumed_align=128,
                ),
                fc2_output.layout,
            )

        fc1_done_counter = self._device_workspace.tensor(self.fc1_done_counter_region)
        fc1_output_sf_storage = self._device_workspace.tensor(self.fc1_output_sf_region)
        if cutlass.const_expr(isinstance(self.hidden_size, int)):
            hidden = self.hidden_size
        else:
            hidden = actual_expert_shape[2]
        if cutlass.const_expr(isinstance(self.intermediate_gateup_size, int)):
            intermediate_gateup = self.intermediate_gateup_size
        else:
            intermediate_gateup = actual_expert_shape[1]
        fc1_output_sf = cute.make_tensor(
            fc1_output_sf_storage.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    fc1_output_sf_storage.shape[0],
                    intermediate_gateup // 2,
                    cutlass.Int32(1),
                ),
                self.epilogue.sf_vec_size,
            ),
        )

        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(thread_idx // 32)
        block_idx = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_coord_in_cluster = (
            cta_rank_in_cluster % self.cluster_shape_mn[0],
            cta_rank_in_cluster // self.cluster_shape_mn[0],
            cutlass.Int32(0),
        )
        linear_cta_idx = cta_rank_in_cluster + block_idx[2] * Int32(self.cluster_size)

        if cutlass.const_expr(self.token_comm.stamps):
            if thread_idx == Int32(0):
                self.token_comm.stamp(
                    self._device_workspace,
                    self.token_comm.stamp_main_entry,
                    linear_cta_idx,
                )
        token_src_metadata = self.token_comm.token_src_metadata_tensor(
            self._device_workspace
        )
        gather_index = self.token_comm.gather_index_tensor(self._device_workspace)
        pool_topk_scores = self.token_comm.fc1_topk_scores_tensor(
            self._device_workspace
        )
        if cutlass.const_expr(self.tail_barrier is not None):
            self.tail_barrier.assign_device_members(
                self._device_workspace, peer_rank_ptr_mapper, round_set
            )

        primary_tma = (
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_atom,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom,
            fc2_tma_b_tensor,
            fc2_tma_b_atom,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom,
            fc2_tma_b_rows_atom,
            fc1_tma_b_rows_atom,
            fc1_tma_b_pool_atom,
            fc1_output_tma_atom,
            fc1_output_tma_tensor,
        )
        primary_shared_tma = (
            shared_fc1_tma_a_tensor,
            shared_fc1_tma_a_atom,
            shared_fc1_tma_sfa_tensor,
            shared_fc1_tma_sfa_atom,
            shared_fc2_tma_a_tensor,
            shared_fc2_tma_a_atom,
            shared_fc2_tma_sfa_tensor,
            shared_fc2_tma_sfa_atom,
            shared_tma_x_atom,
            shared_tma_x_rows_atom,
        )
        alternate_shared_tma = (
            alternate_shared_fc1_tma_a_tensor,
            alternate_shared_fc1_tma_a_atom,
            alternate_shared_fc1_tma_sfa_tensor,
            alternate_shared_fc1_tma_sfa_atom,
            alternate_shared_fc2_tma_a_tensor,
            alternate_shared_fc2_tma_a_atom,
            alternate_shared_fc2_tma_sfa_tensor,
            alternate_shared_fc2_tma_sfa_atom,
            alternate_shared_tma_x_atom,
            alternate_shared_tma_x_rows_atom,
        )
        # A shared expert's entry head starts before the plan, so without an
        # alternate configuration the pipelines come up at once and the scheduler
        # plans the token pool once it has the plan's sizes.
        if cutlass.const_expr(
            self._alternate is None
            and (not self.token_pool_capacity or self.shared_entry_head)
        ):
            self._run_token_tile(
                self._primary,
                primary_tma,
                shared_tma=primary_shared_tma,
                warp_idx=warp_idx,
                thread_idx=thread_idx,
                block_idx=block_idx,
                cta_coord_in_cluster=cta_coord_in_cluster,
                linear_cta_idx=linear_cta_idx,
                smem_base=smem_base,
                round_set=round_set,
                fc2_output=fc2_output,
                fc1_done_counter=fc1_done_counter,
                fc1_output_sf=fc1_output_sf,
                hidden=hidden,
                intermediate_gateup=intermediate_gateup,
                actual_expert_shape=actual_expert_shape,
                scheduler_expert_sizes=scheduler_expert_sizes,
                peer_rank_ptr_mapper=peer_rank_ptr_mapper,
                token_src_metadata=token_src_metadata,
                gather_index=gather_index,
                pool_topk_scores=pool_topk_scores,
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
                fc1_norm_const=fc1_norm_const,
                fc1_weight_address=fc1_weight_address,
                fc1_weight_sf_address=fc1_weight_sf_address,
                pull_activation=pull_activation,
                pull_activation_sf=pull_activation_sf,
                staging_peer_rank_ptr_mapper=staging_peer_rank_ptr_mapper,
                prefetch_entry=True,
            )
        else:
            # The entry weight prefetch does not depend on the plan, so it leaves
            # before the choice waits for the expert sizes.
            if cutlass.const_expr(self.entry_prefetch_bytes > 0):
                if warp_idx < len(self.epilogue_warp_ids):
                    iket.range_push("epilogue.entry_prefetch")
                    self._prefetch_entry(
                        fc1_weight_address,
                        fc1_weight_sf_address,
                        warp_idx,
                        linear_cta_idx,
                        thread_idx % Int32(32),
                    )
                    iket.range_pop()
            # So does the output's zeroing.
            if cutlass.const_expr(self.reduce_topk_in_kernel):
                if warp_idx < len(self.epilogue_warp_ids):
                    self._zero_output(
                        fc2_output.iterator.toint(),
                        thread_idx % Int32(32),
                        peer_rank_ptr_mapper,
                    )
            alternate_tma = (
                alternate_fc1_tma_a_tensor,
                alternate_fc1_tma_a_atom,
                alternate_fc1_tma_sfa_tensor,
                alternate_fc1_tma_sfa_atom,
                alternate_fc2_tma_a_tensor,
                alternate_fc2_tma_a_atom,
                alternate_fc2_tma_sfa_tensor,
                alternate_fc2_tma_sfa_atom,
                alternate_fc1_tma_b_atom,
                alternate_fc1_tma_sfb_tensor,
                alternate_fc1_tma_sfb_atom,
                alternate_fc2_tma_b_tensor,
                alternate_fc2_tma_b_atom,
                alternate_fc2_tma_sfb_tensor,
                alternate_fc2_tma_sfb_atom,
                alternate_fc2_tma_b_rows_atom,
                alternate_fc1_tma_b_rows_atom,
                alternate_fc1_tma_b_pool_atom,
                alternate_fc1_output_tma_atom,
                alternate_fc1_output_tma_tensor,
            )
            use_alternate = self._choose_token_tile(warp_idx, smem_base, linear_cta_idx)
            if cutlass.const_expr(self._alternate is None):
                self._run_token_tile(
                    self._primary,
                    primary_tma,
                    shared_tma=primary_shared_tma,
                    warp_idx=warp_idx,
                    thread_idx=thread_idx,
                    block_idx=block_idx,
                    cta_coord_in_cluster=cta_coord_in_cluster,
                    linear_cta_idx=linear_cta_idx,
                    smem_base=smem_base,
                    round_set=round_set,
                    fc2_output=fc2_output,
                    fc1_done_counter=fc1_done_counter,
                    fc1_output_sf=fc1_output_sf,
                    hidden=hidden,
                    intermediate_gateup=intermediate_gateup,
                    actual_expert_shape=actual_expert_shape,
                    scheduler_expert_sizes=scheduler_expert_sizes,
                    peer_rank_ptr_mapper=peer_rank_ptr_mapper,
                    token_src_metadata=token_src_metadata,
                    gather_index=gather_index,
                    pool_topk_scores=pool_topk_scores,
                    fc1_alpha=fc1_alpha,
                    fc2_alpha=fc2_alpha,
                    fc1_norm_const=fc1_norm_const,
                    fc1_weight_address=fc1_weight_address,
                    fc1_weight_sf_address=fc1_weight_sf_address,
                    pull_activation=pull_activation,
                    pull_activation_sf=pull_activation_sf,
                    staging_peer_rank_ptr_mapper=staging_peer_rank_ptr_mapper,
                    prefetch_entry=False,
                )
            else:
                if use_alternate:
                    self._run_token_tile(
                        self._alternate,
                        alternate_tma,
                        shared_tma=alternate_shared_tma,
                        warp_idx=warp_idx,
                        thread_idx=thread_idx,
                        block_idx=block_idx,
                        cta_coord_in_cluster=cta_coord_in_cluster,
                        linear_cta_idx=linear_cta_idx,
                        smem_base=smem_base,
                        round_set=round_set,
                        fc2_output=fc2_output,
                        fc1_done_counter=fc1_done_counter,
                        fc1_output_sf=fc1_output_sf,
                        hidden=hidden,
                        intermediate_gateup=intermediate_gateup,
                        actual_expert_shape=actual_expert_shape,
                        scheduler_expert_sizes=scheduler_expert_sizes,
                        peer_rank_ptr_mapper=peer_rank_ptr_mapper,
                        token_src_metadata=token_src_metadata,
                        gather_index=gather_index,
                        pool_topk_scores=pool_topk_scores,
                        fc1_alpha=fc1_alpha,
                        fc2_alpha=fc2_alpha,
                        fc1_norm_const=fc1_norm_const,
                        fc1_weight_address=fc1_weight_address,
                        fc1_weight_sf_address=fc1_weight_sf_address,
                        pull_activation=pull_activation,
                        pull_activation_sf=pull_activation_sf,
                        staging_peer_rank_ptr_mapper=staging_peer_rank_ptr_mapper,
                        prefetch_entry=False,
                    )
                else:
                    self._run_token_tile(
                        self._primary,
                        primary_tma,
                        shared_tma=primary_shared_tma,
                        warp_idx=warp_idx,
                        thread_idx=thread_idx,
                        block_idx=block_idx,
                        cta_coord_in_cluster=cta_coord_in_cluster,
                        linear_cta_idx=linear_cta_idx,
                        smem_base=smem_base,
                        round_set=round_set,
                        fc2_output=fc2_output,
                        fc1_done_counter=fc1_done_counter,
                        fc1_output_sf=fc1_output_sf,
                        hidden=hidden,
                        intermediate_gateup=intermediate_gateup,
                        actual_expert_shape=actual_expert_shape,
                        scheduler_expert_sizes=scheduler_expert_sizes,
                        peer_rank_ptr_mapper=peer_rank_ptr_mapper,
                        token_src_metadata=token_src_metadata,
                        gather_index=gather_index,
                        pool_topk_scores=pool_topk_scores,
                        fc1_alpha=fc1_alpha,
                        fc2_alpha=fc2_alpha,
                        fc1_norm_const=fc1_norm_const,
                        fc1_weight_address=fc1_weight_address,
                        fc1_weight_sf_address=fc1_weight_sf_address,
                        pull_activation=pull_activation,
                        pull_activation_sf=pull_activation_sf,
                        staging_peer_rank_ptr_mapper=staging_peer_rank_ptr_mapper,
                        prefetch_entry=False,
                    )

        if cutlass.const_expr(self.token_comm.stamps):
            if thread_idx == Int32(0):
                self.token_comm.stamp(
                    self._device_workspace,
                    self.token_comm.stamp_main_exit,
                    linear_cta_idx,
                )
        # Drain the preceding communication grid (including its final refinement
        # claims) so that this grid's completion, which the end-of-round reduce
        # waits for before clearing counters, implies that grid's completion too.
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.tail_barrier is not None):
            # Every epilogue drained its bulk stores at its last tile boundary; the
            # grid join and system fence inside the barrier publish them to peers,
            # and leaving it means every peer's pushes into this rank have landed.
            cute.arch.sync_threads()
            if warp_idx < len(self.epilogue_warp_ids):
                self.tail_barrier.arrive_and_wait(
                    32 * len(self.epilogue_warp_ids),
                    Int32(self.promised_launchable_sm_count),
                    linear_cta_idx,
                    thread_idx,
                    prologue_grid_sync=True,
                    epilogue_grid_sync=False,
                )
            self.tail_barrier.remove_device_members()
        self._device_workspace.remove_device_members()

    @cute.jit
    def _choose_token_tile(
        self, warp_idx: Int32, smem_base: cute.Pointer, linear_cta_idx: Int32
    ) -> Boolean:
        """Whether this launch runs the alternate token tile; every CTA reaches the same answer.

        Under a token pool with a capacity the scheduler warp also plans the
        pool (``_plan_token_pool``) before the same barrier, and the launch runs
        the primary configuration where there is no alternate.

        The scheduler warp waits for the plan's expert sizes, which the first FC1
        tile needs anyway, applies the choice rule, and a CTA barrier hands the
        answer to every warp before any of them initializes the barriers, the
        TMEM or the work queue of the configuration it then runs. Every CTA
        derives it from the same sizes, so both CTAs of a cluster agree.

        The rule: both configurations run the plan's token tiles, and a tile
        costs the alternate N128 tile less (its AB ring holds more stages at the
        same instruction N), but the alternate holds at most 128 rows of one.
        The launch therefore takes the alternate exactly when every tile of the
        plan fits it. The largest tile of an expert is its first, of
        ``balanced_tile_rows`` routes. ``token_tile_choice`` "primary" bounds the
        largest tile at zero rows instead, so only a launch without routes takes
        the alternate. CTA 0 records the tile N it chose in
        ``token_tile_record_region``.
        """
        choice = cute.make_tensor(
            cute.recast_ptr(
                self._primary.smem_workspace.ptr(
                    self.token_tile_choice_region, smem_base
                ),
                dtype=Int32,
            ),
            cute.make_layout(1),
        )
        if warp_idx == self.scheduler_warp_id:
            spin_wait(
                self._device_workspace.ptr(self.token_comm.sizes_ready_region),
                lambda value: value == Int32(1),
                scope="gpu",
            )
            lane = cute.arch.lane_idx()
            if cutlass.const_expr(self._alternate is not None):
                expert_sizes = self._device_workspace.tensor(
                    self.token_comm.expert_sizes_region
                )
                largest_tile = Int32(0)
                for batch in cutlass.range_constexpr(
                    ceil_div(self.local_expert_count, 32)
                ):
                    expert = Int32(batch * 32) + lane
                    if expert < Int32(self.local_expert_count):
                        largest_tile = cutlass.max(
                            largest_tile,
                            balanced_tile_rows(expert_sizes[expert], self.cta_tile_n),
                        )
                for step in cutlass.range_constexpr(5):
                    largest_tile = cutlass.max(
                        largest_tile,
                        Int32(
                            cute.arch.shuffle_sync_bfly(largest_tile, offset=1 << step)
                        ),
                    )
                if cutlass.const_expr(self.shared_expert):
                    # The shared expert's token tiles run in the same configuration.
                    largest_tile = cutlass.max(
                        largest_tile, Int32(self.shared_expert_tiles.tile_rows)
                    )
                bound = self.alternate_tile_n if self.token_tile_choice == "rule" else 0
                use_alternate = Int32(largest_tile <= Int32(bound))
                if lane == Int32(0):
                    choice[0] = use_alternate
                    if linear_cta_idx == Int32(0):
                        record = self._device_workspace.tensor(
                            self.token_tile_record_region
                        )
                        record[0] = Int32(self.cta_tile_n) + use_alternate * Int32(
                            self.alternate_tile_n - self.cta_tile_n
                        )
            if cutlass.const_expr(self.token_pool_capacity):
                self._plan_token_pool(
                    self._token_pool_plan(smem_base), lane, linear_cta_idx
                )
        cute.arch.sync_threads()
        if cutlass.const_expr(self._alternate is None):
            return Boolean(False)
        return choice[0] != Int32(0)

    def _token_pool_plan(self, smem_base: cute.Pointer) -> cute.Tensor:
        """The launch's token pool plan in SMEM (``_plan_token_pool``): its capacity, then one word per local expert."""
        return cute.make_tensor(
            cute.recast_ptr(
                self._primary.smem_workspace.ptr(
                    self.token_pool_plan_region, smem_base
                ),
                dtype=Int32,
            ),
            cute.make_layout(1 + self.local_expert_count),
        )

    @cute.jit
    def _plan_token_pool(
        self, plan: cute.Tensor, lane: Int32, linear_cta_idx: Int32
    ) -> None:
        """The scheduler warp's plan of a token pool with a capacity, once the expert sizes are final.

        The pool holds the launch's routes in processing order, the slot order
        the scheduler walks (see GenphaseTokenComm, "Pull dispatch"). Word
        ``1 + e`` packs local expert e's first route in that order (high bits)
        with its routes per token tile (``balanced_tile_rows`` of the plan's
        tile N, low eight bits): its token tile t starts at that route plus t
        times that many. Word 0 is the launch's capacity: a token tile whose
        routes end within it loads from the pool, any other gathers from the
        landing slots. Under ``fc1_token_pool_fallback`` "launch" it is zero
        unless all of the launch's routes fit. The communication grid's walk
        derives the same rows from the same sizes (``_stage_pull_chunk_table``).
        CTA 0 records the launch's routes and capacity in
        ``token_pool_record_region``.
        """
        sizes_by_slot = self.token_comm.scheduler_expert_sizes_tensor(
            self._device_workspace
        )
        slot_to_expert = self.token_comm.slot_to_expert_tensor(self._device_workspace)
        carried = Int32(0)
        for batch in cutlass.range_constexpr(ceil_div(self.local_expert_count, 32)):
            slot = Int32(batch * 32) + lane
            size = Int32(0)
            if slot < Int32(self.local_expert_count):
                size = sizes_by_slot[slot]
            route_end = carried + _warp_inclusive_sum(size, lane)
            if slot < Int32(self.local_expert_count):
                plan[Int32(1) + slot_to_expert[slot]] = (
                    (route_end - size) << Int32(8)
                ) | balanced_tile_rows(size, self.cta_tile_n)
            carried = Int32(cute.arch.shuffle_sync(route_end, Int32(31)))
        capacity = Int32(self.token_pool_capacity)
        if cutlass.const_expr(self.token_comm.fc1_token_pool_fallback == "launch"):
            capacity = cutlass.select_(carried <= capacity, capacity, Int32(0))
        if lane == Int32(0):
            plan[0] = capacity
            if linear_cta_idx == Int32(0):
                record = self._device_workspace.tensor(self.token_pool_record_region)
                record[0] = carried
                record[1] = capacity

    @cute.jit
    def _run_token_tile(
        self,
        config: _TokenTileConfig,
        tma: Tuple,
        *,
        warp_idx: Int32,
        thread_idx: Int32,
        block_idx,
        cta_coord_in_cluster,
        linear_cta_idx: Int32,
        smem_base: cute.Pointer,
        round_set: Int32,
        fc2_output: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        hidden,
        intermediate_gateup,
        actual_expert_shape,
        scheduler_expert_sizes: cute.Tensor,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        token_src_metadata: cute.Tensor,
        gather_index: cute.Tensor,
        pool_topk_scores: Optional[cute.Tensor],
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        fc1_weight_address: Int64,
        fc1_weight_sf_address: Int64,
        pull_activation: Optional[cute.Tensor],
        pull_activation_sf: Optional[cute.Tensor],
        staging_peer_rank_ptr_mapper: Optional[SymmetricBufferDevice],
        prefetch_entry: bool,
        shared_tma: Optional[Tuple] = None,
    ) -> None:
        """Every warp role of one token tile configuration, from its barrier initialization to its last tile.

        ``tma`` holds the configuration's TMA atoms and tensors in the order of
        the kernel's parameters, and ``shared_tma`` the shared expert's. ``prefetch_entry``
        makes the epilogue warps issue the entry weight prefetch, unless the kernel already did.
        """
        (
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_atom,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom,
            fc2_tma_b_tensor,
            fc2_tma_b_atom,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom,
            fc2_tma_b_rows_atom,
            fc1_tma_b_rows_atom,
            fc1_tma_b_pool_atom,
            fc1_output_tma_atom,
            fc1_output_tma_tensor,
        ) = tma
        # The launch's token pool plan (``_plan_token_pool``) under a token pool with a capacity.
        token_pool_plan = (
            self._token_pool_plan(smem_base) if self.token_pool_capacity else None
        )
        # Under the split-depth plan the weights and the tokens ride separate
        # pipelines of different depth; under the shared plan `token_pipeline` is
        # the same object, so every role below reads the same as it did before.
        if cutlass.const_expr(config.mainloop.uses_asymmetric_ab_stages):
            ab_pipeline = config.mainloop.create_weight_pipeline(
                config.smem_workspace, smem_base
            )
            token_pipeline = config.mainloop.create_token_pipeline(
                config.smem_workspace, smem_base
            )
        else:
            ab_pipeline = config.mainloop.create_ab_pipeline(
                config.smem_workspace, smem_base
            )
            token_pipeline = ab_pipeline
        tma_a_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, config.mainloop.num_weight_ab_stages
        )
        tma_b_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, config.mainloop.num_token_ab_stages
        )
        gather_b_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, config.mainloop.num_token_ab_stages
        )
        mma_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, config.mainloop.num_weight_ab_stages
        )
        mma_token_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, config.mainloop.num_token_ab_stages
        )
        acc_pipeline = config.mainloop.create_acc_pipeline(
            config.smem_workspace, smem_base
        )
        tmem_allocator = config.mainloop.create_tmem_allocator(
            config.smem_workspace,
            smem_base,
            allocator_warp_id=self.epilogue_warp_ids[0],
        )
        self.scheduler.assign_device_members(
            expert_token_sizes=scheduler_expert_sizes,
            expert_token_prefix_sum=None,
            actual_expert_shape=actual_expert_shape,
            block_idx=block_idx,
            smem_workspace=config.smem_workspace,
            smem_base=smem_base,
            device_workspace=self._device_workspace,
            cache_token_counts=self.just_in_time_claims,
        )
        scheduler = self.scheduler

        fc2_spin_threshold = (
            intermediate_gateup + config.mainloop.cta_tile_m - 1
        ) // config.mainloop.cta_tile_m
        # The scheduler enumerates slots, so the tile it produces has to be
        # rewritten into an expert index before any consumer sees it: the weights,
        # their scale factors, the per-expert alphas and the per-expert
        # scale-factor gate all address an expert. The row offsets it also carries
        # are already resolved and stay as they are.
        expert_remap = self.token_comm.slot_to_expert_tensor(self._device_workspace)
        # No FC1 ready counter: readiness is published per rank, not per tile, and
        # the waits below cover it once per warp.
        kernel_extension = BlockScaledSwapAbFc12Extension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter.iterator,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=None,
            expert_remap=expert_remap,
            shared_expert_marker=self.local_expert_count
            if self.shared_expert
            else None,
        )
        if cutlass.const_expr(self.apply_topk_at_fc1):
            optional_epilogue_args = GatedActEpilogueArgs(
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
                fc1_norm_const=fc1_norm_const,
                topk_scores=pool_topk_scores,
            )
        else:
            # Routing weights after FC2: under the in-kernel reduce the FC2 epilogue
            # weights each route's partial; otherwise the top-k reduce does.
            optional_epilogue_args = GatedActEpilogueArgs(
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
                fc1_norm_const=fc1_norm_const,
                topk_scores=None,
                fc2_topk_scores=pool_topk_scores
                if cutlass.const_expr(self.reduce_topk_in_kernel)
                else None,
            )

        # Each work tile completes one phase of the claim gate with two arrivals:
        # TMA-B once the tile passed its readiness gate, and TMA-A once
        # `claim_lead_k_tiles` of its weight K tiles are left to issue.
        claim_gate = None
        if cutlass.const_expr(self.claim_gate_compiled):
            claim_gate = config.smem_workspace.ptr(self.claim_gate_region, smem_base)
            if thread_idx == Int32(0):
                cute.arch.mbarrier_init(claim_gate, 2)
            cute.arch.mbarrier_init_fence()

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        config.mainloop.assign_device_members(
            config.smem_workspace,
            smem_base,
            cta_coord_in_cluster,
            hidden,
            intermediate_gateup,
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == self.scheduler_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.scheduler")
            if cutlass.const_expr(self.shared_entry_head):
                claim_gate_phase = Int32(0)
                # The shared expert's entry head, before the plan: one tile at a
                # time, the next claimed once the current one has passed its gate
                # and has `claim_lead_k_tiles` K tiles of weights left, so the
                # clusters that start first do not bind several head tiles each.
                iket.range_push("scheduler.shared_head")
                head_tile = scheduler.gen_next_shared_head_work()
                while head_tile.is_valid_tile:
                    scheduler.publish_work(
                        kernel_extension.prepare_work_tile(head_tile)
                    )
                    cute.arch.mbarrier_wait(claim_gate, claim_gate_phase)
                    claim_gate_phase = claim_gate_phase ^ Int32(1)
                    head_tile = scheduler.gen_next_shared_head_work()
                iket.range_pop()
                if cutlass.const_expr(self.token_comm.stamps):
                    if cute.arch.lane_idx() == Int32(0):
                        self.token_comm.stamp(
                            self._device_workspace,
                            self.token_comm.stamp_main_head_done,
                            linear_cta_idx,
                        )
            iket.range_push("scheduler.wait_sizes_ready")
            # One communication CTA (the helper, or planner 0 under the Sentinel
            # dispatch) publishes `expert_sizes` once, so the target is one rather
            # than a derived count.
            spin_wait(
                self._device_workspace.ptr(self.token_comm.sizes_ready_region),
                lambda value: value == Int32(1),
                scope="gpu",
            )
            iket.range_pop()
            if cutlass.const_expr(self.token_comm.stamps):
                if cute.arch.lane_idx() == Int32(0):
                    self.token_comm.stamp(
                        self._device_workspace,
                        self.token_comm.stamp_main_plan_seen,
                        linear_cta_idx,
                    )
            if cutlass.const_expr(
                self.shared_entry_head
                and self.token_pool_capacity
                and self._alternate is None
            ):
                # No configuration choice ran before the pipelines (``_choose_token_tile``),
                # so the pool is planned here; the first routed tile published below
                # orders the plan's SMEM words before every consumer's reads.
                self._plan_token_pool(
                    self._token_pool_plan(smem_base),
                    cute.arch.lane_idx(),
                    linear_cta_idx,
                )
            if cutlass.const_expr(self.die_local_weights):
                scheduler.load_sm_die()
            # Claims made while the SM streams weights map and translate their
            # tiles from register copies of the slot tables, which are final now.
            expert_remap_cache = None
            if cutlass.const_expr(self.just_in_time_claims):
                scheduler.load_token_count_cache()
                expert_remap_cache = kernel_extension.load_expert_remap_cache(
                    self.local_expert_count
                )
            if cutlass.const_expr(self.shared_expert):
                scheduler.load_routed_tile_count()
            iket.range_push("scheduler.gen_work")
            work_tile = scheduler.gen_next_work()
            iket.range_pop()
            claims_just_in_time = Boolean(False)
            if cutlass.const_expr(self.just_in_time_claims):
                fc1_tile_count, fc2_tile_count = scheduler.count_work_tiles()
                claims_just_in_time = self._claims_pay_just_in_time(
                    fc1_tile_count, fc2_tile_count
                )
            if cutlass.const_expr(not self.shared_entry_head):
                claim_gate_phase = Int32(0)
            while work_tile.is_valid_tile:
                iket.range_push("scheduler.publish_work")
                scheduler.publish_work(
                    kernel_extension.prepare_work_tile(work_tile, expert_remap_cache)
                )
                iket.range_pop()
                if cutlass.const_expr(self.just_in_time_claims):
                    # The tile just published completes this phase. Neither arrival
                    # waits on a later claim of this cluster: a tile's gate depends
                    # only on FC1 tiles of lower work IDs, whose clusters complete and
                    # release them (after any FC1 tiles batched with them) without
                    # waiting on this claim. A launch with eager claims never waits,
                    # and the arrivals complete phases nobody observes.
                    if claims_just_in_time:
                        iket.range_push("scheduler.wait_claim_gate")
                        cute.arch.mbarrier_wait(claim_gate, claim_gate_phase)
                        claim_gate_phase = claim_gate_phase ^ Int32(1)
                        iket.range_pop()
                iket.range_push("scheduler.gen_work")
                work_tile = scheduler.gen_next_work()
                iket.range_pop()
            if cutlass.const_expr(self.token_comm.stamps):
                if cute.arch.lane_idx() == Int32(0):
                    self.token_comm.stamp(
                        self._device_workspace,
                        self.token_comm.stamp_main_last_claim,
                        linear_cta_idx,
                    )
            iket.range_push("scheduler.publish_tail")
            scheduler.publish_work(work_tile)
            scheduler.produce_tail()
            iket.range_pop()
            iket.range_pop()

        if warp_idx == self.tma_a_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.tma_a")
            sched_consumer = scheduler.make_consumer()
            config.mainloop.run_tma_a(
                fc1_tma_a_tensor=fc1_tma_a_tensor,
                fc1_tma_a_atom=fc1_tma_a_atom,
                fc1_tma_sfa_tensor=fc1_tma_sfa_tensor,
                fc1_tma_sfa_atom=fc1_tma_sfa_atom,
                fc2_tma_a_tensor=fc2_tma_a_tensor,
                fc2_tma_a_atom=fc2_tma_a_atom,
                fc2_tma_sfa_tensor=fc2_tma_sfa_tensor,
                fc2_tma_sfa_atom=fc2_tma_sfa_atom,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=tma_a_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
                expert_sizes=self._device_workspace.tensor(
                    self.token_comm.expert_sizes_region
                ),
                claim_gate=claim_gate,
                claim_lead_k_tiles=self.claim_lead_k_tiles,
                shared_tma=shared_tma[:8] if self.shared_expert else None,
            )
            iket.range_pop()

        if warp_idx == self.tma_b_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.tma_b")
            sched_consumer = scheduler.make_consumer()
            config.mainloop.run_tma_b(
                fc1_tma_sfb_tensor=fc1_tma_sfb_tensor,
                fc1_tma_sfb_atom=fc1_tma_sfb_atom,
                fc2_tma_b_tensor=fc2_tma_b_tensor,
                fc2_tma_b_atom=fc2_tma_b_atom,
                fc2_tma_sfb_tensor=fc2_tma_sfb_tensor,
                fc2_tma_sfb_atom=fc2_tma_sfb_atom,
                ab_pipeline=token_pipeline,
                ab_pipeline_state=tma_b_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
                fc1_done_counter_pointer=fc1_done_counter.iterator,
                fc2_spin_threshold=fc2_spin_threshold,
                fc1_sf_ready_pointer=self._device_workspace.ptr(
                    self.token_comm.fc1_sf_ready_region
                ),
                expert_sizes=self._device_workspace.tensor(
                    self.token_comm.expert_sizes_region
                ),
                fc1_tile_ready_pointer=(
                    self._device_workspace.ptr(self.token_comm.fc1_tile_ready_region)
                    if self.token_comm.pull_tile_ready
                    else None
                ),
                first_fc1_stamp=self._first_fc1_stamp_pointer(linear_cta_idx),
                claim_gate=claim_gate,
                fc1_ready_counts_routes=self.dispatch_protocol == "pull",
                fc2_tma_b_rows_atom=fc2_tma_b_rows_atom,
                fc1_tma_b_atom=fc1_tma_b_atom,
                fc1_tma_b_rows_atom=fc1_tma_b_rows_atom,
                fc1_tma_b_pool_atom=fc1_tma_b_pool_atom,
                token_pool_plan=token_pool_plan,
                shared_tma=shared_tma if self.shared_expert else None,
                shared_token_rows=self.shared_expert_tiles.tile_rows
                if self.shared_expert
                else 0,
            )
            iket.range_pop()

        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.mma")
            sched_consumer = scheduler.make_consumer()
            if cutlass.const_expr(config.mainloop.uses_asymmetric_ab_stages):
                config.mainloop.run_mma_asymmetric(
                    tmem_allocator=tmem_allocator,
                    weight_pipeline=ab_pipeline,
                    weight_pipeline_state=mma_pipeline_state,
                    token_pipeline=token_pipeline,
                    token_pipeline_state=mma_token_pipeline_state,
                    acc_pipeline=acc_pipeline,
                    sched_consumer=sched_consumer,
                )
            else:
                config.mainloop.run_mma(
                    tmem_allocator=tmem_allocator,
                    ab_pipeline=ab_pipeline,
                    ab_pipeline_state=mma_pipeline_state,
                    acc_pipeline=acc_pipeline,
                    sched_consumer=sched_consumer,
                )
            iket.range_pop()

        if warp_idx < len(self.epilogue_warp_ids):
            cute.arch.warpgroup_reg_alloc(self.epilogue_register_count)
            if cutlass.const_expr(prefetch_entry and self.entry_prefetch_bytes > 0):
                iket.range_push("epilogue.entry_prefetch")
                self._prefetch_entry(
                    fc1_weight_address,
                    fc1_weight_sf_address,
                    warp_idx,
                    linear_cta_idx,
                    thread_idx % Int32(32),
                )
                iket.range_pop()
            if cutlass.const_expr(prefetch_entry and self.reduce_topk_in_kernel):
                self._zero_output(
                    fc2_output.iterator.toint(),
                    thread_idx % Int32(32),
                    peer_rank_ptr_mapper,
                )
            iket.range_push("genphase.epilogue")
            # FC1 reads the pool-aligned top-k scores and FC2 reads the routing
            # records; both land with the same counter. With a shared expert's entry
            # head the epilogue waits for it at its first routed tile instead.
            entry_waits = {}
            if cutlass.const_expr(self.shared_entry_head):
                entry_waits = dict(
                    shared_expert_marker=self.local_expert_count,
                    routed_entry_wait=self._wait_metadata_ready,
                    fc2_entry_wait=self._wait_outputs_zeroed
                    if self.reduce_topk_in_kernel
                    else None,
                )
            else:
                iket.range_push("epilogue.wait_metadata_ready")
                spin_wait(
                    self._device_workspace.ptr(self.token_comm.metadata_ready_region),
                    lambda value: value == Int32(self.token_comm.metadata_ready_target),
                    scope="gpu",
                )
                iket.range_pop()
            sched_consumer = scheduler.make_consumer()
            tmem_allocator.allocate(config.mainloop.num_tmem_alloc_cols)
            barrier_sync_unaligned(
                config.mainloop.tmem_allocation_barrier_id,
                32 * (1 + config.mainloop.num_accumulator_consumer_warps_per_cta),
            )
            tmem_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
            # Under the in-kernel top-k reduce FC2 adds into the output, which the
            # staging allocation holds, once every rank has zeroed its own.
            fc2_peer_rank_ptr_mapper = peer_rank_ptr_mapper
            if cutlass.const_expr(self.reduce_topk_in_kernel):
                fc2_peer_rank_ptr_mapper = staging_peer_rank_ptr_mapper
                if cutlass.const_expr(not self.shared_entry_head):
                    self._wait_outputs_zeroed()
            config.epilogue.run(
                config.smem_workspace,
                smem_base,
                tmem_pointer,
                acc_pipeline,
                sched_consumer,
                kernel_extension,
                fc1_output_tma_atom,
                fc1_output_tma_tensor,
                fc1_output_sf,
                None,
                None,
                fc2_output,
                fc1_done_counter,
                thread_idx,
                token_src_metadata,
                None,
                None,
                fc2_peer_rank_ptr_mapper,
                optional_epilogue_args,
                **entry_waits,
            )
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(
                tmem_allocator.retrieve_ptr(self.acc_dtype),
                config.mainloop.num_tmem_alloc_cols,
            )
            if cutlass.const_expr(self.token_comm.stamps):
                if thread_idx == Int32(0):
                    self.token_comm.stamp(
                        self._device_workspace,
                        self.token_comm.stamp_main_epilogue_done,
                        linear_cta_idx,
                    )

            iket.range_pop()

        if (warp_idx >= Int32(self.gather_b_warp_ids[0])) & (
            warp_idx <= Int32(self.gather_b_warp_ids[-1])
        ):
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            # Every sibling executes warpgroup register deallocation above;
            # only the active gather warps consume work and issue Gather4, and
            # none of them under a token pool of every data row, whose FC1 tokens
            # TMA-B loads. Under a pool with a capacity they gather the token
            # tiles past it.
            if cutlass.const_expr(not self.fc1_token_pool or self.token_pool_capacity):
                if warp_idx < Int32(
                    self.gather_b_warp_ids[0] + self.active_gather_b_warp_count
                ):
                    iket.range_push("genphase.gather_b1")
                    gather_entry_wait = {}
                    if cutlass.const_expr(self.shared_entry_head):
                        # The same waits, at the first routed tile this warp gathers.
                        gather_entry_wait = dict(
                            entry_wait=functools.partial(
                                self._wait_gather_input_ready,
                                warp_idx,
                                thread_idx,
                                linear_cta_idx,
                            )
                        )
                    else:
                        iket.range_push("gather_b1.wait_input_ready")
                        # `gather_index` is local, published by this rank's index-copy CTAs
                        # (the planners under the Sentinel dispatch); the payload flag is
                        # written by peers, hence the wider scope.
                        # Under the Sentinel dispatch payload readiness is per expert: TMA-B
                        # acquires each expert's verified-rows count before releasing the
                        # Gather4 issue barrier. So it is under round gating, where an
                        # expert's refine blocks acquire its round's counters first, and
                        # under the pull dispatch, where this rank fetches the expert's rows
                        # and counts them once they have landed. The whole-payload counter
                        # dispatch publishes everything with one per-peer counter.
                        if cutlass.const_expr(
                            self.dispatch_protocol == "counter"
                            and not self.token_comm.round_gated
                        ):
                            spin_wait(
                                self._device_workspace.ptr(
                                    self.token_comm.token_data_ready_region
                                ),
                                lambda value: value
                                == Int32(self.token_comm.payload_ready_target),
                                scope="sys",
                            )
                        spin_wait(
                            self._device_workspace.ptr(
                                self.token_comm.metadata_ready_region
                            ),
                            lambda value: value
                            == Int32(self.token_comm.metadata_ready_target),
                            scope="gpu",
                        )
                        iket.range_pop()
                        if cutlass.const_expr(self.token_comm.stamps):
                            if (warp_idx == Int32(self.gather_b_warp_ids[0])) & (
                                thread_idx % Int32(32) == Int32(0)
                            ):
                                self.token_comm.stamp(
                                    self._device_workspace,
                                    self.token_comm.stamp_main_gather_open,
                                    linear_cta_idx,
                                )
                    sched_consumer = scheduler.make_consumer()
                    config.mainloop.run_tma_gather_b1(
                        fc1_tma_b_atom=fc1_tma_b_atom,
                        gather_index=gather_index,
                        ab_pipeline=token_pipeline,
                        ab_pipeline_state=gather_b_pipeline_state,
                        sched_consumer=sched_consumer,
                        gather_warp_idx=warp_idx - Int32(self.gather_b_warp_ids[0]),
                        gather_row_offset=round_set * Int32(self.gather_set_row_offset),
                        token_pool_plan=token_pool_plan,
                        **gather_entry_wait,
                    )
                    iket.range_pop()

        if cutlass.const_expr(self.pull_warps > 0):
            if warp_idx >= Int32(self.pull_warp_ids[0]):
                cute.arch.warpgroup_reg_dealloc(self.pull_warp_register_count)
                # Every warp of the group releases its registers before the epilogue
                # warpgroup's allocation can complete; only the first `pull_warps` pull.
                if warp_idx < Int32(self.pull_warp_ids[0] + self.pull_warps):
                    iket.range_push("genphase.pull")
                    self.token_comm.run_pull_warp(
                        self._device_workspace,
                        thread_idx % Int32(32),
                        linear_cta_idx * Int32(self.pull_warps)
                        + warp_idx
                        - Int32(self.pull_warp_ids[0]),
                        self.promised_launchable_sm_count * self.pull_warps,
                        pull_activation.iterator.toint(),
                        pull_activation_sf.iterator.toint(),
                        staging_peer_rank_ptr_mapper,
                    )
                    iket.range_pop()


__all__ = ["BlockScaledSwapAbGenphaseMoeKernel"]
