# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Blackwell generation-phase block-scaled swap-AB FC12 mainloop.

FC1 gathers tokens from the dense received plane through Gather4, or, under the
pull dispatch into the token pool (``fc1_token_pool``), loads them from their
expert-major pool rows with tiled TMA as FC2 loads its tokens; a pool with a
capacity holds only the token tiles within it, and the rest gather. Blackwell
issues K32 MMAs whose instruction N follows each token tile's valid tokens, and
the two MMA CTAs split that row window evenly in both FC1 and FC2 token loads.
The 512-column TMEM plan has seven disjoint accumulator stages at N64, three at
N128 and two at N192 and N240. Communication readiness and per-expert
scale-factor gates follow the ported generation-phase protocol.

A shared expert (``shared_expert_marker``) is a third weight stream through the
same stages: MXFP8 E4M3 weights fill the A stages the unpacked E2M1 weights
fill, so a shared tile differs only in its TMA descriptors, the A format of its
instruction descriptor and the A bytes its stages expect. Its FC1 tokens are the
rank's own staged input, loaded by tiled TMA, and its scale factors sit in
pool rows past the routed ones.
"""

import dataclasses
from typing import Callable, ClassVar, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import Int32, Int64

from .....api import (
    ImplDesc,
    KernelComponent,
    OptionalRequirement,
    ProblemDesc,
    StaticOrRuntimeIntegerType,
)
from .....helpers.cute_py_helpers import (
    Tcgen05MmaInstruction,
    Tcgen05TmemPlan,
    make_smem_layouts,
    tcgen05_block_scaled_acc_dtype,
    tcgen05_smem_alloc_type,
)
from .....helpers.dsl_helpers import mark_alignment, spin_wait, tma_multicast_mask
from ......comm import ptx as sentinel_ptx
from .....helpers.iket_compat import iket
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import (
    ceil_div,
    expert_pool_rows,
    round_up,
    strides_equal_ignoring_singletons,
    token_slot_rows,
)
from .....quant_def import QuantKind
from ....schedulers.base import SchedulerConsumer
from ....schedulers.fc12_mapping import BlockPhase
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .tma_gather import blackwell_tma_gather4_load, blackwell_tma_tile2d_load
from .....helpers.ptx_helpers import barrier_sync_unaligned, umma_mxf8f6f4_2cta

# L2 eviction priority for the weight stream (A and SFA), selected by the ``weight_evict`` impl
# field. A weight tile read for the last time is marked evict-first, so the stream recycles its
# own lines instead of displacing activations and lines that earlier kernels left dirty (whose
# write-back would then compete with the stream). "first" marks every weight load; "adaptive"
# marks only the reads from an expert's last token tile (under dense slots, its last partial
# one), after which no other token tile re-reads that weight tile; "normal" leaves the TMA
# default.
_EVICT_FIRST = 0x12F0000000000000
_EVICT_NORMAL = 0x1000000000000000
WEIGHT_EVICT_MODES = ("normal", "first", "adaptive")


def _as_int32(value: int) -> int:
    """The signed 32-bit integer with the bits of the unsigned ``value``."""
    return value - (1 << 32) if value >= (1 << 31) else value


class BlockScaledSwapAbFc12MainloopGenSpecialized(KernelComponent):
    """Own all Blackwell load, MMA, pipeline, SMEM, and TMEM mainloop state."""

    ab_pipeline_mbarriers_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.ab_pipeline_mbarriers"
    )
    token_pipeline_mbarriers_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.token_pipeline_mbarriers"
    )
    a_smem_tensor_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.a_smem_tensor"
    )
    b_smem_tensor_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.b_smem_tensor"
    )
    sfa_smem_tensor_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.sfa_smem_tensor"
    )
    sfb_smem_tensor_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.sfb_smem_tensor"
    )
    acc_pipeline_mbarriers_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.acc_pipeline_mbarriers"
    )
    tmem_holding_buffer_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.tmem_holding_buffer"
    )
    tmem_deallocation_mbarrier_region: ClassVar[str] = (
        "blackwell.genphase_fc12.mainloop.tmem_deallocation_mbarrier"
    )
    tmem_allocation_barrier_id: ClassVar[int] = 2
    fc1_gather_sync_barrier_id: ClassVar[int] = 3
    # Four warps rather than one because the gather4 row operands live in uniform
    # registers and stay live across the whole K-tile loop: the per-warp row count
    # is the uniform register pressure, so splitting it is what keeps it bounded.
    # Each warp also issues its Gather4 groups one at a time, so the warp count
    # bounds how fast a stage's token rows go out; the kernel may ask for eight.
    num_gather_b_warps: ClassVar[int] = 4
    gather4_width: ClassVar[int] = 4
    # One B row must occupy exactly one swizzle atom of this size; see
    # _plan_gather_geometry for what breaks otherwise.
    gather_atom_bytes: ClassVar[int] = 128
    # Row identifiers are loaded eight at a time, which is one 256-bit access and
    # two gather4 groups. Ownership is therefore granular in eight-row runs.
    gather_index_load_rows: ClassVar[int] = 8
    # Rows of FC2's token box when TMA-B loads only a tile's live rows (see
    # ``fc2_b_live_rows``): one 16-row box is two 128-byte swizzle atoms.
    fc2_b_row_box: ClassVar[int] = 16
    # Default for a descriptor without ``cluster_shape_mn``. ``__init__`` takes the
    # descriptor's shape, which the mega kernel always forwards (the frontend's
    # default is (2, 1)); the mainloop accepts (2, 1) and (4, 1), and cluster N
    # stays 1 under swap-AB.
    cluster_shape_mn: ClassVar[Tuple[int, int]] = (4, 1)
    # Blackwell uses its native 1x block-scaled instruction K.
    mma_k_mode: ClassVar[str] = "1x"
    # Both operands are K-major: an fp4 TCGen05 operand has no other option, and
    # nothing in this kernel wants a different major mode for the wider dtypes.
    a_major_mode: ClassVar[OperandMajorMode] = OperandMajorMode.K
    b_major_mode: ClassVar[OperandMajorMode] = OperandMajorMode.K
    # Token-side depth once the operands stop sharing one. Only the token tile
    # scales B, so a 256-wide tile drags the weight depth down with it even though
    # A's per-stage bytes never change: the shared plan drops from ten stages at
    # 128 to seven at 256. Pinning the token side here and spending the remainder
    # on the weight side is what recovers weight depth.
    #
    # Six leaves the weights nine stages, two more than the shared plan's seven
    # and one short of what the 128 tile gets. The trade runs the other way too:
    # five would give the weights ten, four eleven, three thirteen. Six is where
    # profiling put it -- the token operand is read out of L2, so a handful of
    # stages covers its latency, but not as few as the L2 latency alone suggests.
    asymmetric_token_stages: ClassVar[int] = 6
    # The token tile that always takes this split; others split only through
    # ``token_ab_stages``.
    asymmetric_token_tile: ClassVar[int] = 256

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "quant_kind": str,
            "hidden_size": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            "mma_instruction_mnk": tuple,
            "mma_tiler_mnk": tuple,
            "use_2cta_instrs": bool,
            "tmem_plan": Tcgen05TmemPlan,
            "mainloop_smem_budget_bytes": int,
            "num_accumulator_consumer_warps_per_cta": int,
            "sf_padding_block": int,
            "weight_evict": OptionalRequirement(str),
            "gather_b_warps": OptionalRequirement(int),
            "token_ab_stages": OptionalRequirement(int),
            "token_tile_plan_n": OptionalRequirement(int),
            "die_local_weights": OptionalRequirement(bool),
            "fc1_token_pool": OptionalRequirement(bool),
            "fc1_token_pool_bounded": OptionalRequirement(bool),
            "shared_expert_marker": OptionalRequirement(Optional[int]),
            "shared_expert_token_tiles": OptionalRequirement(int),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)
        self.cluster_shape_mn = tuple(impl_desc.get("cluster_shape_mn", (4, 1)))
        self.weight_evict = impl_desc.get("weight_evict", "normal")
        if self.weight_evict not in WEIGHT_EVICT_MODES:
            raise ValueError(
                f"weight_evict must be one of {WEIGHT_EVICT_MODES}, got {self.weight_evict!r}."
            )

        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.a_dtype = self.quant_kind.weight_dtype
        self.b_dtype = self.quant_kind.activation_dtype
        self.sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.hidden_size = problem_desc["hidden_size"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.num_gather_b_warps = impl_desc.get(
            "gather_b_warps",
            2 if self.mma_tiler_mnk[1] == 64 and self.cluster_shape_mn == (4, 1) else 4,
        )
        self.tmem_plan = impl_desc["tmem_plan"]
        self.mainloop_smem_budget_bytes = impl_desc["mainloop_smem_budget_bytes"]
        # Die-local weights hold expert e's row half h as entry h * E + e of a
        # tensor with half the rows (``_weight_tile``).
        self.die_local_weights = impl_desc.get("die_local_weights", False)
        if self.die_local_weights:
            self.weight_expert_count = problem_desc["expert_count"]
            self.die_half_mma_tiles = (
                self.intermediate_gateup_size // 2 // self.mma_tiler_mnk[0],
                self.hidden_size // 2 // self.mma_tiler_mnk[0],
            )
        self.num_accumulator_consumer_warps_per_cta = impl_desc[
            "num_accumulator_consumer_warps_per_cta"
        ]

        self.architecture = "sm_100"
        self.mma_k_mode = "1x"
        self.mma_cta_count = 2 if self.use_2cta_instrs else 1
        self.instruction_k = self.mma_instruction_mnk[2]
        self.num_mma_instructions_per_ab_stage = (
            self.mma_tiler_mnk[2] // self.instruction_k
        )
        self._validate_configuration()

        self.cta_tile_m = self.mma_tiler_mnk[0] // self.mma_cta_count
        self.cta_tile_n = self.mma_tiler_mnk[1]
        self.mma_tile_k = self.mma_tiler_mnk[2]
        # Token tile N of the plan the communication component lays out and the
        # scheduler walks: it sets each expert's tile count, its tiles' rows and
        # their slots. By default the tile this mainloop is sized for; a narrower
        # tile N may run a wider tile's plan when no tile of it has more rows than
        # this one holds (the kernel's token tile choice).
        self.plan_tile_n = impl_desc.get("token_tile_plan_n", self.cta_tile_n)
        self._validate_plan_tile_n()
        # Pool rows between consecutive token tiles of one expert, in both the
        # data and the scale-factor pool (see token_slot_rows).
        self.token_slot_rows = token_slot_rows(self.plan_tile_n)
        tmem_plan = self.tmem_plan
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = tmem_plan.sfa_columns + tmem_plan.sfb_columns
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_stages = tmem_plan.accumulator_stage_count
        self.num_accumulator_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        if (
            tmem_plan.accumulator_stage_stride_columns
            != tmem_plan.accumulator_stage_columns
        ):
            raise ValueError(
                "Blackwell Genphase MegaMoE does not support overlapping accumulator stages."
            )
        if tmem_plan.accumulator_pipeline_stages != tmem_plan.accumulator_stage_count:
            raise ValueError(
                "Blackwell Genphase MegaMoE requires one pipeline stage per disjoint accumulator stage."
            )
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.accumulator_shape = (
            self.cta_tile_m,
            self.cta_tile_n,
            tmem_plan.accumulator_stage_count,
        )
        self.accumulator_stride = (
            1 << 16,
            1,
            tmem_plan.accumulator_stage_stride_columns,
        )

        self.mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=self.mma_instruction_mnk,
            participates=self.mma_cta_count,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        self.a_smem_alloc_dtype = tcgen05_smem_alloc_type(
            self.a_dtype, self.b_dtype, self.architecture
        )
        self.b_smem_alloc_dtype = tcgen05_smem_alloc_type(
            self.b_dtype, self.a_dtype, self.architecture
        )
        self.sfb_instruction_shape_mnk = (
            self.mma_instruction_mnk[0] // self.mma_cta_count,
            round_up(self.mma_instruction_mnk[1], 128),
            self.instruction_k,
        )
        self.sfb_mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=self.sfb_instruction_shape_mnk,
            participates=1,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        self.mma_tiler_sfb = (
            self.sfb_instruction_shape_mnk[0],
            self.sfb_instruction_shape_mnk[1],
            self.mma_tile_k,
        )
        self.cluster_layout_shape_vmnk = (
            (self.mma_cta_count,),
            self.cluster_shape_mn[0] // self.mma_cta_count,
            self.cluster_shape_mn[1],
            1,
        )
        self.cluster_layout_sfb_shape_vmnk = ((1,), *self.cluster_shape_mn, 1)
        self.num_mcast_ctas_a = self.cluster_layout_shape_vmnk[2]
        self.num_mcast_ctas_b = self.cluster_layout_shape_vmnk[1]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # The 256-wide token tile is always split. Any other tile splits on request:
        # ``token_ab_stages`` (0: none) keeps that many token stages and gives the
        # weight side the rest of the budget, unless that buys the weight side no
        # stage over the shared plan (``_select_ab_stage_plan``).
        self.uses_asymmetric_ab_stages = self.cta_tile_n == self.asymmetric_token_tile
        self.token_ab_stages = impl_desc.get("token_ab_stages", 0)
        if isinstance(self.token_ab_stages, bool) or not isinstance(
            self.token_ab_stages, int
        ):
            raise ValueError(
                f"token_ab_stages must be an int, got {self.token_ab_stages!r}."
            )
        if self.token_ab_stages:
            if self.token_ab_stages < 2:
                raise ValueError(
                    f"token_ab_stages must be at least 2, got {self.token_ab_stages}."
                )
            self.uses_asymmetric_ab_stages = True
        # Two-CTA mxf8f6f4 tiles issue each token tile's MMAs at the instruction N
        # its valid tokens need (``_instruction_n``) rather than the tile N, so a
        # partly filled tile costs the tensor pipe only its live columns. The two
        # MMA CTAs then split that shorter row window evenly: CTA c holds the
        # tile's rows [c * N / 2, (c + 1) * N / 2) in its first N / 2 B rows, which
        # Gather-B (FC1) and TMA-B (FC2) follow. Accumulator column j stays token
        # row j, so the epilogue's view of the tile does not change.
        self.runtime_instruction_n = (
            self.use_2cta_instrs and self.quant_kind.umma_kind == "mxf8f6f4"
        )
        if self.runtime_instruction_n:
            self._plan_runtime_instruction_n()
        # A tile N past one 128-row scale-factor atom (192, 240) copies its second
        # atom into TMEM only for tiles whose instruction N reaches it.
        self.runtime_sfb_window = self.runtime_instruction_n and self.cta_tile_n > 128
        # FC2's tiled TMA-B box is a CTA's whole half of the tile N. Past N128 a
        # tile's instruction N often leaves part of it unread (a 128-row tile of
        # an N192 plan reads 64 of each CTA's 96 rows), so TMA-B instead loads
        # just the CTA's live rows, in 16-row boxes at explicit coordinates, and
        # registers only their bytes.
        self.fc2_b_live_rows = (
            self.runtime_instruction_n
            and self.cta_tile_n > 128
            and not self.is_b_mcast
            and (self.cta_tile_n // self.mma_cta_count) % self.fc2_b_row_box == 0
        )
        # FC1's token operand: Gather4 from the landing slots by index (the
        # Gather-B warps), or with ``fc1_token_pool`` TMA-B's tiled loads of the
        # token tile's consecutive rows of the expert-major token pool, which the
        # pull dispatch's walk stores each route's row into. Those loads follow
        # FC2's: one box of the CTA's whole half of the tile N per stage, or only
        # the CTA's live rows in 16-row boxes where the tile leaves part of that
        # box unread. Unlike FC2 they do so at N128 as well, where a partly
        # filled tile would otherwise move its padding rows' 5120-byte FC1 rows.
        self.fc1_token_pool = impl_desc.get("fc1_token_pool", False)
        # A pool with a capacity holds a launch's token tiles only up to it (the
        # launch's token pool plan says which); the others still gather.
        self.fc1_token_pool_bounded = self.fc1_token_pool and impl_desc.get(
            "fc1_token_pool_bounded", False
        )
        self.fc1_b_live_rows = (
            self.fc1_token_pool
            and self.runtime_instruction_n
            and not self.is_b_mcast
            and (self.cta_tile_n // self.mma_cta_count) % self.fc2_b_row_box == 0
        )
        self._select_ab_stage_plan()
        self.num_tma_load_bytes = self.ab_stage_tma_bytes * self.mma_cta_count
        if self.num_tma_load_bytes != (
            self.a_stage_tma_bytes
            + self.b_stage_tma_bytes
            + self.sfa_stage_tma_bytes
            + self.sfb_stage_tma_bytes
        ):
            raise ValueError(
                "Genphase MegaMoE AB stage transaction byte accounting is inconsistent."
            )
        # Under the split plan each pipeline has a single producer whose expectation
        # is exactly the bytes it issues. The shared plan instead has two producers
        # splitting one combined expectation, so neither share matches the bytes
        # that warp actually sends -- which is why it needs the even split.
        self.weight_pipeline_tx_count = (
            self.a_stage_tma_bytes + self.sfa_stage_tma_bytes
        )
        self.token_pipeline_tx_count = self.b_stage_tma_bytes + self.sfb_stage_tma_bytes
        if not self.uses_asymmetric_ab_stages:
            if self.num_tma_load_bytes % 2 != 0:
                raise ValueError(
                    "Genphase MegaMoE total AB transaction bytes must split evenly across two arrivals."
                )
        self.ab_pipeline_tx_count_per_producer = self.num_tma_load_bytes // 2
        self._plan_gather_geometry()
        # Under the shared plan TMA-B subtracts the FC1 B rows Gather4 skips from
        # its half of the stage expectation. Once a tile's B bytes outweigh the
        # weight bytes (N192, N240), a mostly-padding tile would drive that half
        # negative, which an mbarrier expectation cannot be. FC1 stages then
        # register each producer's own operand bytes instead; the sum, and every
        # FC2 stage, is unchanged.
        max_omitted_b_bytes = (self.cta_tile_n - self.gather4_width) * (
            self.mma_tile_k * int(self.b_dtype.width) // 8
        )
        self.fc1_tx_split_by_operand = (
            not self.uses_asymmetric_ab_stages
            and max_omitted_b_bytes > self.ab_pipeline_tx_count_per_producer
        )
        self.fc1_gather_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.fc1_gather_sync_barrier_id,
            num_threads=32 * (1 + self.num_gather_b_warps),
        )
        # The expert index a shared expert's tiles carry, or None without one.
        self.shared_expert_marker = impl_desc.get("shared_expert_marker")
        if self.shared_expert_marker is not None:
            # Its token tiles: the last one reads each weight tile for the last time.
            self.shared_token_tiles = impl_desc["shared_expert_token_tiles"]
            self._plan_shared_expert()

    def _plan_shared_expert(self) -> None:
        """Fix the static parts of the shared expert's weight and token loads.

        The shared weights are MXFP8 E4M3, one byte per element like the
        unpacked E2M1 containers of the A stages, so a shared stage holds the
        same A tile but its TMA moves twice the bytes of a routed one: each
        producer that registers a stage's A bytes adds ``shared_a_extra_tx``.
        The instruction descriptor switches the A format to E4M3. FC1 loads the
        shared tile's tokens with tiled TMA from the staged input, a box of the
        CTA's half of the tile N per stage or, where a tile leaves part of it
        unread, its live rows in 16-row boxes (``shared_b_live_rows``).
        """
        if not self.runtime_instruction_n or self.is_a_mcast or self.is_b_mcast:
            raise ValueError(
                "A shared expert needs the runtime-N two-CTA mxf8f6f4 MMA at cluster (2, 1), without multicast."
            )
        shared_format = QuantKind.mxfp8_e4m3.weight_format_code
        self.shared_umma_instruction_descriptor_base = (
            self.umma_instruction_descriptor_base & ~(7 << 7)
        ) | (shared_format << 7)
        shared_a_stage_tma_bytes = (
            self.cta_tile_m * self.mma_tile_k * self.mma_cta_count
        )
        self.shared_a_extra_tx = shared_a_stage_tma_bytes - self.a_stage_tma_bytes
        if self.shared_a_extra_tx < 0:
            raise ValueError(
                "Shared-expert A stages cannot move fewer bytes than routed ones."
            )
        self.shared_b_live_rows = (
            self.cta_tile_n // self.mma_cta_count
        ) % self.fc2_b_row_box == 0

    def _validate_plan_tile_n(self) -> None:
        """Check that this tile can run the tiles of a wider plan.

        A plan tile N other than the mainloop's own needs strided slots, whose
        slot is a whole number of this tile's scale-factor atoms, and the runtime
        instruction N, which keeps every MMA inside the tile's live rows.
        """
        plan_n = self.plan_tile_n
        if isinstance(plan_n, bool) or not isinstance(plan_n, int) or plan_n <= 0:
            raise ValueError(
                f"token_tile_plan_n must be a positive int, got {plan_n!r}."
            )
        if plan_n == self.cta_tile_n:
            return
        plan_slot_rows = token_slot_rows(plan_n)
        if (
            plan_n < self.cta_tile_n
            or plan_slot_rows == plan_n
            or self.cta_tile_n != 128
            or plan_slot_rows % self.cta_tile_n
            or not (self.use_2cta_instrs and self.quant_kind.umma_kind == "mxf8f6f4")
        ):
            raise ValueError(
                f"A token tile N {self.cta_tile_n} cannot run the plan of token tile N {plan_n}: only N128 runs "
                "the plan of a wider strided tile (192, 240), with two-CTA mxf8f6f4 MMAs."
            )

    @cute.jit
    def _sfb_tile_index(self, tile_n_idx: Int32) -> Int32:
        """Index of a token tile's first scale-factor tile in the SFB tiling of ``mma_tiler_sfb``.

        N64 tiles share one 128-row atom; a tile running a strided plan whose
        slot spans several of its scale-factor tiles starts at its slot.
        """
        if cutlass.const_expr(self.cta_tile_n == 64):
            return tile_n_idx // Int32(2)
        if cutlass.const_expr(self.token_slot_rows != self.mma_tiler_sfb[1]):
            return tile_n_idx * Int32(self.token_slot_rows // self.mma_tiler_sfb[1])
        return tile_n_idx

    def _plan_runtime_instruction_n(self) -> None:
        """Fix the static parts of the runtime-N MMA issue.

        The instruction descriptor follows the PTX ISA layout for
        ``kind::mxf8f6f4``: A and B formats at bits 7 and 10, the scale format at
        bit 23, M >> 7 at bit 27 and N >> 3 at bit 17, which ``_instruction_n``
        supplies per tile. Both operands are K-major, so their major bits stay
        zero. The scale-factor data IDs (bits 29 and 4) select the byte of the
        32-bit scale-factor column, one per MMA K block.

        Both SMEM descriptors are K-major with a 128-byte swizzle: one B or A row
        is one 128-byte swizzle atom (checked in ``_plan_gather_geometry``), eight
        rows are 1024 bytes apart (the stride byte offset), the leading byte
        offset is the unused 16 bytes CuTe also encodes, and the layout type and
        descriptor version sit in the high bits. The start address field holds
        the byte address >> 4, so one MMA K block advances it by
        ``instruction_k`` one-byte containers.
        """
        if self.mma_instruction_mnk[0] != 256 or self.mma_cta_count != 2:
            raise ValueError(
                "Runtime instruction N is implemented for the two-CTA M256 MMA only."
            )
        if self.mma_tiler_mnk[2] // self.instruction_k != 4:
            raise ValueError(
                "Runtime instruction N expects four MMA K blocks, one scale-factor byte each, per AB stage."
            )
        if self.cta_tile_n % 16 != 0:
            raise ValueError(
                f"Runtime instruction N needs a tile N that is a multiple of 16, got {self.cta_tile_n}."
            )
        self.umma_instruction_descriptor_base = (
            (self.quant_kind.weight_format_code << 7)
            | (self.quant_kind.activation_format_code << 10)
            | (self.quant_kind.scale_format_code << 23)
            | ((self.mma_instruction_mnk[0] >> 7) << 27)
        )
        self.umma_kmajor_sw128_descriptor_bits = (
            (2 << 61) | (1 << 46) | ((1024 >> 4) << 32) | ((16 >> 4) << 16)
        )
        a_container_bytes = (
            int(
                tcgen05_smem_alloc_type(
                    self.a_dtype, self.b_dtype, self.architecture
                ).width
            )
            // 8
        )
        b_container_bytes = (
            int(
                tcgen05_smem_alloc_type(
                    self.b_dtype, self.a_dtype, self.architecture
                ).width
            )
            // 8
        )
        if a_container_bytes != 1 or b_container_bytes != 1:
            raise ValueError(
                "Runtime instruction N expects one-byte A and B SMEM containers."
            )
        self.umma_k_block_descriptor_step = self.instruction_k * a_container_bytes >> 4
        self.a_stage_descriptor_step = (
            self.cta_tile_m * self.mma_tiler_mnk[2] * a_container_bytes
        ) >> 4
        self.b_stage_descriptor_step = (
            self.cta_tile_n
            // self.mma_cta_count
            * self.mma_tiler_mnk[2]
            * b_container_bytes
        ) >> 4

    @cute.jit
    def _instruction_n(self, work_tile) -> Int32:
        """The tile's MMA instruction N: its valid tokens rounded up to the two-CTA M256 granule of 16."""
        return cutlass.max(
            round_up(work_tile.valid_tokens_in_cta_tile, Int32(16)), Int32(16)
        )

    @cute.jit
    def _issue_runtime_n_mma_stage(
        self,
        *,
        accumulator_tmem: Int32,
        a_descriptor: Int64,
        b_descriptor: Int64,
        instruction_descriptor: Int32,
        sfa_tmem: Int32,
        sfb_tmem: Int32,
        accumulate_first,
    ) -> None:
        """Issue one AB stage's four K-block MMAs at the instruction N ``instruction_descriptor`` encodes.

        K block ``k`` reads scale-factor byte ``k`` of each row's 32-bit column:
        its data ID goes into the descriptor and into the top two bits of both
        scale-factor TMEM addresses.
        """
        with cute.arch.elect_one():
            for k_block in cutlass.range_constexpr(
                self.num_mma_instructions_per_ab_stage
            ):
                scale_address_offset = Int32(_as_int32(k_block << 30))
                accumulate = Int32(1)
                if cutlass.const_expr(k_block == 0):
                    accumulate = Int32(accumulate_first)
                umma_mxf8f6f4_2cta(
                    accumulator_tmem,
                    a_descriptor + Int64(k_block * self.umma_k_block_descriptor_step),
                    b_descriptor + Int64(k_block * self.umma_k_block_descriptor_step),
                    instruction_descriptor | Int32((k_block << 29) | (k_block << 4)),
                    sfa_tmem + scale_address_offset,
                    sfb_tmem + scale_address_offset,
                    accumulate,
                )

    @cute.jit
    def _smem_descriptor_base(self, smem_tensor: cute.Tensor) -> Int64:
        """K-major 128-byte-swizzle UMMA descriptor of stage 0's first K block of ``smem_tensor``."""
        start_address = (smem_tensor.iterator.toint() >> Int32(4)) & Int32(0x3FFF)
        return Int64(self.umma_kmajor_sw128_descriptor_bits) + Int64(start_address)

    @cute.jit
    def _runtime_n_tile_operands(
        self, work_tile, accumulator_pointer, accumulator_stage_index
    ):
        """Accumulator and scale-factor TMEM addresses and instruction descriptor of one tile's MMAs."""
        accumulator_base = accumulator_pointer.toint()
        accumulator_tmem = accumulator_base + accumulator_stage_index * Int32(
            self.accumulator_stride[2]
        )
        sfa_tmem = accumulator_base + Int32(self.num_accumulator_tmem_cols)
        sfb_tmem = sfa_tmem + Int32(self.num_sfa_tmem_cols)
        if cutlass.const_expr(self.cta_tile_n == 64):
            # Two N64 tiles share one 128-row scale-factor atom; column c of the
            # atom holds rows [32c, 32c + 32).
            sfb_tmem = sfb_tmem + (work_tile.tile_n_idx % Int32(2)) * Int32(2)
        instruction_descriptor = Int32(self.umma_instruction_descriptor_base) | (
            (self._instruction_n(work_tile) >> Int32(3)) << Int32(17)
        )
        if cutlass.const_expr(self.shared_expert_marker is not None):
            # A shared tile's A operand is E4M3.
            instruction_descriptor = cutlass.select_(
                work_tile.expert_idx == Int32(self.shared_expert_marker),
                Int32(self.shared_umma_instruction_descriptor_base)
                | ((self._instruction_n(work_tile) >> Int32(3)) << Int32(17)),
                instruction_descriptor,
            )
        return accumulator_tmem, sfa_tmem, sfb_tmem, instruction_descriptor

    def _plan_gather_geometry(self) -> None:
        """Derive the interleaved gather4 ownership and check its preconditions.

        Rows a CTA's B SMEM holds is ``cta_tile_n / mma_cta_count``; the CTAs that
        share a B multicast group split that between them. The MMA CTA count
        cancels, so every CTA issues ``cta_tile_n / cluster_m`` rows whichever
        instruction width is in use.

        Ownership is granular in eight-row runs, interleaved across issuing units
        rather than block-partitioned. Eight rows is what one 256-bit access covers,
        so a run's identifiers are one load and its two gather4 destinations are
        adjacent. Interleaving spreads a short row window over every issuing unit
        instead of piling it onto unit zero, and it avoids block partitioning's
        requirement that the live row count be a multiple of the whole span --
        which is what keeps the door open for a dynamic transaction count later.
        """
        self.gather_b_rows_per_cta = self.cta_tile_n // self.mma_cta_count
        self.gather_issuing_units = self.num_mcast_ctas_b * self.num_gather_b_warps
        self.gather_b_groups_per_run = self.gather_index_load_rows // self.gather4_width
        if self.gather_b_rows_per_cta % self.gather_index_load_rows != 0:
            raise ValueError(
                f"Genphase MegaMoE gather runs are {self.gather_index_load_rows} rows: tile N {self.cta_tile_n} "
                f"must give each of {self.mma_cta_count} MMA CTAs a whole number of runs."
            )
        # Runs are dealt round-robin to the issuing units. When they do not divide
        # evenly (15 runs per CTA at N240) the last round is partial and the units
        # past its end skip it; ``gather_b_uneven_runs`` makes that test static.
        self.gather_b_runs_per_cta = (
            self.gather_b_rows_per_cta // self.gather_index_load_rows
        )
        self.gather_b_runs_per_unit = ceil_div(
            self.gather_b_runs_per_cta, self.gather_issuing_units
        )
        self.gather_b_uneven_runs = (
            self.gather_b_runs_per_cta % self.gather_issuing_units != 0
        )
        self.gather_b_rows_per_unit = (
            self.gather_b_runs_per_unit * self.gather_index_load_rows
        )

        # The gather writes four consecutive box images of a (1, mma_tile_k) box
        # under a 128-byte swizzle. That matches the planned B stage only when one
        # row occupies exactly one swizzle atom: above 128 bytes `make_smem_layouts`
        # splits the K mode into interleaved chunks that no single base address can
        # address, and below it the region picks a narrower swizzle than the
        # descriptor's. Both make the flat destination offset silently wrong.
        gather_row_bytes = self.mma_tile_k * int(self.b_smem_alloc_dtype.width) // 8
        if gather_row_bytes != self.gather_atom_bytes:
            raise ValueError(
                f"{self.quant_kind} FC1 gather requires mma_tiler K == "
                f"{self.gather_atom_bytes * 8 // int(self.b_smem_alloc_dtype.width)}, got {self.mma_tile_k}: one B "
                f"row must occupy exactly one {self.gather_atom_bytes}-byte swizzle atom, and this one occupies "
                f"{gather_row_bytes}."
            )

        # Full-tile Gather4 capacity must match FC2's tiled TMA layout. Compare
        # independently derived geometry and SMEM byte counts before allowing
        # FC1 to subtract omitted padding bytes from its stage expectation.
        self.gather_b_stage_bytes = (
            self.cta_tile_n * self.mma_tile_k * int(self.b_dtype.width) // 8
        )
        if self.gather_b_stage_bytes != self.b_stage_tma_bytes:
            raise ValueError(
                f"Genphase MegaMoE FC1 Gather-B moves {self.gather_b_stage_bytes} B bytes per AB stage but FC2's "
                f"TMA-B moves {self.b_stage_tma_bytes}; one static transaction expectation cannot serve both."
            )

    def _validate_configuration(self) -> None:
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

        expected_instruction_k = self.quant_kind.instruction_k(self.mma_k_mode)
        if self.instruction_k != expected_instruction_k:
            raise ValueError(
                f"{self.quant_kind} requires Blackwell 1x instruction K={expected_instruction_k}, got {self.instruction_k}."
            )
        expected_instruction_m = 256 if self.use_2cta_instrs else 128
        if self.mma_instruction_mnk[0] != expected_instruction_m:
            raise ValueError(
                f"{self.mma_cta_count}-CTA Blackwell MMA requires instruction M={expected_instruction_m}, "
                f"got {self.mma_instruction_mnk[0]}."
            )
        if self.mma_instruction_mnk[:2] != self.mma_tiler_mnk[:2]:
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
        if self.mma_tiler_mnk[1] not in (64, 128, 192, 240, 256):
            raise ValueError(
                "Blackwell Genphase MegaMoE supports tile N in (64, 128, 192, 240, 256)."
            )
        if self.mma_tiler_mnk[2] % self.instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
        if self.mainloop_smem_budget_bytes <= 0:
            raise ValueError("mainloop_smem_budget_bytes must be positive.")
        if self.num_accumulator_consumer_warps_per_cta <= 0:
            raise ValueError("num_accumulator_consumer_warps_per_cta must be positive.")
        if isinstance(self.hidden_size, int) and self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if isinstance(self.intermediate_gateup_size, int):
            if self.intermediate_gateup_size <= 0:
                raise ValueError("intermediate_gateup_size must be positive.")
            if self.intermediate_gateup_size % 2 != 0:
                raise ValueError("intermediate_gateup_size must be even.")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register the selected Blackwell mainloop SMEM plan."""
        # Under the shared plan this one array serves all four operands. Under the
        # split plan it becomes the weight side's and the token side gets its own,
        # because the two depths no longer index the same stage.
        smem_workspace.register_mbarrier(
            self.ab_pipeline_mbarriers_region, self.num_weight_ab_stages * 2
        )
        if self.uses_asymmetric_ab_stages:
            smem_workspace.register_mbarrier(
                self.token_pipeline_mbarriers_region, self.num_token_ab_stages * 2
            )
        for region_name, region in (
            (self.a_smem_tensor_region, self.a_smem_region),
            (self.b_smem_tensor_region, self.b_smem_region),
            (self.sfa_smem_tensor_region, self.sfa_smem_region),
            (self.sfb_smem_tensor_region, self.sfb_smem_region),
        ):
            smem_workspace.register_tensor(
                region_name,
                region.dtype,
                region.shape,
                stride=region.stride,
                swizzle=region.swizzle,
                byte_alignment=region.byte_alignment,
            )
        smem_workspace.register_mbarrier(
            self.acc_pipeline_mbarriers_region, self.num_accumulator_pipeline_stages * 2
        )
        smem_workspace.register_mbarrier(self.tmem_deallocation_mbarrier_region, 1)
        smem_workspace.register_tensor(
            self.tmem_holding_buffer_region, cutlass.Int32, (1,), byte_alignment=8
        )

    def _select_ab_stage_plan(self) -> None:
        stage_one_regions = make_smem_layouts(
            self.mma_instruction,
            self.mma_tiler_mnk,
            1,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )
        independent_sfb_region = make_smem_layouts(
            self.sfb_mma_instruction,
            self.mma_tiler_sfb,
            1,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )[3]
        if (
            independent_sfb_region.shape != stage_one_regions[3].shape
            or independent_sfb_region.stride != stage_one_regions[3].stride
            or independent_sfb_region.nbytes != stage_one_regions[3].nbytes
        ):
            raise ValueError(
                "The Blackwell independent SFB plan does not match the primary MMA SFB operand."
            )

        def packed_bytes(region, dtype) -> int:
            return (region.cosize * int(dtype.width) + 7) // 8

        # Native mixed FP4 halves its SMEM region, so the budget can select more AB stages.
        # TMA completion still counts packed source bytes, independent of the SMEM allocation type.
        self.ab_stage_payload_bytes = sum(region.nbytes for region in stage_one_regions)
        stage_operand_tma_bytes = (
            packed_bytes(stage_one_regions[0], self.a_dtype),
            packed_bytes(stage_one_regions[1], self.b_dtype),
            stage_one_regions[2].nbytes,
            stage_one_regions[3].nbytes,
        )
        self.ab_stage_tma_bytes = sum(stage_operand_tma_bytes)
        (
            self.a_stage_tma_bytes,
            self.b_stage_tma_bytes,
            self.sfa_stage_tma_bytes,
            self.sfb_stage_tma_bytes,
        ) = tuple(
            stage_bytes * self.mma_cta_count for stage_bytes in stage_operand_tma_bytes
        )
        mbarrier_bytes = int(cutlass.Int64.width) // 8
        plan_tail_bytes = (
            2 * self.num_accumulator_pipeline_stages + 2
        ) * mbarrier_bytes
        stage_budget_bytes = self.mainloop_smem_budget_bytes - plan_tail_bytes
        # Each pipeline spends two mbarriers per stage, full and empty. The shared
        # plan pays that once for all four operands; the split plan pays it twice,
        # which is the only overhead the split introduces.
        weight_stage_cost_bytes = (
            stage_one_regions[0].nbytes
            + stage_one_regions[2].nbytes
            + 2 * mbarrier_bytes
        )
        token_stage_cost_bytes = (
            stage_one_regions[1].nbytes
            + stage_one_regions[3].nbytes
            + 2 * mbarrier_bytes
        )

        shared_stages = stage_budget_bytes // (
            weight_stage_cost_bytes + token_stage_cost_bytes
        )
        if (
            self.token_ab_stages
            and (stage_budget_bytes - self.token_ab_stages * token_stage_cost_bytes)
            // weight_stage_cost_bytes
            <= shared_stages
        ):
            # A requested split that buys the weight ring no depth over the shared
            # ring falls back to it (its second mbarrier set would cost for nothing).
            self.uses_asymmetric_ab_stages = False
        if self.uses_asymmetric_ab_stages:
            self.num_token_ab_stages = (
                self.token_ab_stages or self.asymmetric_token_stages
            )
            self.num_weight_ab_stages = (
                stage_budget_bytes - self.num_token_ab_stages * token_stage_cost_bytes
            ) // weight_stage_cost_bytes
            if self.num_weight_ab_stages < 1:
                raise ValueError(
                    f"{self.num_token_ab_stages} token stages leave no room for a weight stage in the "
                    f"{self.mainloop_smem_budget_bytes}-byte mainloop budget."
                )
            # Splitting is pointless if it does not buy depth, and a plan that
            # bought none would silently pay the second mbarrier set for nothing.
            if self.num_weight_ab_stages <= shared_stages:
                raise ValueError(
                    f"Asymmetric stages give the weight side {self.num_weight_ab_stages} stages, no more than the "
                    f"{shared_stages} a shared plan reaches; the token depth of {self.num_token_ab_stages} is too deep "
                    f"to be worth splitting."
                )
        else:
            self.num_weight_ab_stages = shared_stages
            self.num_token_ab_stages = shared_stages
            if shared_stages < 1:
                raise ValueError(
                    f"One AB stage needs {weight_stage_cost_bytes + token_stage_cost_bytes + plan_tail_bytes} bytes, "
                    f"exceeding the {self.mainloop_smem_budget_bytes}-byte mainloop budget."
                )
        # Retained for the callers that size one pipeline's state: under the shared
        # plan both depths are this, and under the split plan the weight side is
        # the one the MMA's primary state follows.
        self.num_ab_pipeline_stages = self.num_weight_ab_stages

        (self.a_smem_region, _, self.sfa_smem_region, _) = make_smem_layouts(
            self.mma_instruction,
            self.mma_tiler_mnk,
            self.num_weight_ab_stages,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )
        (_, self.b_smem_region, _, self.sfb_smem_region) = make_smem_layouts(
            self.mma_instruction,
            self.mma_tiler_mnk,
            self.num_token_ab_stages,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )
        self.selected_smem_bytes = sum(
            region.nbytes
            for region in (
                self.a_smem_region,
                self.b_smem_region,
                self.sfa_smem_region,
                self.sfb_smem_region,
            )
        )

    def make_tiled_mma(self) -> cute.TiledMma:
        """Create a context-local SM100/103 block-scaled MMA object."""
        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        helper = sm100_utils
        shape = self.mma_instruction_mnk[:2]
        return helper.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            shape,
        )

    def materialize_codegen_members(self) -> None:
        """Materialize IR-backed Blackwell MMA and layout objects."""
        common_mma_arguments = (
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
        )
        self.tiled_mma = self.make_tiled_mma()
        helper = sm100_utils
        shape = self.sfb_instruction_shape_mnk[:2]
        self.tiled_mma_sfb = helper.make_blockscaled_trivial_tiled_mma(
            *common_mma_arguments, tcgen05.CtaGroup.ONE, shape
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.tiled_mma_sfb.thr_id.shape,),
        )
        if self.cluster_layout_vmnk.shape != self.cluster_layout_shape_vmnk:
            raise ValueError(
                f"Main cluster layout mismatch: Python {self.cluster_layout_shape_vmnk}, "
                f"CuTe {self.cluster_layout_vmnk.shape}."
            )
        if self.cluster_layout_sfb_vmnk.shape != self.cluster_layout_sfb_shape_vmnk:
            raise ValueError(
                f"SFB cluster layout mismatch: Python {self.cluster_layout_sfb_shape_vmnk}, "
                f"CuTe {self.cluster_layout_sfb_vmnk.shape}."
            )

        # Each operand is staged to its own side's depth; under the shared plan the
        # two depths are the same number and this is the previous behaviour.
        self.a_smem_composed_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma,
            self.mma_tiler_mnk,
            self.a_smem_alloc_dtype,
            self.num_weight_ab_stages,
        )
        self.b_smem_composed_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma,
            self.mma_tiler_mnk,
            self.b_smem_alloc_dtype,
            self.num_token_ab_stages,
        )
        self.a_smem_layout_staged = self.a_smem_composed_layout_staged.outer
        self.b_smem_layout_staged = self.b_smem_composed_layout_staged.outer
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma,
            self.mma_tiler_mnk,
            self.sf_vec_size,
            self.num_weight_ab_stages,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma,
            self.mma_tiler_mnk,
            self.sf_vec_size,
            self.num_token_ab_stages,
        )
        for tensor_name, materialized_layout, planned_region in (
            ("A", self.a_smem_layout_staged, self.a_smem_region),
            ("B", self.b_smem_layout_staged, self.b_smem_region),
            ("SFA", self.sfa_smem_layout_staged, self.sfa_smem_region),
            ("SFB", self.sfb_smem_layout_staged, self.sfb_smem_region),
        ):
            if (
                materialized_layout.shape != planned_region.shape
                or not strides_equal_ignoring_singletons(
                    planned_region.shape,
                    materialized_layout.stride,
                    planned_region.stride,
                )
            ):
                raise ValueError(
                    f"{tensor_name} SMEM plan mismatch: Python "
                    f"{planned_region.shape}:{planned_region.stride}, CuTe "
                    f"{materialized_layout.shape}:{materialized_layout.stride}."
                )

    def prepare_tma_load_params(
        self,
        *,
        fc1_a: cute.Tensor,
        fc1_b: cute.Tensor,
        fc1_sfa: cute.Tensor,
        fc1_sfb: cute.Tensor,
        fc2_a: cute.Tensor,
        fc2_b: cute.Tensor,
        fc2_sfa: cute.Tensor,
        fc2_sfb: cute.Tensor,
        fc1_b_pool: Optional[cute.Tensor] = None,
    ) -> Tuple:
        """Return TMA tensor/atom pairs ordered for run_tma_a then run_tma_b.

        Under a token pool with a capacity ``fc1_b`` is the landing plane the
        Gather4 atom reads and ``fc1_b_pool`` the pool, which the last atom
        returned and the live-row atom read; otherwise that atom is None.
        """
        a_stage_layout = cute.slice_(
            self.a_smem_composed_layout_staged, (None, None, None, 0)
        )
        b_stage_layout = cute.slice_(
            self.b_smem_composed_layout_staged, (None, None, None, 0)
        )
        sfa_stage_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        sfb_stage_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )

        a_operation = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        b_operation = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        sfa_operation = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        sfb_operation = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        a_internal_type = (
            self.a_smem_alloc_dtype
            if self.a_smem_alloc_dtype is not self.a_dtype
            else None
        )
        b_internal_type = (
            self.b_smem_alloc_dtype
            if self.b_smem_alloc_dtype is not self.b_dtype
            else None
        )

        fc1_a_atom, fc1_a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            a_operation,
            fc1_a,
            a_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=a_internal_type,
        )
        # FC1 B is read from the dense (source rank, source token) activation plane
        # the communication kernel filled. A one-row descriptor lets the producer
        # warps supply arbitrary source row IDs through TMA TILE_GATHER4 while
        # landing in the native K_SW128 B layout. Under ``fc1_token_pool`` it is
        # the (pool row, K) token pool instead, whose box is the CTA's whole half
        # of the tile N, plus FC2's 16-row box for the live-row loads. Only atoms
        # are returned: the loads supply coordinates directly and never partition
        # a tensor view. Under a pool with a capacity ``fc1_b`` keeps the gather's
        # one-row descriptor and the pool's box descriptor is ``fc1_b_pool_atom``.
        fc1_b_2d = fc1_b[(None, None, 0)]
        fc1_pool_2d = fc1_b_2d
        if cutlass.const_expr(self.fc1_token_pool_bounded):
            fc1_pool_2d = fc1_b_pool[(None, None, 0)]
        pool_box_rows = self.cta_tile_n // self.mma_cta_count
        fc1_b_box_rows = (
            pool_box_rows
            if self.fc1_token_pool and not self.fc1_token_pool_bounded
            else 1
        )
        gather_b_base = cute.make_layout(
            (fc1_b_box_rows, self.mma_tile_k), stride=(self.mma_tile_k, 1)
        )
        gather_b_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, gather_b_base
        )
        fc1_b_atom, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fc1_b_2d,
            gather_b_smem_layout,
            (fc1_b_box_rows, self.mma_tile_k),
        )
        fc1_b_pool_atom = None
        if cutlass.const_expr(self.fc1_token_pool_bounded):
            pool_b_base = cute.make_layout(
                (pool_box_rows, self.mma_tile_k), stride=(self.mma_tile_k, 1)
            )
            pool_b_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3), 0, pool_b_base
            )
            fc1_b_pool_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                fc1_pool_2d,
                pool_b_smem_layout,
                (pool_box_rows, self.mma_tile_k),
            )
        fc1_b_rows_atom = None
        if cutlass.const_expr(self.fc1_b_live_rows):
            fc1_b_rows_base = cute.make_layout(
                (self.fc2_b_row_box, self.mma_tile_k), stride=(self.mma_tile_k, 1)
            )
            fc1_b_rows_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3), 0, fc1_b_rows_base
            )
            fc1_b_rows_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                fc1_pool_2d,
                fc1_b_rows_smem_layout,
                (self.fc2_b_row_box, self.mma_tile_k),
            )
        fc1_sfa_atom, fc1_sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_operation,
            fc1_sfa,
            sfa_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        fc1_sfb_atom, fc1_sfb_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_operation,
            fc1_sfb,
            sfb_stage_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        fc2_a_atom, fc2_a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            a_operation,
            fc2_a,
            a_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=a_internal_type,
        )
        fc2_b_atom, fc2_b_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            b_operation,
            fc2_b,
            b_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=b_internal_type,
        )
        fc2_sfa_atom, fc2_sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_operation,
            fc2_sfa,
            sfa_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        fc2_sfb_atom, fc2_sfb_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_operation,
            fc2_sfb,
            sfb_stage_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        # FC2's live-row token loads: a 2D (pool row, K) descriptor whose box is
        # ``fc2_b_row_box`` rows of one K tile, in the B stage's 128-byte swizzle
        # (one row per swizzle atom row, as the FC1 gather's box).
        fc2_b_rows_atom = None
        if cutlass.const_expr(self.fc2_b_live_rows):
            fc2_b_rows_base = cute.make_layout(
                (self.fc2_b_row_box, self.mma_tile_k), stride=(self.mma_tile_k, 1)
            )
            fc2_b_rows_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3), 0, fc2_b_rows_base
            )
            fc2_b_rows_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                fc2_b[(None, None, 0)],
                fc2_b_rows_smem_layout,
                (self.fc2_b_row_box, self.mma_tile_k),
            )
        return (
            fc1_a_tensor,
            fc1_a_atom,
            fc1_sfa_tensor,
            fc1_sfa_atom,
            fc2_a_tensor,
            fc2_a_atom,
            fc2_sfa_tensor,
            fc2_sfa_atom,
            fc1_b_atom,
            fc1_sfb_tensor,
            fc1_sfb_atom,
            fc2_b_tensor,
            fc2_b_atom,
            fc2_sfb_tensor,
            fc2_sfb_atom,
            fc2_b_rows_atom,
            fc1_b_rows_atom,
            fc1_b_pool_atom,
        )

    def prepare_shared_tma_load_params(
        self,
        *,
        fc1_a: cute.Tensor,
        fc1_sfa: cute.Tensor,
        fc2_a: cute.Tensor,
        fc2_sfa: cute.Tensor,
        token_x: cute.Tensor,
    ) -> Tuple:
        """TMA tensor/atom pairs of a shared expert, ordered for ``run_tma_a`` then ``run_tma_b``.

        ``fc1_a`` / ``fc2_a`` are the shared E4M3 weights as (rows, K, 1) views
        and ``fc1_sfa`` / ``fc2_sfa`` their atom-swizzled scales; ``token_x`` is
        the (token, hidden) view of the staged input. The weight atoms follow the
        routed ones without the unpacking type; the token atoms load a box of the
        CTA's half of the tile N, and 16 live rows where ``shared_b_live_rows``.
        """
        a_stage_layout = cute.slice_(
            self.a_smem_composed_layout_staged, (None, None, None, 0)
        )
        sfa_stage_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        a_operation = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        sfa_operation = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        atoms = []
        for weight, scale in ((fc1_a, fc1_sfa), (fc2_a, fc2_sfa)):
            a_atom, a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
                a_operation,
                weight,
                a_stage_layout,
                self.mma_tiler_mnk,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            sfa_atom, sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_operation,
                scale,
                sfa_stage_layout,
                self.mma_tiler_mnk,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
            atoms.extend((a_tensor, a_atom, sfa_tensor, sfa_atom))
        box_rows = self.cta_tile_n // self.mma_cta_count
        x_box = cute.make_layout(
            (box_rows, self.mma_tile_k), stride=(self.mma_tile_k, 1)
        )
        x_atom, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            token_x,
            cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, x_box),
            (box_rows, self.mma_tile_k),
        )
        x_rows_atom = None
        if cutlass.const_expr(self.shared_b_live_rows):
            x_rows_box = cute.make_layout(
                (self.fc2_b_row_box, self.mma_tile_k), stride=(self.mma_tile_k, 1)
            )
            x_rows_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                token_x,
                cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, x_rows_box),
                (self.fc2_b_row_box, self.mma_tile_k),
            )
        return (*atoms, x_atom, x_rows_atom)

    @cute.jit
    def assign_device_members(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        cta_coord_in_cluster: cute.Coord,
        hidden_size,
        intermediate_gateup_size,
    ) -> None:
        """Root device-side Blackwell SMEM tensors and CTA coordinates."""
        self.a_smem_tensor = smem_workspace.tensor(self.a_smem_tensor_region, smem_base)
        if cutlass.const_expr(self.shared_expert_marker is not None):
            # The A stages typed as the shared expert's E4M3 weights, for its TMA loads.
            a_region = smem_workspace.region(self.a_smem_tensor_region)
            self.shared_a_smem_tensor = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float8E4M3FN,
                    self.a_smem_tensor.iterator.toint(),
                    self.a_smem_tensor.memspace,
                    assumed_align=a_region.byte_alignment,
                    swizzle_=None
                    if a_region.swizzle is None
                    else cute.make_swizzle(*a_region.swizzle),
                ),
                self.a_smem_tensor.layout,
            )
        self.b_smem_tensor = smem_workspace.tensor(self.b_smem_tensor_region, smem_base)
        self.sfa_smem_tensor = smem_workspace.tensor(
            self.sfa_smem_tensor_region, smem_base
        )
        self.sfb_smem_tensor = smem_workspace.tensor(
            self.sfb_smem_tensor_region, smem_base
        )
        self.cta_coord_in_cluster = cta_coord_in_cluster
        self.mma_cta_index = cta_coord_in_cluster[0] % self.mma_cta_count
        self.is_leader_cta = self.mma_cta_index == 0
        self.main_vmnk_coord = (
            self.mma_cta_index,
            cta_coord_in_cluster[0] // self.mma_cta_count,
            cta_coord_in_cluster[1],
            cta_coord_in_cluster[2],
        )
        bound_hidden_size = (
            self.hidden_size if isinstance(self.hidden_size, int) else hidden_size
        )
        bound_intermediate_gateup_size = (
            self.intermediate_gateup_size
            if isinstance(self.intermediate_gateup_size, int)
            else intermediate_gateup_size
        )
        self.fc1_k_tile_count = ceil_div(bound_hidden_size, self.mma_tile_k)
        self.fc2_k_tile_count = ceil_div(
            bound_intermediate_gateup_size // 2, self.mma_tile_k
        )
        self.sfb_vmnk_coord = (
            0,
            cta_coord_in_cluster[0],
            cta_coord_in_cluster[1],
            cta_coord_in_cluster[2],
        )

    @cute.jit
    def create_ab_pipeline(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer
    ) -> pipeline.PipelineAsync:
        """Create the shared two-producer TMA-to-UMMA pipeline.

        Each producer arrival contributes the same accounting share.  The two
        arrivals together register the combined A, B, SFA, and SFB bytes; an
        individual share does not need to match the bytes issued by that warp.
        Gather-B never arrives: it only lets its completions decrement the byte
        count that TMA-B already registered.
        """
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
        num_tma_consumers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_consumers
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=smem_workspace.ptr(
                self.ab_pipeline_mbarriers_region, smem_base
            ),
            num_stages=self.num_ab_pipeline_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.ab_pipeline_tx_count_per_producer,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def create_weight_pipeline(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer
    ) -> pipeline.PipelineAsync:
        """Create the weight half of the split TMA-to-UMMA pipeline.

        One producer, TMA-A, and its expectation is exactly the A and SFA bytes it
        issues -- unlike the shared pipeline, where two producers each register
        half of a combined count that matches neither warp's traffic.
        """
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        num_tma_consumers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_consumers
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=smem_workspace.ptr(
                self.ab_pipeline_mbarriers_region, smem_base
            ),
            num_stages=self.num_weight_ab_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.weight_pipeline_tx_count,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def create_token_pipeline(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer
    ) -> pipeline.PipelineAsync:
        """Create the token half of the split TMA-to-UMMA pipeline.

        One producer, TMA-B, registering the B and SFB bytes. Gather-B still never
        arrives: on FC1 it only lets its completions decrement the count TMA-B
        registered, which is why the two share this pipeline rather than getting
        one each.
        """
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        num_tma_consumers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_consumers
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=smem_workspace.ptr(
                self.token_pipeline_mbarriers_region, smem_base
            ),
            num_stages=self.num_token_ab_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.token_pipeline_tx_count,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def create_acc_pipeline(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer
    ):
        """Create the UMMA-to-epilogue accumulator pipeline."""
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_thread_count = (
            self.num_accumulator_consumer_warps_per_cta * 32 * self.mma_cta_count
        )
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_thread_count
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=smem_workspace.ptr(
                self.acc_pipeline_mbarriers_region, smem_base
            ),
            num_stages=self.num_accumulator_pipeline_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def create_tmem_allocator(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        *,
        allocator_warp_id: int,
    ):
        """Bind the SM100/103 TMEM allocator to finalized SMEM regions."""
        allocation_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_allocation_barrier_id,
            num_threads=32 * (1 + self.num_accumulator_consumer_warps_per_cta),
        )
        return utils.TmemAllocator(
            smem_workspace.ptr(self.tmem_holding_buffer_region, smem_base),
            barrier_for_retrieve=allocation_barrier,
            allocator_warp_id=allocator_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=smem_workspace.ptr(
                self.tmem_deallocation_mbarrier_region, smem_base
            ),
            arch=self.architecture,
        )

    def _s2t_copy_and_partition(
        self, smem_tensor: cute.Tensor, tmem_tensor: cute.Tensor
    ):
        compact_smem_tensor = cute.filter_zeros(smem_tensor)
        compact_tmem_tensor = cute.filter_zeros(tmem_tensor)
        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        copy_atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), self.sf_dtype)
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, compact_tmem_tensor)
        thread_copy = tiled_copy.get_slice(0)

        mn_mode = cute.get(compact_smem_tensor.layout, mode=[0, 0])
        mn_mode = cute.append(mn_mode, cute.make_layout((4,), stride=(0,)))
        broadcast_layout = cute.append(
            cute.group_modes(mn_mode, 0),
            cute.get(compact_smem_tensor.layout, mode=[0, 1]),
        )
        broadcast_layout = cute.append(
            cute.group_modes(broadcast_layout, 0),
            cute.get(compact_smem_tensor.layout, mode=[1]),
        )
        broadcast_layout = cute.append(
            broadcast_layout, cute.get(compact_smem_tensor.layout, mode=[2])
        )
        broadcast_layout = cute.append(
            broadcast_layout, cute.get(compact_smem_tensor.layout, mode=[3])
        )
        broadcast_smem_tensor = cute.make_tensor(
            compact_smem_tensor.iterator, broadcast_layout
        )

        partitioned_smem = thread_copy.partition_S(broadcast_smem_tensor)
        partitioned_smem = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy, partitioned_smem
        )
        partitioned_tmem = thread_copy.partition_D(compact_tmem_tensor)
        return tiled_copy, partitioned_smem, partitioned_tmem

    def _weight_copy_kwargs(self, policy) -> dict:
        if cutlass.const_expr(self.weight_evict == "normal"):
            return {}
        return {"cache_policy": policy}

    @cute.jit
    def _weight_cache_policy(self, work_tile, expert_num_tokens: cute.Tensor):
        if cutlass.const_expr(self.weight_evict == "first"):
            return Int64(_EVICT_FIRST)
        policy = Int64(_EVICT_NORMAL)
        if cutlass.const_expr(self.token_slot_rows != self.plan_tile_n):
            # Strided slots spread an expert's routes over its tiles, so any of
            # them may be partial; the last one is the last reader.
            if (work_tile.tile_n_idx + Int32(1)) * Int32(
                self.plan_tile_n
            ) >= expert_num_tokens[work_tile.expert_idx]:
                policy = Int64(_EVICT_FIRST)
        elif work_tile.valid_tokens_in_cta_tile < Int32(self.plan_tile_n):
            policy = Int64(_EVICT_FIRST)
        return policy

    @cute.jit
    def _weight_tile(self, work_tile, phase_index: int):
        """The tile as the weight loads address it, and its MMA M tile within that expert entry.

        ``phase_index`` is 0 for FC1 and 1 for FC2. Under die-local weights M
        tile ``m`` of the second half is M tile ``m - half`` of entry ``E + e``.
        """
        mma_tile_m = work_tile.tile_m_idx // self.mma_cta_count
        weight_tile = work_tile
        if cutlass.const_expr(self.die_local_weights):
            half_tiles = Int32(self.die_half_mma_tiles[phase_index])
            half = mma_tile_m // half_tiles
            mma_tile_m = mma_tile_m - half * half_tiles
            weight_tile = dataclasses.replace(
                work_tile,
                expert_idx=work_tile.expert_idx
                + half * Int32(self.weight_expert_count),
            )
        return weight_tile, mma_tile_m

    @cute.jit
    def run_tma_a(
        self,
        *,
        fc1_tma_a_tensor: cute.Tensor,
        fc1_tma_a_atom: cute.CopyAtom,
        fc1_tma_sfa_tensor: cute.Tensor,
        fc1_tma_sfa_atom: cute.CopyAtom,
        fc2_tma_a_tensor: cute.Tensor,
        fc2_tma_a_atom: cute.CopyAtom,
        fc2_tma_sfa_tensor: cute.Tensor,
        fc2_tma_sfa_atom: cute.CopyAtom,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        expert_num_tokens: cute.Tensor,
        claim_gate: Optional[cute.Pointer] = None,
        claim_lead_k_tiles: int = 1,
        shared_tma: Optional[Tuple] = None,
    ) -> None:
        """Run the TMA-A warp that loads weights and their scale factors.

        ``claim_gate``, when given, is the scheduler's lazy-claim barrier:
        this warp arrives on it once per work tile, when ``claim_lead_k_tiles`` of
        the tile's K tiles are left to issue (at the tile's start if it has no
        more), so the next claim lands while the stream still has work queued.

        ``shared_tma`` holds the shared expert's weight atoms and tensors, FC1
        then FC2 (``prepare_shared_tma_load_params``), with a shared expert.
        """
        if cutlass.const_expr(self.shared_expert_marker is not None):
            self._run_tma_a_with_shared_expert(
                fc1_tma_a_tensor=fc1_tma_a_tensor,
                fc1_tma_a_atom=fc1_tma_a_atom,
                fc1_tma_sfa_tensor=fc1_tma_sfa_tensor,
                fc1_tma_sfa_atom=fc1_tma_sfa_atom,
                fc2_tma_a_tensor=fc2_tma_a_tensor,
                fc2_tma_a_atom=fc2_tma_a_atom,
                fc2_tma_sfa_tensor=fc2_tma_sfa_tensor,
                fc2_tma_sfa_atom=fc2_tma_sfa_atom,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=ab_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
                expert_num_tokens=expert_num_tokens,
                claim_gate=claim_gate,
                claim_lead_k_tiles=claim_lead_k_tiles,
                shared_tma=shared_tma,
            )
        else:
            multicast_mask = None
            if cutlass.const_expr(self.is_a_mcast or self.use_2cta_instrs):
                multicast_mask = tma_multicast_mask(
                    self.cluster_shape_mn,
                    None,
                    self.cta_coord_in_cluster,
                    None,
                    self.use_2cta_instrs,
                    "a",
                )

            a_cta_layout = cute.make_layout(
                cute.slice_(self.cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            sfa_cta_layout = a_cta_layout
            thread_mma = self.tiled_mma.get_slice(self.mma_cta_index)

            # The two phases keep separate loop bodies rather than selecting atoms from
            # a shared one: `is_fc1` is a runtime predicate, so the atoms and the
            # per-phase tensor views cannot be chosen by a Python conditional.
            work_tile = sched_consumer.consume_work()
            while work_tile.is_valid_tile:
                is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
                if is_fc1:
                    iket.range_push("tma_weight_fc1")
                    k_tile_count = self.fc1_k_tile_count
                    weight_policy = self._weight_cache_policy(
                        work_tile, expert_num_tokens
                    )
                    weight_tile, mma_tile_m = self._weight_tile(work_tile, 0)
                    real_a, a_descriptor_ptr = kernel_extension.get_gmem_tensor(
                        "a", fc1_tma_a_tensor, weight_tile
                    )
                    real_sfa, sfa_descriptor_ptr = kernel_extension.get_gmem_tensor(
                        "sfa", fc1_tma_sfa_tensor, weight_tile
                    )

                    global_a = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
                        (None, None, None),
                    )
                    global_sfa = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
                        (None, None, None),
                    )
                    partitioned_global_a = thread_mma.partition_A(global_a)
                    partitioned_global_sfa = thread_mma.partition_A(global_sfa)

                    partitioned_smem_a, partitioned_global_a = cpasync.tma_partition(
                        fc1_tma_a_atom,
                        self.main_vmnk_coord[2],
                        a_cta_layout,
                        cute.group_modes(self.a_smem_tensor, 0, 3),
                        cute.group_modes(partitioned_global_a, 0, 3),
                    )
                    partitioned_smem_sfa, partitioned_global_sfa = (
                        cpasync.tma_partition(
                            fc1_tma_sfa_atom,
                            self.main_vmnk_coord[2],
                            sfa_cta_layout,
                            cute.group_modes(self.sfa_smem_tensor, 0, 3),
                            cute.group_modes(partitioned_global_sfa, 0, 3),
                        )
                    )
                    partitioned_smem_sfa = cute.filter_zeros(partitioned_smem_sfa)
                    partitioned_global_sfa = cute.filter_zeros(partitioned_global_sfa)

                    global_a_slice = partitioned_global_a[(None, mma_tile_m, None, 0)]
                    global_sfa_slice = partitioned_global_sfa[
                        (None, mma_tile_m, None, 0)
                    ]

                    claim_k_tile = k_tile_count - min(claim_lead_k_tiles, k_tile_count)
                    empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                    for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                        if cutlass.const_expr(claim_gate is not None):
                            if (k_tile_idx == claim_k_tile) & (
                                cute.arch.lane_idx() == Int32(0)
                            ):
                                cute.arch.mbarrier_arrive(claim_gate)
                        if cutlass.const_expr(self.fc1_tx_split_by_operand):
                            ab_pipeline.producer_acquire(
                                ab_pipeline_state,
                                empty_status,
                                expected_tx=Int32(self.weight_pipeline_tx_count),
                            )
                        else:
                            ab_pipeline.producer_acquire(
                                ab_pipeline_state, empty_status
                            )
                        tma_barrier = ab_pipeline.producer_get_barrier(
                            ab_pipeline_state
                        )
                        stage_index = ab_pipeline_state.index
                        ab_pipeline_state.advance()
                        if k_tile_idx + 1 < k_tile_count:
                            empty_status = ab_pipeline.producer_try_acquire(
                                ab_pipeline_state
                            )
                        cute.copy(
                            fc1_tma_a_atom,
                            global_a_slice[(None, k_tile_idx)],
                            partitioned_smem_a[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=a_descriptor_ptr,
                            mcast_mask=multicast_mask,
                            **self._weight_copy_kwargs(weight_policy),
                        )
                        cute.copy(
                            fc1_tma_sfa_atom,
                            global_sfa_slice[(None, k_tile_idx)],
                            partitioned_smem_sfa[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=sfa_descriptor_ptr,
                            mcast_mask=multicast_mask,
                            **self._weight_copy_kwargs(weight_policy),
                        )
                else:
                    iket.range_push("tma_weight_fc2")
                    k_tile_count = self.fc2_k_tile_count
                    weight_policy = self._weight_cache_policy(
                        work_tile, expert_num_tokens
                    )
                    weight_tile, mma_tile_m = self._weight_tile(work_tile, 1)
                    real_a, a_descriptor_ptr = kernel_extension.get_gmem_tensor(
                        "a", fc2_tma_a_tensor, weight_tile
                    )
                    real_sfa, sfa_descriptor_ptr = kernel_extension.get_gmem_tensor(
                        "sfa", fc2_tma_sfa_tensor, weight_tile
                    )

                    global_a = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
                        (None, None, None),
                    )
                    global_sfa = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
                        (None, None, None),
                    )
                    partitioned_global_a = thread_mma.partition_A(global_a)
                    partitioned_global_sfa = thread_mma.partition_A(global_sfa)

                    partitioned_smem_a, partitioned_global_a = cpasync.tma_partition(
                        fc2_tma_a_atom,
                        self.main_vmnk_coord[2],
                        a_cta_layout,
                        cute.group_modes(self.a_smem_tensor, 0, 3),
                        cute.group_modes(partitioned_global_a, 0, 3),
                    )
                    partitioned_smem_sfa, partitioned_global_sfa = (
                        cpasync.tma_partition(
                            fc2_tma_sfa_atom,
                            self.main_vmnk_coord[2],
                            sfa_cta_layout,
                            cute.group_modes(self.sfa_smem_tensor, 0, 3),
                            cute.group_modes(partitioned_global_sfa, 0, 3),
                        )
                    )
                    partitioned_smem_sfa = cute.filter_zeros(partitioned_smem_sfa)
                    partitioned_global_sfa = cute.filter_zeros(partitioned_global_sfa)

                    global_a_slice = partitioned_global_a[(None, mma_tile_m, None, 0)]
                    global_sfa_slice = partitioned_global_sfa[
                        (None, mma_tile_m, None, 0)
                    ]

                    claim_k_tile = k_tile_count - min(claim_lead_k_tiles, k_tile_count)
                    empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                    for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                        if cutlass.const_expr(claim_gate is not None):
                            if (k_tile_idx == claim_k_tile) & (
                                cute.arch.lane_idx() == Int32(0)
                            ):
                                cute.arch.mbarrier_arrive(claim_gate)
                        ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
                        tma_barrier = ab_pipeline.producer_get_barrier(
                            ab_pipeline_state
                        )
                        stage_index = ab_pipeline_state.index
                        ab_pipeline_state.advance()
                        if k_tile_idx + 1 < k_tile_count:
                            empty_status = ab_pipeline.producer_try_acquire(
                                ab_pipeline_state
                            )
                        cute.copy(
                            fc2_tma_a_atom,
                            global_a_slice[(None, k_tile_idx)],
                            partitioned_smem_a[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=a_descriptor_ptr,
                            mcast_mask=multicast_mask,
                            **self._weight_copy_kwargs(weight_policy),
                        )
                        cute.copy(
                            fc2_tma_sfa_atom,
                            global_sfa_slice[(None, k_tile_idx)],
                            partitioned_smem_sfa[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=sfa_descriptor_ptr,
                            mcast_mask=multicast_mask,
                            **self._weight_copy_kwargs(weight_policy),
                        )

                iket.range_pop()
                work_tile = sched_consumer.consume_work()

            ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def _run_tma_a_with_shared_expert(
        self,
        *,
        fc1_tma_a_tensor: cute.Tensor,
        fc1_tma_a_atom: cute.CopyAtom,
        fc1_tma_sfa_tensor: cute.Tensor,
        fc1_tma_sfa_atom: cute.CopyAtom,
        fc2_tma_a_tensor: cute.Tensor,
        fc2_tma_a_atom: cute.CopyAtom,
        fc2_tma_sfa_tensor: cute.Tensor,
        fc2_tma_sfa_atom: cute.CopyAtom,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        expert_num_tokens: cute.Tensor,
        claim_gate: Optional[cute.Pointer] = None,
        claim_lead_k_tiles: int = 1,
        shared_tma: Optional[Tuple] = None,
    ) -> None:
        """``run_tma_a`` with a shared expert: the default loop's tile loads in helpers, which a shared tile's join.

        A routed tile's loads (``_load_routed_weight_tile``) are the default
        loop body's; a shared tile's (``_load_shared_weight_tile``) use its own
        descriptors and stage bytes.
        """
        multicast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.use_2cta_instrs):
            multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                None,
                self.cta_coord_in_cluster,
                None,
                self.use_2cta_instrs,
                "a",
            )

        a_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        sfa_cta_layout = a_cta_layout
        thread_mma = self.tiled_mma.get_slice(self.mma_cta_index)
        tile_loads = dict(
            ab_pipeline=ab_pipeline,
            kernel_extension=kernel_extension,
            thread_mma=thread_mma,
            a_cta_layout=a_cta_layout,
            sfa_cta_layout=sfa_cta_layout,
            multicast_mask=multicast_mask,
            claim_gate=claim_gate,
            claim_lead_k_tiles=claim_lead_k_tiles,
        )

        # The phases keep separate loads rather than selecting atoms from a shared
        # one: `is_fc1` is a runtime predicate, so the atoms and the per-phase
        # tensor views cannot be chosen by a Python conditional.
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            if cutlass.const_expr(self.shared_expert_marker is not None):
                if work_tile.expert_idx == Int32(self.shared_expert_marker):
                    ab_pipeline_state = self._load_shared_weight_tile(
                        work_tile, is_fc1, shared_tma, ab_pipeline_state, **tile_loads
                    )
                else:
                    ab_pipeline_state = self._load_routed_weight_tile(
                        work_tile,
                        is_fc1,
                        (
                            fc1_tma_a_tensor,
                            fc1_tma_a_atom,
                            fc1_tma_sfa_tensor,
                            fc1_tma_sfa_atom,
                        ),
                        (
                            fc2_tma_a_tensor,
                            fc2_tma_a_atom,
                            fc2_tma_sfa_tensor,
                            fc2_tma_sfa_atom,
                        ),
                        expert_num_tokens,
                        ab_pipeline_state,
                        **tile_loads,
                    )
            else:
                ab_pipeline_state = self._load_routed_weight_tile(
                    work_tile,
                    is_fc1,
                    (
                        fc1_tma_a_tensor,
                        fc1_tma_a_atom,
                        fc1_tma_sfa_tensor,
                        fc1_tma_sfa_atom,
                    ),
                    (
                        fc2_tma_a_tensor,
                        fc2_tma_a_atom,
                        fc2_tma_sfa_tensor,
                        fc2_tma_sfa_atom,
                    ),
                    expert_num_tokens,
                    ab_pipeline_state,
                    **tile_loads,
                )
            work_tile = sched_consumer.consume_work()

        ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def _load_routed_weight_tile(
        self,
        work_tile,
        is_fc1,
        fc1_tma: Tuple,
        fc2_tma: Tuple,
        expert_num_tokens: cute.Tensor,
        ab_pipeline_state: pipeline.PipelineState,
        **tile_loads,
    ) -> pipeline.PipelineState:
        """One routed work tile's weight and scale-factor loads; returns the advanced pipeline state."""
        fc1_tma_a_tensor, fc1_tma_a_atom, fc1_tma_sfa_tensor, fc1_tma_sfa_atom = fc1_tma
        fc2_tma_a_tensor, fc2_tma_a_atom, fc2_tma_sfa_tensor, fc2_tma_sfa_atom = fc2_tma
        if is_fc1:
            iket.range_push("tma_weight_fc1")
            weight_policy = self._weight_cache_policy(work_tile, expert_num_tokens)
            weight_tile, mma_tile_m = self._weight_tile(work_tile, 0)
            ab_pipeline_state = self._load_weight_tile(
                fc1_tma_a_tensor,
                fc1_tma_a_atom,
                fc1_tma_sfa_tensor,
                fc1_tma_sfa_atom,
                self.a_smem_tensor,
                weight_tile,
                mma_tile_m,
                self.fc1_k_tile_count,
                weight_policy,
                self.weight_pipeline_tx_count if self.fc1_tx_split_by_operand else None,
                ab_pipeline_state,
                **tile_loads,
            )
        else:
            iket.range_push("tma_weight_fc2")
            weight_policy = self._weight_cache_policy(work_tile, expert_num_tokens)
            weight_tile, mma_tile_m = self._weight_tile(work_tile, 1)
            ab_pipeline_state = self._load_weight_tile(
                fc2_tma_a_tensor,
                fc2_tma_a_atom,
                fc2_tma_sfa_tensor,
                fc2_tma_sfa_atom,
                self.a_smem_tensor,
                weight_tile,
                mma_tile_m,
                self.fc2_k_tile_count,
                weight_policy,
                None,
                ab_pipeline_state,
                **tile_loads,
            )
        iket.range_pop()
        return ab_pipeline_state

    @cute.jit
    def _load_shared_weight_tile(
        self,
        work_tile,
        is_fc1,
        shared_tma: Tuple,
        ab_pipeline_state: pipeline.PipelineState,
        **tile_loads,
    ) -> pipeline.PipelineState:
        """One shared-expert work tile's weight and scale-factor loads; returns the advanced pipeline state.

        The shared weights are one expert entry and are not die-local. Every stage
        expects the producer's usual bytes plus the E4M3 tile's extra A bytes. The
        last shared token tile reads its weight tiles for the last time, so under
        the adaptive policy its loads are evict-first, as an expert's last token
        tile's are.
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
        ) = shared_tma
        weight_tile = dataclasses.replace(work_tile, expert_idx=Int32(0))
        mma_tile_m = work_tile.tile_m_idx // self.mma_cta_count
        policy = Int64(_EVICT_FIRST)
        if cutlass.const_expr(self.weight_evict == "adaptive"):
            policy = Int64(_EVICT_NORMAL)
            if work_tile.tile_n_idx == Int32(self.shared_token_tiles - 1):
                policy = Int64(_EVICT_FIRST)
        weight_tx = (
            self.weight_pipeline_tx_count
            if self.uses_asymmetric_ab_stages
            else self.ab_pipeline_tx_count_per_producer
        )
        if is_fc1:
            iket.range_push("tma_weight_shared_fc1")
            fc1_weight_tx = (
                self.weight_pipeline_tx_count
                if self.fc1_tx_split_by_operand
                else weight_tx
            )
            ab_pipeline_state = self._load_weight_tile(
                fc1_tma_a_tensor,
                fc1_tma_a_atom,
                fc1_tma_sfa_tensor,
                fc1_tma_sfa_atom,
                self.shared_a_smem_tensor,
                weight_tile,
                mma_tile_m,
                self.fc1_k_tile_count,
                policy,
                fc1_weight_tx + self.shared_a_extra_tx,
                ab_pipeline_state,
                **tile_loads,
            )
        else:
            iket.range_push("tma_weight_shared_fc2")
            ab_pipeline_state = self._load_weight_tile(
                fc2_tma_a_tensor,
                fc2_tma_a_atom,
                fc2_tma_sfa_tensor,
                fc2_tma_sfa_atom,
                self.shared_a_smem_tensor,
                weight_tile,
                mma_tile_m,
                self.fc2_k_tile_count,
                policy,
                weight_tx + self.shared_a_extra_tx,
                ab_pipeline_state,
                **tile_loads,
            )
        iket.range_pop()
        return ab_pipeline_state

    @cute.jit
    def _load_weight_tile(
        self,
        tma_a_tensor: cute.Tensor,
        tma_a_atom: cute.CopyAtom,
        tma_sfa_tensor: cute.Tensor,
        tma_sfa_atom: cute.CopyAtom,
        a_smem_tensor: cute.Tensor,
        weight_tile,
        mma_tile_m,
        k_tile_count: int,
        weight_policy,
        expected_tx: Optional[int],
        ab_pipeline_state: pipeline.PipelineState,
        *,
        ab_pipeline: pipeline.PipelineAsync,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        thread_mma,
        a_cta_layout,
        sfa_cta_layout,
        multicast_mask,
        claim_gate: Optional[cute.Pointer],
        claim_lead_k_tiles: int,
    ) -> pipeline.PipelineState:
        """Every K tile of one work tile's weights and scale factors into the AB stages.

        ``expected_tx`` (None: the pipeline's own count) is the stage expectation
        this producer registers. Returns the advanced pipeline state.
        """
        real_a, a_descriptor_ptr = kernel_extension.get_gmem_tensor(
            "a", tma_a_tensor, weight_tile
        )
        real_sfa, sfa_descriptor_ptr = kernel_extension.get_gmem_tensor(
            "sfa", tma_sfa_tensor, weight_tile
        )

        global_a = cute.local_tile(
            real_a, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None, None)
        )
        global_sfa = cute.local_tile(
            real_sfa,
            cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
            (None, None, None),
        )
        partitioned_global_a = thread_mma.partition_A(global_a)
        partitioned_global_sfa = thread_mma.partition_A(global_sfa)

        partitioned_smem_a, partitioned_global_a = cpasync.tma_partition(
            tma_a_atom,
            self.main_vmnk_coord[2],
            a_cta_layout,
            cute.group_modes(a_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_a, 0, 3),
        )
        partitioned_smem_sfa, partitioned_global_sfa = cpasync.tma_partition(
            tma_sfa_atom,
            self.main_vmnk_coord[2],
            sfa_cta_layout,
            cute.group_modes(self.sfa_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_sfa, 0, 3),
        )
        partitioned_smem_sfa = cute.filter_zeros(partitioned_smem_sfa)
        partitioned_global_sfa = cute.filter_zeros(partitioned_global_sfa)

        global_a_slice = partitioned_global_a[(None, mma_tile_m, None, 0)]
        global_sfa_slice = partitioned_global_sfa[(None, mma_tile_m, None, 0)]

        claim_k_tile = k_tile_count - min(claim_lead_k_tiles, k_tile_count)
        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
        for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
            if cutlass.const_expr(claim_gate is not None):
                if (k_tile_idx == claim_k_tile) & (cute.arch.lane_idx() == Int32(0)):
                    cute.arch.mbarrier_arrive(claim_gate)
            if cutlass.const_expr(expected_tx is not None):
                ab_pipeline.producer_acquire(
                    ab_pipeline_state, empty_status, expected_tx=Int32(expected_tx)
                )
            else:
                ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            ab_pipeline_state.advance()
            if k_tile_idx + 1 < k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            cute.copy(
                tma_a_atom,
                global_a_slice[(None, k_tile_idx)],
                partitioned_smem_a[(None, stage_index)],
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=a_descriptor_ptr,
                mcast_mask=multicast_mask,
                **self._weight_copy_kwargs(weight_policy),
            )
            cute.copy(
                tma_sfa_atom,
                global_sfa_slice[(None, k_tile_idx)],
                partitioned_smem_sfa[(None, stage_index)],
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=sfa_descriptor_ptr,
                mcast_mask=multicast_mask,
                **self._weight_copy_kwargs(weight_policy),
            )
        return ab_pipeline_state

    @cute.jit
    def run_tma_b(
        self,
        *,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom: cute.CopyAtom,
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom: cute.CopyAtom,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom: cute.CopyAtom,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        fc1_done_counter_pointer: cute.Pointer,
        fc2_spin_threshold: Int32,
        fc1_sf_ready_pointer: cute.Pointer,
        expert_num_tokens: cute.Tensor,
        fc1_tile_ready_pointer: Optional[cute.Pointer] = None,
        first_fc1_stamp: Optional[cute.Pointer] = None,
        claim_gate: Optional[cute.Pointer] = None,
        fc1_ready_counts_routes: bool = False,
        fc2_tma_b_rows_atom: Optional[cute.CopyAtom] = None,
        fc1_tma_b_atom: Optional[cute.CopyAtom] = None,
        fc1_tma_b_rows_atom: Optional[cute.CopyAtom] = None,
        fc1_tma_b_pool_atom: Optional[cute.CopyAtom] = None,
        token_pool_plan: Optional[cute.Tensor] = None,
        shared_tma: Optional[Tuple] = None,
        shared_token_rows: int = 0,
    ) -> None:
        """Run the TMA-B warp: FC1 scale factors (and token data under the token pool), FC2 token data and scale factors.

        FC1's token payload is Gather-B's job, so this warp contributes only the
        SFB bytes there. It still owns the whole static transaction expectation
        for the stage, which is why the named barrier below has to release the
        Gather-B warps only after ``producer_acquire`` has registered it. Under
        ``fc1_token_pool`` this warp loads FC1's tokens itself from their pool
        rows, through ``fc1_tma_b_atom`` (a box of the CTA's half of the tile N)
        or ``fc1_tma_b_rows_atom`` (16 live rows), and meets no Gather-B warp.
        Under a pool with a capacity it does so for the token tiles the launch's
        ``token_pool_plan`` (the main kernel's ``_plan_token_pool``) places in the
        pool, from rows in processing order through ``fc1_tma_b_pool_atom``, and
        meets the Gather-B warps on every other tile.

        The FC1 scale factors are gated per expert rather than once for the whole
        pool: the communication kernel counts SF blocks into
        ``fc1_sf_ready_pointer``, and both sides walk experts in the same order, so
        this tile only has to outlast its own expert's blocks.

        ``first_fc1_stamp``, when given, receives the %globaltimer at which this
        CTA's first FC1 tile passed that gate.

        ``claim_gate``, when given, is the scheduler's lazy-claim barrier:
        this warp arrives on it once per work tile, when the tile has passed its
        gate (the FC1 scale factors or the FC2 ``fc1_done`` count).

        ``fc1_ready_counts_routes`` makes the gate's target the expert's route
        count rather than its SF blocks, for a dispatch that counts routes.
        ``fc1_tile_ready_pointer``, when given, gates each FC1 tile on its own
        token tile instead: the slot ``cumulative_token_block_count +
        tile_n_idx`` counts the tile's landed routes, and the gate waits for the
        tile's valid tokens.
        ``fc2_tma_b_rows_atom`` is the live-row FC2 token descriptor
        (``fc2_b_live_rows``).

        With a shared expert ``shared_tma`` holds its token atoms over the staged
        input (``prepare_shared_tma_load_params``) and ``shared_token_rows`` the
        routes of its token tiles but the last: a shared FC1 tile passes no gate
        and loads its tokens from there (``_load_shared_fc1_tokens``), and a
        shared FC2 tile is an FC2 tile like any other.
        """
        if cutlass.const_expr(self.shared_expert_marker is not None):
            self._run_tma_b_with_shared_expert(
                fc1_tma_sfb_tensor=fc1_tma_sfb_tensor,
                fc1_tma_sfb_atom=fc1_tma_sfb_atom,
                fc2_tma_b_tensor=fc2_tma_b_tensor,
                fc2_tma_b_atom=fc2_tma_b_atom,
                fc2_tma_sfb_tensor=fc2_tma_sfb_tensor,
                fc2_tma_sfb_atom=fc2_tma_sfb_atom,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=ab_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
                fc1_done_counter_pointer=fc1_done_counter_pointer,
                fc2_spin_threshold=fc2_spin_threshold,
                fc1_sf_ready_pointer=fc1_sf_ready_pointer,
                expert_num_tokens=expert_num_tokens,
                fc1_tile_ready_pointer=fc1_tile_ready_pointer,
                first_fc1_stamp=first_fc1_stamp,
                claim_gate=claim_gate,
                fc1_ready_counts_routes=fc1_ready_counts_routes,
                fc2_tma_b_rows_atom=fc2_tma_b_rows_atom,
                fc1_tma_b_atom=fc1_tma_b_atom,
                fc1_tma_b_rows_atom=fc1_tma_b_rows_atom,
                fc1_tma_b_pool_atom=fc1_tma_b_pool_atom,
                token_pool_plan=token_pool_plan,
                shared_tma=shared_tma,
                shared_token_rows=shared_token_rows,
            )
        else:
            b_multicast_mask = None
            sfb_multicast_mask = None
            if cutlass.const_expr(self.is_b_mcast or self.use_2cta_instrs):
                b_multicast_mask = tma_multicast_mask(
                    self.cluster_shape_mn,
                    None,
                    self.cta_coord_in_cluster,
                    None,
                    self.use_2cta_instrs,
                    "b",
                )
                sfb_multicast_mask = tma_multicast_mask(
                    self.cluster_shape_mn,
                    None,
                    self.cta_coord_in_cluster,
                    None,
                    self.use_2cta_instrs,
                    "sfb",
                )

            b_cta_layout = cute.make_layout(
                cute.slice_(self.cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            sfb_cta_layout = cute.make_layout(
                cute.slice_(self.cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
            )
            thread_mma = self.tiled_mma.get_slice(self.mma_cta_index)
            thread_mma_sfb = self.tiled_mma_sfb.get_slice(self.mma_cta_index)

            fc1_stamped = cutlass.Boolean(False)
            work_tile = sched_consumer.consume_work()
            while work_tile.is_valid_tile:
                is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
                if is_fc1:
                    iket.range_push("tma_token_fc1")
                    iket.range_push("tma_b.wait_expert_sf_ready")
                    expert_idx = work_tile.expert_idx
                    if cutlass.const_expr(fc1_tile_ready_pointer is not None):
                        spin_wait(
                            fc1_tile_ready_pointer
                            + work_tile.cumulative_token_block_count
                            + work_tile.tile_n_idx,
                            lambda value: value == work_tile.valid_tokens_in_cta_tile,
                            scope="gpu",
                        )
                    else:
                        if cutlass.const_expr(fc1_ready_counts_routes):
                            fc1_ready_target = expert_num_tokens[expert_idx]
                        else:
                            fc1_ready_target = expert_pool_rows(
                                expert_num_tokens[expert_idx],
                                Int32(self.sf_padding_block),
                                self.plan_tile_n,
                                self.token_slot_rows,
                            ) // Int32(self.sf_padding_block)
                        spin_wait(
                            fc1_sf_ready_pointer + expert_idx,
                            lambda value: value == fc1_ready_target,
                            scope="gpu",
                        )
                    # Orders the gate's generic acquire before the scale-factor TMA loads of the rows it counts.
                    sentinel_ptx.fence_proxy_async_global()
                    if cutlass.const_expr(claim_gate is not None):
                        if cute.arch.lane_idx() == Int32(0):
                            cute.arch.mbarrier_arrive(claim_gate)
                    if cutlass.const_expr(first_fc1_stamp is not None):
                        if not fc1_stamped:
                            cute.make_tensor(first_fc1_stamp, cute.make_layout(1))[
                                0
                            ] = sentinel_ptx.globaltimer_ns()
                            fc1_stamped = cutlass.Boolean(True)
                    iket.range_pop()
                    k_tile_count = self.fc1_k_tile_count
                    real_sfb, sfb_descriptor_ptr = kernel_extension.get_gmem_tensor(
                        "sfb", fc1_tma_sfb_tensor, work_tile
                    )

                    global_sfb = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    partitioned_global_sfb = thread_mma_sfb.partition_B(global_sfb)

                    partitioned_smem_sfb, partitioned_global_sfb = (
                        cpasync.tma_partition(
                            fc1_tma_sfb_atom,
                            self.sfb_vmnk_coord[1],
                            sfb_cta_layout,
                            cute.group_modes(self.sfb_smem_tensor, 0, 3),
                            cute.group_modes(partitioned_global_sfb, 0, 3),
                        )
                    )
                    partitioned_smem_sfb = cute.filter_zeros(partitioned_smem_sfb)
                    partitioned_global_sfb = cute.filter_zeros(partitioned_global_sfb)

                    global_sfb_slice = partitioned_global_sfb[
                        (None, self._sfb_tile_index(work_tile.tile_n_idx), None, 0)
                    ]

                    fc1_expected_tx = Int32(self.ab_pipeline_tx_count_per_producer)
                    if cutlass.const_expr(
                        self.fc1_tx_split_by_operand or self.uses_asymmetric_ab_stages
                    ):
                        fc1_expected_tx = Int32(self.token_pipeline_tx_count)
                    fc1_stage_rows = self.cta_tile_n // self.mma_cta_count
                    fc1_stage_boxes = fc1_stage_rows // self.fc2_b_row_box
                    if cutlass.const_expr(self.fc1_token_pool):
                        # This CTA's rows of the tile, consecutive in the token pool: the
                        # instruction N's half from c * N / 2, as the MMA splits them. The
                        # static box covers the CTA's whole half of the tile N, and the rows
                        # it loads past the live ones feed no MMA column. A tile that leaves
                        # part of that box unread loads only its live rows, in 16-row boxes,
                        # and registers only their bytes, as FC2 does past N128.
                        fc1_live_boxes = Int32(fc1_stage_boxes)
                        fc1_cta_rows = Int32(fc1_stage_rows)
                        if cutlass.const_expr(self.runtime_instruction_n):
                            fc1_cta_rows = self._instruction_n(work_tile) // Int32(
                                self.mma_cta_count
                            )
                        if cutlass.const_expr(self.fc1_token_pool_bounded):
                            # Under a capacity the tile's rows start at its first route in
                            # processing order, and a tile whose routes end past the
                            # launch's capacity gathers instead (see the gather case below).
                            tile_plan = token_pool_plan[Int32(1) + expert_idx]
                            tile_route = (
                                tile_plan >> Int32(8)
                            ) + work_tile.tile_n_idx * (tile_plan & Int32(255))
                            in_pool = (
                                tile_route + work_tile.valid_tokens_in_cta_tile
                                <= token_pool_plan[0]
                            )
                            gathers = (
                                tile_route + work_tile.valid_tokens_in_cta_tile
                                > token_pool_plan[0]
                            )
                            fc1_tile_row = (
                                tile_route + Int32(self.mma_cta_index) * fc1_cta_rows
                            )
                            gathered_rows = round_up(
                                work_tile.valid_tokens_in_cta_tile,
                                Int32(self.gather4_width),
                            )
                            gather_expected_tx = fc1_expected_tx - (
                                Int32(self.cta_tile_n) - gathered_rows
                            ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
                        else:
                            fc1_tile_row = (
                                work_tile.cumulative_data_physical_row
                                + work_tile.tile_n_idx * Int32(self.token_slot_rows)
                                + Int32(self.mma_cta_index) * fc1_cta_rows
                            )
                        if cutlass.const_expr(self.fc1_b_live_rows):
                            fc1_live_boxes = (
                                fc1_cta_rows + Int32(self.fc2_b_row_box - 1)
                            ) // Int32(self.fc2_b_row_box)
                            fc1_expected_tx = fc1_expected_tx - (
                                Int32(self.cta_tile_n)
                                - fc1_live_boxes
                                * Int32(self.fc2_b_row_box * self.mma_cta_count)
                            ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
                        if cutlass.const_expr(self.fc1_token_pool_bounded):
                            fc1_expected_tx = cutlass.select_(
                                in_pool, fc1_expected_tx, gather_expected_tx
                            )
                    else:
                        # Gather-B issues no gather4 group whose rows are all padding (rows past the
                        # tile's valid tokens). Each skipped row is one B row less in every MMA pair of
                        # the multicast group, so the static expectation drops by that many rows. The
                        # gathered rows are the tile's first round_up(valid, 4) under either row split
                        # between the MMA CTAs, as the runtime instruction N is at least that many.
                        gathered_rows = round_up(
                            work_tile.valid_tokens_in_cta_tile,
                            Int32(self.gather4_width),
                        )
                        fc1_expected_tx = fc1_expected_tx - (
                            Int32(self.cta_tile_n) - gathered_rows
                        ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
                    empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                    for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                        ab_pipeline.producer_acquire(
                            ab_pipeline_state, empty_status, expected_tx=fc1_expected_tx
                        )
                        tma_barrier = ab_pipeline.producer_get_barrier(
                            ab_pipeline_state
                        )
                        stage_index = ab_pipeline_state.index
                        if cutlass.const_expr(not self.fc1_token_pool):
                            # TMA-B has waited for this AB stage to become empty and
                            # contributed one of the two static producer transaction
                            # expectations. Rendezvous with the Gather-B warps before any
                            # of them issues Gather4 against this barrier.
                            barrier_sync_unaligned(
                                self.fc1_gather_sync_barrier_id,
                                32 * (1 + self.num_gather_b_warps),
                            )
                        if cutlass.const_expr(self.fc1_token_pool_bounded):
                            if gathers:
                                barrier_sync_unaligned(
                                    self.fc1_gather_sync_barrier_id,
                                    32 * (1 + self.num_gather_b_warps),
                                )
                        ab_pipeline_state.advance()
                        if k_tile_idx + 1 < k_tile_count:
                            empty_status = ab_pipeline.producer_try_acquire(
                                ab_pipeline_state
                            )
                        if cutlass.const_expr(self.fc1_token_pool_bounded):
                            if in_pool:
                                self._load_fc1_pool_stage(
                                    fc1_tma_b_pool_atom,
                                    fc1_tma_b_rows_atom,
                                    stage_index,
                                    tma_barrier,
                                    k_tile_idx,
                                    fc1_tile_row,
                                    fc1_live_boxes,
                                    b_multicast_mask,
                                )
                        elif cutlass.const_expr(self.fc1_token_pool):
                            self._load_fc1_pool_stage(
                                fc1_tma_b_atom,
                                fc1_tma_b_rows_atom,
                                stage_index,
                                tma_barrier,
                                k_tile_idx,
                                fc1_tile_row,
                                fc1_live_boxes,
                                b_multicast_mask,
                            )
                        cute.copy(
                            fc1_tma_sfb_atom,
                            global_sfb_slice[(None, k_tile_idx)],
                            partitioned_smem_sfb[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=sfb_descriptor_ptr,
                            mcast_mask=sfb_multicast_mask,
                        )
                else:
                    iket.range_push("tma_token_fc2")
                    counter_slot = (
                        work_tile.cumulative_token_block_count + work_tile.tile_n_idx
                    )
                    counter_pointer = fc1_done_counter_pointer + counter_slot
                    iket.range_push("tma_token_fc2_wait")
                    spin_wait(
                        counter_pointer,
                        lambda value: value >= fc2_spin_threshold,
                        sleep_cycles=500,
                        peek_status=work_tile.peek_ready,
                    )
                    # Orders the fc1_done acquire (this wait's or the scheduler peek's) before the TMA loads of FC1's output.
                    sentinel_ptx.fence_proxy_async_global()
                    if cutlass.const_expr(claim_gate is not None):
                        if cute.arch.lane_idx() == Int32(0):
                            cute.arch.mbarrier_arrive(claim_gate)
                    iket.range_pop()

                    k_tile_count = self.fc2_k_tile_count
                    real_b, b_descriptor_ptr = kernel_extension.get_gmem_tensor(
                        "b", fc2_tma_b_tensor, work_tile
                    )
                    real_sfb, sfb_descriptor_ptr = kernel_extension.get_gmem_tensor(
                        "sfb", fc2_tma_sfb_tensor, work_tile
                    )
                    if cutlass.const_expr(self.token_slot_rows != self.cta_tile_n):
                        # Token tiles sit one slot apart in the FC1 output pool, while the
                        # B tiling below steps by the tile itself; shift each tile by the
                        # slot padding of the tiles before it.
                        real_b = cute.domain_offset(
                            (
                                work_tile.tile_n_idx
                                * Int32(self.token_slot_rows - self.cta_tile_n),
                                0,
                                0,
                            ),
                            real_b,
                        )
                    if cutlass.const_expr(self.runtime_instruction_n):
                        # The B partition below gives MMA CTA c the tile's rows from
                        # c * tile N / 2; the runtime-N split starts them at c * N / 2.
                        # The box keeps its static height: the rows it loads past the
                        # first N / 2 of each CTA feed no MMA column.
                        real_b = cute.domain_offset(
                            (
                                Int32(self.mma_cta_index)
                                * (
                                    (
                                        self._instruction_n(work_tile)
                                        - Int32(self.cta_tile_n)
                                    )
                                    // Int32(self.mma_cta_count)
                                ),
                                0,
                                0,
                            ),
                            real_b,
                        )

                    global_b = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler_mnk, (0, None, None)),
                        (None, None, None),
                    )
                    global_sfb = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    partitioned_global_b = thread_mma.partition_B(global_b)
                    partitioned_global_sfb = thread_mma_sfb.partition_B(global_sfb)

                    partitioned_smem_b, partitioned_global_b = cpasync.tma_partition(
                        fc2_tma_b_atom,
                        self.main_vmnk_coord[1],
                        b_cta_layout,
                        cute.group_modes(self.b_smem_tensor, 0, 3),
                        cute.group_modes(partitioned_global_b, 0, 3),
                    )
                    partitioned_smem_sfb, partitioned_global_sfb = (
                        cpasync.tma_partition(
                            fc2_tma_sfb_atom,
                            self.sfb_vmnk_coord[1],
                            sfb_cta_layout,
                            cute.group_modes(self.sfb_smem_tensor, 0, 3),
                            cute.group_modes(partitioned_global_sfb, 0, 3),
                        )
                    )
                    partitioned_smem_sfb = cute.filter_zeros(partitioned_smem_sfb)
                    partitioned_global_sfb = cute.filter_zeros(partitioned_global_sfb)

                    global_b_slice = partitioned_global_b[
                        (None, work_tile.tile_n_idx, None, 0)
                    ]
                    global_sfb_slice = partitioned_global_sfb[
                        (None, self._sfb_tile_index(work_tile.tile_n_idx), None, 0)
                    ]

                    if cutlass.const_expr(self.fc2_b_live_rows):
                        # This CTA's live rows of the tile: the instruction N's half from
                        # c * N / 2, as the static box's partition above starts them; the
                        # boxes past them are skipped and their bytes leave the stage's
                        # expectation, which TMA-B registers for both MMA CTAs.
                        live_rows = self._instruction_n(work_tile) // Int32(
                            self.mma_cta_count
                        )
                        live_boxes = (
                            live_rows + Int32(self.fc2_b_row_box - 1)
                        ) // Int32(self.fc2_b_row_box)
                        tile_row = (
                            work_tile.cumulative_data_physical_row
                            + work_tile.tile_n_idx * Int32(self.token_slot_rows)
                            + Int32(self.mma_cta_index) * live_rows
                        )
                        stage_rows = self.cta_tile_n // self.mma_cta_count
                        fc2_expected_tx = Int32(self.ab_pipeline_tx_count_per_producer)
                        if cutlass.const_expr(self.uses_asymmetric_ab_stages):
                            fc2_expected_tx = Int32(self.token_pipeline_tx_count)
                        fc2_expected_tx = fc2_expected_tx - (
                            Int32(self.cta_tile_n)
                            - live_boxes
                            * Int32(self.fc2_b_row_box * self.mma_cta_count)
                        ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
                    empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                    for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                        if cutlass.const_expr(self.fc2_b_live_rows):
                            ab_pipeline.producer_acquire(
                                ab_pipeline_state,
                                empty_status,
                                expected_tx=fc2_expected_tx,
                            )
                        else:
                            ab_pipeline.producer_acquire(
                                ab_pipeline_state, empty_status
                            )
                        tma_barrier = ab_pipeline.producer_get_barrier(
                            ab_pipeline_state
                        )
                        stage_index = ab_pipeline_state.index
                        ab_pipeline_state.advance()
                        if k_tile_idx + 1 < k_tile_count:
                            empty_status = ab_pipeline.producer_try_acquire(
                                ab_pipeline_state
                            )
                        if cutlass.const_expr(self.fc2_b_live_rows):
                            if live_boxes < Int32(stage_rows // self.fc2_b_row_box):
                                stage_base_elements = stage_index * Int32(
                                    stage_rows * self.mma_tile_k
                                )
                                for box in cutlass.range_constexpr(
                                    stage_rows // self.fc2_b_row_box
                                ):
                                    if Int32(box) < live_boxes:
                                        with cute.arch.elect_one():
                                            blackwell_tma_tile2d_load(
                                                fc2_tma_b_rows_atom,
                                                self.b_smem_tensor.iterator
                                                + stage_base_elements
                                                + Int32(
                                                    box
                                                    * self.fc2_b_row_box
                                                    * self.mma_tile_k
                                                ),
                                                tma_barrier,
                                                k_tile_idx * Int32(self.mma_tile_k),
                                                tile_row
                                                + Int32(box * self.fc2_b_row_box),
                                                use_cta_group_2=self.use_2cta_instrs,
                                                mcast_mask=b_multicast_mask,
                                            )
                            else:
                                # Live rows fill the box: one static box load, as below.
                                cute.copy(
                                    fc2_tma_b_atom,
                                    global_b_slice[(None, k_tile_idx)],
                                    partitioned_smem_b[(None, stage_index)],
                                    tma_bar_ptr=tma_barrier,
                                    tma_desc_ptr=b_descriptor_ptr,
                                    mcast_mask=b_multicast_mask,
                                )
                        else:
                            cute.copy(
                                fc2_tma_b_atom,
                                global_b_slice[(None, k_tile_idx)],
                                partitioned_smem_b[(None, stage_index)],
                                tma_bar_ptr=tma_barrier,
                                tma_desc_ptr=b_descriptor_ptr,
                                mcast_mask=b_multicast_mask,
                            )
                        cute.copy(
                            fc2_tma_sfb_atom,
                            global_sfb_slice[(None, k_tile_idx)],
                            partitioned_smem_sfb[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=sfb_descriptor_ptr,
                            mcast_mask=sfb_multicast_mask,
                        )
                iket.range_pop()
                work_tile = sched_consumer.consume_work()

            ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def _run_tma_b_with_shared_expert(
        self,
        *,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom: cute.CopyAtom,
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom: cute.CopyAtom,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom: cute.CopyAtom,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        fc1_done_counter_pointer: cute.Pointer,
        fc2_spin_threshold: Int32,
        fc1_sf_ready_pointer: cute.Pointer,
        expert_num_tokens: cute.Tensor,
        fc1_tile_ready_pointer: Optional[cute.Pointer] = None,
        first_fc1_stamp: Optional[cute.Pointer] = None,
        claim_gate: Optional[cute.Pointer] = None,
        fc1_ready_counts_routes: bool = False,
        fc2_tma_b_rows_atom: Optional[cute.CopyAtom] = None,
        fc1_tma_b_atom: Optional[cute.CopyAtom] = None,
        fc1_tma_b_rows_atom: Optional[cute.CopyAtom] = None,
        fc1_tma_b_pool_atom: Optional[cute.CopyAtom] = None,
        token_pool_plan: Optional[cute.Tensor] = None,
        shared_tma: Optional[Tuple] = None,
        shared_token_rows: int = 0,
    ) -> None:
        """``run_tma_b`` with a shared expert: the default loop's tile loads in helpers, which a shared FC1 tile's join.

        A routed FC1 tile's loads (``_load_routed_fc1_tokens``) and every FC2
        tile's (``_load_fc2_tokens``) are the default loop body's; a shared FC1
        tile passes no gate and loads its tokens from the staged input
        (``_load_shared_fc1_tokens``).
        """
        b_multicast_mask = None
        sfb_multicast_mask = None
        if cutlass.const_expr(self.is_b_mcast or self.use_2cta_instrs):
            b_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                None,
                self.cta_coord_in_cluster,
                None,
                self.use_2cta_instrs,
                "b",
            )
            sfb_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                None,
                self.cta_coord_in_cluster,
                None,
                self.use_2cta_instrs,
                "sfb",
            )

        b_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        thread_mma = self.tiled_mma.get_slice(self.mma_cta_index)
        thread_mma_sfb = self.tiled_mma_sfb.get_slice(self.mma_cta_index)

        routed_fc1_loads = dict(
            ab_pipeline=ab_pipeline,
            kernel_extension=kernel_extension,
            fc1_tma_sfb_tensor=fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom=fc1_tma_sfb_atom,
            fc1_sf_ready_pointer=fc1_sf_ready_pointer,
            expert_num_tokens=expert_num_tokens,
            fc1_tile_ready_pointer=fc1_tile_ready_pointer,
            first_fc1_stamp=first_fc1_stamp,
            claim_gate=claim_gate,
            fc1_ready_counts_routes=fc1_ready_counts_routes,
            fc1_tma_b_atom=fc1_tma_b_atom,
            fc1_tma_b_rows_atom=fc1_tma_b_rows_atom,
            fc1_tma_b_pool_atom=fc1_tma_b_pool_atom,
            token_pool_plan=token_pool_plan,
            thread_mma_sfb=thread_mma_sfb,
            sfb_cta_layout=sfb_cta_layout,
            b_multicast_mask=b_multicast_mask,
            sfb_multicast_mask=sfb_multicast_mask,
        )
        fc2_loads = dict(
            ab_pipeline=ab_pipeline,
            kernel_extension=kernel_extension,
            fc1_done_counter_pointer=fc1_done_counter_pointer,
            fc2_spin_threshold=fc2_spin_threshold,
            claim_gate=claim_gate,
            fc2_tma_b_tensor=fc2_tma_b_tensor,
            fc2_tma_b_atom=fc2_tma_b_atom,
            fc2_tma_sfb_tensor=fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom=fc2_tma_sfb_atom,
            fc2_tma_b_rows_atom=fc2_tma_b_rows_atom,
            thread_mma=thread_mma,
            thread_mma_sfb=thread_mma_sfb,
            b_cta_layout=b_cta_layout,
            sfb_cta_layout=sfb_cta_layout,
            b_multicast_mask=b_multicast_mask,
            sfb_multicast_mask=sfb_multicast_mask,
        )

        fc1_stamped = cutlass.Boolean(False)
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            if is_fc1:
                if cutlass.const_expr(self.shared_expert_marker is not None):
                    if work_tile.expert_idx == Int32(self.shared_expert_marker):
                        ab_pipeline_state = self._load_shared_fc1_tokens(
                            work_tile,
                            ab_pipeline_state,
                            shared_tma,
                            shared_token_rows,
                            ab_pipeline=ab_pipeline,
                            kernel_extension=kernel_extension,
                            fc1_tma_sfb_tensor=fc1_tma_sfb_tensor,
                            fc1_tma_sfb_atom=fc1_tma_sfb_atom,
                            claim_gate=claim_gate,
                            thread_mma_sfb=thread_mma_sfb,
                            sfb_cta_layout=sfb_cta_layout,
                            b_multicast_mask=b_multicast_mask,
                            sfb_multicast_mask=sfb_multicast_mask,
                        )
                    else:
                        ab_pipeline_state, fc1_stamped = self._load_routed_fc1_tokens(
                            work_tile,
                            ab_pipeline_state,
                            fc1_stamped,
                            **routed_fc1_loads,
                        )
                else:
                    ab_pipeline_state, fc1_stamped = self._load_routed_fc1_tokens(
                        work_tile, ab_pipeline_state, fc1_stamped, **routed_fc1_loads
                    )
            else:
                ab_pipeline_state = self._load_fc2_tokens(
                    work_tile, ab_pipeline_state, **fc2_loads
                )
            iket.range_pop()
            work_tile = sched_consumer.consume_work()

        ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def _load_routed_fc1_tokens(
        self,
        work_tile,
        ab_pipeline_state: pipeline.PipelineState,
        fc1_stamped,
        *,
        ab_pipeline: pipeline.PipelineAsync,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom: cute.CopyAtom,
        fc1_sf_ready_pointer: cute.Pointer,
        expert_num_tokens: cute.Tensor,
        fc1_tile_ready_pointer: Optional[cute.Pointer],
        first_fc1_stamp: Optional[cute.Pointer],
        claim_gate: Optional[cute.Pointer],
        fc1_ready_counts_routes: bool,
        fc1_tma_b_atom: Optional[cute.CopyAtom],
        fc1_tma_b_rows_atom: Optional[cute.CopyAtom],
        fc1_tma_b_pool_atom: Optional[cute.CopyAtom],
        token_pool_plan: Optional[cute.Tensor],
        thread_mma_sfb,
        sfb_cta_layout,
        b_multicast_mask,
        sfb_multicast_mask,
    ):
        """One routed FC1 tile: its readiness gate, then its scale factors (and pooled tokens) per K tile.

        Returns the advanced pipeline state and whether the first FC1 gate pass is stamped.
        """
        iket.range_push("tma_token_fc1")
        iket.range_push("tma_b.wait_expert_sf_ready")
        expert_idx = work_tile.expert_idx
        if cutlass.const_expr(fc1_tile_ready_pointer is not None):
            spin_wait(
                fc1_tile_ready_pointer
                + work_tile.cumulative_token_block_count
                + work_tile.tile_n_idx,
                lambda value: value == work_tile.valid_tokens_in_cta_tile,
                scope="gpu",
            )
        else:
            if cutlass.const_expr(fc1_ready_counts_routes):
                fc1_ready_target = expert_num_tokens[expert_idx]
            else:
                fc1_ready_target = expert_pool_rows(
                    expert_num_tokens[expert_idx],
                    Int32(self.sf_padding_block),
                    self.plan_tile_n,
                    self.token_slot_rows,
                ) // Int32(self.sf_padding_block)
            spin_wait(
                fc1_sf_ready_pointer + expert_idx,
                lambda value: value == fc1_ready_target,
                scope="gpu",
            )
        # Orders the gate's generic acquire before the scale-factor TMA loads of the rows it counts.
        sentinel_ptx.fence_proxy_async_global()
        if cutlass.const_expr(claim_gate is not None):
            if cute.arch.lane_idx() == Int32(0):
                cute.arch.mbarrier_arrive(claim_gate)
        if cutlass.const_expr(first_fc1_stamp is not None):
            if not fc1_stamped:
                cute.make_tensor(first_fc1_stamp, cute.make_layout(1))[0] = (
                    sentinel_ptx.globaltimer_ns()
                )
                fc1_stamped = cutlass.Boolean(True)
        iket.range_pop()
        k_tile_count = self.fc1_k_tile_count
        real_sfb, sfb_descriptor_ptr = kernel_extension.get_gmem_tensor(
            "sfb", fc1_tma_sfb_tensor, work_tile
        )

        global_sfb = cute.local_tile(
            real_sfb,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        partitioned_global_sfb = thread_mma_sfb.partition_B(global_sfb)

        partitioned_smem_sfb, partitioned_global_sfb = cpasync.tma_partition(
            fc1_tma_sfb_atom,
            self.sfb_vmnk_coord[1],
            sfb_cta_layout,
            cute.group_modes(self.sfb_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_sfb, 0, 3),
        )
        partitioned_smem_sfb = cute.filter_zeros(partitioned_smem_sfb)
        partitioned_global_sfb = cute.filter_zeros(partitioned_global_sfb)

        global_sfb_slice = partitioned_global_sfb[
            (None, self._sfb_tile_index(work_tile.tile_n_idx), None, 0)
        ]

        fc1_expected_tx = Int32(self.ab_pipeline_tx_count_per_producer)
        if cutlass.const_expr(
            self.fc1_tx_split_by_operand or self.uses_asymmetric_ab_stages
        ):
            fc1_expected_tx = Int32(self.token_pipeline_tx_count)
        fc1_stage_rows = self.cta_tile_n // self.mma_cta_count
        fc1_stage_boxes = fc1_stage_rows // self.fc2_b_row_box
        if cutlass.const_expr(self.fc1_token_pool):
            # This CTA's rows of the tile, consecutive in the token pool: the
            # instruction N's half from c * N / 2, as the MMA splits them. The
            # static box covers the CTA's whole half of the tile N, and the rows
            # it loads past the live ones feed no MMA column. A tile that leaves
            # part of that box unread loads only its live rows, in 16-row boxes,
            # and registers only their bytes, as FC2 does past N128.
            fc1_live_boxes = Int32(fc1_stage_boxes)
            fc1_cta_rows = Int32(fc1_stage_rows)
            if cutlass.const_expr(self.runtime_instruction_n):
                fc1_cta_rows = self._instruction_n(work_tile) // Int32(
                    self.mma_cta_count
                )
            if cutlass.const_expr(self.fc1_token_pool_bounded):
                # Under a capacity the tile's rows start at its first route in
                # processing order, and a tile whose routes end past the
                # launch's capacity gathers instead (see the gather case below).
                tile_plan = token_pool_plan[Int32(1) + expert_idx]
                tile_route = (tile_plan >> Int32(8)) + work_tile.tile_n_idx * (
                    tile_plan & Int32(255)
                )
                in_pool = (
                    tile_route + work_tile.valid_tokens_in_cta_tile
                    <= token_pool_plan[0]
                )
                gathers = (
                    tile_route + work_tile.valid_tokens_in_cta_tile > token_pool_plan[0]
                )
                fc1_tile_row = tile_route + Int32(self.mma_cta_index) * fc1_cta_rows
                gathered_rows = round_up(
                    work_tile.valid_tokens_in_cta_tile, Int32(self.gather4_width)
                )
                gather_expected_tx = fc1_expected_tx - (
                    Int32(self.cta_tile_n) - gathered_rows
                ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
            else:
                fc1_tile_row = (
                    work_tile.cumulative_data_physical_row
                    + work_tile.tile_n_idx * Int32(self.token_slot_rows)
                    + Int32(self.mma_cta_index) * fc1_cta_rows
                )
            if cutlass.const_expr(self.fc1_b_live_rows):
                fc1_live_boxes = (
                    fc1_cta_rows + Int32(self.fc2_b_row_box - 1)
                ) // Int32(self.fc2_b_row_box)
                fc1_expected_tx = fc1_expected_tx - (
                    Int32(self.cta_tile_n)
                    - fc1_live_boxes * Int32(self.fc2_b_row_box * self.mma_cta_count)
                ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
            if cutlass.const_expr(self.fc1_token_pool_bounded):
                fc1_expected_tx = cutlass.select_(
                    in_pool, fc1_expected_tx, gather_expected_tx
                )
        else:
            # Gather-B issues no gather4 group whose rows are all padding (rows past the
            # tile's valid tokens). Each skipped row is one B row less in every MMA pair of
            # the multicast group, so the static expectation drops by that many rows. The
            # gathered rows are the tile's first round_up(valid, 4) under either row split
            # between the MMA CTAs, as the runtime instruction N is at least that many.
            gathered_rows = round_up(
                work_tile.valid_tokens_in_cta_tile, Int32(self.gather4_width)
            )
            fc1_expected_tx = fc1_expected_tx - (
                Int32(self.cta_tile_n) - gathered_rows
            ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
        for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
            ab_pipeline.producer_acquire(
                ab_pipeline_state, empty_status, expected_tx=fc1_expected_tx
            )
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            if cutlass.const_expr(not self.fc1_token_pool):
                # TMA-B has waited for this AB stage to become empty and
                # contributed one of the two static producer transaction
                # expectations. Rendezvous with the Gather-B warps before any
                # of them issues Gather4 against this barrier.
                barrier_sync_unaligned(
                    self.fc1_gather_sync_barrier_id, 32 * (1 + self.num_gather_b_warps)
                )
            if cutlass.const_expr(self.fc1_token_pool_bounded):
                if gathers:
                    barrier_sync_unaligned(
                        self.fc1_gather_sync_barrier_id,
                        32 * (1 + self.num_gather_b_warps),
                    )
            ab_pipeline_state.advance()
            if k_tile_idx + 1 < k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            if cutlass.const_expr(self.fc1_token_pool_bounded):
                if in_pool:
                    self._load_fc1_pool_stage(
                        fc1_tma_b_pool_atom,
                        fc1_tma_b_rows_atom,
                        stage_index,
                        tma_barrier,
                        k_tile_idx,
                        fc1_tile_row,
                        fc1_live_boxes,
                        b_multicast_mask,
                    )
            elif cutlass.const_expr(self.fc1_token_pool):
                self._load_fc1_pool_stage(
                    fc1_tma_b_atom,
                    fc1_tma_b_rows_atom,
                    stage_index,
                    tma_barrier,
                    k_tile_idx,
                    fc1_tile_row,
                    fc1_live_boxes,
                    b_multicast_mask,
                )
            cute.copy(
                fc1_tma_sfb_atom,
                global_sfb_slice[(None, k_tile_idx)],
                partitioned_smem_sfb[(None, stage_index)],
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=sfb_descriptor_ptr,
                mcast_mask=sfb_multicast_mask,
            )
        return ab_pipeline_state, fc1_stamped

    @cute.jit
    def _load_fc2_tokens(
        self,
        work_tile,
        ab_pipeline_state: pipeline.PipelineState,
        *,
        ab_pipeline: pipeline.PipelineAsync,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        fc1_done_counter_pointer: cute.Pointer,
        fc2_spin_threshold: Int32,
        claim_gate: Optional[cute.Pointer],
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom: cute.CopyAtom,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom: cute.CopyAtom,
        fc2_tma_b_rows_atom: Optional[cute.CopyAtom],
        thread_mma,
        thread_mma_sfb,
        b_cta_layout,
        sfb_cta_layout,
        b_multicast_mask,
        sfb_multicast_mask,
    ) -> pipeline.PipelineState:
        """One FC2 tile: its ``fc1_done`` gate, then its tokens and scale factors per K tile; returns the advanced state."""
        iket.range_push("tma_token_fc2")
        counter_slot = work_tile.cumulative_token_block_count + work_tile.tile_n_idx
        counter_pointer = fc1_done_counter_pointer + counter_slot
        iket.range_push("tma_token_fc2_wait")
        spin_wait(
            counter_pointer,
            lambda value: value >= fc2_spin_threshold,
            sleep_cycles=500,
            peek_status=work_tile.peek_ready,
        )
        # Orders the fc1_done acquire (this wait's or the scheduler peek's) before the TMA loads of FC1's output.
        sentinel_ptx.fence_proxy_async_global()
        if cutlass.const_expr(claim_gate is not None):
            if cute.arch.lane_idx() == Int32(0):
                cute.arch.mbarrier_arrive(claim_gate)
        iket.range_pop()

        k_tile_count = self.fc2_k_tile_count
        real_b, b_descriptor_ptr = kernel_extension.get_gmem_tensor(
            "b", fc2_tma_b_tensor, work_tile
        )
        real_sfb, sfb_descriptor_ptr = kernel_extension.get_gmem_tensor(
            "sfb", fc2_tma_sfb_tensor, work_tile
        )
        if cutlass.const_expr(self.token_slot_rows != self.cta_tile_n):
            # Token tiles sit one slot apart in the FC1 output pool, while the
            # B tiling below steps by the tile itself; shift each tile by the
            # slot padding of the tiles before it.
            real_b = cute.domain_offset(
                (
                    work_tile.tile_n_idx
                    * Int32(self.token_slot_rows - self.cta_tile_n),
                    0,
                    0,
                ),
                real_b,
            )
        if cutlass.const_expr(self.runtime_instruction_n):
            # The B partition below gives MMA CTA c the tile's rows from
            # c * tile N / 2; the runtime-N split starts them at c * N / 2.
            # The box keeps its static height: the rows it loads past the
            # first N / 2 of each CTA feed no MMA column.
            real_b = cute.domain_offset(
                (
                    Int32(self.mma_cta_index)
                    * (
                        (self._instruction_n(work_tile) - Int32(self.cta_tile_n))
                        // Int32(self.mma_cta_count)
                    ),
                    0,
                    0,
                ),
                real_b,
            )

        global_b = cute.local_tile(
            real_b, cute.slice_(self.mma_tiler_mnk, (0, None, None)), (None, None, None)
        )
        global_sfb = cute.local_tile(
            real_sfb,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        partitioned_global_b = thread_mma.partition_B(global_b)
        partitioned_global_sfb = thread_mma_sfb.partition_B(global_sfb)

        partitioned_smem_b, partitioned_global_b = cpasync.tma_partition(
            fc2_tma_b_atom,
            self.main_vmnk_coord[1],
            b_cta_layout,
            cute.group_modes(self.b_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_b, 0, 3),
        )
        partitioned_smem_sfb, partitioned_global_sfb = cpasync.tma_partition(
            fc2_tma_sfb_atom,
            self.sfb_vmnk_coord[1],
            sfb_cta_layout,
            cute.group_modes(self.sfb_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_sfb, 0, 3),
        )
        partitioned_smem_sfb = cute.filter_zeros(partitioned_smem_sfb)
        partitioned_global_sfb = cute.filter_zeros(partitioned_global_sfb)

        global_b_slice = partitioned_global_b[(None, work_tile.tile_n_idx, None, 0)]
        global_sfb_slice = partitioned_global_sfb[
            (None, self._sfb_tile_index(work_tile.tile_n_idx), None, 0)
        ]

        if cutlass.const_expr(self.fc2_b_live_rows):
            # This CTA's live rows of the tile: the instruction N's half from
            # c * N / 2, as the static box's partition above starts them; the
            # boxes past them are skipped and their bytes leave the stage's
            # expectation, which TMA-B registers for both MMA CTAs.
            live_rows = self._instruction_n(work_tile) // Int32(self.mma_cta_count)
            live_boxes = (live_rows + Int32(self.fc2_b_row_box - 1)) // Int32(
                self.fc2_b_row_box
            )
            tile_row = (
                work_tile.cumulative_data_physical_row
                + work_tile.tile_n_idx * Int32(self.token_slot_rows)
                + Int32(self.mma_cta_index) * live_rows
            )
            stage_rows = self.cta_tile_n // self.mma_cta_count
            fc2_expected_tx = Int32(self.ab_pipeline_tx_count_per_producer)
            if cutlass.const_expr(self.uses_asymmetric_ab_stages):
                fc2_expected_tx = Int32(self.token_pipeline_tx_count)
            fc2_expected_tx = fc2_expected_tx - (
                Int32(self.cta_tile_n)
                - live_boxes * Int32(self.fc2_b_row_box * self.mma_cta_count)
            ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
        for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
            if cutlass.const_expr(self.fc2_b_live_rows):
                ab_pipeline.producer_acquire(
                    ab_pipeline_state, empty_status, expected_tx=fc2_expected_tx
                )
            else:
                ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            ab_pipeline_state.advance()
            if k_tile_idx + 1 < k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            if cutlass.const_expr(self.fc2_b_live_rows):
                if live_boxes < Int32(stage_rows // self.fc2_b_row_box):
                    stage_base_elements = stage_index * Int32(
                        stage_rows * self.mma_tile_k
                    )
                    for box in cutlass.range_constexpr(
                        stage_rows // self.fc2_b_row_box
                    ):
                        if Int32(box) < live_boxes:
                            with cute.arch.elect_one():
                                blackwell_tma_tile2d_load(
                                    fc2_tma_b_rows_atom,
                                    self.b_smem_tensor.iterator
                                    + stage_base_elements
                                    + Int32(box * self.fc2_b_row_box * self.mma_tile_k),
                                    tma_barrier,
                                    k_tile_idx * Int32(self.mma_tile_k),
                                    tile_row + Int32(box * self.fc2_b_row_box),
                                    use_cta_group_2=self.use_2cta_instrs,
                                    mcast_mask=b_multicast_mask,
                                )
                else:
                    # Live rows fill the box: one static box load, as below.
                    cute.copy(
                        fc2_tma_b_atom,
                        global_b_slice[(None, k_tile_idx)],
                        partitioned_smem_b[(None, stage_index)],
                        tma_bar_ptr=tma_barrier,
                        tma_desc_ptr=b_descriptor_ptr,
                        mcast_mask=b_multicast_mask,
                    )
            else:
                cute.copy(
                    fc2_tma_b_atom,
                    global_b_slice[(None, k_tile_idx)],
                    partitioned_smem_b[(None, stage_index)],
                    tma_bar_ptr=tma_barrier,
                    tma_desc_ptr=b_descriptor_ptr,
                    mcast_mask=b_multicast_mask,
                )
            cute.copy(
                fc2_tma_sfb_atom,
                global_sfb_slice[(None, k_tile_idx)],
                partitioned_smem_sfb[(None, stage_index)],
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=sfb_descriptor_ptr,
                mcast_mask=sfb_multicast_mask,
            )
        return ab_pipeline_state

    @cute.jit
    def _load_shared_fc1_tokens(
        self,
        work_tile,
        ab_pipeline_state: pipeline.PipelineState,
        shared_tma: Tuple,
        shared_token_rows: int,
        *,
        ab_pipeline: pipeline.PipelineAsync,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom: cute.CopyAtom,
        claim_gate: Optional[cute.Pointer],
        thread_mma_sfb,
        sfb_cta_layout,
        b_multicast_mask,
        sfb_multicast_mask,
    ) -> pipeline.PipelineState:
        """One shared FC1 tile: its tokens from the staged input and its scale factors per K tile.

        The staged input is complete when this kernel starts, so the tile passes
        no gate. Token tile t holds tokens [t * ``shared_token_rows``, ...) of
        the input, split between the MMA CTAs as the instruction N splits them;
        the rows a box loads past the tile's tokens feed no stored column. Its
        scale factors come from the pool rows the staging fills, as a routed
        tile's from its expert's. Returns the advanced pipeline state.
        """
        x_atom, x_rows_atom = shared_tma[8], shared_tma[9]
        iket.range_push("tma_token_shared_fc1")
        if cutlass.const_expr(claim_gate is not None):
            if cute.arch.lane_idx() == Int32(0):
                cute.arch.mbarrier_arrive(claim_gate)
        real_sfb, sfb_descriptor_ptr = kernel_extension.get_gmem_tensor(
            "sfb", fc1_tma_sfb_tensor, work_tile
        )
        global_sfb = cute.local_tile(
            real_sfb,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        partitioned_global_sfb = thread_mma_sfb.partition_B(global_sfb)
        partitioned_smem_sfb, partitioned_global_sfb = cpasync.tma_partition(
            fc1_tma_sfb_atom,
            self.sfb_vmnk_coord[1],
            sfb_cta_layout,
            cute.group_modes(self.sfb_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_sfb, 0, 3),
        )
        partitioned_smem_sfb = cute.filter_zeros(partitioned_smem_sfb)
        partitioned_global_sfb = cute.filter_zeros(partitioned_global_sfb)
        global_sfb_slice = partitioned_global_sfb[
            (None, self._sfb_tile_index(work_tile.tile_n_idx), None, 0)
        ]

        expected_tx = Int32(self.ab_pipeline_tx_count_per_producer)
        if cutlass.const_expr(
            self.fc1_tx_split_by_operand or self.uses_asymmetric_ab_stages
        ):
            expected_tx = Int32(self.token_pipeline_tx_count)
        stage_boxes = self.cta_tile_n // self.mma_cta_count // self.fc2_b_row_box
        cta_rows = self._instruction_n(work_tile) // Int32(self.mma_cta_count)
        live_boxes = Int32(stage_boxes)
        if cutlass.const_expr(self.shared_b_live_rows):
            live_boxes = (cta_rows + Int32(self.fc2_b_row_box - 1)) // Int32(
                self.fc2_b_row_box
            )
            expected_tx = expected_tx - (
                Int32(self.cta_tile_n)
                - live_boxes * Int32(self.fc2_b_row_box * self.mma_cta_count)
            ) * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)
        tile_row = (
            work_tile.tile_n_idx * Int32(shared_token_rows)
            + Int32(self.mma_cta_index) * cta_rows
        )

        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
        for k_tile_idx in cutlass.range(self.fc1_k_tile_count, unroll=1):
            ab_pipeline.producer_acquire(
                ab_pipeline_state, empty_status, expected_tx=expected_tx
            )
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            ab_pipeline_state.advance()
            if k_tile_idx + 1 < self.fc1_k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            self._load_token_rows_stage(
                x_atom,
                x_rows_atom,
                stage_index,
                tma_barrier,
                k_tile_idx,
                tile_row,
                live_boxes,
                b_multicast_mask,
                live_rows=self.shared_b_live_rows,
            )
            cute.copy(
                fc1_tma_sfb_atom,
                global_sfb_slice[(None, k_tile_idx)],
                partitioned_smem_sfb[(None, stage_index)],
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=sfb_descriptor_ptr,
                mcast_mask=sfb_multicast_mask,
            )
        iket.range_pop()
        return ab_pipeline_state

    @cute.jit
    def _load_token_rows_stage(
        self,
        box_atom: cute.CopyAtom,
        rows_atom: Optional[cute.CopyAtom],
        stage_index: Int32,
        tma_barrier: cute.Pointer,
        k_tile_idx: Int32,
        tile_row: Int32,
        live_boxes: Int32,
        b_multicast_mask,
        live_rows: bool,
    ) -> None:
        """One AB stage of consecutive token rows from ``tile_row``: the CTA's whole half box, or its live 16-row boxes."""
        stage_rows = self.cta_tile_n // self.mma_cta_count
        stage_boxes = stage_rows // self.fc2_b_row_box
        stage_base_elements = stage_index * Int32(stage_rows * self.mma_tile_k)
        if live_boxes < Int32(stage_boxes):
            if cutlass.const_expr(live_rows):
                for box in cutlass.range_constexpr(stage_boxes):
                    if Int32(box) < live_boxes:
                        with cute.arch.elect_one():
                            blackwell_tma_tile2d_load(
                                rows_atom,
                                self.b_smem_tensor.iterator
                                + stage_base_elements
                                + Int32(box * self.fc2_b_row_box * self.mma_tile_k),
                                tma_barrier,
                                k_tile_idx * Int32(self.mma_tile_k),
                                tile_row + Int32(box * self.fc2_b_row_box),
                                use_cta_group_2=self.use_2cta_instrs,
                                mcast_mask=b_multicast_mask,
                            )
        else:
            with cute.arch.elect_one():
                blackwell_tma_tile2d_load(
                    box_atom,
                    self.b_smem_tensor.iterator + stage_base_elements,
                    tma_barrier,
                    k_tile_idx * Int32(self.mma_tile_k),
                    tile_row,
                    use_cta_group_2=self.use_2cta_instrs,
                    mcast_mask=b_multicast_mask,
                )

    @cute.jit
    def _load_fc1_pool_stage(
        self,
        fc1_tma_b_atom: cute.CopyAtom,
        fc1_tma_b_rows_atom: Optional[cute.CopyAtom],
        stage_index: Int32,
        tma_barrier: cute.Pointer,
        k_tile_idx: Int32,
        fc1_tile_row: Int32,
        fc1_live_boxes: Int32,
        b_multicast_mask,
    ) -> None:
        """One AB stage of FC1's token tile from the token pool: the CTA's whole half box, or its live 16-row boxes."""
        fc1_stage_rows = self.cta_tile_n // self.mma_cta_count
        fc1_stage_boxes = fc1_stage_rows // self.fc2_b_row_box
        stage_base_elements = stage_index * Int32(fc1_stage_rows * self.mma_tile_k)
        if fc1_live_boxes < Int32(fc1_stage_boxes):
            if cutlass.const_expr(self.fc1_b_live_rows):
                for box in cutlass.range_constexpr(fc1_stage_boxes):
                    if Int32(box) < fc1_live_boxes:
                        with cute.arch.elect_one():
                            blackwell_tma_tile2d_load(
                                fc1_tma_b_rows_atom,
                                self.b_smem_tensor.iterator
                                + stage_base_elements
                                + Int32(box * self.fc2_b_row_box * self.mma_tile_k),
                                tma_barrier,
                                k_tile_idx * Int32(self.mma_tile_k),
                                fc1_tile_row + Int32(box * self.fc2_b_row_box),
                                use_cta_group_2=self.use_2cta_instrs,
                                mcast_mask=b_multicast_mask,
                            )
        else:
            with cute.arch.elect_one():
                blackwell_tma_tile2d_load(
                    fc1_tma_b_atom,
                    self.b_smem_tensor.iterator + stage_base_elements,
                    tma_barrier,
                    k_tile_idx * Int32(self.mma_tile_k),
                    fc1_tile_row,
                    use_cta_group_2=self.use_2cta_instrs,
                    mcast_mask=b_multicast_mask,
                )

    @cute.jit
    def run_tma_gather_b1(
        self,
        *,
        fc1_tma_b_atom: cute.CopyAtom,
        gather_index: cute.Tensor,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        gather_warp_idx: Int32,
        gather_row_offset: Int32,
        token_pool_plan: Optional[cute.Tensor] = None,
        entry_wait: Optional[Callable[[], None]] = None,
    ) -> None:
        """Gather dense-order FC1 activations directly into B SMEM.

        gather_index holds rows of the round's ping-pong buffer; gather_row_offset
        moves them to that buffer's rows of the descriptor, which spans both.

        Under a token pool with a capacity, TMA-B loads the token tiles the
        launch's ``token_pool_plan`` places in the pool (see ``run_tma_b``), and
        this warp keeps its pipeline state aligned on them as on FC2 tiles.

        Every CTA of the B multicast group issues its share and multicasts it to
        the whole group, so the group mask is the ordinary ``"b"`` mask each CTA
        already computes for itself -- no leader election. Ownership is
        interleaved by gather4 group so that a short row window spreads over all
        issuing units rather than landing entirely on group zero.

        Gather-B contributes transaction completions only; it is not an AB
        producer. FC2 is fully produced by TMA-A and TMA-B, so on those tiles this
        warp only keeps its register-held pipeline state aligned, as it does on a
        shared expert's FC1 tiles, whose tokens TMA-B loads.

        ``entry_wait``, when given, runs once before the first tile this warp
        gathers: the waits for the index lists and the payload that a caller
        without a shared expert's entry head passes before the first tile.
        """
        row_ids = cute.make_rmem_tensor(
            cute.make_layout((self.gather_b_rows_per_unit,)), cutlass.Int32
        )
        row_id_runs = cute.zipped_divide(row_ids, (self.gather_index_load_rows,))
        index_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Int32,
            num_bits_per_copy=self.gather_index_load_rows * cutlass.Int32.width,
        )
        index_load_bytes = self.gather_index_load_rows * cutlass.Int32.width // 8
        gather_b_multicast_mask = None
        if cutlass.const_expr(self.is_b_mcast or self.use_2cta_instrs):
            gather_b_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                None,
                self.cta_coord_in_cluster,
                None,
                self.use_2cta_instrs,
                "b",
            )
        issuing_unit = (
            self.main_vmnk_coord[1] * Int32(self.num_gather_b_warps) + gather_warp_idx
        )
        if cutlass.const_expr(entry_wait is not None):
            entry_waited = Int32(0)

        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            if cutlass.const_expr(self.shared_expert_marker is not None):
                shared_fc1 = is_fc1 & (
                    work_tile.expert_idx == Int32(self.shared_expert_marker)
                )
                is_fc1 = is_fc1 & (
                    work_tile.expert_idx != Int32(self.shared_expert_marker)
                )
            if cutlass.const_expr(
                self.fc1_token_pool_bounded and self.shared_expert_marker is not None
            ):
                # A shared tile has no entry in the plan; any expert's stands in.
                tile_plan = token_pool_plan[
                    Int32(1)
                    + cutlass.min(
                        work_tile.expert_idx, Int32(self.shared_expert_marker - 1)
                    )
                ]
                tile_route = (tile_plan >> Int32(8)) + work_tile.tile_n_idx * (
                    tile_plan & Int32(255)
                )
                pooled = is_fc1 & (
                    tile_route + work_tile.valid_tokens_in_cta_tile
                    <= token_pool_plan[0]
                )
                pooled_k_tiles = cutlass.select_(
                    pooled, Int32(self.fc1_k_tile_count), Int32(self.fc2_k_tile_count)
                )
                is_fc1 = is_fc1 & (
                    tile_route + work_tile.valid_tokens_in_cta_tile > token_pool_plan[0]
                )
            elif cutlass.const_expr(self.fc1_token_pool_bounded):
                tile_plan = token_pool_plan[Int32(1) + work_tile.expert_idx]
                tile_route = (tile_plan >> Int32(8)) + work_tile.tile_n_idx * (
                    tile_plan & Int32(255)
                )
                pooled = is_fc1 & (
                    tile_route + work_tile.valid_tokens_in_cta_tile
                    <= token_pool_plan[0]
                )
                pooled_k_tiles = cutlass.select_(
                    pooled, Int32(self.fc1_k_tile_count), Int32(self.fc2_k_tile_count)
                )
                is_fc1 = is_fc1 & (
                    tile_route + work_tile.valid_tokens_in_cta_tile > token_pool_plan[0]
                )
            if is_fc1:
                iket.range_push("tma_gather_b1_fc1")
                if cutlass.const_expr(entry_wait is not None):
                    if entry_waited == Int32(0):
                        entry_wait()
                        entry_waited = Int32(1)
                tile_pool_base = (
                    work_tile.cumulative_data_physical_row
                    + work_tile.tile_n_idx * Int32(self.token_slot_rows)
                )
                # Match the MMA's and FC2 TMA-B's row split between the MMA CTAs: CTA c
                # holds rows [c * N / 2, (c + 1) * N / 2) of an instruction N wide.
                rows_per_mma_cta = Int32(self.cta_tile_n // self.mma_cta_count)
                if cutlass.const_expr(self.runtime_instruction_n):
                    rows_per_mma_cta = self._instruction_n(work_tile) // Int32(
                        self.mma_cta_count
                    )
                cta_row_base = self.mma_cta_index * rows_per_mma_cta
                index_runs = cute.zipped_divide(
                    cute.domain_offset((tile_pool_base + cta_row_base,), gather_index),
                    (self.gather_index_load_rows,),
                )
                for local_run in cutlass.range_constexpr(self.gather_b_runs_per_unit):
                    run_index = (
                        Int32(local_run * self.gather_issuing_units) + issuing_unit
                    )
                    cute.copy(
                        index_load_atom,
                        mark_alignment(index_runs[(None, run_index)], index_load_bytes),
                        row_id_runs[(None, local_run)],
                    )
                for row in cutlass.range_constexpr(self.gather_b_rows_per_unit):
                    row_ids[row] = row_ids[row] + gather_row_offset

                # Padding rows feed MMA columns that are never stored. Their indices are not
                # written by the router, so gathering them would make every tile fetch the same
                # dense row and serialize on its L2 lines; whole-padding gather4 groups are skipped.
                gathered_rows = round_up(
                    work_tile.valid_tokens_in_cta_tile, Int32(self.gather4_width)
                )
                for k_tile_idx in cutlass.range(self.fc1_k_tile_count, unroll=1):
                    # TMA-B has already acquired this stage and contributed its
                    # stage-specific transaction expectation. The named barrier orders that
                    # arrive-and-expect-tx before Gather4.
                    barrier_sync_unaligned(
                        self.fc1_gather_sync_barrier_id,
                        32 * (1 + self.num_gather_b_warps),
                    )
                    if k_tile_idx == Int32(0):
                        # TMA-B acquired this expert's verified-rows count before the
                        # first barrier; the landing bytes were written and checked
                        # through the generic proxy, so fence before Gather4 reads them.
                        sentinel_ptx.fence_proxy_async_global()
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index
                    col_k = k_tile_idx * Int32(self.mma_tile_k)
                    stage_base_elements = stage_index * Int32(
                        self.gather_b_rows_per_cta * self.mma_tile_k
                    )

                    for local_run in cutlass.range_constexpr(
                        self.gather_b_runs_per_unit
                    ):
                        run_index = (
                            Int32(local_run * self.gather_issuing_units) + issuing_unit
                        )
                        # A run is issued only inside this CTA's rows; its whole-padding
                        # gather4 groups are skipped below.
                        issue_rows = True
                        if cutlass.const_expr(self.runtime_instruction_n):
                            issue_rows = (
                                run_index * Int32(self.gather_index_load_rows)
                                < rows_per_mma_cta
                            )
                        elif cutlass.const_expr(
                            self.gather_b_uneven_runs
                            and local_run == self.gather_b_runs_per_unit - 1
                        ):
                            issue_rows = run_index < Int32(self.gather_b_runs_per_cta)
                        if issue_rows:
                            run_base_elements = stage_base_elements + run_index * Int32(
                                self.gather_index_load_rows * self.mma_tile_k
                            )
                            for group_in_run in cutlass.range_constexpr(
                                self.gather_b_groups_per_run
                            ):
                                destination_offset = run_base_elements + Int32(
                                    group_in_run * self.gather4_width * self.mma_tile_k
                                )
                                row_start = (
                                    local_run * self.gather_index_load_rows
                                    + group_in_run * self.gather4_width
                                )
                                issue_group = (
                                    cta_row_base
                                    + run_index * Int32(self.gather_index_load_rows)
                                    + Int32(group_in_run * self.gather4_width)
                                    < gathered_rows
                                )
                                if issue_group:
                                    with cute.arch.elect_one():
                                        blackwell_tma_gather4_load(
                                            fc1_tma_b_atom,
                                            self.b_smem_tensor.iterator
                                            + destination_offset,
                                            tma_barrier,
                                            col_k,
                                            row_ids[row_start],
                                            row_ids[row_start + 1],
                                            row_ids[row_start + 2],
                                            row_ids[row_start + 3],
                                            use_cta_group_2=self.use_2cta_instrs,
                                            mcast_mask=gather_b_multicast_mask,
                                        )
                    ab_pipeline_state.advance()
                iket.range_pop()
            else:
                iket.range_push("tma_gather_b1_fc2_state_advance")
                skipped_k_tiles = Int32(self.fc2_k_tile_count)
                if cutlass.const_expr(self.fc1_token_pool_bounded):
                    skipped_k_tiles = pooled_k_tiles
                if cutlass.const_expr(self.shared_expert_marker is not None):
                    skipped_k_tiles = cutlass.select_(
                        shared_fc1, Int32(self.fc1_k_tile_count), skipped_k_tiles
                    )
                advanced_index = ab_pipeline_state.index + skipped_k_tiles
                stage_count = Int32(ab_pipeline_state.stages)
                wrap_count = advanced_index // stage_count
                ab_pipeline_state._index = advanced_index % stage_count
                ab_pipeline_state._phase = ab_pipeline_state.phase ^ (
                    wrap_count & Int32(1)
                )
                ab_pipeline_state._count += skipped_k_tiles
                iket.range_pop()

            work_tile = sched_consumer.consume_work()

    @cute.jit
    def _issue_mma_window(
        self,
        *,
        quant_kind,
        acc_tensor,
        a_window_frag,
        b_window_frag,
        sfa_window_tensor,
        sfb_window_tensor,
        valid_tokens_in_tile,
        mma_instruction_mnk,
        window_instruction_offset,
        window_instruction_count,
        first_instruction_accumulate,
    ):
        tiled_mma = self.make_tiled_mma()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, first_instruction_accumulate)
        cute.gemm(
            tiled_mma,
            acc_tensor,
            [a_window_frag, sfa_window_tensor],
            [b_window_frag, sfb_window_tensor],
            acc_tensor,
        )

    @cute.jit
    def run_mma(
        self,
        *,
        tmem_allocator,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        sched_consumer: SchedulerConsumer,
    ) -> None:
        """Issue the native Blackwell MMAs of every work tile, at its runtime instruction N where enabled."""
        tiled_mma = self.make_tiled_mma()
        fragment_a = tiled_mma.make_fragment_A(self.a_smem_tensor)
        fragment_b = tiled_mma.make_fragment_B(self.b_smem_tensor)
        if cutlass.const_expr(self.runtime_instruction_n):
            a_descriptor_base = self._smem_descriptor_base(self.a_smem_tensor)
            b_descriptor_base = self._smem_descriptor_base(self.b_smem_tensor)

        barrier_sync_unaligned(
            self.tmem_allocation_barrier_id,
            32 * (1 + self.num_accumulator_consumer_warps_per_cta),
        )
        accumulator_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
        accumulator_layout = cute.make_layout(
            self.accumulator_shape, stride=self.accumulator_stride
        )
        accumulator_base = cute.make_tensor(accumulator_pointer, accumulator_layout)

        sfa_stage_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfa(
                tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, 1
            ),
            (None, None, None, 0),
        )
        sfb_stage_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, 1
            ),
            (None, None, None, 0),
        )
        sfa_pointer = cute.recast_ptr(
            accumulator_pointer + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
        )
        sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, sfa_stage_smem_layout
        )
        tmem_sfa = cute.make_tensor(sfa_pointer, sfa_layout)
        sfb_pointer = cute.recast_ptr(
            accumulator_pointer
            + self.num_accumulator_tmem_cols
            + self.num_sfa_tmem_cols,
            dtype=self.sf_dtype,
        )
        sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, sfb_stage_smem_layout
        )
        tmem_sfb = cute.make_tensor(sfb_pointer, sfb_layout)
        tiled_copy_sfa, partitioned_smem_sfa, partitioned_tmem_sfa = (
            self._s2t_copy_and_partition(self.sfa_smem_tensor, tmem_sfa)
        )
        tiled_copy_sfb, partitioned_smem_sfb, partitioned_tmem_sfb = (
            self._s2t_copy_and_partition(self.sfb_smem_tensor, tmem_sfb)
        )

        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_accumulator_pipeline_stages
        )
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            k_tile_count = Int32(0)
            if is_fc1:
                k_tile_count = self.fc1_k_tile_count
            else:
                k_tile_count = self.fc2_k_tile_count
            accumulator_stage_index = acc_producer_state.index

            if self.is_leader_cta:
                accumulator = accumulator_base[(None, None, accumulator_stage_index)]
                accumulator = cute.tiled_divide(accumulator, accumulator.shape)
                mma_tmem_sfb = tmem_sfb
                if cutlass.const_expr(self.cta_tile_n == 64):
                    sfb_shift = (work_tile.tile_n_idx % Int32(2)) * Int32(2)
                    shifted_sfb_pointer = cute.recast_ptr(
                        accumulator_pointer
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + sfb_shift,
                        dtype=self.sf_dtype,
                    )
                    mma_tmem_sfb = cute.make_tensor(shifted_sfb_pointer, sfb_layout)
                # At tile N 256 a dynamic extent of 128 or less makes the SFB TMEM
                # window compact, so its two K halves come from a different pair of
                # SMEM rest coordinates than the full-width case.
                use_n128_sfb_mapping = False
                if cutlass.const_expr(
                    self.quant_kind == QuantKind.nvfp4 and self.cta_tile_n == 256
                ):
                    use_n128_sfb_mapping = round_up(
                        work_tile.valid_tokens_in_cta_tile, 16
                    ) <= Int32(128)

                if cutlass.const_expr(self.runtime_instruction_n):
                    accumulator_tmem, sfa_tmem, sfb_tmem, instruction_descriptor = (
                        self._runtime_n_tile_operands(
                            work_tile, accumulator_pointer, accumulator_stage_index
                        )
                    )
                if cutlass.const_expr(self.runtime_sfb_window):
                    single_sfb_atom = self._instruction_n(work_tile) <= Int32(128)

                acc_pipeline.producer_acquire(acc_producer_state)
                if use_n128_sfb_mapping:
                    next_ab_pipeline_state = ab_pipeline_state.clone()
                    next_ab_pipeline_state.advance()
                    full_status = ab_pipeline.consumer_try_wait(ab_pipeline_state)
                    for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                        ab_pipeline.consumer_wait(ab_pipeline_state, full_status)
                        next_full_status = cutlass.Boolean(1)
                        if k_tile_idx + 1 < k_tile_count:
                            next_full_status = ab_pipeline.consumer_try_wait(
                                next_ab_pipeline_state
                            )

                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_pipeline_state.index,
                        )
                        cute.copy(
                            tiled_copy_sfa,
                            partitioned_smem_sfa[s2t_stage_coord],
                            partitioned_tmem_sfa,
                        )
                        for instruction_index in cutlass.range_constexpr(
                            self.num_mma_instructions_per_ab_stage
                        ):
                            # Rest_Tiler order: N0K0, N128K0, N0K64, N128K64.
                            lower_n_lower_k_source_coord = (
                                None,
                                0,
                                None,
                                instruction_index,
                                ab_pipeline_state.index,
                            )
                            lower_n_lower_k_destination_coord = (
                                None,
                                0,
                                None,
                                instruction_index,
                            )
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[lower_n_lower_k_source_coord],
                                partitioned_tmem_sfb[lower_n_lower_k_destination_coord],
                            )
                            lower_n_upper_k_source_coord = (
                                None,
                                2,
                                None,
                                instruction_index,
                                ab_pipeline_state.index,
                            )
                            compact_upper_k_destination_coord = (
                                None,
                                1,
                                None,
                                instruction_index,
                            )
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[lower_n_upper_k_source_coord],
                                partitioned_tmem_sfb[compact_upper_k_destination_coord],
                            )

                        a_stage_frag = fragment_a[
                            (None, None, None, ab_pipeline_state.index)
                        ]
                        b_stage_frag = fragment_b[
                            (None, None, None, ab_pipeline_state.index)
                        ]
                        self._issue_mma_window(
                            quant_kind=self.quant_kind,
                            acc_tensor=accumulator,
                            a_window_frag=a_stage_frag,
                            b_window_frag=b_stage_frag,
                            sfa_window_tensor=tmem_sfa,
                            sfb_window_tensor=mma_tmem_sfb,
                            valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                            mma_instruction_mnk=self.mma_instruction_mnk,
                            window_instruction_offset=0,
                            window_instruction_count=self.num_mma_instructions_per_ab_stage,
                            first_instruction_accumulate=k_tile_idx != 0,
                        )

                        ab_pipeline.consumer_release(ab_pipeline_state)
                        ab_pipeline_state.advance()
                        next_ab_pipeline_state.advance()
                        full_status = next_full_status
                else:
                    next_ab_pipeline_state = ab_pipeline_state.clone()
                    next_ab_pipeline_state.advance()
                    full_status = ab_pipeline.consumer_try_wait(ab_pipeline_state)
                    for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                        ab_pipeline.consumer_wait(ab_pipeline_state, full_status)
                        next_full_status = cutlass.Boolean(1)
                        if k_tile_idx + 1 < k_tile_count:
                            next_full_status = ab_pipeline.consumer_try_wait(
                                next_ab_pipeline_state
                            )

                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_pipeline_state.index,
                        )
                        cute.copy(
                            tiled_copy_sfa,
                            partitioned_smem_sfa[s2t_stage_coord],
                            partitioned_tmem_sfa,
                        )
                        if cutlass.const_expr(self.runtime_sfb_window):
                            # Mode 1 walks the 128-row scale-factor atoms; an instruction
                            # N of at most 128 reads only the first.
                            if single_sfb_atom:
                                cute.copy(
                                    tiled_copy_sfb,
                                    partitioned_smem_sfb[
                                        (None, 0, None, None, ab_pipeline_state.index)
                                    ],
                                    partitioned_tmem_sfb[(None, 0, None, None)],
                                )
                            else:
                                cute.copy(
                                    tiled_copy_sfb,
                                    partitioned_smem_sfb[s2t_stage_coord],
                                    partitioned_tmem_sfb,
                                )
                        else:
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[s2t_stage_coord],
                                partitioned_tmem_sfb,
                            )

                        if cutlass.const_expr(self.runtime_instruction_n):
                            self._issue_runtime_n_mma_stage(
                                accumulator_tmem=accumulator_tmem,
                                a_descriptor=a_descriptor_base
                                + Int64(
                                    ab_pipeline_state.index
                                    * Int32(self.a_stage_descriptor_step)
                                ),
                                b_descriptor=b_descriptor_base
                                + Int64(
                                    ab_pipeline_state.index
                                    * Int32(self.b_stage_descriptor_step)
                                ),
                                instruction_descriptor=instruction_descriptor,
                                sfa_tmem=sfa_tmem,
                                sfb_tmem=sfb_tmem,
                                accumulate_first=k_tile_idx != 0,
                            )
                        else:
                            a_stage_frag = fragment_a[
                                (None, None, None, ab_pipeline_state.index)
                            ]
                            b_stage_frag = fragment_b[
                                (None, None, None, ab_pipeline_state.index)
                            ]
                            self._issue_mma_window(
                                quant_kind=self.quant_kind,
                                acc_tensor=accumulator,
                                a_window_frag=a_stage_frag,
                                b_window_frag=b_stage_frag,
                                sfa_window_tensor=tmem_sfa,
                                sfb_window_tensor=mma_tmem_sfb,
                                valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                                mma_instruction_mnk=self.mma_instruction_mnk,
                                window_instruction_offset=0,
                                window_instruction_count=self.num_mma_instructions_per_ab_stage,
                                first_instruction_accumulate=k_tile_idx != 0,
                            )

                        ab_pipeline.consumer_release(ab_pipeline_state)
                        ab_pipeline_state.advance()
                        next_ab_pipeline_state.advance()
                        full_status = next_full_status
                acc_pipeline.producer_commit(acc_producer_state)

            acc_producer_state.advance()
            work_tile = sched_consumer.consume_work()
        acc_pipeline.producer_tail(acc_producer_state)

    @cute.jit
    def run_mma_asymmetric(
        self,
        *,
        tmem_allocator,
        weight_pipeline: pipeline.PipelineAsync,
        weight_pipeline_state: pipeline.PipelineState,
        token_pipeline: pipeline.PipelineAsync,
        token_pipeline_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        sched_consumer: SchedulerConsumer,
    ) -> None:
        """``run_mma`` for the split-depth plan, where the two operands no longer share a stage index.

        The view construction below duplicates ``run_mma``'s; the two have to be
        kept in step. What differs is only the stage bookkeeping: A and SFA are
        indexed by the weight state, B and SFB by the token state, and each stage
        is waited on and released through its own pipeline.

        No ``consumer_try_wait`` here, unlike ``run_mma``. Prefetching the next
        stage's readiness overlaps a barrier wait with MMA issue, which is worth a
        four-variable state machine when there is one pipeline -- but with two it
        doubles into eight, and this shape is weight-bound anyway: the MMA warp is
        waiting on DRAM, not on the barrier. A plain wait is also far easier for
        the compiler to schedule than two predicated ones.
        """
        tiled_mma = self.make_tiled_mma()
        fragment_a = tiled_mma.make_fragment_A(self.a_smem_tensor)
        fragment_b = tiled_mma.make_fragment_B(self.b_smem_tensor)
        if cutlass.const_expr(self.runtime_instruction_n):
            a_descriptor_base = self._smem_descriptor_base(self.a_smem_tensor)
            b_descriptor_base = self._smem_descriptor_base(self.b_smem_tensor)

        barrier_sync_unaligned(
            self.tmem_allocation_barrier_id,
            32 * (1 + self.num_accumulator_consumer_warps_per_cta),
        )
        accumulator_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
        accumulator_layout = cute.make_layout(
            self.accumulator_shape, stride=self.accumulator_stride
        )
        accumulator_base = cute.make_tensor(accumulator_pointer, accumulator_layout)

        sfa_stage_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfa(
                tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, 1
            ),
            (None, None, None, 0),
        )
        sfb_stage_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, 1
            ),
            (None, None, None, 0),
        )
        sfa_pointer = cute.recast_ptr(
            accumulator_pointer + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
        )
        sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, sfa_stage_smem_layout
        )
        tmem_sfa = cute.make_tensor(sfa_pointer, sfa_layout)
        sfb_pointer = cute.recast_ptr(
            accumulator_pointer
            + self.num_accumulator_tmem_cols
            + self.num_sfa_tmem_cols,
            dtype=self.sf_dtype,
        )
        sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, sfb_stage_smem_layout
        )
        tmem_sfb = cute.make_tensor(sfb_pointer, sfb_layout)
        tiled_copy_sfa, partitioned_smem_sfa, partitioned_tmem_sfa = (
            self._s2t_copy_and_partition(self.sfa_smem_tensor, tmem_sfa)
        )
        tiled_copy_sfb, partitioned_smem_sfb, partitioned_tmem_sfb = (
            self._s2t_copy_and_partition(self.sfb_smem_tensor, tmem_sfb)
        )

        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_accumulator_pipeline_stages
        )
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            k_tile_count = Int32(0)
            if is_fc1:
                k_tile_count = self.fc1_k_tile_count
            else:
                k_tile_count = self.fc2_k_tile_count
            accumulator_stage_index = acc_producer_state.index

            if self.is_leader_cta:
                accumulator = accumulator_base[(None, None, accumulator_stage_index)]
                accumulator = cute.tiled_divide(accumulator, accumulator.shape)
                mma_tmem_sfb = tmem_sfb
                if cutlass.const_expr(self.cta_tile_n == 64):
                    sfb_shift = (work_tile.tile_n_idx % Int32(2)) * Int32(2)
                    shifted_sfb_pointer = cute.recast_ptr(
                        accumulator_pointer
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + sfb_shift,
                        dtype=self.sf_dtype,
                    )
                    mma_tmem_sfb = cute.make_tensor(shifted_sfb_pointer, sfb_layout)
                # At tile N 256 a dynamic extent of 128 or less makes the SFB TMEM
                # window compact, so its two K halves come from a different pair of
                # SMEM rest coordinates than the full-width case. Tested inside the
                # loop rather than around it, which ``run_mma`` does to keep its
                # state machine out of the branch: the predicate is warp-uniform
                # and this loop has no state machine to duplicate.
                use_n128_sfb_mapping = False
                if cutlass.const_expr(
                    self.quant_kind == QuantKind.nvfp4 and self.cta_tile_n == 256
                ):
                    use_n128_sfb_mapping = round_up(
                        work_tile.valid_tokens_in_cta_tile, 16
                    ) <= Int32(128)

                if cutlass.const_expr(self.runtime_instruction_n):
                    accumulator_tmem, sfa_tmem, sfb_tmem, instruction_descriptor = (
                        self._runtime_n_tile_operands(
                            work_tile, accumulator_pointer, accumulator_stage_index
                        )
                    )
                if cutlass.const_expr(self.runtime_sfb_window):
                    single_sfb_atom = self._instruction_n(work_tile) <= Int32(128)

                acc_pipeline.producer_acquire(acc_producer_state)
                for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                    weight_pipeline.consumer_wait(weight_pipeline_state)
                    token_pipeline.consumer_wait(token_pipeline_state)

                    weight_stage_coord = (
                        None,
                        None,
                        None,
                        None,
                        weight_pipeline_state.index,
                    )
                    cute.copy(
                        tiled_copy_sfa,
                        partitioned_smem_sfa[weight_stage_coord],
                        partitioned_tmem_sfa,
                    )
                    if use_n128_sfb_mapping:
                        for instruction_index in cutlass.range_constexpr(
                            self.num_mma_instructions_per_ab_stage
                        ):
                            # Rest_Tiler order: N0K0, N128K0, N0K64, N128K64.
                            lower_n_lower_k_source_coord = (
                                None,
                                0,
                                None,
                                instruction_index,
                                token_pipeline_state.index,
                            )
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[lower_n_lower_k_source_coord],
                                partitioned_tmem_sfb[
                                    (None, 0, None, instruction_index)
                                ],
                            )
                            lower_n_upper_k_source_coord = (
                                None,
                                2,
                                None,
                                instruction_index,
                                token_pipeline_state.index,
                            )
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[lower_n_upper_k_source_coord],
                                partitioned_tmem_sfb[
                                    (None, 1, None, instruction_index)
                                ],
                            )
                    else:
                        token_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            token_pipeline_state.index,
                        )
                        if cutlass.const_expr(self.runtime_sfb_window):
                            # Mode 1 walks the 128-row scale-factor atoms; an instruction
                            # N of at most 128 reads only the first.
                            if single_sfb_atom:
                                cute.copy(
                                    tiled_copy_sfb,
                                    partitioned_smem_sfb[
                                        (
                                            None,
                                            0,
                                            None,
                                            None,
                                            token_pipeline_state.index,
                                        )
                                    ],
                                    partitioned_tmem_sfb[(None, 0, None, None)],
                                )
                            else:
                                cute.copy(
                                    tiled_copy_sfb,
                                    partitioned_smem_sfb[token_stage_coord],
                                    partitioned_tmem_sfb,
                                )
                        else:
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[token_stage_coord],
                                partitioned_tmem_sfb,
                            )

                    if cutlass.const_expr(self.runtime_instruction_n):
                        self._issue_runtime_n_mma_stage(
                            accumulator_tmem=accumulator_tmem,
                            a_descriptor=a_descriptor_base
                            + Int64(
                                weight_pipeline_state.index
                                * Int32(self.a_stage_descriptor_step)
                            ),
                            b_descriptor=b_descriptor_base
                            + Int64(
                                token_pipeline_state.index
                                * Int32(self.b_stage_descriptor_step)
                            ),
                            instruction_descriptor=instruction_descriptor,
                            sfa_tmem=sfa_tmem,
                            sfb_tmem=sfb_tmem,
                            accumulate_first=k_tile_idx != 0,
                        )
                    else:
                        a_stage_frag = fragment_a[
                            (None, None, None, weight_pipeline_state.index)
                        ]
                        b_stage_frag = fragment_b[
                            (None, None, None, token_pipeline_state.index)
                        ]
                        self._issue_mma_window(
                            quant_kind=self.quant_kind,
                            acc_tensor=accumulator,
                            a_window_frag=a_stage_frag,
                            b_window_frag=b_stage_frag,
                            sfa_window_tensor=tmem_sfa,
                            sfb_window_tensor=mma_tmem_sfb,
                            valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                            mma_instruction_mnk=self.mma_instruction_mnk,
                            window_instruction_offset=0,
                            window_instruction_count=self.num_mma_instructions_per_ab_stage,
                            first_instruction_accumulate=k_tile_idx != 0,
                        )

                    weight_pipeline.consumer_release(weight_pipeline_state)
                    token_pipeline.consumer_release(token_pipeline_state)
                    weight_pipeline_state.advance()
                    token_pipeline_state.advance()
                acc_pipeline.producer_commit(acc_producer_state)

            acc_producer_state.advance()
            work_tile = sched_consumer.consume_work()
        acc_pipeline.producer_tail(acc_producer_state)


__all__ = ["BlockScaledSwapAbFc12MainloopGenSpecialized"]
