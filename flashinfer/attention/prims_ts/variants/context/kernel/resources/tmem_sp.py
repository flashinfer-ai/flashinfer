# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TMEM S/P: QK scores, the online softmax and P written back for PV."""

import cutlass.cute as cute
import cutlass
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ..config import FmhaConfig
from ..device.expcast import (
    add_f16x2_to_f32x2,
    exp2_f16x2_word,
    expcast_e4m3_quad_relu,
    f16x2_sum4,
    f16x2_to_f32x2,
    pack_f16x2x2_to_fp8_e4m3,
    split_bf16_hi_lo_word,
)
from ..device.windows import (
    bottom_right_window_left_bound,
    bottom_right_window_tile_start,
    variable_window_cta_min_start,
)
from ..device.common import (
    MX_SCALE_TILE_BYTES,
    SOFTMAX_MASK_VALUE,
    SoftmaxScalar,
    TmemAddr,
    TmemPtr,
    resolve_work_tile_coords,
    NUM_SOFTMAX_WARPS,
    UNBOUNDED_K,
)
from .smem_kv import SmemKVResource
from .smem_p import SmemPResource
from ..device.sage_scales import (
    SAGE_SCORE_BLOCK_COLS,
    load_sage_k_tile,
    prefetch_sage_k_tile,
    sage_k_block_word,
    sage_k_section_address,
    sage_k_tile_bytes,
)
from ..device.expcast import E4M3_CODES_PER_OCTAVE, VC_EXPCAST_CODE_BIAS
from ..device.vc_scales import VCKScaleTable, vc_flat_k_base, vc_flat_q_slot
from .....device.packing import freeze_smem_descriptor, pack_float4_to_fp8_e4m3
from .....device.placeholders import placeholder_local_array, placeholder_tmem_ptr
from .....device.stage import WorkStage
from .....device.tcgen05 import tcgen05_ld_32x32b_max
from ....sage.kernel.index import load_sage_q_scale, sage_sequence_words
from ....sage.kernel.int32_scores import (
    INT32_SCORE_BIAS,
    fill_sage_score_seed_tile,
    issue_sage_score_seed_mma,
    sage_score_seed_words,
)
from ....sage.layout import FLOAT4_WORDS
from ....vc_attention import VC_MEAN_GROUP_TILES, VC_MEAN_MMA_K, VC_MEAN_OPERANDS
from typing import Any, Optional, TypeAlias
from .....device.mask import (
    kv_tile_needs_right_mask,
    pack_kv_tile_range,
    packed_kv_tile_in_range,
    unmasked_kv_tile_range,
)


MX_SCALE_TMEM_COL = 120
SoftmaxChunk: TypeAlias = cutlass.Vector
SoftmaxChunks: TypeAlias = list[SoftmaxChunk]
SoftmaxRowSumContribution: TypeAlias = SoftmaxChunks | SoftmaxScalar

# Trace-time storage for TmemSP s_data vectors (S/P chunks).
# Stored at module level (not on self) to avoid adding a non-dynamic-expression
# field to the dataclass, which breaks the framework's scf.if handling.
tmem_sp_sdata: dict[int, list] = {}
# Packed fp8 P words carried from exp2_p to store_p when P is staged in SMEM.
tmem_sp_pwords: dict[int, tuple] = {}
# VC-Attention: the log2 softmax scale carried from the row-max pass to exp2_p
# and the fp32 tile row sum carried from exp2_p to store_p.
tmem_sp_tile_scale: dict[int, Any] = {}


# ---------------------------------------------------------------------------
# TmemSPResource -- TMEM S/P ping-pong buffer with UmmaAsync pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemSPResource(MemoryResource):
    """TMEM S/P ping-pong buffer with UmmaAsync pipeline.

    Producer: MMA warp writes S = Q*K scores, then reads P for P*V.
    Consumer: Softmax warp reads S, computes P = softmax(S), writes P back.

    Self-edge in dependency graph enables ping-pong validation:
    MMA acquires -> writes S -> commits -> Softmax waits -> reads S,
    writes P -> releases -> MMA re-acquires -> reads P for P*V -> commits.
    """

    # Static configuration
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    enable_early_tile_sum: Constexpr[bool] = False
    old_row_max: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_chunk: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_lo: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    # 0 for SP0 (uses Q0), 1 for SP1 (uses Q1).
    q_half: Constexpr[int] = 0
    q_offset: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_max: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_sum: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    sage_k_first_atom: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    sage_k_words: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    sage_q_scale: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seqlen_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    smem_k_stages: int = field(init=False, default=0)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    # Softmax warp per-warp P address.
    tmem_p_addr_cached: TmemAddr | None = field(init=False, default=None)
    tmem_p_offset: Constexpr[int] = field(init=False, default=None)
    # Precomputed TMEM pointers/addresses (set by auxiliary work). Avoids
    # per-iteration inttoptr + address math.
    # MMA warp pointer for QK to S.
    tmem_ptr_s_cached: TmemPtr | None = field(init=False, default=None)
    # Softmax warp per-warp S address.
    tmem_s_addr_cached: TmemAddr | None = field(init=False, default=None)
    tmem_s_offset: Constexpr[int] = field(init=False, default=None)
    variable_window_end: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    variable_window_start: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    # VC-Attention-QK8: sfK of the next K/V tile, read from the SMEM table one
    # tile ahead so no shared load sits on the row-max critical chain.
    vc_k_scale_next: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_kept_row_sum: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    # Row sums of the sixteen K/V tiles of the current mean group, in the running max's
    # units. The mean restore is one bf16 K=16 UMMA per group, O += sum_t rowsum(P_t) * mu_t
    # with these sums as the A operand and the tile means as B, so each tile's sum stays live
    # until its group closes and is rescaled by every later max correction. One register
    # array of VC_MEAN_GROUP_TILES values.
    vc_meansums: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_prev_tile_sum: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    # VC-Attention: the row's softmax scale in log2 units (per work tile, times
    # sfQ under QK8) and the previous tile's fp32 row sum awaiting its deferred
    # mean step.
    vc_row_scale: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    # Linked resources
    # INT32 scores: the first instance owns the seed MMA's constant operand
    # tile and the second reads it.
    score_seed_owner: Optional["TmemSPResource"] = field(init=False, default=None)
    # p_in_smem: this group's SMEM P tile.
    smem_p: Optional[SmemPResource] = field(init=False, default=None)

    # Runtime scalars
    # Valid K/V rows of the final tile, computed once when the lengths bind.
    k_tail_rows: int | Int32 = field(init=False, default=0)
    max_seq_len_kv: int | Int32 = field(init=False, default=0)
    num_kv_heads: int | Int32 = field(init=False, default=0)
    q_offset_default: int | Int32 = field(init=False, default=0)
    scale_softmax_log2: Float32 | None = field(init=False, default=None)
    variable_window_q_stride: int | Int32 = field(init=False, default=0)

    # GMEM pointers and tensors
    cum_seqlen_k: cute.Tensor | None = field(init=False, default=None)
    cum_seqlen_q: cute.Tensor | None = field(init=False, default=None)
    # Sage attention: the prepared scales and the KV head count that strides
    # their sequences. The lane's ``sfQ`` word and the address of the
    # sequence's first ``sfK`` atom are cached per work tile; each tile's
    # ``sfK`` words pass from their load to the max pass and, when the
    # softmax scales blocks, to the P pass.
    sage_scales: cute.Pointer | None = field(init=False, default=None)
    seq_lens_kv: cute.Pointer | None = field(init=False, default=None)
    variable_window_cta_starts: cute.Tensor | None = field(init=False, default=None)
    variable_window_token_ends: cute.Tensor | None = field(init=False, default=None)
    variable_window_token_starts: cute.Tensor | None = field(init=False, default=None)
    vc_k_scale: cute.Tensor | None = field(init=False, default=None)
    # VC-Attention-QK8 dequant scales in the flat layout: Q [Hq, numel] and K/V
    # tile [Hkv, numel], both fp32.
    vc_q_scale: cute.Tensor | None = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[Optional[TmemAllocation]] = field(init=False, default=None)
    # One packed ``end << 16 | begin`` word per softmax warp: the LOOP tiles
    # that need no score mask for the warp's rows. Kept in SMEM rather than a
    # loop-carried register, which the S tile's register budget cannot spare.
    mask_range_alloc: Constexpr[Optional[SmemAllocation]] = field(
        init=False, default=None
    )
    seed_alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)
    # MXFP8 Q/K: the K ring allocation and depth, behind which the K scale atoms
    # sit, and the destination of each QK step's scale copy in the peer S tail.
    smem_k_alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        tmem_s_offset: int,
        tmem_p_offset: int,
        q_half: int = 0,
        q_offset: int | Int32 = 0,
        cum_seqlen_q: cute.Tensor | None = None,
        cum_seqlen_k: cute.Tensor | None = None,
        seq_lens_kv: cute.Pointer | None = None,
        max_seq_len_kv: int | Int32 = 0,
        variable_window_token_starts: cute.Tensor | None = None,
        variable_window_token_ends: cute.Tensor | None = None,
        variable_window_cta_starts: cute.Tensor | None = None,
        variable_window_q_stride: int | Int32 = 0,
        scale_softmax_log2: Float32 | None = None,
        smem_p: Optional[SmemPResource] = None,
        vc_q_scale: cute.Tensor | None = None,
        vc_k_scale: cute.Tensor | None = None,
        vc_repair_tiles: int = 0,
        sage_scales: cute.Pointer | None = None,
        num_kv_heads: int | Int32 = 0,
        score_seed_owner: Optional["TmemSPResource"] = None,
        smem_k: Optional[SmemKVResource] = None,
        **kwargs: Any,
    ) -> None:
        """Bind S/P TMEM offsets, Q peer index, and optional varlen metadata.

        The repair tile count is a compile-time constant. It says how many
        K/V tiles at the end of the loop are VC-Attention outlier tiles. It is
        zero on plans that restore the tile means. VC-Attention-QK8 plans carry
        the fixed Q and K/V lengths on ``cfg``, which address the scale tables.
        ``smem_k`` is the K staging resource whose scale atoms an MXFP8 QK step
        copies.
        """
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.vc_repair_tiles = vc_repair_tiles
        self.score_seed_owner = score_seed_owner
        if smem_k is not None:
            self.smem_k_alloc = smem_k.alloc
            self.smem_k_stages = smem_k.pipeline_config.num_stages
        self.tmem_s_offset = tmem_s_offset
        self.tmem_p_offset = tmem_p_offset
        self.q_half = q_half
        self.enable_early_tile_sum = cfg.enable_early_tile_sum
        self.q_offset_default = q_offset
        self.cum_seqlen_q = cum_seqlen_q
        self.cum_seqlen_k = cum_seqlen_k
        self.seq_lens_kv = seq_lens_kv
        self.max_seq_len_kv = max_seq_len_kv
        self.k_tail_rows = Int32(0)
        self.variable_window_token_starts = variable_window_token_starts
        self.variable_window_token_ends = variable_window_token_ends
        self.variable_window_cta_starts = variable_window_cta_starts
        self.variable_window_q_stride = variable_window_q_stride
        self.scale_softmax_log2 = scale_softmax_log2
        self.sage_scales = sage_scales
        self.num_kv_heads = num_kv_heads
        self.sage_q_scale = TaskLocalVariable(
            dtype=Float32,
            default=Float32(1.0),
            docs="Sage ``sfQ`` of this lane's Q row, cached per work tile.",
        )
        self.sage_k_first_atom = TaskLocalVariable(
            dtype=Int64,
            default=Int64(0),
            docs="Address of the work tile's first Sage ``sfK`` atom.",
        )
        self.sage_k_words = TaskLocalVariable(
            dtype=cutlass.Vector,
            default_factory=lambda: placeholder_unit_vector(
                cfg.sage_k_tile_words if cfg.use_sage_attention else 1
            ),
            docs="Sage ``sfK`` of each word of the current K tile.",
        )
        self.vc_q_scale = vc_q_scale
        self.vc_k_scale = vc_k_scale
        self.alloc = TmemAllocation(
            f"tmem_sp_q{q_half}",
            cfg.qk_mma_tiler[1] * cfg.mma_softmax_stage,
        )
        self.tmem_addr_cached = Int32(0)
        self.tmem_ptr_s_cached = placeholder_tmem_ptr()
        self.tmem_s_addr_cached = Int32(0)
        self.tmem_p_addr_cached = Int32(0)
        self.smem_p = smem_p
        self.old_row_max = TaskLocalVariable(
            dtype=Float32,
            default=Float32(SOFTMAX_MASK_VALUE),
            docs="Softmax row maximum from the previous K/V tile.",
        )
        self.row_max = TaskLocalVariable(
            dtype=Float32,
            default=Float32(SOFTMAX_MASK_VALUE),
            docs="Softmax row maximum for the current K/V tile.",
        )
        self.row_sum = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="Accumulated softmax denominator for the current row.",
        )
        if self.enable_early_tile_sum:
            self.p_chunk = TaskLocalVariable(
                dtype=Float32,
                default=Float32(0.0),
                docs="FP32 sum of the current probability tile.",
            )
        else:
            self.p_chunk = TaskLocalVariable(
                dtype=list,
                default_factory=lambda: placeholder_softmax_chunks(cfg),
                docs="P fragments retained for post-release row-sum reduction.",
            )
        self.p_lo = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="FP32 sum of the first half of the current probability tile.",
        )
        self.q_offset = TaskLocalVariable(
            dtype=Int32,
            default=Int32(self.q_offset_default),
            docs="Causal Q/K sequence offset for the current work tile.",
        )
        self.seqlen_k = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Request-local K/V sequence length for packed dense masking.",
        )
        self.variable_window_start = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Inclusive first K position for this Q row.",
        )
        self.variable_window_end = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Inclusive last K position for this Q row.",
        )
        self.scale_softmax_log2_value = TaskLocalVariable(
            dtype=Float32,
            # Placeholder before load_scale_softmax_log2 reads the runtime tensor.
            default=Float32(0.0),
            docs="Softmax scale cached from the runtime scale tensor.",
        )
        self.vc_row_scale = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention: the softmax scale in log2 units.",
        )
        self.vc_prev_tile_sum = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention: previous tile's row sum, stored one tile late.",
        )
        self.vc_kept_row_sum = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention V repair, row sum over the original tokens.",
        )
        self.vc_k_scale_next = TaskLocalVariable(
            dtype=Float32,
            default=Float32(1.0),
            docs="VC-Attention-QK8: K dequant scale of the next K/V tile.",
        )
        self.vc_meansums = TaskLocalVariable(
            dtype=cutlass.Array,
            default=placeholder_local_array(Float32, VC_MEAN_GROUP_TILES),
            docs="VC-Attention: row sums of the tiles of the current mean group, "
            "in the running max's units, lanes of the K=16 mean UMMA A operand.",
        )

    @property
    def loop_mask_kind(self) -> Optional[str]:
        """Return the LOOP-section score mask, in schedule dispatch order.

        ``"variable_window"``, ``"left_window"``, ``"varlen_right"``,
        ``"query_paired_causal"``, ``"fixed_dense_k_tail"``,
        ``"packed_dense_k"``, or ``None`` for an unmasked loop. Every
        softmax schedule dispatches its LOOP row max on this one value.
        """
        if self.uses_variable_window:
            return "variable_window"
        if self.uses_left_window_loop_mask:
            return "left_window"
        if self.uses_varlen_loop_right_mask:
            return "varlen_right"
        if self.uses_query_paired_q_offset_loop_mask:
            return "query_paired_causal"
        if self.uses_fixed_dense_k_tail_mask:
            return "fixed_dense_k_tail"
        if self.uses_packed_dense_k_mask:
            return "packed_dense_k"
        return None

    @property
    def loop_offset_sensitive(self) -> bool:
        """Return true because MMA and masking decisions use loop_offset."""
        # producer_work and head-paired masking use loop_offset for K tile indices.
        return True

    @property
    def needs_window_tail_left_mask(self) -> bool:
        """Return whether a sliding-window TAIL can cross its left edge.

        For fixed equal-length 128x128 tiling, a window of at least M-1
        tokens places the entire final causal tile on or to the right of the
        left bound. Packed and bottom-right-offset inputs retain the general
        two-sided mask because their runtime tile origin can shift.
        """
        return self.cfg.window_size_left > 0 and (
            self.cfg.has_varlen
            or self.cfg.has_q_offset
            or self.cfg.q_tile_m != self.cfg.kv_tile_n
            or self.cfg.window_size_left < self.cfg.q_tile_m - 1
        )

    @property
    def uses_fixed_dense_k_tail_mask(self) -> bool:
        """Return whether fixed dense attention has a partial final K/V tile."""
        return (
            not self.cfg.is_causal
            and not self.cfg.has_varlen
            and not self.cfg.has_variable_window
            and self.cfg.fixed_dense_k_tail > 0
        )

    @property
    def uses_head_paired_causal_tail_mask(self) -> bool:
        """Return whether TAIL should use head-paired causal masking."""
        return self.cfg.is_causal and self.cfg.head_paired

    @property
    def uses_left_window_loop_mask(self) -> bool:
        """Return whether loop iterations need the left sliding-window mask."""
        return self.cfg.kv_tile_start_window_size_left > 0

    @property
    def uses_loop_unmasked_tile_range(self) -> bool:
        """Return whether LOOP masks select tiles by an SMEM-cached unmasked range.

        Two-sided (window) masks build their ``[begin, end)`` tile range once
        per work tile in HEAD. Masks with a single right bound test it in key
        space with ``kv_tile_needs_right_mask``, because their masked tiles
        need that bound anyway: the query-paired causal mask builds it from
        the cached ``q_offset`` (``causal_k_end``) and the packed dense K mask
        uses the cached ``seqlen_k``. A fixed dense K tail masks only the last
        LOOP tile, which a constant tile-count test already selects.
        """
        return self.loop_mask_kind in ("variable_window", "left_window", "varlen_right")

    @property
    def uses_packed_dense_k_mask(self) -> bool:
        """Return whether packed or paged dense attention needs local K bounds."""
        return (
            self.cfg.has_varlen
            and not self.cfg.is_causal
            and self.cfg.packed_dense_k_mask
        )

    @property
    def uses_query_paired_causal_tail_mask(self) -> bool:
        """Return whether TAIL should use query-paired causal masking."""
        return self.cfg.is_causal and not self.cfg.head_paired and self.q_half == 0

    @property
    def uses_query_paired_invalid_tail(self) -> bool:
        """Return whether peer0 needs the extra wholly-invalid tail slot."""
        return self.cfg.skip_causal_invalid_peer0 and self.q_half == 0

    @property
    def uses_query_paired_q_offset_loop_mask(self) -> bool:
        """Return whether query-paired loop iterations need q-offset masking.

        Mixed packed batches use a request-local domain, but paired-tail
        alignment and partial Q tiles can conservatively retain a K/V tile
        that crosses the causal right edge. Either peer can therefore need
        the right mask inside LOOP rather than only in peer0 TAIL. A uniform
        tile-aligned shift preserves the ordinary tail placement and compiles
        this per-iteration mask away.
        """
        return (
            self.cfg.has_q_offset
            and not self.cfg.head_paired
            and not self.cfg.has_tile_aligned_uniform_q_offset
        )

    @property
    def uses_variable_window(self) -> bool:
        """Return whether softmax consumes explicit packed-Q row bounds."""
        return self.cfg.has_variable_window

    @property
    def uses_varlen_loop_right_mask(self) -> bool:
        """Return whether loop iterations need mixed-varlen right masking."""
        return self.cfg.head_paired and self.cfg.has_varlen and self.cfg.has_q_offset

    @property
    def uses_varlen_q_offset_cache(self) -> bool:
        """Return whether masks need a per-work-tile varlen Q/K offset."""
        return self.cfg.has_varlen and self.cfg.has_q_offset

    @property
    def vc_k_scales(self) -> VCKScaleTable:
        """This softmax group's staged ``sfK`` table (see ``vc_scales``)."""
        return VCKScaleTable(self.cfg, self.smem_p, self.q_half)

    @cute.jit
    def apply_causal_mask(
        self,
        stage_info: StageInfo,
        s_data: SoftmaxChunks,
        q_offset: Int32,
    ) -> SoftmaxChunks:
        """Apply the tail-stage causal mask to a loaded S tile."""
        return self.apply_causal_mask_for_kv_tile(
            stage_info, s_data, stage_info.loop_end, q_offset
        )

    @cute.jit
    def apply_causal_mask_for_kv_tile(
        self,
        stage_info: StageInfo,
        s_data: SoftmaxChunks,
        kv_tile_idx: Int32,
        q_offset: Int32,
    ) -> SoftmaxChunks:
        """Apply the query-paired right-edge causal mask to a loaded S tile.

        Query-paired maps q_half=1 to the next sequence tile, so the row index
        includes q_half * q_tile_m. Head-paired causal tails use
        right_masked_row_max() instead. q_offset is cached once per work tile
        so varlen masking does not reload cum_seqlen_q/k in every K/V loop.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        num_softmax_warps = 4
        warp_id_in_sg = cute.arch.warp_idx() % num_softmax_warps
        seq_coord, _, _ = resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        kv_base = kv_tile_idx * self.cfg.kv_tile_n
        q_min = (
            q_offset
            + seq_coord * self.cfg.cta_tiler[0]
            + self.q_half * self.cfg.q_tile_m
        )
        k_max = kv_base + self.cfg.qk_mma_tiler[1] - Int32(1)
        need_mask = q_min <= k_max
        if need_mask:
            q_idx = q_min + warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx()
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                k_chunk_base = kv_base + chunk_idx * tmem_x
                num_valid = cute.math.min(
                    cute.math.max(q_idx - k_chunk_base + Int32(1), Int32(0)),
                    Int32(tmem_x),
                )
                causal_mask = cutlass.vector.create_mask([tmem_x], [num_valid])
                masked_values_vec = cutlass.vector.full_like(
                    s_data[chunk_idx], Float32(SOFTMAX_MASK_VALUE)
                )
                s_data[chunk_idx] = cutlass.vector.where(
                    causal_mask, s_data[chunk_idx], masked_values_vec
                )
        return s_data

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=q_offset)
    @cute.jit
    def cache_q_offset(self, stage_info: StageInfo) -> Int32:
        """Cache the per-work-tile causal Q/K sequence offset for masks.

        Mixed-varlen batches cannot use the uniform kernel q_offset because
        each batch can have a different S_kv - S_q. This pre-wait hook runs
        once in the softmax task HEAD, before the K/V loop, so loop and tail
        masks reuse the cached offset instead of rereading the request metadata.
        """
        if cutlass.const_expr(self.cfg.has_uniform_varlen):
            return Int32(self.cfg.uniform_seq_len_k - self.cfg.uniform_seq_len_q)
        batch_coord = self.varlen_batch_coord(stage_info)
        if cutlass.const_expr(self.cfg.has_uniform_varlen):
            seqlen_q = Int32(self.cfg.uniform_seq_len_q)
        else:
            cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord])
            seqlen_q = Int32(self.cum_seqlen_q[batch_coord + Int32(1)]) - cuseqlen_q
        if cutlass.const_expr(self.cfg.use_paged_kv):
            seqlen_k = Int32(self.seq_lens_kv[batch_coord])
        elif cutlass.const_expr(self.cfg.has_uniform_varlen):
            seqlen_k = Int32(self.cfg.uniform_seq_len_k)
        else:
            cuseqlen_k = Int32(self.cum_seqlen_k[batch_coord])
            seqlen_k = Int32(self.cum_seqlen_k[batch_coord + Int32(1)]) - cuseqlen_k
        return seqlen_k - seqlen_q

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(sage_q_scale, sage_k_first_atom),
    )
    @cute.jit
    def cache_sage_scales(self, stage_info: StageInfo) -> tuple[Float32, Int64]:
        """Cache this lane's Sage ``sfQ`` word and the work tile's first ``sfK`` atom.

        The lane's Q row is row ``(warp % 4) * 32 + lane`` of token range
        ``2 * seq + q_half``, at most the head's last token range, of head
        ``head``, which is local head ``head % h_r`` of KV head ``head //
        h_r``; ``_SageQSection.tile_index`` names
        the prepared Q tile holding it. The ``sfK`` atoms are the
        sequence's K-token section (``_SageKSection``), padded to whole K
        tiles; the max pass addresses each tile's atoms from the first one.
        """
        cfg = self.cfg
        layout = cfg.sage_scale_layout
        seq_coord, head_coord, batch_coord = resolve_work_tile_coords(
            cfg, stage_info.work_tile.tile_idx
        )
        kv_head_coord = head_coord // Int32(cfg.h_r)
        local_head_coord = head_coord - kv_head_coord * Int32(cfg.h_r)
        row_idx = (
            cute.arch.warp_idx() % 4
        ) * cute.arch.WARP_SIZE + cute.arch.lane_idx()
        sequence_words = sage_sequence_words(
            layout,
            batch_idx=batch_coord,
            kv_head_idx=kv_head_coord,
            num_kv_heads=self.num_kv_heads,
        )
        # The second Q half of a sequence's last work tile can start past its
        # last Q tile; its rows are past the sequence and their output is
        # dropped, so it reads the head's last tile instead of the next one's.
        token_tile = cute.math.min(
            seq_coord * Int32(cfg.work_tile_q_seq_tiles) + Int32(self.q_half),
            Int32(layout.q_rows.token_tiles - 1),
        )
        sage_q_scale = load_sage_q_scale(
            layout,
            self.sage_scales,
            sequence_words=sequence_words,
            q_tile_idx=layout.q_rows.tile_index(local_head_coord, token_tile),
            row_idx=row_idx,
        )
        first_atom = sage_k_section_address(
            layout, self.sage_scales, sequence_words=sequence_words
        )
        return sage_q_scale, first_atom

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=seqlen_k)
    @cute.jit
    def cache_seqlen_k(self, stage_info: StageInfo) -> Int32:
        """Cache the request-local K/V extent once per work tile."""
        if cutlass.const_expr(self.cfg.has_uniform_varlen):
            return Int32(self.cfg.uniform_seq_len_k)
        if cutlass.const_expr(self.cfg.use_paged_kv):
            batch_coord = self.varlen_batch_coord(stage_info)
            return Int32(self.seq_lens_kv[batch_coord])
        batch_coord = self.varlen_batch_coord(stage_info)
        cuseqlen_k = Int32(self.cum_seqlen_k[batch_coord])
        return Int32(self.cum_seqlen_k[batch_coord + Int32(1)]) - cuseqlen_k

    # The cache_unmasked_tile_range_* methods run once per work tile in the
    # softmax HEAD; each LOOP mask then tests its tile against the cached
    # range instead of rebuilding the row's bounds every iteration. Every
    # work-method input is a required routed token, so each mask kind's
    # inputs get their own entry point.

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def cache_unmasked_tile_range_q_offset(
        self, stage_info: StageInfo, *, q_offset: Int32
    ) -> None:
        """Cache the unmasked LOOP range of q-offset causal/window masks."""
        self.store_unmasked_tile_range(
            stage_info, self.unmasked_tile_range(stage_info, q_offset=q_offset)
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def cache_unmasked_tile_range_window(
        self, stage_info: StageInfo, *, window_start: Int32, window_end: Int32
    ) -> None:
        """Cache the unmasked LOOP range of the VariableWindow mask."""
        self.store_unmasked_tile_range(
            stage_info,
            self.unmasked_tile_range(
                stage_info, window_start=window_start, window_end=window_end
            ),
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(variable_window_start, variable_window_end),
    )
    @cute.jit
    def cache_variable_window_bounds(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32]:
        """Load this lane's bounds relative to the CTA's first K/V tile."""
        seq_coord, _, batch_coord = resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        warp_id_in_sg = cute.arch.warp_idx() % 4
        row_in_tile = warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx()
        local_q = (
            seq_coord * self.cfg.q_tile_m * self.cfg.work_tile_q_seq_tiles
            + self.q_half * self.cfg.peer_q_seq_tile_stride * self.cfg.q_tile_m
            + row_in_tile
        )
        local_q = cute.math.min(
            local_q,
            self.variable_window_q_stride - Int32(1),
        )
        packed_q = batch_coord * self.variable_window_q_stride + local_q
        min_window_start = variable_window_cta_min_start(
            self.variable_window_cta_starts,
            batch_coord=batch_coord,
            seq_coord=seq_coord,
            q_stride=self.variable_window_q_stride,
            tile_size_q=self.cfg.cta_tiler[0],
        )
        kv_base = (min_window_start // self.cfg.kv_tile_n) * self.cfg.kv_tile_n
        return (
            Int32(self.variable_window_token_starts[packed_q]) - kv_base,
            Int32(self.variable_window_token_ends[packed_q]) - kv_base,
        )

    @cute.jit
    def causal_k_end(self, stage_info: StageInfo, q_offset: Int32) -> Int32:
        """Return the keys ``[0, end)`` visible to the Q tile's first row.

        The query-paired causal right bound grows by one key per row, so row
        ``r`` of the tile sees ``end + r`` keys and a tile whose keys all lie
        below ``end`` needs no mask. Built from the per-work-tile ``q_offset``
        and tile coordinates on the uniform datapath: caching ``end`` across
        the LOOP instead measured 3-8% slower on the register-bound D256
        LDTM.STAT softmax, whose S tile the allocator then copies every tile.
        """
        cfg = self.cfg
        seq_coord, _, _ = resolve_work_tile_coords(cfg, stage_info.work_tile.tile_idx)
        return (
            Int32(q_offset)
            + seq_coord * cfg.cta_tiler[0]
            + self.q_half * cfg.q_tile_m
            + Int32(1)
        )

    @cute.jit
    def causal_k_end_needs_mask(
        self, kv_tile_idx: Int32, causal_k_end: Int32
    ) -> cutlass.Boolean:
        """Return whether K/V tile ``kv_tile_idx`` crosses the tile's causal end."""
        return kv_tile_needs_right_mask(
            kv_tile_idx * self.cfg.kv_tile_n,
            Int32(self.cfg.qk_mma_tiler[1]),
            causal_k_end,
        )

    @cute.jit
    def causal_row_k_end(self, causal_k_end: Int32) -> Int32:
        """Return the keys this thread's Q row sees: ``causal_k_end + row``."""
        return (
            causal_k_end
            + (cute.arch.warp_idx() % NUM_SOFTMAX_WARPS) * cute.arch.WARP_SIZE
            + cute.arch.lane_idx()
        )

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def compute_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Main K-loop stage: load S from TMEM and compute unmasked row_max."""
        if cutlass.const_expr(self.cfg.uses_ldtm_stat):
            return self.load_s_chunks_and_reduce_row_max(stage_info, row_max)
        s_data = self.load_s_chunks(stage_info)
        return self.reduce_row_max(s_data, row_max)

    @cute.jit
    def default_p_chunk(self) -> SoftmaxRowSumContribution:
        if cutlass.const_expr(self.enable_early_tile_sum):
            return Float32(0.0)
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x

        # PERF NOTE: These P chunk vectors become iter_args of the scf.while
        # persistent loop. The MLIR compiler materializes zero-initialization
        # HEAD code (~128 add.rn.f32x2 instructions adding 0+0) and
        # TAIL finalization code (runs once after LOOP exit). The K-loop
        # body instruction count is unaffected — identical with or without
        # these iter_args. The HEAD/TAIL overhead may affect performance
        # through i-cache pressure (~+3% PTX footprint), register allocation
        # changes (ptxas sees more live values at scf.while boundary), and
        # pipeline warm-up timing shifts.
        p_chunk = []
        for _chunk_idx in cutlass.range_constexpr(num_chunks):
            zeros = tuple(self.cfg.qk_acc_dtype(0.0) for _ in range(tmem_x))
            p_chunk.append(cutlass.Vector.from_elements(zeros, self.cfg.qk_acc_dtype))
        return p_chunk

    @cute.jit
    def dequantize_s_chunks(
        self, s_data: SoftmaxChunks, sage_k_words: cutlass.Vector
    ) -> SoftmaxChunks:
        """Multiply the loaded scores by ``sfK`` in place (``SageScoreScaling.DEQUANTIZED``).

        ``sage_k_words`` holds the tile's ``sfK`` words; an atom's words are
        unpadded at these block sizes, so word ``w`` scales the columns
        ``w * tokens_per_word`` onward. The scores take the words in packed
        pairs. ``sfQ`` is not applied here: the max pass folds it into the
        tile maximum and the P pass into the exp2 scale, which saves one
        multiply per word; the lazy correction sees dequantized row maxima.
        INT32 scores drop their bias here: words that cover several tokens
        fold ``-bias * sfK`` into one FMA per score pair (one rounding per
        word, below half a quantized score step), one-token words subtract
        it first, exactly, which is cheaper than a product per word.
        """
        cfg = self.cfg
        tmem_x = cfg.tmem_x_load_s
        num_chunks = cfg.qk_mma_tiler[1] // tmem_x
        tokens_per_word = cfg.sage_scale_layout.k_tokens.tokens_per_word
        folds_bias = cfg.uses_int32_scores and tokens_per_word > 1
        subtracts_bias = cfg.uses_int32_scores and not folds_bias
        if cutlass.const_expr(cfg.uses_int32_scores):
            minus_bias_pair = (Float32(-INT32_SCORE_BIAS), Float32(-INT32_SCORE_BIAS))
        if cutlass.const_expr(folds_bias):
            bias_words: tuple[Any, ...] = ()
            for word_idx in cutlass.range_constexpr(0, cfg.sage_k_tile_words, 2):
                bias_words += cute.arch.mul_packed_f32x2(
                    (
                        Float32(sage_k_words[word_idx]),
                        Float32(sage_k_words[word_idx + 1]),
                    ),
                    minus_bias_pair,
                    rnd="rn",
                    ftz=False,
                )
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            scaled: tuple[Any, ...] = ()
            for elem_idx in cutlass.range_constexpr(0, tmem_x, 2):
                col = chunk_idx * tmem_x + elem_idx
                words = (col // tokens_per_word, (col + 1) // tokens_per_word)
                score_pair = (
                    s_data[chunk_idx][elem_idx],
                    s_data[chunk_idx][elem_idx + 1],
                )
                word_pair = (
                    Float32(sage_k_words[words[0]]),
                    Float32(sage_k_words[words[1]]),
                )
                if cutlass.const_expr(folds_bias):
                    scaled += cute.arch.fma_packed_f32x2(
                        score_pair,
                        word_pair,
                        (bias_words[words[0]], bias_words[words[1]]),
                        rnd="rn",
                        ftz=False,
                    )
                else:
                    if cutlass.const_expr(subtracts_bias):
                        score_pair = cute.arch.add_packed_f32x2(
                            score_pair, minus_bias_pair, rnd="rn", ftz=False
                        )
                    scaled += cute.arch.mul_packed_f32x2(
                        score_pair, word_pair, rnd="rn", ftz=False
                    )
            s_data[chunk_idx] = cutlass.Vector.from_elements(scaled, cfg.qk_acc_dtype)
        return s_data

    @consumer_work(returns=p_chunk)
    @cute.jit
    def exp2_p(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """Apply exp2 using the runtime scale cached before the K/V loop."""
        return self.exp2_p_store(
            self.stage_col_offset(stage_info), row_max, scale_softmax_log2
        )

    @consumer_work(returns=p_chunk)
    @cute.jit
    def exp2_p_hi(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
        p_lo: Float32,
    ) -> Float32:
        """Second half of exp2_p: keys [N/2, N). Returns the full tile sum."""
        return self.exp2_p_store(
            self.stage_col_offset(stage_info),
            row_max,
            scale_softmax_log2,
            chunk_lo=self.cfg.qk_mma_tiler[1] // self.cfg.tmem_x_load_s // 2,
            sum_in=p_lo,
        )

    @consumer_work(returns=p_lo)
    @cute.jit
    def exp2_p_lo(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> Float32:
        """First half of exp2_p: keys [0, N/2). Stores those P columns and
        fences so the MMA can start PV on them. Returns their row sum."""
        return self.exp2_p_store(
            self.stage_col_offset(stage_info),
            row_max,
            scale_softmax_log2,
            chunk_hi=self.cfg.qk_mma_tiler[1] // self.cfg.tmem_x_load_s // 2,
            sum_in=Float32(0.0),
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_chunk)
    @cute.jit
    def exp2_p_smem(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """exp2_p with P in SMEM, run on the loaded S after the stage release."""
        return self.exp2_p_store(
            self.stage_col_offset(stage_info), row_max, scale_softmax_log2
        )

    @cute.jit
    def exp2_p_store(
        self,
        stage_col_offset: TmemAddr,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
        chunk_lo: cutlass.Constexpr[int] = 0,
        chunk_hi: cutlass.Constexpr[int | None] = None,
        sum_in: Float32 | None = None,
        sage_q_scale: Float32 | None = None,
        sage_k_block_scales: cutlass.Vector | None = None,
    ) -> SoftmaxRowSumContribution:
        """Apply exp2 softmax P, fold the PV P scale, and store P to TMEM.

        ``chunk_lo``/``chunk_hi`` select the key chunks to process; the whole
        tile by default. A partial range fences its stores and returns
        ``sum_in`` plus its row sum, for the two-half publish.

        With block-scaled Sage attention each 16-column block's raw scores
        take the fused multiplier ``sfQ * sfK_block * scale_softmax_log2`` in
        place of the scalar scale, so the exp2 argument stays one FMA per
        score; the bias is unchanged because the row max is already
        dequantized. Each block's multiplier is formed where it is first
        used, so no per-block array stays live across the pass. A plan that
        multiplied its scores by ``sfK`` in place passes ``sage_q_scale``
        alone: the exp2 scale takes ``sfQ`` while the bias keeps the
        dequantized row max.
        """
        tmem_p_addr = self.tmem_p_addr_cached + stage_col_offset
        tmem_shape = "32x32b"
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        if cutlass.const_expr(chunk_hi is None):
            chunk_hi = num_chunks
        partial = chunk_lo > 0 or chunk_hi < num_chunks
        p_packing_ratio = self.cfg.qk_acc_dtype.width // self.cfg.v_dtype.width
        scale = scale_softmax_log2
        if cutlass.const_expr(self.cfg.vc_attention):
            # The row-max pass kept row_max in log2 units of the scaled scores.
            scale = tmem_sp_tile_scale.pop(id(self))
        block_scaled = sage_k_block_scales is not None
        blocks_per_chunk = tmem_x // SAGE_SCORE_BLOCK_COLS
        if cutlass.const_expr(block_scaled):
            sage_q_scale_log2 = sage_q_scale * scale_softmax_log2
        if cutlass.const_expr(not partial and self.cfg.uses_d256_fp8_softmax_cadence):
            return self.exp2_p_store_d256_fp8_cadence(
                tmem_p_addr,
                row_max,
                scale,
            )
        if cutlass.const_expr(not partial and self.cfg.uses_d128_fp8_softmax_cadence):
            return self.exp2_p_store_d128_fp8_cadence(
                tmem_p_addr,
                row_max,
                scale,
            )
        p_data_f32 = cutlass.Array(self.cfg.qk_acc_dtype, tmem_x, alignment=16)
        p_data_packed = cutlass.Array(
            p_data_f32.data_ptr(),
            shape=(tmem_x * p_packing_ratio,),
            dtype=self.cfg.v_dtype,
        )
        p_scale_log2 = Float32(self.cfg.pv_p_scale_log2)
        if cutlass.const_expr(self.cfg.vc_attention):
            # The paper's 2^8 P scaling is part of VC_EXPCAST_CODE_BIAS, so
            # p_scale_log2 is not added here.
            minus_row_max = Float32(0.0) - row_max
            if cutlass.const_expr(self.cfg.uses_int32_scores):
                # INT32 scores carry the float bias; fold its removal into the
                # exp2 addend so the per-score FFMA stays one instruction.
                minus_row_max = minus_row_max - Float32(INT32_SCORE_BIAS) * scale
            return self.exp2_p_store_expcast(scale, minus_row_max)
        minus_row_max_scale = softmax_exp2_bias(row_max, scale, p_scale_log2)
        # The exp2 addend of a score pair; a block-scaled INT32 plan folds
        # the score bias into it per block.
        addend_pair = (minus_row_max_scale, minus_row_max_scale)
        if cutlass.const_expr(sage_q_scale is not None and not block_scaled):
            scale = scale_softmax_log2 * sage_q_scale
        if cutlass.const_expr(chunk_hi < num_chunks):
            # A later range reads the same S data.
            s_data = tmem_sp_sdata[id(self)]
        else:
            s_data = tmem_sp_sdata.pop(id(self))
        if cutlass.const_expr(
            self.cfg.exp2_packed_f16x2
            and not block_scaled
            and (self.enable_early_tile_sum or partial)
        ):
            return self.exp2_p_store_packed_words(
                tmem_p_addr,
                scale,
                minus_row_max_scale,
                s_data,
                chunk_lo,
                chunk_hi,
                sum_in,
            )
        if cutlass.const_expr(self.enable_early_tile_sum or partial):
            # Four packed float2 accumulators keep eight independent FADD2
            # chains, so no add waits on the MUFU result it consumes.
            local_sum_pair_0 = (Float32(0.0), Float32(0.0))
            local_sum_pair_1 = (Float32(0.0), Float32(0.0))
            local_sum_pair_2 = (Float32(0.0), Float32(0.0))
            local_sum_pair_3 = (Float32(0.0), Float32(0.0))
        for chunk_idx in cutlass.range_constexpr(chunk_lo, chunk_hi):
            p_vals = ()
            for elem_idx in cutlass.range_constexpr(0, tmem_x, 2):
                if cutlass.const_expr(
                    block_scaled and elem_idx % SAGE_SCORE_BLOCK_COLS == 0
                ):
                    block_idx = (
                        chunk_idx * blocks_per_chunk + elem_idx // SAGE_SCORE_BLOCK_COLS
                    )
                    scale = Float32(sage_k_block_scales[block_idx]) * sage_q_scale_log2
                    if cutlass.const_expr(self.cfg.uses_int32_scores):
                        # One rounding per block instead of one per score; it
                        # stays below half a quantized score step.
                        addend_pair = cute.arch.fma_packed_f32x2(
                            (Float32(-INT32_SCORE_BIAS), Float32(-INT32_SCORE_BIAS)),
                            (scale, scale),
                            (minus_row_max_scale, minus_row_max_scale),
                            rnd="rn",
                            ftz=False,
                        )
                fma_pair = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][elem_idx],
                        s_data[chunk_idx][elem_idx + 1],
                    ),
                    (scale, scale),
                    addend_pair,
                    rnd="rn",
                    ftz=False,
                )
                if cutlass.const_expr(
                    elem_idx // 2 >= tmem_x // 2 - self.cfg.exp2_fma_pairs
                ):
                    p0, p1 = exp2_fma_packed(fma_pair[0], fma_pair[1])
                else:
                    p0 = cute.math.exp2(fma_pair[0], fastmath=True)
                    p1 = cute.math.exp2(fma_pair[1], fastmath=True)
                if cutlass.const_expr(self.enable_early_tile_sum or partial):
                    pair_idx = chunk_idx * (tmem_x // 2) + elem_idx // 2
                    if cutlass.const_expr(pair_idx % 4 == 0):
                        local_sum_pair_0 = cute.arch.add_packed_f32x2(
                            local_sum_pair_0,
                            (p0, p1),
                            rnd="rn",
                            ftz=False,
                        )
                    elif cutlass.const_expr(pair_idx % 4 == 1):
                        local_sum_pair_1 = cute.arch.add_packed_f32x2(
                            local_sum_pair_1,
                            (p0, p1),
                            rnd="rn",
                            ftz=False,
                        )
                    elif cutlass.const_expr(pair_idx % 4 == 2):
                        local_sum_pair_2 = cute.arch.add_packed_f32x2(
                            local_sum_pair_2, (p0, p1), rnd="rn", ftz=False
                        )
                    else:
                        local_sum_pair_3 = cute.arch.add_packed_f32x2(
                            local_sum_pair_3, (p0, p1), rnd="rn", ftz=False
                        )
                p_vals += (p0, p1)
            s_data[chunk_idx] = cutlass.Vector.from_elements(
                p_vals, self.cfg.qk_acc_dtype
            )
        use_fused_d128_fp8x4_pack = (
            not self.cfg.single_qkv_instance
            and self.cfg.v_dtype == cutlass.Float8E4M3FN
        )
        # One word holds four fp8 P values, so a pair of chunks is tmem_x words
        # and a chunk range inside a pair is a word sub-range of its store.
        for pair_idx in cutlass.range_constexpr(
            chunk_lo // p_packing_ratio,
            (chunk_hi + p_packing_ratio - 1) // p_packing_ratio,
        ):
            word_lo = max(chunk_lo - pair_idx * p_packing_ratio, 0) * tmem_x // 4
            word_hi = (
                min(chunk_hi - pair_idx * p_packing_ratio, p_packing_ratio)
                * tmem_x
                // 4
            )
            if cutlass.const_expr(use_fused_d128_fp8x4_pack):
                # Match the handwritten D128 pack: merge both FP8x2
                # conversions in one side-effecting block so ptxas can retain
                # the 32-bit word without a PRMT between temporary vectors.
                packed_words: tuple[Any, ...] = ()
                for word_idx in cutlass.range_constexpr(word_lo, word_hi):
                    flat_idx = word_idx * 4
                    chunk_idx = pair_idx * p_packing_ratio + flat_idx // tmem_x
                    elem_idx = flat_idx % tmem_x
                    packed_word = pack_float4_to_fp8_e4m3(
                        s_data[chunk_idx][elem_idx],
                        s_data[chunk_idx][elem_idx + 1],
                        s_data[chunk_idx][elem_idx + 2],
                        s_data[chunk_idx][elem_idx + 3],
                    )
                    packed_words += (packed_word,)
                store_fragment = cutlass.Vector.from_elements(packed_words, Int32)
                if cutlass.const_expr(self.cfg.p_in_smem):
                    tmem_sp_pwords[id(self)] = packed_words
            else:
                for slice_idx in cutlass.range_constexpr(p_packing_ratio):
                    chunk_idx = pair_idx * p_packing_ratio + slice_idx
                    p_chunk_dtype = s_data[chunk_idx].to(self.cfg.v_dtype)
                    if cutlass.const_expr(self.cfg.v_dtype.width == 8):
                        p_chunk_i8 = p_chunk_dtype.bitcast(cutlass.Int8)
                        p_data_packed[slice_idx * tmem_x : tmem_x] = p_chunk_i8
                    else:
                        p_data_packed[slice_idx * tmem_x : tmem_x] = p_chunk_dtype
                store_fragment = p_data_f32[0:tmem_x]
            if cutlass.const_expr(not self.cfg.p_in_smem):
                prims.tcgen05_st(
                    tmem_shape,
                    prims.make_tmem_ptr(
                        tmem_p_addr + pair_idx * tmem_x + word_lo, cutlass.Int8
                    ),
                    store_fragment,
                )
        if cutlass.const_expr(self.enable_early_tile_sum or partial):
            local_sum_pair_0 = cute.arch.add_packed_f32x2(
                local_sum_pair_0, local_sum_pair_2, rnd="rn", ftz=False
            )
            local_sum_pair_1 = cute.arch.add_packed_f32x2(
                local_sum_pair_1, local_sum_pair_3, rnd="rn", ftz=False
            )
            local_sum_pair = cute.arch.add_packed_f32x2(
                local_sum_pair_0, local_sum_pair_1, rnd="rn", ftz=False
            )
            tile_sum = local_sum_pair[0] + local_sum_pair[1]
        if cutlass.const_expr(partial):
            # Make this range's P visible before the barrier arrive that follows.
            cute.arch.fence_view_async_tmem_store()
            return sum_in + tile_sum
        if cutlass.const_expr(
            self.enable_early_tile_sum or self.cfg.has_tmem_p_pipeline
        ):
            # Publish TMEM store through the task-pipeline barrier without a blocking
            # store wait. The P-ready consumer pipeline orders the UMMA warp
            # after every store in the staged D256 path.
            cute.arch.fence_view_async_tmem_store()
        else:
            # Preserve the legacy publication sequence for paths that retain
            # P fragments until after SP release.
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        if cutlass.const_expr(self.enable_early_tile_sum):
            return tile_sum
        result = []
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            result.append(s_data[chunk_idx])
        return result

    @cute.jit
    def exp2_p_store_d128_fp8_cadence(
        self,
        tmem_p_addr: TmemAddr,
        row_max: SoftmaxScalar,
        scale: SoftmaxScalar,
    ) -> Float32:
        """Use TRT's D128 FP8 softmax arithmetic cadence.

        Prefetch eight FMA values and retire FP8 conversions and scalar sums
        eight values behind EXP2. Four independent sum chains preserve the
        dependency depth of the reference implementation.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_values = self.cfg.qk_mma_tiler[1]
        fma_lookahead = 8
        retirement_delay = 8
        store_group_words = 4
        words_per_chunk = 2 * store_group_words
        p_scale_log2 = Float32(self.cfg.pv_p_scale_log2)
        minus_row_max_scale = softmax_exp2_bias(row_max, scale, p_scale_log2)
        s_data = tmem_sp_sdata.pop(id(self))
        local_sum_chains = cutlass.Array(
            Float32,
            4,
            space=cutlass.AddressSpace.rmem,
        )
        for chain_idx in cutlass.range_constexpr(4):
            local_sum_chains[chain_idx] = Float32(0.0)

        fma_ring = cute.make_rmem_tensor((fma_lookahead,), Float32)
        exp_ring = cute.make_rmem_tensor((retirement_delay,), Float32)
        p_output_words_lo = cute.make_rmem_tensor((store_group_words,), Int32)
        p_output_words_hi = cute.make_rmem_tensor((store_group_words,), Int32)
        num_chunks = num_values // tmem_x
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            for local_idx in cutlass.range_constexpr(0, fma_lookahead, 2):
                fma_pair = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][local_idx],
                        s_data[chunk_idx][local_idx + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                    rnd="rn",
                    ftz=False,
                )
                fma_ring[local_idx] = fma_pair[0]
                fma_ring[local_idx + 1] = fma_pair[1]

            for local_idx in cutlass.range_constexpr(0, tmem_x, 2):
                fma_idx = local_idx % fma_lookahead
                exp_idx = local_idx % retirement_delay
                if cutlass.const_expr(
                    local_idx >= retirement_delay
                    and (local_idx - retirement_delay) % 4 == 0
                ):
                    delayed_idx = local_idx - retirement_delay
                    word_idx = delayed_idx // 4
                    if cutlass.const_expr(word_idx < store_group_words):
                        p_output_words_lo[word_idx] = pack_float4_to_fp8_e4m3(
                            exp_ring[exp_idx],
                            exp_ring[exp_idx + 1],
                            exp_ring[exp_idx + 2],
                            exp_ring[exp_idx + 3],
                        )
                    else:
                        p_output_words_hi[word_idx - store_group_words] = (
                            pack_float4_to_fp8_e4m3(
                                exp_ring[exp_idx],
                                exp_ring[exp_idx + 1],
                                exp_ring[exp_idx + 2],
                                exp_ring[exp_idx + 3],
                            )
                        )

                p_0 = cute.math.exp2(fma_ring[fma_idx], fastmath=True)
                # Preserve the odd value from the current pair before the
                # circular lookahead slot is refilled with the future pair.
                fma_1 = fma_ring[fma_idx + 1]
                if cutlass.const_expr(local_idx + fma_lookahead < tmem_x):
                    future_idx = local_idx + fma_lookahead
                    fma_pair = cute.arch.fma_packed_f32x2(
                        (
                            s_data[chunk_idx][future_idx],
                            s_data[chunk_idx][future_idx + 1],
                        ),
                        (scale, scale),
                        (minus_row_max_scale, minus_row_max_scale),
                        rnd="rn",
                        ftz=False,
                    )
                    fma_ring[fma_idx] = fma_pair[0]
                    fma_ring[fma_idx + 1] = fma_pair[1]
                p_1 = cute.math.exp2(fma_1, fastmath=True)

                if cutlass.const_expr(local_idx >= retirement_delay):
                    delayed_idx = local_idx - retirement_delay
                    chain_base = ((delayed_idx // 2) % 2) * 2
                    local_sum_chains[chain_base] += exp_ring[exp_idx]
                    local_sum_chains[chain_base + 1] += exp_ring[exp_idx + 1]
                exp_ring[exp_idx] = p_0
                exp_ring[exp_idx + 1] = p_1

            for delayed_idx in cutlass.range_constexpr(
                tmem_x - retirement_delay,
                tmem_x,
                4,
            ):
                exp_idx = delayed_idx % retirement_delay
                word_idx = delayed_idx // 4
                if cutlass.const_expr(word_idx < store_group_words):
                    p_output_words_lo[word_idx] = pack_float4_to_fp8_e4m3(
                        exp_ring[exp_idx],
                        exp_ring[exp_idx + 1],
                        exp_ring[exp_idx + 2],
                        exp_ring[exp_idx + 3],
                    )
                else:
                    p_output_words_hi[word_idx - store_group_words] = (
                        pack_float4_to_fp8_e4m3(
                            exp_ring[exp_idx],
                            exp_ring[exp_idx + 1],
                            exp_ring[exp_idx + 2],
                            exp_ring[exp_idx + 3],
                        )
                    )
            for delayed_idx in cutlass.range_constexpr(
                tmem_x - retirement_delay,
                tmem_x,
                2,
            ):
                exp_idx = delayed_idx % retirement_delay
                chain_base = ((delayed_idx // 2) % 2) * 2
                local_sum_chains[chain_base] += exp_ring[exp_idx]
                local_sum_chains[chain_base + 1] += exp_ring[exp_idx + 1]

            prims.tcgen05_st(
                "32x32b",
                prims.make_tmem_ptr(
                    tmem_p_addr + chunk_idx * words_per_chunk,
                    cutlass.Int8,
                ),
                p_output_words_lo.load(),
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_st(
                "32x32b",
                prims.make_tmem_ptr(
                    tmem_p_addr + chunk_idx * words_per_chunk + store_group_words,
                    cutlass.Int8,
                ),
                p_output_words_hi.load(),
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)

        local_sum_pair = cute.arch.add_packed_f32x2(
            (local_sum_chains[0], local_sum_chains[1]),
            (local_sum_chains[2], local_sum_chains[3]),
            rnd="rn",
            ftz=False,
        )
        return local_sum_pair[0] + local_sum_pair[1]

    @cute.jit
    def exp2_p_store_d256_fp8_cadence(
        self,
        tmem_p_addr: TmemAddr,
        row_max: SoftmaxScalar,
        scale: SoftmaxScalar,
    ) -> Float32:
        """Interleave D256 FP8 EXP2, conversion, and tile-sum retirement.

        The eight-value lookahead mirrors the handwritten D256 FMHA cadence
        while retaining immutable SSA values. This avoids the long all-EXP2
        burst without reintroducing the mutable cross-typed fragment that was
        nondeterministic under Task Scheduling control flow.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_values = self.cfg.qk_mma_tiler[1]
        p_scale_log2 = Float32(self.cfg.pv_p_scale_log2)
        minus_row_max_scale = softmax_exp2_bias(row_max, scale, p_scale_log2)
        s_data = tmem_sp_sdata.pop(id(self))

        fma_values: tuple[Any, ...] = ()
        for flat_idx in cutlass.range_constexpr(0, 8, 2):
            chunk_idx = flat_idx // tmem_x
            elem_idx = flat_idx % tmem_x
            fma_pair = cute.arch.fma_packed_f32x2(
                (
                    s_data[chunk_idx][elem_idx],
                    s_data[chunk_idx][elem_idx + 1],
                ),
                (scale, scale),
                (minus_row_max_scale, minus_row_max_scale),
                rnd="rn",
                ftz=False,
            )
            fma_values += (fma_pair[0], fma_pair[1])

        p_values: tuple[Any, ...] = ()
        packed_words: tuple[Any, ...] = ()
        local_sum_pair_0 = (Float32(0.0), Float32(0.0))
        local_sum_pair_1 = (Float32(0.0), Float32(0.0))
        use_two_sum_pairs = not self.cfg.single_qkv_instance
        for flat_idx in cutlass.range_constexpr(0, num_values, 2):
            if cutlass.const_expr(flat_idx >= 8):
                delayed_idx = flat_idx - 8
                if cutlass.const_expr(flat_idx % 4 == 0):
                    packed_word = pack_float4_to_fp8_e4m3(
                        p_values[delayed_idx],
                        p_values[delayed_idx + 1],
                        p_values[delayed_idx + 2],
                        p_values[delayed_idx + 3],
                    )
                    packed_words += (packed_word,)
                if cutlass.const_expr(
                    use_two_sum_pairs and (delayed_idx // 2) % 2 == 1
                ):
                    local_sum_pair_1 = cute.arch.add_packed_f32x2(
                        local_sum_pair_1,
                        (p_values[delayed_idx], p_values[delayed_idx + 1]),
                        rnd="rn",
                        ftz=False,
                    )
                else:
                    local_sum_pair_0 = cute.arch.add_packed_f32x2(
                        local_sum_pair_0,
                        (p_values[delayed_idx], p_values[delayed_idx + 1]),
                        rnd="rn",
                        ftz=False,
                    )

            p0 = cute.math.exp2(fma_values[flat_idx], fastmath=True)
            if cutlass.const_expr(flat_idx + 8 < num_values):
                future_idx = flat_idx + 8
                chunk_idx = future_idx // tmem_x
                elem_idx = future_idx % tmem_x
                fma_pair = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][elem_idx],
                        s_data[chunk_idx][elem_idx + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                    rnd="rn",
                    ftz=False,
                )
                fma_values += (fma_pair[0], fma_pair[1])
            p1 = cute.math.exp2(fma_values[flat_idx + 1], fastmath=True)
            p_values += (p0, p1)

        for delayed_idx in cutlass.range_constexpr(num_values - 8, num_values, 2):
            if cutlass.const_expr(delayed_idx % 4 == 0):
                packed_word = pack_float4_to_fp8_e4m3(
                    p_values[delayed_idx],
                    p_values[delayed_idx + 1],
                    p_values[delayed_idx + 2],
                    p_values[delayed_idx + 3],
                )
                packed_words += (packed_word,)
            if cutlass.const_expr(use_two_sum_pairs and (delayed_idx // 2) % 2 == 1):
                local_sum_pair_1 = cute.arch.add_packed_f32x2(
                    local_sum_pair_1,
                    (p_values[delayed_idx], p_values[delayed_idx + 1]),
                    rnd="rn",
                    ftz=False,
                )
            else:
                local_sum_pair_0 = cute.arch.add_packed_f32x2(
                    local_sum_pair_0,
                    (p_values[delayed_idx], p_values[delayed_idx + 1]),
                    rnd="rn",
                    ftz=False,
                )

        if cutlass.const_expr(use_two_sum_pairs):
            local_sum_pair_0 = cute.arch.add_packed_f32x2(
                local_sum_pair_0,
                local_sum_pair_1,
                rnd="rn",
                ftz=False,
            )

        store_fragment = cutlass.Vector.from_elements(packed_words, Int32)
        prims.tcgen05_st(
            "32x32b",
            prims.make_tmem_ptr(tmem_p_addr, cutlass.Int8),
            store_fragment,
        )
        cute.arch.fence_view_async_tmem_store()
        return local_sum_pair_0[0] + local_sum_pair_0[1]

    @cute.jit
    def exp2_p_store_expcast(
        self, scale: SoftmaxScalar, minus_row_max_scale: SoftmaxScalar
    ) -> Float32:
        """VC-Attention ExpCast-FP8 P with one packed FFMA per element.

        ``code = 8 * (s * scale - m) + VC_EXPCAST_CODE_BIAS``

        Packed P words go to ``store_p``. Returns the tile row sum of the
        quantized P.
        """
        assert self.cfg.p_in_smem and self.enable_early_tile_sum
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        # The exact per-tile row max keeps u <= 0, so codes stay at or below
        # the code bias.
        code_scale = scale * Float32(E4M3_CODES_PER_OCTAVE)
        code_bias = minus_row_max_scale * Float32(E4M3_CODES_PER_OCTAVE) + Float32(
            VC_EXPCAST_CODE_BIAS
        )
        s_data = tmem_sp_sdata.pop(id(self))
        packed_words: tuple[Any, ...] = ()
        acc: list[Any] = [Int32(0)] * 4
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            for elem_idx in cutlass.range_constexpr(0, tmem_x, 4):
                quad_idx = (chunk_idx * tmem_x + elem_idx) // 4
                chain = 2 * (quad_idx % 2)
                c0, c1 = cute.arch.fma_packed_f32x2(
                    (s_data[chunk_idx][elem_idx], s_data[chunk_idx][elem_idx + 1]),
                    (code_scale, code_scale),
                    (code_bias, code_bias),
                )
                c2, c3 = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][elem_idx + 2],
                        s_data[chunk_idx][elem_idx + 3],
                    ),
                    (code_scale, code_scale),
                    (code_bias, code_bias),
                )
                word, acc_lo, acc_hi = expcast_e4m3_quad_relu(
                    c0, c1, c2, c3, acc[chain], acc[chain + 1], quad_idx < 2
                )
                acc[chain] = acc_lo
                acc[chain + 1] = acc_hi
                packed_words += (word,)
        tmem_sp_pwords[id(self)] = packed_words
        return f16x2_sum4(acc[0], acc[1], acc[2], acc[3])

    @cute.jit
    def exp2_p_store_packed_words(
        self,
        tmem_p_addr: TmemAddr,
        scale: SoftmaxScalar,
        minus_row_max_scale: Float32,
        s_data: list,
        chunk_lo: cutlass.Constexpr[int],
        chunk_hi: cutlass.Constexpr[int],
        sum_in: Float32 | None,
    ) -> Float32:
        """exp2 softmax P in f16x2 words with one MUFU op per score pair. The row
        sum adds the packed halves and the fp8 P is packed from the words. Returns
        the tile sum."""
        assert self.cfg.v_dtype.width == 8
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        partial = chunk_lo > 0 or chunk_hi < num_chunks
        # Four accumulator pairs keep eight independent sum chains, each seeded
        # by its first pair.
        sums: list[Any] = [None] * 4
        p_words: dict[int, tuple[Any, ...]] = {}
        for chunk_idx in cutlass.range_constexpr(chunk_lo, chunk_hi):
            words: tuple[Any, ...] = ()
            for elem_idx in cutlass.range_constexpr(0, tmem_x, 2):
                fma_pair = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][elem_idx],
                        s_data[chunk_idx][elem_idx + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                    rnd="rn",
                    ftz=False,
                )
                word = exp2_f16x2_word(fma_pair[0], fma_pair[1])
                pair_idx = (chunk_idx - chunk_lo) * (tmem_x // 2) + elem_idx // 2
                if cutlass.const_expr(pair_idx < 4):
                    sums[pair_idx] = f16x2_to_f32x2(word)
                else:
                    sums[pair_idx % 4] = add_f16x2_to_f32x2(sums[pair_idx % 4], word)
                words += (word,)
            p_words[chunk_idx] = words
        p_packing_ratio = self.cfg.qk_acc_dtype.width // self.cfg.v_dtype.width
        for pair_idx in cutlass.range_constexpr(
            chunk_lo // p_packing_ratio,
            (chunk_hi + p_packing_ratio - 1) // p_packing_ratio,
        ):
            word_lo = max(chunk_lo - pair_idx * p_packing_ratio, 0) * tmem_x // 4
            word_hi = (
                min(chunk_hi - pair_idx * p_packing_ratio, p_packing_ratio)
                * tmem_x
                // 4
            )
            frag_words: tuple[Any, ...] = ()
            for word_idx in cutlass.range_constexpr(word_lo, word_hi):
                flat_idx = word_idx * 4
                chunk_words = p_words[pair_idx * p_packing_ratio + flat_idx // tmem_x]
                half_idx = (flat_idx % tmem_x) // 2
                frag_words += (
                    pack_f16x2x2_to_fp8_e4m3(
                        chunk_words[half_idx], chunk_words[half_idx + 1]
                    ),
                )
            if cutlass.const_expr(self.cfg.p_in_smem):
                tmem_sp_pwords[id(self)] = frag_words
            else:
                prims.tcgen05_st(
                    "32x32b",
                    prims.make_tmem_ptr(
                        tmem_p_addr + pair_idx * tmem_x + word_lo, cutlass.Int8
                    ),
                    cutlass.Vector.from_elements(frag_words, Int32),
                )
        sum_02 = cute.arch.add_packed_f32x2(sums[0], sums[2], rnd="rn", ftz=False)
        sum_13 = cute.arch.add_packed_f32x2(sums[1], sums[3], rnd="rn", ftz=False)
        local_sum_pair = cute.arch.add_packed_f32x2(sum_02, sum_13, rnd="rn", ftz=False)
        tile_sum = local_sum_pair[0] + local_sum_pair[1]
        if cutlass.const_expr(partial):
            # Make this range's P visible before the barrier arrive that follows.
            cute.arch.fence_view_async_tmem_store()
            return sum_in + tile_sum
        if cutlass.const_expr(
            self.enable_early_tile_sum or self.cfg.has_tmem_p_pipeline
        ):
            cute.arch.fence_view_async_tmem_store()
        else:
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        return tile_sum

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def fixed_dense_k_tail_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        section: cutlass.Constexpr[WorkStage] = WorkStage.Loop,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Exclude TMA zero-fill lanes in a partial fixed dense K/V tile."""
        s_data = self.load_s_chunks(stage_info)
        s_data = self.mask_fixed_dense_k_tail(stage_info, s_data, section=section)
        return self.reduce_row_max(s_data, row_max)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=row_max)
    @cute.jit
    def freeze_row_max(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        scale_softmax_log2: Float32,
    ) -> SoftmaxScalar:
        """Keep the old row max when the tile raises it by at most the threshold."""
        _ = stage_info
        threshold = Float32(self.cfg.corr_skip_threshold_log2)
        frozen = row_max
        if cutlass.const_expr(self.cfg.vc_attention):
            # VC-Attention keeps the row max in log2 units.
            scale_softmax_log2 = Float32(1.0)
        if (old_row_max != Float32(SOFTMAX_MASK_VALUE)) & (
            (row_max - old_row_max) * scale_softmax_log2 <= threshold
        ):
            frozen = old_row_max
        return frozen

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the LOOP mask ranges and the INT32 score seed tile.

        The second instance shares the first's seed tile.
        """
        requirements = []
        if self.uses_loop_unmasked_tile_range:
            if self.mask_range_alloc is None:
                self.mask_range_alloc = SmemAllocation(
                    f"{self.name}unmasked_tile_range",
                    4 * NUM_SOFTMAX_WARPS,
                    alignment=16,
                )
            requirements.append(self.mask_range_alloc)
        if not self.cfg.uses_int32_scores or self.score_seed_owner is not None:
            return requirements
        if self.seed_alloc is None:
            self.seed_alloc = SmemAllocation(
                f"{self.name}_score_seed",
                self.cfg.sage_score_seed_tile_bytes,
                alignment=128,
            )
        return requirements + [self.seed_alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return the TMEM allocation for this S/P ping-pong resource."""
        return [self.alloc]

    @cute.jit
    def init_function_state(self, stage_info: StageInfo) -> None:
        """Precompute TMEM pointers/addresses (once, before persistent loop).

        Runs on all warps via init_variables, after tmem_addr_cached is set.
        Only the MMA-warp pointer is computed here (needed ungated).
        Softmax-warp addresses are deferred to per-work-tile auxiliary work
        so they are computed after setmaxnreg and avoid crossing the register
        budget boundary.  The fields are initialized to Int32(0) here so the
        DSL sees a consistent type structure before the scf.while loop.

        Emits the softmax-side state variables (old_row_max, row_max,
        row_sum, p_chunk, q_offset) consumed by Softmax tasks; producer-side
        desc_q_base / desc_k_base slots are auto-mirrored from
        SmemQ / SmemKV by Task.init_variables (with explicit aliasing
        from desc_q0_base / desc_q1_base).
        """
        # MMA warp: tmem_ptr_s for QK→S producer_work
        self.tmem_ptr_s_cached = prims.make_tmem_ptr(
            self.tmem_addr_cached, cutlass.Int8
        ).subview(self.tmem_s_offset)
        # Initialize to establish DSL type; real values are set per work tile.
        self.tmem_s_addr_cached = Int32(0)
        self.tmem_p_addr_cached = Int32(0)
        _ = stage_info

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self.init_function_state(stage_info)
        if cutlass.const_expr(
            self.cfg.uses_int32_scores and self.score_seed_owner is None
        ):
            # The MMA task is one warp; it fills the tile it will read.
            tidx, _, _ = cute.arch.thread_idx()
            fill_sage_score_seed_tile(
                self.score_seed_words(stage_info.context),
                self.cfg.sage_score_seed_tile_bytes // 4,
                tidx % Int32(32),
                32,
            )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_work_tile_state(self, stage_info: StageInfo) -> None:
        self.init_work_tile_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_chunk)
    @cute.jit
    def init_softmax_state(self, stage_info: StageInfo) -> SoftmaxRowSumContribution:
        self.init_function_state(stage_info)
        return self.default_p_chunk()

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_softmax_state_early(self, stage_info: StageInfo) -> None:
        """Initialize softmax TMEM state without a function-lifetime P value."""
        self.init_function_state(stage_info)
        if cutlass.const_expr(self.cfg.vc_restores_means):
            # The row-sum operand exists behind the P tile only when the tile
            # means are restored; V repair plans have no such buffer.
            self.zero_vc_rowsum_row(stage_info)

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_row_max, row_max, row_sum, q_offset),
    )
    @cute.jit
    def init_softmax_work_tile_state(
        self, stage_info: StageInfo
    ) -> tuple[Float32, Float32, Float32, Int32]:
        self.init_work_tile_state(stage_info)
        if cutlass.const_expr(self.cfg.fixed_dense_k_tail > 0):
            # Bind after the softmax register allocation, like its TMEM addresses.
            # V repair pads the K/V rows to whole tiles and the row count holds
            # no tail. A repair plan is fixed and knows the tail at compile time.
            if cutlass.const_expr(self.vc_repair_tiles > 0):
                self.k_tail_rows = Int32(self.cfg.fixed_dense_k_tail)
            else:
                self.k_tail_rows = (Int32(self.max_seq_len_kv) - Int32(1)) % Int32(
                    self.cfg.kv_tile_n
                ) + Int32(1)
        return (
            Float32(SOFTMAX_MASK_VALUE),
            Float32(SOFTMAX_MASK_VALUE),
            Float32(0.0),
            Int32(self.q_offset_default),
        )

    @cute.jit
    def init_work_tile_state(self, stage_info: StageInfo) -> None:
        """Reset softmax state and recompute per-warp TMEM addresses each tile.

        Softmax-warp addresses are computed here (inside the persistent loop,
        after setmaxnreg) to avoid spilling them across the register-budget
        boundary in ungated HEAD. The returned q_offset defaults to the
        uniform kernel argument; varlen causal masks overwrite it once per
        work tile via cache_q_offset().
        """
        num_softmax_warps = 4
        warp_id_in_sg = cute.arch.warp_idx() % num_softmax_warps
        tmem_raw_addr = self.tmem_addr_cached
        tmem_base_row = tmem_raw_addr >> 16
        tmem_base_col = tmem_raw_addr & Int32(0xFFFF)
        row_id = tmem_base_row + warp_id_in_sg * cute.arch.WARP_SIZE
        self.tmem_s_addr_cached = (row_id << 16) | (tmem_base_col + self.tmem_s_offset)
        self.tmem_p_addr_cached = (row_id << 16) | (tmem_base_col + self.tmem_p_offset)
        _ = stage_info

    @consumer_work
    @cute.jit
    def invalid_exp2_p(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> None:
        """Tail stage for softmax group 0: no-op because MMA will not read P."""
        pass

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def invalid_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Tail stage for softmax group 0: identity row_max, no S load."""
        _ = stage_info
        return row_max, row_max

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def left_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        q_offset: Int32,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Apply the bottom-right-aligned sliding-window left mask.

        Only LOOP tiles outside the warp's cached unmasked range are masked.
        """
        s_data = self.load_s_chunks(stage_info)
        if self.loop_tile_needs_mask(stage_info):
            index_q, kv_tile_start, window_bound_left = self.row_window_geometry(
                stage_info, q_offset
            )
            base_k = (kv_tile_start + stage_info.loop_offset) * self.cfg.kv_tile_n
            window_end = None
            if cutlass.const_expr(self.cfg.has_varlen or self.cfg.has_q_offset):
                # Packed requests share a worst-case window span, and fixed
                # bottom-right windows can begin at a non-aligned Q/K offset.
                window_end = index_q + q_offset + Int32(1)
            s_data = self.mask_window_chunks(
                s_data, base_k, window_bound_left, window_end
            )
        return self.reduce_row_max(s_data, row_max)

    @cute.jit
    def load_s_chunks(self, stage_info: StageInfo) -> SoftmaxChunks:
        """Load ALL S chunks from TMEM into register vectors."""
        tmem_s_addr = self.tmem_s_addr_cached + self.stage_col_offset(stage_info)
        tmem_shape = "32x32b"
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        s_data = [None] * num_chunks
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            _chunk = cutlass.Array(self.cfg.qk_acc_dtype, tmem_x)
            _chunk[0:tmem_x] = prims.tcgen05_ld(
                tmem_shape,
                prims.make_tmem_ptr(
                    tmem_s_addr + chunk_idx * tmem_x, self.cfg.qk_acc_dtype
                ),
                num=tmem_x,
            )
            s_data[chunk_idx] = _chunk
        cute.arch.fence_view_async_tmem_load()
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            s_data[chunk_idx] = s_data[chunk_idx][0:tmem_x]
        return s_data

    @cute.jit
    def load_s_chunks_and_reduce_row_max(
        self,
        stage_info: StageInfo,
        row_max: SoftmaxScalar,
        *,
        causal_loop_mask: cutlass.Constexpr[bool] = False,
        causal_k_end: Int32 = 0,
        tile_scale: SoftmaxScalar | None = None,
        dense_k_mask: cutlass.Constexpr[bool] = False,
        seqlen_k: Int32 = 0,
        section: cutlass.Constexpr[WorkStage] = WorkStage.Loop,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """LDTM.STAT: fuse S load and per-chunk row_max via tcgen05.ld.red.max.

        Instead of loading each S chunk with ``tcgen05.ld`` and then folding
        a per-lane max in registers, use ``tcgen05.ld.red.max`` so the
        reduction happens as part of the TMEM load. For causal or packed dense
        masking, only tiles crossing the runtime right bound need a software
        reduction of masked scores; full tiles retain the hardware maximum.

        VC-Attention keeps the row max in log2 units. ``tile_scale`` is
        the softmax scale, applied to the hardware tile maximum before the
        merge and carried to ``exp2_p`` like the loaded S chunks.
        """
        tmem_s_addr = self.tmem_s_addr_cached + self.stage_col_offset(stage_info)
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        old_row_max = row_max
        s_data: SoftmaxChunks = [None] * num_chunks
        chunk_maxima: tuple[Any, ...] = ()
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            loaded = tcgen05_ld_32x32b_max(tmem_s_addr + chunk_idx * tmem_x, tmem_x)
            s_data[chunk_idx] = cutlass.Vector.from_elements(
                loaded[:tmem_x], dtype=self.cfg.qk_acc_dtype
            )
            chunk_maxima += (loaded[tmem_x],)
        cute.arch.fence_view_async_tmem_load()
        # Reduction results are asynchronous, just like the loaded scores.
        tile_max = cutlass.Vector.from_elements(
            chunk_maxima, self.cfg.qk_acc_dtype
        ).reduce("max")
        if cutlass.const_expr(self.cfg.vc_attention):
            if cutlass.const_expr(self.cfg.uses_int32_scores):
                tile_max = tile_max - Float32(INT32_SCORE_BIAS)
            tile_max = tile_max * tile_scale
            tmem_sp_tile_scale[id(self)] = tile_scale
        row_max = cute.math.max(row_max, tile_max)
        if cutlass.const_expr(causal_loop_mask or dense_k_mask):
            if cutlass.const_expr(causal_loop_mask):
                kv_base = stage_info.loop_offset * self.cfg.kv_tile_n
                needs_mask = self.causal_k_end_needs_mask(
                    stage_info.loop_offset, causal_k_end
                )
            else:
                if cutlass.const_expr(section == WorkStage.Loop):
                    kv_tile_idx = stage_info.loop_offset
                else:
                    kv_tile_idx = stage_info.loop_end
                kv_base = kv_tile_idx * self.cfg.kv_tile_n
                needs_mask = kv_tile_needs_right_mask(
                    kv_base, Int32(self.cfg.kv_tile_n), seqlen_k
                )
            if needs_mask:
                if cutlass.const_expr(causal_loop_mask):
                    right_bound = self.causal_row_k_end(causal_k_end)
                else:
                    right_bound = seqlen_k
                # Invalid lanes can contain another request's scores or TMA
                # padding. Neither may contribute to the maximum or softmax.
                masked_row_max = old_row_max
                for chunk_idx in cutlass.range_constexpr(num_chunks):
                    chunk_base = kv_base + Int32(chunk_idx * tmem_x)
                    valid_in_chunk = cute.math.min(
                        cute.math.max(right_bound - chunk_base, Int32(0)),
                        Int32(tmem_x),
                    )
                    mask = cutlass.vector.create_mask([tmem_x], [valid_in_chunk])
                    masked_values = cutlass.vector.full_like(
                        s_data[chunk_idx], Float32(SOFTMAX_MASK_VALUE)
                    )
                    s_data[chunk_idx] = cutlass.vector.where(
                        mask, s_data[chunk_idx], masked_values
                    )
                    masked_row_max = cute.math.max(
                        masked_row_max, s_data[chunk_idx].reduce("max")
                    )
                row_max = masked_row_max
        tmem_sp_sdata[id(self)] = s_data
        return old_row_max, row_max

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=sage_k_words)
    @cute.jit
    def load_sage_k_words(
        self,
        stage_info: StageInfo,
        *,
        sage_k_first_atom: Int64,
        section: cutlass.Constexpr[WorkStage] = WorkStage.Loop,
    ) -> cutlass.Vector:
        """Load the current K tile's ``sfK`` words and warm L1 for the next tile.

        The global transport (``SageKTransport.GMEM``), scheduled before
        the wait for the tile's scores: the wait and the TMEM load then hide
        the load latency, which the compiler would otherwise expose by
        sinking the loads next to their first use. ``section=Tail`` loads the
        tile the schedule materializes after the loop, which has no next tile.
        """
        layout = self.cfg.sage_scale_layout
        if cutlass.const_expr(section == WorkStage.Loop):
            kv_tile_idx = stage_info.loop_offset
        else:
            kv_tile_idx = stage_info.loop_end
        tile_bytes = Int64(sage_k_tile_bytes(layout))
        tile_address = sage_k_first_atom + Int64(kv_tile_idx) * tile_bytes
        words = load_sage_k_tile(layout, tile_address)
        if cutlass.const_expr(section == WorkStage.Loop):
            prefetch_sage_k_tile(layout, tile_address + tile_bytes)
        values: tuple[Any, ...] = ()
        for word_idx in cutlass.range_constexpr(self.cfg.sage_k_tile_words):
            values += (Float32(words[word_idx]),)
        return cutlass.Vector.from_elements(values, Float32)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns="scale_softmax_log2_value")
    @cute.jit
    def load_scale_softmax_log2(self, stage_info: StageInfo) -> Float32:
        """Load the runtime softmax scale once before the K/V loop."""
        _ = stage_info
        if cutlass.const_expr(self.scale_softmax_log2 is None):
            # Safe fallback for validation-only resource construction.
            return Float32(0.0)
        return self.scale_softmax_log2

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def loop_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        q_offset: Int32,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Loop stage: apply causal masking for mixed Q right-offset batches."""
        if cutlass.const_expr(self.cfg.uses_ldtm_stat):
            return self.load_s_chunks_and_reduce_row_max(
                stage_info,
                row_max,
                causal_loop_mask=True,
                causal_k_end=self.causal_k_end(stage_info, q_offset),
            )
        s_data = self.load_s_chunks(stage_info)
        s_data = self.apply_causal_mask_for_kv_tile(
            stage_info, s_data, kv_tile_idx=stage_info.loop_offset, q_offset=q_offset
        )
        return self.reduce_row_max(s_data, row_max)

    @cute.jit
    def loop_tile_needs_mask(self, stage_info: StageInfo) -> cutlass.Boolean:
        """Return whether the current LOOP tile lies outside the warp's cached range.

        Each lane reads back the packed word it wrote in this work tile's HEAD.
        """
        words, warp = self.mask_range_word(stage_info)
        return not packed_kv_tile_in_range(stage_info.loop_offset, words[warp])

    @cute.jit
    def loop_visible_k_interval(
        self,
        stage_info: StageInfo,
        *,
        q_offset: Int32,
        window_start: Int32,
        window_end: Int32,
    ) -> tuple[Int32, Int32]:
        """Return the keys ``[lo, hi)`` visible to all of this warp's rows.

        LOOP tile ``i`` covers keys ``[i * kv_tile_n, (i + 1) * kv_tile_n)``
        of the same coordinates the ``loop_mask_kind`` mask uses. Causal and
        sliding-window bounds grow with the Q row, so the warp's first row
        gives ``hi`` and its last row gives ``lo``; both come from
        warp-uniform values, which keeps the cached range out of the vector
        registers that hold the S tile. VariableWindow starts are unordered,
        so its per-row bounds are reduced across the warp.
        """
        cfg = self.cfg
        kind = self.loop_mask_kind
        first_row = (
            cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        ) * cute.arch.WARP_SIZE
        last_row = first_row + Int32(cute.arch.WARP_SIZE - 1)
        lo = Int32(0)
        hi = Int32(UNBOUNDED_K)
        if cutlass.const_expr(kind == "variable_window"):
            # Inclusive per-row bounds, relative to the CTA's first K/V tile.
            lo = cute.arch.warp_reduction(
                window_start, lambda a, b: cute.math.max(a, b)
            )
            hi = cute.arch.warp_reduction(
                window_end + Int32(1), lambda a, b: cute.math.min(a, b)
            )
        elif cutlass.const_expr(kind == "left_window"):
            first_q, kv_tile_start, _ = self.row_window_geometry(
                stage_info, q_offset, first_row
            )
            _, _, last_window_left = self.row_window_geometry(
                stage_info, q_offset, last_row
            )
            base_k = kv_tile_start * cfg.kv_tile_n
            lo = last_window_left - base_k
            if cutlass.const_expr(cfg.has_varlen or cfg.has_q_offset):
                hi = first_q + q_offset + Int32(1) - base_k
        elif cutlass.const_expr(kind == "varlen_right" and cfg.is_causal):
            first_q, kv_tile_start, _ = self.row_window_geometry(
                stage_info, q_offset, first_row
            )
            base_k = kv_tile_start * cfg.kv_tile_n
            if cutlass.const_expr(self.needs_window_tail_left_mask):
                _, _, last_window_left = self.row_window_geometry(
                    stage_info, q_offset, last_row
                )
                lo = last_window_left - base_k
            hi = first_q + q_offset + Int32(1) - base_k
        return lo, hi

    @cute.jit
    def mask_fixed_dense_k_tail(
        self,
        stage_info: StageInfo,
        s_data: SoftmaxChunks,
        *,
        section: cutlass.Constexpr[WorkStage],
    ) -> SoftmaxChunks:
        """Replace the TMA zero-fill lanes of the partial last K/V tile by -inf.

        ``section=Tail``: the last tile, masked unconditionally.
        ``section=Loop``: masked only on the last iteration.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        if cutlass.const_expr(section == WorkStage.Tail):
            is_tail_tile = True
        else:
            is_tail_tile = stage_info.loop_offset == stage_info.loop_end - Int32(1)
        if is_tail_tile:
            masked_values = cutlass.vector.full(
                [tmem_x],
                self.cfg.qk_acc_dtype(SOFTMAX_MASK_VALUE),
                dtype=self.cfg.qk_acc_dtype,
            )
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                capacity = min(
                    tmem_x, max(0, self.cfg.fixed_dense_k_tail - chunk_idx * tmem_x)
                )
                if cutlass.const_expr(capacity == 0):
                    s_data[chunk_idx] = masked_values
                else:
                    bits = ()
                    for column in cutlass.range_constexpr(tmem_x):
                        if cutlass.const_expr(column < capacity):
                            bits += (
                                self.k_tail_rows > Int32(chunk_idx * tmem_x + column),
                            )
                        else:
                            bits += (cutlass.Boolean(False),)
                    mask = cutlass.Vector.from_elements(bits, cutlass.Boolean)
                    s_data[chunk_idx] = cutlass.vector.where(
                        mask, s_data[chunk_idx], masked_values
                    )
        return s_data

    @cute.jit
    def mask_range_word(self, stage_info: StageInfo) -> tuple[cutlass.Array, Int32]:
        """Return the SMEM range words and this warp's slot in them."""
        words = cutlass.Array(
            stage_info.context.smem_base.data_ptr() + self.mask_range_alloc.offset,
            dtype=Int32,
            shape=(NUM_SOFTMAX_WARPS,),
            addrspace=3,
        )
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % NUM_SOFTMAX_WARPS
        return words, warp

    @cute.jit
    def mask_variable_window_chunks(
        self,
        stage_info: StageInfo,
        s_data: SoftmaxChunks,
        window_start: Int32,
        window_end: Int32,
    ) -> SoftmaxChunks:
        """Mask keys outside the inclusive per-row VariableWindow bounds."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        kv_tile_base = stage_info.loop_offset * self.cfg.kv_tile_n
        tile_n = self.cfg.qk_mma_tiler[1]
        left_oob = cute.math.min(
            cute.math.max(window_start - kv_tile_base, Int32(0)),
            Int32(tile_n),
        )
        right_valid = cute.math.min(
            cute.math.max(window_end + Int32(1) - kv_tile_base, Int32(0)),
            Int32(tile_n),
        )
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            chunk_base = Int32(chunk_idx * tmem_x)
            chunk_left = cute.math.min(
                cute.math.max(left_oob - chunk_base, Int32(0)),
                Int32(tmem_x),
            )
            chunk_right = cute.math.min(
                cute.math.max(right_valid - chunk_base, Int32(0)),
                Int32(tmem_x),
            )
            valid_bits = bmsk_clamp(chunk_left, chunk_right - chunk_left)
            chunk = s_data[chunk_idx]
            masked_scores = []
            for quad_idx in cutlass.range_constexpr(tmem_x // 4):
                quad_base = quad_idx * 4
                masked_scores.extend(
                    mask_score_quad(
                        valid_bits >> Int32(quad_base),
                        chunk[quad_base],
                        chunk[quad_base + 1],
                        chunk[quad_base + 2],
                        chunk[quad_base + 3],
                    )
                )
            s_data[chunk_idx] = cutlass.Vector.from_elements(
                tuple(masked_scores), self.cfg.qk_acc_dtype
            )
        return s_data

    @cute.jit
    def mask_window_chunks(
        self,
        s_data: SoftmaxChunks,
        base_k: Int32,
        window_start: Optional[Int32],
        window_end: Optional[Int32],
    ) -> SoftmaxChunks:
        """Mask scores of keys ``base_k + [0, N)`` outside ``[window_start, window_end)``.

        ``None`` leaves that side unbounded. With both bounds, each 32-column
        chunk builds one valid-bit word, so a score costs a bit test and a
        predicated mask value. A single bound keeps one create_mask and a select,
        which is as cheap and measured faster on fixed-length BF16 plans.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        if cutlass.const_expr(window_start is None or window_end is None):
            masked_values = cutlass.vector.full(
                [tmem_x],
                self.cfg.qk_acc_dtype(SOFTMAX_MASK_VALUE),
                dtype=self.cfg.qk_acc_dtype,
            )
            all_true_mask = cutlass.vector.create_mask([tmem_x], [tmem_x])
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            chunk_base = base_k + Int32(chunk_idx * tmem_x)
            if cutlass.const_expr(window_end is None):
                left_mask_inverted = cutlass.vector.create_mask(
                    [tmem_x], [window_start - chunk_base]
                )
                mask = left_mask_inverted ^ all_true_mask
                s_data[chunk_idx] = cutlass.vector.where(
                    mask, s_data[chunk_idx], masked_values
                )
            elif cutlass.const_expr(window_start is None):
                right_oob_start_idx = cute.math.min(
                    cute.math.max(window_end - chunk_base, Int32(0)),
                    Int32(tmem_x),
                )
                mask = cutlass.vector.create_mask([tmem_x], [right_oob_start_idx])
                s_data[chunk_idx] = cutlass.vector.where(
                    mask, s_data[chunk_idx], masked_values
                )
            else:
                chunk_left = cute.math.min(
                    cute.math.max(window_start - chunk_base, Int32(0)),
                    Int32(tmem_x),
                )
                chunk_right = cute.math.min(
                    cute.math.max(window_end - chunk_base, Int32(0)),
                    Int32(tmem_x),
                )
                valid_bits = bmsk_clamp(chunk_left, chunk_right - chunk_left)
                chunk = s_data[chunk_idx]
                masked_scores = []
                for quad_idx in cutlass.range_constexpr(tmem_x // 4):
                    quad_base = quad_idx * 4
                    masked_scores.extend(
                        mask_score_quad(
                            valid_bits,
                            chunk[quad_base],
                            chunk[quad_base + 1],
                            chunk[quad_base + 2],
                            chunk[quad_base + 3],
                            quad_base,
                        )
                    )
                s_data[chunk_idx] = cutlass.Vector.from_elements(
                    tuple(masked_scores), self.cfg.qk_acc_dtype
                )
        return s_data

    @consumer_work(returns=p_chunk)
    @cute.jit
    def masked_exp2_p(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """Tail stage: apply exp2 using the cached runtime softmax scale."""
        return self.exp2_p_store(
            self.stage_col_offset(stage_info), row_max, scale_softmax_log2
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_chunk)
    @cute.jit
    def masked_exp2_p_smem(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """masked_exp2_p with P in SMEM, after the stage release."""
        return self.exp2_p_store(
            self.stage_col_offset(stage_info), row_max, scale_softmax_log2
        )

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        q_offset: Int32,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Tail stage: load S, apply causal mask, and compute row_max."""
        s_data = self.load_s_chunks(stage_info)
        if cutlass.const_expr(self.cfg.is_causal):
            s_data = self.apply_causal_mask(stage_info, s_data, q_offset)
        return self.reduce_row_max(s_data, row_max)

    @cute.jit
    def mx_copy_scales(
        self,
        desc_q_base: prims.Tcgen05SmemDesc,
        desc_k_base: prims.Tcgen05SmemDesc,
        tmem_ptr_sf: TmemPtr,
    ) -> None:
        """Copy this QK step's Q and K scale atoms to ``tmem_ptr_sf``, the peer S tail.

        Each atom sits behind the ring its tile belongs to, so the start field of
        the tile's descriptor locates it. One 32x128b copy broadcast to the four
        lane quarters fills the four columns of one operand.
        """
        cfg = self.cfg
        q_bytes = Int32((desc_q_base & Int64(0x3FFF)) << 4)
        k_bytes = Int32((desc_k_base & Int64(0x3FFF)) << 4)
        k_stages = self.smem_k_stages
        k_stage = (k_bytes - Int32(self.smem_k_alloc.offset)) // cfg.tma_copy_kv_bytes
        slots = (
            q_bytes
            + (cfg.sQ_shape[0] - self.q_half) * cfg.tma_copy_q_bytes
            + self.q_half * MX_SCALE_TILE_BYTES,
            k_bytes
            + k_stages * cfg.tma_copy_kv_bytes
            - k_stage * (cfg.tma_copy_kv_bytes - MX_SCALE_TILE_BYTES),
        )
        for operand in cutlass.range_constexpr(2):
            prims.tcgen05_cp(
                prims.Tcgen05CpShape.SHAPE_32X128B,
                tmem_ptr_sf.subview(4 * operand),
                prims.Tcgen05SmemDesc.build(
                    slots[operand],
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    layout=prims.Tcgen05SmemSwizzle.NONE,
                ),
                multicast=prims.Tcgen05CpMulticast.WARPX4,
            )

    @producer_work
    @cute.jit
    def p_read(self, stage_info: StageInfo) -> None:
        """P-read sync: no-op. SP handle held from QK, consumed by softmax."""
        pass

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def packed_dense_k_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        seqlen_k: Int32,
        section: cutlass.Constexpr[WorkStage],
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Mask scores beyond one packed request's K/V right edge."""
        if cutlass.const_expr(self.cfg.uses_ldtm_stat):
            return self.load_s_chunks_and_reduce_row_max(
                stage_info,
                row_max,
                dense_k_mask=True,
                seqlen_k=seqlen_k,
                section=section,
            )
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        s_data = self.load_s_chunks(stage_info)
        if cutlass.const_expr(section == WorkStage.Loop):
            kv_tile_idx = stage_info.loop_offset
        else:
            # Some schedules materialize their final score tile in TAIL.
            kv_tile_idx = stage_info.loop_end
        kv_base = kv_tile_idx * self.cfg.kv_tile_n
        if kv_tile_needs_right_mask(kv_base, Int32(self.cfg.kv_tile_n), seqlen_k):
            masked_values = cutlass.vector.full(
                [tmem_x],
                self.cfg.qk_acc_dtype(SOFTMAX_MASK_VALUE),
                dtype=self.cfg.qk_acc_dtype,
            )
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                chunk_base = kv_base + Int32(chunk_idx * tmem_x)
                valid_in_chunk = cute.math.min(
                    cute.math.max(seqlen_k - chunk_base, Int32(0)),
                    Int32(tmem_x),
                )
                mask = cutlass.vector.create_mask([tmem_x], [valid_in_chunk])
                s_data[chunk_idx] = cutlass.vector.where(
                    mask, s_data[chunk_idx], masked_values
                )
        return self.reduce_row_max(s_data, row_max)

    @producer_work
    @cute.jit
    def qk_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_q_base: prims.Tcgen05SmemDesc,
        desc_k_base: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[WorkStage],
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Issue one QK slice, clipping MMA phases to the logical head dimension."""
        self.qk_mma_impl(
            stage_info,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            section=section,
            head_dim_stage_idx=head_dim_stage_idx,
            is_tail=is_tail,
        )

    @cute.jit
    def qk_mma_impl(
        self,
        stage_info: StageInfo,
        *,
        desc_q_base: prims.Tcgen05SmemDesc,
        desc_k_base: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[WorkStage],
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """QK MMA: compute Q*K -> S in TMEM.

        The schedule aliases either desc_q0_base or desc_q1_base into the
        logical desc_q_base producer arg based on which SP instance is being
        driven.

        In causal mode with no Q right offset, skips QK0→S0 MMA in the last
        LOOP iteration, since Softmax0's domain is N-2 but MMA's domain is N-1.
        The task domain pads partial final CTAs so this slot is always outside
        peer0's causal reach.
        """
        skip_qk0_invalid = False
        if cutlass.const_expr(self.cfg.skip_causal_invalid_peer0 and self.q_half == 0):
            if cutlass.const_expr(section == WorkStage.Loop):
                if not is_tail:
                    skip_qk0_invalid = stage_info.loop_offset == (
                        stage_info.loop_end - 1
                    )

        if not skip_qk0_invalid:
            tmem_ptr_s = self.tmem_ptr_s_cached.subview(
                self.stage_col_offset(stage_info)
            )

            acc_format = (
                cutlass.Int32 if self.cfg.uses_int32_scores else cutlass.Float32
            )
            if cutlass.const_expr(self.cfg.uses_int32_scores):
                mma_kind = prims.Tcgen05MMAKind.INT8
                ab_format = cutlass.Int8
            elif cutlass.const_expr(self.cfg.q_dtype.width == 8):
                mma_kind = prims.Tcgen05MMAKind.F8F6F4
                # E4M3 operands use the Float16 encoding handle.
                ab_format = cutlass.Float16
            else:
                mma_kind = prims.Tcgen05MMAKind.F16
                if cutlass.const_expr(self.cfg.q_dtype == cutlass.BFloat16):
                    ab_format = cutlass.BFloat16
                else:
                    ab_format = cutlass.Float16

            idesc_qk = prims.Tcgen05InstrDesc.build(
                c_dtype=acc_format,
                a_dtype=ab_format,
                b_dtype=ab_format,
                n_dim=self.cfg.qk_mma_tiler[1],
                m_dim=self.cfg.qk_mma_tiler[0] * self.cfg.cta_group_size,
            )

            k_dim_per_mma = 16
            if cutlass.const_expr(self.cfg.q_dtype.width != 16):
                k_dim_per_mma = 32
            inc_bytes_qk = k_dim_per_mma * self.cfg.q_dtype.width // 8

            num_kphases_per_tma = self.cfg.tma_copy_q_granu_inner // k_dim_per_mma
            # Byte stride between TMA fragments of the Q and K tiles. They differ only
            # in the two-CTA form, where a K stage holds half the rows.
            chunk_bytes_q = inc_bytes_qk * num_kphases_per_tma
            chunk_bytes_k = chunk_bytes_q
            if cutlass.const_expr(self.cfg.tma_copy_qkv_iters != 1):
                chunk_bytes_q = (
                    self.cfg.tma_copy_q_granu_elems * self.cfg.q_dtype.width // 8
                )
                chunk_bytes_k = (
                    self.cfg.tma_copy_kv_bytes // self.cfg.tma_copy_kv_stage_iters
                )
            cta_group = (
                prims.CTAGroup.CTA_2
                if cutlass.const_expr(self.cfg.two_cta_umma)
                else prims.CTAGroup.CTA_1
            )
            issue_mma = cutlass.Boolean(True)
            if cutlass.const_expr(self.cfg.two_cta_umma):
                issue_mma = (
                    cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
                )
            num_tma_iters_qk = self.cfg.tma_copy_qkv_iters
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                num_tma_iters_qk = self.cfg.tma_copy_kv_stage_iters

            # Prevent LLVM from rematerializing descriptor
            # computations inside each elect_sync basic block.
            # Without this, NVPTX recomputes shr+and+cvt+or from
            # __dynamic_shmem__0 inside every elect BB (~5 extra
            # instructions per MMA call).
            desc_q_base_ = freeze_smem_descriptor(desc_q_base)
            desc_k_base_ = freeze_smem_descriptor(desc_k_base)
            if cutlass.const_expr(self.cfg.use_mxfp8_qk):
                # The scales of this QK step sit in the tail of the peer S tile.
                tmem_ptr_sf = self.tmem_ptr_s_cached.subview(
                    (
                        self.cfg.tmem_s1_offset
                        if self.q_half == 0
                        else self.cfg.tmem_s0_offset
                    )
                    - self.tmem_s_offset
                    + MX_SCALE_TMEM_COL
                )
                if prims.elect_sync():
                    self.mx_copy_scales(desc_q_base, desc_k_base, tmem_ptr_sf)

            # INT32 scores accumulate onto the seeded bias; FP32 scores
            # overwrite S with the first K phase.
            scale_d = self.cfg.uses_int32_scores
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                scale_d = scale_d or head_dim_stage_idx != 0
            for tma_iter in cutlass.range_constexpr(num_tma_iters_qk):
                q_tma_iter = head_dim_stage_idx * num_tma_iters_qk + tma_iter
                q_tma_iter_offset = chunk_bytes_q * q_tma_iter
                k_tma_iter_offset = chunk_bytes_k * tma_iter
                # The final 128-wide K stage may contain only 64 logical
                # elements (non-absorbed MLA). Do not issue MMAs on padding.
                valid_kphases = min(
                    num_kphases_per_tma,
                    max(
                        0,
                        (
                            self.cfg.logical_head_dim_qk
                            - q_tma_iter * self.cfg.tma_copy_q_granu_inner
                        )
                        // k_dim_per_mma,
                    ),
                )
                for k_idx in cutlass.range_constexpr(valid_kphases):
                    local_increment = inc_bytes_qk * k_idx
                    dq = desc_q_base_ + ((local_increment + q_tma_iter_offset) >> 4)
                    dk = desc_k_base_ + ((local_increment + k_tma_iter_offset) >> 4)
                    if issue_mma:
                        if prims.elect_sync():
                            if cutlass.const_expr(self.cfg.use_mxfp8_qk):
                                # K step k reads scale byte k of each lane's column.
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4,
                                    cta_group,
                                    tmem_ptr_s,
                                    dq,
                                    dk,
                                    prims.Tcgen05MxInstrDesc.build(
                                        a_dtype=self.cfg.q_dtype,
                                        b_dtype=self.cfg.q_dtype,
                                        scale_format=prims.MxScaleFormat.E8M0,
                                        n_dim=self.cfg.qk_mma_tiler[1],
                                        m_dim=self.cfg.qk_mma_tiler[0],
                                        a_sf_id=k_idx,
                                        b_sf_id=k_idx,
                                    ),
                                    scale_d,
                                    tmem_ptr_sf,
                                    tmem_ptr_sf.subview(4),
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                            else:
                                prims.tcgen05_mma(
                                    mma_kind,
                                    cta_group,
                                    tmem_ptr_s,
                                    dq,
                                    dk,
                                    idesc_qk,
                                    scale_d,
                                )
                    scale_d = True

    @producer_work
    @cute.jit
    def qk_mma_stage(
        self,
        stage_info: StageInfo,
        *,
        desc_q_base: prims.Tcgen05SmemDesc,
        desc_k_stage: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[WorkStage],
        head_dim_stage_idx: cutlass.Constexpr[int],
    ) -> None:
        """Accumulate a later K slice without replacing the first slice's binding."""
        self.qk_mma_impl(
            stage_info,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_stage,
            section=section,
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=sage_k_words)
    @cute.jit
    def read_staged_sage_k_words(
        self, stage_info: StageInfo, *, sage_k_stage_words: Int32
    ) -> cutlass.Vector:
        """Read the current K tile's ``sfK`` words from its waited ring stage.

        The ring transport (``SageKTransport.SMEM``, protocol on
        ``SmemSageKScalesResource``): ``sage_k_stage_words`` is the stage's
        shared-memory byte offset; one float4 load per four words.
        """
        tile_words = self.cfg.sage_k_tile_words
        stage = cutlass.Array(
            stage_info.context.smem_base.data_ptr() + sage_k_stage_words,
            dtype=Float32,
            shape=(tile_words,),
            addrspace=3,
        )
        values: tuple[Any, ...] = ()
        for base in cutlass.range_constexpr(0, tile_words, FLOAT4_WORDS):
            loaded = stage.load(base, vector_size=FLOAT4_WORDS, alignment=16)
            for elem in cutlass.range_constexpr(FLOAT4_WORDS):
                values += (Float32(loaded[elem]),)
        return cutlass.Vector.from_elements(values, Float32)

    @cute.jit
    def reduce_row_max(
        self,
        s_data: SoftmaxChunks,
        row_max: SoftmaxScalar,
        tile_scale: SoftmaxScalar | None = None,
        *,
        sage_q_scale: Float32 | None = None,
        sage_k_block_scales: cutlass.Vector | None = None,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Reduce per-chunk maximums into row_max, stash s_data.

        VC-Attention passes ``tile_scale``: the raw tile maximum is scaled into
        log2 units (the scale is positive, so max commutes with it) before the
        running maximum merge, and the scale is stashed for ``exp2_p``.

        With block-scaled Sage attention the maximum is of the dequantized
        scores (``sage_dequantized_tile_max``) and the stashed scores stay
        raw. A Sage plan that multiplied its scores by ``sfK`` in place
        passes ``sage_q_scale`` alone: the tile maximum takes ``sfQ`` once,
        so the row max stays in dequantized units while the stashed scores
        carry ``sfK`` only.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        old_row_max = row_max
        if cutlass.const_expr(sage_k_block_scales is not None):
            tile_row_max = self.sage_dequantized_tile_max(
                s_data,
                sage_q_scale=sage_q_scale,
                sage_k_block_scales=sage_k_block_scales,
            )
            row_max = cute.math.max(row_max, tile_row_max)
        elif cutlass.const_expr(
            self.cfg.uses_d128_fp8_softmax_cadence
            or self.cfg.uses_d256_fp8_softmax_cadence
        ):
            max_0 = row_max
            max_1 = row_max
            max_2 = row_max
            max_3 = row_max
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                for elem_idx in cutlass.range_constexpr(0, tmem_x, 4):
                    max_0 = cute.math.max(max_0, s_data[chunk_idx][elem_idx], ftz=True)
                    max_1 = cute.math.max(
                        max_1, s_data[chunk_idx][elem_idx + 1], ftz=True
                    )
                    max_2 = cute.math.max(
                        max_2, s_data[chunk_idx][elem_idx + 2], ftz=True
                    )
                    max_3 = cute.math.max(
                        max_3, s_data[chunk_idx][elem_idx + 3], ftz=True
                    )
            max_0 = cute.math.max(max_0, max_2, ftz=True)
            max_1 = cute.math.max(max_1, max_3, ftz=True)
            row_max = cute.math.max(max_0, max_1, ftz=True)
        else:
            row_values: tuple[Any, ...] = ()
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                for elem_idx in cutlass.range_constexpr(tmem_x):
                    row_values += (s_data[chunk_idx][elem_idx],)
            row_vector = cutlass.Vector.from_elements(row_values, self.cfg.qk_acc_dtype)
            tile_row_max = row_vector.reduce("max")
            if cutlass.const_expr(self.cfg.vc_attention):
                if cutlass.const_expr(self.cfg.uses_int32_scores):
                    tile_row_max = tile_row_max - Float32(INT32_SCORE_BIAS)
                tile_row_max = tile_row_max * tile_scale
                tmem_sp_tile_scale[id(self)] = tile_scale
            if cutlass.const_expr(sage_q_scale is not None):
                tile_row_max = tile_row_max * sage_q_scale
            row_max = cute.math.max(row_max, tile_row_max)
        tmem_sp_sdata[id(self)] = s_data
        return old_row_max, row_max

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def right_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        q_offset: Int32,
        section: cutlass.Constexpr[WorkStage],
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Head-paired stage: apply causal/window mask and compute row_max.

        Unlike the query-paired tail mask, this keeps Q0/Q1 on the same
        sequence tile; q_half selects a Q head, not a later sequence tile.
        Sliding-window tails need both bounds because the final tile can also
        contain keys to the left of the visible window. LOOP tiles inside the
        warp's cached unmasked range skip the mask.
        """
        s_data = self.load_s_chunks(stage_info)
        if cutlass.const_expr(self.cfg.is_causal):
            needs_mask = True
            if cutlass.const_expr(
                section == WorkStage.Loop and self.uses_loop_unmasked_tile_range
            ):
                needs_mask = self.loop_tile_needs_mask(stage_info)
            if needs_mask:
                index_q, kv_tile_start, window_bound_left = self.row_window_geometry(
                    stage_info, q_offset
                )
                if cutlass.const_expr(section == WorkStage.Loop):
                    kv_tile_idx = kv_tile_start + stage_info.loop_offset
                else:
                    # Tail uses the first tile after the loop domain.
                    kv_tile_idx = kv_tile_start + stage_info.loop_end
                window_start = None
                if cutlass.const_expr(self.needs_window_tail_left_mask):
                    window_start = window_bound_left
                s_data = self.mask_window_chunks(
                    s_data,
                    kv_tile_idx * self.cfg.kv_tile_n,
                    window_start,
                    index_q + q_offset + Int32(1),
                )
        return self.reduce_row_max(s_data, row_max)

    @cute.jit
    def row_sum_reduction(
        self,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        p_chunk: SoftmaxChunks,
        scale_softmax_log2: SoftmaxScalar,
    ) -> Float32:
        """Accumulate row_sum from P chunks saved by consumer_work."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        scale = scale_softmax_log2
        acc_scale_ = scale * (old_row_max - row_max)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        scaled_sum = row_sum * acc_scale
        if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
            # Use four independent float2 accumulation chains for D256. A
            # single 64-pair chain serializes every FADD behind the
            # preceding result and leaves no row-sum ILP after P publication.
            local_sum_0 = (scaled_sum, scaled_sum)
            local_sum_1 = (Float32(0.0), Float32(0.0))
            local_sum_2 = (Float32(0.0), Float32(0.0))
            local_sum_3 = (Float32(0.0), Float32(0.0))
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                p_chunk_vec = p_chunk[chunk_idx]
                for elem_idx in cutlass.range_constexpr(0, tmem_x, 8):
                    local_sum_0 = cute.arch.add_packed_f32x2(
                        local_sum_0,
                        (p_chunk_vec[elem_idx], p_chunk_vec[elem_idx + 1]),
                        rnd="rn",
                        ftz=False,
                    )
                    local_sum_1 = cute.arch.add_packed_f32x2(
                        local_sum_1,
                        (p_chunk_vec[elem_idx + 2], p_chunk_vec[elem_idx + 3]),
                        rnd="rn",
                        ftz=False,
                    )
                    local_sum_2 = cute.arch.add_packed_f32x2(
                        local_sum_2,
                        (p_chunk_vec[elem_idx + 4], p_chunk_vec[elem_idx + 5]),
                        rnd="rn",
                        ftz=False,
                    )
                    local_sum_3 = cute.arch.add_packed_f32x2(
                        local_sum_3,
                        (p_chunk_vec[elem_idx + 6], p_chunk_vec[elem_idx + 7]),
                        rnd="rn",
                        ftz=False,
                    )
            local_sum_0 = cute.arch.add_packed_f32x2(
                local_sum_0, local_sum_1, rnd="rn", ftz=False
            )
            local_sum_2 = cute.arch.add_packed_f32x2(
                local_sum_2, local_sum_3, rnd="rn", ftz=False
            )
            local_sum_0 = cute.arch.add_packed_f32x2(
                local_sum_0, local_sum_2, rnd="rn", ftz=False
            )
            return local_sum_0[0] + local_sum_0[1]

        local_sum = (scaled_sum, scaled_sum)
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            p_chunk_vec = p_chunk[chunk_idx]
            for idx in cutlass.range_constexpr(tmem_x // 2):
                local_sum = cute.arch.add_packed_f32x2(
                    local_sum,
                    (p_chunk_vec[2 * idx], p_chunk_vec[2 * idx + 1]),
                    rnd="rn",
                    ftz=False,
                )
        return local_sum[0] + local_sum[1]

    @cute.jit
    def row_window_geometry(
        self,
        stage_info: StageInfo,
        q_offset: Int32,
        row_in_tile: Optional[Int32] = None,
    ) -> tuple[Int32, Int32, Int32]:
        """Return a row's Q index, the window's first K/V tile and left bound.

        ``row_in_tile`` defaults to this thread's row. LOOP tile ``i`` is K/V
        tile ``kv_tile_start + i``; without a left window ``kv_tile_start`` is
        zero. The left bound is inclusive.
        """
        cfg = self.cfg
        if cutlass.const_expr(row_in_tile is None):
            row_in_tile = (
                cute.arch.warp_idx() % 4
            ) * cute.arch.WARP_SIZE + cute.arch.lane_idx()
        seq_tile_coord, _, _ = resolve_work_tile_coords(
            cfg, stage_info.work_tile.tile_idx
        )
        index_q = seq_tile_coord * cfg.q_tile_m + row_in_tile
        kv_tile_start = Int32(0)
        window_bound_left = Int32(0)
        if cutlass.const_expr(cfg.window_size_left > 0):
            if cutlass.const_expr(cfg.has_varlen or cfg.has_q_offset):
                window_bound_left = bottom_right_window_left_bound(
                    index_q,
                    q_offset,
                    cfg.window_size_left,
                )
                kv_tile_start = bottom_right_window_tile_start(
                    seq_coord=seq_tile_coord,
                    q_tile_m=cfg.q_tile_m,
                    kv_tile_n=cfg.kv_tile_n,
                    q_offset=q_offset,
                    window_size_left=cfg.window_size_left,
                )
            else:
                # Exact fixed equal-length fast path: q_offset is constexpr zero.
                window_bound_left = index_q - Int32(cfg.window_size_left)
                kv_tile_start = cute.math.max(
                    Int32(0),
                    (seq_tile_coord * cfg.q_tile_m - cfg.window_size_left)
                    // cfg.kv_tile_n,
                )
        return index_q, kv_tile_start, window_bound_left

    @cute.jit
    def sage_block_scales(self, sage_k_words: cutlass.Vector) -> cutlass.Vector:
        """Return ``sfK`` of each 16-column score block of the tile, from its words."""
        layout = self.cfg.sage_scale_layout
        values: tuple[Any, ...] = ()
        for block_idx in cutlass.range_constexpr(self.cfg.sage_score_blocks):
            values += (Float32(sage_k_words[sage_k_block_word(layout, block_idx)]),)
        return cutlass.Vector.from_elements(values, Float32)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_chunk)
    @cute.jit
    def sage_dequantized_exp2_p_smem(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
        sage_q_scale: Float32,
    ) -> SoftmaxRowSumContribution:
        """``exp2_p_smem`` on scores that carry ``sfK`` only: ``sfQ`` joins the exp2 scale."""
        return self.exp2_p_store(
            self.stage_col_offset(stage_info),
            row_max,
            scale_softmax_log2,
            sage_q_scale=sage_q_scale,
        )

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def sage_dequantized_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        sage_q_scale: Float32,
        sage_k_words: cutlass.Vector,
        section: cutlass.Constexpr[WorkStage] = WorkStage.Loop,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Row max of scores multiplied by ``sfK`` in place (``SageScoreScaling.DEQUANTIZED``).

        ``sage_k_words`` is the tile's ``sfK`` per word, from either
        transport; ``sfQ`` scales the tile maximum. ``section=Tail`` masks
        the zero-filled lanes of the dense tail tile.
        """
        s_data = self.load_s_chunks(stage_info)
        s_data = self.dequantize_s_chunks(s_data, sage_k_words)
        if cutlass.const_expr(section == WorkStage.Tail):
            s_data = self.mask_fixed_dense_k_tail(stage_info, s_data, section=section)
        return self.reduce_row_max(s_data, row_max, sage_q_scale=sage_q_scale)

    @cute.jit
    def sage_dequantized_tile_max(
        self,
        s_data: SoftmaxChunks,
        *,
        sage_q_scale: Float32,
        sage_k_block_scales: cutlass.Vector,
    ) -> Float32:
        """Return the row's maximum dequantized score of the tile.

        Each 16-column block's raw maximum is scaled by its ``sfK`` word, the
        scaled block maxima are reduced, and the result is scaled by ``sfQ``:
        one multiply per block, as positive scales commute with the maximum.
        The row max therefore stays in the dequantized units
        ``scale_softmax_log2`` expects, so the lazy correction and the
        statistics are unchanged by Sage. Both reductions are vector
        reductions, which lower to plain FMNMX chains; a scalar ``max`` per
        block would add NaN-handling compare-and-select pairs.
        """
        tmem_x = self.cfg.tmem_x_load_s
        blocks_per_chunk = tmem_x // SAGE_SCORE_BLOCK_COLS
        block_maxima: tuple[Any, ...] = ()
        for block_idx in cutlass.range_constexpr(self.cfg.sage_score_blocks):
            chunk_idx = block_idx // blocks_per_chunk
            first_elem = (block_idx % blocks_per_chunk) * SAGE_SCORE_BLOCK_COLS
            block_values: tuple[Any, ...] = ()
            for elem_idx in cutlass.range_constexpr(
                first_elem, first_elem + SAGE_SCORE_BLOCK_COLS
            ):
                block_values += (s_data[chunk_idx][elem_idx],)
            block_maxima += (
                cutlass.Vector.from_elements(
                    block_values, self.cfg.qk_acc_dtype
                ).reduce("max"),
            )
        return self.sage_scale_block_maxima(
            block_maxima,
            sage_q_scale=sage_q_scale,
            sage_k_block_scales=sage_k_block_scales,
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_chunk)
    @cute.jit
    def sage_exp2_p_smem(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
        sage_q_scale: Float32,
        sage_k_words: cutlass.Vector,
    ) -> SoftmaxRowSumContribution:
        """``exp2_p_smem`` of block-scaled scores: the tile's ``sfK`` words of the
        max pass (``sage_row_max``) give the fused per-block multipliers."""
        return self.exp2_p_store(
            self.stage_col_offset(stage_info),
            row_max,
            scale_softmax_log2,
            sage_q_scale=sage_q_scale,
            sage_k_block_scales=self.sage_block_scales(sage_k_words),
        )

    @cute.jit
    def sage_load_s_chunks_and_reduce_row_max(
        self,
        stage_info: StageInfo,
        row_max: SoftmaxScalar,
        *,
        sage_q_scale: Float32,
        sage_k_block_scales: cutlass.Vector,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """LDTM.STAT for Sage: load S in 16-column pieces with their maxima.

        ``tcgen05.ld.red.max`` over each 16-column block returns the block's
        raw maximum with its scores, so the per-block scaling of
        ``sage_dequantized_tile_max`` costs no software reduction. The
        pieces of one 32-column chunk are joined into the chunk vector the P
        pass expects.
        """
        tmem_s_addr = self.tmem_s_addr_cached + self.stage_col_offset(stage_info)
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        blocks_per_chunk = tmem_x // SAGE_SCORE_BLOCK_COLS
        old_row_max = row_max
        s_data: SoftmaxChunks = [None] * num_chunks
        block_maxima: tuple[Any, ...] = ()
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            chunk_values: tuple[Any, ...] = ()
            for block_in_chunk in cutlass.range_constexpr(blocks_per_chunk):
                loaded = tcgen05_ld_32x32b_max(
                    tmem_s_addr
                    + chunk_idx * tmem_x
                    + block_in_chunk * SAGE_SCORE_BLOCK_COLS,
                    SAGE_SCORE_BLOCK_COLS,
                )
                chunk_values += loaded[:SAGE_SCORE_BLOCK_COLS]
                block_maxima += (loaded[SAGE_SCORE_BLOCK_COLS],)
            s_data[chunk_idx] = cutlass.Vector.from_elements(
                chunk_values, dtype=self.cfg.qk_acc_dtype
            )
        cute.arch.fence_view_async_tmem_load()
        tile_max = self.sage_scale_block_maxima(
            block_maxima,
            sage_q_scale=sage_q_scale,
            sage_k_block_scales=sage_k_block_scales,
        )
        row_max = cute.math.max(row_max, tile_max)
        tmem_sp_sdata[id(self)] = s_data
        row_max_safe = row_max
        if row_max == -Float32.inf:
            row_max_safe = Float32(0.0)
        return old_row_max, row_max_safe

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def sage_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        sage_q_scale: Float32,
        sage_k_words: cutlass.Vector,
        section: cutlass.Constexpr[WorkStage] = WorkStage.Loop,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Sage row max of block-scaled scores (``SageScoreScaling.BLOCK``).

        ``sage_q_scale`` is the work tile's cached ``sfQ`` (``cache_sage_scales``)
        and ``sage_k_words`` the tile's ``sfK`` words, which the P pass reuses;
        each 16-column block takes its word's scale. The maximum is of the
        dequantized scores; the stashed scores stay raw. ``section=Tail``
        masks the zero-filled lanes of the dense tail tile.
        """
        sage_k_block_scales = self.sage_block_scales(sage_k_words)
        if cutlass.const_expr(
            section == WorkStage.Loop and self.cfg.sage_uses_ldtm_stat
        ):
            return self.sage_load_s_chunks_and_reduce_row_max(
                stage_info,
                row_max,
                sage_q_scale=sage_q_scale,
                sage_k_block_scales=sage_k_block_scales,
            )
        s_data = self.load_s_chunks(stage_info)
        if cutlass.const_expr(section == WorkStage.Tail):
            s_data = self.mask_fixed_dense_k_tail(stage_info, s_data, section=section)
        return self.reduce_row_max(
            s_data,
            row_max,
            sage_q_scale=sage_q_scale,
            sage_k_block_scales=sage_k_block_scales,
        )

    @cute.jit
    def sage_scale_block_maxima(
        self,
        block_maxima: tuple,
        *,
        sage_q_scale: Float32,
        sage_k_block_scales: cutlass.Vector,
    ) -> Float32:
        """Return the tile's dequantized maximum from its raw 16-column block maxima."""
        scaled_block_maxima: tuple[Any, ...] = ()
        for block_idx in cutlass.range_constexpr(self.cfg.sage_score_blocks):
            block_max = block_maxima[block_idx]
            if cutlass.const_expr(self.cfg.uses_int32_scores):
                # Exact: biased scores have unit spacing.
                block_max = block_max - Float32(INT32_SCORE_BIAS)
            scaled_block_maxima += (
                block_max * Float32(sage_k_block_scales[block_idx]),
            )
        tile_max = cutlass.Vector.from_elements(
            scaled_block_maxima, self.cfg.qk_acc_dtype
        ).reduce("max")
        return tile_max * sage_q_scale

    @cute.jit
    def score_seed_words(self, context: Any) -> cutlass.Array:
        """Return the seed MMA's operand tile; the second instance reads the first's."""
        owner = self if self.score_seed_owner is None else self.score_seed_owner
        return sage_score_seed_words(
            context.smem_base,
            owner.seed_alloc.offset,
            self.cfg.sage_score_seed_tile_bytes,
        )

    @producer_work
    @cute.jit
    def seed_scores(self, stage_info: StageInfo) -> None:
        """Write ``INT32_SCORE_BIAS`` into the acquired S stage before its K wait."""
        issue_sage_score_seed_mma(
            self.tmem_ptr_s_cached.subview(self.stage_col_offset(stage_info)),
            self.score_seed_words(stage_info.context),
            self.cfg.qk_mma_tiler[0],
            self.cfg.qk_mma_tiler[1],
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=old_row_max)
    @cute.jit
    def softmax_aux_identity(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> SoftmaxScalar:
        """Auxiliary identity path (no P-chunk reduction)."""
        _ = stage_info
        return row_max

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=row_sum)
    @cute.jit
    def softmax_aux_reduce(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        p_chunk: SoftmaxRowSumContribution,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxScalar:
        """Accumulate row_sum from vector P fragments or their scalar sum."""
        _ = stage_info
        if cutlass.const_expr(self.cfg.vc_attention):
            # VC-Attention keeps the row max in log2 units.
            scale_softmax_log2 = Float32(1.0)
        if cutlass.const_expr(self.enable_early_tile_sum):
            acc_scale = cute.math.exp2(
                scale_softmax_log2 * (old_row_max - row_max),
                fastmath=True,
            )
            return row_sum * acc_scale + p_chunk
        return self.row_sum_reduction(
            old_row_max=old_row_max,
            row_max=row_max,
            row_sum=row_sum,
            p_chunk=p_chunk,
            scale_softmax_log2=scale_softmax_log2,
        )

    @cute.jit
    def stage_col_offset(self, stage_info: StageInfo) -> Int32 | int:
        """Return the TMEM column offset for a pipelined S/P stage."""
        stage_col_offset = Int32(0)
        if cutlass.const_expr(self.cfg.mma_softmax_stage > 1):
            stage_col_offset = stage_info.stage_idx * self.cfg.qk_mma_tiler[1]
        return stage_col_offset

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def store_p(self, stage_info: StageInfo) -> None:
        """Store the P row from exp2_p into SMEM after the P-tile acquire; the
        proxy fence orders the stores before the UMMA reads them. Auxiliary work
        because the S stage is already released and P goes to the SMEM tile."""
        self.store_p_row_smem(stage_info, tmem_sp_pwords.pop(id(self)))
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )

    @cute.jit
    def store_p_row_smem(
        self, stage_info: StageInfo, packed_words: tuple[Any, ...]
    ) -> None:
        """Write this thread's 128-byte P row in the SW128 K-major layout the PV
        descriptor expects: 16-byte chunk c of row r lands at chunk c xor (r mod 8)."""
        assert self.smem_p is not None
        context = stage_info.context
        assert context is not None and context.smem_base is not None
        view = cutlass.Array(
            context.smem_base.data_ptr() + self.smem_p.alloc.offset,
            dtype=Int32,
            shape=(self.cfg.smem_p_bytes // 4,),
            addrspace=3,
        )
        # One P row per softmax thread, indexed like its TMEM lane.
        warp_id_in_sg = cute.arch.warp_idx() % len(self.cfg.softmax0_warp_ids)
        row = Int32(warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx())
        row_words = row * Int32(32)
        swz = row & Int32(7)
        for chunk_idx in cutlass.range_constexpr(len(packed_words) // 4):
            chunk_words = cutlass.Vector.from_elements(
                packed_words[chunk_idx * 4 : chunk_idx * 4 + 4], Int32
            )
            phys_chunk = Int32(chunk_idx) ^ swz
            view.subview(row_words + phys_chunk * Int32(4)).data_ptr().store(
                chunk_words, alignment=16
            )

    @cute.jit
    def store_unmasked_tile_range(self, stage_info: StageInfo, packed: Int32) -> None:
        """Store this warp's packed range; every lane writes the same value."""
        words, warp = self.mask_range_word(stage_info)
        words[warp] = packed

    @cute.jit
    def store_vc_rowsum_group(self, stage_info: StageInfo, sums: cutlass.Array) -> None:
        """Write the group's row sums as bf16 hi/lo pairs: tile ``8o+i`` in K slots
        ``2i``, ``2i+1`` of operand ``o``, as one 16-byte store per core matrix
        (slots 0-7 in the first, 8-15 in the next).

        The split truncates and the low half carries the remainder, so each
        pair is exact to 2^-16 and runs on the integer pipe.
        """
        view = cutlass.Array(
            self.vc_rowsum_view(stage_info).data_ptr(),
            dtype=Int32,
            shape=(self.cfg.vc_rowsum_tile_bytes // 4,),
            addrspace=3,
        )
        base = self.vc_row_base() >> Int32(1)
        tiles_per_operand = VC_MEAN_MMA_K // 2
        for operand in cutlass.range_constexpr(VC_MEAN_OPERANDS):
            words = [
                split_bf16_hi_lo_word(sums[operand * tiles_per_operand + i])
                for i in range(tiles_per_operand)
            ]
            row = base + Int32(operand * self.cfg.vc_rowsum_operand_bytes // 4)
            for half in cutlass.range_constexpr(2):
                vec = cutlass.Vector.from_elements(
                    tuple(words[half * 4 : half * 4 + 4]), Int32
                )
                view.subview(row + Int32(half * 32)).data_ptr().store(vec, alignment=16)

    @cute.jit
    def unmasked_tile_range(
        self,
        stage_info: StageInfo,
        *,
        q_offset: Int32 = 0,
        window_start: Int32 = 0,
        window_end: Int32 = 0,
    ) -> Int32:
        """Return this warp's packed LOOP tiles ``[begin, end)`` needing no mask.

        Tiles left of the range cross some row's left bound and tiles from
        ``end`` on cross some row's right bound; the per-tile branch is
        warp-uniform.
        """
        lo, hi = self.loop_visible_k_interval(
            stage_info,
            q_offset=Int32(q_offset),
            window_start=Int32(window_start),
            window_end=Int32(window_end),
        )
        begin, end = unmasked_kv_tile_range(lo, hi, Int32(self.cfg.kv_tile_n))
        return pack_kv_tile_range(begin, end)

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def variable_window_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        window_start: Int32,
        window_end: Int32,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Mask S using inclusive per-row VariableWindow bounds.

        Only LOOP tiles outside the warp's cached unmasked range are masked.
        """
        s_data = self.load_s_chunks(stage_info)
        if self.loop_tile_needs_mask(stage_info):
            s_data = self.mask_variable_window_chunks(
                stage_info, s_data, window_start, window_end
            )
        return self.reduce_row_max(s_data, row_max)

    @cute.jit
    def varlen_batch_coord(self, stage_info: StageInfo) -> Int32:
        """Return the batch coordinate for the active tile-order policy."""
        _, _, batch_coord = resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        return batch_coord

    @consumer_work(returns=(old_row_max, row_max, vc_k_scale_next))
    @cute.jit
    def vc_compute_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        vc_row_scale: SoftmaxScalar,
        vc_k_scale_next: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar, Float32]:
        """VC-Attention K-loop row max in log2 units. Under QK8 the tile maximum is
        scaled by ``vc_row_scale * sfK[tile]`` (prefetched) and the next tile's
        ``sfK`` is read."""
        tile_scale = vc_row_scale
        k_scale_after = vc_k_scale_next
        if cutlass.const_expr(self.cfg.vc_qk8):
            k_scale_after = self.vc_k_scales.next_scale(
                stage_info, Int32(stage_info.loop_offset)
            )
            tile_scale = vc_row_scale * vc_k_scale_next
        if cutlass.const_expr(self.cfg.uses_ldtm_stat):
            old_row_max, new_row_max = self.load_s_chunks_and_reduce_row_max(
                stage_info, row_max, tile_scale=tile_scale
            )
        else:
            s_data = self.load_s_chunks(stage_info)
            old_row_max, new_row_max = self.reduce_row_max(s_data, row_max, tile_scale)
        return old_row_max, new_row_max, k_scale_after

    @cute.jit
    def vc_coords(self, stage_info: StageInfo) -> tuple[Int32, Int32, Int32]:
        """Return ``(batch, head, q_row)`` of this softmax thread's row."""
        seq_coord, head_coord, batch_coord = resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        warp_id_in_sg = cute.arch.warp_idx() % len(self.cfg.softmax0_warp_ids)
        row_in_tile = warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx()
        row = (
            seq_coord * self.cfg.q_tile_m * self.cfg.work_tile_q_seq_tiles
            + self.q_half * self.cfg.peer_q_seq_tile_stride * self.cfg.q_tile_m
            + row_in_tile
        )
        head = head_coord * self.cfg.work_tile_q_heads
        return batch_coord, head, row

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def vc_fixed_dense_k_tail_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        vc_row_scale: SoftmaxScalar,
        vc_k_scale_next: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """VC-Attention dense tail: mask the zero-filled lanes of the partial last
        K/V tile before the row max; under QK8 the tile's prefetched ``sfK`` scales it."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        tile_scale = vc_row_scale
        if cutlass.const_expr(self.cfg.vc_qk8):
            tile_scale = vc_row_scale * vc_k_scale_next
        s_data = self.load_s_chunks(stage_info)
        neg_inf = cutlass.vector.full(
            [tmem_x],
            self.cfg.qk_acc_dtype(-Float32.inf),
            dtype=self.cfg.qk_acc_dtype,
        )
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            valid_in_chunk = cute.math.min(
                cute.math.max(
                    self.k_tail_rows - Int32(chunk_idx * tmem_x),
                    Int32(0),
                ),
                Int32(tmem_x),
            )
            mask = cutlass.vector.create_mask([tmem_x], [valid_in_chunk])
            s_data[chunk_idx] = cutlass.vector.where(mask, s_data[chunk_idx], neg_inf)
        return self.reduce_row_max(s_data, row_max, tile_scale)

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            vc_row_scale,
            vc_prev_tile_sum,
            vc_meansums,
            vc_kept_row_sum,
            vc_k_scale_next,
        ),
    )
    @cute.jit
    def vc_init_row_scale(self, stage_info: StageInfo) -> tuple[Float32, ...]:
        """VC-Attention work-tile setup. Returns the softmax scale in log2 units,
        a zero previous-tile row sum, the zeroed group row sum array, a zero kept row sum
        (V repair) and tile 0's sfK.

        Under VC-Attention-QK8 the row scale also carries this row's sfQ, and the
        group's SMEM sfK table is filled with one coalesced pass. Both named
        barrier syncs of the fill cover the group's four warps: the first keeps
        a fast warp from refilling while a peer still reads the previous work
        tile's entries, the second publishes the new table.
        """
        row_scale = self.scale_softmax_log2
        k_scale_next = Float32(1.0)
        if cutlass.const_expr(self.cfg.vc_qk8):
            batch_coord, head, row = self.vc_coords(stage_info)
            kv_head = head // self.cfg.h_r
            q_slot = vc_flat_q_slot(
                batch_coord,
                row,
                self.cfg.uniform_seq_len_q,
                self.cfg.kv_tile_n.bit_length() - 1,
            )
            row_scale = row_scale * Float32(self.vc_q_scale[(head, q_slot)])
            k_base = vc_flat_k_base(
                batch_coord, self.cfg.uniform_seq_len_k, self.cfg.kv_tile_n
            )
            num_tiles = (
                self.cfg.uniform_seq_len_k + self.cfg.kv_tile_n - 1
            ) // self.cfg.kv_tile_n
            k_scale_next = self.vc_k_scales.fill(
                stage_info, self.vc_k_scale, kv_head, k_base, num_tiles
            )
        sums = cutlass.Array(
            Float32, VC_MEAN_GROUP_TILES, space=cutlass.AddressSpace.rmem
        )
        for i in cutlass.range_constexpr(VC_MEAN_GROUP_TILES):
            sums[i] = Float32(0.0)
        return (row_scale, Float32(0.0), sums, Float32(0.0), k_scale_next)

    @cute.jit
    def vc_row_base(self) -> Int32:
        """Element offset of this thread's row inside the row-sum operand."""
        warp_id_in_sg = cute.arch.warp_idx() % len(self.cfg.softmax0_warp_ids)
        row = Int32(warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx())
        return (row >> Int32(3)) * Int32(128) + (row & Int32(7)) * Int32(8)

    @cute.jit
    def vc_rowsum_view(self, stage_info: StageInfo) -> cutlass.Array:
        """This group's bf16 row-sum operand: K-major 32-byte rows in 8-row core
        matrices (``k`` 0-7 in the first 128 bytes of a group, 8-15 in the next)."""
        assert self.smem_p is not None
        context = stage_info.context
        return cutlass.Array(
            context.smem_base.data_ptr()
            + self.smem_p.alloc.offset
            + self.smem_p.rowsum_tile_offset,
            dtype=cutlass.BFloat16,
            shape=(self.cfg.vc_rowsum_tile_bytes // 2,),
            addrspace=3,
        )

    @consumer_work(
        returns=(
            vc_prev_tile_sum,
            row_sum,
            vc_meansums,
        )
    )
    @cute.jit
    def vc_store_p(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        vc_prev_tile_sum: SoftmaxScalar,
        p_chunk: SoftmaxRowSumContribution,
        vc_meansums: cutlass.Array,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> tuple[Float32, ...]:
        """VC-Attention P store. Writes the P row, rescales the group row sums
        by this tile's max correction and slots in the previous tile's sum. Every
        8th tile writes the completed group operand under the same P-ready
        handoff. Returns this tile's sum, the updated row sum and the group sums.
        """
        self.store_p_row_smem(stage_info, tmem_sp_pwords.pop(id(self)))
        acc_scale = cute.math.exp2(old_row_max - row_max, fastmath=True)
        # The peeled tail entry carries the loop offset of the previous tile.
        if cutlass.const_expr(is_tail):
            tile = Int32(stage_info.loop_end)
        else:
            tile = Int32(stage_info.loop_offset)
        slot = (tile - Int32(1)) & Int32(VC_MEAN_GROUP_TILES - 1)
        new_sum = vc_prev_tile_sum * acc_scale
        for i in cutlass.range_constexpr(VC_MEAN_GROUP_TILES):
            scaled = vc_meansums[i] * acc_scale
            if slot == Int32(i):
                scaled = new_sum
            vc_meansums[i] = scaled
        keep = Float32(1.0)
        if ((tile & Int32(VC_MEAN_GROUP_TILES - 1)) == Int32(0)) & (tile > Int32(0)):
            self.store_vc_rowsum_group(stage_info, vc_meansums)
            keep = Float32(0.0)
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )
        for i in cutlass.range_constexpr(VC_MEAN_GROUP_TILES):
            vc_meansums[i] = vc_meansums[i] * keep
        return (p_chunk, row_sum * acc_scale + p_chunk, vc_meansums)

    @consumer_work(returns=(vc_kept_row_sum, row_sum))
    @cute.jit
    def vc_store_p_repair(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        vc_kept_row_sum: SoftmaxScalar,
        p_chunk: SoftmaxRowSumContribution,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> tuple[Float32, Float32]:
        """VC-Attention V repair P store. The last ``vc_repair_tiles`` of the
        loop domain stay out of the kept row sum that normalizes the output."""
        self.store_p_row_smem(stage_info, tmem_sp_pwords.pop(id(self)))
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )
        acc_scale = cute.math.exp2(old_row_max - row_max, fastmath=True)
        kept_chunk = p_chunk
        if cutlass.const_expr(not is_tail):
            first_repair = Int32(stage_info.loop_end) - Int32(self.vc_repair_tiles)
            if Int32(stage_info.loop_offset) >= first_repair:
                kept_chunk = Float32(0.0)
        return (
            vc_kept_row_sum * acc_scale + kept_chunk,
            row_sum * acc_scale + p_chunk,
        )

    @consumer_work
    @cute.jit
    def vc_store_rowsum_final(
        self,
        stage_info: StageInfo,
        *,
        vc_prev_tile_sum: SoftmaxScalar,
        vc_meansums: cutlass.Array,
    ) -> None:
        """VC-Attention: slot the last tile's row sum into its group and
        store the group operands for the tail mean step."""
        last_tile = Int32(stage_info.loop_end)
        if cutlass.const_expr(not self.uses_fixed_dense_k_tail_mask):
            last_tile = last_tile - Int32(1)
        slot = last_tile & Int32(VC_MEAN_GROUP_TILES - 1)
        for i in cutlass.range_constexpr(VC_MEAN_GROUP_TILES):
            value = vc_meansums[i]
            if slot == Int32(i):
                value = vc_prev_tile_sum
            vc_meansums[i] = value
        self.store_vc_rowsum_group(stage_info, vc_meansums)
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )

    @cute.jit
    def zero_vc_rowsum_row(self, stage_info: StageInfo) -> None:
        """Clear this thread's 32-byte row of every row-sum operand once."""
        view = self.vc_rowsum_view(stage_info)
        base = self.vc_row_base()
        zero = cutlass.BFloat16(0.0)
        zeros = cutlass.Vector.from_elements((zero,) * 8, cutlass.BFloat16)
        for operand in cutlass.range_constexpr(VC_MEAN_OPERANDS):
            row = base + Int32(operand * self.cfg.vc_rowsum_operand_bytes // 2)
            view.subview(row).data_ptr().store(zeros, alignment=16)
            view.subview(row + Int32(64)).data_ptr().store(zeros, alignment=16)


@cute.jit
def bmsk_clamp(start: Int32, width: Int32) -> Int32:
    """Create a contiguous 32-bit mask with clamped bounds."""
    return cute.arch.inline_ptx(
        "bmsk.clamp.b32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Int32],
        read_only_args=[start, width],
    )


@cute.jit
def exp2_fma_packed(x0: Float32, x1: Float32) -> tuple[Float32, Float32]:
    """exp2 of two fp32 values on the FMA pipe, 2^x = 2^floor(x) * poly(x - floor(x)).
    Same routine as ex2_emulation_f32x2_value in the SM110 GQA decode kernel."""
    # Below -127 the result flushes to zero in e4m3 anyway.
    x0 = cute.math.max(x0, Float32(-127.0), ftz=True)
    x1 = cute.math.max(x1, Float32(-127.0), ftz=True)
    # Adding 1.5 * 2^23 with round-down leaves floor(x) in the low mantissa bits.
    bias = Float32(1.5 * 2.0**23)
    t = cute.arch.add_packed_f32x2((x0, x1), (bias, bias), rnd="rm", ftz=False)
    n = cute.arch.add_packed_f32x2(t, (-bias, -bias), rnd="rn", ftz=False)
    f = cute.arch.fma_packed_f32x2(
        n, (Float32(-1.0), Float32(-1.0)), (x0, x1), rnd="rn", ftz=False
    )
    # Degree-3 fit of 2^f on [0, 1).
    c1 = Float32(0.695146143436431884765625)
    c2 = Float32(0.227564394474029541015625)
    c3 = Float32(0.077119089663028717041015625)
    p = cute.arch.fma_packed_f32x2(f, (c3, c3), (c2, c2), rnd="rn", ftz=False)
    p = cute.arch.fma_packed_f32x2(p, f, (c1, c1), rnd="rn", ftz=False)
    p = cute.arch.fma_packed_f32x2(
        p, f, (Float32(1.0), Float32(1.0)), rnd="rn", ftz=False
    )
    # The bias has zero low bits, so bits(t) << 23 is floor(x) << 23.
    return (
        f32_from_bits(f32_bits(p[0]) + (f32_bits(t[0]) << 23)),
        f32_from_bits(f32_bits(p[1]) + (f32_bits(t[1]) << 23)),
    )


@cute.jit
def f32_bits(x: Float32) -> Int32:
    return cutlass.Vector.from_elements((x,), Float32).bitcast(Int32)[0]


@cute.jit
def f32_from_bits(x: Int32) -> Float32:
    return cutlass.Vector.from_elements((x,), Int32).bitcast(Float32)[0]


@cute.jit
def mask_score_quad(
    valid_bits: Int32,
    score0: Float32,
    score1: Float32,
    score2: Float32,
    score3: Float32,
    first_bit: cutlass.Constexpr[int] = 0,
) -> tuple[Float32, Float32, Float32, Float32]:
    """Expand bits ``first_bit + [0, 4)`` with setp and replace invalid scores."""
    compare = "".join(
        f"and.b32 bit, {{$r0}}, {1 << (first_bit + i):#x};\n"
        f"setp.ne.u32 valid{i}, bit, 0;\n"
        for i in range(4)
    )
    return cute.arch.inline_ptx(
        """
        {
            .reg .pred valid<4>;
            .reg .b32 bit;
            mov.b32 {$w0}, {$r1};
            mov.b32 {$w1}, {$r2};
            mov.b32 {$w2}, {$r3};
            mov.b32 {$w3}, {$r4};
        """
        + compare
        + """
            @!valid0 mov.b32 {$w0}, 0xff7fffff;
            @!valid1 mov.b32 {$w1}, 0xff7fffff;
            @!valid2 mov.b32 {$w2}, 0xff7fffff;
            @!valid3 mov.b32 {$w3}, 0xff7fffff;
        }
        """,
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[valid_bits, score0, score1, score2, score3],
    )


def placeholder_softmax_chunks(cfg: Any) -> SoftmaxChunks:
    """Build zero P chunks with the same structure as runtime softmax chunks."""
    try:
        tmem_x = cfg.tmem_x_load_s
        num_chunks = cfg.qk_mma_tiler[1] // tmem_x
        chunks = []
        for _ in range(num_chunks):
            zeros = tuple(cfg.qk_acc_dtype(0.0) for _ in range(tmem_x))
            chunks.append(cutlass.Vector.from_elements(zeros, cfg.qk_acc_dtype))
        return chunks
    except RuntimeError:
        return []


def placeholder_unit_vector(length: int) -> cutlass.Vector | None:
    """Build a vector of ``length`` unit scales with the runtime vector's structure."""
    try:
        return cutlass.Vector.from_elements(
            tuple(Float32(1.0) for _ in range(length)), Float32
        )
    except RuntimeError:
        return None


@cute.jit
def softmax_exp2_bias(row_max: Float32, scale: Float32, p_scale: Float32) -> Float32:
    """Use a zero exponent anchor for masked rows without changing their max."""
    return Float32(
        cutlass.select_(
            row_max == Float32(SOFTMAX_MASK_VALUE),
            p_scale,
            (Float32(0.0) - row_max) * scale + p_scale,
        )
    )
