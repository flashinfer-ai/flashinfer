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

"""Static configuration of the task-scheduled FMHA context kernel."""

import cutlass.cute as cute
import cutlass
import math
from cutlass import Int32
from dataclasses import dataclass
from .device.sage_scales import (
    SAGE_SCORE_BLOCK_COLS,
    sage_k_tile_bytes,
    sage_k_tile_words,
    sage_k_transport,
    sage_score_scaling,
)
from ...sage.kernel.int32_scores import sage_score_seed_tile_bytes
from ...sage.layout import SageScaleLayout
from ...sage.modes import SageKTransport, SageScoreScaling
from ...vc_attention import VC_MEAN_MMA_K, VC_MEAN_OPERANDS


# ---------------------------------------------------------------------------
# FmhaConfig -- kernel-wide configuration
# ---------------------------------------------------------------------------


@dataclass
class FmhaConfig:
    """Compile-time and runtime configuration for the FMHA kernel.

    Mirrors the attributes from BlackwellFusedMultiHeadAttentionForward.__init__
    and _setup_attributes, collected into a single portable dataclass.

    All fields are marked Constexpr so that tree_flatten does not try to
    recursively extract MLIR values from dtype classes or plain Python ints/tuples.
    """

    # Data types
    q_dtype: type | None = None
    k_dtype: type | None = None
    v_dtype: type | None = None
    o_dtype: type | None = None
    # The dtype the softmax reads S as. INT8 Q/K accumulate INT32 scores onto
    # ``INT32_SCORE_BIAS``, whose words are the same FP32 values.
    qk_acc_dtype: type | None = None
    pv_acc_dtype: type | None = None

    # Tile shapes
    logical_head_dim_qk: int = 128
    head_dim_per_stage_o: int = 128
    qk_mma_tiler: tuple[int, int, int] = (128, 128, 64)
    pv_mma_tiler: tuple[int, int, int] = (128, 64, 128)
    epi_tile: tuple[int, int] = (128, 64)
    # Number of interleaved Q/KV/O instances per CTA: two selects the paired
    # schedule, while one selects the single-instance schedule used for D>128.
    num_qkv_instances: int = 2

    # Pipeline stages
    q_stage: int = 2
    kv_stage: int = 3
    # Hand the S0/S1 pacing token back before the exp2/P work on the fp8 P-in-SMEM path
    # (as the TMEM-P fp8 cadence does), so the peer softmax group starts its row max
    # while this group computes P.
    fp8_psmem_early_token: bool = False
    # K/V ring depth. 0 derives it from the shared-memory budget.
    kv_ring_stages: int = 0
    # K=64 per fp8 PV UMMA instruction through bit 29 of the instruction
    # descriptor, halving the PV issue count.
    umma_fp8_k64: bool = False
    # One TMA pipeline has one expected-transaction byte count per stage, so K
    # and V share a ring of kv_stage stages only while their dtype widths
    # match. Mixed widths set split_kv_pipelines and size one ring per side.
    split_kv_pipelines: bool = False
    kv_stage_k: int = 3
    kv_stage_v: int = 3
    mma_softmax_stage: int = 1
    # Stage fp8 P in SMEM so softmax releases the S stage right after loading it.
    p_in_smem: bool = False
    # Use the two-stage loop-carried S/P schedule and an independent P-ready
    # handoff, allowing QK(i+1) and PV(i) to operate on opposite S/P stages.
    has_tmem_p_pipeline: bool = False
    stats_via_smem: bool = False
    stage_scoped_tmem_stats: bool = False
    softmax_corr_stage: int = 1
    mma_corr_stage: int = 2

    # TMA copy granularity
    tma_copy_qkv_iters: int = 1
    tma_copy_q_granu_inner: int = 128
    tma_copy_q_elements: int = 0
    tma_copy_q_granu_elems: int = 0
    tma_copy_q_bytes: int = 0
    tma_copy_kv_granu_inner: int = 128
    tma_copy_kv_elements: int = 0
    tma_copy_kv_granu_elems: int = 0
    tma_copy_kv_bytes: int = 0
    # V-specific TMA copy granularity. V may use a different dtype than K,
    # e.g. QK-BF16/PV-FP8, but these fields are otherwise ignored.
    tma_copy_v_iters: int = 1
    tma_copy_v_granu_inner: int = 128
    tma_copy_v_stage_iters: int = 0
    tma_copy_v_granu_elems: int = 0
    tma_copy_v_bytes: int = 0
    tma_copy_o_iters: int = 1
    tma_copy_o_granu_inner: int = 0
    tma_copy_o_elements: int = 0
    tma_copy_o_granu_elems: int = 0
    q_tile_m: int = 128
    kv_tile_n: int = 128

    # Warp assignments
    softmax0_warp_ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    softmax1_warp_ids: tuple[int, int, int, int] = (4, 5, 6, 7)
    correction_warp_ids: tuple[int, int, int, int] = (8, 9, 10, 11)
    mma_warp_id: int = 12
    load_warp_id: int = 13
    epilogue_warp_id: int = 14
    empty_warp_id: int = 15

    # Register budgets
    num_regs_softmax: int = 192
    num_regs_correction: int = 96
    num_regs_other: int = 32

    # TMEM layout
    tmem_alloc_cols: int = 512
    tmem_stats_cols: int = 4
    tmem_s0_offset: int = 0
    tmem_s1_offset: int = 128
    tmem_o0_offset: int = 256
    tmem_o1_offset: int = 384
    tmem_p0_offset: int = 32
    tmem_p1_offset: int = 160
    tmem_vec0_offset: int = 0
    tmem_vec1_offset: int = 128

    # SMEM shapes (set during __init__ of FmhaTs)
    sO_stage_elements: int = 0
    sQ_shape: tuple[int, int] = (2, 0)
    sK_shape: tuple[int, int] = (3, 0)

    # Misc
    buffer_align_bytes: int = 1024
    tmem_bar_id: int = 2
    cluster_shape_mn: tuple[int, int] = (1, 1)
    block_warps: int = 16

    # GQA: head ratio h_q // h_kv (1 = MHA, >1 = GQA)
    h_r: int = 1

    # Causal masking: when True, mask out positions where k_idx > q_idx
    is_causal: bool = False
    # Explicit packed-Q inclusive [start, end] bounds replace static masks.
    has_variable_window: bool = False
    # Causal balancing uses head_batch_seq logical tile order and reverses Q
    # sequence tiles.
    balance_causal_workload: bool = False
    num_seq_tiles: int | Int32 = 0
    # Skip correction optimization: when True, skip rescale if old_max == new_max
    enable_skip_correction: bool = True
    # Keep the running row max while a tile raises it by at most this many log2 units.
    corr_skip_threshold_log2: float = 0.0
    two_cta_umma: bool = False
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    # exp2 pairs per 16-pair softmax chunk computed on the FMA pipe.
    exp2_fma_pairs: int = 0
    # Softmax exp2 as one f16x2 MUFU op per score pair.
    exp2_packed_f16x2: bool = False
    # VC-Attention mode on the E4M3 V kernel with 16-bit or E4M3 Q/K. V stores
    # residuals around a per-tile mean and the output scale is a per-channel
    # table. In "means" mode the kernel adds each tile mean back into the
    # output with one UMMA step. In "outliers" mode the kernel skips the means
    # and instead consumes extra K/V tiles of the largest residuals that
    # follow the sequence. None turns VC-Attention off.
    vc_attention: str | None = None
    # Sage attention: the prepared scale buffer's layout, else None. The
    # softmax dequantizes each 16-column score block by its ``sfK`` word and
    # the row's ``sfQ`` word, and the epilogue scales each output column by
    # ``sfV`` and adds ``v_mean``.
    sage_scale_layout: SageScaleLayout | None = None
    # Stages of the load warp's ``sfK`` ring (``SageKTransport.SMEM``): the K
    # tiles the shared K/V ring keeps in flight, so the ring never throttles
    # the K loads.
    sage_k_scale_stages: int = 4
    # MXFP8 Q/K, selected by a Float8E8M0FNU Q/K input dtype. The QK UMMA is the
    # block-scaled kind with one E8M0 scale per 32 channels of the E4M3
    # operands. The scale atoms ride the Q and K loads and the MMA warp copies
    # them into tensor memory before each QK step.
    use_mxfp8_qk: bool = False

    # Variable sequence length mode stores Q/K/V/O as flattened
    # [sum_seqlen, head, dim] tensors and uses cum_seqlen_* for per-batch
    # sequence offsets.
    has_varlen: bool = False
    # Uniform packed plans retain the ragged tensor-map ABI but derive their
    # cumulative offsets arithmetically, avoiding dependent indptr loads on
    # every persistent work tile.
    # VC-Attention-QK8 fixed plans also set the two lengths while
    # has_uniform_varlen stays False. There they are the Q tokens and the K/V
    # rows of the fixed plan, the rows including the repair tiles, and they fold
    # the flat scale table slots at compile time so the P store addresses stay
    # out of the loop.
    has_uniform_varlen: bool = False
    uniform_seq_len_q: int = 0
    uniform_seq_len_k: int = 0

    # When true, map each work tile to two Q heads sharing one K/V head.
    # This enables the grouped-query/sliding-window context flavor while
    # reusing the unified FMHA context implementation.
    head_paired: bool = False

    # Concrete kernel policy derived from paired geometry and V dtype.  The
    # resource consumes this flag directly so the row-sum algorithm and the
    # register budget selected by FmhaTs cannot diverge.
    enable_early_tile_sum: bool = False

    seq_tile_n: int = 128
    tmem_x_load_s: int = 32
    # Opt-in to `tcgen05.ld.red.max` (LDTM.STAT) for the non-masked
    # `compute_row_max` path: fuses the per-chunk max into the TMEM load.
    # Requires SM103+/SM110+ with tcgen05.ld.red support; the primitive is
    # emitted with the 32dp x 32bit x 32rep shape only.
    uses_ldtm_stat: bool = False
    # Causal S_q < S_kv shifts Q rows right by q_offset = S_kv - S_q.
    # This flag selects the shifted causal mask; there is no second causal mode.
    has_q_offset: bool = False
    # Fixed causal attention with exactly one K/V tile does not need the
    # synthetic peer0 tail slot used by the general query-paired schedule.
    causal_single_kv_tile: bool = False
    window_size_left: int = 0
    # Upper bound on valid K/V rows in the final dense tile. Zero
    # means that the K/V extent is tile-aligned (or that this specialization
    # does not use the fixed dense-tail mask).
    fixed_dense_k_tail: int = 0
    # Packed-contiguous and paged dense attention normally mask scores past
    # each request's logical K length. Plans with uniform, tile-aligned K
    # lengths can compile that mask away because every K tile is fully valid.
    packed_dense_k_mask: bool = True

    # ------------------------------------------------------------------
    # Paged KV cache (vLLM-style logical->physical page indirection)
    #
    # When use_paged_kv is True, K/V live in a fixed-size page pool
    # [num_pages_in_pool, h_kv, num_tokens_per_page, d] and the kernel follows
    # a fixed row-strided block table to resolve logical (b, s) -> physical page
    # id at TMA-issue time.
    #
    # Staged D256 assigns page-offset prefetch to its empty/padding warp.
    # Paired D128 reads page IDs directly from the page table in its load task.
    # ------------------------------------------------------------------
    use_paged_kv: bool = False
    # The caller guarantees that request-invalid rows in every active final V
    # page contain zero. This lets consumers omit the defensive post-TMA clear.
    paged_v_tail_is_zero: bool = False
    # D256 uses a single Q/KV instance and can issue the final O TMA store
    # from one correction warp after the four-warp correction group has
    # staged O.  This frees the standalone epilogue warp for scheduling.
    fuse_epilogue_into_correction: bool = False
    num_tokens_per_page: int = 32
    # Static upper bound derived from max_kv_len during kernel construction.
    # Runtime active-page bounds come from seq_lens_kv.
    max_num_pages_per_seq_kv: int = 1
    page_offsets_num_warps: int = 1
    # Selected internally from the staged topology, static page geometry, and
    # exact SMEM capacity. It is derived during kernel construction and is not
    # a public tuning input.
    page_table_window_entries: int = 32

    @property
    def cta_group_size(self) -> int:
        """CTAs cooperating on one UMMA: 2 in the two-CTA form, else 1."""
        return 2 if self.two_cta_umma else 1

    @property
    def cta_tiler(self) -> tuple[int, int, int]:
        """Derive the CTA tile from the MMA tile and work-tile mapping."""
        if self.single_qkv_instance or self.head_paired:
            return self.qk_mma_tiler
        return (
            self.num_qkv_instances * self.qk_mma_tiler[0],
            self.qk_mma_tiler[1],
            self.qk_mma_tiler[2],
        )

    @property
    def gmem_o_store_wait_after_write(self) -> bool:
        """Return whether each O store must wait for the matching SMEM write."""
        return self.head_paired or self.stage_o_by_head_dim

    @property
    def has_tile_aligned_uniform_q_offset(self) -> bool:
        """Whether a uniform causal shift preserves K/V tile boundaries.

        Uniform packed plans retain fixed Q/K lengths under their replay
        contract.  When their bottom-right shift is an exact K/V-tile
        multiple, every query tile's causal diagonal has the same placement
        as the zero-offset schedule: query-paired peer 0 can use its explicit
        diagonal/invalid-tail protocol, and all other diagonals remain in
        TAIL.  No LOOP iteration then needs a causal right mask.
        """
        return (
            self.has_q_offset
            and self.has_uniform_varlen
            and self.uniform_seq_len_q % self.cta_tiler[0] == 0
            and (self.uniform_seq_len_k - self.uniform_seq_len_q) % self.kv_tile_n == 0
        )

    @property
    def kv_tile_rows_per_cta(self) -> int:
        """K rows one CTA stages per K/V tile."""
        return self.kv_tile_n // self.cta_group_size

    @property
    def kv_tile_start_window_size_left(self) -> int:
        """Return the left-window width used to compute the first K/V tile."""
        return self.window_size_left if self.head_paired else 0

    @property
    def needs_paged_v_tail_clear(self) -> bool:
        """Whether paged V tiles can contain request-invalid rows.

        A caller-owned zero-tail contract makes the clear redundant. Otherwise,
        only exact-full causal grids omit it. Requiring complete Q work tiles
        also excludes a padded final query-paired domain.
        """
        if not self.use_paged_kv or self.paged_v_tail_is_zero:
            return False
        q_work_tile_m = self.q_tile_m * self.work_tile_q_seq_tiles
        return not (
            self.has_uniform_varlen
            and self.is_causal
            and not self.has_q_offset
            and self.uniform_seq_len_k % self.kv_tile_n == 0
            and self.uniform_seq_len_q % q_work_tile_m == 0
        )

    @property
    def page_offset_pipeline_stage_counts(self) -> tuple[int, ...]:
        """Return the physical page-ID ring depths for this topology."""
        if not self.use_paged_kv or not self.single_qkv_instance:
            return ()
        # The staged schedule holds one credit for every K/V head-dimension
        # slice plus one K-ahead boundary credit.  A reused page-table window
        # needs independent K-ahead and V-delayed rings; the ordinary path
        # shares the same total number of credits. Paired D128 loads page IDs
        # directly in its load task and therefore has no physical page ring.
        k_stages = self.num_head_dim_stages_k + 1
        v_stages = self.num_head_dim_stages_v
        if self.reuses_page_table_windows:
            return (k_stages, v_stages)
        return (k_stages + v_stages,)

    @property
    def page_table_window_candidate_entries(self) -> int:
        """Return the widest page-ID window admitted by static topology.

        A split-D K/V schedule consumes two head-dimension stages for each
        logical tile.  When the static domain can cover a complete window,
        let each producer lane fetch one ID per D stage so one page-window
        handoff spans both stages. Short domains admit only the natural
        one-warp window. The kernel's capacity pass makes the final selection.
        """
        natural_entries = cute.arch.WARP_SIZE
        if (
            not self.use_paged_kv
            or self.is_causal
            or self.num_tokens_per_page <= 0
            or not (self.single_qkv_instance and self.has_tmem_p_pipeline)
        ):
            return natural_entries
        pages_per_tile = self.kv_tile_n // self.num_tokens_per_page
        staged_entries = natural_entries * self.num_head_dim_stages_k
        staged_period = staged_entries // pages_per_tile
        static_num_kv_tiles = (
            self.max_num_pages_per_seq_kv + pages_per_tile - 1
        ) // pages_per_tile
        if (
            staged_entries % pages_per_tile == 0
            and static_num_kv_tiles >= staged_period
            and static_num_kv_tiles % staged_period == 0
        ):
            return staged_entries
        return natural_entries

    @property
    def peer_q_head_stride(self) -> int:
        """Return the Q-head stride between the two peer Q/O tiles."""
        return 1 if self.head_paired and not self.single_qkv_instance else 0

    @property
    def peer_q_seq_tile_stride(self) -> int:
        """Return the Q-sequence tile stride between the two peer Q/O tiles."""
        return 0 if self.head_paired or self.single_qkv_instance else 1

    @property
    def pv_half_overlap(self) -> bool:
        """Publish P in two 64-key halves so PV can start on the first half.
        Not used when P is staged in SMEM."""
        return (
            not self.single_qkv_instance
            and self.enable_early_tile_sum
            and not self.is_causal
            and not self.has_varlen
            and not self.has_tmem_p_pipeline
            and not self.p_in_smem
            and self.v_dtype is not None
            and self.v_dtype.width in (8, 16)
            and not self.uses_d128_fp8_softmax_cadence
            and not self.uses_d256_fp8_softmax_cadence
            # The head-dim-staged PV path issues every K slice per call.
            and not self.stage_kv_by_head_dim
            # Softmax halves the QK N tile and PV halves its K slices, so the
            # two extents must match and split evenly.
            and self.qk_mma_tiler[1] == self.pv_mma_tiler[2]
            and (self.pv_mma_tiler[2] // (16 if self.v_dtype.width == 16 else 32)) % 2
            == 0
        )

    @property
    def pv_n_per_cta(self) -> int:
        """V head-dim columns one CTA stages per tile."""
        return self.pv_mma_tiler[1] // self.cta_group_size

    @property
    def pv_p_scale(self) -> float:
        """Return the P scale applied before PV MMA."""
        if self.v_dtype is not None and self.v_dtype.width == 8:
            # E4M3 saturates at 448. Exact correction keeps P <= 1, so P scales
            # by 448. Lazy correction lets the softmax row max lag the true one
            # by up to corr_skip_threshold_log2, so P <= 2^threshold. Keep a
            # power-of-two scale there. The largest with P * scale <= 448 is
            # 2^floor(log2(448) - threshold), which is 1 at threshold 8.
            if self.corr_skip_threshold_log2 == 0.0:
                return 448.0
            return 2.0 ** max(
                0.0, math.floor(math.log2(448.0) - self.corr_skip_threshold_log2)
            )
        # Non-FP8 V uses P directly, so the PV-side P scale is identity.
        return 1.0

    @property
    def pv_p_scale_log2(self) -> float:
        """Return log2(P scale) for folding into exp2 softmax P."""
        return math.log2(self.pv_p_scale)

    @property
    def reuses_page_table_windows(self) -> bool:
        """Whether dense paged-KV admits structural page-ID windows."""
        if (
            not self.use_paged_kv
            or self.is_causal
            or self.num_tokens_per_page <= 0
            or self.kv_tile_n % self.num_tokens_per_page != 0
        ):
            return False
        pages_per_tile = self.kv_tile_n // self.num_tokens_per_page
        window_entries = self.page_table_window_entries
        if (
            pages_per_tile <= 0
            # A one-tile window cannot retain distinct K/V page IDs.
            or pages_per_tile >= window_entries
            or window_entries % pages_per_tile != 0
        ):
            return False
        window_period = window_entries // pages_per_tile
        if self.single_qkv_instance and self.has_tmem_p_pipeline and window_period < 3:
            # The K-ahead/V-delayed HEAD requires distinct K0, K1, and tail
            # positions. Smaller periods retain the ordinary per-tile path.
            return False
        static_num_kv_tiles = (
            self.max_num_pages_per_seq_kv + pages_per_tile - 1
        ) // pages_per_tile
        return (
            static_num_kv_tiles >= window_period
            and static_num_kv_tiles % window_period == 0
        )

    @property
    def sage_k_tile_bytes(self) -> int:
        """Return the bytes of one K tile's ``sfK`` words, one SMEM ring stage."""
        return sage_k_tile_bytes(self.sage_scale_layout)

    @property
    def sage_k_tile_words(self) -> int:
        """Return the ``sfK`` words of one K tile."""
        return sage_k_tile_words(self.sage_scale_layout)

    @property
    def sage_k_transport(self) -> SageKTransport | None:
        """Return how the softmax obtains a K tile's ``sfK`` words (``sage_k_transport``)."""
        if not self.use_sage_attention:
            return None
        return sage_k_transport(self.sage_scale_layout)

    @property
    def sage_score_blocks(self) -> int:
        """Return the 16-column score blocks of one K tile, one ``sfK`` word each."""
        return self.qk_mma_tiler[1] // SAGE_SCORE_BLOCK_COLS

    @property
    def sage_score_scaling(self) -> SageScoreScaling | None:
        """Return how the softmax applies ``sfK`` to the scores (``sage_score_scaling``)."""
        if not self.use_sage_attention:
            return None
        return sage_score_scaling(self.sage_scale_layout)

    @property
    def sage_score_seed_tile_bytes(self) -> int:
        """Return the bytes of the BF16 operand tile that seeds INT32 scores."""
        return sage_score_seed_tile_bytes(self.qk_mma_tiler[0], self.qk_mma_tiler[1])

    @property
    def sage_uses_ldtm_stat(self) -> bool:
        """Return whether the Sage max pass takes the block maxima from LDTM.STAT pieces."""
        return (
            self.uses_ldtm_stat
            and self.sage_score_scaling is SageScoreScaling.BLOCK
            and not self.uses_int32_scores
        )

    @property
    def single_qkv_instance(self) -> bool:
        """Return whether one work tile carries a single Q/KV/O instance."""
        return self.num_qkv_instances == 1

    @property
    def skip_causal_invalid_peer0(self) -> bool:
        """Return whether query-paired causal peer0 may skip extra loop work."""
        if (
            not self.is_causal
            or self.head_paired
            or (self.has_q_offset and not self.has_tile_aligned_uniform_q_offset)
            or self.single_qkv_instance
            or self.causal_single_kv_tile
        ):
            return False
        peer0_kv_tiles = (self.q_tile_m + self.kv_tile_n - 1) // self.kv_tile_n
        paired_kv_tiles = (self.cta_tiler[0] + self.kv_tile_n - 1) // self.kv_tile_n
        extra_peer1_kv_tiles = paired_kv_tiles - peer0_kv_tiles
        if extra_peer1_kv_tiles > 2:
            raise ValueError(
                "query-paired causal scheduling supports peer1 at most two "
                "K/V tiles ahead of peer0; got "
                f"{extra_peer1_kv_tiles} extra K/V tiles"
            )
        return extra_peer1_kv_tiles > 0

    @property
    def smem_o_head_dim(self) -> int:
        """Physical O extent; the final TMA store masks channels beyond D."""
        fragment = min(1024 // self.o_dtype.width, self.epi_tile[1])
        return (self.epi_tile[1] + fragment - 1) // fragment * fragment

    @property
    def smem_p_bytes(self) -> int:
        """Bytes of one query group's SMEM P tile: q rows by k keys of the V dtype."""
        return self.qk_mma_tiler[0] * self.qk_mma_tiler[1] * self.v_dtype.width // 8

    @property
    def smem_q_head_dim(self) -> int:
        """Q storage rounded to one 128-byte TMA fragment."""
        if self.logical_head_dim_qk == 192:
            fragment_elements = 1024 // self.q_dtype.width
            return (
                (self.logical_head_dim_qk + fragment_elements - 1)
                // fragment_elements
                * fragment_elements
            )
        return self.qk_mma_tiler[2]

    @property
    def smem_v_head_dim(self) -> int:
        """Physical V extent in complete 128-byte TMA fragments."""
        fragment = min(1024 // self.v_dtype.width, self.pv_mma_tiler[1])
        return (self.pv_mma_tiler[1] + fragment - 1) // fragment * fragment

    @property
    def stages_page_offsets_in_smem(self) -> bool:
        """Whether a dedicated warp stages page IDs for the load warp.

        Staged D256 uses the coalesced SMEM page-window path. Paired D128 loads
        page IDs directly in its K/V producer.
        """
        return self.use_paged_kv and self.single_qkv_instance

    @property
    def use_sage_attention(self) -> bool:
        """Return whether the softmax and epilogue apply Sage scales."""
        return self.sage_scale_layout is not None

    @property
    def uses_causal_reversed_head_batch_seq_tile_order(self) -> bool:
        """Return whether causal head_batch_seq tiles reverse Q sequence order."""
        return self.is_causal and self.balance_causal_workload

    @property
    def uses_d128_fp8_softmax_cadence(self) -> bool:
        """Paired D128 fp8 with P in TMEM retires softmax interleaved. Only the
        shapes that do not stage P in SMEM take this path."""
        return (
            not self.single_qkv_instance
            and not self.p_in_smem
            and self.enable_early_tile_sum
            and self.q_dtype == cutlass.Float8E4M3FN
            and self.k_dtype == cutlass.Float8E4M3FN
            and self.v_dtype == cutlass.Float8E4M3FN
        )

    @property
    def uses_d256_fp8_softmax_cadence(self) -> bool:
        """Return whether staged D256 FP8 uses interleaved softmax retirement."""
        return (
            self.single_qkv_instance
            and self.has_tmem_p_pipeline
            and self.stage_kv_by_head_dim
            and self.qk_mma_tiler == (128, 128, 256)
            and self.q_dtype == cutlass.Float8E4M3FN
            and self.k_dtype == cutlass.Float8E4M3FN
            and self.v_dtype == cutlass.Float8E4M3FN
        )

    @property
    def uses_early_tile_sum(self) -> bool:
        """Return whether paired M128 geometry supports early V-tile reduction."""
        return (
            not self.single_qkv_instance
            and self.q_tile_m == 128
            and self.v_dtype
            in (
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
            )
        )

    @property
    def uses_head_batch_seq_tile_order(self) -> bool:
        """Return whether work tiles use head_batch_seq coordinates. Two-CTA
        stays seq-first so the grid padding and the cluster axis agree."""
        return self.uses_paired_fp8_head_batch_seq_tile_order or (
            self.is_causal and self.balance_causal_workload and not self.two_cta_umma
        )

    @property
    def uses_int32_scores(self) -> bool:
        """Return whether QK accumulates INT8 Q/K into INT32 scores on a float bias.

        A Sage recipe's INT8 Q/K accumulate onto the bit pattern of
        ``INT32_SCORE_BIAS``, so the softmax reads the scores as FP32 and
        removes the bias where it applies ``sfK``.
        """
        return self.q_dtype == cutlass.Int8

    @property
    def uses_paired_fp8_head_batch_seq_tile_order(self) -> bool:
        """Return whether paired FP8 benefits from head-local tile order.

        Causal work uses this order for load balancing. Dense GQA uses it to
        keep Q-head groups that share the same K/V head adjacent. Dense MHA
        has no cross-head K/V reuse and retains its sequence-local order. So
        does the two-CTA form under GQA, whose clusters must pair adjacent Q
        tiles of one head to share one K/V head.
        """
        return (
            (self.is_causal or self.h_r > 1)
            and not self.two_cta_umma
            and not self.single_qkv_instance
            and self.q_dtype is not None
            and self.k_dtype is not None
            and self.v_dtype is not None
            and self.q_dtype.width == 8
            and self.k_dtype.width == 8
            and self.v_dtype.width == 8
        )

    # Work-tile mapping for the two peer Q/O tiles handled by each CTA:
    # query-paired maps peers to two sequence tiles in one Q head, while
    # head-paired mode maps peers to two Q heads at one sequence tile.
    @property
    def vc_kscale_table_bytes(self) -> int:
        """Bytes of one softmax group's fp32 K-scale table, one word per K/V
        tile of the K/V length the kernel was built with, 16-byte padded. The
        table is sized at construction from ``max_kv_len``, before the fixed
        plan lengths reach ``cfg``."""
        max_kv_len = self.max_num_pages_per_seq_kv * self.num_tokens_per_page
        entries = max((max_kv_len + self.kv_tile_n - 1) // self.kv_tile_n, 1)
        return (entries * 4 + 15) // 16 * 16

    @property
    def vc_mean_operand_bytes(self) -> int:
        """Bytes of one bf16 [D x 16] K-major mean operand (B of one mean UMMA step)."""
        return self.pv_mma_tiler[1] * VC_MEAN_MMA_K * 2

    @property
    def vc_mean_stages(self) -> int:
        """Tile-mean ring depth: one operand per staged V tile."""
        return self.kv_stage_v if self.split_kv_pipelines else self.kv_stage_k

    @property
    def vc_mean_tile_bytes(self) -> int:
        """Bytes of the mean operands of one tile group (one ring stage)."""
        return VC_MEAN_OPERANDS * self.vc_mean_operand_bytes

    @property
    def vc_qk8(self) -> bool:
        """VC-Attention-QK8, the E4M3 or INT8 Q/K recipe with per-block dequant
        scales. INT8 scores arrive on the ``INT32_SCORE_BIAS`` float bias, which
        the row max and the exp2 remove where they apply the tile scale."""
        return self.vc_attention is not None and self.q_dtype.width == 8

    @property
    def vc_restores_means(self) -> bool:
        """Whether the kernel adds the tile means back into O."""
        return self.vc_attention == "means"

    @property
    def vc_rowsum_operand_bytes(self) -> int:
        """Bytes of one bf16 [128 x 16] K-major row-sum operand (A of one mean UMMA step)."""
        return self.qk_mma_tiler[0] * VC_MEAN_MMA_K * 2

    @property
    def vc_rowsum_tile_bytes(self) -> int:
        """Bytes of the row-sum operands of one tile group."""
        return VC_MEAN_OPERANDS * self.vc_rowsum_operand_bytes

    @property
    def work_tile_coord_indices(self) -> tuple[int, int, int]:
        """Return work-tile indices for logical ``(seq, head, batch)``."""
        if self.uses_head_batch_seq_tile_order:
            return 2, 0, 1
        return 0, 1, 2

    @property
    def work_tile_q_heads(self) -> int:
        """Return the number of Q heads represented by one work tile."""
        if self.single_qkv_instance:
            return 1
        return 2 if self.head_paired else 1

    @property
    def work_tile_q_seq_tiles(self) -> int:
        """Return the number of Q sequence tiles represented by one work tile."""
        if self.single_qkv_instance:
            return 1
        return 1 if self.head_paired else 2

    def validate_vc_profile(self) -> None:
        """Validate the VC-Attention recipe against the configured kernel."""
        if self.vc_attention is None:
            return
        if self.vc_attention not in ("means", "outliers"):
            raise ValueError(f"unknown vc_attention mode {self.vc_attention!r}")
        if (
            self.is_causal
            or self.has_variable_window
            or self.head_paired
            or self.use_paged_kv
            or self.h_r != 1
        ):
            raise ValueError(
                "VC-Attention requires the dense contiguous query-paired context "
                "kernel with equal Q and K/V head counts"
            )
        # The packed mean operand and the row-sum operand are laid out for 128
        # channels and 128 query rows.
        if self.logical_head_dim_qk != 128 or self.pv_mma_tiler[1] != 128:
            raise ValueError("VC-Attention requires head_dim 128")
        if self.q_dtype.width not in (8, 16) or self.v_dtype.width != 8:
            raise ValueError("VC-Attention requires 16-bit or E4M3 Q/K and E4M3 V")
        if not self.p_in_smem:
            raise ValueError(
                "VC-Attention requires P staged in SMEM, the dense paired D128 "
                "schedule with a 16-bit output"
            )
