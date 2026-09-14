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

"""Configuration for the throughput 2CTA MLA decode TS kernel.

The throughput 2CTA policy uses a 2CTA M128 schedule. BF16 and FP8 inputs share
the same explicit configuration structure and output-reduction contract.
"""

from dataclasses import dataclass
from typing import Tuple

from ..helpers.constants import MAX_MLA_SPLITS_KV, SUPPORTED_MLA_PAGE_SIZES
from ..helpers.mask import MaskType, normalize_mask_type


# Softmax converts natural-scale scores to exp2 with log2(e).
LOG2_E = 1.4426950408889634074

# The separate reducer follows the public MLA output contract: partial O is
# stored as BF16 while LSE and the final accumulation remain FP32.  One
# 512-thread CTA owns eight D512 rows, with each thread moving one 16-byte
# BF16 vector.  Keeping these values derived makes the row packing explicit
# and prevents the launch geometry from drifting away from the workspace
# representation.
PARTIAL_O_BITS = 16
REDUCTION_THREADS_PER_CTA = 512
REDUCTION_VECTOR_BYTES = 16
REDUCTION_VALUES_PER_THREAD = REDUCTION_VECTOR_BYTES * 8 // PARTIAL_O_BITS
REDUCTION_THREADS_PER_ROW = 512 // REDUCTION_VALUES_PER_THREAD
REDUCTION_ROWS_PER_CTA = REDUCTION_THREADS_PER_CTA // REDUCTION_THREADS_PER_ROW

# PV consumes V from SMEM in 32-token K blocks.  Physical KV pages may be
# smaller or larger, but TMA must assemble this fixed block geometry before
# tcgen05 advances the V descriptor to the next K block.
V_SMEM_K_BLOCK_TOKENS = 32

# Each transpose-TMA issue stages one 64-element slice of the latent dimension.
V_TMA_LATENT_ELEMENTS = 64


def ceil_div(a: int, b: int) -> int:
    """Return ``ceil(a / b)`` for positive integer divisors."""

    if b <= 0:
        raise ValueError(f"divisor must be positive, got {b}")
    return (a + b - 1) // b


def compute_split_kv(
    *,
    batch_size: int,
    num_q_tiles: int,
    seq_len_kv: int,
    mma_qk_tiler_mn: Tuple[int, int] = (128, 128),
    max_active_blocks: int,
) -> int:
    """Choose the throughput 2CTA split-KV count for a concrete launch.

    The heuristic tries to expose enough K-split work to fill the available CTA
    slots without creating extra partial K waves. The result is capped to keep
    the reduction grid bounded.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_q_tiles <= 0:
        raise ValueError(f"num_q_tiles must be positive, got {num_q_tiles}")
    if seq_len_kv <= 0:
        raise ValueError(f"seq_len_kv must be positive, got {seq_len_kv}")
    if max_active_blocks <= 0:
        raise ValueError(f"max_active_blocks must be positive, got {max_active_blocks}")
    if mma_qk_tiler_mn[1] <= 0:
        raise ValueError(
            f"mma_qk_tiler_mn[1] must be positive, got {mma_qk_tiler_mn[1]}"
        )

    max_splits = ceil_div(seq_len_kv, mma_qk_tiler_mn[1])
    blocks_per_batch = max(1, max_active_blocks // batch_size // (num_q_tiles * 2))
    split_heur = min(max_splits, blocks_per_batch)
    k_waves = ceil_div(max_splits, split_heur)
    split_wave_aware = ceil_div(max_splits, k_waves)
    return min(split_wave_aware, MAX_MLA_SPLITS_KV)


def compute_workspace_size(
    *,
    tile_size_q: int,
    num_q_tiles: int,
    latent_dim: int,
    batch_size: int,
    split_kv: int,
    partial_o_dtype,
    lse_dtype,
) -> int:
    """Return the physical flat-tile split-KV workspace size in bytes."""

    if split_kv == 1:
        return 0
    if tile_size_q <= 0:
        raise ValueError(f"tile_size_q must be positive, got {tile_size_q}")
    if num_q_tiles <= 0:
        raise ValueError(f"num_q_tiles must be positive, got {num_q_tiles}")
    partial_rows = batch_size * tile_size_q * num_q_tiles * split_kv
    return partial_rows * (
        latent_dim * partial_o_dtype.width // 8 + lse_dtype.width // 8
    )


@dataclass
class MlaDecodeConfig:
    """MLA decode kernel configuration.

    All values are plain Python ints/tuples so they can be used as
    ``Constexpr`` in DSL code.
    """

    # Architecture.  Dense MLA uses 512 latent channels plus 64 RoPE channels;
    # the throughput path uses one 2-CTA cluster per M128 tile.
    latent_dim: int = 512
    rope_dim: int = 64
    num_mma_ctas: int = 2
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)

    # MMA tile shapes.  QK runs over 128x128 K tiles, while PV writes the 512 V
    # head in two 256-column passes.
    mma_qk_tiler_mn: Tuple[int, int] = (128, 128)
    mma_pv_tiler_mn: Tuple[int, int] = (128, 256)
    mma_qk_tiler_k: int = 64  # = rope_dim
    mma_qk_tiler: Tuple[int, int, int] = (128, 128, 64)
    mma_qk_rope_tiler: Tuple[int, int, int] = (128, 128, 64)
    mma_pv_tiler: Tuple[int, int, int] = (128, 256, 32)

    # Iteration counts derived from the fixed dense MLA dimensions and MMA
    # tile shapes above.
    iterations_qk_latent: int = 8  # latent_dim / mma_qk_tiler_k = 512/64
    iterations_qk_rope: int = 1  # rope_dim / mma_qk_tiler_k = 64/64
    iterations_qk: int = 9  # latent + rope
    iterations_pv_k: int = 4  # mma_qk_tiler[1] / mma_pv_tiler[2] = 128/32
    iterations_pv_n: int = 2  # latent_dim / mma_pv_tiler[1] = 512/256
    # BF16 keeps the hardware-legal K64/PV-K32 transactions, but publishes two
    # consecutive transactions under one TMA/UMMA pipeline stage.  This
    # matches a 128-element head-dimension stage without requiring a TMA box
    # wider than the 128-byte swizzle permits. FP8 already uses native K128
    # QK stages and separate whole-tile K/V resources.
    kv_subtiles_per_stage: int = 2
    iterations_qk_latent_stages: int = 4
    iterations_qk_stages: int = 5
    iterations_pv_stages: int = 4

    # Pipeline stage counts for the captured schedule resources.  The combined
    # K/V stage count keeps enough delayed-V stages live for K-before-V overlap.
    load_q_stage: int = 1
    load_k_stage: int = 3
    load_v_stage: int = 2
    load_kv_stage: int = 7
    mma_s_stage: int = 2
    p_mma_stage: int = 2
    p_cor_stage: int = 2
    mma_o_stage: int = 1

    # DSV4 CSA's W9 owns an independent cp.async page-index ring.  Each
    # stage is two consecutive sparse K128 tiles (256 int32 token indices),
    # and the generated kernel keeps six such stages live.
    dsv4_page_offsets_stages: int = 6
    dsv4_page_offsets_entries_per_stage: int = 256
    dsv4_page_offsets_pair_tiles: int = 2
    # DSV4 BMM2 consumes one K128 operand for each D256 V panel.  The raw
    # Gather4 path deposits precisely that layout, so it is the required path
    # for the source-matched fused W8 schedule.
    # Raw V is valid only with the K128 P/V layout because its Gather4 SMEM
    # permutation is the source BMM2 operand ABI.  The public DSV4 entry
    # point selects that source-shaped route by default; the factory keeps a
    # false default so generic MLA callers must opt in deliberately.
    dsv4_use_raw_v_gather: bool = False
    # Select the source BMM2 K128 operand and whole-O handoff.  The public
    # DSV4 wrapper enables this; K64 x two remains an explicit diagnostic
    # fallback for isolating regressions in generic MLA configurations.
    dsv4_use_source_pv_k128: bool = False
    # Source W9/W12--W15 pair FSM.  It requires the raw K128/whole-O
    # specialization and is selected by the public DSV4 wrapper by default;
    # false remains a legacy per-tile-ring diagnostic only.
    dsv4_use_source_pair_ring: bool = False
    # Source WorkId phase-A. The public DSV4 wrapper selects its one-stage CLC
    # response pipeline and 960-thread consumer arrival protocol by default;
    # false is the explicit static grid-stride diagnostic fallback. The source
    # throttle remains a distinct W12--W15 CTA0 -> W10 edge below.
    dsv4_use_source_workid: bool = False
    # Source's separate W12--W15 -> W10 stage-1 throttle.  This must remain
    # distinct from the WorkId response pipeline and from dense W8 throttling.
    dsv4_use_source_throttle: bool = False
    # Diagnostic source S/P handoff ordering: softmax keeps an S stage until
    # after the P shared-memory store/fence. This remains separate from the
    # later direct-P experiment, which would remove the P pipeline itself.
    dsv4_release_s_after_p: bool = False
    # Diagnostic source W8 S-state machine. It keeps P's existing pipeline as
    # a safety net while matching source's pre-acquire/shifted-acquire order.
    dsv4_use_source_s_handoff: bool = False
    # Experimental source P handoff.  Retains the two physical P buffers and
    # its async-shared fence, but uses source S ownership rather than a second
    # P mbarrier pipeline.  It is valid only with the source S handoff.
    dsv4_use_source_direct_p: bool = False
    # Source Softmax -> Correction handoff.  Each K tile publishes
    # (old_max, new_max) before P materialization and the final tile publishes
    # a separate terminal (row_sum, row_max) token.  This also selects the
    # generated kernel's S/stats/O TMEM column layout.
    dsv4_use_source_early_corr: bool = False
    # Source TmemS row-max exchange: two S-stage-local 128-float buffers and
    # 64-thread W0/W2, W1/W3 named barriers.  The older full-warpgroup form
    # remains only as a diagnostic control for performance attribution.
    dsv4_use_source_softmax_pair_reduction: bool = False
    # Source TmemS/TmemP/TmemSoftmax software schedule: sequential four-way
    # max chains, FMA4/MUFU/F2FP16 overlap, direct packed-P store, and the
    # four-accumulator sum tree.  It is isolated from local PTXAS pragmas so
    # instruction order and compiler scheduling annotations can be measured
    # independently.
    dsv4_use_source_softmax_pipeline: bool = False
    # Emit TRTLLM-gen's contraction-eligible packed MUL+ADD PTX instead of a
    # directly expressed packed FMA. Both are expected to become FFMA2, but
    # their PTX SSA and final scheduling can differ.
    dsv4_use_source_contractible_ffma: bool = False
    # Source ReduceRow assigns W0/W2 and W1/W3 independent 64-thread named
    # barriers and has no post-load WAR barrier in the terminal epilogue.
    dsv4_use_source_epilogue_pair_barrier: bool = False
    # Source W9 selector copies use cp.async.cg (L1 bypass) with the
    # L2::128B prefetch-size hint.  Keep this isolated from the six-stage
    # pair-ring cadence so cache-policy A/B does not change its dataflow.
    dsv4_use_source_page_offsets_cache_policy: bool = False
    # Source CutlassCpAsyncPipeline publishes W9 with a counted
    # cp.async.mbarrier.arrive followed by one ordinary mbarrier.arrive per
    # lane.  CUTLASS DSL's AsyncLoad pipeline folds the same completion into
    # one .noinc arrive.  Keep the two lowering protocols independently
    # selectable so their SASS and performance can be compared directly.
    dsv4_use_source_page_offsets_transcnt: bool = False
    # Source TMEM exit ownership: W8 signals a 160-thread CTA named barrier,
    # W4--W7 join after the final O store, and W4 performs the two-CTA
    # cluster handshake/deallocation.  The generic graph instead drains the
    # TmemO producer pipeline and deallocates from W8.
    dsv4_use_source_tmem_lifecycle: bool = False
    # Source W9/W12--W15 exit immediately after their final page-index/Q/K/V
    # publication.  The generic TaskManager instead drains every destination
    # producer pipeline at the end of the persistent loop.  Keep source's
    # loader-only terminal policy isolated from TMEM/softmax lifecycle knobs.
    dsv4_use_source_loader_no_tail: bool = False
    # Use the generated kernel's one setmaxnreg action per warp role.  The
    # generic TS prologue first lowers all producer warps to the Gather4 floor,
    # then task entry lowers W12--W15 a second time; source relies solely on
    # the task-local 5x INC + 1x DEC protocol.
    dsv4_use_source_register_protocol: bool = False
    # Prefetch only the tensor maps consumed by the source-shaped DSV4 path:
    # Q plus the two unique raw Gather4 maps.  The generic MLA prologue warms
    # nine descriptors, eight of which belong to compile-time inactive rope,
    # transpose, or high-level sparse paths for this specialization.
    dsv4_use_active_tma_prefetch: bool = False
    # Construct the active DSV4 pipeline sync objects without their generic
    # per-resource elected-thread initialization, then initialize the unified
    # barrier block from one source-shaped W0 lane-distributed region.  This
    # lets MbarrierInitRegMapping see one CFG instead of a sequence of small
    # resource-local CFGs while preserving every pipeline's pointer/count.
    dsv4_use_source_mbarrier_init: bool = False
    # PTXAS scheduling pragmas emitted by the generated source.  Keep this a
    # separate specialization so their effect can be measured independently
    # of the logical pipeline DAG.
    dsv4_use_source_ptx_knobs: bool = False
    # Function-local PTXAS scheduling hints in TmemS/TmemP/TmemCorr.  Keep
    # these separate from the three kernel-entry global knobs: their placement
    # is part of the register dependency schedule and needs an independent A/B.
    dsv4_use_source_local_ptx_knobs: bool = False
    # Compile source's skip-correction specialization.  The threshold itself
    # remains a runtime scalar, exactly like KernelParams::mSkipCorrThreshold;
    # this flag controls the extra max-freeze and warp-vote instruction paths.
    dsv4_enable_skip_correction: bool = False
    # Public callers that do not request LSE must not pay for invisible
    # softmax-stat writes.  The flag is static because it changes the
    # epilogue instruction stream.
    stores_lse: bool = True
    # Production DSV4 uses W9-fed raw Gather4 for K.  The high-level atom is
    # retained strictly as a porting isolator for the same dataflow/SMEM tile.
    dsv4_use_raw_k_gather: bool = True

    # Base BF16 warp assignments for the 12-warp CTA. Softmax and correction
    # each own a contiguous four-warp group; the remaining warps issue MMA and
    # TMA (including register-held page IDs) or provide scheduler/alignment
    # roles. The FP8 factory extends the CTA to 16 warps for its second softmax
    # group and split QK/PV schedule.
    compute_warp_ids: Tuple[int, ...] = (0, 1, 2, 3)
    correction_warp_ids: Tuple[int, ...] = (4, 5, 6, 7)
    mma_warp_id: int = 8
    load_tma_warp_id: int = 9
    load_v_warp_id: int = 10
    load_v_num_warps: int = 1
    scheduler_warp_id: int = 10
    padding_warp_id: int = 11
    pv_mma_warp_id: int = 11
    empty_warp_ids: Tuple[int, ...] = (11,)
    second_compute_warp_ids: Tuple[int, ...] = ()
    num_softmax_groups: int = 1

    num_compute_warps: int = 4
    threads_per_warp: int = 32
    threads_per_cta: int = 384  # 12 warps * 32
    warps_in_n: int = 2

    # Register budgets passed to setmaxnreg for the high-register softmax and
    # correction groups; all other warps use the lower shared budget.
    softmax_reg_num: int = 192
    correction_reg_num: int = 208
    other_reg_num: int = 96
    # The DSV4 generated reference assigns the high producer budget to its
    # MMA/page/scheduler/padding warps, and a distinct low budget to the four
    # Gather4 TMA issuer warps.  Keep these separate: ``other_reg_num`` is
    # the prologue decrease budget, whereas each task below requests its own
    # final budget through Task Scheduling.
    producer_reg_num: int = 96
    gather4_reg_num: int = 96

    # Named barrier IDs and thread counts.  IDs are local to this kernel's
    # manual synchronization protocol and are kept away from the TMEM barrier.
    softmax_sync_bar_id: int = 2
    softmax_sync_threads: int = 128  # 4 warps * 32
    epilogue_sync_bar_id: int = 3
    epilogue_sync_threads: int = 128  # 4 warps * 32
    softmax_order_bar_0_id: int = 5
    softmax_order_bar_1_id: int = 6

    # TMEM sync barrier (for alloc/dealloc)
    tmem_sync_bar_id: int = 1
    tmem_sync_bar_threads: int = 0  # computed in make_config
    tmem_dealloc_join_bar_id: int = 12
    tmem_dealloc_join_threads: int = 160  # W8 + W4--W7

    # TMEM layout offsets.  The full 512-column TMEM budget is reserved so S,
    # O, and correction-factor columns can use fixed offsets.
    num_tmem_cols: int = 512
    tmem_o_offset: int = 0  # computed
    correction_factor_offset: int = 0  # computed

    # SMEM element counts (per-stage or total)
    smem_q_latent_elems: int = 0
    smem_q_rope_elems: int = 0
    smem_kc_elems: int = 0
    smem_vc_elems: int = 0
    smem_k_stage_elems: int = 0
    smem_v_stage_elems: int = 0
    smem_p_elems: int = 0
    softmax_exchange_elems: int = 128

    # Page geometry.  Physical page-table geometry is independent of the V
    # transpose-TMA microtile used to assemble the tcgen05 SMEM operand.
    page_size: int = 32
    kc_page_tile_size: int = 32
    v_tma_token_count: int = 32

    # Data types
    qkv_dtype: str = "bf16"
    o_dtype: str = "bf16"
    qkv_dtype_bytes: int = 2
    o_dtype_bytes: int = 2
    use_bf16_output: int = 1
    use_fp8_output: int = 0

    # TMA byte counts
    tma_copy_q_bytes: int = 0
    tma_copy_kc_bytes: int = 0
    tma_copy_vc_bytes: int = 0
    tma_copy_k_tile_bytes: int = 0
    tma_copy_v_tile_bytes: int = 0
    tma_kc_subtile_bytes: int = 0
    tma_vc_subtile_bytes: int = 0

    # Scheduling.  The runner normally supplies max_active_clusters from
    # HardwareInfo; 56 is the SM100-class fallback used by local construction
    # paths that do not query hardware.
    use_fp8_split_mma_schedule: bool = False
    use_fp8_dual_softmax_schedule: bool = False
    max_active_clusters: int = 56
    is_persistent: bool = True
    is_var_seq: bool = False
    # Use block_split_kvs[batch] as a per-batch cap before runtime K contracts
    # the useful split prefix. Grid and workspace geometry retain the maximum.
    is_var_split_kv: bool = False
    # Causal is bottom-right aligned for speculative decode. Dense still masks
    # the ordinary per-batch KV tail at ``cache_seqs[batch]``.
    mask_type: str = MaskType.CAUSAL.value

    # Dynamic-token sparse (DSV4 CSA) specialization.  Sparse routing treats
    # every page-table entry as one physical token and stores one routing row
    # per logical query token.  The first ``sparse_swa_topk`` entries address
    # the SWA pool; the remaining entries address the compressed pool.
    is_dynamic_token_sparse: bool = False
    sparse_swa_topk: int = 128
    # Source Contract-F fused epilogue: normalize FP32 O, apply inverse RoPE
    # to D[448:512], quantize each contiguous D128 block to E4M3, and write
    # the grouped physical O/FP32-scale layouts consumed by the next GEMM.
    dsv4_fuses_inv_rope_fp8_quant: bool = False

    @property
    def tokens_per_k_tile(self) -> int:
        """Return the logical KV-token count covered by one QK tile."""

        return self.mma_qk_tiler[1]

    @property
    def tokens_per_k_cta(self) -> int:
        """Return the KV-token count owned by one CTA in the 2CTA cluster."""

        return self.tokens_per_k_tile // self.num_mma_ctas

    @property
    def pages_per_k_tile(self) -> int:
        """Return the physical page count spanned by one logical K tile."""

        return self.tokens_per_k_tile // self.page_size

    @property
    def pages_per_k_cta(self) -> int:
        """Return page IDs consumed by one CTA, including shared-page K tiles."""

        return max(1, ceil_div(self.pages_per_k_tile, self.num_mma_ctas))

    @property
    def tokens_per_v_tile(self) -> int:
        """Return the logical KV-token count covered by all PV K iterations."""

        return self.mma_pv_tiler[2] * self.iterations_pv_k

    @property
    def pages_per_v_tile(self) -> int:
        """Return the physical page count spanned by one logical V tile."""

        return self.tokens_per_v_tile // self.page_size

    @property
    def pages_per_v_subtile(self) -> int:
        """Return physical page IDs consumed by one staged BF16 PV iteration."""

        return max(1, ceil_div(self.pages_per_v_tile, self.iterations_pv_k))

    @property
    def v_subtiles_per_page(self) -> int:
        """Return staged BF16 PV iterations that share one physical page ID."""

        return max(1, ceil_div(self.iterations_pv_k, self.pages_per_v_tile))

    @property
    def v_tma_copies_per_subtile(self) -> int:
        """Return transpose-TMA copies needed to assemble one PV K subtile."""

        return self.mma_pv_tiler[2] // self.v_tma_token_count

    def is_fp8_qkv(self) -> bool:
        """Return whether Q/K/V tensors use E4M3 data."""

        return self.qkv_dtype == "e4m3"


def make_mla_decode_config(
    mma_qk_tiler_mn: Tuple[int, int] = (128, 128),
    mma_pv_tiler_mn: Tuple[int, int] = (128, 256),
    rope_dim: int = 64,
    page_size: int = 32,
    qkv_dtype: str = "bf16",
    o_dtype: str = "bf16",
    max_active_clusters: int = 56,
    is_persistent: bool = True,
    is_var_seq: bool = False,
    is_var_split_kv: bool = False,
    mask_type: MaskType | str = MaskType.CAUSAL,
    is_dynamic_token_sparse: bool = False,
    sparse_swa_topk: int = 128,
    dsv4_use_source_pv_k128: bool = False,
    dsv4_use_source_pair_ring: bool = False,
    dsv4_use_source_workid: bool = False,
    dsv4_use_source_throttle: bool = False,
    dsv4_release_s_after_p: bool = False,
    dsv4_use_source_s_handoff: bool = False,
    dsv4_use_source_direct_p: bool = False,
    dsv4_use_source_early_corr: bool = False,
    dsv4_use_source_softmax_pair_reduction: bool = False,
    dsv4_use_source_softmax_pipeline: bool = False,
    dsv4_use_source_contractible_ffma: bool = False,
    dsv4_use_source_epilogue_pair_barrier: bool = False,
    dsv4_use_source_page_offsets_cache_policy: bool = False,
    dsv4_use_source_page_offsets_transcnt: bool = False,
    dsv4_use_source_tmem_lifecycle: bool = False,
    dsv4_use_source_loader_no_tail: bool = False,
    dsv4_use_source_register_protocol: bool = False,
    dsv4_use_active_tma_prefetch: bool = False,
    dsv4_use_source_mbarrier_init: bool = False,
    dsv4_use_source_ptx_knobs: bool = False,
    dsv4_use_source_local_ptx_knobs: bool = False,
    dsv4_enable_skip_correction: bool = False,
    dsv4_fuses_inv_rope_fp8_quant: bool = False,
    stores_lse: bool = True,
) -> MlaDecodeConfig:
    """Create and populate a MlaDecodeConfig from problem parameters."""
    cfg = MlaDecodeConfig()
    cfg.mma_qk_tiler_mn = mma_qk_tiler_mn
    cfg.mma_pv_tiler_mn = mma_pv_tiler_mn
    cfg.rope_dim = rope_dim
    cfg.page_size = page_size
    cfg.qkv_dtype = qkv_dtype
    cfg.o_dtype = o_dtype
    cfg.max_active_clusters = max_active_clusters
    cfg.is_persistent = is_persistent
    cfg.is_var_seq = is_var_seq
    cfg.is_var_split_kv = is_var_split_kv
    cfg.mask_type = normalize_mask_type(mask_type)
    cfg.is_dynamic_token_sparse = is_dynamic_token_sparse
    cfg.sparse_swa_topk = sparse_swa_topk
    cfg.dsv4_use_source_pv_k128 = dsv4_use_source_pv_k128
    cfg.dsv4_use_source_pair_ring = dsv4_use_source_pair_ring
    cfg.dsv4_use_source_workid = dsv4_use_source_workid
    cfg.dsv4_use_source_throttle = dsv4_use_source_throttle
    cfg.dsv4_release_s_after_p = dsv4_release_s_after_p
    cfg.dsv4_use_source_s_handoff = dsv4_use_source_s_handoff
    cfg.dsv4_use_source_direct_p = dsv4_use_source_direct_p
    cfg.dsv4_use_source_early_corr = dsv4_use_source_early_corr
    cfg.dsv4_use_source_softmax_pair_reduction = dsv4_use_source_softmax_pair_reduction
    cfg.dsv4_use_source_softmax_pipeline = dsv4_use_source_softmax_pipeline
    cfg.dsv4_use_source_contractible_ffma = dsv4_use_source_contractible_ffma
    cfg.dsv4_use_source_epilogue_pair_barrier = dsv4_use_source_epilogue_pair_barrier
    cfg.dsv4_use_source_page_offsets_cache_policy = (
        dsv4_use_source_page_offsets_cache_policy
    )
    cfg.dsv4_use_source_page_offsets_transcnt = dsv4_use_source_page_offsets_transcnt
    cfg.dsv4_use_source_tmem_lifecycle = dsv4_use_source_tmem_lifecycle
    cfg.dsv4_use_source_loader_no_tail = dsv4_use_source_loader_no_tail
    cfg.dsv4_use_source_register_protocol = dsv4_use_source_register_protocol
    cfg.dsv4_use_active_tma_prefetch = dsv4_use_active_tma_prefetch
    cfg.dsv4_use_source_mbarrier_init = dsv4_use_source_mbarrier_init
    cfg.dsv4_use_source_ptx_knobs = dsv4_use_source_ptx_knobs
    cfg.dsv4_use_source_local_ptx_knobs = dsv4_use_source_local_ptx_knobs
    cfg.dsv4_enable_skip_correction = dsv4_enable_skip_correction
    cfg.dsv4_fuses_inv_rope_fp8_quant = dsv4_fuses_inv_rope_fp8_quant
    cfg.stores_lse = stores_lse
    if dsv4_use_source_pair_ring and not dsv4_use_source_pv_k128:
        raise ValueError("DSV4 source pair-ring requires dsv4_use_source_pv_k128=True")
    if dsv4_use_source_throttle and not dsv4_use_source_workid:
        raise ValueError("DSV4 source throttle requires dsv4_use_source_workid=True")
    if dsv4_use_source_s_handoff and not dsv4_use_source_pv_k128:
        raise ValueError("DSV4 source S handoff requires dsv4_use_source_pv_k128=True")
    if dsv4_use_source_direct_p and not dsv4_use_source_s_handoff:
        raise ValueError("DSV4 source direct-P requires dsv4_use_source_s_handoff=True")
    if dsv4_use_source_early_corr and (
        not is_dynamic_token_sparse or not dsv4_use_source_pv_k128
    ):
        raise ValueError("DSV4 source early-correction requires dynamic sparse K128")
    if dsv4_use_source_ptx_knobs and not is_dynamic_token_sparse:
        raise ValueError("DSV4 source PTX knobs require dynamic sparse MLA")
    if dsv4_use_source_mbarrier_init and not is_dynamic_token_sparse:
        raise ValueError("DSV4 source mbarrier init requires dynamic sparse MLA")
    if dsv4_use_source_local_ptx_knobs and not is_dynamic_token_sparse:
        raise ValueError("DSV4 source local PTX knobs require dynamic sparse MLA")
    if dsv4_enable_skip_correction and (
        not is_dynamic_token_sparse or qkv_dtype != "e4m3"
    ):
        raise ValueError("DSV4 skip correction requires dynamic sparse E4M3 MLA")
    if dsv4_fuses_inv_rope_fp8_quant and (
        not is_dynamic_token_sparse
        or qkv_dtype != "e4m3"
        or o_dtype != "e4m3"
        or rope_dim != 0
    ):
        raise ValueError(
            "DSV4 inverse-RoPE FP8 quant fusion requires dynamic sparse "
            "E4M3 MLA, E4M3 output, and rope_dim=0"
        )
    if dsv4_use_source_softmax_pair_reduction and not is_dynamic_token_sparse:
        raise ValueError(
            "DSV4 source softmax pair reduction requires dynamic sparse MLA"
        )
    if dsv4_use_source_softmax_pipeline and (
        not is_dynamic_token_sparse
        or not dsv4_use_source_pv_k128
        or not dsv4_use_source_direct_p
        or not dsv4_use_source_early_corr
        or qkv_dtype != "e4m3"
    ):
        raise ValueError(
            "DSV4 source softmax pipeline requires dynamic sparse FP8 "
            "K128 direct-P with early correction"
        )
    if dsv4_use_source_contractible_ffma and not dsv4_use_source_softmax_pipeline:
        raise ValueError(
            "DSV4 source contractible FFMA requires the source softmax pipeline"
        )
    if dsv4_use_source_epilogue_pair_barrier and not is_dynamic_token_sparse:
        raise ValueError(
            "DSV4 source epilogue pair barrier requires dynamic-token sparse MLA"
        )
    if dsv4_use_source_page_offsets_cache_policy and (
        not is_dynamic_token_sparse or not dsv4_use_source_pair_ring
    ):
        raise ValueError(
            "DSV4 source page-offset cache policy requires the dynamic sparse pair-ring"
        )
    if dsv4_use_source_page_offsets_transcnt and (
        not is_dynamic_token_sparse or not dsv4_use_source_pair_ring
    ):
        raise ValueError(
            "DSV4 source page-offset transaction-count arrival requires "
            "the dynamic sparse pair-ring"
        )
    if dsv4_use_source_tmem_lifecycle and (
        not is_dynamic_token_sparse
        or not dsv4_use_source_s_handoff
        or not dsv4_use_source_direct_p
        or not dsv4_use_source_early_corr
    ):
        raise ValueError(
            "DSV4 source TMEM lifecycle requires dynamic sparse S-handoff, "
            "direct-P, and early correction"
        )
    if dsv4_use_source_loader_no_tail and (
        not is_dynamic_token_sparse or not dsv4_use_source_pair_ring
    ):
        raise ValueError(
            "DSV4 source loader no-tail requires the dynamic sparse pair-ring"
        )
    if dsv4_use_source_register_protocol and not is_dynamic_token_sparse:
        raise ValueError("DSV4 source register protocol requires dynamic sparse MLA")
    if dsv4_use_active_tma_prefetch and (
        not is_dynamic_token_sparse or not dsv4_use_source_pv_k128
    ):
        raise ValueError(
            "DSV4 active TMA prefetch requires dynamic sparse source PV K128"
        )

    def _require_positive(name: str, value: int) -> None:
        """Validate that a named configuration value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def _require_divisible(
        dividend_name: str, dividend: int, divisor_name: str, divisor: int
    ) -> None:
        """Validate that one named configuration value divides another."""
        _require_positive(dividend_name, dividend)
        _require_positive(divisor_name, divisor)
        if dividend % divisor != 0:
            raise ValueError(
                f"{dividend_name}={dividend} must be divisible by "
                f"{divisor_name}={divisor}"
            )

    for idx, value in enumerate(mma_qk_tiler_mn):
        _require_positive(f"mma_qk_tiler_mn[{idx}]", value)
    for idx, value in enumerate(mma_pv_tiler_mn):
        _require_positive(f"mma_pv_tiler_mn[{idx}]", value)
    if rope_dim < 0:
        raise ValueError(f"rope_dim must be non-negative, got {rope_dim}")
    _require_positive("page_size", page_size)
    if page_size not in SUPPORTED_MLA_PAGE_SIZES and not (
        is_dynamic_token_sparse and page_size == 1
    ):
        raise ValueError(
            f"page_size must be one of {SUPPORTED_MLA_PAGE_SIZES}, got {page_size}"
        )
    if is_dynamic_token_sparse:
        if page_size != 1:
            raise ValueError(
                f"dynamic-token sparse MLA requires page_size=1, got {page_size}"
            )
        if rope_dim != 0:
            raise ValueError(
                "dynamic-token sparse MLA requires rope_dim=0 because DSV4 "
                "stores the complete pre-rotated H512 vector"
            )
        if sparse_swa_topk != mma_qk_tiler_mn[1]:
            raise ValueError(
                "dynamic-token sparse MLA currently requires one full SWA "
                f"tile ({mma_qk_tiler_mn[1]} entries), got {sparse_swa_topk}"
            )
    if page_size > mma_qk_tiler_mn[1] or mma_qk_tiler_mn[1] % page_size != 0:
        raise ValueError(
            "page_size must exactly partition the throughput 2CTA K tile: "
            f"mma_qk_tiler_mn[1]={mma_qk_tiler_mn[1]}, page_size={page_size}"
        )

    if qkv_dtype not in ("bf16", "e4m3"):
        raise ValueError(f"unsupported qkv_dtype={qkv_dtype!r}")
    if o_dtype not in ("bf16", "e4m3"):
        raise ValueError(f"unsupported o_dtype={o_dtype!r}")
    cfg.qkv_dtype_bytes = 1 if qkv_dtype == "e4m3" else 2
    cfg.o_dtype_bytes = 1 if o_dtype == "e4m3" else 2
    cfg.use_bf16_output = int(o_dtype == "bf16")
    cfg.use_fp8_output = int(o_dtype == "e4m3")
    cfg.use_fp8_split_mma_schedule = qkv_dtype == "e4m3"
    # DSV4 assigns W12-W15 to distributed Gather4 V loads.  Dense FP8 keeps
    # the original odd/even dual-softmax schedule on those warps.
    cfg.use_fp8_dual_softmax_schedule = (
        qkv_dtype == "e4m3" and not is_dynamic_token_sparse
    )
    cfg.empty_warp_ids = () if cfg.use_fp8_split_mma_schedule else (cfg.pv_mma_warp_id,)
    if cfg.use_fp8_split_mma_schedule:
        cfg.threads_per_cta = cfg.threads_per_warp * 16
        cfg.softmax_reg_num = 160
        cfg.correction_reg_num = 160
        cfg.other_reg_num = 32
        cfg.mma_o_stage = 2
        cfg.epilogue_sync_bar_id = 4
        if cfg.use_fp8_dual_softmax_schedule:
            cfg.second_compute_warp_ids = (12, 13, 14, 15)
            cfg.num_softmax_groups = 2
        elif is_dynamic_token_sparse:
            # Match the generated DSV4 CSA async pipeline contract:
            # Q=1, K=2, V=2, S=2, P=2, O=1.  ``load_k_stage`` and
            # ``mma_o_stage`` deliberately differ from dense FP8, whose
            # split-QK/PV schedule needs an additional K/O stage.
            cfg.load_k_stage = 2
            cfg.load_v_stage = 2
            cfg.mma_s_stage = 2
            cfg.p_mma_stage = 2
            cfg.p_cor_stage = 2
            cfg.mma_o_stage = 1

            if dsv4_use_source_softmax_pair_reduction:
                # TmemS in the generated DSV4 kernel reduces each M128 row
                # across the 2x2 warp-group layout with two 64-thread named
                # barriers: W0/W2 use slot 1 and W1/W3 use slot 2.  The
                # scratch is double-buffered with the two S stages, so it
                # needs no second WAR barrier before the next KV tile.
                cfg.softmax_sync_bar_id = 1
                cfg.softmax_sync_threads = 64

            if dsv4_use_source_epilogue_pair_barrier:
                # Generated ReduceRow.h uses barrier slots 3/4 for W0/W2 and
                # W1/W3, respectively.  The pairs have no data dependency.
                cfg.epilogue_sync_bar_id = 3
                cfg.epilogue_sync_threads = 64

            # The generated kernel's setmaxnreg layout is:
            # W0-W3 softmax=152, W4-W7 correction=144, W8-W11
            # MMA/page/scheduler/padding=136, W12-W15 Gather4=72.
            # Start all producer-side warps at the Gather4 floor, then let
            # their individual Task instances raise to the documented budget.
            cfg.softmax_reg_num = 152
            cfg.correction_reg_num = 144
            cfg.other_reg_num = 72
            cfg.producer_reg_num = 136
            cfg.gather4_reg_num = 72
            cfg.load_v_warp_id = 12
            cfg.load_v_num_warps = 4
            cfg.scheduler_warp_id = 10
            # Source uses W11 as Padding after fusing QK/PV into W8.
            cfg.padding_warp_id = 11
            cfg.pv_mma_warp_id = 11
            cfg.empty_warp_ids = ()
            # Source BMM2 consumes one P[K128] operand with each V[D256]
            # panel.  The current K64 x two implementation is retained only
            # as an explicit fallback while its replacement is qualified.
            cfg.dsv4_use_raw_v_gather = dsv4_use_source_pv_k128

    # Derived MMA tilers. FP8 latent QK uses K=128 while the separate RoPE MMA
    # keeps K=64.
    if cfg.rope_dim > 0:
        cfg.mma_qk_tiler_k = cfg.rope_dim * (2 if qkv_dtype == "e4m3" else 1)
    else:
        cfg.mma_qk_tiler_k = 128 if qkv_dtype == "e4m3" else 64
    _require_divisible(
        "latent_dim", cfg.latent_dim, "mma_qk_tiler_k", cfg.mma_qk_tiler_k
    )
    _require_divisible(
        "mma_qk_tiler_mn[1] * mma_qk_tiler_k",
        mma_qk_tiler_mn[1] * cfg.mma_qk_tiler_k,
        "mma_pv_tiler_mn[1]",
        mma_pv_tiler_mn[1],
    )
    cfg.mma_qk_tiler = (mma_qk_tiler_mn[0], mma_qk_tiler_mn[1], cfg.mma_qk_tiler_k)
    cfg.mma_qk_rope_tiler = (mma_qk_tiler_mn[0], mma_qk_tiler_mn[1], cfg.rope_dim)
    pv_k = (
        mma_qk_tiler_mn[1]
        if is_dynamic_token_sparse and dsv4_use_source_pv_k128
        else mma_qk_tiler_mn[1] * cfg.mma_qk_tiler_k // mma_pv_tiler_mn[1]
    )
    _require_divisible("mma_qk_tiler_mn[1]", mma_qk_tiler_mn[1], "pv_k", pv_k)
    _require_divisible(
        "latent_dim", cfg.latent_dim, "mma_pv_tiler_mn[1]", mma_pv_tiler_mn[1]
    )
    cfg.mma_pv_tiler = (mma_pv_tiler_mn[0], mma_pv_tiler_mn[1], pv_k)

    # Iteration counts
    cfg.iterations_qk_latent = cfg.latent_dim // cfg.mma_qk_tiler_k
    cfg.iterations_qk_rope = 1 if cfg.rope_dim > 0 else 0
    cfg.iterations_qk = cfg.iterations_qk_latent + cfg.iterations_qk_rope
    cfg.iterations_pv_k = cfg.mma_qk_tiler[1] // cfg.mma_pv_tiler[2]
    cfg.iterations_pv_n = cfg.latent_dim // cfg.mma_pv_tiler[1]
    cfg.kv_subtiles_per_stage = 1 if qkv_dtype == "e4m3" else 2
    _require_divisible(
        "iterations_qk_latent",
        cfg.iterations_qk_latent,
        "kv_subtiles_per_stage",
        cfg.kv_subtiles_per_stage,
    )
    _require_divisible(
        "iterations_pv_k * iterations_pv_n",
        cfg.iterations_pv_k * cfg.iterations_pv_n,
        "kv_subtiles_per_stage",
        cfg.kv_subtiles_per_stage,
    )
    cfg.iterations_qk_latent_stages = (
        cfg.iterations_qk_latent // cfg.kv_subtiles_per_stage
    )
    cfg.iterations_qk_stages = cfg.iterations_qk_latent_stages + cfg.iterations_qk_rope
    cfg.iterations_pv_stages = (
        cfg.iterations_pv_k * cfg.iterations_pv_n // cfg.kv_subtiles_per_stage
    )
    if cfg.tokens_per_v_tile != cfg.tokens_per_k_tile:
        raise ValueError(
            "PV K iterations must cover the same token tile as QK: "
            f"tokens_per_v_tile={cfg.tokens_per_v_tile}, "
            f"tokens_per_k_tile={cfg.tokens_per_k_tile}"
        )

    # Page-offset tile sizes
    num_mma_ctas = cfg.cluster_shape_mnk[0]
    cfg.num_mma_ctas = num_mma_ctas
    _require_divisible(
        "mma_qk_tiler_mn[0]", cfg.mma_qk_tiler[0], "num_mma_ctas", num_mma_ctas
    )
    _require_divisible(
        "mma_qk_tiler_mn[1]", cfg.mma_qk_tiler[1], "num_mma_ctas", num_mma_ctas
    )
    if (
        cfg.page_size != cfg.tokens_per_k_tile
        and cfg.tokens_per_k_cta % cfg.page_size != 0
    ):
        raise ValueError(
            "page_size must partition each CTA's K tile unless both CTAs share "
            "one full-tile page: "
            f"tokens_per_k_cta={cfg.tokens_per_k_cta}, page_size={cfg.page_size}"
        )
    _require_divisible(
        "mma_pv_tiler_mn[0]", cfg.mma_pv_tiler[0], "num_mma_ctas", num_mma_ctas
    )
    _require_divisible(
        "mma_pv_tiler_mn[1]", cfg.mma_pv_tiler[1], "num_mma_ctas", num_mma_ctas
    )
    cfg.kc_page_tile_size = min(cfg.page_size, cfg.tokens_per_k_cta)
    cfg.v_tma_token_count = min(cfg.page_size, V_SMEM_K_BLOCK_TOKENS)
    _require_divisible(
        "V_SMEM_K_BLOCK_TOKENS",
        V_SMEM_K_BLOCK_TOKENS,
        "v_tma_token_count",
        cfg.v_tma_token_count,
    )
    _require_divisible(
        "mma_pv_tiler[2]",
        cfg.mma_pv_tiler[2],
        "v_tma_token_count",
        cfg.v_tma_token_count,
    )

    # SMEM sizes (elements)
    cfg.smem_q_latent_elems = (
        (cfg.mma_qk_tiler[0] // num_mma_ctas)
        * cfg.mma_qk_tiler[2]
        * cfg.iterations_qk_latent
        * cfg.load_q_stage
    )
    cfg.smem_q_rope_elems = (
        (cfg.mma_qk_rope_tiler[0] // num_mma_ctas)
        * cfg.mma_qk_rope_tiler[2]
        * cfg.load_q_stage
    )
    k_latent_subtile_elems = cfg.mma_qk_tiler[1] // num_mma_ctas * cfg.mma_qk_tiler[2]
    k_rope_subtile_elems = (
        cfg.mma_qk_rope_tiler[1] // num_mma_ctas * cfg.mma_qk_rope_tiler[2]
    )
    if qkv_dtype == "e4m3":
        cfg.smem_k_stage_elems = (
            k_latent_subtile_elems * cfg.iterations_qk_latent
            + k_rope_subtile_elems * cfg.iterations_qk_rope
        )
        cfg.smem_kc_elems = cfg.smem_k_stage_elems * cfg.load_k_stage
    else:
        cfg.smem_k_stage_elems = k_latent_subtile_elems * cfg.kv_subtiles_per_stage
        cfg.smem_kc_elems = cfg.smem_k_stage_elems * cfg.load_kv_stage
    v_subtile_elems = cfg.mma_pv_tiler[1] // num_mma_ctas * cfg.mma_pv_tiler[2]
    cfg.smem_v_stage_elems = v_subtile_elems * cfg.iterations_pv_k * cfg.iterations_pv_n
    cfg.smem_vc_elems = (
        cfg.smem_v_stage_elems * cfg.load_v_stage if qkv_dtype == "e4m3" else 0
    )
    cfg.smem_p_elems = (
        (cfg.mma_pv_tiler[0] // num_mma_ctas)
        * cfg.mma_pv_tiler[2]
        * cfg.iterations_pv_k
        * cfg.p_mma_stage
    )
    cfg.softmax_exchange_elems = cfg.num_compute_warps * cfg.threads_per_warp
    if is_dynamic_token_sparse and dsv4_use_source_softmax_pair_reduction:
        # One 128-float reduceWarpGrp2x2 scratch per TmemS stage.
        cfg.softmax_exchange_elems *= cfg.mma_s_stage
    elif cfg.use_fp8_dual_softmax_schedule:
        cfg.softmax_exchange_elems *= 2

    # TMEM layout
    if is_dynamic_token_sparse and dsv4_use_source_early_corr:
        # Generated DSV4 CSA TMEM map (columns): S0/S1=[0,128), the two
        # TmemSoftmaxLocal stages=[128,160) and [160,192), and the two D256
        # O panels=[192,320) and [320,448).  A stats stage reserves one
        # tcgen05 32-column row even though its payload is two floats.
        cfg.correction_factor_offset = 128
        cfg.tmem_o_offset = 192
    else:
        cfg.tmem_o_offset = cfg.mma_s_stage * cfg.mma_qk_tiler[1] // cfg.warps_in_n
        cfg.correction_factor_offset = (
            cfg.tmem_o_offset + cfg.latent_dim // cfg.warps_in_n
        )

    # TMA byte counts
    q_latent_tile_bytes = (
        cfg.mma_qk_tiler[0] // num_mma_ctas * cfg.mma_qk_tiler[2] * cfg.qkv_dtype_bytes
    )
    q_rope_tile_bytes = (
        cfg.mma_qk_rope_tiler[0]
        // num_mma_ctas
        * cfg.mma_qk_rope_tiler[2]
        * cfg.qkv_dtype_bytes
    )
    cfg.tma_copy_q_bytes = (
        q_latent_tile_bytes * num_mma_ctas * cfg.iterations_qk_latent
        + q_rope_tile_bytes * num_mma_ctas * cfg.iterations_qk_rope
    )
    cfg.tma_copy_kc_bytes = (
        cfg.mma_qk_tiler[1]
        // num_mma_ctas
        * cfg.mma_qk_tiler[2]
        * cfg.qkv_dtype_bytes
        * num_mma_ctas
    )
    cfg.tma_copy_vc_bytes = (
        cfg.mma_pv_tiler[1]
        // num_mma_ctas
        * cfg.mma_pv_tiler[2]
        * cfg.qkv_dtype_bytes
        * num_mma_ctas
    )
    # Generic dense decode uses a shared K/V subtile contract.  DSV4 keeps
    # independent K128 and V(K128,D256) Gather4 pipelines, whose individual
    # transaction sizes differ while their full-tile completion sizes match.
    if (
        cfg.tma_copy_kc_bytes != cfg.tma_copy_vc_bytes
        and not cfg.is_dynamic_token_sparse
    ):
        raise ValueError(
            "K and V TMA subtile byte counts must match: "
            f"tma_copy_kc_bytes={cfg.tma_copy_kc_bytes}, "
            f"tma_copy_vc_bytes={cfg.tma_copy_vc_bytes}"
        )
    cfg.tma_kc_subtile_bytes = cfg.tma_copy_kc_bytes * cfg.kv_subtiles_per_stage
    cfg.tma_vc_subtile_bytes = cfg.tma_copy_vc_bytes * cfg.kv_subtiles_per_stage
    cfg.tma_copy_k_tile_bytes = (
        cfg.smem_k_stage_elems * cfg.qkv_dtype_bytes * num_mma_ctas
    )
    cfg.tma_copy_v_tile_bytes = (
        cfg.tma_copy_vc_bytes * cfg.iterations_pv_k * cfg.iterations_pv_n
    )

    # The generated DSV4 kernel uses __syncthreads() immediately after its
    # CTA_2 TMEM allocation.  Preserve that full-CTA arrival set instead of
    # letting the Gather4 producers race ahead of the alloc publication.
    if cfg.is_dynamic_token_sparse:
        cfg.tmem_sync_bar_threads = cfg.threads_per_cta
    else:
        # TMEM sync barrier thread count: QK MMA + softmax + correction, plus
        # the FP8 PV-MMA warp when it participates in TMEM O production.
        cfg.tmem_sync_bar_threads = (
            cfg.threads_per_warp * (2 if cfg.use_fp8_split_mma_schedule else 1)
            + cfg.threads_per_warp * cfg.num_compute_warps
            + cfg.threads_per_warp * cfg.num_compute_warps
        )
        if cfg.use_fp8_dual_softmax_schedule:
            cfg.tmem_sync_bar_threads += cfg.threads_per_warp * cfg.num_compute_warps

    return cfg
