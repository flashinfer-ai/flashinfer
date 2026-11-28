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

"""FMHA decode TS kernel assembly.

Assembles all resources, tasks, pipeline configs, and the dependency graph
into a TaskManager for the SwapsMmaAb decode kernel.

SwapsMmaAb, also shortened to swapsAb, names the MMA layout choice that maps
the logical attention operands onto the opposite MMA A/B roles from the
textbook QK form. BMM1 issues K as the MMA A operand and Q as the MMA B
operand, producing S = K * Q^T so KV tokens occupy the MMA M axis and the GQA
head group occupies the small N axis. BMM2 follows the same convention with V
as A and P as B.

Entry points:
  - build_decode_task_manager()  — pure Python, validation only (no GPU)
  - FmhaDecodeTs                — GPU kernel class with @cute.jit + @cute.kernel
"""

import math
from dataclasses import replace

import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
from .device.direct_sparse_metadata import (
    DirectSparseMetadataView,
    HeadIndexedMetadataView,
)
import cutlass.utils as utils
from cuda.bindings import driver as cuda_drv
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
)
from cutlass.experimental.task_scheduling.task_manager import TaskManager
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as mlir_T

from ...block_sparse.common import (
    block_sparse_contiguous_kv_copy_geometry,
    block_sparse_proxy_summary_geometry,
)
from ....device.tensor_map import (
    create_tensor_map_tiled,
    create_tensor_map_tiled_from_view,
)
from .config import FmhaDecodeConfig
from .device.constants import KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES
from .resources import (
    TmemCorrResource,
    TmemOResource,
    SmemPResource,
    TmemSResource,
    TmemSoftmaxGlobalResource,
)
from .device.common import split_kv_grid_coord
from .device.q_rows import UniformQueryOffsets, q_group_token_base, q_seq_bounds
from .device.kv_tile_idx import runtime_active_splits_kv, runtime_split_kv_is_active
from .tasks.common import prefetch_prepared_sparse_row
from .reduction import (  # noqa: F401
    decode_gen_separate_reduction_kernel,
    launch_separate_reduction,
)
from .schedule import build_decode_gen_schedule


def block_sparse_bshd_tma_strides(
    *,
    q_seq: cutlass.Integer | int,
    h_q: cutlass.Integer | int,
    h_k: cutlass.Integer | int,
    s_k: cutlass.Integer | int,
    d: cutlass.Integer | int,
    element_bytes: int,
) -> tuple[
    tuple[cutlass.Integer | int, ...],
    tuple[cutlass.Integer | int, ...],
]:
    """Build BSHD TensorMap strides in 16-byte units using Int64 math.

    The raw TensorMap API omits the implicit contiguous stride and takes the
    remaining strides in 16-byte units, so one unit holds ``16 /
    element_bytes`` elements of the 16-bit or 8-bit Q/K/V (headDim=128). Keep
    every returned value in Int64: the outer batch stride can exceed the
    signed Int32 range even though each public tensor dimension is Int32.
    """

    elements_per_stride_unit = 16 // element_bytes
    d_units = Int64(d // elements_per_stride_unit)
    h_r = h_q // h_k
    return (
        (
            d_units,
            Int64(h_r) * d_units,
            Int64(h_q) * d_units,
            Int64(q_seq) * Int64(h_q) * d_units,
        ),
        (
            Int64(h_k) * d_units,
            d_units,
            Int64(s_k) * Int64(h_k) * d_units,
        ),
    )


def build_decode_task_manager(
    cfg: FmhaDecodeConfig,
    seq_len_kv: int = 2048,
    batch_size: int = 8,
    num_heads_kv: int = 8,
    verbose: bool = True,
    skip_validation: bool = False,
    exhaustive_deadlock_race_check: bool = True,
) -> TaskManager:
    """Build and validate the decode TS TaskManager (pure Python, no GPU).

    Parameters
    ----------
    cfg : FmhaDecodeConfig
        Pre-built configuration. Build it externally with
        ``make_decode_config`` so callers can apply their own
        overrides without monkey-patching the kernel module.
    seq_len_kv : int
        KV sequence length.
    exhaustive_deadlock_race_check : bool
        Run exhaustive interleaving validation where TaskManager can model the
        complete synchronization protocol. Structural checks always run.

    Returns
    -------
    TaskManager
        Structurally validated task manager, exhaustively checked when the
        profile does not use a private TMEM-P alias protocol.
    """
    effective_seq_len_kv = configure_static_sliding_window(cfg, seq_len_kv)
    total_kv_tiles = compute_total_kv_tiles(effective_seq_len_kv, cfg.tile_size_kv)
    cfg.total_kv_tiles = total_kv_tiles

    (
        task_list,
        resource_dependency_graph,
        dma_consumer_release_labels,
        smem_allocator,
        tmem_allocator,
        _eager_init_resources,
    ) = build_decode_gen_schedule(
        cfg,
        total_kv_tiles,
        tile_sched_params=None,
        num_heads_kv=Int32(num_heads_kv),
        lse_ptr=None,
    )

    tm = TaskManager(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        dma_consumer_release_labels=dma_consumer_release_labels,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        verbose=verbose,
        skip_validation=skip_validation,
        exhaustive_deadlock_race_check=(
            exhaustive_deadlock_race_check
            and not has_unmodeled_tmem_p_alias_protocol(cfg)
        ),
    )

    return tm


def compute_static_num_skipped_kv_tiles(cfg: FmhaDecodeConfig, seq_len_kv: int) -> int:
    """Return full leading KV tiles skipped by a static sliding window."""
    if not cfg.use_sliding_window_causal or cfg.is_multi_token_q:
        return 0
    return max(seq_len_kv - cfg.attention_window_size, 0) // cfg.tile_size_kv


def compute_static_window_start_idx(cfg: FmhaDecodeConfig, seq_len_kv: int) -> int:
    """Return the token index where a static sliding window begins."""
    if not cfg.use_sliding_window_causal or cfg.is_multi_token_q:
        return 0
    return max(seq_len_kv - cfg.attention_window_size, 0)


def compute_total_kv_tiles(seq_len_kv: int, tile_size_kv: int) -> int:
    """Number of KV tiles needed for a fixed-length launch."""
    return (seq_len_kv + tile_size_kv - 1) // tile_size_kv


def configure_static_sliding_window(
    cfg: FmhaDecodeConfig, seq_len_kv: int | None, bias_kv_tma: bool = False
) -> int:
    """Populate fixed-length sliding metadata and return effective seqLenKv."""
    if seq_len_kv is None:
        cfg.use_static_sliding_kv_tma_bias = False
        cfg.static_num_skipped_kv_tiles = 0
        cfg.static_window_start_idx = 0
        return 0
    skipped_tiles = compute_static_num_skipped_kv_tiles(cfg, seq_len_kv)
    skipped_tokens = skipped_tiles * cfg.tile_size_kv
    window_start_idx = compute_static_window_start_idx(cfg, seq_len_kv)
    effective_seq_len_kv = seq_len_kv - skipped_tokens
    cfg.use_static_sliding_kv_tma_bias = bias_kv_tma and skipped_tiles > 0
    cfg.static_seq_len_kv = (
        effective_seq_len_kv if cfg.use_static_sliding_kv_tma_bias else seq_len_kv
    )
    cfg.static_num_skipped_kv_tiles = (
        0 if cfg.use_static_sliding_kv_tma_bias else skipped_tiles
    )
    cfg.static_window_start_idx = (
        window_start_idx - skipped_tokens
        if cfg.use_static_sliding_kv_tma_bias
        else window_start_idx
    )
    return effective_seq_len_kv


@cute.jit
def deallocate_decode_tmem(
    tmem_ptr_i32: cute.Pointer,
    tmem_alloc_cols: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Retire one CTA's TMEM after all of its task users have stopped."""
    warp_size = 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    dealloc_barrier_id = 13
    if cutlass.const_expr(cfg.use_keeps_mma_ab):
        prims.barrier_cta_sync(dealloc_barrier_id)
        if (
            warp_idx >= cfg.correction_warp_idx
            and warp_idx < cfg.correction_warp_idx + cfg.correction_num_warps
        ):
            tidx, _, _ = cute.arch.thread_idx()
            correction_thread_idx = tidx - cfg.correction_warp_idx * warp_size
            if correction_thread_idx < warp_size:
                tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
                prims.tcgen05_dealloc(tmem_ptr, tmem_alloc_cols)
    else:
        if (
            warp_idx >= cfg.correction_warp_idx
            and warp_idx < cfg.correction_warp_idx + cfg.correction_num_warps
        ):
            prims.barrier_cta_sync(
                dealloc_barrier_id,
                thread_count=cfg.correction_num_warps * warp_size,
            )
            tidx, _, _ = cute.arch.thread_idx()
            correction_thread_idx = tidx - cfg.correction_warp_idx * warp_size
            if correction_thread_idx < warp_size:
                tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
                prims.tcgen05_dealloc(tmem_ptr, tmem_alloc_cols)


@cute.kernel
def decode_gen_kernel(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    softmax_scale_operand: Float32 | cute.Pointer,
    output_scale: Float32 | cute.Pointer,
    g_seqlens_kv: cute.Pointer | DirectSparseMetadataView,
    g_cu_seqlens_q: cute.Pointer | UniformQueryOffsets,
    g_page_idx_kv: cute.Pointer | DirectSparseMetadataView,
    g_k_sf: cute.Pointer,
    g_v_sf: cute.Pointer,
    g_page_table_stride: Int64,
    g_page_table_capacity: Int32,
    g_q_token_kv_block_sparse_page_memberships: cute.Pointer,
    g_q_token_kv_block_sparse_page_membership_stride: Int32,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_split_kv_counter: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int | None] = 2048,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    use_native_paged_kv: cutlass.Constexpr[bool] = False,
    use_static_native_seqlens_kv: cutlass.Constexpr[bool] = False,
    g_sparse_row_route_offsets: cute.Pointer | None = None,
    g_sparse_row_route_counts: cute.Pointer | None = None,
    g_sparse_route_metadata: cute.Pointer | None = None,
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    g_sage_scales: cute.Pointer | None = None,
    lse_ptr: cute.Pointer | None = None,
    requested_splits=None,
    g_max_q_len: Int32 = 1,
) -> None:
    """Dispatch one static Q/split tile and drain padded launch slots safely."""
    if cutlass.const_expr(not cfg.fixed_seqlen):
        from copy import copy

        cfg = copy(cfg)
        cfg.static_seq_len_kv = g_s_k
        cfg.uniform_query_offsets = isinstance(g_cu_seqlens_q, UniformQueryOffsets)
        if cutlass.const_expr(cfg.groups_tokens_heads_q):
            cfg.runtime_q_ctas = Int32(
                (cutlass.Uint32(g_max_q_len) + cfg.q_tokens_per_cta - 1)
                // cfg.q_tokens_per_cta
            )
        else:
            cfg.runtime_q_ctas = g_max_q_len * (
                (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
            )
    q_group_cta_idx, h_k_idx, b_idx = split_kv_grid_coord(cfg, cute.arch.block_idx())
    if cutlass.const_expr(cfg.widens_sparse_metadata_rows_per_head):
        if cutlass.const_expr(isinstance(g_page_idx_kv, DirectSparseMetadataView)):
            g_page_idx_kv = g_page_idx_kv.with_head(h_k_idx)
        else:
            g_seqlens_kv = HeadIndexedMetadataView(g_seqlens_kv, g_h_k, h_k_idx)
            g_page_idx_kv = g_page_idx_kv + Int64(h_k_idx) * g_page_table_stride
            g_page_table_stride = g_page_table_stride * Int64(g_h_k)
            g_q_token_kv_block_sparse_page_memberships = (
                g_q_token_kv_block_sparse_page_memberships
                + Int64(h_k_idx)
                * Int64(g_q_token_kv_block_sparse_page_membership_stride)
            )
            g_q_token_kv_block_sparse_page_membership_stride = (
                g_q_token_kv_block_sparse_page_membership_stride * g_h_k
            )
    q_group_idx = q_group_cta_idx
    if cutlass.const_expr(cfg.uses_compact_split_grid):
        q_group_idx = Int32(cute.arch.block_idx()[2]) % Int32(cfg.num_q_ctas)
    elif cutlass.const_expr(cfg.use_split_kv):
        q_group_idx = q_group_cta_idx // Int32(cfg.splits_kv)

    q_token_offset = Int32(0)
    seq_len_q = Int32(0)
    if cutlass.const_expr(not cfg.use_variable_seqlens_q):
        seq_len_q = Int32(cfg.max_seq_len_q)
    q_token_base = Int32(0)
    q_tile_is_active = cutlass.Boolean(True)
    # QToken metadata writes every route's KV length, at least 1 (see
    # q_token_kv_block_sparse_metadata.cuh), so it never empties a Q tile.
    # Skipping that check keeps the producer acquire out of the prologue:
    # CTAs initialize their resources while the metadata grid runs and
    # acquire it before TaskManager reads any metadata. Direct G1 lengths come
    # from the raw inputs and keep the check.
    kv_len_prunes_q_tiles = (
        use_variable_seqlens_kv
        or (use_native_paged_kv and not use_static_native_seqlens_kv)
    ) and not (
        cfg.use_q_token_kv_block_sparse_route
        and not isinstance(g_page_idx_kv, DirectSparseMetadataView)
    )
    if cutlass.const_expr(not cfg.use_persistent_scheduler):
        if cutlass.const_expr(
            cfg.use_pdl
            and (
                (
                    cfg.use_variable_seqlens_q
                    and not isinstance(g_cu_seqlens_q, UniformQueryOffsets)
                )
                or kv_len_prunes_q_tiles
            )
        ):
            # Static tile pruning reads producer-owned Q/KV metadata here;
            # persistent schedules defer those reads until TaskManager runs.
            prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
        if cutlass.const_expr(cfg.use_variable_seqlens_q):
            q_token_offset, seq_len_q = q_seq_bounds(cfg, g_cu_seqlens_q, b_idx)
        q_token_base = q_group_token_base(cfg, q_group_idx)
        q_tile_is_active = q_token_base < seq_len_q
        if cutlass.const_expr(kv_len_prunes_q_tiles):
            q_tile_is_active = q_tile_is_active & (
                Int32(g_seqlens_kv[b_idx]) > Int32(0)
            )

    # Persistent block coordinates identify physical workers rather than
    # logical tiles; their WorkQueue owns the equivalent Q predicate. Split-KV
    # and persistent scheduling are mutually exclusive in supported profiles.
    if q_tile_is_active:
        if cutlass.const_expr(cfg.use_split_kv and static_full_split_prefix):
            run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
                tma_desc_k_atom,
                tma_desc_v_atom,
                o_iter,
                g_s_k,
                g_h_k,
                softmax_scale_operand,
                output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                Int32(cfg.splits_kv)
                if requested_splits is None
                else requested_splits[0],
                requested_splits is None,
                False,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                lse_ptr=lse_ptr,
                requested_splits=requested_splits,
                g_max_q_len=g_max_q_len,
            )
        elif cutlass.const_expr(cfg.use_split_kv and cfg.use_pdl):
            # The immediately preceding grid may produce seq_lens. Send every
            # configured split through resource initialization and defer
            # useful-prefix pruning until after the PDL acquire.
            run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
                tma_desc_k_atom,
                tma_desc_v_atom,
                o_iter,
                g_s_k,
                g_h_k,
                softmax_scale_operand,
                output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                Int32(cfg.splits_kv),
                False,
                True,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                g_sage_scales=g_sage_scales,
                lse_ptr=lse_ptr,
                requested_splits=requested_splits,
                g_max_q_len=g_max_q_len,
            )
        else:
            run_decode_gen_runtime_prefix(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
                tma_desc_k_atom,
                tma_desc_v_atom,
                o_iter,
                g_s_k,
                g_h_k,
                softmax_scale_operand,
                output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_cta_idx,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                q_token_base,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                g_sage_scales=g_sage_scales,
                lse_ptr=lse_ptr,
                requested_splits=requested_splits,
                g_max_q_len=g_max_q_len,
            )
    else:
        # Inactive requests publish no output or partials; dependent reducers
        # use the same live-length predicate. Every physical CTA still signals PDL.
        signal_padded_pdl_producer(cfg)


def decode_min_blocks_per_mp(cfg: FmhaDecodeConfig, seq_len_kv: int | None) -> int:
    """Return the launch bound needed for dynamic register reallocation.

    ``setmaxnreg`` needs a kernel-entry occupancy bound before ptxas can infer
    the initial per-thread register allocation. KV256 only pays that fixed
    hand-off cost for a long enough mainloop; the established profiles below
    retain their existing unconditional launch bounds.
    """
    kv256_reallocation = cfg.tile_size_kv == 256 and (
        seq_len_kv is None
        or compute_total_kv_tiles(seq_len_kv, cfg.tile_size_kv)
        >= KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES
    )
    return int(
        cfg.uses_decode_kv_converter
        or kv256_reallocation
        or cfg.tile_size_q == 8
        or (cfg.tile_size_q == 16 and cfg.q_dtype_bytes == 1)
        or (cfg.use_keeps_mma_ab and cfg.tile_size_q == 128)
    )


@cute.jit
def exit_cta_if_inactive(is_active: cutlass.Boolean) -> None:
    """Retire all threads in a CTA when a uniform activity predicate is false."""
    active_i32 = Int32(is_active)
    llvm.inline_asm(
        mlir_T.i32(),
        [active_i32.ir_value()],
        "{ .reg .pred _pexit; setp.eq.s32 _pexit, $1, 0;"
        " @_pexit exit; mov.u32 $0, 0; }",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def fmha_block_sparse_launch(
    problem_shape: tuple[Int32, Int32, Int32, Int32, Int32],
    q_iter: cute.Pointer,
    k_iter: cute.Pointer,
    v_iter: cute.Pointer,
    k_summary_iter: cute.Pointer,
    v_summary_iter: cute.Pointer,
    o_iter: cute.Pointer,
    row_route_offsets_iter: cute.Pointer,
    row_route_counts_iter: cute.Pointer,
    route_metadata_iter: cute.Pointer,
    softmax_scale: Float32 | cute.Pointer,
    output_scale: Float32 | cute.Pointer,
    stream: cuda_drv.CUstream,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int | None],
    g_seqlens_kv: cute.Pointer | None = None,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    num_physical_kv_pages: Int64 = 0,
    k_page_stride: Int64 = 0,
    v_page_stride: Int64 = 0,
    sage_scales_iter: cute.Pointer | None = None,
) -> None:
    """Launch attention over exact and typed exact/proxy prepared KV routes.

    A preceding prepare kernel has already resolved each BSR row into compact
    logical atom origins, storage locators, validity flags, and optional token
    words. Exact routes address K/V; proxy routes address summary K/V. Both
    layouts execute the same ``decode_gen_kernel`` schedule and
    physical copy policy. Exact builds constexpr-elide summary TensorMaps.
    ``sage_scales_iter`` is used only by a Sage attention config, which takes
    its Q, K and V scales from the prepared scales (``_SageScaleLayout``).
    """
    if cutlass.const_expr(not cfg.use_block_sparse):
        raise ValueError("fmha_block_sparse_launch requires block-sparse config")
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes and cfg.use_paged_kv):
        raise ValueError("block-sparse proxy routes require contiguous K/V")

    softmax_scale_operand = softmax_scale
    if cutlass.const_expr(not isinstance(softmax_scale, cute.Pointer)):
        softmax_scale_operand = Float32(softmax_scale * math.log2(math.e))
    b, h_q, h_k, s_k, d = problem_shape
    h_r = h_q // h_k
    q_seq = Int32(cfg.max_seq_len_q)
    q_strides, _ = block_sparse_bshd_tma_strides(
        q_seq=q_seq,
        h_q=h_q,
        h_k=h_k,
        s_k=s_k,
        d=d,
        element_bytes=cfg.q_dtype_bytes,
    )
    _, kv_strides = block_sparse_bshd_tma_strides(
        q_seq=q_seq,
        h_q=h_q,
        h_k=h_k,
        s_k=s_k,
        d=d,
        element_bytes=cfg.kv_dtype_bytes,
    )

    # H128 uses a 128-byte inner box: 64 two-byte or 128 one-byte elements.
    # The sparse profile validator rejects other head dimensions.
    tma_box0_q = min(128 // cfg.q_dtype_bytes, cfg.headdim)
    tma_box0 = min(128 // cfg.kv_dtype_bytes, cfg.headdim)
    tma_swizzle = cuda.TensorMapSwizzle.s128b
    # Public tensors are contiguous BSHD. Q is factored into (Hr, Hkv) so the
    # unchanged grouped-Q resource can address one KV head's grouped-Q tile.
    q_desc = create_tensor_map_tiled(
        global_address=q_iter.toint(),
        dtype=cfg.q_dtype,
        global_dims=(d, h_r, h_k, q_seq, b),
        global_strides=q_strides,
        box_dims=(
            tma_box0_q,
            cfg.heads_q_per_kv,
            1,
            cfg.q_tokens_per_cta,
            1,
        ),
        swizzle=tma_swizzle,
    )

    (
        primary_kv_box_size,
        kv_atom_size,
        uses_atom_desc,
    ) = block_sparse_contiguous_kv_copy_geometry(
        kv_block_size=cfg.kv_block_size,
        kv_route_size=cfg.tile_size_kv,
    )
    k_desc_summary_primary = None
    v_desc_summary_primary = None
    k_desc_summary_atom = None
    v_desc_summary_atom = None
    if cutlass.const_expr(cfg.use_paged_kv):
        # Paged HND storage is addressed as (D, token-in-page, Hkv, page).
        # Prepared routes already contain each atom's physical page ID, so no
        # dense page table is passed to or staged by the attention kernel.
        kv_shape = (
            d,
            Int32(cfg.num_tokens_per_page),
            h_k,
            num_physical_kv_pages,
        )
        k_layout = cute.make_layout(
            kv_shape,
            stride=(
                1,
                d,
                d * Int32(cfg.num_tokens_per_page),
                k_page_stride,
            ),
        )
        v_layout = cute.make_layout(
            kv_shape,
            stride=(
                1,
                d,
                d * Int32(cfg.num_tokens_per_page),
                v_page_stride,
            ),
        )
        k_desc_atom = create_tensor_map_tiled_from_view(
            cute.make_tensor(k_iter, k_layout),
            box_dims=(tma_box0, kv_atom_size, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle,
        )
        v_desc_atom = create_tensor_map_tiled_from_view(
            cute.make_tensor(v_iter, v_layout),
            box_dims=(tma_box0, kv_atom_size, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle,
        )
        k_desc_primary = k_desc_atom
        v_desc_primary = v_desc_atom
    else:
        # Exact and summary tensors form one logical segmented KV coordinate
        # space. Each physical source owns the same primary/atom descriptor
        # pair; the prepared route kind selects the pair, while the loader
        # retains the existing KV128/fine/KV256 copy policy.
        kv_dims = (d, s_k, h_k, b)
        k_desc_primary = create_tensor_map_tiled(
            global_address=k_iter.toint(),
            dtype=cfg.k_dtype,
            global_dims=kv_dims,
            global_strides=kv_strides,
            box_dims=(tma_box0, primary_kv_box_size, 1, 1),
            swizzle=tma_swizzle,
        )
        v_desc_primary = create_tensor_map_tiled(
            global_address=v_iter.toint(),
            dtype=cfg.v_dtype,
            global_dims=kv_dims,
            global_strides=kv_strides,
            box_dims=(tma_box0, primary_kv_box_size, 1, 1),
            swizzle=tma_swizzle,
        )
        k_desc_atom = k_desc_primary
        v_desc_atom = v_desc_primary
        if cutlass.const_expr(uses_atom_desc):
            # KV256 always stages four semantic KV64 atoms. KV128 needs this
            # map only when a route may join unrelated BSR entries.
            k_desc_atom = create_tensor_map_tiled(
                global_address=k_iter.toint(),
                dtype=cfg.k_dtype,
                global_dims=kv_dims,
                global_strides=kv_strides,
                box_dims=(tma_box0, kv_atom_size, 1, 1),
                swizzle=tma_swizzle,
            )
            v_desc_atom = create_tensor_map_tiled(
                global_address=v_iter.toint(),
                dtype=cfg.v_dtype,
                global_dims=kv_dims,
                global_strides=kv_strides,
                box_dims=(tma_box0, kv_atom_size, 1, 1),
                swizzle=tma_swizzle,
            )

        if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
            num_kv_blocks, _ = block_sparse_proxy_summary_geometry(
                s_k,
                cfg.kv_block_size,
            )
            _, summary_kv_strides = block_sparse_bshd_tma_strides(
                q_seq=q_seq,
                h_q=h_q,
                h_k=h_k,
                s_k=num_kv_blocks,
                d=d,
                element_bytes=cfg.kv_dtype_bytes,
            )
            summary_dims = (d, num_kv_blocks, h_k, b)
            k_desc_summary_primary = create_tensor_map_tiled(
                global_address=k_summary_iter.toint(),
                dtype=cfg.k_dtype,
                global_dims=summary_dims,
                global_strides=summary_kv_strides,
                box_dims=(tma_box0, primary_kv_box_size, 1, 1),
                swizzle=tma_swizzle,
            )
            v_desc_summary_primary = create_tensor_map_tiled(
                global_address=v_summary_iter.toint(),
                dtype=cfg.v_dtype,
                global_dims=summary_dims,
                global_strides=summary_kv_strides,
                box_dims=(tma_box0, primary_kv_box_size, 1, 1),
                swizzle=tma_swizzle,
            )
            k_desc_summary_atom = k_desc_summary_primary
            v_desc_summary_atom = v_desc_summary_primary
            if cutlass.const_expr(uses_atom_desc):
                k_desc_summary_atom = create_tensor_map_tiled(
                    global_address=k_summary_iter.toint(),
                    dtype=cfg.k_dtype,
                    global_dims=summary_dims,
                    global_strides=summary_kv_strides,
                    box_dims=(tma_box0, kv_atom_size, 1, 1),
                    swizzle=tma_swizzle,
                )
                v_desc_summary_atom = create_tensor_map_tiled(
                    global_address=v_summary_iter.toint(),
                    dtype=cfg.v_dtype,
                    global_dims=summary_dims,
                    global_strides=summary_kv_strides,
                    box_dims=(tma_box0, kv_atom_size, 1, 1),
                    swizzle=tma_swizzle,
                )

    q_groups = Int32(
        (q_seq + Int32(cfg.q_tokens_per_cta - 1)) // Int32(cfg.q_tokens_per_cta)
    )
    if cutlass.const_expr(cfg.use_persistent_scheduler):
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (q_groups, h_k, b),
            (1, 1, 1),
        )
        grid = tile_sched_params.get_grid_shape()
    else:
        tile_sched_params = None
        grid = (q_groups, h_k, b)

    null_i32_ptr = cute.make_ptr(
        Int32,
        0,
        mem_space=cutlass.AddressSpace.gmem,
    )
    null_f32_ptr = cute.make_ptr(
        Float32,
        0,
        mem_space=cutlass.AddressSpace.gmem,
    )
    seqlens_kv_iter = (
        g_seqlens_kv if cutlass.const_expr(use_variable_seqlens_kv) else null_i32_ptr
    )
    decode_gen_kernel(
        q_desc,
        k_desc_primary,
        v_desc_primary,
        k_desc_primary,
        v_desc_primary,
        k_desc_atom,
        v_desc_atom,
        o_iter,
        s_k,
        h_k,
        softmax_scale_operand,
        output_scale,
        seqlens_kv_iter,
        null_i32_ptr,
        null_i32_ptr,
        null_f32_ptr,
        null_f32_ptr,
        Int64(0),  # g_page_table_stride
        Int32(0),  # g_page_table_capacity
        null_i32_ptr,
        Int32(0),  # g_q_token_kv_block_sparse_page_membership_stride
        o_iter,
        null_f32_ptr,
        null_i32_ptr,
        null_f32_ptr,
        h_r,
        tile_sched_params,
        cfg,
        seq_len_kv,
        use_variable_seqlens_kv,
        False,  # use_native_paged_kv
        False,  # use_static_native_seqlens_kv
        row_route_offsets_iter,
        row_route_counts_iter,
        route_metadata_iter,
        False,  # static_full_split_prefix
        tma_desc_k_summary=k_desc_summary_primary,
        tma_desc_v_summary=v_desc_summary_primary,
        tma_desc_k_summary_atom=k_desc_summary_atom,
        tma_desc_v_summary_atom=v_desc_summary_atom,
        g_sage_scales=sage_scales_iter,
        lse_ptr=None,
        g_max_q_len=q_seq,
    ).launch(
        grid=grid,
        block=[cfg.threads_per_cta, 1, 1],
        cluster=[1, 1, 1],
        stream=stream,
        # Reuse dense decode's entry-occupancy contract, including the long
        # KV256 profiles that execute dynamic register reallocation.
        min_blocks_per_mp=decode_min_blocks_per_mp(cfg, seq_len_kv),
        use_pdl=False,
    )


def has_unmodeled_tmem_p_alias_protocol(cfg: FmhaDecodeConfig) -> bool:
    """Whether exhaustive TS checking would report a known false P/S race.

    The staged D256 path selects one of two physical P/S stages at runtime.
    Static streamed profiles instead order P fragments with private mbarriers and
    reuses the matching TmemO-full barrier as the next-QK overwrite credit.
    Those intra-work protocols are below TaskManager's resource transitions,
    so its allocation-level checker cannot prove them. Persistent streaming has
    enough task-level ordering for the checker and remains covered.
    """
    return cfg.uses_staged_one_inst_tmem_p or (
        cfg.streams_tmem_p_fragments and not cfg.use_persistent_scheduler
    )


@cute.jit
def launch_decode_kernel(
    problem_shape: tuple[Int32, Int32, Int32, Int32, Int32],
    q_iter: cute.Pointer,
    k_iter: cute.Pointer,
    v_iter: cute.Pointer,
    k_sf_iter: cute.Pointer,
    v_sf_iter: cute.Pointer,
    o_iter: cute.Pointer,
    seqlens_kv_iter: cute.Pointer | DirectSparseMetadataView,
    cu_seqlens_q_iter: cute.Pointer | UniformQueryOffsets,
    total_q_tokens: Int32,
    page_idx_kv_iter: cute.Pointer | DirectSparseMetadataView,
    q_token_kv_block_sparse_page_memberships_iter: cute.Pointer,
    partial_o_iter: cute.Pointer,
    partial_stats_iter: cute.Pointer,
    split_kv_counter_iter: cute.Pointer,
    attention_sinks_iter: cute.Pointer,
    softmax_scale: Float32 | cute.Pointer,
    output_scale: Float32 | cute.Pointer,
    kv_b_stride: Int32,
    max_active_clusters: Int32,
    stream: cuda_drv.CUstream,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int | None] = 2048,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    use_native_paged_kv: cutlass.Constexpr[bool] = False,
    page_table_stride: Int64 = 0,
    page_table_capacity: Int32 = 0,
    q_token_kv_block_sparse_page_membership_stride: Int32 = 0,
    num_physical_kv_pages: Int64 = 0,
    k_page_stride: Int64 = 0,
    k_head_stride: Int64 = 0,
    k_token_stride: Int64 = 0,
    v_page_stride: Int64 = 0,
    v_head_stride: Int64 = 0,
    v_token_stride: Int64 = 0,
    k_sf_page_stride: Int64 = 0,
    v_sf_page_stride: Int64 = 0,
    k_sf_head_stride: Int64 = 0,
    k_sf_token_stride: Int64 = 0,
    v_sf_head_stride: Int64 = 0,
    v_sf_token_stride: Int64 = 0,
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
    use_static_native_seqlens_kv: cutlass.Constexpr[bool] = False,
    sage_scales_iter: cute.Pointer | None = None,
    lse_ptr: cute.Pointer | None = None,
    requested_splits=None,
    max_q_len: Int32 = 0,
    runtime_q_capacity_rows: Int32 = 0,
) -> None:
    """Standalone JIT launcher for FMHA decode TS.

    ``max_q_len`` bounds the launched Q-token grid. ``runtime_q_capacity_rows``
    is the partial-output row stride per request/KV head and can exceed that
    grid's live rows; both operands remain runtime values.
    ``sage_scales_iter`` is used only by a Sage attention config, which takes
    its Q, K and V scales from the prepared scales (``_SageScaleLayout``).
    """
    log2_e = math.log2(math.e)
    softmax_scale_operand = softmax_scale
    if cutlass.const_expr(not isinstance(softmax_scale, cute.Pointer)):
        softmax_scale_operand = Float32(softmax_scale * log2_e)
    if cutlass.const_expr(requested_splits is not None):
        # Prepare the reciprocal once on the host; counts stay runtime operands.
        requested_splits = (
            requested_splits,
            cute.fast_divmod_create_divisor(requested_splits * Int32(cfg.num_insts_kv)),
        )
    b, h_q, h_k, s_k, d = problem_shape
    h_r = h_q // h_k
    bias_static_sliding_kv_tma = (
        cfg.use_sliding_window_causal
        and cfg.fixed_seqlen
        and cfg.max_seq_len_q == 1
        and not cfg.use_paged_kv
        and not cfg.use_split_kv
        and not use_variable_seqlens_kv
    )
    effective_seq_len_kv = configure_static_sliding_window(
        cfg, seq_len_kv, bias_static_sliding_kv_tma
    )

    q_seq = max_q_len if not cfg.fixed_seqlen else Int32(cfg.max_seq_len_q)
    if cutlass.const_expr(cfg.use_paged_kv):
        kv_tma_d = d
        if cutlass.const_expr(cfg.use_nvfp4_interleaved_kv):
            # Interleaved rows hold the packed payload and then its scales.
            kv_tma_d = Int32(cfg.nvfp4_interleaved_row_bytes)
        elif cutlass.const_expr(cfg.uses_packed_nvfp4_raw_payload):
            kv_tma_d = d // Int32(2)
        if cutlass.const_expr(use_native_paged_kv):
            total_pages = num_physical_kv_pages
            storage_tokens_per_page = Int32(cfg.effective_storage_tokens_per_page)
            k_tma_page_stride = k_page_stride
            k_tma_head_stride = k_head_stride
            k_tma_token_stride = k_token_stride
            v_tma_page_stride = v_page_stride
            v_tma_head_stride = v_head_stride
            v_tma_token_stride = v_token_stride
            if cutlass.const_expr(
                cfg.store_transformed_kv_in_tmem
                and not cfg.use_nvfp4_interleaved_kv
                and not cfg.use_packed_nvfp4_raw_payload
            ):
                # B4X16_P64 tensor-map strides are expressed in logical FP4
                # lanes while the native cache reports packed-byte strides.
                k_tma_page_stride = k_page_stride * Int64(2)
                k_tma_head_stride = k_head_stride * Int64(2)
                k_tma_token_stride = k_token_stride * Int64(2)
                v_tma_page_stride = v_page_stride * Int64(2)
                v_tma_head_stride = v_head_stride * Int64(2)
                v_tma_token_stride = v_token_stride * Int64(2)
            if cutlass.const_expr(cfg.use_flat_native_kv_tma):
                # Reinterpret compact HND rows, without copying/repacking KV
                # or requiring contiguous sparse selections. Planning checked
                # signed-32-bit row bounds; other bindings use the general map.
                # For S tokens/page, F tokens/fragment and H heads, metadata
                # emits A = page*H*(S/F) + subpage. Attention addresses row
                # (A + head*(S/F))*F, avoiding per-copy page/subpage divmod.
                # Bytes and TMA count are unchanged; rank-2 issuing below is a
                # separate optimization of this descriptor-only flat view.
                flat_tokens = (
                    num_physical_kv_pages
                    * Int64(storage_tokens_per_page)
                    * Int64(cfg.flat_native_kv_num_heads)
                )
                kv_shape = (kv_tma_d, flat_tokens, Int32(1), Int32(1))
                k_tma_head_stride = k_tma_token_stride * flat_tokens
                k_tma_page_stride = k_tma_head_stride
                v_tma_head_stride = v_tma_token_stride * flat_tokens
                v_tma_page_stride = v_tma_head_stride
            else:
                kv_shape = (
                    kv_tma_d,
                    storage_tokens_per_page,
                    h_k,
                    total_pages,
                )
            if cutlass.const_expr(cfg.uses_2d_flat_kv_tma):
                # The prepared flat view has no head/page coordinates. Keep
                # the same physical rows and SW128 tile without repacking KV.
                k_layout = cute.make_layout(
                    (kv_shape[0], kv_shape[1]), stride=(1, k_tma_token_stride)
                )
                v_layout = cute.make_layout(
                    (kv_shape[0], kv_shape[1]), stride=(1, v_tma_token_stride)
                )
            else:
                k_layout = cute.make_layout(
                    kv_shape,
                    stride=(
                        1,
                        k_tma_token_stride,
                        k_tma_head_stride,
                        k_tma_page_stride,
                    ),
                )
                v_layout = cute.make_layout(
                    kv_shape,
                    stride=(
                        1,
                        v_tma_token_stride,
                        v_tma_head_stride,
                        v_tma_page_stride,
                    ),
                )
            k_tma = cute.make_tensor(k_iter, k_layout)
            v_tma = cute.make_tensor(v_iter, v_layout)
        else:
            total_pages = b * Int32(cfg.max_num_pages_per_seq_kv)
            kv_layout = cute.make_layout(
                (kv_tma_d, Int32(cfg.num_tokens_per_page), h_k, total_pages),
                stride=(
                    1,
                    kv_tma_d,
                    kv_tma_d * Int32(cfg.num_tokens_per_page),
                    kv_tma_d * Int32(cfg.num_tokens_per_page) * h_k,
                ),
            )
            k_tma_iter = k_iter
            v_tma_iter = v_iter
            k_tma = cute.make_tensor(k_tma_iter, kv_layout)
            v_tma = cute.make_tensor(v_tma_iter, kv_layout)
    else:
        kv_s_for_tma = s_k
        k_tma_iter = k_iter
        v_tma_iter = v_iter
        if cutlass.const_expr(cfg.use_static_sliding_kv_tma_bias):
            skipped_tokens = Int32(
                compute_static_num_skipped_kv_tiles(cfg, seq_len_kv) * cfg.tile_size_kv
            )
            skipped_elems = skipped_tokens * d * h_k
            k_tma_iter = k_iter + skipped_elems
            v_tma_iter = v_iter + skipped_elems
            kv_s_for_tma = Int32(effective_seq_len_kv)
        # Contiguous K/V are compact BSHD, matching the public PrimTS tensor
        # contract and the Q layout above.
        kv_layout = cute.make_layout(
            (d, kv_s_for_tma, h_k, b), stride=(1, d * h_k, d, kv_b_stride)
        )
        k_tma = cute.make_tensor(k_tma_iter, kv_layout)
        v_tma = cute.make_tensor(v_tma_iter, kv_layout)

    # Keep the TMA inner box at 128B when possible, but never exceed headDim.
    # box_dim is expressed in elements of the source dtype, not bytes.
    tma_box0_q = min(128 // cfg.q_dtype_bytes, cfg.headdim)
    tma_box0_k = min(128 // cfg.k_dtype_bytes, cfg.headdim)
    tma_box0_v = min(128 // cfg.v_dtype_bytes, cfg.headdim)
    tma_swizzle_q = cuda.TensorMapSwizzle.s128b
    if cutlass.const_expr(cfg.q_dtype_bytes == 1 and cfg.headdim == 64):
        tma_swizzle_q = cuda.TensorMapSwizzle.s64b
    tma_swizzle_k = cuda.TensorMapSwizzle.s128b
    if cutlass.const_expr(cfg.k_dtype_bytes == 1 and cfg.headdim == 64):
        tma_swizzle_k = cuda.TensorMapSwizzle.s64b
    tma_swizzle_v = cuda.TensorMapSwizzle.s128b
    if cutlass.const_expr(cfg.v_dtype_bytes == 1 and cfg.headdim == 64):
        tma_swizzle_v = cuda.TensorMapSwizzle.s64b
    if cutlass.const_expr(cfg.use_nvfp4_interleaved_kv):
        # One plain-byte copy per fragment: complete rows, no swizzle.
        tma_box0_k = cfg.nvfp4_interleaved_row_bytes
        tma_box0_v = cfg.nvfp4_interleaved_row_bytes
        tma_swizzle_k = cuda.TensorMapSwizzle.none
        tma_swizzle_v = cuda.TensorMapSwizzle.none
    elif cutlass.const_expr(cfg.use_nvfp4_kv):
        if cutlass.const_expr(not cfg.uses_packed_nvfp4_raw_payload):
            # Unpacking TMA consumes 128 logical FP4 lanes and materializes
            # one s128b-swizzled byte per lane in the raw SMEM stage.
            tma_box0_k = cfg.head_dim_kv_stage
            tma_box0_v = cfg.head_dim_kv_stage
            tma_swizzle_k = cuda.TensorMapSwizzle.s128b
            tma_swizzle_v = cuda.TensorMapSwizzle.s128b
        else:
            # Packed E2M1: two channels per byte, one raw head-dim stage.
            # Whole-tile raw stages issue one 64-byte D128 slice box per
            # slice (a 64B-swizzled box cannot be wider than 64 bytes).
            packed_box_hd = (
                cfg.head_dim_kv_stage
                if cfg.use_whole_tile_nvfp4_raw_stage
                else cfg.raw_kv_head_dim_stage
            )
            tma_box0_k = packed_box_hd // 2
            tma_box0_v = packed_box_hd // 2
            tma_swizzle_k = cuda.TensorMapSwizzle.none
            if cutlass.const_expr(cfg.uses_sw128_nvfp4_raw_payload):
                tma_swizzle_k = cuda.TensorMapSwizzle.s128b
            elif cutlass.const_expr(
                cfg.uses_fast_smem_nvfp4_transform or cfg.use_packed_nvfp4_raw_payload
            ):
                # 64-byte packed rows; the 64B swizzle gives conflict-free
                # 16-byte row loads in the fast SMEM converter and in the
                # packed-payload TMEM converters.
                tma_swizzle_k = cuda.TensorMapSwizzle.s64b
            tma_swizzle_v = tma_swizzle_k
    if cutlass.const_expr(cfg.tile_size_kv == 256):
        # The 2x2 datapath consumes K in a (0, 2, 1, 3) KV64 permutation.
        # A KV64 TensorMap atom lets the shared load resource place each
        # semantic block directly in its physical slot for both paged and
        # contiguous layouts.
        tma_kv_tokens = min(
            cfg.num_tokens_per_page if cfg.use_paged_kv else cfg.tile_size_kv,
            64,
        )
    else:
        tma_kv_tokens = (
            cfg.num_tokens_per_page
            if cutlass.const_expr(cfg.use_paged_kv)
            else cfg.tile_size_kv
        )
    q_box_dims: tuple[object, ...]
    if cutlass.const_expr(cfg.use_variable_seqlens_q):
        # Bound packed Q by its physical [D, Hq, total_q_tokens] allocation.
        # Per-route Q masks handle tiles that cross into the next request.
        q_tma = cute.make_tensor(
            q_iter,
            cute.make_layout(
                (d, h_q, total_q_tokens),
                stride=(1, d, h_q * d),
            ),
        )
        if cutlass.const_expr(cfg.groups_tokens_heads_q):
            q_box_dims = (
                tma_box0_q,
                cfg.heads_q_per_kv,
                cfg.q_tma_tokens_per_cta,
            )
            q_groups = (q_seq + Int32(cfg.q_tokens_per_cta - 1)) // Int32(
                cfg.q_tokens_per_cta
            )
        else:
            q_box_dims = (tma_box0_q, cfg.tile_size_q, 1)
            head_ctas_per_token = Int32(
                (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
            )
            q_groups = q_seq * head_ctas_per_token
        tma_desc_q = create_tensor_map_tiled_from_view(
            q_tma,
            box_dims=q_box_dims,
            stride_order=(0, 1, 2),
            swizzle=tma_swizzle_q,
        )
    else:
        q_tma = cute.make_tensor(
            q_iter,
            cute.make_layout(
                (d, h_r, h_k, q_seq, b),
                stride=(1, d, h_r * d, h_q * d, q_seq * h_q * d),
            ),
        )
        if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
            q_box_dims = (
                tma_box0_q,
                cfg.heads_q_per_kv,
                1,
                cfg.q_tokens_per_cta,
                1,
            )
            q_groups = (q_seq + Int32(cfg.q_tokens_per_cta - 1)) // Int32(
                cfg.q_tokens_per_cta
            )
        else:
            q_box_dims = (tma_box0_q, cfg.tile_size_q, 1, 1, 1)
            q_groups = (
                (h_r + Int32(cfg.tile_size_q - 1)) // Int32(cfg.tile_size_q)
            ) * q_seq
        tma_desc_q = create_tensor_map_tiled_from_view(
            q_tma,
            box_dims=q_box_dims,
            stride_order=(0, 1, 2, 3, 4),
            swizzle=tma_swizzle_q,
        )
    if cutlass.const_expr(cfg.uses_2d_flat_kv_tma):
        k_box_dims = (tma_box0_k, tma_kv_tokens)
        v_box_dims = (tma_box0_v, tma_kv_tokens)
        kv_stride_order = (0, 1)
    else:
        k_box_dims = (tma_box0_k, tma_kv_tokens, 1, 1)
        v_box_dims = (tma_box0_v, tma_kv_tokens, 1, 1)
        kv_stride_order = (0, 1, 2, 3)
    if cutlass.const_expr(
        cfg.store_transformed_kv_in_tmem
        and not cfg.use_nvfp4_interleaved_kv
        and not cfg.use_packed_nvfp4_raw_payload
    ):
        tma_desc_k = create_tensor_map_tiled_from_view(
            k_tma,
            box_dims=k_box_dims,
            stride_order=kv_stride_order,
            swizzle=tma_swizzle_k,
            dtype=cutlass.Float4E2M1FNx2,
            tma_format=cuda.TensorMapDataFormat.B4X16_P64,
        )
        tma_desc_v = create_tensor_map_tiled_from_view(
            v_tma,
            box_dims=v_box_dims,
            stride_order=kv_stride_order,
            swizzle=tma_swizzle_v,
            dtype=cutlass.Float4E2M1FNx2,
            tma_format=cuda.TensorMapDataFormat.B4X16_P64,
        )
    else:
        tma_desc_k = create_tensor_map_tiled_from_view(
            k_tma,
            box_dims=k_box_dims,
            stride_order=kv_stride_order,
            swizzle=tma_swizzle_k,
        )
        tma_desc_v = create_tensor_map_tiled_from_view(
            v_tma,
            box_dims=v_box_dims,
            stride_order=kv_stride_order,
            swizzle=tma_swizzle_v,
        )
    if cutlass.const_expr(cfg.uses_nvfp4_gather4_kv):
        # Compact HND rows make every page fragment one contiguous row of a
        # flat 2-D map (D256 four-token fragments: 576 bytes; 32-bit words keep
        # the box within 256 elements). gather4 copies take four row
        # coordinates each.
        frag_bytes = cfg.num_tokens_per_page * cfg.nvfp4_interleaved_row_bytes
        frag_rows = Int32(
            num_physical_kv_pages
            * Int64(cfg.effective_storage_tokens_per_page // cfg.num_tokens_per_page)
            * Int64(cfg.flat_native_kv_num_heads)
        )
        tma_desc_k = create_tensor_map_tiled(
            global_address=k_iter.toint(),
            dtype=Int32,
            global_dims=[frag_bytes // 4, frag_rows],
            global_strides=[frag_bytes // 16],
            box_dims=[frag_bytes // 4, 1],
            swizzle=cuda.TensorMapSwizzle.none,
        )
        tma_desc_v = create_tensor_map_tiled(
            global_address=v_iter.toint(),
            dtype=Int32,
            global_dims=[frag_bytes // 4, frag_rows],
            global_strides=[frag_bytes // 16],
            box_dims=[frag_bytes // 4, 1],
            swizzle=cuda.TensorMapSwizzle.none,
        )

    # NVFP4 scale factors are staged into SMEM by the same SmemKv TMA pipeline
    # as the packed K/V. The SF tensor is logically
    # (headdim // 16, storage_tokens_per_page, h_k, total_pages) E4M3. K uses
    # token-major layout and V uses TRT-LLM's 4-token interleaved layout. One
    # copy per page fragment stages its rows token-major, folding
    # ``sf_tma_reshape_factor`` token rows into one box row when a row is
    # below TMA's 16-byte minimum (FmhaDecodeConfig.sf_tma_reshape_factor).
    tma_desc_k_sf = tma_desc_k
    tma_desc_v_sf = tma_desc_v
    if cutlass.const_expr(cfg.uses_nvfp4_scale_tensors):
        sf_per_token = cfg.headdim // 16
        sf_r = cfg.sf_tma_reshape_factor
        sf_storage_tokens_per_page = cfg.num_tokens_per_page
        if cutlass.const_expr(use_native_paged_kv):
            sf_storage_tokens_per_page = cfg.effective_storage_tokens_per_page
        # Native paged launches use the caller's scale-factor strides; other
        # launches use compact HND scale-factor pages. A fold needs compact
        # token rows, which the plan validates.
        compact_sf_head_stride = Int64(sf_per_token * sf_storage_tokens_per_page)
        k_sf_tma_strides = (
            Int64(sf_per_token),
            compact_sf_head_stride,
            compact_sf_head_stride * Int64(h_k),
        )
        v_sf_tma_strides = k_sf_tma_strides
        if cutlass.const_expr(use_native_paged_kv):
            k_sf_tma_strides = (k_sf_token_stride, k_sf_head_stride, k_sf_page_stride)
            v_sf_tma_strides = (v_sf_token_stride, v_sf_head_stride, v_sf_page_stride)
        sf_shape = (
            Int32(sf_per_token * sf_r),
            Int32(sf_storage_tokens_per_page // sf_r),
            h_k,
            total_pages,
        )
        # The descriptor spans a physical page, but each transaction copies
        # only the semantic fragment selected by the decoded page locator.
        sf_box = (sf_per_token * sf_r, cfg.num_tokens_per_page // sf_r, 1, 1)

        def sf_tma_desc(sf_iter, strides, swizzle):
            """TensorMap of one scale tensor with (token, head, page) ``strides``."""
            token_stride, head_stride, page_stride = strides
            layout = cute.make_layout(
                sf_shape,
                stride=(1, token_stride * Int64(sf_r), head_stride, page_stride),
            )
            return cuda.create_tensor_map_tiled_from_view(
                cute.make_tensor(sf_iter, layout),
                box_dims=sf_box,
                stride_order=(0, 1, 2, 3),
                swizzle=swizzle,
            )

        # Gather4 launches encode only their two flat scale maps; stage-sliced
        # scales copy one D128 stage's columns per transaction; other
        # profiles copy complete scale rows.
        if cutlass.const_expr(cfg.uses_nvfp4_scale_gather4):
            # Compact scale tensors: each page fragment's scale rows are one
            # contiguous row of a flat 2-D map (32-bit words), gathered four
            # rows per copy (FmhaDecodeConfig.uses_nvfp4_scale_gather4).
            sf_frag_bytes = cfg.num_tokens_per_page * sf_per_token
            sf_frag_rows = Int32(
                Int64(total_pages)
                * Int64(sf_storage_tokens_per_page // cfg.num_tokens_per_page)
                * Int64(cfg.flat_nvfp4_scale_num_heads)
            )

            def sf_gather4_desc(sf_iter):
                """Flat fragment-row TensorMap of one compact scale tensor."""
                return create_tensor_map_tiled(
                    global_address=sf_iter.toint(),
                    dtype=Int32,
                    global_dims=[sf_frag_bytes // 4, sf_frag_rows],
                    global_strides=[sf_frag_bytes // 16],
                    box_dims=[sf_frag_bytes // 4, 1],
                    swizzle=cuda.TensorMapSwizzle.none,
                )

            tma_desc_k_sf = sf_gather4_desc(k_sf_iter)
            tma_desc_v_sf = sf_gather4_desc(v_sf_iter)
        elif cutlass.const_expr(cfg.uses_stage_sliced_nvfp4_sf):
            # K: one token per box row, 16 B (the half row holding a D128
            # stage's 8 scales); the loader selects the half by column.
            k_token_stride = k_sf_tma_strides[0]
            tma_desc_k_sf = cuda.create_tensor_map_tiled_from_view(
                cute.make_tensor(
                    k_sf_iter,
                    cute.make_layout(
                        (
                            Int32(sf_per_token),
                            Int32(sf_storage_tokens_per_page),
                            h_k,
                            total_pages,
                        ),
                        stride=(
                            1,
                            k_token_stride,
                            k_sf_tma_strides[1],
                            k_sf_tma_strides[2],
                        ),
                    ),
                ),
                box_dims=(16, cfg.num_tokens_per_page, 1, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=cuda.TensorMapSwizzle.none,
            )
            # V: folded 4-token rows (128 B); a stage's scales are the 32 B
            # at column 32 * stage of each row.
            v_token_stride, v_head_stride, v_page_stride = v_sf_tma_strides
            tma_desc_v_sf = cuda.create_tensor_map_tiled_from_view(
                cute.make_tensor(
                    v_sf_iter,
                    cute.make_layout(
                        sf_shape,
                        stride=(
                            1,
                            v_token_stride * Int64(sf_r),
                            v_head_stride,
                            v_page_stride,
                        ),
                    ),
                ),
                box_dims=(32, cfg.num_tokens_per_page // sf_r, 1, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=cuda.TensorMapSwizzle.none,
            )
        else:
            tma_desc_k_sf = sf_tma_desc(
                k_sf_iter, k_sf_tma_strides, cuda.TensorMapSwizzle.none
            )
            # D512 V scales use the 128 B swizzle (FmhaDecodeConfig.swizzles_v_scale_factors).
            tma_desc_v_sf = sf_tma_desc(
                v_sf_iter,
                v_sf_tma_strides,
                cuda.TensorMapSwizzle.s128b
                if cutlass.const_expr(cfg.swizzles_v_scale_factors)
                else cuda.TensorMapSwizzle.none,
            )

    grid_x = q_groups
    grid_batch = b
    if cutlass.const_expr(cfg.use_split_kv):
        grid_x = q_groups * Int32(cfg.splits_kv)
        if cutlass.const_expr(cfg.uses_compact_split_grid):
            # X contains only live policy splits. Q/batch share Z with a
            # compile-time Q stride, avoiding runtime-count coordinate division.
            grid_x = (
                Int32(cfg.splits_kv)
                if requested_splits is None
                else requested_splits[0]
            )
            grid_batch = b * q_groups

    if cutlass.const_expr(cfg.use_persistent_scheduler):
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (grid_x, h_k, b),
            (1, 1, 1),
        )
        grid = tile_sched_params.get_grid_shape()
    else:
        tile_sched_params = None
        grid = (grid_x, h_k, grid_batch)
    # cluster distributed-SMEM reduction groups the splits_kv split CTAs
    # of each sequence (contiguous in grid-X) into one cluster so they can write
    # partials into each other's shared memory via prims.mapa.
    cluster_shape = [1, 1, 1]
    if cutlass.const_expr(cfg.use_cluster_smem_reduction):
        cluster_shape = [cfg.splits_kv, 1, 1]
    null_sparse_route_ptr = cute.make_ptr(
        Int32,
        0,
        mem_space=cutlass.AddressSpace.gmem,
    )
    # Both schedules are compiled together and selected by runtime bounds.
    # A CTA-relative 32-ID stage covers any instruction-aligned short span;
    # longer spans retain the ordinary staged pipeline. Counts never enter
    # the compile identity or require another JIT when this choice changes.
    can_hold_runtime_window = (
        cfg.runtime_split_kv
        and requested_splits is not None
        and use_native_paged_kv
        and cfg.use_split_kv
        and not cfg.use_sliding_window_causal
        and not cfg.uses_scattered_page_route
        and not cfg.use_block_sparse
        and cfg.num_insts_kv * (cfg.tile_size_kv // cfg.num_tokens_per_page)
        <= cfg.runtime_page_window_pages
    )
    use_runtime_window = cutlass.Boolean(False)
    if cutlass.const_expr(can_hold_runtime_window):
        pages_per_tile = cfg.tile_size_kv // cfg.num_tokens_per_page
        window_tiles = (
            cfg.runtime_page_window_pages // pages_per_tile // cfg.num_insts_kv
        ) * cfg.num_insts_kv
        use_runtime_window = s_k <= requested_splits[0] * Int32(
            window_tiles * cfg.tile_size_kv
        )
    for profile in cutlass.range_constexpr(2 if can_hold_runtime_window else 1):
        launch_cfg = replace(cfg, runtime_page_window=(profile == 1))
        if use_runtime_window == cutlass.Boolean(profile == 1):
            decode_gen_kernel(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
                tma_desc_k,
                tma_desc_v,
                o_iter,
                s_k,
                h_k,
                softmax_scale_operand,
                output_scale,
                seqlens_kv_iter,
                cu_seqlens_q_iter,
                page_idx_kv_iter,
                k_sf_iter,
                v_sf_iter,
                page_table_stride,
                page_table_capacity,
                q_token_kv_block_sparse_page_memberships_iter,
                q_token_kv_block_sparse_page_membership_stride,
                partial_o_iter,
                partial_stats_iter,
                split_kv_counter_iter,
                attention_sinks_iter,
                (runtime_q_capacity_rows if not cfg.fixed_seqlen else h_r),
                tile_sched_params,
                launch_cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                null_sparse_route_ptr,
                null_sparse_route_ptr,
                null_sparse_route_ptr,
                static_full_split_prefix,
                g_sage_scales=sage_scales_iter,
                lse_ptr=lse_ptr,
                requested_splits=requested_splits,
                g_max_q_len=max_q_len,
            ).launch(
                grid=grid,
                block=[cfg.threads_per_cta, 1, 1],
                cluster=cluster_shape,
                stream=stream,
                min_blocks_per_mp=decode_min_blocks_per_mp(
                    cfg, None if seq_len_kv is None else effective_seq_len_kv
                ),
                # PDL-capable attention initializes all local task resources before
                # acquiring its immediately preceding producer. Split attention also
                # defers runtime prefix pruning until after that acquire.
                use_pdl=cfg.use_pdl,
            )


@cute.jit
def release_decode_dependents(
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Uniformly release a following PDL grid after one CTA's true tail."""
    if cutlass.const_expr(cfg.use_parallel_separate_reduction_pdl):
        prims.barrier_cta_sync(14)
        prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def retire_inactive_pdl_split(
    is_active: cutlass.Boolean,
    tmem_ptr_i32: cute.Pointer,
    tmem_alloc_cols: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Cleanly retire a post-acquire split without staging TaskManager."""
    if not is_active:
        deallocate_decode_tmem(tmem_ptr_i32, tmem_alloc_cols, cfg)
        release_decode_dependents(cfg)
    # A runtime branch around ``TaskManager.run`` would make the DSL carry the
    # Python TaskManager meta object through ``scf.if``. Retiring only the
    # inactive, CTA-uniform path here leaves the active task graph straight-line.
    exit_cta_if_inactive(is_active)


def round_up_tmem_columns(num_columns: int) -> int:
    """tcgen05_alloc requires a power-of-two column count in [32, 512]."""
    return max(32, 1 << (num_columns - 1).bit_length())


@cute.jit
def run_decode_gen_active(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    softmax_scale_operand: Float32 | cute.Pointer,
    output_scale: Float32 | cute.Pointer,
    g_seqlens_kv: cute.Pointer | DirectSparseMetadataView,
    g_cu_seqlens_q: cute.Pointer | UniformQueryOffsets,
    g_page_idx_kv: cute.Pointer | DirectSparseMetadataView,
    g_k_sf: cute.Pointer,
    g_v_sf: cute.Pointer,
    g_page_table_stride: Int64,
    g_page_table_capacity: Int32,
    g_q_token_kv_block_sparse_page_memberships: cute.Pointer,
    g_q_token_kv_block_sparse_page_membership_stride: Int32,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_split_kv_counter: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    q_group_idx: Int32,
    h_k_idx: Int32,
    b_idx: Int32,
    q_token_offset: Int32,
    seq_len_q: Int32,
    active_splits_kv: Int32,
    static_full_split_prefix: cutlass.Constexpr[bool],
    defer_runtime_split_pruning: cutlass.Constexpr[bool],
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int | None] = 2048,
    use_variable_seqlens_kv: cutlass.Constexpr[bool] = False,
    use_native_paged_kv: cutlass.Constexpr[bool] = False,
    use_static_native_seqlens_kv: cutlass.Constexpr[bool] = False,
    g_sparse_row_route_offsets: cute.Pointer | None = None,
    g_sparse_row_route_counts: cute.Pointer | None = None,
    g_sparse_route_metadata: cute.Pointer | None = None,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    g_sage_scales: cute.Pointer | None = None,
    lse_ptr: cute.Pointer | None = None,
    requested_splits=None,
    g_max_q_len: Int32 = 1,
) -> None:
    """Run the complete decode body for one runtime-valid Q tile.

    Builds resources/tasks via `schedule.build_decode_gen_schedule` (shared with the
    validation path), then owns the matched TMA prefetch, SMEM/TMEM setup,
    TaskManager execution, and TMEM teardown lifecycle. Keeping that lifecycle
    in a void JIT helper lets the kernel wrapper omit it entirely for an
    overlaunched packed-Q tile without threading TaskManager state through a
    dynamic branch.
    """

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    bias_static_sliding_kv_tma = (
        cfg.use_sliding_window_causal
        and cfg.fixed_seqlen
        and cfg.max_seq_len_q == 1
        and not cfg.use_paged_kv
        and not cfg.use_split_kv
        and not use_variable_seqlens_kv
    )
    effective_seq_len_kv = configure_static_sliding_window(
        cfg, seq_len_kv, bias_static_sliding_kv_tma
    )
    total_kv_tiles = compute_total_kv_tiles(effective_seq_len_kv, cfg.tile_size_kv)
    cfg.total_kv_tiles = total_kv_tiles
    use_runtime_seqlens_kv = use_variable_seqlens_kv or (
        use_native_paged_kv and not use_static_native_seqlens_kv
    )
    runtime_seqlens_kv = (
        g_seqlens_kv if cutlass.const_expr(use_runtime_seqlens_kv) else None
    )
    runtime_max_seq_len_kv = (
        g_s_k
        if cutlass.const_expr(use_runtime_seqlens_kv or not cfg.fixed_seqlen)
        else Int32(cfg.static_seq_len_kv)
    )
    tma_desc_k_summary_ptr = None
    tma_desc_v_summary_ptr = None
    tma_desc_k_summary_atom_ptr = None
    tma_desc_v_summary_atom_ptr = None
    if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
        assert tma_desc_k_summary is not None
        assert tma_desc_v_summary is not None
        assert tma_desc_k_summary_atom is not None
        assert tma_desc_v_summary_atom is not None
        tma_desc_k_summary_ptr = tma_desc_k_summary.get_ptr()
        tma_desc_v_summary_ptr = tma_desc_v_summary.get_ptr()
        tma_desc_k_summary_atom_ptr = tma_desc_k_summary_atom.get_ptr()
        tma_desc_v_summary_atom_ptr = tma_desc_v_summary_atom.get_ptr()

    # Prefetch TMA
    uses_atom_desc = False
    if cutlass.const_expr(cfg.use_block_sparse):
        _, _, uses_atom_desc = block_sparse_contiguous_kv_copy_geometry(
            kv_block_size=cfg.kv_block_size,
            kv_route_size=cfg.tile_size_kv,
        )
    init_warp = 1
    if warp_idx == init_warp:
        prims.prefetch_tensormap(tma_desc_q.get_ptr())
        prims.prefetch_tensormap(tma_desc_k.get_ptr())
        prims.prefetch_tensormap(tma_desc_v.get_ptr())
        if cutlass.const_expr(cfg.uses_nvfp4_scale_tensors):
            prims.prefetch_tensormap(tma_desc_k_sf.get_ptr())
            prims.prefetch_tensormap(tma_desc_v_sf.get_ptr())
        if cutlass.const_expr(cfg.use_block_sparse and uses_atom_desc):
            # KV256 and non-aligned coarse KV128 may select the exact atom maps.
            prims.prefetch_tensormap(tma_desc_k_atom.get_ptr())
            prims.prefetch_tensormap(tma_desc_v_atom.get_ptr())
        if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
            if cutlass.const_expr(cfg.tile_size_kv != 256):
                prims.prefetch_tensormap(tma_desc_k_summary_ptr)
                prims.prefetch_tensormap(tma_desc_v_summary_ptr)
            if cutlass.const_expr(uses_atom_desc):
                prims.prefetch_tensormap(tma_desc_k_summary_atom_ptr)
                prims.prefetch_tensormap(tma_desc_v_summary_atom_ptr)
    init_warp += 1

    q_output_rows = g_h_r
    if cutlass.const_expr(cfg.fixed_seqlen and cfg.is_multi_token_q):
        q_output_rows = g_h_r * Int32(cfg.max_seq_len_q)

    # Static block-sparse tiles read their prepared row header here, before
    # TMEM allocation and barrier setup, so that global-memory round trip is
    # hidden instead of stalling every task at its first schedule step.
    sparse_row_route_begin = None
    sparse_route_count = None
    if cutlass.const_expr(
        cfg.use_block_sparse
        and not cfg.use_persistent_scheduler
        and g_sparse_row_route_offsets is not None
        and g_sparse_row_route_counts is not None
    ):
        sparse_row_route_begin, sparse_route_count = prefetch_prepared_sparse_row(
            cfg,
            g_sparse_row_route_offsets,
            g_sparse_row_route_counts,
            q_group_idx,
            h_k_idx,
            b_idx,
            g_h_k,
        )

    (
        task_list,
        dep_graph,
        dma_consumer_release_labels,
        smem_allocator,
        tmem_allocator,
        eager_init_resources,
    ) = build_decode_gen_schedule(
        cfg,
        total_kv_tiles,
        scale_softmax_log2=(
            Float32(1.0)
            if isinstance(softmax_scale_operand, cute.Pointer)
            else softmax_scale_operand
        ),
        o_ptr=o_iter,
        output_scale=(
            Float32(1.0) if isinstance(output_scale, cute.Pointer) else output_scale
        ),
        partial_o_ptr=g_partial_o,
        partial_stats_ptr=g_partial_stats,
        split_kv_counter_ptr=g_split_kv_counter,
        attention_sinks_ptr=g_attention_sinks,
        seqlens_kv=runtime_seqlens_kv,
        cu_seqlens_q=g_cu_seqlens_q,
        max_seq_len_kv=runtime_max_seq_len_kv,
        corr_max_seq_len_kv=runtime_max_seq_len_kv,
        num_heads_kv=g_h_k,
        h_r=q_output_rows,
        max_q_len=g_max_q_len,
        tma_desc_q=tma_desc_q.get_ptr(),
        tma_desc_k=tma_desc_k.get_ptr(),
        tma_desc_v=tma_desc_v.get_ptr(),
        tma_desc_k_sf=tma_desc_k_sf.get_ptr(),
        tma_desc_v_sf=tma_desc_v_sf.get_ptr(),
        tma_desc_k_atom=tma_desc_k_atom.get_ptr(),
        tma_desc_v_atom=tma_desc_v_atom.get_ptr(),
        tma_desc_k_summary=tma_desc_k_summary_ptr,
        tma_desc_v_summary=tma_desc_v_summary_ptr,
        tma_desc_k_summary_atom=tma_desc_k_summary_atom_ptr,
        tma_desc_v_summary_atom=tma_desc_v_summary_atom_ptr,
        sage_scales_ptr=g_sage_scales,
        page_idx_kv=g_page_idx_kv,
        page_table_stride=g_page_table_stride,
        page_table_capacity=g_page_table_capacity,
        q_token_kv_block_sparse_page_memberships=g_q_token_kv_block_sparse_page_memberships,
        q_token_kv_block_sparse_page_membership_stride=g_q_token_kv_block_sparse_page_membership_stride,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        q_group_idx=q_group_idx,
        q_token_offset=q_token_offset,
        seq_len_q=seq_len_q,
        active_splits_kv=(None if defer_runtime_split_pruning else active_splits_kv),
        static_full_split_prefix=static_full_split_prefix,
        tile_sched_params=tile_sched_params,
        use_variable_seqlens_kv=use_variable_seqlens_kv,
        use_native_paged_kv=use_native_paged_kv,
        use_static_native_seqlens_kv=use_static_native_seqlens_kv,
        sparse_row_route_offsets=g_sparse_row_route_offsets,
        sparse_row_route_counts=g_sparse_row_route_counts,
        sparse_route_metadata=g_sparse_route_metadata,
        sparse_row_route_begin=sparse_row_route_begin,
        sparse_route_count=sparse_route_count,
        lse_ptr=lse_ptr,
        requested_splits=requested_splits,
    )

    smem_allocator.allocate()

    tmem_cols = tmem_allocator.total_tmem_columns
    tmem_alloc_cols = round_up_tmem_columns(tmem_cols)
    tmem_ptr_alloc = smem_allocator.tmem_ptr_alloc
    assert tmem_ptr_alloc is not None
    tmem_ptr_i32 = smem_allocator.get(tmem_ptr_alloc)
    if warp_idx == init_warp:
        prims.tcgen05_alloc(tmem_ptr_i32, Int32(tmem_alloc_cols))
        prims.tcgen05_relinquish_alloc_permit()
    init_warp += 1

    task_manager = TaskManager(
        tasks=task_list,
        resource_dependency_graph=dep_graph,
        dma_consumer_release_labels=dma_consumer_release_labels,
        skip_validation=True,
        verbose=False,
        # Kernel construction is a latency-sensitive JIT path and may run once
        # per CUDA-graph bucket on every TP rank.  Keep the cheap structural
        # checks here, but leave the exhaustive interleaving proof to
        # build_decode_task_manager() and its offline tests.  `skip_validation`
        # only changes failures into warnings; it does not skip the expensive
        # state-space search.
        exhaustive_deadlock_race_check=False,
        assume_pdl_wait_completed=cfg.use_pdl,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
    )

    task_manager.setup_resources_and_tasks()
    resource_context = ResourceContext(
        smem_base=smem_allocator.smem_base,
        tmem_ptr_i32=tmem_ptr_i32,
    )
    for resource in eager_init_resources:
        # Materialize manually managed resource state before TS tasks start.
        # This covers correction's cluster transaction barriers and KV256's
        # per-fragment P-ready barriers.
        resource.create_function_variables(resource_context)
    # Ensure every CTA thread observes initialized mbarriers and SMEM resource
    # bases before any task body can use them.
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(cfg.supports_cluster_smem_reduction):
        # Cluster-wide visibility point for the per-CTA transaction mbarriers.
        # After this, peers may safely signal an owner CTA's mbarrier via mapa.
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()
    if cutlass.const_expr(cfg.use_pdl):
        # Delay the acquire until every CTA-local pipeline barrier and
        # SMEM/TMEM resource is initialized, but keep it before TaskManager
        # resolves any value published by the preceding producer grid.
        prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
    if cutlass.const_expr(defer_runtime_split_pruning):
        # A split launch cannot read its producer-owned sequence length until
        # the dependency above is acquired. Every physical split CTA
        # initializes the same static task resources first; only the useful
        # runtime prefix enters TaskManager. Pruned CTAs take the equivalent
        # TMEM teardown and dependent-release helpers, then retire before the
        # straight-line task graph.
        runtime_seq_len_kv = g_s_k
        if cutlass.const_expr(
            use_variable_seqlens_kv
            or (use_native_paged_kv and not use_static_native_seqlens_kv)
        ):
            runtime_seq_len_kv = Int32(g_seqlens_kv[b_idx])
        split_idx = Int32(cute.arch.block_idx()[0])
        if cutlass.const_expr(not cfg.uses_compact_split_grid):
            split_idx = split_idx % Int32(cfg.splits_kv)
        if cutlass.const_expr(cfg.runtime_split_kv and requested_splits is not None):
            active_splits = runtime_active_splits_kv(
                cfg,
                runtime_seq_len_kv,
                seq_len_q,
                q_group_token_base(cfg, q_group_idx),
                requested_splits=requested_splits,
            )
            run_task_graph = cute.arch.make_warp_uniform(split_idx < active_splits)
        else:
            run_task_graph = cute.arch.make_warp_uniform(
                runtime_split_kv_is_active(
                    cfg,
                    runtime_seq_len_kv,
                    seq_len_q,
                    q_group_token_base(cfg, q_group_idx),
                    split_idx,
                )
            )
        retire_inactive_pdl_split(
            run_task_graph,
            tmem_ptr_i32,
            Int32(tmem_alloc_cols),
            cfg,
        )
    # Scales are first consumed by the task bodies. Bind their device values
    # after the producer acquire so resource initialization can overlap it.
    if cutlass.const_expr(isinstance(softmax_scale_operand, cute.Pointer)):
        scale_softmax_log2 = cute.make_tensor(
            softmax_scale_operand, cute.make_layout(1)
        )[0] * Float32(math.log2(math.e))
        for resource in task_manager.resources:
            if cutlass.const_expr(
                isinstance(
                    resource,
                    (
                        TmemSResource,
                        SmemPResource,
                        TmemOResource,
                        TmemSoftmaxGlobalResource,
                        TmemCorrResource,
                    ),
                )
            ):
                resource.scale_softmax_log2 = scale_softmax_log2
    if cutlass.const_expr(isinstance(output_scale, cute.Pointer)):
        resolved_output_scale = cute.make_tensor(output_scale, cute.make_layout(1))[0]
        for resource in task_manager.resources:
            if cutlass.const_expr(isinstance(resource, TmemCorrResource)):
                resource.output_scale = resolved_output_scale
    task_manager.run()

    # Every peer async-stores into an owner CTA's distributed SMEM and charges
    # the bytes to that owner's transaction mbarrier. Producer-only CTAs are
    # never remote-SMEM targets, so they may retire after their store issues;
    # each owner remains resident until its mbarrier completes and its local
    # reduction consumes all delivered partials. A final cluster rendezvous
    # would therefore only make completed producers wait for the owners.

    # Keeps aliases score/stat TMEM columns across task warp groups, while the
    # swaps path needs only its correction-warp rendezvous. In both cases the
    # following PDL grid is released only after deallocation completes.
    deallocate_decode_tmem(tmem_ptr_i32, Int32(tmem_alloc_cols), cfg)
    release_decode_dependents(cfg)


@cute.jit
def run_decode_gen_inactive_cluster_rank() -> None:
    """Join cluster initialization, then retire an inactive physical split rank."""
    # The physical cluster remains configured-max sized. Initialize a local
    # zero-traffic barrier and join the same cluster visibility point as active
    # ranks. Active peers never address this rank after runtime contraction.
    inactive_mbarrier = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    thread_idx, _, _ = cute.arch.thread_idx()
    if thread_idx == Int32(0):
        prims.mbarrier_init(inactive_mbarrier.data_ptr(), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()


@cute.jit
def run_decode_gen_runtime_prefix(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_sf: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k_atom: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v_atom: cutlass.GridConstant[cuda.TensorMap],
    o_iter: cute.Pointer,
    g_s_k: Int32,
    g_h_k: Int32,
    softmax_scale_operand: Float32 | cute.Pointer,
    output_scale: Float32 | cute.Pointer,
    g_seqlens_kv: cute.Pointer | DirectSparseMetadataView,
    g_cu_seqlens_q: cute.Pointer | UniformQueryOffsets,
    g_page_idx_kv: cute.Pointer | DirectSparseMetadataView,
    g_k_sf: cute.Pointer,
    g_v_sf: cute.Pointer,
    g_page_table_stride: Int64,
    g_page_table_capacity: Int32,
    g_q_token_kv_block_sparse_page_memberships: cute.Pointer,
    g_q_token_kv_block_sparse_page_membership_stride: Int32,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_split_kv_counter: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    q_group_cta_idx: Int32,
    q_group_idx: Int32,
    h_k_idx: Int32,
    b_idx: Int32,
    q_token_offset: Int32,
    seq_len_q: Int32,
    q_token_base: Int32,
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    seq_len_kv: cutlass.Constexpr[int | None],
    use_variable_seqlens_kv: cutlass.Constexpr[bool],
    use_native_paged_kv: cutlass.Constexpr[bool],
    use_static_native_seqlens_kv: cutlass.Constexpr[bool],
    g_sparse_row_route_offsets: cute.Pointer | None,
    g_sparse_row_route_counts: cute.Pointer | None,
    g_sparse_route_metadata: cute.Pointer | None,
    tma_desc_k_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_k_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    tma_desc_v_summary_atom: cutlass.GridConstant[cuda.TensorMap] | None = None,
    g_sage_scales: cute.Pointer | None = None,
    lse_ptr: cute.Pointer | None = None,
    requested_splits=None,
    g_max_q_len: Int32 = 1,
) -> None:
    """Run the general runtime split-prefix producer or retire its suffix."""

    split_is_active = cutlass.Boolean(True)
    active_splits_kv = Int32(1)
    if cutlass.const_expr(cfg.use_split_kv):
        runtime_seq_len_kv = g_s_k
        if cutlass.const_expr(
            use_variable_seqlens_kv
            or (use_native_paged_kv and not use_static_native_seqlens_kv)
        ):
            runtime_seq_len_kv = Int32(g_seqlens_kv[b_idx])
        active_splits_kv = runtime_active_splits_kv(
            cfg,
            runtime_seq_len_kv,
            seq_len_q,
            q_token_base,
            requested_splits=requested_splits,
        )
        if cutlass.const_expr(not cfg.use_separate_reduction_kernel):
            # Preserve one neutral producer for fused empty-K semantics.
            active_splits_kv = cute.math.max(active_splits_kv, Int32(1))
            if cutlass.const_expr(
                cfg.uses_q_token_kv_block_sparse_page_route
                and not cfg.supports_cluster_smem_reduction
            ):
                # When only the final configured split is empty, retaining its
                # neutral producer lets the fused GMEM reducer use the fully
                # unrolled prefix instead of its dynamically guarded fallback.
                # This also avoids changing the grid or adding a reducer launch.
                if active_splits_kv == Int32(cfg.splits_kv - 1):
                    active_splits_kv = Int32(cfg.splits_kv)
        split_idx = Int32(cute.arch.block_idx()[0])
        if cutlass.const_expr(not cfg.uses_compact_split_grid):
            split_idx = q_group_cta_idx % Int32(cfg.splits_kv)
        split_is_active = split_idx < active_splits_kv

    if cutlass.const_expr(cfg.supports_cluster_smem_reduction):
        if split_is_active:
            run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
                tma_desc_k_atom,
                tma_desc_v_atom,
                o_iter,
                g_s_k,
                g_h_k,
                softmax_scale_operand,
                output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                active_splits_kv,
                False,
                False,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                g_sage_scales=g_sage_scales,
                lse_ptr=lse_ptr,
                requested_splits=requested_splits,
                g_max_q_len=g_max_q_len,
            )
        else:
            run_decode_gen_inactive_cluster_rank()
    else:
        if split_is_active:
            run_decode_gen_active(
                tma_desc_q,
                tma_desc_k,
                tma_desc_v,
                tma_desc_k_sf,
                tma_desc_v_sf,
                tma_desc_k_atom,
                tma_desc_v_atom,
                o_iter,
                g_s_k,
                g_h_k,
                softmax_scale_operand,
                output_scale,
                g_seqlens_kv,
                g_cu_seqlens_q,
                g_page_idx_kv,
                g_k_sf,
                g_v_sf,
                g_page_table_stride,
                g_page_table_capacity,
                g_q_token_kv_block_sparse_page_memberships,
                g_q_token_kv_block_sparse_page_membership_stride,
                g_partial_o,
                g_partial_stats,
                g_split_kv_counter,
                g_attention_sinks,
                g_h_r,
                q_group_idx,
                h_k_idx,
                b_idx,
                q_token_offset,
                seq_len_q,
                active_splits_kv,
                False,
                False,
                tile_sched_params,
                cfg,
                seq_len_kv,
                use_variable_seqlens_kv,
                use_native_paged_kv,
                use_static_native_seqlens_kv,
                g_sparse_row_route_offsets,
                g_sparse_row_route_counts,
                g_sparse_route_metadata,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                g_sage_scales=g_sage_scales,
                lse_ptr=lse_ptr,
                requested_splits=requested_splits,
                g_max_q_len=g_max_q_len,
            )
        else:
            signal_padded_pdl_producer(cfg)


@cute.jit
def signal_padded_pdl_producer(cfg: cutlass.Constexpr[FmhaDecodeConfig]) -> None:
    """Preserve PDL ordering through a zero-work producer CTA."""
    if cutlass.const_expr(cfg.use_pdl):
        # This CTA has no task resources or producer-owned work, but it must
        # acquire the incoming dependency before retiring.
        prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
    if cutlass.const_expr(cfg.use_parallel_separate_reduction_pdl):
        prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)
