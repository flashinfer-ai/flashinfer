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

"""Split-KV reduction body for the throughput 2CTA MLA TS kernel."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims

from .config import (
    REDUCTION_THREADS_PER_ROW,
    REDUCTION_VALUES_PER_THREAD,
    REDUCTION_VECTOR_BYTES,
)
from ..helpers.constants import SPLIT_REDUCTION_SCALE_BARRIER_ID
from ..helpers.math import ceil_div
from ..helpers.mask import MaskType, mask_visible_k_length
from ..helpers.ops import (
    fmax_f32,
    warp_reduce_max_f32,
    warp_reduce_sum_f32,
    vector_from_scalars,
)
from ..helpers.query import (
    flat_query_row_state,
    query_batch_bounds,
)
from .work_partition import (
    runtime_equal_split_row_prefix_active_split_count,
    runtime_row_prefix_active_split_count,
    runtime_split_kv_cap,
)


@cute.jit
def zero_balanced_output(
    kernel,
    output,
    lse,
    cu_seqlens_q,
    cfg,
    batch_idx,
    rows_per_cta: cutlass.Constexpr[int],
):
    """Publish deterministic empty-attention output from one reducer CTA.

    The caller uses the reducer's physical batch slot rather than a compact
    combine descriptor. Each thread retains the normal reducer's row/vector
    ownership, but no partial workspace or shared memory is touched.
    """

    reduction_group_idx, query_tile_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row_in_cta = tidx // Int32(REDUCTION_THREADS_PER_ROW)
    row_thread_idx = tidx - row_in_cta * Int32(REDUCTION_THREADS_PER_ROW)
    physical_tile_size_q = (
        cfg.mma_qk_tiler[0] if hasattr(cfg, "mma_qk_tiler") else cfg.tile_size_q
    )
    physical_tile_rows = Int32(physical_tile_size_q)
    local_flat_query_row = reduction_group_idx * Int32(rows_per_cta) + row_in_cta
    row_is_valid = local_flat_query_row < physical_tile_rows
    row_in_tile = cute.math.min(local_flat_query_row, physical_tile_rows - Int32(1))
    logical_num_heads = kernel.num_heads
    logical_seq_len_q = kernel.seq_len_q
    if cutlass.const_expr(hasattr(kernel, "logical_num_heads")):
        logical_num_heads = kernel.logical_num_heads
        logical_seq_len_q = kernel.logical_seq_len_q
    (
        storage_flat_query_row,
        _,
        _,
        _,
        mapped_query_is_valid,
    ) = flat_query_row_state(
        row_in_tile,
        query_tile_idx,
        physical_tile_size_q,
        logical_num_heads,
        logical_seq_len_q,
        cu_seqlens_q,
        batch_idx,
    )
    query_is_valid = row_is_valid and mapped_query_is_valid
    public_flat_query_row = storage_flat_query_row
    if cutlass.const_expr(cu_seqlens_q is None):
        public_flat_query_row += batch_idx * Int32(
            logical_seq_len_q * logical_num_heads
        )

    if query_is_valid:
        element_idx = row_thread_idx * Int32(REDUCTION_VALUES_PER_THREAD)
        zero = vector_from_scalars(
            (
                Float32(0.0),
                Float32(0.0),
                Float32(0.0),
                Float32(0.0),
                Float32(0.0),
                Float32(0.0),
                Float32(0.0),
                Float32(0.0),
            ),
            dtype=Float32,
        )
        output_elem_offset = Int64(public_flat_query_row) * Int64(
            cfg.latent_dim
        ) + Int64(element_idx)
        (output.iterator.raw_ptr() + output_elem_offset).store(
            zero.to(output.element_type),
            alignment=REDUCTION_VECTOR_BYTES,
        )
        if row_thread_idx == Int32(0):
            (lse.iterator.raw_ptr() + Int64(public_flat_query_row)).store(
                -kernel.lse_dtype.inf
            )


@cute.jit
def zero_balanced_inactive_outputs(
    kernel,
    output,
    lse,
    cache_seqs,
    cu_seqlens_q,
    cfg,
    reducer_capacity: cutlass.Constexpr[int],
    rows_per_cta: cutlass.Constexpr[int],
):
    """Publish empty requests from a compact, graph-stable reducer grid.

    Zero-KV requests intentionally have neither a producer work descriptor nor
    a combine descriptor. Consequently no mainloop or ordinary reduction CTA
    owns their public output. Grid-striding them here replaces output left by a
    previous replay with zero and replaces private LSE with negative infinity.
    """

    _, _, reducer_slot = cute.arch.block_idx()
    batch_idx = reducer_slot
    while batch_idx < Int32(kernel.batch_size):
        if cache_seqs[batch_idx] == Int32(0):
            zero_balanced_output(
                kernel,
                output,
                lse,
                cu_seqlens_q,
                cfg,
                batch_idx,
                rows_per_cta,
            )
        batch_idx += Int32(reducer_capacity)


@cute.jit
def run_reduction_kernel(
    kernel,
    output,
    lse,
    acc_output,
    acc_lse,
    split_kv,
    cache_seqs,
    cu_seqlens_q,
    block_split_kvs,
    balanced_split_count,
    balanced_split_begin,
    cfg,
    max_splits: cutlass.Constexpr[int],
    rows_per_cta: cutlass.Constexpr[int],
    balanced_batch_idx,
):
    """Combine consecutive logical split rows in one reference-reducer CTA.

    Each 64-thread row group owns one D512 output row.  Its first warp computes
    FP32 LSE rescale factors, both warps consume one contiguous 16-byte BF16
    fragment per thread, and the final accumulation remains FP32.  The launch
    is flattened over logical query rows, then each row is decomposed back into
    its physical M128 workspace tile and row.  Only the last CTA can contain
    inactive row groups; those groups still synchronize but never access the
    workspace or publish public output.
    """
    reduction_group_idx, balanced_query_tile_idx, batch_idx = cute.arch.block_idx()
    if cutlass.const_expr(kernel.use_balanced_scheduler):
        batch_idx = balanced_batch_idx
    tidx, _, _ = cute.arch.thread_idx()
    row_in_cta = tidx // Int32(REDUCTION_THREADS_PER_ROW)
    row_thread_idx = tidx - row_in_cta * Int32(REDUCTION_THREADS_PER_ROW)
    threads_per_warp = cfg.threads_per_warp if hasattr(cfg, "threads_per_warp") else 32
    row_warp_idx = cute.arch.make_warp_uniform(
        row_thread_idx // Int32(threads_per_warp)
    )
    lane_idx = row_thread_idx % Int32(threads_per_warp)

    physical_tile_size_q = (
        cfg.mma_qk_tiler[0] if hasattr(cfg, "mma_qk_tiler") else cfg.tile_size_q
    )
    physical_tile_rows = Int32(physical_tile_size_q)
    local_flat_query_row = reduction_group_idx * Int32(rows_per_cta) + row_in_cta
    if cutlass.const_expr(kernel.use_balanced_scheduler):
        is_one_cta = hasattr(kernel, "logical_num_heads")
        logical_num_heads = kernel.num_heads
        logical_seq_len_q = kernel.seq_len_q
        if cutlass.const_expr(is_one_cta):
            logical_num_heads = kernel.logical_num_heads
            logical_seq_len_q = kernel.logical_seq_len_q
        row_is_valid = local_flat_query_row < physical_tile_rows
        row_in_tile = cute.math.min(local_flat_query_row, physical_tile_rows - Int32(1))
        query_tile_idx = balanced_query_tile_idx
        (
            storage_flat_query_row,
            _,
            logical_q_idx,
            _,
            mapped_query_is_valid,
        ) = flat_query_row_state(
            row_in_tile,
            query_tile_idx,
            physical_tile_size_q,
            logical_num_heads,
            logical_seq_len_q,
            cu_seqlens_q,
            batch_idx,
        )
    else:
        logical_query_rows = Int32(kernel.num_heads * kernel.seq_len_q)
        row_is_valid = local_flat_query_row < logical_query_rows
        safe_local_flat_query_row = cute.math.min(
            local_flat_query_row, logical_query_rows - Int32(1)
        )
        if cutlass.const_expr(
            kernel.num_heads * kernel.seq_len_q <= cfg.mma_qk_tiler[0]
        ):
            query_tile_idx = Int32(0)
            row_in_tile = safe_local_flat_query_row
        else:
            query_tile_idx = safe_local_flat_query_row // physical_tile_rows
            row_in_tile = (
                safe_local_flat_query_row - query_tile_idx * physical_tile_rows
            )
        if cutlass.const_expr(cu_seqlens_q is None):
            storage_flat_query_row = safe_local_flat_query_row
            logical_q_idx = storage_flat_query_row // Int32(kernel.num_heads)
            mapped_query_is_valid = True
        else:
            (
                storage_flat_query_row,
                _,
                logical_q_idx,
                _,
                mapped_query_is_valid,
            ) = flat_query_row_state(
                row_in_tile,
                query_tile_idx,
                cfg.mma_qk_tiler[0],
                kernel.num_heads,
                kernel.seq_len_q,
                cu_seqlens_q,
                batch_idx,
            )
    query_is_valid = row_is_valid and mapped_query_is_valid
    public_flat_query_row = storage_flat_query_row
    if cutlass.const_expr(cu_seqlens_q is None):
        public_seq_len_q = kernel.seq_len_q
        public_num_heads = kernel.num_heads
        if cutlass.const_expr(hasattr(kernel, "logical_num_heads")):
            # The balanced 1CTA producer pads logical flat-Q rows to its
            # physical tile shape. Public tensors retain the caller's logical
            # [B, SQ, H, ...] layout; only split-partial workspace uses the
            # padded dimensions.
            public_seq_len_q = kernel.logical_seq_len_q
            public_num_heads = kernel.logical_num_heads
        public_flat_query_row = public_flat_query_row + batch_idx * Int32(
            public_seq_len_q * public_num_heads
        )

    # The scalar split count remains the grid/workspace capacity. A variable-
    # split launch optionally contracts it with block_split_kvs[batch].
    if cutlass.const_expr(kernel.use_balanced_scheduler):
        split_kv_cap = Int32(balanced_split_count)
    elif cutlass.const_expr(
        kernel.static_split_kv is not None and not kernel.is_var_split_kv
    ):
        split_kv_cap = Int32(max_splits)
    else:
        split_kv_cap = runtime_split_kv_cap(
            split_kv,
            kernel.is_var_split_kv,
            block_split_kvs,
            batch_idx,
        )
    # Producer splits are sized from the group's largest K domain, while each
    # logical row consumes only the prefix containing its visible K tiles.
    tile_k = cache_seqs[batch_idx]
    row_k = tile_k
    tile_num_heads = kernel.num_heads
    tile_seq_len_q = kernel.seq_len_q
    if cutlass.const_expr(hasattr(kernel, "logical_num_heads")):
        tile_num_heads = kernel.logical_num_heads
        tile_seq_len_q = kernel.logical_seq_len_q
    if cutlass.const_expr(
        cfg.mask_type == MaskType.CAUSAL.value and tile_seq_len_q > 1
    ):
        _, runtime_logical_seq_len_q = query_batch_bounds(
            cu_seqlens_q,
            batch_idx,
            tile_seq_len_q,
        )
        _, _, tile_last_logical_q_idx, _, _ = flat_query_row_state(
            physical_tile_rows - Int32(1),
            query_tile_idx,
            physical_tile_size_q,
            tile_num_heads,
            tile_seq_len_q,
            cu_seqlens_q,
            batch_idx,
        )
        tile_k = mask_visible_k_length(
            cfg.mask_type,
            tile_k,
            tile_last_logical_q_idx,
            runtime_logical_seq_len_q,
        )
        row_k = mask_visible_k_length(
            cfg.mask_type,
            cache_seqs[batch_idx],
            logical_q_idx,
            runtime_logical_seq_len_q,
        )
    kv_tile_size = (
        cfg.mma_qk_tiler[1] if hasattr(cfg, "mma_qk_tiler") else cfg.tile_size_kv
    )
    tile_k_tile_total = (tile_k + kv_tile_size - 1) // kv_tile_size
    row_k_tile_total = (row_k + kv_tile_size - 1) // kv_tile_size
    if cutlass.const_expr(kernel.use_balanced_scheduler):
        # Balanced descriptors use quotient/remainder equal splits over the
        # full request. Reconstruct that exact prefix: deriving a uniform span
        # from the row-visible or tile-visible causal domain can omit the last
        # partially intersecting descriptor.
        request_k_tile_total = (
            cache_seqs[batch_idx] + kv_tile_size - 1
        ) // kv_tile_size
        local_split_kv = runtime_equal_split_row_prefix_active_split_count(
            row_k_tile_total,
            request_k_tile_total,
            split_kv_cap,
        )
    else:
        # Ordinary kernels use configured-span partitioning over the producer
        # tile's visible K domain.
        local_split_kv = runtime_row_prefix_active_split_count(
            row_k_tile_total,
            tile_k_tile_total,
            split_kv_cap,
        )

    smem_lse_scale = cutlass.Array(
        kernel.lse_dtype,
        rows_per_cta * max_splits,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    row_scale_offset = row_in_cta * Int32(max_splits)

    split_begin = Int32(0)
    acc_lse_batch_idx = batch_idx
    if cutlass.const_expr(kernel.use_balanced_scheduler):
        split_begin = Int32(balanced_split_begin)
        acc_lse_batch_idx = Int32(0)
    acc_lse_tile = acc_lse[row_in_tile, None, query_tile_idx, acc_lse_batch_idx]
    if row_warp_idx == 0:
        # The first warp for each row owns its log-sum-exp merge.  It publishes
        # one rescale factor per active split for the row's second warp too.
        lse_per_thread = ceil_div(max_splits, threads_per_warp)
        local_lse = cutlass.Array(kernel.lse_dtype, lse_per_thread)
        lse_max = kernel.lse_dtype(-kernel.lse_dtype.inf)
        for i in cutlass.range_constexpr(lse_per_thread):
            split_kv_idx = lane_idx + i * threads_per_warp
            active_slot = query_is_valid & cute.elem_less(
                split_kv_idx,
                local_split_kv,
            )
            local_lse[i] = -kernel.lse_dtype.inf
            # Keep the workspace access inside the dynamic predicate. Split
            # capacities need not be warp-aligned, so padded lanes must not
            # form an address beyond the producer allocation.
            if active_slot:
                local_lse[i] = acc_lse_tile[split_begin + split_kv_idx]
            lse_max = fmax_f32(lse_max, local_lse[i])
        lse_max = warp_reduce_max_f32(lse_max)
        lse_max = lse_max if lse_max != -kernel.lse_dtype.inf else 0.0
        sum_lse = kernel.lse_dtype(0.0)
        for i in cutlass.range_constexpr(lse_per_thread):
            sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
        sum_lse = warp_reduce_sum_f32(sum_lse)
        has_finite_mass = sum_lse == sum_lse and sum_lse != kernel.lse_dtype(0.0)
        global_lse = (
            lse_max + cute.math.log2(sum_lse, fastmath=True)
            if has_finite_mass
            else -kernel.lse_dtype.inf
        )
        if lane_idx == 0 and query_is_valid:
            (lse.iterator.raw_ptr() + Int64(public_flat_query_row)).store(global_lse)
        for i in cutlass.range_constexpr(lse_per_thread):
            split_kv_idx = lane_idx + i * threads_per_warp
            if cute.elem_less(split_kv_idx, local_split_kv):
                smem_lse_scale[row_scale_offset + split_kv_idx] = (
                    cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                    if has_finite_mass
                    else kernel.lse_dtype(0.0)
                )

    # Independent writer warps publish scales before any row consumes
    # them.  Invalid/padded rows participate so this remains a full CTA barrier.
    prims.barrier_cta_sync(SPLIT_REDUCTION_SCALE_BARRIER_ID)

    element_idx = row_thread_idx * Int32(REDUCTION_VALUES_PER_THREAD)
    acc_vec = vector_from_scalars(
        (
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
        ),
        dtype=Float32,
    )
    acc_output_ptr = acc_output.iterator.raw_ptr()
    if query_is_valid:
        if cutlass.const_expr(kernel.use_balanced_scheduler):
            partial_capacity = Int64(kernel.balanced_partial_capacity)
            partial_elem_offset_base = (
                Int64(query_tile_idx)
                * Int64(physical_tile_rows)
                * partial_capacity
                * Int64(cfg.latent_dim)
                + Int64(row_in_tile) * partial_capacity * Int64(cfg.latent_dim)
                + Int64(split_begin) * Int64(cfg.latent_dim)
                + Int64(element_idx)
            )
        else:
            partial_elem_offset_base = (
                Int64(batch_idx)
                * Int64(cute.size(acc_output.shape[3]))
                * Int64(physical_tile_rows)
                * Int64(split_kv)
                * Int64(cfg.latent_dim)
                + Int64(query_tile_idx)
                * Int64(physical_tile_rows)
                * Int64(split_kv)
                * Int64(cfg.latent_dim)
                + Int64(row_in_tile) * Int64(split_kv) * Int64(cfg.latent_dim)
                + Int64(element_idx)
            )
        partial_output_ptr = acc_output_ptr + partial_elem_offset_base
        # S2 is the common two-wave 2CTA decode reducer.  Materialize its two
        # fixed partials directly; larger reducers retain the compact dynamic
        # loop because fully unrolling S4 increases instruction pressure.
        if cutlass.const_expr(max_splits == 2):
            for i in cutlass.range_constexpr(max_splits):
                if Int32(i) < local_split_kv:
                    partial_vec = (
                        (partial_output_ptr + Int64(i * cfg.latent_dim))
                        .load(
                            count=REDUCTION_VALUES_PER_THREAD,
                            alignment=REDUCTION_VECTOR_BYTES,
                        )
                        .to(Float32)
                    )
                    scale = Float32(smem_lse_scale[row_scale_offset + Int32(i)])
                    acc_vec = acc_vec + partial_vec * scale
        else:
            for i in range(local_split_kv):
                partial_vec = (
                    (partial_output_ptr + Int64(i) * Int64(cfg.latent_dim))
                    .load(
                        count=REDUCTION_VALUES_PER_THREAD,
                        alignment=REDUCTION_VECTOR_BYTES,
                    )
                    .to(Float32)
                )
                scale = Float32(smem_lse_scale[row_scale_offset + i])
                acc_vec = acc_vec + partial_vec * scale

        output_elem_offset = Int64(public_flat_query_row) * Int64(
            cfg.latent_dim
        ) + Int64(element_idx)
        (output.iterator.raw_ptr() + output_elem_offset).store(
            acc_vec.to(output.element_type),
            alignment=REDUCTION_VECTOR_BYTES,
        )
