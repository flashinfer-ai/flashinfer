"""Selected-position specialization of the CuTe paged-forward engine.

The engine supports direct BF16 output for large QSA prefills and FP32 split
partials for latency-sensitive small launches. Cache strides specialize the
compiled callable so both entries consume zero-copy per-layer views from an
interleaved BLHNC allocation without carrying layout arithmetic as runtime
state. Every page-scaled cache offset is widened to Int64 before multiplication.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import (
    BFloat16,
    Float32,
    Float8E4M3FN,
    Int32,
    Int64,
    Uint32,
    const_expr,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import (
    warp_mma_gemm,
)
from b12x.attention._shared.contiguous.softmax import Softmax
from b12x.attention._shared.cute.ops import fmax
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compiler import run_compiled
from b12x._lib.compile_plan import compile_only_launches_enabled, program_keys, record_program
from b12x._lib.intrinsics import (
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    ld_global_nc_v4_u32,
    shared_ptr_to_u32,
    st_shared_v4_u32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._selected_forward_config import HEAD_DIM as _HEAD_DIM
from ._selected_forward_config import MAX_SPLITS as _NUM_SPLITS
from ._selected_forward_config import SELECTION_WIDTH as _SELECTION_WIDTH


_THREADS = 128
_WARPS = _THREADS // 32
_LOG2_E = 1.4426950408889634
_LOCK = RLock()
_KERNEL_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
_WARMED: dict[tuple[object, ...], Callable[..., None]] = {}
_MERGE_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
_MERGE_WARMED: dict[tuple[object, ...], Callable[..., None]] = {}


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


@dsl_user_op
def _exp2_approx(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pointer_as_int64(pointer, offset, *, loc=None, ip=None) -> Int64:
    element_pointer = pointer + offset
    return Int64(llvm.ptrtoint(T.i64(), element_pointer.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _cp_async_load_128b_zfill(
    smem_addr: Int32,
    gmem_addr: Int64,
    src_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _SelectedPositionPagedForwardKernel:
    """Paged-forward entry over caller-selected logical K/V positions."""

    def __init__(
        self,
        *,
        q_heads: int,
        kv_heads: int,
        kv_is_fp8: bool,
        direct_output: bool,
        return_lse: bool,
        kv_warps: int,
        page_size: int,
        key_strides: tuple[int, int, int],
        value_strides: tuple[int, int, int],
        selection_width: int = _SELECTION_WIDTH,
    ) -> None:
        self.q_heads = int(q_heads)
        self.selection_width = int(selection_width)
        if self.selection_width < _SELECTION_WIDTH:
            raise ValueError("selected-position capacity must cover the QSA selection")
        self.kv_heads = int(kv_heads)
        self.heads_per_kv = self.q_heads // self.kv_heads
        self.kv_is_fp8 = bool(kv_is_fp8)
        self.direct_output = bool(direct_output)
        self.return_lse = bool(return_lse)
        self.kv_warps = int(kv_warps)
        if self.kv_warps not in (1, 2, 4):
            raise ValueError(
                "selected-position paged forward requires 1, 2, or 4 KV warps"
            )
        self.tile_n = 16 * self.kv_warps
        self.page_size = int(page_size)
        if self.page_size <= 0:
            raise ValueError(
                "selected-position paged forward page_size must be positive"
            )
        self.key_page_stride, self.key_token_stride, self.key_head_stride = map(
            int, key_strides
        )
        (
            self.value_page_stride,
            self.value_token_stride,
            self.value_head_stride,
        ) = map(int, value_strides)
        self.shared_cache_strides = tuple(map(int, key_strides)) == tuple(
            map(int, value_strides)
        )
        self.qk_warps = 1 if self.heads_per_kv <= 16 else 2
        self.pv_warps = self.qk_warps * 4
        self.tile_m = self.qk_warps * 16
        self.threads = self.pv_warps * 32

    def _tiled_mma_qk(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.qk_warps, 1, 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )

    def _tiled_mma_pv(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.qk_warps, 4, 1),
            permutation_mnk=(self.tile_m, _HEAD_DIM, 16),
        )

    @cute.jit
    def _load_fp8_vector_to_bf16_shared(
        self,
        source: cute.Pointer,
        source_base: Int64,
        destination: cute.Tensor,
        token: Int32,
        dimension_base: Int32,
    ) -> None:
        x0 = Uint32(0)
        x1 = Uint32(0)
        x2 = Uint32(0)
        x3 = Uint32(0)
        if source_base >= Int64(0):
            x0, x1, x2, x3 = ld_global_nc_v4_u32(
                _pointer_as_int64(source, source_base + dimension_base.to(Int64))
            )
        y00, y01 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x0)
        y10, y11 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x1)
        y20, y21 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x2)
        y30, y31 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x3)
        destination0 = shared_ptr_to_u32(
            destination.iterator
            + cute.crd2idx((token, dimension_base), destination.layout)
        )
        destination1 = shared_ptr_to_u32(
            destination.iterator
            + cute.crd2idx(
                (token, dimension_base + Int32(8)),
                destination.layout,
            )
        )
        st_shared_v4_u32(destination0, y00, y01, y10, y11)
        st_shared_v4_u32(destination1, y20, y21, y30, y31)

    def _shared_layouts(
        self,
    ) -> tuple[
        cute.ComposedLayout,
        cute.ComposedLayout,
        cute.ComposedLayout,
        cute.ComposedLayout,
    ]:
        head_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                _HEAD_DIM,
            ),
            BFloat16,
        )
        probability_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                self.tile_n,
            ),
            BFloat16,
        )
        query_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_m, _HEAD_DIM),
            order=(0, 1),
        )
        key_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_n, _HEAD_DIM),
            order=(0, 1),
        )
        value_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_n, _HEAD_DIM),
            order=(0, 1),
        )
        probability_layout = cute.tile_to_shape(
            probability_layout_atom,
            (self.tile_m, self.tile_n),
            order=(0, 1),
        )
        return query_layout, key_layout, value_layout, probability_layout

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        k_descale: cute.Pointer,
        v_descale: cute.Pointer,
        block_table: cute.Pointer,
        request_ids: cute.Pointer,
        selected_positions: cute.Pointer,
        query_positions: cute.Pointer,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        num_cache_pages: Int64,
        table_batch: Int64,
        table_width: Int64,
        softmax_scale: Float32,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        query_layout, key_layout, value_layout, probability_layout = (
            self._shared_layouts()
        )
        tiled_mma_qk = self._tiled_mma_qk()
        tiled_mma_pv = self._tiled_mma_pv()
        self.kernel(
            query,
            key_cache,
            value_cache,
            k_descale,
            v_descale,
            block_table,
            request_ids,
            selected_positions,
            query_positions,
            partial_output,
            partial_lse,
            output,
            num_cache_pages,
            table_batch,
            table_width,
            softmax_scale,
            splits,
            query_layout,
            key_layout,
            value_layout,
            probability_layout,
            tiled_mma_qk,
            tiled_mma_pv,
        ).launch(
            grid=(rows, self.kv_heads, splits),
            block=(self.threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _mma_selected_tile(
        self,
        tile: Int32,
        row: Int32,
        kv_head: Int32,
        request_id: Int64,
        request_valid,
        query_position: Int64,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        block_table: cute.Pointer,
        selected_positions: cute.Pointer,
        key_bases: cute.Tensor,
        value_bases: cute.Tensor,
        shared_query: cute.Tensor,
        shared_key: cute.Tensor,
        shared_value: cute.Tensor,
        shared_probability: cute.Tensor,
        shared_row_scale: cute.Tensor,
        num_cache_pages: Int64,
        table_width: Int64,
        thread: Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        register_query: cute.Tensor,
        register_key: cute.Tensor,
        register_probability: cute.Tensor,
        register_value: cute.Tensor,
        copy_query: cute.TiledCopy,
        copy_key: cute.TiledCopy,
        copy_probability: cute.TiledCopy,
        copy_value: cute.TiledCopy,
        copy_source_query: cute.Tensor,
        copy_source_key: cute.Tensor,
        copy_source_probability: cute.Tensor,
        copy_source_value: cute.Tensor,
        accumulator_output: cute.Tensor,
        softmax: Softmax,
        is_first: cutlass.Constexpr,
    ) -> None:
        column = thread
        if column < Int32(self.tile_n):
            selected_column = tile * Int32(self.tile_n) + column
            valid = request_valid & (selected_column < Int32(self.selection_width))
            logical_position = Int64(-1)
            if valid:
                selected_offset = row.to(Int64) * Int64(
                    self.selection_width
                ) + selected_column.to(Int64)
                logical_position = selected_positions[selected_offset].to(Int64)
                valid = (logical_position >= Int64(0)) & (
                    logical_position <= query_position
                )

            physical_page = Int64(-1)
            page_offset = Int64(0)
            if valid:
                logical_page = logical_position // Int64(self.page_size)
                valid = logical_page < table_width
                if valid:
                    table_offset = request_id * table_width + logical_page
                    physical_page = block_table[table_offset].to(Int64)
                    valid = (physical_page >= Int64(0)) & (
                        physical_page < num_cache_pages
                    )
                    page_offset = logical_position % Int64(self.page_size)

            key_base = Int64(-1)
            value_base = Int64(-1)
            if valid:
                key_base = (
                    physical_page * Int64(self.key_page_stride)
                    + page_offset * Int64(self.key_token_stride)
                    + kv_head.to(Int64) * Int64(self.key_head_stride)
                )
                if const_expr(self.shared_cache_strides):
                    value_base = key_base
                else:
                    value_base = (
                        physical_page * Int64(self.value_page_stride)
                        + page_offset * Int64(self.value_token_stride)
                        + kv_head.to(Int64) * Int64(self.value_head_stride)
                    )
            key_bases[column] = key_base
            if const_expr(not self.shared_cache_strides):
                value_bases[column] = value_base
        cute.arch.sync_threads()

        if const_expr(not self.kv_is_fp8):
            vectors_per_row = _HEAD_DIM // 8
            for vector_iter in cutlass.range_constexpr(
                self.tile_n * vectors_per_row // self.threads
            ):
                linear_vector = thread + Int32(vector_iter * self.threads)
                token = linear_vector // Int32(vectors_per_row)
                vector = linear_vector % Int32(vectors_per_row)
                dimension_base = vector * Int32(8)
                key_base = Int64(key_bases[token])
                value_base = key_base
                if const_expr(not self.shared_cache_strides):
                    value_base = Int64(value_bases[token])
                key_valid = key_base >= Int64(0)
                value_valid = value_base >= Int64(0)
                safe_key_base = Int64(cutlass.select_(key_valid, key_base, Int64(0)))
                safe_value_base = Int64(
                    cutlass.select_(value_valid, value_base, Int64(0))
                )
                key_destination = shared_ptr_to_u32(
                    shared_key.iterator
                    + cute.crd2idx(
                        (token, dimension_base),
                        shared_key.layout,
                    )
                )
                value_destination = shared_ptr_to_u32(
                    shared_value.iterator
                    + cute.crd2idx(
                        (token, dimension_base),
                        shared_value.layout,
                    )
                )
                _cp_async_load_128b_zfill(
                    key_destination,
                    _pointer_as_int64(
                        key_cache,
                        safe_key_base + dimension_base.to(Int64),
                    ),
                    cutlass.select_(key_valid, Int32(16), Int32(0)),
                )
                _cp_async_load_128b_zfill(
                    value_destination,
                    _pointer_as_int64(
                        value_cache,
                        safe_value_base + dimension_base.to(Int64),
                    ),
                    cutlass.select_(value_valid, Int32(16), Int32(0)),
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        else:
            vectors_per_row = _HEAD_DIM // 16
            for vector_iter in cutlass.range_constexpr(
                self.tile_n * vectors_per_row // self.threads
            ):
                linear_vector = thread + Int32(vector_iter * self.threads)
                token = linear_vector // Int32(vectors_per_row)
                dimension_base = (linear_vector % Int32(vectors_per_row)) * Int32(16)
                key_base = Int64(key_bases[token])
                value_base = key_base
                if const_expr(not self.shared_cache_strides):
                    value_base = Int64(value_bases[token])
                self._load_fp8_vector_to_bf16_shared(
                    key_cache,
                    key_base,
                    shared_key,
                    token,
                    dimension_base,
                )
                self._load_fp8_vector_to_bf16_shared(
                    value_cache,
                    value_base,
                    shared_value,
                    token,
                    dimension_base,
                )
        cute.arch.sync_threads()

        if thread < Int32(self.qk_warps * 32):
            accumulator_score = cute.make_rmem_tensor(
                tiled_mma_qk.get_slice(thread).partition_shape_C(
                    (self.tile_m, self.tile_n)
                ),
                Float32,
            )
            accumulator_score.fill(0.0)
            warp_mma_gemm(
                tiled_mma_qk,
                accumulator_score,
                register_query,
                register_key,
                copy_source_query,
                copy_source_key,
                copy_query,
                copy_key,
                A_in_regs=not is_first,
            )
            score_mn = layout_utils.reshape_acc_to_mn(accumulator_score)
            score_coordinates = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(thread).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    if (coordinate[0] >= Int32(self.heads_per_kv)) | (
                        Int64(key_bases[coordinate[1]]) < Int64(0)
                    ):
                        score_mn[m, n] = -Float32.inf

            row_scale = softmax.online_softmax(
                accumulator_score,
                is_first=is_first,
                check_inf=True,
            )
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    shared_probability[coordinate[0], coordinate[1]] = BFloat16(
                        score_mn[m, n]
                    )
                if score_coordinates[m, 0][1] == Int32(0):
                    shared_row_scale[score_coordinates[m, 0][0]] = row_scale[m]

        cute.arch.sync_threads()

        output_mn = layout_utils.reshape_acc_to_mn(accumulator_output)
        output_coordinates = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(
                cute.make_identity_tensor((self.tile_m, _HEAD_DIM))
            )
        )
        if const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(output_mn.shape[0])):
                scale = Float32(shared_row_scale[output_coordinates[m, 0][0]])
                output_mn[m, None].store(output_mn[m, None].load() * scale)
        warp_mma_gemm(
            tiled_mma_pv,
            accumulator_output,
            register_probability,
            register_value,
            copy_source_probability,
            copy_source_value,
            copy_probability,
            copy_value,
        )
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        query: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        k_descale: cute.Pointer,
        v_descale: cute.Pointer,
        block_table: cute.Pointer,
        request_ids: cute.Pointer,
        selected_positions: cute.Pointer,
        query_positions: cute.Pointer,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        num_cache_pages: Int64,
        table_batch: Int64,
        table_width: Int64,
        softmax_scale: Float32,
        splits: Int32,
        query_layout: cute.ComposedLayout,
        key_layout: cute.ComposedLayout,
        value_layout: cute.ComposedLayout,
        probability_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
    ) -> None:
        row_idx, kv_head_idx, split_idx = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        thread = Int32(thread_idx)
        split = Int32(split_idx)
        kv_head = Int32(kv_head_idx)
        row = Int32(row_idx)
        request_id = request_ids[row].to(Int64)
        query_position = query_positions[row].to(Int64)
        request_valid = (request_id >= Int64(0)) & (request_id < table_batch)

        k_scale = Float32(1.0)
        v_scale = Float32(1.0)
        if const_expr(self.kv_is_fp8):
            k_scale = Float32(k_descale[0])
            v_scale = Float32(v_descale[0])

        allocator = cutlass.utils.SmemAllocator()
        shared_query = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=query_layout,
            byte_alignment=1024,
        )
        shared_key = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=key_layout,
            byte_alignment=1024,
        )
        shared_value = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=value_layout,
            byte_alignment=1024,
        )
        shared_probability = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=probability_layout,
            byte_alignment=1024,
        )
        shared_row_scale = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((self.tile_m,), stride=(1,)),
            byte_alignment=16,
        )
        key_bases = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((self.tile_n,), stride=(1,)),
            byte_alignment=16,
        )
        value_bases = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((self.tile_n,), stride=(1,)),
            byte_alignment=16,
        )

        query_element = thread
        first_query_head = kv_head * Int32(self.heads_per_kv)
        while query_element < Int32(self.tile_m * _HEAD_DIM):
            local_head = query_element // Int32(_HEAD_DIM)
            dimension = query_element - local_head * Int32(_HEAD_DIM)
            query_value = Float32(0.0)
            if local_head < Int32(self.heads_per_kv):
                query_head = first_query_head + local_head
                query_offset = (
                    row.to(Int64) * Int64(self.q_heads) + query_head.to(Int64)
                ) * Int64(_HEAD_DIM) + dimension.to(Int64)
                query_value = Float32(query[query_offset])
            shared_query[local_head, dimension] = BFloat16(query_value)
            query_element += Int32(self.threads)
        cute.arch.sync_threads()

        qk_thread = thread % Int32(self.qk_warps * 32)
        thread_mma_qk = tiled_mma_qk.get_slice(qk_thread)
        thread_mma_pv = tiled_mma_pv.get_slice(thread)
        register_query = thread_mma_qk.make_fragment_A(
            thread_mma_qk.partition_A(shared_query)
        )
        register_key = thread_mma_qk.make_fragment_B(
            thread_mma_qk.partition_B(shared_key)
        )
        shared_value_transposed = layout_utils.transpose_view(shared_value)
        register_probability = thread_mma_pv.make_fragment_A(
            thread_mma_pv.partition_A(shared_probability)
        )
        register_value = thread_mma_pv.make_fragment_B(
            thread_mma_pv.partition_B(shared_value_transposed)
        )
        copy_atom_query_key = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            BFloat16,
        )
        copy_atom_value = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            BFloat16,
        )
        copy_query = cute.make_tiled_copy_A(
            copy_atom_query_key, tiled_mma_qk
        ).get_slice(qk_thread)
        copy_key = cute.make_tiled_copy_B(copy_atom_query_key, tiled_mma_qk).get_slice(
            qk_thread
        )
        copy_probability = cute.make_tiled_copy_A(
            copy_atom_query_key, tiled_mma_pv
        ).get_slice(thread)
        copy_value = cute.make_tiled_copy_B(copy_atom_value, tiled_mma_pv).get_slice(
            thread
        )
        copy_source_query = copy_query.partition_S(shared_query)
        copy_source_key = copy_key.partition_S(shared_key)
        copy_source_probability = copy_probability.partition_S(shared_probability)
        copy_source_value = copy_value.partition_S(shared_value_transposed)
        accumulator_output = cute.make_rmem_tensor(
            thread_mma_pv.partition_shape_C((self.tile_m, _HEAD_DIM)),
            Float32,
        )
        accumulator_output.fill(0.0)
        score_layout = thread_mma_qk.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_n))
        ).layout
        score_mn_layout = layout_utils.convert_layout_acc_mn(score_layout)
        score_scale = Float32(softmax_scale * Float32(_LOG2_E))
        if const_expr(self.kv_is_fp8):
            score_scale *= k_scale
        softmax = Softmax.create(
            score_scale,
            cute.size(score_mn_layout.shape[0]),
            arch=120,
        )
        softmax.reset()

        tile_count = Int32((self.selection_width + self.tile_n - 1) // self.tile_n)
        tiles_per_split = (tile_count - split + splits - Int32(1)) // splits
        self._mma_selected_tile(
            split,
            row,
            kv_head,
            request_id,
            request_valid,
            query_position,
            key_cache,
            value_cache,
            block_table,
            selected_positions,
            key_bases,
            value_bases,
            shared_query,
            shared_key,
            shared_value,
            shared_probability,
            shared_row_scale,
            num_cache_pages,
            table_width,
            thread,
            tiled_mma_qk,
            tiled_mma_pv,
            register_query,
            register_key,
            register_probability,
            register_value,
            copy_query,
            copy_key,
            copy_probability,
            copy_value,
            copy_source_query,
            copy_source_key,
            copy_source_probability,
            copy_source_value,
            accumulator_output,
            softmax,
            is_first=True,
        )
        for local_tile in cutlass.range(tiles_per_split - Int32(1), unroll=1):
            tile = split + (Int32(local_tile) + Int32(1)) * splits
            self._mma_selected_tile(
                tile,
                row,
                kv_head,
                request_id,
                request_valid,
                query_position,
                key_cache,
                value_cache,
                block_table,
                selected_positions,
                key_bases,
                value_bases,
                shared_query,
                shared_key,
                shared_value,
                shared_probability,
                shared_row_scale,
                num_cache_pages,
                table_width,
                thread,
                tiled_mma_qk,
                tiled_mma_pv,
                register_query,
                register_key,
                register_probability,
                register_value,
                copy_query,
                copy_key,
                copy_probability,
                copy_value,
                copy_source_query,
                copy_source_key,
                copy_source_probability,
                copy_source_value,
                accumulator_output,
                softmax,
                is_first=False,
            )

        if thread < Int32(self.qk_warps * 32):
            output_scale = softmax.finalize()
            score_coordinates = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(qk_thread).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            if score_coordinates[0, 0][1] == Int32(0):
                for m in cutlass.range_constexpr(cute.size(softmax.row_sum)):
                    local_head = score_coordinates[m, 0][0]
                    scale = output_scale[m]
                    if const_expr(self.kv_is_fp8):
                        scale *= v_scale
                    shared_row_scale[local_head] = scale
                    if const_expr(
                        not self.direct_output or self.return_lse
                    ) and local_head < Int32(self.heads_per_kv):
                        query_head = first_query_head + local_head
                        lse_offset = (
                            row.to(Int64) * splits.to(Int64) + split.to(Int64)
                        ) * Int64(self.q_heads) + query_head.to(Int64)
                        partial_lse[lse_offset] = Float32(softmax.row_sum[m])
        cute.arch.sync_threads()

        output_mn = layout_utils.reshape_acc_to_mn(accumulator_output)
        output_coordinates = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(
                cute.make_identity_tensor((self.tile_m, _HEAD_DIM))
            )
        )
        output_base = (
            row.to(Int64) * Int64(self.q_heads)
            if const_expr(self.direct_output)
            else (row.to(Int64) * splits.to(Int64) + split.to(Int64))
            * Int64(self.q_heads)
        ) * Int64(_HEAD_DIM)
        for m in cutlass.range_constexpr(cute.size(output_mn.shape[0])):
            scale = Float32(shared_row_scale[output_coordinates[m, 0][0]])
            output_mn[m, None].store(output_mn[m, None].load() * scale)
            for n in cutlass.range_constexpr(cute.size(output_mn.shape[1])):
                coordinate = output_coordinates[m, n]
                if coordinate[0] < Int32(self.heads_per_kv):
                    query_head = first_query_head + coordinate[0]
                    offset = (
                        output_base
                        + query_head.to(Int64) * Int64(_HEAD_DIM)
                        + coordinate[1].to(Int64)
                    )
                    if const_expr(self.direct_output):
                        output[offset] = BFloat16(output_mn[m, n])
                    else:
                        partial_output[offset] = Float32(output_mn[m, n])


class _SparseGqaMergeKernel:
    """Merge FP32 split-softmax partials into caller-owned BF16 output."""

    def __init__(self, *, q_heads: int, return_lse: bool) -> None:
        self.q_heads = int(q_heads)
        self.return_lse = bool(return_lse)

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(partial_output, partial_lse, output, output_lse, splits).launch(
            grid=(self.q_heads, rows, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        splits: Int32,
    ) -> None:
        query_head, row, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread_i = Int32(thread)
        row_i = Int64(row)
        head_i = Int64(query_head)
        lane = thread_i % Int32(32)
        warp_id = thread_i // Int32(32)
        weights = cute.make_rmem_tensor((_NUM_SPLITS // 32,), Float32)
        local_maximum = Float32(-Float32.inf)
        for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
            split = lane + Int32(slot * 32)
            weight = Float32(-Float32.inf)
            if split < splits:
                offset = (row_i * splits.to(Int64) + split.to(Int64)) * Int64(
                    self.q_heads
                ) + head_i
                weight = Float32(partial_lse[offset])
            weights[slot] = weight
            local_maximum = fmax(local_maximum, weight)
        maximum = warp_reduce(local_maximum, fmax)

        local_denominator = Float32(0.0)
        for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
            weight = Float32(0.0)
            if weights[slot] > Float32(-Float32.inf):
                weight = _exp2_approx(
                    (Float32(weights[slot]) - maximum) * Float32(_LOG2_E)
                )
            weights[slot] = weight
            local_denominator += weight
        denominator = warp_reduce(local_denominator, _add)
        inverse = Float32(0.0)
        if denominator > Float32(0.0):
            inverse = Float32(1.0) / denominator
        if const_expr(self.return_lse) and thread_i == Int32(0):
            merged_lse = Float32(-Float32.inf)
            if denominator > Float32(0.0):
                merged_lse = maximum + cute.math.log2(
                    denominator, fastmath=True
                ) * Float32(0.6931471805599453)
            output_lse[row_i * Int64(self.q_heads) + head_i] = merged_lse

        allocator = cutlass.utils.SmemAllocator()
        shared_weights = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((_NUM_SPLITS,), stride=(1,)),
            byte_alignment=16,
        )
        if warp_id == Int32(0):
            for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
                shared_weights[lane + Int32(slot * 32)] = weights[slot]
        cute.arch.sync_threads()

        # Adjacent lanes load adjacent head dimensions. Preserve the original
        # split-pair accumulation and ascending butterfly reduction order.
        for tile in cutlass.range_constexpr(_HEAD_DIM // _THREADS):
            dimension = thread_i + Int32(tile * _THREADS)
            partials = cute.make_rmem_tensor((32,), Float32)
            for split_lane in cutlass.range_constexpr(32):
                total = Float32(0.0)
                for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
                    split = Int32(split_lane + slot * 32)
                    if split < splits:
                        offset = (
                            (row_i * splits.to(Int64) + split.to(Int64))
                            * Int64(self.q_heads)
                            + head_i
                        ) * Int64(_HEAD_DIM) + dimension.to(Int64)
                        total += Float32(partial_output[offset]) * Float32(
                            shared_weights[split]
                        )
                partials[split_lane] = total
            for stage in cutlass.range_constexpr(5):
                for group in cutlass.range_constexpr(32 // (2 << stage)):
                    left = group * (2 << stage)
                    right = left + (1 << stage)
                    partials[left] = Float32(partials[left]) + Float32(partials[right])
            output_offset = (row_i * Int64(self.q_heads) + head_i) * Int64(
                _HEAD_DIM
            ) + dimension.to(Int64)
            output[output_offset] = BFloat16(Float32(partials[0]) * inverse)


class _CooperativeSparseGqaMergeKernel(_SparseGqaMergeKernel):
    """Distribute each ordered split reduction across four warps and dimension tiles."""

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(partial_output, partial_lse, output, output_lse, splits).launch(
            grid=(self.q_heads, rows, _HEAD_DIM // 32),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        splits: Int32,
    ) -> None:
        query_head, row, tile = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane = Int32(thread) % Int32(32)
        warp = Int32(thread) // Int32(32)
        row_i, head_i = Int64(row), Int64(query_head)
        weights = cute.make_rmem_tensor((2,), Float32)
        maximum = Float32(-Float32.inf)
        for slot in cutlass.range_constexpr(2):
            split = lane + Int32(slot * 32)
            weight = Float32(-Float32.inf)
            if split < splits:
                offset = (row_i * splits.to(Int64) + split.to(Int64)) * Int64(
                    self.q_heads
                ) + head_i
                weight = Float32(partial_lse[offset])
            weights[slot] = weight
            maximum = fmax(maximum, weight)
        maximum = warp_reduce(maximum, fmax)
        denominator = Float32(0.0)
        for slot in cutlass.range_constexpr(2):
            weight = Float32(0.0)
            if weights[slot] > Float32(-Float32.inf):
                weight = _exp2_approx(
                    (Float32(weights[slot]) - maximum) * Float32(_LOG2_E)
                )
            weights[slot] = weight
            denominator += weight
        denominator = warp_reduce(denominator, _add)
        inverse = Float32(0.0)
        if denominator > Float32(0.0):
            inverse = Float32(1.0) / denominator
        if const_expr(self.return_lse) and (warp == Int32(0)) & (
            lane == Int32(0)
        ):
            merged_lse = Float32(-Float32.inf)
            if denominator > Float32(0.0):
                merged_lse = maximum + cute.math.log2(
                    denominator, fastmath=True
                ) * Float32(0.6931471805599453)
            output_lse[row_i * Int64(self.q_heads) + head_i] = merged_lse

        allocator = cutlass.utils.SmemAllocator()
        shared_weights = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((64,), stride=(1,)),
            byte_alignment=16,
        )
        shared_partials = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((4, 32), stride=(32, 1)),
            byte_alignment=16,
        )
        if warp == Int32(0):
            for slot in cutlass.range_constexpr(2):
                shared_weights[lane + Int32(slot * 32)] = weights[slot]
        cute.arch.sync_threads()

        dimension = Int64(tile) * Int64(32) + lane.to(Int64)
        partials = cute.make_rmem_tensor((8,), Float32)
        for local in cutlass.range_constexpr(8):
            total = Float32(0.0)
            for slot in cutlass.range_constexpr(2):
                split = warp * Int32(8) + Int32(local + slot * 32)
                if split < splits:
                    offset = (
                        (row_i * splits.to(Int64) + split.to(Int64))
                        * Int64(self.q_heads)
                        + head_i
                    ) * Int64(_HEAD_DIM) + dimension
                    total += Float32(partial_output[offset]) * Float32(
                        shared_weights[split]
                    )
            partials[local] = total
        # Pair s with s+32; ascending butterfly stages 0..2 stay warp-local.
        for stage in cutlass.range_constexpr(3):
            for group in cutlass.range_constexpr(8 // (2 << stage)):
                left = group * (2 << stage)
                partials[left] = Float32(partials[left]) + Float32(
                    partials[left + (1 << stage)]
                )
        shared_partials[warp, lane] = partials[0]
        cute.arch.sync_threads()
        if warp == Int32(0):
            # Stages 3 and 4 join the four warp-owned contiguous groups.
            column = lane
            low = Float32(shared_partials[0, column]) + Float32(
                shared_partials[1, column]
            )
            high = Float32(shared_partials[2, column]) + Float32(
                shared_partials[3, column]
            )
            dimension = Int64(tile) * Int64(32) + column.to(Int64)
            offset = (row_i * Int64(self.q_heads) + head_i) * Int64(
                _HEAD_DIM
            ) + dimension
            output[offset] = BFloat16((low + high) * inverse)


class _ShapeAdaptiveSparseGqaMergeKernel:
    """Trade split parallelism for independent query rows in one compiled callable."""

    def __init__(self, *, q_heads: int, return_lse: bool) -> None:
        self.q_heads = q_heads
        self.return_lse = bool(return_lse)
        self.serial = _SparseGqaMergeKernel(
            q_heads=q_heads, return_lse=return_lse
        )
        self.cooperative = _CooperativeSparseGqaMergeKernel(
            q_heads=q_heads, return_lse=return_lse
        )

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        # Independent query rows supply parallelism without multiplying CTAs
        # across head dimensions. Both branches preserve the same reduction tree.
        if rows > splits:
            self.serial.kernel(
                partial_output, partial_lse, output, output_lse, splits
            ).launch(
                grid=(self.q_heads, rows, 1),
                block=(_THREADS, 1, 1),
                stream=stream,
            )
        else:
            self.cooperative.kernel(
                partial_output, partial_lse, output, output_lse, splits
            ).launch(
                grid=(self.q_heads, rows, _HEAD_DIM // 32),
                block=(_THREADS, 1, 1),
                stream=stream,
            )


_MERGE_KERNEL = _ShapeAdaptiveSparseGqaMergeKernel


def _cache_key(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    *,
    direct_output: bool,
    return_lse: bool,
    kv_warps: int,
    selection_width: int = _SELECTION_WIDTH,
) -> tuple[object, ...]:
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        int(device_index),
        int(query.shape[1]),
        int(key_cache.shape[2]),
        int(request_ids.element_size() * 8),
        key_cache.dtype,
        int(key_cache.shape[1]),
        tuple(map(int, key_cache.stride()[:3])),
        tuple(map(int, value_cache.stride()[:3])),
        bool(direct_output),
        bool(return_lse) if direct_output else True,
        int(kv_warps),
        int(selection_width),
    )


def _pointer(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
) -> cute.Pointer:
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype,
        16,
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _compile(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    direct_output: bool,
    return_lse: bool,
    kv_warps: int,
    selection_width: int = _SELECTION_WIDTH,
) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=direct_output,
        return_lse=return_lse,
        kv_warps=kv_warps,
        selection_width=selection_width,
    )
    with _LOCK:
        cached = _KERNEL_CACHE.get(key)
        if cached is not None:
            return key, cached
        (
            _,
            q_heads,
            kv_heads,
            request_id_bits,
            kv_dtype,
            page_size,
            key_strides,
            value_strides,
            direct_output,
            return_lse,
            kv_warps,
            selection_width,
        ) = key
        request_id_type = Int32 if request_id_bits == 32 else Int64
        kv_type = Float8E4M3FN if kv_dtype == torch.float8_e4m3fn else BFloat16
        kernel = _SelectedPositionPagedForwardKernel(
            q_heads=q_heads,
            kv_heads=kv_heads,
            kv_is_fp8=kv_dtype == torch.float8_e4m3fn,
            direct_output=direct_output,
            return_lse=return_lse,
            kv_warps=kv_warps,
            page_size=page_size,
            key_strides=key_strides,
            value_strides=value_strides,
            selection_width=selection_width,
        )
        with torch.cuda.device(key[0]):
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=kernel,
                cache_key=key,
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(BFloat16),
                _fake_pointer(kv_type),
                _fake_pointer(kv_type),
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(Int32),
                _fake_pointer(request_id_type),
                _fake_pointer(Int32),
                _fake_pointer(Int64),
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(BFloat16),
                Int64(1),
                Int64(1),
                Int64(1),
                Float32(1.0),
                Int32(1),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.paged.selected_forward",
                    4,
                    (
                        q_heads,
                        kv_heads,
                        request_id_bits,
                        str(kv_dtype),
                        page_size,
                        key_strides,
                        value_strides,
                        direct_output,
                        return_lse,
                        kv_warps,
                        selection_width,
                    ),
                    labels=(
                        "q_heads",
                        "kv_heads",
                        "request_id_bits",
                        "kv_dtype",
                        "page_size",
                        "key_strides",
                        "value_strides",
                        "direct_output",
                        "return_lse",
                        "kv_warps",
                        "selection_width",
                    ),
                ),
            )
        _KERNEL_CACHE[key] = raw
        return key, raw


def precompile_sparse_gqa_split(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    selection_width: int = _SELECTION_WIDTH,
) -> None:
    """Compile the caller's planned selection capacity without storage writes."""
    with torch.cuda.device(query.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CuTe sparse GQA compilation is forbidden during capture"
            )
        _compile(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            request_ids=request_ids,
            direct_output=False,
            return_lse=True,
            kv_warps=2,
            selection_width=selection_width,
        )


def launch_sparse_gqa_split(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    softmax_scale: float,
    splits: int,
) -> None:
    """Launch the specialized split core without allocation or fallback."""
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=False,
        return_lse=True,
        kv_warps=2,
        selection_width=int(selected_positions.shape[1]),
    )
    with torch.cuda.device(query.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _KERNEL_CACHE.get(key)
            warmed = raw is not None and _WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "CuTe sparse GQA must be compiled and warm-run before CUDA graph "
                "capture"
            )
        if raw is None:
            key, raw = _compile(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                request_ids=request_ids,
                direct_output=False,
                return_lse=True,
                kv_warps=2,
                selection_width=int(selected_positions.shape[1]),
            )
        request_id_type = Int32 if request_ids.dtype == torch.int32 else Int64
        kv_type = Float8E4M3FN if key_cache.dtype == torch.float8_e4m3fn else BFloat16
        run_compiled(
            raw,
            (
                _pointer(query, BFloat16),
                _pointer(key_cache, kv_type),
                _pointer(value_cache, kv_type),
                _pointer(k_descale, Float32)
                if k_descale is not None
                else _fake_pointer(Float32),
                _pointer(v_descale, Float32)
                if v_descale is not None
                else _fake_pointer(Float32),
                _pointer(block_table, Int32),
                _pointer(request_ids, request_id_type),
                _pointer(selected_positions, Int32),
                _pointer(query_positions, Int64),
                _pointer(partial_output, Float32),
                _pointer(partial_lse, Float32),
                _fake_pointer(BFloat16),
                int(key_cache.shape[0]),
                int(block_table.shape[0]),
                int(block_table.shape[1]),
                float(softmax_scale),
                int(query.shape[0]),
                int(splits),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _KERNEL_CACHE.get(key) is raw:
                _WARMED[key] = raw


def launch_selected_paged_gqa_direct(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor | None,
    softmax_scale: float,
    kv_warps: int,
) -> None:
    """Launch direct selected-position paged attention without split scratch."""
    kv_warps = int(kv_warps)
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=True,
        return_lse=output_lse is not None,
        kv_warps=kv_warps,
        selection_width=int(selected_positions.shape[1]),
    )
    with torch.cuda.device(query.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _KERNEL_CACHE.get(key)
            warmed = raw is not None and _WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "selected-position paged attention must be compiled and warm-run "
                "before CUDA graph capture"
            )
        if raw is None:
            key, raw = _compile(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                request_ids=request_ids,
                direct_output=True,
                return_lse=output_lse is not None,
                kv_warps=kv_warps,
                selection_width=int(selected_positions.shape[1]),
            )
        request_id_type = Int32 if request_ids.dtype == torch.int32 else Int64
        kv_type = Float8E4M3FN if key_cache.dtype == torch.float8_e4m3fn else BFloat16
        run_compiled(
            raw,
            (
                _pointer(query, BFloat16),
                _pointer(key_cache, kv_type),
                _pointer(value_cache, kv_type),
                _pointer(k_descale, Float32)
                if k_descale is not None
                else _fake_pointer(Float32),
                _pointer(v_descale, Float32)
                if v_descale is not None
                else _fake_pointer(Float32),
                _pointer(block_table, Int32),
                _pointer(request_ids, request_id_type),
                _pointer(selected_positions, Int32),
                _pointer(query_positions, Int64),
                _fake_pointer(Float32),
                _pointer(output_lse, Float32)
                if output_lse is not None
                else _fake_pointer(Float32),
                _pointer(output, BFloat16),
                int(key_cache.shape[0]),
                int(block_table.shape[0]),
                int(block_table.shape[1]),
                float(softmax_scale),
                int(query.shape[0]),
                1,
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _KERNEL_CACHE.get(key) is raw:
                _WARMED[key] = raw


def launch_sparse_gqa_merge(
    *,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor | None,
    rows: int,
    splits: int,
) -> None:
    """Merge the Qwen split layout without allocation or fallback."""
    device_index = partial_output.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    q_heads = int(partial_output.shape[2])
    return_lse = output_lse is not None
    key = (
        int(device_index),
        q_heads,
        _HEAD_DIM,
        _MERGE_KERNEL.__name__,
        return_lse,
    )
    with torch.cuda.device(device_index):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _MERGE_CACHE.get(key)
            warmed = raw is not None and _MERGE_WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "CuTe sparse GQA merge must be compiled and warm-run before "
                "CUDA graph capture"
            )
        if raw is None:
            kernel = _MERGE_KERNEL(q_heads=q_heads, return_lse=return_lse)
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=kernel,
                cache_key=key,
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(BFloat16),
                _fake_pointer(Float32),
                Int32(1),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.qsa.sparse_gqa_merge",
                    11,
                    key[1:],
                ),
            )
            with _LOCK:
                _MERGE_CACHE[key] = raw
        if compile_only_launches_enabled():
            for program in program_keys(raw):
                record_program(program)
            return raw
        run_compiled(
            raw,
            (
                _pointer(partial_output, Float32),
                _pointer(partial_lse, Float32),
                _pointer(output, BFloat16),
                _pointer(output_lse, Float32)
                if output_lse is not None
                else _fake_pointer(Float32),
                int(rows),
                int(splits),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _MERGE_CACHE.get(key) is raw:
                _MERGE_WARMED[key] = raw


def clear_caches() -> None:
    """Clear process-local compiled and warm-launch state for focused tests."""
    with _LOCK:
        _KERNEL_CACHE.clear()
        _WARMED.clear()
        _MERGE_CACHE.clear()
        _MERGE_WARMED.clear()


__all__ = [
    "clear_caches",
    "launch_selected_paged_gqa_direct",
    "launch_sparse_gqa_split",
    "launch_sparse_gqa_merge",
    "precompile_sparse_gqa_split",
]
