"""Standalone persistent merge kernel for the primary paged backend.

This module intentionally does not share implementation with the archived
`PagedAttentionCombineKernel`. It follows FlashInfer's split-state merge model:

- partial outputs are normalized attention outputs,
- partial scores are base-2 log-sum-exp values,
- CTAs walk `(row, head)` work persistently,
- each CTA cooperatively folds multiple partial states into one final state.

The first version here ports the control flow and state arithmetic faithfully
but keeps the partial-vector loads direct from global memory. The cp.async
staging layer can be added on top of this contract without changing the API.
"""

from __future__ import annotations

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm

from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import Int64, T, dsl_user_op

from b12x.attention._shared.cute import ops as attention_ops
from b12x._lib.intrinsics import (
    get_ptr_as_int64,
    ld_shared_v4_u32,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_shared_v4_u32,
)


@dsl_user_op
def _cp_async_load_128b(smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_global_v2_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _exp2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _log2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _unpack_bfloat2_to_float2(
    packed: Uint32, *, loc=None, ip=None
) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@cute.jit
def _load_shared_bf16x8_to_f32(dst: cute.Tensor, smem_addr: Int32):
    p0, p1, p2, p3 = ld_shared_v4_u32(smem_addr)
    dst[0], dst[1] = _unpack_bfloat2_to_float2(p0)
    dst[2], dst[3] = _unpack_bfloat2_to_float2(p1)
    dst[4], dst[5] = _unpack_bfloat2_to_float2(p2)
    dst[6], dst[7] = _unpack_bfloat2_to_float2(p3)


@cute.jit
def _store_shared_f32x8_as_bf16(smem_addr: Int32, src: cute.Tensor):
    st_shared_v4_u32(
        smem_addr,
        pack_f32x2_to_bfloat2(src[0], src[1]),
        pack_f32x2_to_bfloat2(src[2], src[3]),
        pack_f32x2_to_bfloat2(src[4], src[5]),
        pack_f32x2_to_bfloat2(src[6], src[7]),
    )


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def default_paged_persistent_ctas(
    *,
    total_rows: int,
    num_heads: int,
    device: torch.device | int | None = None,
) -> int:
    if device is None:
        device = torch.cuda.current_device()
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    total_work = max(int(total_rows) * int(num_heads), 1)
    # Match FlashInfer's persistent merge launch shape more closely: size the
    # resident grid by useful blocks-per-SM rather than pinning it to 1 CTA/SM.
    blocks_per_sm = min(3, _ceil_div(total_work, num_sms))
    persistent_ctas = int(num_sms * max(blocks_per_sm, 1))
    if int(total_rows) == 8:
        return int(min(persistent_ctas, total_work))
    return persistent_ctas


@cute.jit
def _state_init(
    state_o: cute.Tensor,
):
    state_o.fill(0.0)
    return Float32(-Float32.inf), Float32(1.0)


@cute.jit
def _state_get_lse_base2(
    state_m: Float32,
    state_d: Float32,
) -> Float32:
    return Float32(state_m + _log2_approx_ftz_f32(Float32(state_d)))


@cute.jit
def _state_merge(
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    other_o: cute.Tensor,
    other_m: Float32,
    other_d: Float32,
) -> tuple[Float32, Float32]:
    prev_m = state_m
    prev_d = state_d
    if prev_m == -Float32.inf:
        if other_m == -Float32.inf:
            state_m = Float32(prev_m)
            state_d = Float32(prev_d)
        else:
            for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
                state_o[vec_idx] = other_o[vec_idx]
            state_m = Float32(other_m)
            state_d = Float32(other_d)
    elif other_m == -Float32.inf:
        state_m = Float32(prev_m)
        state_d = Float32(prev_d)
    else:
        state_m = attention_ops.fmax(prev_m, other_m)
        prev_scale = _exp2_approx_ftz_f32(prev_m - state_m)
        other_scale = _exp2_approx_ftz_f32(other_m - state_m)
        state_d = Float32(prev_d * prev_scale + other_d * other_scale)
        for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
            state_o[vec_idx] = (
                state_o[vec_idx] * prev_scale + other_o[vec_idx] * other_scale
            )
    return Float32(state_m), state_d


@cute.jit
def _state_merge_normalized_lse_base2(
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    other_o: cute.Tensor,
    other_lse: Float32,
) -> tuple[Float32, Float32]:
    return _state_merge(
        state_o,
        state_m,
        state_d,
        other_o,
        other_lse,
        Float32(1.0),
    )


@cute.jit
def _state_merge_finite_normalized_lse_base2(
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    other_o: cute.Tensor,
    other_lse: Float32,
) -> tuple[Float32, Float32]:
    """Merge a normalized state when both the current input and its LSE are live.

    The exact Laguna verifier schedule never exposes a neutral partial to the
    merge kernel: inactive forward CTAs are outside ``num_index_sets`` and
    every active chunk contains at least one visible key.  Keeping that fact
    in the compiled entry removes two per-partial data-dependent branches.
    """
    merged_m = attention_ops.fmax(state_m, other_lse)
    prev_scale = _exp2_approx_ftz_f32(state_m - merged_m)
    other_scale = _exp2_approx_ftz_f32(other_lse - merged_m)
    merged_d = Float32(state_d * prev_scale + other_scale)
    for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
        state_o[vec_idx] = (
            state_o[vec_idx] * prev_scale + other_o[vec_idx] * other_scale
        )
    return Float32(merged_m), merged_d


@cute.jit
def _state_normalize(
    state_o: cute.Tensor,
    state_d: Float32,
):
    inv_d = cute.arch.rcp_approx(Float32(state_d))
    for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
        state_o[vec_idx] = state_o[vec_idx] * inv_d


@cute.jit
def _threadblock_sync_state(
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    s_partial: cute.Tensor,
    s_lse: cute.Tensor,
    *,
    head_dim: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    bdy: cutlass.Constexpr[int],
    finite_states: cutlass.Constexpr[bool],
) -> tuple[Float32, Float32]:
    tx, ty, _ = cute.arch.thread_idx()
    base_k = tx * vec_size
    _state_normalize(state_o, state_d)
    if const_expr(finite_states and vec_size == 8):
        _store_shared_f32x8_as_bf16(
            shared_ptr_to_u32(
                s_partial.iterator + Int32(ty * head_dim + base_k)
            ),
            state_o,
        )
    else:
        for vec_idx in cutlass.range_constexpr(vec_size):
            s_partial[ty, base_k + vec_idx] = state_o[vec_idx].to(
                s_partial.element_type
            )
    s_lse[ty] = _state_get_lse_base2(state_m, state_d)
    state_m, state_d = _state_init(state_o)
    cute.arch.sync_threads()

    for iter_idx in cutlass.range_constexpr(bdy):
        other_o = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), Float32
        )
        if const_expr(finite_states and vec_size == 8):
            _load_shared_bf16x8_to_f32(
                other_o,
                shared_ptr_to_u32(
                    s_partial.iterator + Int32(iter_idx * head_dim + base_k)
                ),
            )
        else:
            for vec_idx in cutlass.range_constexpr(vec_size):
                other_o[vec_idx] = s_partial[
                    iter_idx, base_k + vec_idx
                ].to(Float32)
        if const_expr(finite_states):
            state_m, state_d = _state_merge_finite_normalized_lse_base2(
                state_o,
                state_m,
                state_d,
                other_o,
                s_lse[iter_idx],
            )
        else:
            state_m, state_d = _state_merge_normalized_lse_base2(
                state_o,
                state_m,
                state_d,
                other_o,
                s_lse[iter_idx],
            )
    return state_m, state_d


@cute.jit
def _merge_async_slot(
    *,
    iter_idx: Int32,
    start_idx,
    num_heads,
    head_idx,
    num_index_sets,
    num_stage_iters,
    base_k,
    head_dim,
    vec_size: cutlass.Constexpr[int],
    bdx: cutlass.Constexpr[int],
    bdy: cutlass.Constexpr[int],
    bytes_per_vec: cutlass.Constexpr[int],
    num_smem_stages: cutlass.Constexpr[int],
    finite_states: cutlass.Constexpr[bool],
    s_stage_partial: cute.Tensor,
    s_stage_lse: cute.Tensor,
    mV_partial: cute.Tensor,
    mLSE_partial: cute.Tensor,
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    slot: cutlass.Constexpr[int],
) -> tuple[Float32, Float32]:
    tx, ty, _ = cute.arch.thread_idx()
    cur_iter = iter_idx + Int32(slot)
    if cur_iter < num_stage_iters:
        if cur_iter % bdx == 0:
            lse_linear_idx = cur_iter * bdy + ty * bdx + tx
            s_stage_lse[ty * bdx + tx] = (
                mLSE_partial[start_idx + lse_linear_idx, head_idx]
                if lse_linear_idx < num_index_sets
                else Float32(0.0)
            )
            cute.arch.sync_threads()

        cute.arch.cp_async_wait_group(num_smem_stages - 1)
        cute.arch.sync_threads()

        if cur_iter * bdy + ty < num_index_sets:
            other_o = cute.make_rmem_tensor(
                cute.make_layout((vec_size,), stride=(1,)),
                Float32,
            )
            if const_expr(finite_states and bytes_per_vec == 16 and vec_size == 8):
                _load_shared_bf16x8_to_f32(
                    other_o,
                    shared_ptr_to_u32(
                        s_stage_partial.iterator
                        + Int32(
                            (
                                (cur_iter % num_smem_stages) * bdy + ty
                            )
                            * head_dim
                            + base_k
                        )
                    ),
                )
            else:
                for vec_idx in cutlass.range_constexpr(vec_size):
                    other_o[vec_idx] = s_stage_partial[
                        cur_iter % num_smem_stages, ty, base_k + vec_idx
                    ].to(Float32)
            if const_expr(finite_states):
                state_m, state_d = _state_merge_finite_normalized_lse_base2(
                    state_o,
                    state_m,
                    state_d,
                    other_o,
                    s_stage_lse[(cur_iter % bdx) * bdy + ty],
                )
            else:
                state_m, state_d = _state_merge_normalized_lse_base2(
                    state_o,
                    state_m,
                    state_d,
                    other_o,
                    s_stage_lse[(cur_iter % bdx) * bdy + ty],
                )

        cute.arch.sync_threads()
        next_linear_idx = (cur_iter + num_smem_stages) * bdy + ty
        if const_expr(bytes_per_vec == 8):
            load_base_k = tx * (vec_size * 2)
            smem_addr = shared_ptr_to_u32(
                s_stage_partial.iterator
                + Int32(
                    ((cur_iter % num_smem_stages) * bdy + ty) * head_dim + load_base_k
                )
            )
            if tx < bdx // 2 and next_linear_idx < num_index_sets:
                partial_idx = start_idx + next_linear_idx
                gmem_addr = get_ptr_as_int64(
                    mV_partial,
                    (Int64(partial_idx) * Int64(num_heads) + Int64(head_idx))
                    * Int64(head_dim)
                    + Int64(load_base_k),
                )
                _cp_async_load_128b(smem_addr, gmem_addr)
        else:
            smem_addr = shared_ptr_to_u32(
                s_stage_partial.iterator
                + Int32(((cur_iter % num_smem_stages) * bdy + ty) * head_dim + base_k)
            )
            if next_linear_idx < num_index_sets:
                partial_idx = start_idx + next_linear_idx
                gmem_addr = get_ptr_as_int64(
                    mV_partial,
                    (Int64(partial_idx) * Int64(num_heads) + Int64(head_idx))
                    * Int64(head_dim)
                    + Int64(base_k),
                )
                _cp_async_load_128b(smem_addr, gmem_addr)
        cute.arch.cp_async_commit_group()
    return state_m, state_d


class PagedPersistentMergeKernel:
    """Faithful row/head-persistent merge for the current paged backend path.

    The kernel expects:
    - `mV_partial`: `(nnz_partial_rows, num_heads, head_dim)`
    - `mLSE_partial`: `(nnz_partial_rows, num_heads)` in base-2 log domain
    - `mMergeIndptr`: `(max_total_rows + 1,)`
    - `mO`: `(max_total_rows, num_heads, head_dim)`
    - `mLSE`: `(num_heads, max_total_rows)` in base-2 log domain
    - `mTotalRowsPtr`: optional `(1,)` dynamic sequence length for graph replay
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        *,
        head_dim: int,
        vec_size: int = 8,
        bdx: int = 32,
        bdy: int = 4,
        num_smem_stages: int = 4,
        persistent_ctas: int | None = None,
        direct_grid: bool = False,
        regular_decode_graph: bool = False,
        analytic_laguna_verify_graph: bool = False,
        analytic_laguna_decode_graph: bool = False,
        pair_bf16_partial_loads: bool = False,
    ):
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.vec_size = vec_size
        self.bdx = bdx
        self.bdy = bdy
        self.num_smem_stages = num_smem_stages
        self.persistent_ctas = (
            int(persistent_ctas) if persistent_ctas is not None else 0
        )
        self.direct_grid = bool(direct_grid)
        self.regular_decode_graph = bool(regular_decode_graph)
        self.analytic_laguna_verify_graph = bool(analytic_laguna_verify_graph)
        self.analytic_laguna_decode_graph = bool(
            analytic_laguna_decode_graph
        )
        self.pair_bf16_partial_loads = bool(pair_bf16_partial_loads)
        self.finite_bf16_decode_graph = bool(
            (
                self.regular_decode_graph
                or self.analytic_laguna_decode_graph
            )
            and self.dtype is cutlass.BFloat16
            and self.dtype_partial is cutlass.BFloat16
            and self.head_dim == 128
            and self.vec_size == 8
            and self.bdx == 16
            and self.bdy == 8
        )

    @staticmethod
    def can_implement(
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        *,
        head_dim: int,
        vec_size: int,
        bdx: int,
        bdy: int,
        num_smem_stages: int,
        persistent_ctas: int,
        direct_grid: bool,
    ) -> bool:
        if dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        if dtype_partial not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        if (
            head_dim <= 0
            or vec_size <= 0
            or bdx <= 0
            or bdy <= 0
            or num_smem_stages <= 0
        ):
            return False
        if head_dim != bdx * vec_size:
            return False
        if (bdx * bdy) % 32 != 0:
            return False
        if not direct_grid and persistent_ctas <= 0:
            return False
        return True

    def _get_shared_storage_cls(self):
        # The staged path issues 16-byte cp.async transactions into this field.
        # Encode that requirement in the typed 4.6 allocation contract instead
        # of relying only on the dynamic-SMEM root's implicit alignment.
        stage_partial_storage = cute.struct.Align[
            cute.struct.MemRange[
                self.dtype_partial,
                int(self.num_smem_stages * self.bdy * self.head_dim),
            ],
            16,
        ]
        stage_lse_storage = cute.struct.MemRange[
            cutlass.Float32, int(self.bdx * self.bdy)
        ]
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sStagePartial": stage_partial_storage,
            "sStageLSE": stage_lse_storage,
        }
        if not (
            self.analytic_laguna_verify_graph
            or self.analytic_laguna_decode_graph
            or self.finite_bf16_decode_graph
        ):
            SharedStorage.__annotations__.update(
                {
                    "sPartial": cute.struct.MemRange[
                        cutlass.Float32, int(self.bdy * self.head_dim)
                    ],
                    "sLSE": cute.struct.MemRange[cutlass.Float32, int(self.bdy)],
                }
            )

        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mV_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mMergeIndptr: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mKvChunkSizePtr: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mTotalRowsPtr: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        if const_expr(len(mV_partial.shape) != 3):
            raise ValueError(
                "mV_partial must have shape (nnz_partial_rows, num_heads, head_dim)"
            )
        if const_expr(len(mLSE_partial.shape) != 2):
            raise ValueError(
                "mLSE_partial must have shape (nnz_partial_rows, num_heads)"
            )
        if const_expr(len(mMergeIndptr.shape) != 1):
            raise ValueError("mMergeIndptr must have shape (max_total_rows + 1,)")
        if const_expr(len(mCacheSeqlens.shape) != 1):
            raise ValueError("mCacheSeqlens must have shape (total_rows,)")
        if const_expr(len(mKvChunkSizePtr.shape) != 1):
            raise ValueError("mKvChunkSizePtr must have shape (1,)")
        if const_expr(len(mO.shape) != 3):
            raise ValueError("mO must have shape (max_total_rows, num_heads, head_dim)")
        if const_expr(len(mLSE.shape) != 2):
            raise ValueError("mLSE must have shape (num_heads, max_total_rows)")
        if const_expr(mTotalRowsPtr is not None and len(mTotalRowsPtr.shape) != 1):
            raise ValueError("mTotalRowsPtr must have shape (1,)")
        if const_expr(mV_partial.element_type != self.dtype_partial):
            raise TypeError("mV_partial dtype must match dtype_partial")
        if const_expr(mO.element_type != self.dtype):
            raise TypeError("mO dtype must match dtype")
        if const_expr(
            mLSE_partial.element_type != Float32 or mLSE.element_type != Float32
        ):
            raise TypeError("mLSE tensors must be Float32")
        if const_expr(
            not self.can_implement(
                self.dtype,
                self.dtype_partial,
                head_dim=self.head_dim,
                vec_size=self.vec_size,
                bdx=self.bdx,
                bdy=self.bdy,
                num_smem_stages=self.num_smem_stages,
                persistent_ctas=self.persistent_ctas,
                direct_grid=self.direct_grid,
            )
        ):
            raise TypeError("paged merge kernel configuration is not supported")

        self.kernel(
            mV_partial,
            mLSE_partial,
            mMergeIndptr,
            mCacheSeqlens,
            mKvChunkSizePtr,
            mO,
            mLSE,
            mTotalRowsPtr,
        ).launch(
            grid=(
                (mO.shape[1], mO.shape[0], 1)
                if self.direct_grid
                else (self.persistent_ctas, 1, 1)
            ),
            block=[self.bdx, self.bdy, 1],
            use_pdl=self.analytic_laguna_verify_graph,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mV_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mMergeIndptr: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mKvChunkSizePtr: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mTotalRowsPtr: cute.Tensor | None,
    ):
        tx, ty, _ = cute.arch.thread_idx()
        block_x, block_y, _ = cute.arch.block_idx()
        if const_expr(not self.direct_grid):
            num_ctas, _, _ = cute.arch.grid_dim()
        max_total_rows = mO.shape[0]
        num_heads = mO.shape[1]
        head_dim = self.head_dim
        total_rows = (
            mTotalRowsPtr[0]
            if const_expr(mTotalRowsPtr is not None)
            else max_total_rows
        )

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        s_stage_partial = storage.sStagePartial.get_tensor(
            cute.make_layout(
                (self.num_smem_stages, self.bdy, head_dim),
                stride=(self.bdy * head_dim, head_dim, 1),
            )
        )
        s_stage_lse = storage.sStageLSE.get_tensor(
            cute.make_layout((self.bdx * self.bdy,), stride=(1,))
        )
        if const_expr(
            self.analytic_laguna_verify_graph
            or self.analytic_laguna_decode_graph
            or self.finite_bf16_decode_graph
        ):
            # The mainloop is complete before the cross-warp fold, so reuse
            # its BF16 stage and LSE storage exactly as FI does.
            s_partial = cute.make_tensor(
                s_stage_partial.iterator,
                cute.make_layout((self.bdy, head_dim), stride=(head_dim, 1)),
            )
            s_lse = cute.make_tensor(
                s_stage_lse.iterator,
                cute.make_layout((self.bdy,), stride=(1,)),
            )
        else:
            s_partial = storage.sPartial.get_tensor(
                cute.make_layout((self.bdy, head_dim), stride=(head_dim, 1))
            )
            s_lse = storage.sLSE.get_tensor(
                cute.make_layout((self.bdy,), stride=(1,))
            )

        if const_expr(not self.direct_grid):
            cute.arch.griddepcontrol_wait()

        total_work = total_rows * num_heads
        base_k = tx * self.vec_size
        work_linear_idx = (
            Int32(block_y * num_heads + block_x)
            if const_expr(self.direct_grid)
            else Int32(block_x)
        )
        max_chunks_per_req = Int32(0)
        analytic_decode_base_chunks = Int32(0)
        analytic_decode_extra_rows = Int32(0)
        if const_expr(
            self.regular_decode_graph
            or self.analytic_laguna_verify_graph
        ):
            max_chunks_per_req = Int32(mV_partial.shape[0] // mO.shape[0])
        if const_expr(self.analytic_laguna_decode_graph):
            analytic_decode_base_chunks = Int32(
                mV_partial.shape[0] // mO.shape[0]
            )
            analytic_decode_extra_rows = Int32(
                mV_partial.shape[0]
                - analytic_decode_base_chunks * mO.shape[0]
            )
        live_verify_chunks = Int32(0)
        if const_expr(self.analytic_laguna_verify_graph):
            # Match the exact verifier forward kernel's graph-static grid and
            # device-only wave balancing.  No merge-indptr or live host plan is
            # needed, and every row uses the same fixed scratch stride.
            live_stage_tiles = (
                mCacheSeqlens[0] + Int32(63)
            ) // Int32(64)
            live_verify_chunks = cutlass.select_(
                live_stage_tiles < max_chunks_per_req,
                live_stage_tiles,
                max_chunks_per_req,
            )
        while work_linear_idx < total_work:
            cute.arch.sync_threads()

            if const_expr(self.direct_grid):
                row_idx = Int32(block_y)
                head_idx = Int32(block_x)
            else:
                row_idx = work_linear_idx // num_heads
                head_idx = work_linear_idx % num_heads
            if const_expr(self.analytic_laguna_verify_graph):
                start_idx = row_idx * max_chunks_per_req
                num_index_sets = live_verify_chunks
                end_idx = start_idx + num_index_sets
            elif const_expr(self.analytic_laguna_decode_graph):
                request_chunk_capacity = (
                    analytic_decode_base_chunks
                    + cutlass.select_(
                        row_idx < analytic_decode_extra_rows,
                        Int32(1),
                        Int32(0),
                    )
                )
                start_idx = (
                    row_idx * analytic_decode_base_chunks
                    + cutlass.select_(
                        row_idx < analytic_decode_extra_rows,
                        row_idx,
                        analytic_decode_extra_rows,
                    )
                )
                live_decode_stage_tiles = (
                    mCacheSeqlens[row_idx] + Int32(63)
                ) // Int32(64)
                num_index_sets = cutlass.select_(
                    live_decode_stage_tiles < request_chunk_capacity,
                    live_decode_stage_tiles,
                    request_chunk_capacity,
                )
                end_idx = start_idx + num_index_sets
            elif const_expr(self.regular_decode_graph):
                start_idx = row_idx * max_chunks_per_req
                num_index_sets = mMergeIndptr[row_idx + 1] - mMergeIndptr[row_idx]
                end_idx = start_idx + num_index_sets
            else:
                start_idx = mMergeIndptr[row_idx]
                end_idx = mMergeIndptr[row_idx + 1]
                num_index_sets = end_idx - start_idx

            if num_index_sets == 0:
                if const_expr(self.dtype is cutlass.BFloat16 and self.vec_size == 4):
                    _st_global_v2_u32(
                        get_ptr_as_int64(
                            mO,
                            (Int64(row_idx) * Int64(num_heads) + Int64(head_idx))
                            * Int64(head_dim)
                            + Int64(base_k),
                        ),
                        Uint32(0),
                        Uint32(0),
                    )
                else:
                    for vec_idx in cutlass.range_constexpr(self.vec_size):
                        mO[row_idx, head_idx, base_k + vec_idx] = self.dtype(0.0)
                if tx == 0 and ty == 0:
                    mLSE[head_idx, row_idx] = -Float32.inf
            elif num_index_sets == 1:
                for vec_idx in cutlass.range_constexpr(self.vec_size):
                    mO[row_idx, head_idx, base_k + vec_idx] = mV_partial[
                        start_idx, head_idx, base_k + vec_idx
                    ].to(self.dtype)
                if tx == 0 and ty == 0:
                    mLSE[head_idx, row_idx] = mLSE_partial[start_idx, head_idx]
            else:
                state_o = cute.make_rmem_tensor(
                    cute.make_layout((self.vec_size,), stride=(1,)),
                    Float32,
                )
                state_m, state_d = _state_init(state_o)
                bytes_per_vec = self.vec_size * self.dtype_partial.width // 8
                can_stage_async = bytes_per_vec == 16 or (
                    bytes_per_vec == 8
                    and self.dtype_partial is cutlass.BFloat16
                    and self.pair_bf16_partial_loads
                )
                if can_stage_async and (
                    bytes_per_vec == 16 or num_index_sets >= Int32(8)
                ):
                    for stage_idx in cutlass.range_constexpr(self.num_smem_stages):
                        staged_linear_idx = stage_idx * self.bdy + ty
                        if const_expr(bytes_per_vec == 8):
                            load_base_k = tx * (self.vec_size * 2)
                            smem_addr = shared_ptr_to_u32(
                                s_stage_partial.iterator
                                + Int32(
                                    (stage_idx * self.bdy + ty) * head_dim + load_base_k
                                )
                            )
                            if (
                                tx < self.bdx // 2
                                and staged_linear_idx < num_index_sets
                            ):
                                partial_idx = start_idx + staged_linear_idx
                                gmem_addr = get_ptr_as_int64(
                                    mV_partial,
                                    (
                                        Int64(partial_idx) * Int64(num_heads)
                                        + Int64(head_idx)
                                    )
                                    * Int64(head_dim)
                                    + Int64(load_base_k),
                                )
                                _cp_async_load_128b(smem_addr, gmem_addr)
                        else:
                            smem_addr = shared_ptr_to_u32(
                                s_stage_partial.iterator
                                + Int32((stage_idx * self.bdy + ty) * head_dim + base_k)
                            )
                            if staged_linear_idx < num_index_sets:
                                partial_idx = start_idx + staged_linear_idx
                                gmem_addr = get_ptr_as_int64(
                                    mV_partial,
                                    (
                                        Int64(partial_idx) * Int64(num_heads)
                                        + Int64(head_idx)
                                    )
                                    * Int64(head_dim)
                                    + Int64(base_k),
                                )
                                _cp_async_load_128b(smem_addr, gmem_addr)
                        cute.arch.cp_async_commit_group()

                    num_stage_iters = (num_index_sets + self.bdy - 1) // self.bdy
                    iter_idx = Int32(0)
                    while iter_idx < num_stage_iters:
                        state_m, state_d = _merge_async_slot(
                            iter_idx=iter_idx,
                            start_idx=start_idx,
                            num_heads=num_heads,
                            head_idx=head_idx,
                            num_index_sets=num_index_sets,
                            num_stage_iters=num_stage_iters,
                            base_k=base_k,
                            head_dim=head_dim,
                            vec_size=self.vec_size,
                            bdx=self.bdx,
                            bdy=self.bdy,
                            bytes_per_vec=bytes_per_vec,
                            num_smem_stages=self.num_smem_stages,
                            finite_states=(
                                self.analytic_laguna_verify_graph
                                or self.finite_bf16_decode_graph
                            ),
                            s_stage_partial=s_stage_partial,
                            s_stage_lse=s_stage_lse,
                            mV_partial=mV_partial,
                            mLSE_partial=mLSE_partial,
                            state_o=state_o,
                            state_m=state_m,
                            state_d=state_d,
                            slot=0,
                        )
                        state_m, state_d = _merge_async_slot(
                            iter_idx=iter_idx,
                            start_idx=start_idx,
                            num_heads=num_heads,
                            head_idx=head_idx,
                            num_index_sets=num_index_sets,
                            num_stage_iters=num_stage_iters,
                            base_k=base_k,
                            head_dim=head_dim,
                            vec_size=self.vec_size,
                            bdx=self.bdx,
                            bdy=self.bdy,
                            bytes_per_vec=bytes_per_vec,
                            num_smem_stages=self.num_smem_stages,
                            finite_states=(
                                self.analytic_laguna_verify_graph
                                or self.finite_bf16_decode_graph
                            ),
                            s_stage_partial=s_stage_partial,
                            s_stage_lse=s_stage_lse,
                            mV_partial=mV_partial,
                            mLSE_partial=mLSE_partial,
                            state_o=state_o,
                            state_m=state_m,
                            state_d=state_d,
                            slot=1,
                        )
                        state_m, state_d = _merge_async_slot(
                            iter_idx=iter_idx,
                            start_idx=start_idx,
                            num_heads=num_heads,
                            head_idx=head_idx,
                            num_index_sets=num_index_sets,
                            num_stage_iters=num_stage_iters,
                            base_k=base_k,
                            head_dim=head_dim,
                            vec_size=self.vec_size,
                            bdx=self.bdx,
                            bdy=self.bdy,
                            bytes_per_vec=bytes_per_vec,
                            num_smem_stages=self.num_smem_stages,
                            finite_states=(
                                self.analytic_laguna_verify_graph
                                or self.finite_bf16_decode_graph
                            ),
                            s_stage_partial=s_stage_partial,
                            s_stage_lse=s_stage_lse,
                            mV_partial=mV_partial,
                            mLSE_partial=mLSE_partial,
                            state_o=state_o,
                            state_m=state_m,
                            state_d=state_d,
                            slot=2,
                        )
                        state_m, state_d = _merge_async_slot(
                            iter_idx=iter_idx,
                            start_idx=start_idx,
                            num_heads=num_heads,
                            head_idx=head_idx,
                            num_index_sets=num_index_sets,
                            num_stage_iters=num_stage_iters,
                            base_k=base_k,
                            head_dim=head_dim,
                            vec_size=self.vec_size,
                            bdx=self.bdx,
                            bdy=self.bdy,
                            bytes_per_vec=bytes_per_vec,
                            num_smem_stages=self.num_smem_stages,
                            finite_states=(
                                self.analytic_laguna_verify_graph
                                or self.finite_bf16_decode_graph
                            ),
                            s_stage_partial=s_stage_partial,
                            s_stage_lse=s_stage_lse,
                            mV_partial=mV_partial,
                            mLSE_partial=mLSE_partial,
                            state_o=state_o,
                            state_m=state_m,
                            state_d=state_d,
                            slot=3,
                        )
                        iter_idx += Int32(4)

                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                else:
                    partial_idx = start_idx + ty
                    while partial_idx < end_idx:
                        other_o = cute.make_rmem_tensor(
                            cute.make_layout((self.vec_size,), stride=(1,)),
                            Float32,
                        )
                        for vec_idx in cutlass.range_constexpr(self.vec_size):
                            other_o[vec_idx] = mV_partial[
                                partial_idx, head_idx, base_k + vec_idx
                            ].to(Float32)
                        state_m, state_d = _state_merge_normalized_lse_base2(
                            state_o,
                            state_m,
                            state_d,
                            other_o,
                            mLSE_partial[partial_idx, head_idx],
                        )
                        partial_idx += self.bdy

                state_m, state_d = _threadblock_sync_state(
                    state_o,
                    state_m,
                    state_d,
                    s_partial,
                    s_lse,
                    head_dim=head_dim,
                    vec_size=self.vec_size,
                    bdy=self.bdy,
                    finite_states=(
                        self.analytic_laguna_verify_graph
                        or self.finite_bf16_decode_graph
                    ),
                )
                _state_normalize(state_o, state_d)
                if const_expr(self.dtype is cutlass.BFloat16 and self.vec_size == 4):
                    _st_global_v2_u32(
                        get_ptr_as_int64(
                            mO,
                            (Int64(row_idx) * Int64(num_heads) + Int64(head_idx))
                            * Int64(head_dim)
                            + Int64(base_k),
                        ),
                        pack_f32x2_to_bfloat2(state_o[0], state_o[1]),
                        pack_f32x2_to_bfloat2(state_o[2], state_o[3]),
                    )
                else:
                    for vec_idx in cutlass.range_constexpr(self.vec_size):
                        mO[row_idx, head_idx, base_k + vec_idx] = state_o[vec_idx].to(
                            self.dtype
                        )
                if tx == 0 and ty == 0:
                    mLSE[head_idx, row_idx] = _state_get_lse_base2(state_m, state_d)
            if const_expr(self.direct_grid):
                work_linear_idx = total_work
            else:
                work_linear_idx += num_ctas

        cute.arch.griddepcontrol_launch_dependents()


class LagunaVerifierMergeKernel:
    """Narrow graph-safe merge for the exact Laguna q=8 surface.

    The forward specialization writes one fixed-stride
    ``[batch, 8, max_splits]`` scratch matrix and derives each request's
    useful split count from its device cache length.  This entry mirrors that
    mapping directly and deliberately omits the generic indptr, chunk-size,
    and total-row ABI.
    """

    head_dim = 128
    vec_size = 8
    bdx = 16
    bdy = 8
    num_smem_stages = 4

    def __init__(
        self,
        persistent_ctas: int,
        *,
        two_wave_b1: bool = False,
    ):
        self.persistent_ctas = int(persistent_ctas)
        self.two_wave_b1 = bool(two_wave_b1)

    @staticmethod
    def _get_shared_storage_cls():
        stage_partial_storage = cute.struct.Align[
            cute.struct.MemRange[
                cutlass.BFloat16,
                int(
                    LagunaVerifierMergeKernel.num_smem_stages
                    * LagunaVerifierMergeKernel.bdy
                    * LagunaVerifierMergeKernel.head_dim
                ),
            ],
            16,
        ]
        stage_lse_storage = cute.struct.MemRange[
            cutlass.Float32,
            int(LagunaVerifierMergeKernel.bdx * LagunaVerifierMergeKernel.bdy),
        ]

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sStagePartial": stage_partial_storage,
            "sStageLSE": stage_lse_storage,
        }

        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mV_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(
            len(mV_partial.shape) != 3
            or mV_partial.element_type != cutlass.BFloat16
        ):
            raise TypeError("Laguna verifier partial output must be BF16 [*,24,128]")
        if const_expr(
            len(mLSE_partial.shape) != 2
            or mLSE_partial.element_type != cutlass.Float32
        ):
            raise TypeError("Laguna verifier partial LSE must be FP32 [*,24]")
        if const_expr(
            len(mCacheSeqlens.shape) != 1
            or mCacheSeqlens.element_type != cutlass.Int32
        ):
            raise TypeError("Laguna verifier cache lengths must be Int32 [batch]")
        if const_expr(
            len(mO.shape) != 3
            or mO.element_type != cutlass.BFloat16
        ):
            raise TypeError("Laguna verifier output must be BF16 [8,24,128]")
        if const_expr(
            len(mLSE.shape) != 2
            or mLSE.element_type != cutlass.Float32
        ):
            raise TypeError("Laguna verifier output LSE must be FP32 [24,8]")

        self.kernel(
            mV_partial,
            mLSE_partial,
            mCacheSeqlens,
            mO,
            mLSE,
        ).launch(
            grid=(self.persistent_ctas, 1, 1),
            block=[self.bdx, self.bdy, 1],
            use_pdl=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mV_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
    ):
        tx, ty, _ = cute.arch.thread_idx()
        work_idx, _, _ = cute.arch.block_idx()
        num_heads = Int32(24)
        row_idx = Int32(work_idx) // num_heads
        head_idx = Int32(work_idx) - row_idx * num_heads
        base_k = tx * Int32(self.vec_size)
        max_chunks_per_row = Int32(mV_partial.shape[0] // mO.shape[0])
        request_idx = row_idx // Int32(8)
        live_stage_tiles = (
            mCacheSeqlens[request_idx] + Int32(63)
        ) // Int32(64)
        live_chunk_cap = max_chunks_per_row
        if const_expr(self.two_wave_b1):
            one_wave_chunks = max_chunks_per_row // Int32(2)
            live_chunk_cap = cutlass.select_(
                live_stage_tiles <= Int32(1024),
                one_wave_chunks,
                max_chunks_per_row,
            )
        num_index_sets = cutlass.select_(
            live_stage_tiles < live_chunk_cap,
            live_stage_tiles,
            live_chunk_cap,
        )
        start_idx = row_idx * max_chunks_per_row

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._get_shared_storage_cls())
        s_stage_partial = storage.sStagePartial.get_tensor(
            cute.make_layout(
                (self.num_smem_stages, self.bdy, self.head_dim),
                stride=(self.bdy * self.head_dim, self.head_dim, 1),
            )
        )
        s_stage_lse = storage.sStageLSE.get_tensor(
            cute.make_layout((self.bdx * self.bdy,), stride=(1,))
        )
        s_partial = cute.make_tensor(
            s_stage_partial.iterator,
            cute.make_layout(
                (self.bdy, self.head_dim), stride=(self.head_dim, 1)
            ),
        )
        s_lse = cute.make_tensor(
            s_stage_lse.iterator,
            cute.make_layout((self.bdy,), stride=(1,)),
        )
        cute.arch.griddepcontrol_wait()
        if num_index_sets == Int32(0):
            for vec_idx in cutlass.range_constexpr(self.vec_size):
                mO[row_idx, head_idx, base_k + vec_idx] = cutlass.BFloat16(0.0)
            if tx == Int32(0) and ty == Int32(0):
                mLSE[head_idx, row_idx] = -Float32.inf
        elif num_index_sets == Int32(1):
            for vec_idx in cutlass.range_constexpr(self.vec_size):
                mO[row_idx, head_idx, base_k + vec_idx] = mV_partial[
                    start_idx, head_idx, base_k + vec_idx
                ]
            if tx == Int32(0) and ty == Int32(0):
                mLSE[head_idx, row_idx] = mLSE_partial[start_idx, head_idx]
        else:
            state_o = cute.make_rmem_tensor(
                cute.make_layout((self.vec_size,), stride=(1,)), Float32
            )
            state_m, state_d = _state_init(state_o)

            for stage_idx in cutlass.range_constexpr(self.num_smem_stages):
                staged_linear_idx = stage_idx * self.bdy + ty
                smem_addr = shared_ptr_to_u32(
                    s_stage_partial.iterator
                    + Int32(
                        (stage_idx * self.bdy + ty) * self.head_dim + base_k
                    )
                )
                if staged_linear_idx < num_index_sets:
                    partial_idx = start_idx + staged_linear_idx
                    gmem_addr = get_ptr_as_int64(
                        mV_partial,
                        (
                            Int64(partial_idx) * Int64(24) + Int64(head_idx)
                        )
                        * Int64(self.head_dim)
                        + Int64(base_k),
                    )
                    _cp_async_load_128b(smem_addr, gmem_addr)
                cute.arch.cp_async_commit_group()

            num_stage_iters = (
                num_index_sets + Int32(self.bdy - 1)
            ) // Int32(self.bdy)
            iter_idx = Int32(0)
            while iter_idx < num_stage_iters:
                state_m, state_d = _merge_async_slot(
                    iter_idx=iter_idx,
                    start_idx=start_idx,
                    num_heads=num_heads,
                    head_idx=head_idx,
                    num_index_sets=num_index_sets,
                    num_stage_iters=num_stage_iters,
                    base_k=base_k,
                    head_dim=self.head_dim,
                    vec_size=self.vec_size,
                    bdx=self.bdx,
                    bdy=self.bdy,
                    bytes_per_vec=16,
                    num_smem_stages=self.num_smem_stages,
                    finite_states=True,
                    s_stage_partial=s_stage_partial,
                    s_stage_lse=s_stage_lse,
                    mV_partial=mV_partial,
                    mLSE_partial=mLSE_partial,
                    state_o=state_o,
                    state_m=state_m,
                    state_d=state_d,
                    slot=0,
                )
                state_m, state_d = _merge_async_slot(
                    iter_idx=iter_idx,
                    start_idx=start_idx,
                    num_heads=num_heads,
                    head_idx=head_idx,
                    num_index_sets=num_index_sets,
                    num_stage_iters=num_stage_iters,
                    base_k=base_k,
                    head_dim=self.head_dim,
                    vec_size=self.vec_size,
                    bdx=self.bdx,
                    bdy=self.bdy,
                    bytes_per_vec=16,
                    num_smem_stages=self.num_smem_stages,
                    finite_states=True,
                    s_stage_partial=s_stage_partial,
                    s_stage_lse=s_stage_lse,
                    mV_partial=mV_partial,
                    mLSE_partial=mLSE_partial,
                    state_o=state_o,
                    state_m=state_m,
                    state_d=state_d,
                    slot=1,
                )
                state_m, state_d = _merge_async_slot(
                    iter_idx=iter_idx,
                    start_idx=start_idx,
                    num_heads=num_heads,
                    head_idx=head_idx,
                    num_index_sets=num_index_sets,
                    num_stage_iters=num_stage_iters,
                    base_k=base_k,
                    head_dim=self.head_dim,
                    vec_size=self.vec_size,
                    bdx=self.bdx,
                    bdy=self.bdy,
                    bytes_per_vec=16,
                    num_smem_stages=self.num_smem_stages,
                    finite_states=True,
                    s_stage_partial=s_stage_partial,
                    s_stage_lse=s_stage_lse,
                    mV_partial=mV_partial,
                    mLSE_partial=mLSE_partial,
                    state_o=state_o,
                    state_m=state_m,
                    state_d=state_d,
                    slot=2,
                )
                state_m, state_d = _merge_async_slot(
                    iter_idx=iter_idx,
                    start_idx=start_idx,
                    num_heads=num_heads,
                    head_idx=head_idx,
                    num_index_sets=num_index_sets,
                    num_stage_iters=num_stage_iters,
                    base_k=base_k,
                    head_dim=self.head_dim,
                    vec_size=self.vec_size,
                    bdx=self.bdx,
                    bdy=self.bdy,
                    bytes_per_vec=16,
                    num_smem_stages=self.num_smem_stages,
                    finite_states=True,
                    s_stage_partial=s_stage_partial,
                    s_stage_lse=s_stage_lse,
                    mV_partial=mV_partial,
                    mLSE_partial=mLSE_partial,
                    state_o=state_o,
                    state_m=state_m,
                    state_d=state_d,
                    slot=3,
                )
                iter_idx += Int32(4)

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            state_m, state_d = _threadblock_sync_state(
                state_o,
                state_m,
                state_d,
                s_partial,
                s_lse,
                head_dim=self.head_dim,
                vec_size=self.vec_size,
                bdy=self.bdy,
                finite_states=True,
            )
            _state_normalize(state_o, state_d)
            output_offset = (
                (Int64(row_idx) * Int64(24) + Int64(head_idx))
                * Int64(self.head_dim)
                + Int64(base_k)
            )
            _st_global_v2_u32(
                get_ptr_as_int64(mO, output_offset),
                pack_f32x2_to_bfloat2(state_o[0], state_o[1]),
                pack_f32x2_to_bfloat2(state_o[2], state_o[3]),
            )
            _st_global_v2_u32(
                get_ptr_as_int64(mO, output_offset + Int64(4)),
                pack_f32x2_to_bfloat2(state_o[4], state_o[5]),
                pack_f32x2_to_bfloat2(state_o[6], state_o[7]),
            )
            if tx == Int32(0) and ty == Int32(0):
                mLSE[head_idx, row_idx] = _state_get_lse_base2(
                    state_m, state_d
                )

        cute.arch.griddepcontrol_launch_dependents()
