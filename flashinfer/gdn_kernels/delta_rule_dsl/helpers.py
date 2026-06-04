import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp, warpgroup
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm


def round_down(a: int, b: int) -> int:
    return (a // b) * b


@cute.jit
def select_tensor_10(t: cute.Tensor) -> cute.Tensor:
    """select_tensor<1,0>: swap first two modes of a 2-D tensor."""
    return cute.make_tensor(
        t.iterator.align(t.iterator.max_alignment),
        cute.make_layout(
            (t.layout.shape[1], t.layout.shape[0]) + t.layout.shape[2:],
            stride=(t.layout.stride[1], t.layout.stride[0]) + t.layout.stride[2:],
        ),
    )


@cute.jit
def smid():
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def tensormap_replace_global_dim_1(
    tensormap_ptr: cute.Pointer,
    new_val: cutlass.Int32,
):
    ptr_i64 = tensormap_ptr.toint().ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64, new_val.ir_value()],
        "tensormap.replace.tile.global_dim.global.b1024.b32 [$0], 1, $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def load_tensor_as_c(
    sTensor: cute.Tensor,
    tiled_mma,
    thread_idx: cutlass.Int32,
    c_shape,
    src_dtype,
    is_src_n_major: bool,
    dst_dtype=None,
) -> cute.Tensor:
    if cutlass.const_expr(dst_dtype is None):
        dst_dtype = src_dtype
    if cutlass.const_expr(is_src_n_major):
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), src_dtype
        )
    else:
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), src_dtype
        )
    tiled_copy = cute.make_tiled_copy_C(ldsm_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(thread_idx)
    tCrSrc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(c_shape), src_dtype)
    tCrSrc_cv = thr_copy.retile(tCrSrc)
    tCsC = thr_copy.partition_S(sTensor)
    cute.copy(tiled_copy, tCsC, tCrSrc_cv)
    if cutlass.const_expr(dst_dtype is src_dtype):
        return tCrSrc
    tCrC = cute.make_rmem_tensor_like(tCrSrc, dst_dtype)
    for i in cutlass.range(cute.size(tCrC), unroll_full=True):
        tCrC[i] = dst_dtype(tCrSrc[i])
    return tCrC


@cute.jit
def load_tensor_as_a(
    sTensor: cute.Tensor,
    tiled_mma,
    thread_idx: cutlass.Int32,
    a_shape,
    src_dtype,
    is_src_k_major: bool,
    dst_dtype=None,
) -> cute.Tensor:
    if cutlass.const_expr(dst_dtype is None):
        dst_dtype = src_dtype
    if cutlass.const_expr(is_src_k_major):
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), src_dtype
        )
    else:
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), src_dtype
        )
    tiled_copy = cute.make_tiled_copy_A(ldsm_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(thread_idx)
    tArSrc = cute.make_rmem_tensor(tiled_mma.partition_shape_A(a_shape), src_dtype)
    tArSrc_cv = thr_copy.retile(tArSrc)
    tAsA = thr_copy.partition_S(sTensor)
    cute.copy(tiled_copy, tAsA, tArSrc_cv)
    if cutlass.const_expr(dst_dtype is src_dtype):
        return tArSrc
    tArA = cute.make_rmem_tensor_like(tArSrc, dst_dtype)
    for i in cutlass.range(cute.size(tArA), unroll_full=True):
        tArA[i] = dst_dtype(tArSrc[i])
    return tArA


class SM80:
    @staticmethod
    @cute.jit
    def convert_c_layout_to_a_layout(c_layout, tiled_mma):
        c_frag_atom_size = cute.size(c_layout, mode=[0])
        a_frag_atom_size = cute.size(tiled_mma.tv_layout_A, mode=[1])
        ratio = a_frag_atom_size // c_frag_atom_size
        if cutlass.const_expr(ratio == 1):
            return c_layout

        divided = cute.logical_divide(c_layout, (None, None, ratio))
        frag_layout = cute.flatten(
            cute.make_layout(
                (divided.shape[0], divided.shape[2][0]),
                stride=(divided.stride[0], divided.stride[2][0]),
            )
        )
        return cute.make_layout(
            (frag_layout.shape, divided.shape[1], divided.shape[2][1]),
            stride=(
                frag_layout.stride,
                divided.stride[1],
                divided.stride[2][1],
            ),
        )

    @staticmethod
    @cute.jit
    def make_acc_into_op(acc: cute.Tensor, tiled_mma, dtype) -> cute.Tensor:
        operand = cute.make_fragment_like(
            SM80.convert_c_layout_to_a_layout(acc.layout, tiled_mma),
            dtype,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(dtype))
        return operand


class SM90:
    @staticmethod
    @cute.jit
    def wgmma_gemm(
        tiled_mma,
        C: cute.Tensor,
        A: cute.Tensor,
        B: cute.Tensor,
        accumulate: bool,
    ):
        for k_block_idx in cutlass.range(cute.size(A, mode=[2]), unroll_full=True):
            tiled_mma.set(
                warpgroup.Field.ACCUMULATE,
                accumulate or k_block_idx != 0,
            )
            cute.gemm(
                tiled_mma,
                C,
                A[None, None, k_block_idx],
                B[None, None, k_block_idx],
                C,
            )

    @staticmethod
    @cute.jit
    def wgmma_gemm_zero_acc(
        tiled_mma,
        C: cute.Tensor,
        A: cute.Tensor,
        B: cute.Tensor,
    ):
        SM90.wgmma_gemm(tiled_mma, C, A, B, False)

    @staticmethod
    @cute.jit
    def _warpgroup_fence_reg_f32(reg: cutlass.Float32) -> cutlass.Float32:
        return cutlass.Float32(
            llvm.inline_asm(
                T.f32(),
                [reg.ir_value()],
                "",
                "=f,0",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @staticmethod
    @cute.jit
    def _warpgroup_fence_reg_u32(reg: cutlass.Uint32) -> cutlass.Uint32:
        return cutlass.Uint32(
            llvm.inline_asm(
                T.i32(),
                [reg.ir_value()],
                "",
                "=r,0",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @staticmethod
    @cute.jit
    def warpgroup_fence_operand(frg: cute.Tensor):
        if cutlass.const_expr(frg.element_type is cutlass.Float32):
            f32_frg = cute.recast_tensor(frg, cutlass.Float32)
            for i in cutlass.range(cute.size(f32_frg), unroll_full=True):
                f32_frg[i] = SM90._warpgroup_fence_reg_f32(f32_frg[i])
        else:
            u32_frg = cute.recast_tensor(frg, cutlass.Uint32)
            for i in cutlass.range(cute.size(u32_frg), unroll_full=True):
                u32_frg[i] = SM90._warpgroup_fence_reg_u32(u32_frg[i])

    @staticmethod
    @cute.jit
    def convert_c_layout_to_a_layout(c_layout, operand_layout):
        return cute.make_layout(
            (
                operand_layout,
                c_layout.shape[1],
                (
                    c_layout.shape[2],
                    cute.size(c_layout, mode=[0]) // cute.size(operand_layout),
                ),
            ),
            stride=(
                c_layout.stride[0],
                c_layout.stride[1],
                (
                    c_layout.stride[2],
                    cute.size(operand_layout, mode=[2]) * c_layout.stride[0][2],
                ),
            ),
        )

    @staticmethod
    @cute.jit
    def make_acc_into_op(acc: cute.Tensor, tiled_mma, dtype) -> cute.Tensor:
        operand = cute.make_rmem_tensor_like(
            SM90.convert_c_layout_to_a_layout(
                acc.layout,
                tiled_mma.tv_layout_A.shape[1],
            ),
            dtype,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(dtype))
        return operand
