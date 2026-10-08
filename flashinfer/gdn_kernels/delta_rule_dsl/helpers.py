from dataclasses import dataclass

import torch
import cutlass
import cutlass.cute as cute
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute import core as cute_core
from cutlass.cute.atom import Trait, make_atom
from cutlass.cute.nvgpu import warp, warpgroup
from cutlass.cute.typing import Shape
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


def round_down(a: int, b: int) -> int:
    return (a // b) * b


def state_dtype_to_cutlass(dtype: torch.dtype) -> type[cutlass.Numeric]:
    state_dtypes = {
        torch.float32: cutlass.Float32,
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e5m2: cutlass.Float8E5M2,
    }
    if dtype not in state_dtypes:
        raise ValueError(
            f"Unsupported state dtype {dtype}, expected float32, bfloat16, "
            "float16, float8_e4m3fn, or float8_e5m2"
        )
    return state_dtypes[dtype]


@dataclass(frozen=True)
class WarpMmaTF32Op(warp.WarpMmaOp):
    shape_mnk: Shape

    def __post_init__(self) -> None:
        if self.shape_mnk != (16, 8, 8):
            raise ValueError(
                f"WarpMmaTF32Op only supports (16, 8, 8), got {self.shape_mnk}"
            )

    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        shape_mnk = cute_core._pack_shape(self.shape_mnk, loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.MmaAtomSM80Type.get(
            shape_mnk.type.attribute,
            cutlass.TFloat32.mlir_type,
            cutlass.TFloat32.mlir_type,
            cutlass.Float32.mlir_type,
        )
        return WarpMmaTF32Trait(make_atom(ty, loc=loc, ip=ip))

    def _verify_fragment_A(self, input, *, loc=None, ip=None):
        return True

    def _verify_fragment_B(self, input, *, loc=None, ip=None):
        return True


class WarpMmaTF32Trait(Trait):
    pass


class TF32:
    @staticmethod
    @cute.jit
    def round_to_tf32_f32(value: cutlass.Float32) -> cutlass.Float32:
        bits = llvm.inline_asm(
            T.i32(),
            [value.ir_value()],
            "cvt.rz.tf32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return cutlass.Float32(llvm.bitcast(T.f32(), bits))

    @staticmethod
    @cute.jit
    def convert_fp32_to_tf32_residual(tensor: cute.Tensor):
        residual = cute.make_rmem_tensor_like(tensor, cutlass.Float32)
        for i in cutlass.range_constexpr(cute.size(residual)):
            value = tensor[i]
            residual[i] = value - TF32.round_to_tf32_f32(value)
        return cute.recast_tensor(residual, cutlass.TFloat32)

    @staticmethod
    @cute.jit
    def convert_tf32_c_to_kpermuted_a(tCrC: cute.Tensor, tCrA: cute.Tensor):
        for i in cutlass.range(cute.size(tCrA), unroll_full=True):
            tCrA[i] = tCrC[i]
        for m in cutlass.range_constexpr(cute.size(tCrA, mode=[1])):
            for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                tmp = tCrA[(1, 0), m, k]
                tCrA[(1, 0), m, k] = tCrA[(0, 1), m, k]
                tCrA[(0, 1), m, k] = tmp

    @staticmethod
    @cute.jit
    def load_tf32_kpermuted_b(
        tCrB: cute.Tensor,
        sB_NK: cute.Tensor,
        lane_idx: cutlass.Int32,
    ):
        sB_8x8 = cute.flat_divide(sB_NK, (8, 8))
        n = lane_idx // 4
        k = lane_idx - n * 4
        for iter_n in cutlass.range_constexpr(cute.size(tCrB, mode=[1])):
            for iter_k in cutlass.range_constexpr(cute.size(tCrB, mode=[2])):
                tCrB[0, iter_n, iter_k] = sB_8x8[n, k * 2, iter_n, iter_k]
                tCrB[1, iter_n, iter_k] = sB_8x8[n, k * 2 + 1, iter_n, iter_k]


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


@cute.jit
def load_tensor_as_b(
    sTensor: cute.Tensor,
    tiled_mma,
    thread_idx: cutlass.Int32,
    b_shape,
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
    tiled_copy = cute.make_tiled_copy_B(ldsm_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(thread_idx)
    tBrSrc = cute.make_rmem_tensor(tiled_mma.partition_shape_B(b_shape), src_dtype)
    tBrSrc_cv = thr_copy.retile(tBrSrc)
    tBsB = thr_copy.partition_S(sTensor)
    cute.copy(tiled_copy, tBsB, tBrSrc_cv)
    if cutlass.const_expr(dst_dtype is src_dtype):
        return tBrSrc
    tBrB = cute.make_rmem_tensor_like(tBrSrc, dst_dtype)
    for i in cutlass.range(cute.size(tBrB), unroll_full=True):
        tBrB[i] = dst_dtype(tBrSrc[i])
    return tBrB


@cute.jit
def gemm_f16acc_carry(
    tiled_mma,
    acc32: cute.Tensor,
    tA: cute.Tensor,
    tB: cute.Tensor,
    group: cutlass.Constexpr,
):
    """acc32 += A @ B^T with FP16-accumulate m16n8k16 MMAs and an FP32 carry.

    ``tiled_mma`` is built from an FP16-accumulator atom
    (``warp.MmaF16BF16Op(Float16, Float16, (16, 8, 16))``), ``tA`` / ``tB`` are FP16
    operand fragments and ``acc32`` is the FP32 accumulator fragment holding the
    running sum (same C layout as the plain FP32 path). Every ``group`` consecutive
    K steps accumulate into a zeroed FP16 fragment, which is then added to ``acc32``,
    so the sum carried across the whole K extent stays FP32.
    """
    k_blocks = cute.size(tA, mode=[2])
    part = cute.make_rmem_tensor_like(acc32, cutlass.Float16)
    for k0 in cutlass.range_constexpr(0, k_blocks, group):
        part.fill(cutlass.Float16(0.0))
        for k in cutlass.range_constexpr(k0, min(k0 + group, k_blocks)):
            cute.gemm(tiled_mma, part, tA[None, None, k], tB[None, None, k], part)
        for i in cutlass.range_constexpr(cute.size(part)):
            acc32[i] = acc32[i] + cutlass.Float32(part[i])


@dsl_user_op
def _cvt_bf16x2_to_f16x2_sat(x: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Packed bf16x2 -> f16x2 (round to nearest, saturating at +-65504)."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint32(x).ir_value(loc=loc, ip=ip)],
            "{\n"
            ".reg .b32 lo, hi;\n"
            ".reg .f32 flo, fhi;\n"
            "shl.b32 lo, $1, 16;\n"
            "and.b32 hi, $1, 0xffff0000;\n"
            "mov.b32 flo, lo;\n"
            "mov.b32 fhi, hi;\n"
            "cvt.rn.satfinite.f16x2.f32 $0, fhi, flo;\n"
            "}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def convert_smem_inplace(
    sTile_flat: cute.Tensor,
    src_dtype,
    dst_dtype,
    thread_idx: cutlass.Int32,
    num_threads: cutlass.Constexpr,
):
    """Convert a flat shared-memory tile from BF16 to FP16 in place.

    Only ``src_dtype=BFloat16`` / ``dst_dtype=Float16`` is implemented (both 16 bits
    wide). ``sTile_flat`` is a flat 16-byte-aligned tensor of the tile (its size a
    multiple of ``8 * num_threads``). The tile is treated as a 1-D array of 128-bit
    vectors (8 elements); thread ``t`` converts vectors ``t, t + num_threads, ...`` so
    that consecutive lanes touch consecutive 16-byte chunks (no bank conflicts). Values
    outside the FP16 range saturate to +-65504. The caller orders the writes against
    the readers (``fence_view_async_shared`` + a barrier of the converting threads).
    """
    if cutlass.const_expr(
        not (src_dtype is cutlass.BFloat16 and dst_dtype is cutlass.Float16)
    ):
        raise NotImplementedError("convert_smem_inplace: only BFloat16 -> Float16")
    vec = 8
    num_vec = cute.size(sTile_flat) // vec
    src_ptr = sTile_flat.iterator
    dst_ptr = cute.recast_ptr(src_ptr, dtype=dst_dtype)
    vec_layout = cute.make_layout(vec)
    copy_atom_src = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), src_dtype, num_bits_per_copy=128
    )
    copy_atom_dst = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dst_dtype, num_bits_per_copy=128
    )
    for j in cutlass.range_constexpr(num_vec // num_threads):
        off = cute.assume((thread_idx + j * num_threads) * vec, divby=vec)
        s_src = cute.make_tensor(src_ptr + off, vec_layout)
        s_dst = cute.make_tensor(dst_ptr + off, vec_layout)
        r_src = cute.make_rmem_tensor(vec, src_dtype)
        cute.copy(copy_atom_src, s_src, r_src)
        r_dst = cute.make_rmem_tensor(vec, dst_dtype)
        r_src_u32 = cute.recast_tensor(r_src, cutlass.Uint32)
        r_dst_u32 = cute.recast_tensor(r_dst, cutlass.Uint32)
        for i in cutlass.range_constexpr(vec // 2):
            r_dst_u32[i] = _cvt_bf16x2_to_f16x2_sat(r_src_u32[i])
        cute.copy(copy_atom_dst, r_dst, s_dst)


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


@cute.jit
def _mn_gemm_f16acc_to_f16(
    tiled_mma,
    c_shape,
    tA: cute.Tensor,
    tB: cute.Tensor,
    group: cutlass.Constexpr,
) -> cute.Tensor:
    """A @ B^T with FP16-accumulate MMAs, returned as an FP16 fragment.

    For a result that is rounded to FP16 right away (an MMA operand or an smem
    tile): one K group is the FP16 partial sum itself, two groups are added in FP16
    (one rounding of the exact sum, the same value as an FP32 carry rounded to FP16),
    more groups are carried in FP32 (``gemm_f16acc_carry``) and rounded at the end.
    ``c_shape`` is ``thr_mma.partition_shape_C(...)`` of the result tile.
    """
    k_blocks = cute.size(tA, mode=[2])
    num_groups = (k_blocks + group - 1) // group
    if cutlass.const_expr(num_groups <= 2):
        part0 = cute.make_rmem_tensor(c_shape, cutlass.Float16)
        part0.fill(cutlass.Float16(0.0))
        for k in cutlass.range_constexpr(0, min(group, k_blocks)):
            cute.gemm(tiled_mma, part0, tA[None, None, k], tB[None, None, k], part0)
        out = part0
        if cutlass.const_expr(num_groups == 2):
            part1 = cute.make_rmem_tensor(c_shape, cutlass.Float16)
            part1.fill(cutlass.Float16(0.0))
            for k in cutlass.range_constexpr(group, k_blocks):
                cute.gemm(tiled_mma, part1, tA[None, None, k], tB[None, None, k], part1)
            out = cute.make_rmem_tensor_like(part0, cutlass.Float16)
            out.store(part0.load() + part1.load())
    else:
        acc32 = cute.make_rmem_tensor(c_shape, cutlass.Float32)
        acc32.fill(cutlass.Float32(0.0))
        gemm_f16acc_carry(tiled_mma, acc32, tA, tB, group)
        out = cute.make_rmem_tensor_like(acc32, cutlass.Float16)
        out.store(acc32.load().to(cutlass.Float16))
    return out


@cute.jit
def _mn_gemm_f16acc_carry_scaled(
    tiled_mma,
    acc32: cute.Tensor,
    tA: cute.Tensor,
    tB: cute.Tensor,
    group: cutlass.Constexpr,
    scale: cutlass.Float32,
):
    """``gemm_f16acc_carry`` that adds ``scale * partial`` into ``acc32``."""
    k_blocks = cute.size(tA, mode=[2])
    part = cute.make_rmem_tensor_like(acc32, cutlass.Float16)
    for k0 in cutlass.range_constexpr(0, k_blocks, group):
        part.fill(cutlass.Float16(0.0))
        for k in cutlass.range_constexpr(k0, min(k0 + group, k_blocks)):
            cute.gemm(tiled_mma, part, tA[None, None, k], tB[None, None, k], part)
        for i in cutlass.range_constexpr(cute.size(part)):
            acc32[i] = acc32[i] + scale * cutlass.Float32(part[i])
