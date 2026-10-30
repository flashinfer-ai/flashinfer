# SPDX-License-Identifier: Apache-2.0
# CuTe MMA/copy primitives extracted from the local validated pipeline32 work.
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import nvvm


@dsl_user_op
def _sync(*, barrier_id, number_of_threads, loc=None, ip=None):
    nvvm.barrier_cta_sync(
        cutlass.Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=cutlass.Int32(number_of_threads).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _transpose(t: cute.Tensor):
    return cute.make_tensor(
        t.iterator,
        cute.make_layout(
            (cute.size(t, mode=[1]), cute.size(t, mode=[0])),
            stride=(t.layout.stride[1], t.layout.stride[0]),
        ),
    )


@cute.jit
def _mma(dtype: cutlass.Constexpr, m: cutlass.Constexpr, n: cutlass.Constexpr):
    return cute.make_tiled_mma(
        warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
        cute.make_layout((2, 2, 1)),
        permutation_mnk=(m, n, 16),
    )


@cute.jit
def _gemm(
    a: cute.Tensor,
    b_nk: cute.Tensor,
    acc: cute.Tensor,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    k: cutlass.Constexpr,
    use_ldmatrix: cutlass.Constexpr,
):
    """acc += A[M,K] B[N,K]^T, all 128 threads in the calling WG.

    Load one K=16 fragment at a time to bound register lifetimes. LDSM moves
    8x8 matrices directly into MMA register layout. Transposed shared views use
    the transposing atom. Pitches and K-tile offsets remain 16-byte aligned.
    The generic branch reproduces the original loader for paired ablations.
    """
    tid, _, _ = cute.arch.thread_idx()
    mma = _mma(a.element_type, m, n)
    thr = mma.get_slice(tid % 128)
    for block in cutlass.range_constexpr(k // 16):
        sa = cute.local_tile(a, (m, 16), (0, block))
        sb = cute.local_tile(b_nk, (n, 16), (0, block))
        ra = thr.make_fragment_A(thr.partition_A(sa))
        rb = thr.make_fragment_B(thr.partition_B(sb))
        if cutlass.const_expr(use_ldmatrix):
            atom_a = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(
                    transpose=(a.layout.stride[0] == 1), num_matrices=4
                ),
                a.element_type,
            )
            atom_b = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(
                    transpose=(b_nk.layout.stride[0] == 1), num_matrices=4
                ),
                b_nk.element_type,
            )
            copy_a = cute.make_tiled_copy_A(atom_a, mma)
            copy_b = cute.make_tiled_copy_B(atom_b, mma)
            ta, tb = copy_a.get_slice(tid % 128), copy_b.get_slice(tid % 128)
            cute.copy(copy_a, ta.partition_S(sa), ta.retile(ra))
            cute.copy(copy_b, tb.partition_S(sb), tb.retile(rb))
        else:
            cute.autovec_copy(thr.partition_A(sa), ra)
            cute.autovec_copy(thr.partition_B(sb), rb)
        cute.gemm(mma, acc, ra, rb, acc)


@cute.jit
def _acc(dtype: cutlass.Constexpr, m: cutlass.Constexpr, n: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    thr = _mma(dtype, m, n).get_slice(tid % 128)
    out = cute.make_rmem_tensor(thr.partition_shape_C((m, n)), cutlass.Float32)
    out.fill(0.0)
    return out


@cute.jit
def _coords(dtype: cutlass.Constexpr, m: cutlass.Constexpr, n: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    return (
        _mma(dtype, m, n)
        .get_slice(tid % 128)
        .partition_C(cute.make_identity_tensor((m, n)))
    )


@cute.jit
def _store(
    acc: cute.Tensor, dst: cute.Tensor, m: cutlass.Constexpr, n: cutlass.Constexpr
):
    tid, _, _ = cute.arch.thread_idx()
    atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), dst.element_type
    )
    copy = cute.make_tiled_copy_C(atom, _mma(dst.element_type, m, n))
    thread = copy.get_slice(tid % 128)
    target = thread.partition_D(dst)
    target = cute.make_tensor(target.iterator.align(16), target.layout)
    converted = cute.make_fragment_like(acc, dst.element_type)
    for i in cutlass.range_constexpr(cute.size(acc)):
        converted[i] = dst.element_type(acc[i])
    cute.copy(copy, thread.retile(converted), target)


@cute.jit
def _vector_load_tile(
    src, dst, offset, row_stride: cutlass.Constexpr, width: cutlass.Constexpr, lane
):
    """Copy a complete aligned 32-row tile with 128-bit global/shared transfers."""
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), src.element_type, num_bits_per_copy=128
    )
    threads_per_row = width // 8
    copy = cute.make_tiled_copy_tv(
        atom,
        cute.make_layout(
            (128 // threads_per_row, threads_per_row), stride=(threads_per_row, 1)
        ),
        cute.make_layout((1, 8), stride=(8, 1)),
    )
    source = cute.make_tensor(
        src.iterator + cute.assume(offset, divby=8),
        cute.make_layout((32, width), stride=(row_stride, 1)),
    )
    thr = copy.get_slice(lane)
    cute.copy(copy, thr.partition_S(source), thr.partition_D(dst))
