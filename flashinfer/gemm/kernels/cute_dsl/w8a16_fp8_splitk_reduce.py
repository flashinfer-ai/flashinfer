# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Split-K reduce for the W8A16 GEMM: sums the fp32 partials of the K-slices, applies
the per-tensor scale and the optional per-N bias, and casts to the activation dtype.

    out[m, n] = cast(scale * sum_l partials[l, m, n] + bias[n])

``partials`` is the GEMM kernel's C for ``splits`` slices: [splits, M_padded, N] fp32,
N contiguous (the kernel's C[n, splits * M_padded] column-major; M_padded >= M).  One
thread handles eight consecutive n (one 16-byte store, two 16-byte loads per slice).
The slice count is a compile-time constant so the slice loop unrolls and every
slice's loads are in flight at once (the kernel is latency-bound: 128 CTAs for a
1024 x 256 output)."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

_VEC = 8
_THREADS = 256


class W8A16SplitKReduce:
    def __init__(self, has_bias: bool, splits: int):
        self.has_bias = has_bias
        self.splits = splits

    @cute.jit
    def wrapper(
        self,
        partials_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        slice_vecs: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        """``slice_vecs``: elements between two slices' partials, in 8-element vectors
        (M_padded * N / 8)."""
        mn_vecs = m * n // _VEC
        num_blocks = cute.ceil_div(mn_vecs, _THREADS)
        self.kernel(
            partials_ptr, scale_ptr, bias_ptr, out_ptr, mn_vecs, n // _VEC, slice_vecs
        ).launch(grid=[num_blocks, 1, 1], block=[_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        partials_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        mn_vecs: cutlass.Int64,
        n_vecs: cutlass.Int64,
        slice_vecs: cutlass.Int64,
    ):
        # The views are built here, in units of 8-element vectors, with every stride
        # spelled as 8 * (a vector count): the compiler then sees that each vector
        # is 16-byte aligned and keeps the 16-byte loads and stores.  (A stride
        # computed on the host arrives as an opaque integer and the second slice's
        # loads came out as eight 4-byte loads.)
        partials = cute.make_tensor(
            partials_ptr,
            cute.make_layout(
                (_VEC, mn_vecs, self.splits), stride=(1, _VEC, _VEC * slice_vecs)
            ),
        )
        out = cute.make_tensor(
            out_ptr, cute.make_layout((_VEC, mn_vecs), stride=(1, _VEC))
        )
        bias = cute.make_tensor(
            bias_ptr, cute.make_layout((_VEC, n_vecs), stride=(1, _VEC))
        )
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = bidx * _THREADS + tidx
        if idx < mn_vecs:
            acc = partials[(None, idx, 0)].load()
            for s in cutlass.range_constexpr(1, self.splits):
                acc = acc + partials[(None, idx, s)].load()
            acc = cutlass.Float32(scale[0]) * acc
            if cutlass.const_expr(self.has_bias):
                acc = acc + bias[(None, idx % n_vecs)].load().to(cutlass.Float32)
            out[(None, idx)].store(acc.to(out.element_type))
