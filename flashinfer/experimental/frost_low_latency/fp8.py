# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Packed E4M3 products with FP32 accumulation for small-M projections."""

import functools
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from flashinfer.cute_dsl import fp4_common
from flashinfer.cute_dsl.fp4_common import ld_global_v4_u32, get_ptr_as_int64
from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from flashinfer.utils import get_compute_capability


@dsl_user_op
def fp8x4_product_to_float4(x: cutlass.Uint32, w: cutlass.Uint32, *, loc=None, ip=None):
    # Finite E4M3 products have <=8 significant bits and fit BF16's exponent
    # range. Packed BF16 multiply is exact here; all sums remain FP32.
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            cutlass.Uint32(x).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w).ir_value(loc=loc, ip=ip),
        ],
        """{
          .reg .b16 xl, xh, wl, wh;
          .reg .b32 xb0, xb1, wb0, wb1, p0, p1;
          mov.b32 {xl, xh}, $4;
          mov.b32 {wl, wh}, $5;
          cvt.rn.bf16x2.e4m3x2 xb0, xl;
          cvt.rn.bf16x2.e4m3x2 xb1, xh;
          cvt.rn.bf16x2.e4m3x2 wb0, wl;
          cvt.rn.bf16x2.e4m3x2 wb1, wh;
          mul.rn.bf16x2 p0, xb0, wb0;
          mul.rn.bf16x2 p1, xb1, wb1;
          shl.b32 $0, p0, 16;
          and.b32 $1, p0, 0xffff0000;
          shl.b32 $2, p1, 16;
          and.b32 $3, p1, 0xffff0000;
        }""",
        "=f,=f,=f,=f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@cute.kernel
def kernel(
    x: cute.Tensor,
    w: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    y: cute.Tensor,
    K: cutlass.Constexpr,
    N: cutlass.Constexpr,
    WARPS: cutlass.Constexpr,
    WARP_N: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    block, row, _ = cute.arch.block_idx()
    lane = tid % 32
    warp = tid // 32
    col = (block * WARPS + warp) * WARP_N
    accum = cute.make_rmem_tensor((WARP_N, 4), cutlass.Float32)
    accum.fill(0)
    for chunk in cutlass.range_constexpr(K // 512):
        k = chunk * 512 + lane * 16
        xv = ld_global_v4_u32(get_ptr_as_int64(x, cutlass.Int64(row) * K + k))
        for c in cutlass.range_constexpr(WARP_N):
            wv = ld_global_v4_u32(get_ptr_as_int64(w, cutlass.Int64(col + c) * K + k))
            for j in cutlass.range_constexpr(4):
                products = fp8x4_product_to_float4(xv[j], wv[j])
                for e in cutlass.range_constexpr(4):
                    accum[c, e] = accum[c, e] + products[e]
    alpha = sa[0] * sb[0]
    for c in cutlass.range_constexpr(WARP_N):
        total = (accum[c, 0] + accum[c, 1]) + (accum[c, 2] + accum[c, 3])
        for step in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << step)
        if lane == 0:
            y[cutlass.Int64(row) * N + col + c] = (total * alpha).to(cutlass.BFloat16)


@cute.jit
def launch(
    x,
    w,
    sa,
    sb,
    y,
    M: cutlass.Constexpr,
    K: cutlass.Constexpr,
    N: cutlass.Constexpr,
    WARPS: cutlass.Constexpr,
    WARP_N: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    kernel(x, w, sa, sb, y, K, N, WARPS, WARP_N).launch(
        grid=(N // (WARPS * WARP_N), M, 1), block=(WARPS * 32, 1, 1), stream=stream
    )


@functools.cache
def _compiled(m, k, n, device_index, capability):
    # Cache immutable geometry/device metadata, never runtime tensor addresses.
    def fake(dtype, length, alignment):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (length,), assumed_align=alignment
        )

    with torch.cuda.device(device_index):
        return build_and_load_cute_dsl_kernel(
            "frost_low_latency_fp8",
            f"m{m}_k{k}_n{n}_w4_c2",
            lambda: cute.compile(
                launch,
                fake(cutlass.Float8E4M3FN, m * k, 16),
                fake(cutlass.Float8E4M3FN, n * k, 16),
                fake(cutlass.Float32, 1, 4),
                fake(cutlass.Float32, 1, 4),
                fake(cutlass.BFloat16, m * n, 2),
                m,
                k,
                n,
                4,
                2,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            ),
            extra_key_files=(__file__, fp4_common.__file__),
        )


def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
    if out is None:
        out = torch.empty((1, A.shape[1], B.shape[2]), device=A.device, dtype=dtype)
    m, k, n = A.shape[1], A.shape[2], B.shape[2]
    with torch.cuda.device(A.device):
        fn = _compiled(m, k, n, A.device.index, get_compute_capability(A.device))
        fn(
            A.reshape(-1),
            B.transpose(-2, -1).reshape(-1),
            A_scale.reshape(1),
            B_scale.reshape(1),
            out.reshape(-1),
        )
    return out
