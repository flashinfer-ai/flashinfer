# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Test-owned SM90 FP8 instruction-sequence reference.

This deliberately shares no MegaMoE decode, routing, scheduling, or epilogue
code. Independent Torch-decoded E4M3 operands enter a simple synchronous RS
WGMMA loop. Each K32 instruction accumulates into the same FP32 registers.
Changing GEMM tiling/promotion with torch._scaled_mm is not an instruction-exact
oracle at long K; subsequent SwiGLU and FP8 rounding can amplify the difference.
FC2 scale promotion uses explicit fma.rn.f32, not separate multiply/add kernels.
This models the implementation's arithmetic, not an ideal full-precision GEMM.
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import OperandMajorMode, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm


@cute.kernel
def _kernel(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor, mma: cute.TiledMma):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    atom = warpgroup.make_smem_layout_atom(
        warpgroup.SmemLayoutAtomKind.K_SW32, cutlass.Float8E4M3FN
    )
    b_layout = cute.tile_to_shape(atom, (64, 32), order=(0, 1))
    sb = smem.allocate_tensor(
        cutlass.Float8E4M3FN, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner
    )
    thr = mma.get_slice(tid)
    rb = mma.make_fragment_B(thr.partition_B(sb))
    ra = mma.make_fragment_A(mma.partition_shape_A((64, 32)))
    gc = thr.partition_C(cute.local_tile(output, (64, 64), (block, 0)))
    accum = cute.make_rmem_tensor(gc.shape, cutlass.Float32)
    accum.fill(0.0)
    mma.set(warpgroup.Field.ACCUMULATE, False)
    warpgroup.fence()
    for k32 in cutlass.range(cute.size(a, mode=[1]) // 32, unroll=1):
        for i in cutlass.range_constexpr(16):
            linear = tid + i * 128
            row = linear // 32
            col = linear % 32
            sb[row, col] = b[row, k32 * 32 + col]
        ga = thr.partition_A(cute.local_tile(a, (64, 32), (block, k32)))
        for i in cutlass.range_constexpr(cute.size(ra)):
            ra[i] = ga[i]
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        warpgroup.fence()
        cute.gemm(mma, accum, ra, rb, accum)
        mma.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
        cute.arch.sync_threads()
    for i in cutlass.range_constexpr(cute.size(accum)):
        gc[i] = accum[i]


@cute.jit
def _launch(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor):
    mma = utils.sm90.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        (1, 1, 1),
        (64, 64),
        a_source=warpgroup.OperandSource.RMEM,
    )
    _kernel(a, b, output, mma).launch(
        grid=(cute.size(a, mode=[0]) // 64, 1, 1), block=(128, 1, 1)
    )


_cache = {}


def rs_k32_mm(a, b):
    """Reference [tokens,K] @ [K,channels] for at most64 routed rows."""
    assert a.is_cuda and b.is_cuda and a.device == b.device
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    tokens, k = a.shape
    assert 0 < tokens <= 64 and b.shape[0] == k and k % 32 == 0
    channels = b.shape[1]
    assert channels % 64 == 0
    weights = b.T.contiguous()
    activations = torch.zeros((64, k), dtype=a.dtype, device=a.device)
    activations[:tokens].view(torch.uint8).copy_(a.contiguous().view(torch.uint8))
    output = torch.empty((channels, 64), dtype=torch.float32, device=a.device)
    args = tuple(from_dlpack(item) for item in (weights, activations, output))
    key = (a.device.index, channels, k)
    if key not in _cache:
        _cache[key] = cute.compile(_launch, *args)
    _cache[key](*args)
    return output.T[:tokens].contiguous()


@cute.kernel
def _fma_kernel(
    accum: cute.Tensor, value: cute.Tensor, scale: cute.Tensor, output: cute.Tensor
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 256 + tid
    if index < cute.size(accum):
        output[index] = cutlass.Float32(
            llvm.inline_asm(
                cutlass.Float32.mlir_type,
                [
                    value[index].ir_value(),
                    scale[index].ir_value(),
                    accum[index].ir_value(),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )


@cute.jit
def _fma_launch(
    accum: cute.Tensor, value: cute.Tensor, scale: cute.Tensor, output: cute.Tensor
):
    _fma_kernel(accum, value, scale, output).launch(
        grid=((cute.size(accum) + 255) // 256, 1, 1), block=(256, 1, 1)
    )


_fma_cache = {}


def fma_add(accum, value, scale):
    """One-round FP32 reference for accum + value * broadcast(scale)."""
    assert accum.dtype == value.dtype == scale.dtype == torch.float32
    assert accum.device == value.device == scale.device and accum.is_cuda
    assert accum.shape == value.shape
    expanded_scale = scale.expand_as(value).contiguous()
    output = torch.empty_like(accum)
    operands = tuple(
        item.contiguous().view(-1) for item in (accum, value, expanded_scale, output)
    )
    args = tuple(from_dlpack(item) for item in operands)
    key = (accum.device.index, accum.numel())
    if key not in _fma_cache:
        _fma_cache[key] = cute.compile(_fma_launch, *args)
    _fma_cache[key](*args)
    return output
