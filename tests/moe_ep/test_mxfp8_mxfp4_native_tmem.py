# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check every native FC1 TMEM element against an independently written pattern."""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.device_epilogue import (
    DeviceEpilogueConfig,
    PersistentMxfp8Mxfp4Epilogue,
)

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.fc2_store import (
    stage_native_bf16,
)


class _NativeLoadPattern:
    def __init__(self, tokens, warps, store_mode="direct", padding=0):
        self.tokens = tokens
        self.warps = warps
        self.store_mode = store_mode
        self.padding = padding
        self.epilogue = PersistentMxfp8Mxfp4Epilogue(
            DeviceEpilogueConfig(cta_tile_tokens=tokens, epilogue_warps=warps)
        )

    @cute.jit
    def __call__(self, output, stream):
        self.kernel(output).launch(
            grid=(1, 1, 1), block=(self.warps * 32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, output):
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.padding):
            smem.allocate_tensor(cutlass.Uint8, cute.make_layout(self.padding), 16)
        if cutlass.const_expr(self.store_mode != "direct"):
            staging = smem.allocate_tensor(
                cutlass.BFloat16,
                cute.make_layout(self.warps * 512),
                16,
            )
        holding = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1), 4)
        barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.warps * 32)
        allocator = cutlass.utils.TmemAllocator(
            holding.iterator, barrier_for_retrieve=barrier, allocator_warp_id=0
        )
        allocator.allocate(512)
        allocator.wait_for_alloc()
        base = allocator.retrieve_ptr(cutlass.Float32)
        tensor = cute.make_tensor(base, cute.make_layout(512))
        if warp < 4:
            for stage in cutlass.range_constexpr(2):
                for group in cutlass.range_constexpr(self.tokens // 32):
                    registers = cute.make_rmem_tensor(
                        self.epilogue._rmem_layout().shape, cutlass.Float32
                    )
                    for token in cutlass.range_constexpr(32):
                        value = (
                            (warp * 32 + lane) * 1024
                            + stage * self.tokens
                            + group * 32
                            + token
                        )
                        if cutlass.const_expr(self.store_mode == "direct"):
                            registers[token] = cutlass.Float32(value)
                        else:
                            # Distinguish columns even at high feature rows,
                            # while exercising FP32 -> BF16 rounding.
                            registers[token] = (
                                cutlass.Float32(value % 251 - 125) * 0.0313
                            )
                    offset = warp * (32 << 16) + stage * self.tokens + group * 32
                    target = cute.make_tensor(
                        base + cute.assume(offset, divby=32),
                        self.epilogue._tmem_layout(),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32
                        ),
                        registers,
                        target,
                    )
            cute.arch.fence_view_async_tmem_store()
        barrier.arrive_and_wait()
        for stage in cutlass.range_constexpr(2):
            for group in cutlass.range_constexpr(self.tokens // 32):
                if cutlass.const_expr(self.store_mode == "direct"):
                    for half in cutlass.range_constexpr(2):
                        registers = self.epilogue._load_fc1_native_half(
                            tensor,
                            warp,
                            cutlass.Int32(group),
                            cutlass.Int32(stage),
                            half,
                        )
                        for register in cutlass.range_constexpr(16):
                            output[tid, stage, group, half, register] = registers[
                                register
                            ]
                else:
                    for chunk in cutlass.range_constexpr(2):
                        for half in cutlass.range_constexpr(2):
                            registers = self.epilogue._load_fc1_native_half(
                                tensor,
                                warp,
                                cutlass.Int32(group),
                                cutlass.Int32(stage),
                                half,
                            )
                            stage_native_bf16(
                                registers,
                                staging.iterator,
                                warp,
                                lane,
                                half,
                                chunk,
                            )
                        barrier.arrive_and_wait()
                        for row in cutlass.range_constexpr(16):
                            token = row + chunk * 16
                            byte_offset = (
                                (warp // 4) * 4096
                                + row * 256
                                + (warp % 4) * 64
                                + lane * 2
                            )
                            mask = 0x70
                            offset = byte_offset ^ ((byte_offset >> 3) & mask)
                            output[stage, group, warp, token, lane] = staging[
                                offset // 2
                            ]
                        barrier.arrive_and_wait()
        cute.arch.fence_view_async_tmem_load()
        barrier.arrive_and_wait()
        allocator.relinquish_alloc_permit()
        allocator.free(base, 512)


@pytest.mark.parametrize("tokens", [64, 128, 192])
@pytest.mark.parametrize("warps", [4, 8])
def test_native_fc1_tmem_element_mapping(tokens, warps):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100/SM103")
    output = torch.empty((warps * 32, 2, tokens // 32, 2, 16), device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    view = from_dlpack(output)
    run = cute.compile(_NativeLoadPattern(tokens, warps), view, stream)
    run(view, stream)
    tid, stage, group, half, reg = torch.meshgrid(
        *[torch.arange(n) for n in output.shape], indexing="ij"
    )
    row = (tid // 32 % 4) * 32 + half * 16 + (tid % 32) // 4 + 8 * (reg // 2 % 2)
    col = stage * tokens + group * 32 + (tid % 4) * 2 + reg % 2 + 8 * (reg // 4)
    torch.testing.assert_close(output.cpu(), (row * 1024 + col).float(), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [64, 128, 192])
@pytest.mark.parametrize("warps,padding", [(4, 0), (8, 16), (8, 128)])
def test_native_fc2_tmem_to_bf16_shared_mapping(tokens, warps, padding):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100/SM103")
    output = torch.empty(
        (2, tokens // 32, warps, 32, 32), dtype=torch.bfloat16, device="cuda"
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    view = from_dlpack(output)
    run = cute.compile(
        _NativeLoadPattern(tokens, warps, "staged256", padding), view, stream
    )
    run(view, stream)
    stage, group, warp, token, feature = torch.meshgrid(
        *[torch.arange(n, device="cuda") for n in output.shape], indexing="ij"
    )
    value = ((warp % 4) * 32 + feature) * 1024 + stage * tokens + group * 32 + token
    expected = ((value % 251 - 125).float() * 0.0313).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
