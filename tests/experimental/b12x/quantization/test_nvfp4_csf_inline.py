"""Register scale reads match an independent NVFP4 byte oracle."""

from dataclasses import replace

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import numpy as np
import pytest
import torch
from cutlass.cutlass_dsl import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.quant.nvfp4_csf_inline import (
    InlineNvfp4Reader,
    prepare_inline_scales,
)
from b12x._lib.quant.nvfp4_csf import make_nvfp4_csf_batch
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils import current_cuda_stream, make_ptr
from .test_nvfp4_csf import _fixture


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows,columns", [(128, 4), (640, 20), (1024, 256)])
def test_indexed_expansion_routes_tail_and_frozen_replay(ids_dtype, rows, columns):
    from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder

    first, expected13 = _fixture(rows, columns, 0, "cuda")
    second, expected2 = _fixture(256, 16, 0, "cuda")
    outputs = tuple(torch.empty_like(t).view(torch.float8_e4m3fn)
                    for t in (expected13, expected2))
    decoder = Nvfp4CsfDecoder.prepare(
        first, second, *outputs,
        inline_scales=tuple(prepare_inline_scales(p) for p in (first, second)),
    )
    barriers = tuple(torch.empty(97, dtype=torch.int32, device="cuda") for _ in range(2))
    for count in (1, 5, 31, 63, 64, 128):
        routes = torch.ones(count, dtype=ids_dtype, device="cuda")
        if count > 1:
            routes[0] = -1
            routes[-1] = 2**32 + 3 if ids_dtype == torch.int64 else 4
        active = [1] if count < 64 else [0, 1, 2, 3]
        with kernel_resolution_guard("Indexed NVFP4 expansion with mutable routes"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                decoder.decode(routes, *outputs, barriers=barriers)
            for _ in range(2):
                routes.copy_(routes.flip(0))
                for tensor in outputs:
                    tensor.view(torch.uint8).fill_(0xD6)
                for tensor in barriers:
                    tensor.fill_(123)
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                for output, expected in zip(outputs, (expected13, expected2), strict=True):
                    assert torch.equal(output.view(torch.uint8)[active], expected[active])
                    if count < 64:
                        assert (output.view(torch.uint8)[[0, 2, 3]] == 0xD6).all()
                assert all(torch.count_nonzero(t) == 0 for t in barriers)
            graph.reset()


class GatherWords:
    def __init__(self, rows, columns):
        self.reader = InlineNvfp4Reader(rows, columns)

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        output: cute.Pointer,
        byte_count: Int64,
        experts: Int32,
        stream,
    ):
        storage = cute.make_tensor(source, cute.make_layout((byte_count,)))
        out = cute.make_tensor(
            output,
            cute.make_layout(
                (Int64(experts) * Int64(self.reader.rows * self.reader.columns // 4),)
            ),
        )
        self.kernel(storage, out, experts).launch(
            grid=(experts * Int32(self.reader.tiles), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, storage, output, experts: Int32):
        tid, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        expert = Int32(tile) // Int32(self.reader.tiles)
        within = Int32(tile) % Int32(self.reader.tiles)
        row = within // Int32(self.reader.columns // 4)
        column = within % Int32(self.reader.columns // 4)
        value = self.reader.word(storage, experts, expert, row, column, Int32(tid))
        output[Int64(tile) * Int64(128) + Int64(tid)] = value


@pytest.mark.parametrize("rows,columns", [(128, 16), (1024, 256), (4096, 32)])
def test_native_scale_bytes_and_runtime_expert_extent(rows, columns):
    batch, expected = _fixture(rows, columns, 0, "cuda")
    kernel = GatherWords(rows, columns)
    program = b12x_compile(
        kernel,
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quant.nvfp4_csf_inline_oracle", 1, (rows, columns)
        ),
    )
    for experts in (1, 4):
        plane = prepare_inline_scales(
            replace(
                batch,
                fixed=batch.fixed[:experts],
                task_offsets=batch.task_offsets[:experts],
            )
        )
        output = torch.empty_like(expected[:experts])

        def run():
            program(
                make_ptr(
                    cutlass.Uint8,
                    plane.storage.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cutlass.Uint32,
                    output.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                plane.storage.numel(),
                experts,
                current_cuda_stream(),
            )

        with kernel_resolution_guard("Inline NVFP4 scale read"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            output.fill_(0xD6)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(output, expected[:experts])
            graph.reset()


class StagedWords:
    def __init__(self, rows, columns, reuse_slots=False):
        self.reader = InlineNvfp4Reader(rows, columns)
        self.reuse_slots = reuse_slots

    @cute.jit
    def __call__(
        self, source: cute.Pointer, output: cute.Pointer, experts: Int32, stream
    ):
        kernel = self.reused_kernel if self.reuse_slots else self.kernel
        blocks = (self.reader.rows // 128 + 1) * (1 if self.reuse_slots else (self.reader.columns + 7) // 8)
        kernel(source, output, experts).launch(
            grid=(experts * Int32(blocks), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source, output, experts: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        tiles = (self.reader.rows // 128 + 1) * ((self.reader.columns + 7) // 8)
        expert, within = Int32(block) // Int32(tiles), Int32(block) % Int32(tiles)
        row = within // Int32((self.reader.columns + 7) // 8)
        column = within % Int32((self.reader.columns + 7) // 8)
        allocator = cutlass_utils.SmemAllocator()
        shared = allocator.allocate_tensor(
            cutlass.Uint8, cute.make_layout(2048), byte_alignment=16
        )
        barrier = allocator.allocate(cutlass.Int64, byte_alignment=8)
        if tid == 0:
            cute.arch.mbarrier_init(barrier, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, Int32(704))
        cute.arch.sync_threads()
        stage = Int32(block) % Int32(2)
        if tid < 32:
            self.reader.stage(
                source, experts, expert, row, column, shared.iterator, stage, barrier
            )
        cute.arch.mbarrier_wait(barrier, phase=0)
        records = source.toint() + Int64(1024) + Int64(experts) * Int64(
            self.reader.fixed_bytes + self.reader.tiles * 32
        )
        value = self.reader.shared_word(
            shared.iterator.toint() + stage * Int32(1024), Int32(tid), records
        )
        target = cute.make_tensor(
            output + Int64(block) * Int64(256), cute.make_layout(256)
        )
        target[tid] = value

    @cute.kernel
    def reused_kernel(self, source, output, experts: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        expert = Int32(block) // Int32(self.reader.rows // 128 + 1)
        row = Int32(block) % Int32(self.reader.rows // 128 + 1)
        allocator = cutlass_utils.SmemAllocator()
        shared = allocator.allocate_tensor(
            cutlass.Uint8, cute.make_layout(2048), byte_alignment=16
        )
        barrier = allocator.allocate(cutlass.Int64, byte_alignment=8)
        if tid == 0:
            cute.arch.mbarrier_init(barrier, 1)
        cute.arch.sync_threads()
        records = source.toint() + Int64(1024) + Int64(experts) * Int64(
            self.reader.fixed_bytes + self.reader.tiles * 32
        )
        for column in range((self.reader.columns + 7) // 8, unroll=4):
            if tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, Int32(704))
            cute.arch.sync_threads()
            stage = column % Int32(2)
            if tid < 32:
                self.reader.stage(
                    source,
                    experts,
                    expert,
                    row,
                    column,
                    shared.iterator,
                    stage,
                    barrier,
                )
            cute.arch.mbarrier_wait(barrier, phase=column % Int32(2))
            value = self.reader.shared_word(
                shared.iterator.toint() + stage * Int32(1024), Int32(tid), records
            )
            offset = (
                Int64(block) * Int64((self.reader.columns + 7) // 8) + Int64(column)
            ) * Int64(256)
            target = cute.make_tensor(output + offset, cute.make_layout(256))
            target[tid] = value
            cute.arch.sync_threads()


@pytest.mark.parametrize("rows,columns", [(256, 16), (256, 20), (128, 4)])
@pytest.mark.parametrize("period", [0, 1, 17, 97])
@pytest.mark.parametrize("reuse_slots", [False, True])
def test_staged_words_cover_empty_dense_and_partial_payloads(period, reuse_slots, rows, columns):
    fixed, exceptions, logical = [], [], []
    for expert in range(4):
        base = np.full(rows, 10 + expert, dtype=np.uint8)
        codes = (np.arange(rows * columns) % 16).astype(np.uint8).reshape(rows, columns)
        values = base[:, None] + codes
        positions = (
            np.arange(0, rows * columns, period, dtype=np.uint32)
            if period
            else np.empty(0, dtype=np.uint32)
        )
        values.reshape(-1)[positions] = (128 + positions % 128).astype(np.uint8)
        packed = codes[:, ::2] | (codes[:, 1::2] << 4)
        fixed.append(
            np.concatenate(
                (base.reshape(-1, 16), packed.reshape(rows // 16, -1)), axis=1
            )
        )
        exceptions.append(
            (positions | (values.reshape(-1)[positions].astype(np.uint32) << 24))
            .astype("<u4")
            .view(np.uint8)
        )
        logical.append(values)
    batch = make_nvfp4_csf_batch(
        fixed, exceptions, rows=rows, columns=columns, device="cuda"
    )
    expected = (
        torch.from_numpy(np.stack(logical))
        .cuda()
        .reshape(4, rows // 128, 4, 32, columns // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(4, rows, columns)
    )
    atoms = expected.reshape(4, rows // 128, columns // 4, 512)
    padded = torch.zeros(4, rows // 128 + 1, ((columns + 7) // 8) * 2, 512, device="cuda", dtype=torch.uint8)
    padded[:, : rows // 128, : columns // 4].copy_(atoms)
    expected = padded
    program = b12x_compile(
        StagedWords(rows, columns, reuse_slots),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quant.nvfp4_csf_staged_oracle", 3, (rows, columns, reuse_slots)
        ),
    )
    for experts in (1, 4):
        plane = prepare_inline_scales(
            replace(
                batch,
                fixed=batch.fixed[:experts],
                task_offsets=batch.task_offsets[:experts],
            )
        )
        output = torch.empty_like(expected[:experts])
        with kernel_resolution_guard("Staged NVFP4 scale reads"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program(
                    make_ptr(
                        cutlass.Uint8,
                        plane.storage.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    make_ptr(
                        cutlass.Uint32,
                        output.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    experts,
                    current_cuda_stream(),
                )
            output.fill_(0xD6)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(output, expected[:experts])
            graph.reset()
