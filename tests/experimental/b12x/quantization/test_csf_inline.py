"""Inline readers preserve native bytes without whole-expert expansion scratch."""

import cutlass
import cutlass.cute as cute
import cutlass.utils
import pytest
import torch
from cutlass.cutlass_dsl import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane, upload_csf_block_payload
from b12x._lib.quant.csf_inline import CsfInlineReader, Nvfp4CsfStageReader
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from .test_csf_blocks import planes_and_native
from ..conftest import require_b12x


class ReadInline:
    def __init__(self, geometry):
        self.reader = CsfInlineReader(geometry)

    @cute.jit
    def __call__(self, source: cute.Pointer, output: cute.Pointer, stream):
        self.kernel(source, output).launch(
            grid=(
                self.reader.plane.storage.experts * self.reader.plane.storage.groups,
                1,
                1,
            ),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source, output):
        lane, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        groups = self.reader.plane.storage.groups * self.reader.plane.sections
        expert, group = Int32(block) // Int32(groups), Int32(block) % Int32(groups)
        shared = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Uint32, cute.make_layout(128), byte_alignment=16
        )
        target = cute.recast_ptr(shared.iterator, dtype=cutlass.Uint8)
        self.reader.stage_compressed(source, expert, group, target, Int32(lane))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        self.reader.expand_compressed(source, group, target, Int32(lane))
        row_group = (
            group
            % Int32(self.reader.plane.storage.groups)
            // Int32(self.reader.plane.storage.column_groups)
        )
        column_group = group % Int32(self.reader.plane.storage.column_groups)
        nr = cutlass.min(
            Int32(128), Int32(self.reader.plane.storage.rows) - row_group * Int32(128)
        )
        nc = cutlass.min(
            Int32(4), Int32(self.reader.plane.columns) - column_group * Int32(4)
        )
        for quarter in cutlass.range_constexpr(4):
            position = Int32(lane) * Int32(16) + Int32(quarter * 4)
            valid = (
                position < nr * nc
                if cutlass.const_expr(self.reader.plane.destination in (1, 3))
                else True
            )
            if valid:
                within = position
                if cutlass.const_expr(self.reader.plane.destination == 1):
                    within = (
                        position // nr * Int32(self.reader.plane.rows) + position % nr
                    )
                offset = (
                    Int64(expert * self.reader.expert_bytes)
                    + self.reader.plane.group_output_offset(group)
                    + within.to(Int64)
                )
                value = shared[position // Int32(4)]
                cute.make_tensor(
                    cute.recast_ptr(output + offset, dtype=cutlass.Uint32),
                    cute.make_layout(1),
                )[0] = value


@pytest.mark.parametrize("preparation", ["upload", "gpu"])
@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize(
    "destination,rows,columns,sections",
    [(0, 256, 16, 1), (1, 320, 10, 1), (2, 512, 16, 2), (3, 1152, 18, 2)],
)
@pytest.mark.parametrize("dense", [False, True])
def test_inline_native_bytes(
    quantization, destination, rows, columns, sections, dense, preparation
):
    device = require_b12x()
    planes, _ = planes_and_native(
        rows, columns, 2, dense=dense, quantization=quantization
    )
    payload = upload_csf_block_payload(
        planes,
        device,
        native_codes=True,
        group_record_bytes=3,
        destination=destination,
        group_rows=rows // sections,
    )
    if preparation == "gpu":
        from b12x.loader._csf_source import upload_csf_planes
        from b12x._lib.quant._csf_prepare import prepare_csf_payload

        payload = prepare_csf_payload(
            upload_csf_planes(planes, device),
            destination=destination,
            group_rows=rows // sections,
            value_table=torch.arange(256, device=device).clamp_max(247).to(torch.uint8),
        )
    plane = Nvfp4CompactBlockPlane(payload, layout=int(destination == 1))
    expected = torch.empty(
        (2, plane.rows, plane.columns), dtype=torch.uint8, device=device
    )
    other = torch.empty_like(expected)
    Nvfp4CsfDecoder.prepare(plane, plane, expected, other).decode_all(expected, other)
    output = torch.full_like(expected, 0xD3)
    pointer = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    program = b12x_compile(
        ReadInline(plane.geometry),
        pointer,
        pointer,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "test.csf_inline", 2, (plane.geometry,)
        ),
    )
    source_ptr = make_ptr(
        cutlass.Uint8,
        payload.storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    output_ptr = make_ptr(
        cutlass.Uint8, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    program(source_ptr, output_ptr, current_cuda_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program(source_ptr, output_ptr, current_cuda_stream())
    output.fill_(0xD3)
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


class ReadNvfp4Stage:
    def __init__(self, rows, columns, experts):
        self.reader = Nvfp4CsfStageReader(rows, columns, experts)
        self.rows, self.columns, self.experts = rows, columns, experts

    @cute.jit
    def __call__(self, source: cute.Pointer, output: cute.Pointer, stream):
        self.kernel(source, output).launch(
            grid=(self.experts * self.rows * self.columns // 1024, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source, output):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        alloc = cutlass.utils.SmemAllocator()
        shared = alloc.allocate_tensor(
            cutlass.Uint32, cute.make_layout(256), byte_alignment=16
        )
        barrier = alloc.allocate_tensor(
            cutlass.Int64, cute.make_layout(1), byte_alignment=8
        )
        if tid == 0:
            cute.arch.mbarrier_init(barrier.iterator, 65)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        expert = Int32(block) // Int32(self.rows * self.columns // 1024)
        tile = Int32(block) % Int32(self.rows * self.columns // 1024)
        out = cute.make_tensor(
            cute.recast_ptr(output, dtype=cutlass.Uint32),
            cute.make_layout(4 * self.experts * self.rows * self.columns // 4),
        )
        for phase in cutlass.range_constexpr(4):
            if tid < 32:
                if tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(barrier.iterator, 0)
                self.reader.stage(
                    source,
                    Int32(self.experts),
                    (expert + Int32(phase)) % Int32(self.experts),
                    tile // Int32(self.columns // 8),
                    tile % Int32(self.columns // 8),
                    shared.iterator,
                    Int32(0),
                    barrier.iterator,
                )
            else:
                cute.arch.mbarrier_wait(barrier.iterator, phase=phase % 2)
                for half in cutlass.range_constexpr(2):
                    target = cute.recast_ptr(
                        shared.iterator, dtype=cutlass.Uint8
                    ) + Int32(half * 512)
                    self.reader.expand_compressed(
                        source,
                        tile * Int32(2) + Int32(half),
                        target,
                        Int32(tid) - Int32(32),
                    )
                for i in cutlass.range_constexpr(8):
                    word = Int32(tid) - Int32(32) + Int32(i * 32)
                    out[
                        Int32(phase * self.experts * self.rows * self.columns // 4)
                        + Int32(block) * Int32(256)
                        + word
                    ] = shared[word]
            cute.arch.sync_threads()


def test_nvfp4_stage_publication():
    device = require_b12x()
    planes, expected = planes_and_native(256, 16, 2, dense=True)
    payload = upload_csf_block_payload(
        planes, device, native_codes=True, group_record_bytes=3
    )
    output = torch.empty((4, *expected.shape), dtype=expected.dtype, device=device)
    pointer = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    program = b12x_compile(
        ReadNvfp4Stage(256, 16, 2),
        pointer,
        pointer,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("test.csf_inline_stage", 2, ()),
    )
    program(
        make_ptr(
            cutlass.Uint8,
            payload.storage.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        current_cuda_stream(),
    )
    torch.testing.assert_close(
        output.cpu(),
        torch.stack([expected.roll(-phase, dims=0) for phase in range(4)]),
        rtol=0,
        atol=0,
    )
