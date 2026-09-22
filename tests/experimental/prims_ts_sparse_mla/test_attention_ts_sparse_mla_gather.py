# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.experimental import cuda, primitives as prims
from cutlass.utils import SmemAllocator

from flashinfer.attention.prims_ts.kernels.mla_decode.helpers.gather import gather4
from flashinfer.attention.prims_ts.kernels.tensor_map import (
    create_tensor_map_tiled_from_view,
)


class GatherCopy:
    def __init__(self, issue_warps, cta_group):
        self.issue_warps = issue_warps
        self.cta_group = cta_group

    @cute.jit
    def __call__(self, x, indices, out, stream):
        x_tma = cute.make_tensor(x.iterator, cute.select(x.layout, mode=[1, 0]))
        descriptor = create_tensor_map_tiled_from_view(
            x_tma,
            box_dims=(x.shape[1], 1),
            stride_order=(0, 1),
            swizzle=cuda.TensorMapSwizzle.none,
        )
        self.copy(descriptor, indices, out).launch(
            grid=(self.cta_group, 1, 1),
            block=(32 * self.issue_warps, 1, 1),
            cluster=(self.cta_group, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def copy(self, descriptor: cutlass.GridConstant[cuda.TensorMap], indices, out):
        tid = cute.arch.thread_idx()[0]
        cta = cute.arch.block_idx()[0]
        warp = tid // 32
        allocator = SmemAllocator()
        tile = allocator.allocate_tensor(
            out.element_type,
            cute.make_layout((128, out.shape[2]), stride=(out.shape[2], 1)),
            byte_alignment=128,
        )
        barrier = allocator.allocate(Int64, byte_alignment=8)
        if tid == 0:
            cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if cta == 0 and tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                barrier,
                self.cta_group * 128 * out.shape[2] * (out.element_type.width // 8),
            )
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for local_group in cutlass.range_constexpr(32 // self.issue_warps):
            group = warp + local_group * self.issue_warps
            if prims.elect_sync():
                gather4(
                    tile.iterator + group * 4 * out.shape[2],
                    descriptor.get_ptr(),
                    Int32(0),
                    indices[cta, group * 4],
                    indices[cta, group * 4 + 1],
                    indices[cta, group * 4 + 2],
                    indices[cta, group * 4 + 3],
                    barrier,
                    cta_group=self.cta_group,
                )
        if cta == 0:
            cute.arch.mbarrier_wait(barrier, 0)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for i in cutlass.range(tid, 128 * out.shape[2], 32 * self.issue_warps):
            out[cta, i // out.shape[2], i % out.shape[2]] = tile[
                i // out.shape[2], i % out.shape[2]
            ]
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("issue_warps", [1, 2, 4])
@pytest.mark.parametrize("cta_group", [1, 2])
def test_gather4_native(dtype, issue_warps, cta_group):
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver

    width = 128 // torch.empty((), dtype=dtype).element_size()
    x = (
        ((torch.arange(257 * width, device="cuda") % 31).float() - 15)
        .reshape(257, width)
        .to(dtype)
    )
    indices = (
        ((torch.arange(cta_group * 128, device="cuda") * 37) % 257)
        .to(torch.int32)
        .view(cta_group, 128)
    )
    indices[:, 3::11] = -1
    out = torch.empty(cta_group, 128, width, device="cuda", dtype=dtype)
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    tensors = []
    for t in (x, indices, out):
        view = t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
        tensor = from_dlpack(view, assumed_align=16)
        if t.dtype == torch.float8_e4m3fn:
            tensor.element_type = cutlass.Float8E4M3FN
        tensors.append(tensor)
    kernel = cute.compile[cute.FrontendNext](
        GatherCopy(issue_warps, cta_group), *tensors, stream
    )
    kernel(*tensors, stream)
    torch.cuda.synchronize()
    expected = x.float()[indices.clamp_min(0).long()]
    expected[indices < 0] = 0
    torch.testing.assert_close(out.float(), expected, atol=0, rtol=0)
