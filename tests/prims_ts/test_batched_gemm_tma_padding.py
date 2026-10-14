# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify routed TMA data and zero-filled padding, including partial gathers."""

import pytest
import torch
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.utils import SmemAllocator
from cutlass.cute.runtime import from_dlpack

from flashinfer.prims_ts import is_prims_ts_device_supported
from flashinfer.prims_ts.batched_gemm.smem_ab_resources import (
    SmemTmaGatherResource,
    _tma_gather4_cta,
)
from flashinfer.prims_ts.batched_gemm.smem_sf_resources import (
    SmemSfGatherResource,
    _tma_gather4_cta as _tma_gather4_sf_cta,
)


class _GatherProbe:
    def __init__(self, scale_factors):
        self.load_row = (
            SmemSfGatherResource._load_routed_row_or_oob
            if scale_factors
            else SmemTmaGatherResource._load_routed_row_or_oob
        )
        self.copy_rows = _tma_gather4_sf_cta if scale_factors else _tma_gather4_cta

    @cute.kernel
    def gather(
        self,
        desc: cutlass.GridConstant[cuda.TensorMap],
        routes: cute.Tensor,
        output: cute.Tensor,
        valid_rows: cutlass.Int32,
    ):
        allocator = SmemAllocator()
        data = allocator.allocate_tensor(
            output.element_type, cute.make_layout((64,)), byte_alignment=128
        )
        barrier = allocator.allocate_array(cutlass.Uint64, 1)
        tid = cute.arch.thread_idx()[0]
        if tid == 0:
            cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        self.route_map = cutlass.make_array_view(routes)
        row = cute.arch.block_idx()[0] * 4
        r0 = self.load_row(self, row, row, valid_rows)
        r1 = self.load_row(self, row + 1, row + 1, valid_rows)
        r2 = self.load_row(self, row + 2, row + 2, valid_rows)
        r3 = self.load_row(self, row + 3, row + 3, valid_rows)
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                barrier, 64 * output.element_type.width // 8
            )
            self.copy_rows(
                cutlass.make_array_view(data),
                desc.get_ptr(),
                cutlass.Int32(0),
                r0,
                r1,
                r2,
                r3,
                cutlass.make_array_view(
                    cute.make_tensor(barrier, cute.make_layout((1,)))
                ),
            )
        cute.arch.mbarrier_wait(barrier, 0)
        for i in cutlass.range_constexpr(2):
            index = tid + i * 32
            output[row + index // 16, index % 16] = data[index]

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        routes: cute.Tensor,
        output: cute.Tensor,
        valid_rows: cutlass.Int32,
    ):
        desc = cuda.create_tensor_map_tiled_from_view(source, box_dims=(1, 16))
        self.gather(desc, routes, output, valid_rows).launch(
            grid=(2, 1, 1), block=(32, 1, 1)
        )


@pytest.mark.parametrize("scale_factors", [False, True])
@pytest.mark.parametrize("valid_rows", [0, 1, 3, 4, 5, 8])
def test_tma_gather_zero_fills_padding(scale_factors, valid_rows):
    if not torch.cuda.is_available() or not is_prims_ts_device_supported(
        torch.device("cuda")
    ):
        pytest.skip("PrimsTS device support required")
    dtype = torch.uint8 if scale_factors else torch.bfloat16
    source = torch.arange(1, 129, device="cuda").reshape(8, 16).to(dtype)
    ids = [7, 0, 7, 2, 6, 1, 3, 5]
    routes = torch.tensor(ids[: max(1, valid_rows)], dtype=torch.int32, device="cuda")
    output = torch.empty_like(source)
    args = (*map(from_dlpack, (source, routes, output)), cutlass.Int32(valid_rows))
    probe = cute.compile(_GatherProbe(scale_factors), *args)
    probe(*args)
    expected = torch.zeros_like(source)
    expected[:valid_rows] = source[ids[:valid_rows]]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
