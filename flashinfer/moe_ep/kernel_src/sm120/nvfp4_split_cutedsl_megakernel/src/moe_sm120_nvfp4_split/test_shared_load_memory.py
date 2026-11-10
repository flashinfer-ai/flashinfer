"""A lane owns one shared word and overwrites it between two raw loads.

There is no data race or missing GPU synchronization: no other lane accesses
this word. This isolates LLVM memory effects from warp/TMA synchronization.
"""

import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
from cuda.bindings import driver as cuda
from moe_sm120_nvfp4_split.sm120_ptx_helpers import lds_b32_raw


@cute.kernel
def probe_kernel(output: cute.Tensor, iterations: Int32):
    lane, _, _ = cute.arch.thread_idx()
    smem = SmemAllocator()
    shared = smem.allocate_tensor(
        element_type=Int32, layout=cute.make_layout(32), byte_alignment=128
    )
    ptr = shared.iterator + lane
    total = Int32(0)
    for epoch in range(iterations):
        shared[lane] = epoch * 2 + lane
        before = lds_b32_raw(ptr)
        shared[lane] = epoch * 2 + lane + 1
        after = lds_b32_raw(ptr)
        total += after - before
    output[lane] = total


@cute.jit
def probe(output: cute.Tensor, iterations: Int32, stream: cuda.CUstream):
    probe_kernel(output, iterations).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )


@pytest.fixture(scope="module")
def shared_load_runner():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires an SM12x CUDA device")
    output = torch.empty(32, device="cuda", dtype=torch.int32)
    tensor = from_dlpack(output)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(probe, tensor, Int32(257), stream)
    return compiled, tensor, stream, output


@pytest.mark.parametrize("iterations", [1, 2, 17, 257, 1024])
def test_shared_load_observes_same_address_overwrite(shared_load_runner, iterations):
    compiled, tensor, stream, output = shared_load_runner
    compiled(tensor, Int32(iterations), stream)
    torch.cuda.synchronize()
    assert torch.equal(output, torch.full_like(output, iterations))
