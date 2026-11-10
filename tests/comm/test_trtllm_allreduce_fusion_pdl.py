"""Ordering test for the fused all-reduce kernels under Programmatic Dependent Launch.

With ``launch_with_pdl=True`` the fused kernel may start executing before the
kernel that produces ``residual_in`` has finished writing it, so every read of
``residual_in`` has to be issued after ``cudaGridDependencySynchronize()``.

The producer kernel below widens that window on purpose: it triggers the
dependent launch first, spins, and only then writes ``residual_in``. A read
issued before the sync therefore observes whatever was in the buffer
beforehand, which this test sets to NaN so a stale read shows up in the output
rather than being merely numerically wrong.
"""

import multiprocessing as mp
import pathlib
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

import flashinfer.comm as comm
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda:0"))[0] not in (9, 10, 12),
    reason="trtllm_comm kernels support SM90/SM100/SM12x only",
)

MAX_TOKEN_NUM = 128
TOKEN_NUM = 128
SF_VEC_SIZE = 16
TEST_LOOP = 20
# Long enough that the fused kernel is resident and past its prologue before the
# producer writes anything, short enough to keep the test quick.
SPIN_CYCLES = 4_000_000

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FLASHINFER_INCLUDE = str(_REPO_ROOT / "include")
_SPDLOG_INCLUDE = str(_REPO_ROOT / "3rdparty" / "spdlog" / "include")
_CUDA_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

_CPP_SOURCE = r"""
void pdl_delayed_producer(at::Tensor out, at::Tensor value, int64_t spin_cycles);
"""

# The copy is done on raw 16-bit words, so one kernel serves both float16 and
# bfloat16 without a dtype dispatch.
_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

__global__ void trigger_then_write_kernel(uint16_t* out, const uint16_t* value, int64_t n,
                                          int64_t spin_cycles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Let the dependent fused kernel start now, before anything is written.
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  int64_t start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = value[i];
  }
}

void pdl_delayed_producer(at::Tensor out, at::Tensor value, int64_t spin_cycles) {
  TORCH_CHECK(out.is_cuda() && value.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(out.is_contiguous() && value.is_contiguous(), "expected contiguous tensors");
  TORCH_CHECK(out.scalar_type() == value.scalar_type(), "dtype mismatch");
  TORCH_CHECK(out.element_size() == 2, "expected a 16-bit dtype");
  TORCH_CHECK(out.numel() == value.numel(), "size mismatch");
  trigger_then_write_kernel<<<64, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<uint16_t*>(out.data_ptr()), static_cast<const uint16_t*>(value.data_ptr()),
      out.numel(), spin_cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _load_producer():
    major, minor = torch.cuda.get_device_capability()
    gencode = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return load_inline(
        name="test_trtllm_allreduce_fusion_pdl_producer",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE, _SPDLOG_INCLUDE],
        extra_cuda_cflags=[*_CUDA_FLAGS, gencode],
        functions=["pdl_delayed_producer"],
        verbose=False,
    )


def _run_pdl_ordering_worker(
    world_size,
    rank,
    dtype,
    hidden_dim,
    distributed_init_port,
    launch_with_pdl=True,
    gpu_offset=0,
):
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    producer = _load_producer()

    try:
        ipc_handles, workspace_tensor, workspace_metadata = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                MAX_TOKEN_NUM,
                hidden_dim,
                group=group,
                use_fp32_lamport=False,
                create_metadata=True,
            )
        )

        message_size = TOKEN_NUM * hidden_dim
        allreduce_in = torch.randn(message_size, dtype=dtype, device=device)
        residual_value = torch.randn(message_size, dtype=dtype, device=device)
        rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)
        rms_eps = 1e-3

        residual_in = torch.empty(message_size, dtype=dtype, device=device)
        allreduce_out = torch.empty(message_size, dtype=dtype, device=device)
        residual_out = torch.empty(message_size, dtype=dtype, device=device)
        norm_out = torch.empty(message_size, dtype=dtype, device=device)
        quant_out = torch.empty(message_size, dtype=dtype, device=device)
        scale_out = torch.empty(message_size // SF_VEC_SIZE, dtype=dtype, device=device)

        # use_oneshot True and False select the two kernels that read
        # residual_in ahead of the grid dependency sync.
        #
        # A failure is recorded rather than raised here: the race can be lost on
        # one rank and not another, and leaving the loop early would strand the
        # passing rank on a barrier nobody else reaches, turning the regression
        # into a hang instead of a failure.
        failures = []
        for use_oneshot in (True, False):
            stale = 0
            for _ in range(TEST_LOOP):
                residual_in.fill_(float("nan"))
                dist.barrier(group=group)
                torch.cuda.synchronize()

                producer.pdl_delayed_producer(residual_in, residual_value, SPIN_CYCLES)
                comm.trtllm_allreduce_fusion(
                    allreduce_in=allreduce_in,
                    world_size=world_size,
                    world_rank=rank,
                    token_num=TOKEN_NUM,
                    hidden_dim=hidden_dim,
                    workspace_ptrs=workspace_tensor,
                    launch_with_pdl=launch_with_pdl,
                    use_oneshot=use_oneshot,
                    trigger_completion_at_end=False,
                    fp32_acc=False,
                    pattern_code=comm.AllReduceFusionPattern.kARResidualRMSNorm,
                    allreduce_out=allreduce_out,
                    residual_in=residual_in,
                    residual_out=residual_out,
                    norm_out=norm_out,
                    quant_out=quant_out,
                    scale_out=scale_out,
                    rms_gamma=rms_gamma,
                    rms_eps=rms_eps,
                    weight_bias=0.0,
                    scale_factor=None,
                    layout_code=comm.QuantizationSFLayout.LINEAR,
                    metadata=workspace_metadata,
                )
                torch.cuda.synchronize()

                if not (
                    torch.isfinite(residual_out).all()
                    and torch.isfinite(norm_out).all()
                ):
                    stale += 1

            if stale:
                failures.append(f"use_oneshot={use_oneshot}: {stale}/{TEST_LOOP}")
            dist.barrier(group=group)

        assert not failures, (
            f"rank {rank}: residual_in was read before the PDL grid dependency "
            f"sync (launch_with_pdl={launch_with_pdl}) in " + ", ".join(failures)
        )
    finally:
        dist.barrier(group=group)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
            ipc_handles, group=group
        )
        dist.destroy_process_group(group=group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    dtype: torch.dtype,
    hidden_dim: int,
    test_target: Any,
    target_args: tuple = (),
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (
            (
                world_size,
                i,
                dtype,
                hidden_dim,
                distributed_init_port,
            )
            + target_args
            + (gpu_offset,)
        )
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


# Run as: pytest tests/comm/test_trtllm_allreduce_fusion_pdl.py
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_dim", [4096])
@pytest.mark.parametrize("launch_with_pdl", [True, False])
def test_trtllm_allreduce_fusion_pdl_ordering(
    world_size, dtype, hidden_dim, launch_with_pdl
):
    """residual_in must not be read before cudaGridDependencySynchronize().

    launch_with_pdl=False is the control arm: plain stream ordering makes the
    producer's writes visible regardless, so it passes either way and shows the
    harness is not simply reporting NaN unconditionally.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    # Warm the extension cache in the parent so the workers do not all compile.
    _load_producer()

    multi_process_parallel(
        world_size,
        dtype,
        hidden_dim,
        _run_pdl_ordering_worker,
        target_args=(launch_with_pdl,),
    )
    print(f"pdl ordering tp={world_size} launch_with_pdl={launch_with_pdl}: OK")


if __name__ == "__main__":
    test_trtllm_allreduce_fusion_pdl_ordering(2, torch.bfloat16, 4096, True)
