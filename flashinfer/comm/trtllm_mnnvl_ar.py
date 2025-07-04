"""
MNNVL (Multi-Node NVLink) communication operations for FlashInfer.

"""

import math
import os

import sys
import threading
from typing import List, Tuple

import torch
from mpi4py import MPI

from flashinfer.comm.mapping import Mapping

from ..utils import register_custom_op

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import atexit
import functools
from types import SimpleNamespace

from flashinfer.comm.mnnvl import McastGPUBuffer

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec, sm100a_nvcc_flags
from ..utils import register_custom_op

_thread_local = threading.local()


def mpi_barrier():
    """MPI barrier - could potentially be replaced with dist.barrier()"""
    MPI.COMM_WORLD.Barrier()


def gen_trtllm_mnnvl_comm_module() -> JitSpec:
    return gen_jit_spec(
        "trtllm_mnnvl_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_mnnvl_allreduce.cu",
        ],
    )


@functools.cache
def get_trtllm_mnnvl_comm_module():
    module = gen_trtllm_mnnvl_comm_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_mnnvl_all_reduce",
        mutates_args=[
            "inp",
            "out",
            "multicast_buffer_ptr",
            "buffer_ptrs_dev",
            "buffer_mnnvl",
            "buffer_flags_mnnvl",
            "nranks",
            "rank",
            "wait_for_results",
            "launch_with_pdl",
        ],
    )
    def trtllm_mnnvl_all_reduce(
        inp: torch.Tensor,
        out: torch.Tensor,
        multicast_buffer_ptr: int,  # Pointer address as integer
        buffer_ptrs_dev: int,  # Pointer address as integer
        buffer_mnnvl: torch.Tensor,
        buffer_flags_mnnvl: torch.Tensor,
        nranks: int,
        rank: int,
        wait_for_results: bool,
        launch_with_pdl: bool,
    ) -> None:
        module.trtllm_mnnvl_all_reduce(
            inp,
            out,
            multicast_buffer_ptr,
            buffer_ptrs_dev,
            buffer_mnnvl,
            buffer_flags_mnnvl,
            nranks,
            rank,
            wait_for_results,
            launch_with_pdl,
        )

    return SimpleNamespace(
        trtllm_mnnvl_all_reduce=trtllm_mnnvl_all_reduce,
    )


def get_allreduce_mnnvl_workspace(
    mapping: Mapping, dtype: torch.dtype
) -> Tuple[McastGPUBuffer, torch.Tensor, torch.Tensor, int]:
    """Get or create MNNVL all-reduce workspace buffers"""
    if not hasattr(_thread_local, f"allreduce_mnnvl_workspaces_{mapping.pp_rank}"):
        setattr(_thread_local, f"allreduce_mnnvl_workspaces_{mapping.pp_rank}", {})

    force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"

    allreduce_mnnvl_workspaces = getattr(
        _thread_local, f"allreduce_mnnvl_workspaces_{mapping.pp_rank}"
    )
    if mapping not in allreduce_mnnvl_workspaces:
        # buffer shape: [3, 2, buffer_tokens, hidden_dim]
        stride = 3 * 2 * dtype.itemsize
        # LCM for hidden_dim: 2048, 4096, 5120, 7168, 8192 = 286720
        # max_num_elements must be a multiple of 286720
        lcm_hidden_dim = 286720
        buffer_size_in_bytes = math.ceil(12_000_000 / (lcm_hidden_dim * stride)) * (
            lcm_hidden_dim * stride
        )
        max_num_elements = buffer_size_in_bytes // stride

        mcast_buffer = McastGPUBuffer(
            buffer_size_in_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            torch.device("cuda", mapping.local_rank),
            mapping.is_multi_node() or force_mn,
        )

        buffer = mcast_buffer.get_uc_buffer(
            mapping.tp_rank, (3, 2, max_num_elements), dtype, 0
        )
        # Only initialize the buffer when we need to resize it
        buffer.fill_(-0.0)
        # CPU barrier since we assume this should not be called in cuda graph
        torch.cuda.synchronize()
        mpi_barrier()

        # This is a buffer to maintain the state of this allreduce Op
        # Should have the same lifetime with self._buffer
        # [Buffer_ptr, Clear_ptr, Buffer_size, atomic access counter]
        buffer_flags = torch.tensor(
            [0, 2, max_num_elements, 0],
            dtype=torch.uint32,
            device=torch.device("cuda", mapping.local_rank),
        )

        allreduce_mnnvl_workspaces[mapping] = (
            mcast_buffer,
            buffer,
            buffer_flags,
            max_num_elements,
        )
    return allreduce_mnnvl_workspaces[mapping]


def trtllm_mnnvl_all_reduce(
    inp: torch.Tensor,
    out: torch.Tensor,
    multicast_buffer_ptr: int,  # Pointer address as integer
    buffer_ptrs_dev: int,  # Pointer address as integer
    buffer_mnnvl: torch.Tensor,
    buffer_flags_mnnvl: torch.Tensor,
    nranks: int,
    rank: int,
    wait_for_results: bool,
    launch_with_pdl: bool,
) -> None:
    """MNNVL all-reduce operation"""
    module = get_trtllm_mnnvl_comm_module()
    module.trtllm_mnnvl_all_reduce(
        inp,
        out,
        multicast_buffer_ptr,
        buffer_ptrs_dev,
        buffer_mnnvl,
        buffer_flags_mnnvl,
        nranks,
        rank,
        wait_for_results,
        launch_with_pdl,
    )


def cleanup_allreduce_mnnvl_workspaces():
    """Cleanup function to explicitly release MNNVL workspace resources"""
    if hasattr(_thread_local, "__dict__"):
        for attr_name in list(_thread_local.__dict__.keys()):
            if attr_name.startswith("allreduce_mnnvl_workspaces_"):
                workspaces = getattr(_thread_local, attr_name)
                for mapping, (
                    mcast_buffer,
                    buffer,
                    buffer_flags,
                    max_num_elements,
                ) in workspaces.items():
                    try:
                        # Explicitly delete the McastGPUBuffer to trigger cleanup
                        del mcast_buffer
                    except Exception as e:
                        print(f"Error cleaning up MNNVL workspace: {e}")
                workspaces.clear()
                delattr(_thread_local, attr_name)


# Register cleanup function to be called at exit
atexit.register(cleanup_allreduce_mnnvl_workspaces)
