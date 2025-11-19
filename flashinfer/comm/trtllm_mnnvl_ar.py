"""
MNNVL (Multi-Node NVLink) communication operations for FlashInfer.

"""

import functools
import math
import logging
from types import SimpleNamespace
from typing import Optional
from enum import Enum

import torch

from flashinfer.comm.mapping import Mapping

from ..jit import gen_trtllm_mnnvl_comm_module
from ..utils import register_custom_op
from .mnnvl import McastGPUBuffer, CommBackend


def mpi_barrier():
    from mpi4py import MPI

    """MPI barrier - could potentially be replaced with dist.barrier()"""
    MPI.COMM_WORLD.Barrier()


class MNNVLAllreduceFusionStrategy(Enum):
    ONESHOT = 0
    TWOSHOT = 1
    AUTO = 99

    @staticmethod
    def is_one_shot(tp_size: int, num_tokens: int, hidden_dim: int, dtype: torch.dtype) -> bool:
        elem_size = torch.tensor([], dtype=dtype).element_size()
        return num_tokens * hidden_dim * tp_size * elem_size <= kMNNVLOneShotThreshold


# Empirical result calculated from num_tokens * hidden_dim * tp_size * elem_size
kMNNVLOneShotThreshold = 64 * 1024 * 8 * 2


class MNNVLAllreduceFusionWorkspace:
    NUM_LAMPORT_BUFFERS = 3

    def __init__(self, mapping: Mapping, buffer_size_in_bytes: Optional[int] = None):
        """
        Initialize the MNNVL Allreduce Fusion Workspace. COMM_WORLD will be used for creating the workspace and synchronization. The process might hang if the intended communication group in mapping is not COMM_WORLD.

        Args:
            mapping: Mapping configuration containing rank info
            buffer_size_in_bytes: The size in bytes for each lamport buffer. The actual allocation size will be NUM_LAMPORT_BUFFERS * buffer_size_in_bytes.
        """
        if buffer_size_in_bytes is None:
            # Default to 16MB workspace size if not provided
            buffer_size_in_bytes = 16 * (1024**2)
        else:
            # Round up to the nearest multiple of 8MB
            buffer_size_in_bytes = math.ceil(buffer_size_in_bytes / (8 * (1024**2)) * (8 * (1024**2)))

        if buffer_size_in_bytes > (2**32 - 1):
            raise ValueError(
                f"The buffer size in bytes {buffer_size_in_bytes} is greater than the maximum supported size (UINT32_MAX)."
            )

        self.buffer_size_bytes = buffer_size_in_bytes
        self.workspace_size_bytes = buffer_size_in_bytes * self.NUM_LAMPORT_BUFFERS
        self.rank = mapping.tp_rank
        self.tp_size = mapping.tp_size
        logging.debug(
            f"[MNNVL Allreduce] TP size: {mapping.tp_size}, rank: {mapping.tp_rank}, Allocating workspace with size {buffer_size_in_bytes} bytes."
        )
        self.mcast_buffer_handle = McastGPUBuffer(
            self.workspace_size_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            torch.device("cuda", mapping.local_rank),
            mapping.is_multi_node(),
        )

        # We use FP32 for sentinel value regardless of the real dtype
        self.mcast_buffer_handle.lamport_initialize(mapping.tp_rank, torch.float32)
        # Wait until the initialization is done
        torch.cuda.synchronize()
        # FIXME: We are assuming using the COMM_WORLD.
        mpi_barrier()

        # This is a buffer to maintain the state of this allreduce Op
        # Should have the same lifetime with self._buffer
        # The flag should be binded to each buffer allocation
        # Layout: [cur idx, dirty idx, bytes per buffer, dirty num stages, numBytesToClear[4], access count ptr]
        num_bytes_to_clear = [0] * 4
        self.buffer_flags = torch.tensor(
            [0, 2, self.buffer_size_bytes, 0, *num_bytes_to_clear, 0],
            dtype=torch.uint32,
            device=torch.device("cuda", mapping.local_rank),
        )

        self.uc_ptrs_dev = self.mcast_buffer_handle.get_buffer_ptrs_dev()
        self.uc_ptr_local = self.mcast_buffer_handle.get_unicast_ptr(self.rank)
        self.mc_ptr = self.mcast_buffer_handle.get_multicast_ptr()

    @staticmethod
    def get_required_buffer_size_bytes(
        tp_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
    ) -> int:
        """
        Calculate the required buffer size for a given problem size.
        """
        elem_size = torch.tensor([], dtype=dtype).element_size()
        is_one_shot = MNNVLAllreduceFusionStrategy.is_one_shot(tp_size, num_tokens, hidden_dim, dtype)
        if strategy == MNNVLAllreduceFusionStrategy.ONESHOT or (
            strategy == MNNVLAllreduceFusionStrategy.AUTO and is_one_shot
        ):
            # For one-shot, each rank needs to store num_tokens * tp_size tokens
            buffer_size = num_tokens * hidden_dim * tp_size * elem_size
        else:
            # For two-shot, each rank stores a slices of tokens. We need to round up to the nearest tp_size.
            # 2 Stage is required for the two-shot allreduce.
            buffer_size = 2 * math.ceil(num_tokens / tp_size) * tp_size * hidden_dim * elem_size
        return buffer_size


@functools.cache
def get_trtllm_mnnvl_comm_module():
    module = gen_trtllm_mnnvl_comm_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_mnnvl_allreduce_fusion",
        mutates_args=[
            "inp",
            "multicast_buffer_ptr",
            "buffer_ptrs_dev",
            "buffer_ptr_local",
            "buffer_flags_mnnvl",
            "nranks",
            "rank",
            "rmsnorm_fusion",
            "launch_with_pdl",
            "use_oneshot",
            "out",
            "residual_out",
            "residual_in",
            "gamma",
            "epsilon",
        ],
    )
    def trtllm_mnnvl_allreduce_fusion(
        inp: torch.Tensor,
        multicast_buffer_ptr: int,  # Pointer address as integer
        buffer_ptrs_dev: int,  # Pointer address as integer
        buffer_ptr_local: int,  # Pointer address as integer
        buffer_flags_mnnvl: torch.Tensor,
        nranks: int,
        rank: int,
        rmsnorm_fusion: bool,
        launch_with_pdl: bool,
        use_oneshot: bool,
        out: Optional[torch.Tensor],
        residual_out: Optional[torch.Tensor],
        residual_in: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor],
        epsilon: Optional[float],
    ) -> None:
        """
        Perform a multi-node NVLink all-reduce operation with fusion.
        Args:
            inp: Input tensor
            multicast_buffer_ptr: Pointer to the multicast buffer as an integer
            buffer_ptrs_dev: Pointer to the device array of buffer pointers as an integer
            buffer_ptr_local: Pointer to local buffer as an integer
            buffer_flags_mnnvl: Buffer flags tensor for synchronization
            nranks: Total number of ranks participating in the all-reduce
            rank: Current process rank
            rmsnorm_fusion: Whether to perform RMSNorm fusion
            launch_with_pdl: Whether to launch with PDL
            use_oneshot: Whether to use one-shot (true) or two-shot (false)
            outp: Output tensor
            residual_out: Residual output tensor (if rmsnorm)
            gamma: Gamma tensor (if rmsnorm)
            epsilon: Epsilon value (if rmsnorm)
        """
        module.trtllm_mnnvl_allreduce_fusion(
            inp,
            multicast_buffer_ptr,
            buffer_ptrs_dev,
            buffer_ptr_local,
            buffer_flags_mnnvl,
            nranks,
            rank,
            rmsnorm_fusion,
            launch_with_pdl,
            use_oneshot,
            out,
            residual_out,
            residual_in,
            gamma,
            epsilon,
        )

    return SimpleNamespace(
        trtllm_mnnvl_allreduce_fusion=trtllm_mnnvl_allreduce_fusion,
    )


def trtllm_mnnvl_all_reduce(
    inp: torch.Tensor,
    workspace: MNNVLAllreduceFusionWorkspace,
    launch_with_pdl: bool,
    out: Optional[torch.Tensor] = None,
    strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
) -> None:
    """Perform a multi-node NVLink all-reduce operation across multiple GPUs.

    This function performs an all-reduce (sum) operation using NVIDIA's multi-node NVLink (MNNVL)
    technology to efficiently combine tensors across multiple GPUs and nodes.

    There are 2 variants: One-shot and Two-shot:
     - One-shot: Each rank stores local shard to all other ranks. Each ranks will receive all shards at the end of the communication round and perfom local reduction. Suitable for small data size and is optimized for low latency.
     - Two-shot: There will be 3 steps:
        1. Scatter each GPU's input shard to other ranks. Each rank will received all shards of a slice of tokens.
        2. Each rank perform reduction on the local tokens.
        3. Each rank broadcast the result to all ranks.
        Suitable for large data size and is optimized for balancing throughput and latency.

    Args:
        inp: Local Input Shard [num_tokens, hidden_dim]
        workspace: MNNVLAllreduceFusionWorkspace
        launch_with_pdl: Whether to launch with PDL
        out: Output tensor to store the result
        strategy: MNNVLAllreduceFusionStrategy. Internal heuristics will be used if not provided.
    """

    if len(inp.shape) != 2:
        raise ValueError(f"The input tensor must be 2D, got {len(inp.shape)}D. The shape is {inp.shape}.")

    module = get_trtllm_mnnvl_comm_module()

    use_oneshot = strategy == MNNVLAllreduceFusionStrategy.ONESHOT or (
        strategy == MNNVLAllreduceFusionStrategy.AUTO
        and MNNVLAllreduceFusionStrategy.is_one_shot(workspace.tp_size, inp.shape[0], inp.shape[1], inp.dtype)
    )
    module.trtllm_mnnvl_allreduce_fusion(
        inp,
        workspace.mc_ptr,
        workspace.uc_ptrs_dev,
        workspace.uc_ptr_local,
        workspace.buffer_flags,
        workspace.tp_size,
        workspace.rank,
        False,  # No RMSNorm Fusion
        launch_with_pdl,
        use_oneshot,
        out,
        None,
        None,
        None,
        None,
    )


def trtllm_mnnvl_fused_allreduce_rmsnorm(
    prenorm_output: torch.Tensor,
    normed_output: torch.Tensor,
    shard_input: torch.Tensor,
    workspace: MNNVLAllreduceFusionWorkspace,
    gamma: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
    launch_with_pdl: bool,
    strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
) -> None:
    """Performs MNNVL Allreduce + RMSNorm.

    This function performs a multi-node all-reduce (sum) operation by first calling trtllm_mnnvl_all_reduce on the shard_input.
    After this, it performs RMSNorm on the all-reduced result, reading it directly from the multicast buffer.
    Note: multicast buffer is the same as the unicast buffer for the current rank.

    Args:
        prenorm_output: Output tensor for prenorm results
        normed_output: Output tensor for normalized results
        shard_input: Input tensor shard
        workspace: MNNVLAllreduceFusionWorkspace
        gamma: The gamma (norm weight) parameter for RMSNorm
        epsilon: The epsilon parameter for RMSNorm
        residual: The residual tensor to add
        launch_with_pdl: Whether to launch with PDL

    """
    if len(shard_input.shape) != 2:
        raise ValueError(
            f"The input tensor must be 2D, got {len(shard_input.shape)}D. The shape is {shard_input.shape}."
        )

    module = get_trtllm_mnnvl_comm_module()

    use_oneshot = strategy == MNNVLAllreduceFusionStrategy.ONESHOT or (
        strategy == MNNVLAllreduceFusionStrategy.AUTO
        and MNNVLAllreduceFusionStrategy.is_one_shot(
            workspace.tp_size,
            shard_input.shape[0],
            shard_input.shape[1],
            shard_input.dtype,
        )
    )

    module.trtllm_mnnvl_allreduce_fusion(
        shard_input,
        workspace.mc_ptr,
        workspace.uc_ptrs_dev,
        workspace.uc_ptr_local,
        workspace.buffer_flags,
        workspace.tp_size,
        workspace.rank,
        True,  # RMSNorm Fusion
        launch_with_pdl,
        use_oneshot,
        normed_output,
        prenorm_output,
        residual,
        gamma,
        epsilon,
    )
