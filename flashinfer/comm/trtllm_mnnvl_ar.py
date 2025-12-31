"""
MNNVL (Multi-Node NVLink) communication operations for FlashInfer.

"""

import functools
import math
import logging
from types import SimpleNamespace
from typing import Optional, Tuple
from enum import Enum

import torch
from typing_extensions import deprecated

from flashinfer.comm.mapping import Mapping

from ..jit import gen_trtllm_mnnvl_comm_module
from ..utils import register_custom_op
from .mnnvl import McastGPUBuffer, CommBackend, MPIBackend
from .workspace_base import AllReduceFusionWorkspace


def mpi_barrier():
    from mpi4py import MPI

    """MPI barrier - could potentially be replaced with dist.barrier()"""
    MPI.COMM_WORLD.Barrier()


class MNNVLAllreduceFusionStrategy(Enum):
    ONESHOT = 0
    TWOSHOT = 1
    AUTO = 99

    @staticmethod
    def select_strategy(
        tp_size: int, num_tokens: int, hidden_dim: int, dtype: torch.dtype
    ) -> "MNNVLAllreduceFusionStrategy":
        elem_size = torch.tensor([], dtype=dtype).element_size()
        if num_tokens * hidden_dim * tp_size * elem_size <= MNNVL_ONE_SHOT_THRESHOLD:
            return MNNVLAllreduceFusionStrategy.ONESHOT
        else:
            return MNNVLAllreduceFusionStrategy.TWOSHOT


# Empirical result calculated from num_tokens * hidden_dim * tp_size * elem_size
MNNVL_ONE_SHOT_THRESHOLD = 64 * 1024 * 8 * 2


class MNNVLAllReduceFusionWorkspace(AllReduceFusionWorkspace):
    NUM_LAMPORT_BUFFERS = 3

    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        buffer_size_in_bytes: Optional[int] = None,
        comm_backend: Optional[CommBackend] = None,
    ):
        """
        Initialize the MNNVL Allreduce Fusion Workspace. The workspace will be allocated and initialized based on the provided problem size. If max_num_tokens is larger than the one-shot threshold, the workspace will be created according to the max of required one-shot size at threshold, or the required two-shot size. Note that the workspace is not bind to the given problem size. It can be reused for different problem size without reinitialization given the allocated size is sufficient.

        If the buffer_size_in_bytes is provided, the workspace will be created according to the provided size. The user is expected to use the utility function get_required_buffer_size_bytes to calculate the required size. The actual allocation size may be larger due to alignment requirements. This covers the advanced used case, for example, the user may want to enforce oneshot strategy and ignore the heuristics.

        Either max_num_tokens or buffer_size_in_bytes must be provided.

        comm_backend will be used for creating the workspace and synchronization. If not provided, MPIBackend will be used which will use COMM_WORLD for synchronization.

        Args:
            mapping: Mapping configuration containing rank info
            max_num_tokens: The maximum number of tokens in the input tensor.
            hidden_dim: The hidden dimension of the tensors to be reduced.
            dtype: The data type of the tensors to be reduced.
            buffer_size_in_bytes: The requested size in bytes for each lamport buffer. The actual allocation size may be larger due to alignment requirements. The actual usable size will be NUM_LAMPORT_BUFFERS * actual_buffer_size_per_lamport_buffer.
        """
        super().__init__(mapping.world_size, mapping.rank)

        if buffer_size_in_bytes is None:
            assert (
                max_num_tokens is not None
                and hidden_dim is not None
                and dtype is not None
            ), (
                "max_num_tokens, hidden_dim, and dtype must be provided if buffer_size_in_bytes is not provided."
            )

            # If the user want to explictly use one-shot pass the threshold, which requires larger workspace size,
            # We expect the user to set workspace size manually.
            elem_size = torch.tensor([], dtype=dtype).element_size()
            oneshot_max_num_tokens = min(
                MNNVL_ONE_SHOT_THRESHOLD // (mapping.tp_size * elem_size * hidden_dim),
                max_num_tokens,
            )
            one_shot_size_bytes = self.get_required_buffer_size_bytes(
                mapping.tp_size,
                oneshot_max_num_tokens,
                hidden_dim,
                dtype,
                MNNVLAllreduceFusionStrategy.ONESHOT,
            )
            two_shot_size_bytes = self.get_required_buffer_size_bytes(
                mapping.tp_size,
                max_num_tokens,
                hidden_dim,
                dtype,
                MNNVLAllreduceFusionStrategy.TWOSHOT,
            )
            # We don't do roundup here as it will happen at the allocation.
            buffer_size_in_bytes = max(one_shot_size_bytes, two_shot_size_bytes)
        else:
            logging.debug(
                f"[MNNVL Allreduce] Using provided buffer size override in bytes: {buffer_size_in_bytes} bytes."
            )

        if comm_backend is None:
            comm_backend = MPIBackend()
        if buffer_size_in_bytes > (2**32 - 1):
            raise ValueError(
                f"The buffer size in bytes {buffer_size_in_bytes} is greater than the maximum supported size (UINT32_MAX)."
            )

        # Calculate total requested workspace size
        requested_workspace_size = buffer_size_in_bytes * self.NUM_LAMPORT_BUFFERS

        self.rank = mapping.tp_rank
        self.tp_size = mapping.tp_size
        logging.debug(
            f"[MNNVL Allreduce] TP size: {mapping.tp_size}, rank: {mapping.tp_rank}, Allocating workspace with requested size {buffer_size_in_bytes} bytes per buffer."
        )

        # Allocate the workspace
        self.mcast_buffer_handle = McastGPUBuffer(
            requested_workspace_size,
            mapping.tp_size,
            mapping.tp_rank,
            torch.device("cuda", mapping.local_rank),
            comm_backend,
        )

        # Get the actual usable buffer size after allocation (buf_size is updated by McastGPUBuffer)
        allocated_size = self.mcast_buffer_handle.buf_size
        # We want the buffer size to be aligned to 16B which is the granularity for buffer management.
        self.buffer_size_bytes = (
            math.floor(allocated_size / self.NUM_LAMPORT_BUFFERS) // 16 * 16
        )
        # This workspace size is used for checking the buffer. We need to set it to the actual size in use. The buffer free logic does not rely on this size.
        self.workspace_size_bytes = self.buffer_size_bytes * self.NUM_LAMPORT_BUFFERS

        logging.debug(
            f"[MNNVL Allreduce] Actual allocated size: {allocated_size} bytes, Actual buffer size per lamport buffer: {self.buffer_size_bytes} bytes, total workspace: {self.workspace_size_bytes} bytes."
        )

        # We use FP32 for sentinel value regardless of the real dtype
        self.mcast_buffer_handle.lamport_initialize(mapping.tp_rank, torch.float32)
        # Wait until the initialization is done
        torch.cuda.synchronize()
        comm_backend.barrier()

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

    @functools.cache
    def is_buffer_size_sufficient(
        self,
        tp_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
    ) -> bool:
        """
        Calculate the required buffer size for a given problem size.
        """
        required_buffer_size = self.get_required_buffer_size_bytes(
            tp_size, num_tokens, hidden_dim, dtype, strategy
        )
        if required_buffer_size > self.buffer_size_bytes:
            return False
        else:
            return True

    @staticmethod
    @functools.cache
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
        if strategy == MNNVLAllreduceFusionStrategy.AUTO:
            strategy = MNNVLAllreduceFusionStrategy.select_strategy(
                tp_size, num_tokens, hidden_dim, dtype
            )

        if strategy == MNNVLAllreduceFusionStrategy.ONESHOT:
            # For one-shot, each rank needs to store num_tokens * tp_size tokens
            buffer_size = num_tokens * hidden_dim * tp_size * elem_size
        else:
            # For two-shot, each rank stores a slices of tokens. We need to round up to the nearest tp_size.
            # 2 Stage is required for the two-shot allreduce.
            buffer_size = (
                2 * math.ceil(num_tokens / tp_size) * tp_size * hidden_dim * elem_size
            )
        return buffer_size

    @property
    def backend(self) -> str:
        return "mnnvl"

    def destroy(self) -> None:
        """Destroy workspace and free resources."""
        if getattr(self, "_destroyed", False):
            return  # Already destroyed, nothing to do

        del self.mcast_buffer_handle
        del self.buffer_flags
        del self.uc_ptrs_dev
        del self.uc_ptr_local
        del self.mc_ptr
        self._destroyed = True


@functools.cache
def get_trtllm_mnnvl_comm_module():
    module = gen_trtllm_mnnvl_comm_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_mnnvl_allreduce_fusion",
        mutates_args=[
            "input",
            "multicast_buffer_ptr",
            "buffer_ptrs_dev",
            "buffer_ptr_local",
            "buffer_flags_mnnvl",
            "nranks",
            "rank",
            "rmsnorm_fusion",
            "launch_with_pdl",
            "use_oneshot",
            "output",
            "residual_out",
            "residual_in",
            "gamma",
            "epsilon",
        ],
    )
    def trtllm_mnnvl_allreduce_fusion(
        input: torch.Tensor,
        multicast_buffer_ptr: int,  # Pointer address as integer
        buffer_ptrs_dev: int,  # Pointer address as integer
        buffer_ptr_local: int,  # Pointer address as integer
        buffer_flags_mnnvl: torch.Tensor,
        nranks: int,
        rank: int,
        rmsnorm_fusion: bool,
        launch_with_pdl: bool,
        use_oneshot: bool,
        output: torch.Tensor,
        residual_out: Optional[torch.Tensor],
        residual_in: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor],
        epsilon: Optional[float],
    ) -> None:
        """
        Perform a multi-node NVLink all-reduce operation with fusion.
        Args:
            input: Input tensor
            multicast_buffer_ptr: Pointer to the multicast buffer as an integer
            buffer_ptrs_dev: Pointer to the device array of buffer pointers as an integer
            buffer_ptr_local: Pointer to local buffer as an integer
            buffer_flags_mnnvl: Buffer flags tensor for synchronization
            nranks: Total number of ranks participating in the all-reduce
            rank: Current process rank
            rmsnorm_fusion: Whether to perform RMSNorm fusion
            launch_with_pdl: Whether to launch with PDL
            use_oneshot: Whether to use one-shot (true) or two-shot (false)
            output: Output tensor
            residual_out: Residual output tensor (if rmsnorm)
            gamma: Gamma tensor (if rmsnorm)
            epsilon: Epsilon value (if rmsnorm)
        """
        module.trtllm_mnnvl_allreduce_fusion(
            input,
            multicast_buffer_ptr,
            buffer_ptrs_dev,
            buffer_ptr_local,
            buffer_flags_mnnvl,
            nranks,
            rank,
            rmsnorm_fusion,
            launch_with_pdl,
            use_oneshot,
            output,
            residual_out,
            residual_in,
            gamma,
            epsilon,
        )

    return SimpleNamespace(
        trtllm_mnnvl_allreduce_fusion=trtllm_mnnvl_allreduce_fusion,
    )


def trtllm_mnnvl_allreduce(
    input: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    launch_with_pdl: bool,
    output: Optional[torch.Tensor] = None,
    strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
) -> torch.Tensor:
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
        input: Local Input Shard [num_tokens, hidden_dim]
        workspace: MNNVLAllReduceFusionWorkspace
        launch_with_pdl: Whether to launch with PDL
        output: Output tensor to store the result, empty tensor will be created if not provided.
        strategy: MNNVLAllreduceFusionStrategy. Internal heuristics will be used if not provided.
    Returns:
        output: Reduced tensor [num_tokens, hidden_dim]
    """

    # Check ndims here as the shape check is done in the kernel launch code.
    if len(input.shape) != 2:
        raise ValueError(
            f"The input tensor must be 2D, got {len(input.shape)}D. The shape is {input.shape}."
        )

    if output is None:
        output = torch.empty_like(input)
    elif len(output.shape) != 2:
        raise ValueError(
            f"The output tensor must be 2D, got {len(output.shape)}D. The shape is {output.shape}."
        )

    module = get_trtllm_mnnvl_comm_module()

    if strategy == MNNVLAllreduceFusionStrategy.AUTO:
        strategy = MNNVLAllreduceFusionStrategy.select_strategy(
            workspace.tp_size, input.shape[0], input.shape[1], input.dtype
        )

    if not workspace.is_buffer_size_sufficient(
        workspace.tp_size, input.shape[0], input.shape[1], input.dtype, strategy
    ):
        raise ValueError(
            f"The buffer size in the given workspace is insufficient for the given problem size. Buffer: {workspace.buffer_size_bytes} bytes, Required: {workspace.get_required_buffer_size_bytes(workspace.tp_size, input.shape[0], input.shape[1], input.dtype, strategy)} bytes."
        )

    module.trtllm_mnnvl_allreduce_fusion(
        input,
        workspace.mc_ptr,
        workspace.uc_ptrs_dev,
        workspace.uc_ptr_local,
        workspace.buffer_flags,
        workspace.tp_size,
        workspace.rank,
        False,  # No RMSNorm Fusion
        launch_with_pdl,
        strategy == MNNVLAllreduceFusionStrategy.ONESHOT,
        output,
        None,
        None,
        None,
        None,
    )

    return output


def trtllm_mnnvl_fused_allreduce_add_rmsnorm(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    epsilon: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = False,
    strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs MNNVL Allreduce + Residual + RMSNorm.

    This function performs a multi-node all-reduce (sum) operation by first calling trtllm_mnnvl_allreduce on the shard_input.
    After this, it performs residual addition and RMSNorm on the all-reduced result, reading it directly from the multicast buffer.
    Note: multicast buffer is the same as the unicast buffer for the current rank.

    Args:
        input: Input tensor [num_tokens, hidden_dim]
        residual_in: Residual input tensor [num_tokens, hidden_dim]
        gamma: Gamma tensor [hidden_dim]
        workspace: MNNVLAllReduceFusionWorkspace
        epsilon: The epsilon parameter for RMSNorm, torch.finfo.eps will be used if not provided.
        output: Output tensor for normalized results [num_tokens, hidden_dim], empty tensor will be created if not provided.
        residual_out: Residual output tensor [num_tokens, hidden_dim], empty tensor will be created if not provided.
        launch_with_pdl: Whether to launch with PDL
        strategy: MNNVLAllreduceFusionStrategy. Internal heuristics will be used if not provided.

    Returns:
        output: Add-residual and normalized tensor [num_tokens, hidden_dim]
        residual_out: Add-residual tensor [num_tokens, hidden_dim]
    """

    if epsilon is None:
        epsilon = torch.finfo(input.dtype).eps

    if len(input.shape) != 2:
        raise ValueError(
            f"The input tensor must be 2D, got {len(input.shape)}D. The shape is {input.shape}."
        )
    if len(residual_in.shape) != 2:
        raise ValueError(
            f"The residual input tensor must be 2D, got {len(residual_in.shape)}D. The shape is {residual_in.shape}."
        )
    if gamma.numel() != input.shape[1]:
        raise ValueError(
            f"The gamma tensor must have the same number of elements as the hidden dimension, got {gamma.numel()} elements but expected {input.shape[1]} elements."
        )
    if output is None:
        output = torch.empty_like(input)
    elif len(output.shape) != 2:
        raise ValueError(
            f"The output tensor must be 2D, got {len(output.shape)}D. The shape is {output.shape}."
        )
    if residual_out is None:
        residual_out = torch.empty_like(residual_in)
    elif len(residual_out.shape) != 2:
        raise ValueError(
            f"The residual output tensor must be 2D, got {len(residual_out.shape)}D. The shape is {residual_out.shape}."
        )

    module = get_trtllm_mnnvl_comm_module()

    if strategy == MNNVLAllreduceFusionStrategy.AUTO:
        strategy = MNNVLAllreduceFusionStrategy.select_strategy(
            workspace.tp_size, input.shape[0], input.shape[1], input.dtype
        )
    if not workspace.is_buffer_size_sufficient(
        workspace.tp_size, input.shape[0], input.shape[1], input.dtype, strategy
    ):
        raise ValueError(
            f"The buffer size in the given workspace is insufficient for the given problem size. Buffer: {workspace.buffer_size_bytes} bytes, Required: {workspace.get_required_buffer_size_bytes(workspace.tp_size, input.shape[0], input.shape[1], input.dtype, strategy)} bytes."
        )

    module.trtllm_mnnvl_allreduce_fusion(
        input,
        workspace.mc_ptr,
        workspace.uc_ptrs_dev,
        workspace.uc_ptr_local,
        workspace.buffer_flags,
        workspace.tp_size,
        workspace.rank,
        True,  # RMSNorm Fusion
        launch_with_pdl,
        strategy == MNNVLAllreduceFusionStrategy.ONESHOT,
        output,
        residual_out,
        residual_in,
        gamma,
        epsilon,
    )
    return output, residual_out


# Legacy API that has been deprecated; Left for backward compatibility
@deprecated(
    "get_allreduce_mnnvl_workspace is deprecated, use MNNVLAllReduceFusionWorkspace class to manage the workspace instead"
)
def get_allreduce_mnnvl_workspace(
    mapping: Mapping,
    dtype: torch.dtype,
    comm_backend_for_handle_transfer: Optional[CommBackend] = None,
    buffer_size_in_bytes: Optional[int] = None,
) -> Tuple[McastGPUBuffer, torch.Tensor, int]:
    """Get workspace buffers needed for multi-node NVLink all-reduce operation.

    This function allocates and initializes the workspace buffers required for performing
    multi-node NVLink all-reduce operations. It creates:
    1. A multicast GPU buffer for communication between nodes
    2. A flags tensor to track buffer state
    3. Maximum number of elements that can fit in the buffer

    The buffer size is calculated to efficiently handle common hidden dimensions
    (2048, 4096, 5120, 7168, 8192) by using their LCM of 286720.

    Args:
        mapping: Tensor parallel mapping configuration containing rank info
        dtype: Data type of the tensors being reduced
        buffer_size_in_bytes: Optional buffer size. Practically, assign this to 3 * 2 * dtype.itemsize * hidden_dim * max_tokens

    Returns:
        Tuple containing:
        - McastGPUBuffer: Multicast buffer for inter-node communication
        - torch.Tensor: Buffer flags tensor tracking state
        - int: Maximum number of elements that can fit in buffer
    """
    # buffer shape: [3, 2, buffer_tokens, hidden_dim]
    stride = 3 * 2 * dtype.itemsize
    # LCM for hidden_dim: 2048, 4096, 5120, 7168, 8192 = 286720
    # max_num_elements must be a multiple of 286720
    lcm_hidden_dim = 286720
    TARGET_WORKSPACE_SIZE_BYTES = (
        buffer_size_in_bytes if buffer_size_in_bytes is not None else 12_000_000
    )
    buffer_size_in_bytes = math.ceil(
        TARGET_WORKSPACE_SIZE_BYTES / (lcm_hidden_dim * stride)
    ) * (lcm_hidden_dim * stride)

    # Redirect to the new workspace allocation logic. The new kernel needs the new flag buffer layout.
    workspace = MNNVLAllReduceFusionWorkspace(
        mapping,
        buffer_size_in_bytes=buffer_size_in_bytes,
        comm_backend=comm_backend_for_handle_transfer,
    )

    mcast_buffer = workspace.mcast_buffer_handle
    buffer_flags = workspace.buffer_flags
    # this is calculated using the legacy behavior. We do not use the actual allocated size.
    max_num_elements = workspace.buffer_size_bytes // stride

    return (
        mcast_buffer,
        buffer_flags,
        max_num_elements,
    )


@deprecated(
    "trtllm_mnnvl_all_reduce is deprecated, use trtllm_mnnvl_allreduce instead. This function will be removed in the future."
)
def trtllm_mnnvl_all_reduce(
    inp: torch.Tensor,
    multicast_buffer_ptr: int,  # Pointer address as integer
    buffer_ptrs_dev: int,  # Pointer address as integer
    buffer_M: int,
    buffer_flags_mnnvl: torch.Tensor,
    nranks: int,
    rank: int,
    wait_for_results: bool,
    launch_with_pdl: bool,
    out: Optional[torch.Tensor] = None,
) -> None:
    """Perform a multi-node NVLink all-reduce operation across multiple GPUs.

    This function performs an all-reduce (sum) operation using NVIDIA's multi-node NVLink (MNNVL)
    technology to efficiently combine tensors across multiple GPUs and nodes.

    There are 3 steps:
    1. scatter each GPU's input shard to the right unicast buffer
    2. perform all-reduce on each GPU
    3. broadcast the result to all GPUs

    Args:
        inp: Local Input Shard
        multicast_buffer_ptr: Pointer to the multicast buffer as an integer
        buffer_ptrs_dev: Pointer to device buffer pointers as an integer
        buffer_M: Maximum number of elements // hidden_dim
        buffer_flags_mnnvl: Tensor containing buffer state flags
        nranks: Total number of ranks participating in the all-reduce
        rank: Current process rank
        wait_for_results: If True, store the result to out
        launch_with_pdl: If True, launch using Programmatic Dependent Launch
        [Optional] out: Output tensor to store the result (required if wait_for_results is True)

    """

    if len(inp.shape) != 2:
        raise ValueError(
            f"The input tensor must be 2D, got {len(inp.shape)}D. The shape is {inp.shape}."
        )

    # buffer_M is no longer used in this kernel but let's keep this check for consistency in behavior.
    if inp.shape[0] > buffer_M:
        raise ValueError(
            f"The number of tokens in the input tensor {inp.shape[0]} is greater than the buffer_M {buffer_M}. This is not supported. Please increase the workspace size, or decrease the amount of tokens to at most {buffer_M}."
        )

    # Even in legacy code, this should only be used when we implement the fused allreduce+rmsnorm.
    assert wait_for_results and (out is not None), (
        "Calling the legacy trtllm_mnnvl_all_reduce with wait_for_results=False is not supported. Please use trtllm_mnnvl_allreduce instead."
    )
    module = get_trtllm_mnnvl_comm_module()
    module.trtllm_mnnvl_allreduce_fusion(
        inp,
        multicast_buffer_ptr,
        buffer_ptrs_dev,
        0,  # Allreduce kernel itself does not use this local pointer; still this could be risky but it is only used for legacy code compatibility.
        buffer_flags_mnnvl,
        nranks,
        rank,
        False,  # No RMSNorm Fusion
        launch_with_pdl,
        False,  # Use two-shot
        out,
        None,
        None,
        None,
        None,
    )


@deprecated(
    "trtllm_mnnvl_fused_allreduce_rmsnorm is deprecated, use trtllm_mnnvl_fused_allreduce_add_rmsnorm instead. This function will be removed in the future."
)
def trtllm_mnnvl_fused_allreduce_rmsnorm(
    prenorm_output: torch.Tensor,
    normed_output: torch.Tensor,
    shard_input: torch.Tensor,
    multicast_buffer_ptr: int,  # Pointer address as integer
    buffer_ptrs_dev: int,  # Pointer address as integer
    unicast_ptr: int,  # Local unicast buffer pointer
    buffer_M: int,
    buffer_flags_mnnvl: torch.Tensor,
    nranks: int,
    rank: int,
    gamma: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
    launch_with_pdl: bool,
) -> None:
    """Performs MNNVL TwoShot Allreduce + RMSNorm.

    This function performs a multi-node all-reduce (sum) operation by first calling trtllm_mnnvl_all_reduce on the shard_input.
    After this, it performs RMSNorm on the all-reduced result, reading it directly from the multicast buffer.
    Note: multicast buffer is the same as the unicast buffer for the current rank.

    Args:
        prenorm_output: Output tensor for prenorm results
        normed_output: Output tensor for normalized results
        shard_input: Input tensor shard
        multicast_buffer_ptr: Pointer address as integer for multicast buffer
        buffer_ptrs_dev: Pointer address as integer for device buffer pointers
        unicast_ptr: Pointer address as integer for unicast buffer
        buffer_M: Maximum number of elements // hidden_dim
        buffer_flags_mnnvl: Buffer flags for synchronization
        nranks: Number of ranks in the tensor parallel group
        rank: Current rank in the tensor parallel group
        gamma: The gamma (norm weight) parameter for RMSNorm
        epsilon: The epsilon parameter for RMSNorm
        residual: The residual tensor to add
        launch_with_pdl: Whether to launch with PDL

    """
    if len(shard_input.shape) != 2:
        raise ValueError(
            f"The input tensor must be 2D, got {len(shard_input.shape)}D. The shape is {shard_input.shape}."
        )

    # buffer_M is no longer used in this kernel but let's keep this check for consistency in behavior.
    if shard_input.shape[0] > buffer_M:
        raise ValueError(
            f"The number of tokens in the input tensor {shard_input.shape[0]} is greater than the buffer_M {buffer_M}. This is not supported. Please increase the workspace size, or decrease the amount of tokens to at most {buffer_M}."
        )

    if len(residual.shape) != 2:
        raise ValueError(
            f"The residual input tensor must be 2D, got {len(residual.shape)}D. The shape is {residual.shape}."
        )
    if gamma.numel() != shard_input.shape[1]:
        raise ValueError(
            f"The gamma tensor must have the same number of elements as the hidden dimension, got {gamma.numel()} elements but expected {shard_input.shape[1]} elements."
        )

    if len(normed_output.shape) != 2:
        raise ValueError(
            f"The output tensor must be 2D, got {len(normed_output.shape)}D. The shape is {normed_output.shape}."
        )

    if len(prenorm_output.shape) != 2:
        raise ValueError(
            f"The prenorm output tensor must be 2D, got {len(prenorm_output.shape)}D. The shape is {prenorm_output.shape}."
        )

    module = get_trtllm_mnnvl_comm_module()

    module.trtllm_mnnvl_allreduce_fusion(
        shard_input,
        multicast_buffer_ptr,
        buffer_ptrs_dev,
        unicast_ptr,
        buffer_flags_mnnvl,
        nranks,
        rank,
        True,  # RMSNorm Fusion
        launch_with_pdl,
        False,
        normed_output,
        prenorm_output,
        residual,
        gamma,
        epsilon,
    )
