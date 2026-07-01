"""
MNNVL (Multi-Node NVLink) communication operations for FlashInfer.

"""

import functools
import math
import logging
from types import SimpleNamespace
from typing import Optional, Tuple, Union
from enum import Enum

import torch
from typing_extensions import deprecated

from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import TorchDistBackend

from ..jit import gen_trtllm_mnnvl_comm_module
from ..utils import register_custom_op
from ..fp4_quantization import _compute_swizzled_layout_sf_size
from .mnnvl import CommBackend, MPIBackend
from .trtllm_ar import QuantizationSFLayout
from .workspace_base import AllReduceFusionWorkspace
from .torch_symmetric_memory import _alloc_symm_buffer_bytes


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


class MNNVLQuantType:
    NONE = 0
    FP8 = 1
    NVFP4 = 2
    DYNAMIC_FP8 = 3


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
        # Use torch.cuda.current_device() instead of mapping.local_rank to
        # support base_gpu_id != 0 scenarios where the actual CUDA device
        # index differs from the TP rank / local_rank.
        device = torch.device("cuda", torch.cuda.current_device())
        if isinstance(comm_backend, TorchDistBackend):
            group = (
                comm_backend._group
                if comm_backend._group is not None
                else torch.distributed.group.WORLD
            )
            group_name = group.group_name
        else:
            group_name = torch.distributed.group.WORLD.group_name
        self.ptrs, self.tensor, self.handle = _alloc_symm_buffer_bytes(
            requested_workspace_size,
            mapping.tp_size,
            torch.float32,
            device,
            group_name,
        )

        # handle.buffer_size is the usable data size. torch symmetric memory
        # allocator places signal_pad on top of it, not carved from within.
        allocated_size = self.handle.buffer_size
        # We want the buffer size to be aligned to 16B which is the granularity for buffer management.
        self.buffer_size_bytes = (
            math.floor(allocated_size / self.NUM_LAMPORT_BUFFERS) // 16 * 16
        )
        # This workspace size is used for checking the buffer. We need to set it to the actual size in use. The buffer free logic does not rely on this size.
        self.workspace_size_bytes = self.buffer_size_bytes * self.NUM_LAMPORT_BUFFERS

        logging.debug(
            f"[MNNVL Allreduce] Actual allocated size: {allocated_size} bytes, Actual buffer size per lamport buffer: {self.buffer_size_bytes} bytes, total workspace: {self.workspace_size_bytes} bytes."
        )

        # lamport initialize tensor to negative zero.
        self.tensor.fill_(-0.0)
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
            device=torch.device("cuda", torch.cuda.current_device()),
        )

        self.uc_ptrs_dev = self.handle.buffer_ptrs_dev
        self.uc_ptr_local = self.handle.buffer_ptrs[self.rank]
        self.mc_ptr = self.handle.multicast_ptr

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

        del self.buffer_flags
        del self.uc_ptrs_dev
        del self.uc_ptr_local
        del self.mc_ptr
        del self.tensor
        del self.handle
        del self.ptrs
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
            "quant_type",
            "quant_out",
            "sf_out",
            "output_scale",
            "layout_code",
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
        output: Optional[torch.Tensor],
        residual_out: Optional[torch.Tensor],
        residual_in: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor],
        epsilon: Optional[float],
        weight_bias: Optional[float] = None,
        quant_type: int = MNNVLQuantType.NONE,
        quant_out: Optional[torch.Tensor] = None,
        sf_out: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        layout_code: int = QuantizationSFLayout.SWIZZLED_128x4,
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
            weight_bias,
            quant_type,
            quant_out,
            sf_out,
            output_scale,
            layout_code,
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
    """Perform an MNNVL all-reduce sum across tensor-parallel ranks.

    ``input`` must be a 2-D local shard with shape ``[num_tokens, hidden_dim]``.
    The result has the same shape and is written to ``output`` when provided, or
    to a newly allocated tensor otherwise. ``workspace`` must be an
    :class:`MNNVLAllReduceFusionWorkspace` created for the same tensor-parallel
    group and with enough buffer capacity for the selected strategy.

    MNNVL supports two execution strategies:

    * ``ONESHOT`` stores each rank's local shard to the peer-visible workspace
      and each rank performs the reduction locally. This is the low-latency path
      used for smaller payloads.
    * ``TWOSHOT`` scatters token slices, reduces each slice on its destination
      rank, then broadcasts the reduced result. This is the throughput-oriented
      path for larger payloads.

    With ``AUTO``, FlashInfer selects the strategy from the payload size. Both
    ``ONESHOT`` and ``TWOSHOT`` are fully deterministic across ranks.

    This allreduce-only helper does not perform quantization. Use
    :func:`trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant`, or the unified
    :func:`flashinfer.comm.allreduce_fusion` API with FP8/NVFP4
    ``AllReduceFusionPattern`` values, when the post-RMSNorm quantized output is
    needed.

    Determinism:

    * ``ONESHOT`` and ``TWOSHOT`` use the exact same reduction order on each
      rank.
    * ``ONESHOT`` keeps the local rank's value in registers; only remote ranks
      are volatile-loaded from the Lamport buffer.
    * ``ONESHOT`` uses a rank-specialized fast path for ``world_size <= 8``.
      Larger world sizes use a compact deterministic fallback because the
      runtime benefit is thin and specializing every rank significantly
      increases JIT compile time.

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
    weight_bias: float = 0.0,
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
        weight_bias: Bias added to gamma before scaling. 0.0 (default) for standard
            RMSNorm (gamma * x * rsqrt(...)); 1.0 for Gemma / Qwen3.5 RMSNorm
            ((1 + gamma) * x * rsqrt(...)).

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
        weight_bias,
    )
    return output, residual_out


def trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant(
    input: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    epsilon: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    output_scale: Union[torch.Tensor, float, None] = None,
    layout_code: int = QuantizationSFLayout.SWIZZLED_128x4,
    quant_type: int = MNNVLQuantType.NONE,
    launch_with_pdl: bool = False,
    strategy: MNNVLAllreduceFusionStrategy = MNNVLAllreduceFusionStrategy.AUTO,
    weight_bias: float = 0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Perform MNNVL AllReduce + Residual + RMSNorm + FP8/NVFP4 quantization.

    Quantization is applied after RMSNorm. ``output`` is optional; pass it only
    when the normalized non-quantized tensor is also needed. The quantized result
    is always returned as ``quant_out``.

    Args:
        input: Input tensor with shape ``[num_tokens, hidden_dim]``.
        residual_in: Residual tensor with the same shape as ``input``.
        gamma: RMSNorm gamma tensor with shape ``[hidden_dim]``.
        workspace: MNNVL workspace for the tensor-parallel group.
        epsilon: RMSNorm epsilon. Defaults to ``torch.finfo(input.dtype).eps``.
        output: Optional normalized output tensor with shape
            ``[num_tokens, hidden_dim]``.
        residual_out: Optional residual output tensor with shape
            ``[num_tokens, hidden_dim]``.
        quant_out: Optional quantized output. For FP8, shape must be
            ``[num_tokens, hidden_dim]`` and dtype ``torch.float8_e4m3fn``. For
            NVFP4, shape must be ``[num_tokens, hidden_dim // 2]`` and dtype
            ``torch.uint8`` or ``torch.float4_e2m1fn_x2``.
        scale_out: Optional scale output. Dynamic FP8 uses ``[num_tokens, 1]``
            float32. NVFP4 uses ``[num_tokens, hidden_dim // 16]`` for
            ``LINEAR`` layout, or a 1-D tensor large enough for the padded
            ``SWIZZLED_128x4`` layout. Static FP8 ignores this argument.
        output_scale: Scalar float or float32 tensor used as the quantization
            output scale. Defaults to ``1.0``.
        layout_code: NVFP4 scale layout. MNNVL supports ``SWIZZLED_128x4`` and
            ``LINEAR``; ``SWIZZLED_8x4`` is not supported.
        quant_type: ``MNNVLQuantType.FP8``, ``MNNVLQuantType.NVFP4``, or
            ``MNNVLQuantType.DYNAMIC_FP8``.
        launch_with_pdl: Whether to launch with PDL.
        strategy: MNNVL execution strategy. ``AUTO`` uses internal heuristics.
        weight_bias: Bias added to gamma before scaling. ``0.0`` for standard
            RMSNorm; ``1.0`` for Gemma / Qwen3.5 RMSNorm.

    Returns:
        A tuple ``(quant_out, scale_out, residual_out, output)``. ``scale_out``
        is ``None`` for static FP8; ``output`` is ``None`` unless requested.
    """

    if epsilon is None:
        epsilon = torch.finfo(input.dtype).eps
    if output_scale is None and quant_type != MNNVLQuantType.DYNAMIC_FP8:
        output_scale = 1.0

    if len(input.shape) != 2:
        raise ValueError(
            f"The input tensor must be 2D, got {len(input.shape)}D. The shape is {input.shape}."
        )
    if len(residual_in.shape) != 2:
        raise ValueError(
            f"The residual input tensor must be 2D, got {len(residual_in.shape)}D. The shape is {residual_in.shape}."
        )
    if residual_in.shape != input.shape:
        raise ValueError(
            f"residual_in shape must match input shape, got {residual_in.shape} and {input.shape}."
        )
    if gamma.numel() != input.shape[1]:
        raise ValueError(
            f"The gamma tensor must have the same number of elements as the hidden dimension, got {gamma.numel()} elements but expected {input.shape[1]} elements."
        )

    if layout_code == QuantizationSFLayout.SWIZZLED_8x4:
        raise ValueError(
            "MNNVL NVFP4 quantization supports SWIZZLED_128x4 or LINEAR scale layouts, not SWIZZLED_8x4."
        )

    if residual_out is None:
        residual_out = torch.empty_like(input)
    elif residual_out.shape != input.shape:
        raise ValueError(
            f"residual_out shape must match input shape, got {residual_out.shape} and {input.shape}."
        )

    if output is not None and output.shape != input.shape:
        raise ValueError(
            f"output shape must match input shape, got {output.shape} and {input.shape}."
        )

    if quant_type == MNNVLQuantType.DYNAMIC_FP8:
        output_scale_tensor = None
    elif isinstance(output_scale, torch.Tensor):
        if output_scale.numel() < 1:
            raise ValueError("output_scale must contain at least one element")
        output_scale_tensor = output_scale.to(device=input.device, dtype=torch.float32)
    else:
        output_scale_tensor = torch.tensor(
            [float(output_scale)], dtype=torch.float32, device=input.device
        )

    token_num, hidden_dim = input.shape
    if quant_type == MNNVLQuantType.FP8:
        if quant_out is None:
            quant_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        elif quant_out.shape != input.shape:
            raise ValueError(
                f"quant_out shape must be {tuple(input.shape)} for FP8, got {tuple(quant_out.shape)}."
            )
        if quant_out.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"quant_out dtype for FP8 must be float8_e4m3fn, got {quant_out.dtype}."
            )
        if not quant_out.is_contiguous():
            raise ValueError("quant_out must be contiguous for FP8.")
        scale_out = None
    elif quant_type == MNNVLQuantType.NVFP4:
        if input.dtype == torch.float32:
            raise ValueError("MNNVL NVFP4 quantization requires fp16 or bf16 input.")
        if hidden_dim % 16 != 0:
            raise ValueError(
                f"MNNVL NVFP4 quantization requires hidden_dim divisible by 16, got {hidden_dim}."
            )
        expected_quant_shape = (token_num, hidden_dim // 2)
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quant_out is None:
            quant_out = torch.empty(
                expected_quant_shape, dtype=torch.uint8, device=input.device
            )
        elif tuple(quant_out.shape) != expected_quant_shape:
            raise ValueError(
                f"quant_out shape must be {expected_quant_shape} for NVFP4, got {tuple(quant_out.shape)}."
            )
        if quant_out.dtype != torch.uint8 and not (
            fp4_dtype is not None and quant_out.dtype == fp4_dtype
        ):
            raise ValueError(
                f"quant_out dtype for NVFP4 must be uint8 or float4_e2m1fn_x2, got {quant_out.dtype}."
            )
        if not quant_out.is_contiguous():
            raise ValueError("quant_out must be contiguous for NVFP4.")

        expected_scale_out_numel = (
            token_num * hidden_dim // 16
            if layout_code == QuantizationSFLayout.LINEAR
            else _compute_swizzled_layout_sf_size(token_num, hidden_dim // 16)
        )
        if scale_out is None:
            if layout_code == QuantizationSFLayout.LINEAR:
                scale_out = torch.empty(
                    (token_num, hidden_dim // 16),
                    dtype=torch.float8_e4m3fn,
                    device=input.device,
                )
            else:
                scale_out = torch.empty(
                    _compute_swizzled_layout_sf_size(token_num, hidden_dim // 16),
                    dtype=torch.float8_e4m3fn,
                    device=input.device,
                )
        elif scale_out.numel() < expected_scale_out_numel:
            raise ValueError(
                f"scale_out is too small for NVFP4: got {scale_out.numel()} elements, need at least {expected_scale_out_numel}."
            )
        if scale_out.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"scale_out dtype for NVFP4 must be float8_e4m3fn, got {scale_out.dtype}."
            )
    elif quant_type == MNNVLQuantType.DYNAMIC_FP8:
        if quant_out is None:
            quant_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        elif quant_out.shape != input.shape:
            raise ValueError(
                f"quant_out shape must be {tuple(input.shape)} for dynamic FP8, got {tuple(quant_out.shape)}."
            )
        if quant_out.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"quant_out dtype for dynamic FP8 must be float8_e4m3fn, got {quant_out.dtype}."
            )
        if not quant_out.is_contiguous():
            raise ValueError("quant_out must be contiguous for dynamic FP8.")
        expected_scale_shape = (token_num, 1)
        if scale_out is None:
            scale_out = torch.empty(
                expected_scale_shape, dtype=torch.float32, device=input.device
            )
        elif tuple(scale_out.shape) != expected_scale_shape:
            raise ValueError(
                f"scale_out shape must be {expected_scale_shape} for dynamic FP8, got {tuple(scale_out.shape)}."
            )
        if scale_out.dtype != torch.float32:
            raise ValueError(
                f"scale_out dtype for dynamic FP8 must be float32, got {scale_out.dtype}."
            )
        if not scale_out.is_contiguous():
            raise ValueError("scale_out must be contiguous for dynamic FP8.")
    else:
        raise ValueError(f"Unsupported MNNVL quant_type: {quant_type}")

    if strategy == MNNVLAllreduceFusionStrategy.AUTO:
        strategy = MNNVLAllreduceFusionStrategy.select_strategy(
            workspace.tp_size, token_num, hidden_dim, input.dtype
        )
    if not workspace.is_buffer_size_sufficient(
        workspace.tp_size, token_num, hidden_dim, input.dtype, strategy
    ):
        raise ValueError(
            f"The buffer size in the given workspace is insufficient for the given problem size. Buffer: {workspace.buffer_size_bytes} bytes, Required: {workspace.get_required_buffer_size_bytes(workspace.tp_size, token_num, hidden_dim, input.dtype, strategy)} bytes."
        )

    for tensor, name in (
        (residual_in, "residual_in"),
        (gamma, "gamma"),
        (output, "output"),
        (residual_out, "residual_out"),
        (quant_out, "quant_out"),
        (scale_out, "scale_out"),
    ):
        if tensor is not None and tensor.device != input.device:
            raise ValueError(f"{name} must be on {input.device}, got {tensor.device}.")

    module = get_trtllm_mnnvl_comm_module()
    module.trtllm_mnnvl_allreduce_fusion(
        input,
        workspace.mc_ptr,
        workspace.uc_ptrs_dev,
        workspace.uc_ptr_local,
        workspace.buffer_flags,
        workspace.tp_size,
        workspace.rank,
        True,
        launch_with_pdl,
        strategy == MNNVLAllreduceFusionStrategy.ONESHOT,
        output,
        residual_out,
        residual_in,
        gamma,
        epsilon,
        weight_bias,
        quant_type,
        quant_out,
        scale_out,
        output_scale_tensor,
        layout_code,
    )
    return quant_out, scale_out, residual_out, output


# Legacy API that has been deprecated; Left for backward compatibility
@deprecated(
    "get_allreduce_mnnvl_workspace is deprecated, use MNNVLAllReduceFusionWorkspace class to manage the workspace instead"
)
def get_allreduce_mnnvl_workspace(
    mapping: Mapping,
    dtype: torch.dtype,
    comm_backend_for_handle_transfer: Optional[CommBackend] = None,
    buffer_size_in_bytes: Optional[int] = None,
) -> Tuple[MNNVLAllReduceFusionWorkspace, torch.Tensor, int]:
    """Get workspace buffers needed for multi-node NVLink all-reduce operation.

    Deprecated: use :class:`MNNVLAllReduceFusionWorkspace` to manage the
    workspace instead. This legacy helper is kept for backward compatibility
    and may be removed in a future release.

    Args:
        mapping: Tensor parallel mapping configuration containing rank info
        dtype: Data type of the tensors being reduced
        comm_backend_for_handle_transfer: Communication backend for handle transfer
        buffer_size_in_bytes: Optional buffer size. Practically, assign this to 3 * 2 * dtype.itemsize * hidden_dim * max_tokens

    Returns:
        Tuple containing:
        - MNNVLAllReduceFusionWorkspace: The workspace object backed by torch symmetric memory
        - torch.Tensor: Buffer flags tensor tracking state
        - int: Maximum number of elements that can fit in buffer
    """
    # buffer shape: [3, 2, buffer_tokens, hidden_dim]
    stride = 3 * 2 * dtype.itemsize
    lcm_hidden_dim = 286720
    # LCM for hidden_dim: 2048, 4096, 5120, 7168, 8192 = 286720
    # max_num_elements must be a multiple of 286720
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

    max_num_elements = workspace.buffer_size_bytes // stride

    return (
        workspace,
        workspace.buffer_flags,
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
    """Deprecated pointer-based MNNVL all-reduce API.

    Deprecated:
        Use :func:`trtllm_mnnvl_allreduce` with
        :class:`MNNVLAllReduceFusionWorkspace` instead. This legacy function is
        kept for compatibility and may be removed in a future release.

    Perform a multi-node NVLink all-reduce operation across multiple GPUs.

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

    Deprecated: use :func:`trtllm_mnnvl_fused_allreduce_add_rmsnorm` instead.
    This legacy function is kept for backward compatibility and may be removed
    in a future release.

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
