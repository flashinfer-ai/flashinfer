"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Unified AllReduce Fusion API

This module provides a unified interface for AllReduce + RMSNorm fusion operations
across different backends (TensorRT-LLM, MNNVL).

Example usage:
    >>> # Auto-select best backend based on topology
    >>> workspace = create_allreduce_fusion_workspace(
    ...     backend="auto",
    ...     world_size=8,
    ...     rank=0,
    ...     max_token_num=2048,
    ...     hidden_dim=4096,
    ...     dtype=torch.bfloat16,
    ...     topology="single_node"
    ... )
    >>>
    >>> # Perform AllReduce + RMSNorm fusion
    >>> prenorm = torch.empty_like(hidden_states)
    >>> normed = torch.empty_like(hidden_states)
    >>> output = allreduce_fusion(
    ...     input=hidden_states,
    ...     workspace=workspace,
    ...     launch_with_pdl=True,
    ...     residual_out=prenorm,
    ...     norm_out=normed,
    ...     residual_in=residual,
    ...     rms_gamma=norm_weight
    ... )
    >>>
    >>> destroy_allreduce_fusion_workspace(workspace)
"""

from typing import Union, Literal, Optional
from abc import ABC, abstractmethod

import torch

from ..utils import backend_requirement, supported_compute_capability


# ============================================================================
# WORKSPACE BASE CLASS
# ============================================================================


class AllReduceFusionWorkspace(ABC):
    """Base class for AllReduce fusion workspaces."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    @property
    @abstractmethod
    def backend(self) -> str:
        """Return backend name."""
        pass


class TRTLLMAllReduceFusionWorkspace(AllReduceFusionWorkspace):
    """TensorRT-LLM workspace for AllReduce fusion."""

    def __init__(self, world_size: int, rank: int, workspace_ptrs, metadata):
        super().__init__(world_size, rank)
        self.workspace_ptrs = workspace_ptrs
        self.metadata = metadata

    @property
    def backend(self) -> str:
        return "trtllm"


class MNNVLAllReduceFusionWorkspace(AllReduceFusionWorkspace):
    """MNNVL workspace for AllReduce fusion."""

    def __init__(
        self,
        world_size: int,
        rank: int,
        multicast_buffer_ptr: int,
        buffer_ptrs_dev: int,
        unicast_ptr: int,
        buffer_M: int,
        buffer_flags,
    ):
        super().__init__(world_size, rank)
        self.multicast_buffer_ptr = multicast_buffer_ptr
        self.buffer_ptrs_dev = buffer_ptrs_dev
        self.unicast_ptr = unicast_ptr
        self.buffer_M = buffer_M
        self.buffer_flags = buffer_flags

    @property
    def backend(self) -> str:
        return "mnnvl"


# ============================================================================
# BACKEND CHECKS - Hard requirements for decorator
# ============================================================================


@supported_compute_capability([80, 86, 89, 90, 100])
def _trtllm_workspace_check(
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: Optional[torch.device],
    topology: str,
    **kwargs,
) -> bool:
    """
    Check if trtllm backend CAN be used for workspace creation.

    Hard requirements:
    - SM80+ compute capability (checked by decorator)
    - Single-node topology
    - Module availability
    """
    # trtllm is optimized for single-node
    if topology == "multi_node":
        return False

    return True


@supported_compute_capability([90, 100])
def _mnnvl_workspace_check(
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: Optional[torch.device],
    topology: str,
    **kwargs,
) -> bool:
    """
    Check if mnnvl backend CAN be used for workspace creation.

    Hard requirements:
    - SM90+ compute capability (checked by decorator)
    - Multi-node topology
    - Module availability
    """
    # MNNVL is designed for multi-node
    if topology == "single_node":
        return False

    return True


# ============================================================================
# HEURISTIC - Performance-based selection for decorator
# ============================================================================


def _workspace_creation_heuristic(
    suitable_backends: list[str],
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: Optional[torch.device],
    topology: str,
    **kwargs,
) -> list[str]:
    """
    Select best backend for workspace creation based on performance.

    Called by decorator after checking which backends pass requirements.
    Uses benchmarking data to pick fastest option.

    Args:
        suitable_backends: List of backends that passed hard requirement checks
        backend: Requested backend ("auto", "trtllm", or "mnnvl")
        world_size: Number of ranks
        rank: Current rank
        max_token_num: Maximum number of tokens
        hidden_dim: Hidden dimension size
        dtype: Data type
        device: CUDA device
        topology: Network topology ("single_node" or "multi_node")
        **kwargs: Additional arguments

    Returns:
        List containing the selected backend (single element)
    """
    if not suitable_backends:
        return []

    if len(suitable_backends) == 1:
        return suitable_backends

    # Decision tree based on benchmark data
    # TODO: Replace with actual benchmarking results

    # Multi-node: MNNVL is designed for this
    if topology == "multi_node":
        if "mnnvl" in suitable_backends:
            return ["mnnvl"]

    # Single-node scenarios
    problem_size = max_token_num * hidden_dim

    # Large problems (>4M elements): trtllm optimized for throughput
    if problem_size > 4 * 1024 * 1024:
        if "trtllm" in suitable_backends:
            return ["trtllm"]

    # Small token counts (<128): trtllm one-shot has better latency
    if max_token_num < 128:
        if "trtllm" in suitable_backends:
            return ["trtllm"]

    # Small world sizes (<=4): trtllm one-shot efficient
    if world_size <= 4:
        if "trtllm" in suitable_backends:
            return ["trtllm"]

    # Default: return first available
    return [suitable_backends[0]]


# ============================================================================
# WORKSPACE CREATION - Uses decorator for all validation
# ============================================================================


@backend_requirement(
    backend_checks={
        "trtllm": _trtllm_workspace_check,
        "mnnvl": _mnnvl_workspace_check,
    },
    heuristic_func=_workspace_creation_heuristic,
)
def create_allreduce_fusion_workspace(
    backend: Literal["trtllm", "mnnvl", "auto"] = "auto",
    world_size: int = None,
    rank: int = None,
    max_token_num: int = None,
    hidden_dim: int = None,
    dtype: torch.dtype = None,
    device: Optional[torch.device] = None,
    topology: str = "single_node",
    process_group: Optional["torch.distributed.ProcessGroup"] = None,
    **backend_kwargs,
) -> AllReduceFusionWorkspace:
    """
    Create workspace for AllReduce fusion operations.

    Backend selection (checks + heuristics) handled by @backend_requirement decorator.

    Args:
        backend: Backend to use ("trtllm", "mnnvl", or "auto")
                 "auto" uses heuristic to select best backend based on topology
                 and problem size
        world_size: Number of ranks in the process group
        rank: Current rank ID
        max_token_num: Maximum number of tokens to support
        hidden_dim: Hidden dimension size
        dtype: Data type for communication tensors
        device: CUDA device (defaults to current CUDA device)
        topology: Network topology hint for backend selection
                  "single_node" - All ranks on one node (default)
                  "multi_node" - Ranks span multiple nodes
        process_group: PyTorch distributed process group
        **backend_kwargs: Additional backend-specific arguments

    Returns:
        Workspace object (TRTLLMAllReduceFusionWorkspace or MNNVLAllReduceFusionWorkspace)
        The workspace type determines which backend will be used in allreduce_fusion()

    Raises:
        BackendSupportedError: If no suitable backend available for the configuration
        ValueError: If problem size not supported for the specified backend

    Examples:
        >>> # Auto-select best backend based on topology
        >>> workspace = create_allreduce_fusion_workspace(
        ...     backend="auto",
        ...     world_size=8,
        ...     rank=0,
        ...     max_token_num=2048,
        ...     hidden_dim=4096,
        ...     dtype=torch.bfloat16,
        ...     topology="single_node"
        ... )
        >>> print(workspace.backend)  # "trtllm"

        >>> # Explicit backend selection
        >>> workspace = create_allreduce_fusion_workspace(
        ...     backend="mnnvl",
        ...     world_size=16,
        ...     rank=0,
        ...     max_token_num=2048,
        ...     hidden_dim=4096,
        ...     dtype=torch.bfloat16,
        ...     topology="multi_node"
        ... )
        >>> print(workspace.backend)  # "mnnvl"
    """
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Decorator has validated backend - now create workspace
    # If backend="auto", decorator has selected the best one and stored it

    # Get actual backend (decorator resolved "auto" to concrete backend)
    if backend == "auto":
        # Decorator stored the selected backend in suitable_auto_backends
        actual_backend = create_allreduce_fusion_workspace.suitable_auto_backends[0]
    else:
        actual_backend = backend

    # Create workspace for selected backend
    if actual_backend == "trtllm":
        from .trtllm_ar import trtllm_create_ipc_workspace_for_all_reduce_fusion

        workspace = trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_size=world_size,
            tp_rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=device,
            process_group=process_group,
            **backend_kwargs,
        )
        # Ensure workspace has required attributes for our API
        if not hasattr(workspace, "world_size"):
            workspace.world_size = world_size
        if not hasattr(workspace, "rank"):
            workspace.rank = rank
        return workspace

    elif actual_backend == "mnnvl":
        # TODO: Implement create_mnnvl_allreduce_fusion_workspace
        # For now, raise NotImplementedError with instructions
        raise NotImplementedError(
            "MNNVL workspace creation needs to be implemented. "
            "Expected function: trtllm_mnnvl_ar.create_mnnvl_allreduce_fusion_workspace"
        )
        # from .trtllm_mnnvl_ar import create_mnnvl_allreduce_fusion_workspace
        # return create_mnnvl_allreduce_fusion_workspace(
        #     world_size=world_size,
        #     rank=rank,
        #     max_token_num=max_token_num,
        #     hidden_dim=hidden_dim,
        #     dtype=dtype,
        #     device=device,
        #     **backend_kwargs
        # )
    else:
        raise RuntimeError(f"Unknown backend: {actual_backend}")


# ============================================================================
# WORKSPACE DESTRUCTION
# ============================================================================


def destroy_allreduce_fusion_workspace(workspace: AllReduceFusionWorkspace) -> None:
    """
    Destroy workspace and free resources.

    Automatically detects workspace type from the object and calls
    appropriate cleanup function.

    Args:
        workspace: Workspace object to destroy

    Example:
        >>> workspace = create_allreduce_fusion_workspace(...)
        >>> # ... use workspace ...
        >>> destroy_allreduce_fusion_workspace(workspace)
    """
    if isinstance(workspace, TRTLLMAllReduceFusionWorkspace):
        from .trtllm_ar import trtllm_destroy_ipc_workspace_for_all_reduce_fusion

        trtllm_destroy_ipc_workspace_for_all_reduce_fusion(workspace)
    elif isinstance(workspace, MNNVLAllReduceFusionWorkspace):
        # TODO: Implement MNNVL workspace destruction
        raise NotImplementedError("MNNVL workspace destruction not yet implemented")
        # from .trtllm_mnnvl_ar import destroy_mnnvl_allreduce_fusion_workspace
        # destroy_mnnvl_allreduce_fusion_workspace(workspace)
    else:
        raise TypeError(f"Unknown workspace type: {type(workspace)}")


# ============================================================================
# MAIN API - NO backend parameter, infers from workspace type
# ============================================================================


def allreduce_fusion(
    input: torch.Tensor,
    workspace: AllReduceFusionWorkspace,
    launch_with_pdl: bool = False,
    # ===== OUTPUT tensors (pre-allocated, will be filled) =====
    output: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    # ===== INPUT parameters =====
    residual_in: Optional[torch.Tensor] = None,
    rms_gamma: Optional[torch.Tensor] = None,
    rms_eps: float = 1e-6,
    scale_factor: Optional[Union[torch.Tensor, float]] = None,
    layout_code: Optional[int] = None,
    # ===== Control parameters =====
    pattern: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
    fp32_acc: bool = False,
    metadata: Optional[dict] = None,
) -> torch.Tensor:
    """
    AllReduce + RMSNorm fusion operation.

    Backend is automatically determined from workspace type.
    No backend parameter needed!

    Supports multiple fusion patterns:
    - AllReduce only
    - AllReduce + Residual + RMSNorm
    - AllReduce + Residual + RMSNorm + Quantization (FP8/FP4)

    Args:
        input: Input tensor [token_num, hidden_dim]
        workspace: Workspace object (type determines backend)
        launch_with_pdl: Use Persistent Device Launch

        # ===== OUTPUT tensors (pre-allocated, filled by function) =====
        output: AllReduce output [token_num, hidden_dim]
        residual_out: Prenorm output (after residual add, before norm) [token_num, hidden_dim]
        norm_out: Normalized output [token_num, hidden_dim]
        quant_out: Quantized output [token_num, hidden_dim] [trtllm only]
        scale_out: Quantization scale factors [trtllm only]

        # ===== INPUT parameters =====
        residual_in: Residual tensor to ADD [token_num, hidden_dim]
        rms_gamma: RMSNorm weight [hidden_dim]
        rms_eps: RMSNorm epsilon for numerical stability
        scale_factor: Input scale factor for quantization [trtllm only]
        layout_code: Scale factor layout (QuantizationSFLayout) [trtllm only]

        # ===== Control parameters =====
        pattern: Fusion pattern (AllReduceFusionPattern)
                 If None, auto-detected based on provided output tensors
        use_oneshot: [trtllm only] Use oneshot strategy vs twoshot
                     If None, uses internal heuristics
        fp32_acc: [trtllm only] Use FP32 accumulation for AllReduce
        metadata: [trtllm only] Workspace metadata for validation

    Returns:
        Output tensor (typically norm_out for fusion cases, output otherwise)

    Examples:
        >>> # Basic AllReduce + Residual + RMSNorm
        >>> workspace = create_allreduce_fusion_workspace(
        ...     backend="auto",
        ...     world_size=8,
        ...     rank=0,
        ...     max_token_num=2048,
        ...     hidden_dim=4096,
        ...     dtype=torch.bfloat16,
        ...     topology="single_node"
        ... )
        >>>
        >>> # Pre-allocate output tensors
        >>> prenorm = torch.empty_like(hidden_states)
        >>> normed = torch.empty_like(hidden_states)
        >>>
        >>> # Call fusion - backend inferred from workspace type
        >>> output = allreduce_fusion(
        ...     input=hidden_states,
        ...     workspace=workspace,
        ...     launch_with_pdl=True,
        ...     residual_out=prenorm,
        ...     norm_out=normed,
        ...     residual_in=residual,
        ...     rms_gamma=norm_weight
        ... )
        >>> # output == normed (final result)

        >>> # With FP8 quantization
        >>> quant = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
        >>> scales = torch.empty(token_num * hidden_dim // 16, dtype=torch.float16)
        >>>
        >>> output = allreduce_fusion(
        ...     input=hidden_states,
        ...     workspace=workspace,
        ...     norm_out=normed,
        ...     quant_out=quant,
        ...     scale_out=scales,
        ...     residual_in=residual,
        ...     rms_gamma=norm_weight,
        ...     scale_factor=scale_tensor
        ... )
    """
    # Auto-detect pattern if not provided
    if pattern is None:
        pattern = _infer_fusion_pattern(
            output, residual_in, residual_out, norm_out, quant_out, scale_out
        )

    # Infer backend from workspace type and dispatch
    if isinstance(workspace, TRTLLMAllReduceFusionWorkspace):
        return _allreduce_fusion_trtllm(
            input=input,
            workspace=workspace,
            launch_with_pdl=launch_with_pdl,
            output=output,
            residual_in=residual_in,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            scale_factor=scale_factor,
            layout_code=layout_code,
            pattern=pattern,
            use_oneshot=use_oneshot,
            fp32_acc=fp32_acc,
            metadata=metadata,
        )
    elif isinstance(workspace, MNNVLAllReduceFusionWorkspace):
        return _allreduce_fusion_mnnvl(
            input=input,
            workspace=workspace,
            launch_with_pdl=launch_with_pdl,
            residual_in=residual_in,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
        )
    else:
        raise TypeError(
            f"Unknown workspace type: {type(workspace)}. "
            f"Expected TRTLLMAllReduceFusionWorkspace or MNNVLAllReduceFusionWorkspace"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _infer_fusion_pattern(
    output, residual_in, residual_out, norm_out, quant_out, scale_out
) -> int:
    """
    Automatically infer fusion pattern from provided tensors.

    Returns AllReduceFusionPattern value based on which output tensors are provided.
    """
    from .trtllm_ar import AllReduceFusionPattern

    if quant_out is not None:
        # Quantization patterns
        if norm_out is not None and residual_out is not None:
            # Has separate norm output and residual output
            return AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant  # 4
        else:
            # Quant without separate outputs
            return AllReduceFusionPattern.kARResidualRMSNormFP8Quant  # 2
    elif norm_out is not None:
        # RMS Norm without quantization
        return AllReduceFusionPattern.kARResidualRMSNorm  # 1
    else:
        # Just AllReduce
        return AllReduceFusionPattern.kAllReduce  # 0


def _allreduce_fusion_trtllm(
    input: torch.Tensor,
    workspace: TRTLLMAllReduceFusionWorkspace,
    launch_with_pdl: bool,
    output: Optional[torch.Tensor],
    residual_in: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    rms_gamma: Optional[torch.Tensor],
    rms_eps: float,
    scale_factor: Optional[Union[torch.Tensor, float]],
    layout_code: Optional[int],
    pattern: int,
    use_oneshot: Optional[bool],
    fp32_acc: bool,
    metadata: Optional[dict],
) -> torch.Tensor:
    """TensorRT-LLM backend implementation."""
    from .trtllm_ar import trtllm_allreduce_fusion

    token_num, hidden_dim = input.shape

    if output is None:
        output = torch.empty_like(input)

    trtllm_allreduce_fusion(
        allreduce_in=input,
        world_size=workspace.world_size,
        world_rank=workspace.rank,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=workspace.workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        trigger_completion_at_end=launch_with_pdl,  # Same meaning
        fp32_acc=fp32_acc,
        pattern_code=pattern,
        use_oneshot=use_oneshot,
        allreduce_out=output,
        residual_in=residual_in,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        layout_code=layout_code,
        metadata=metadata,
    )

    # Return the most downstream output
    if norm_out is not None:
        return norm_out
    elif quant_out is not None:
        return quant_out
    else:
        return output


def _allreduce_fusion_mnnvl(
    input: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    launch_with_pdl: bool,
    residual_in: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    rms_gamma: Optional[torch.Tensor],
    rms_eps: float,
) -> torch.Tensor:
    """
    MNNVL backend implementation.

    Calls trtllm_mnnvl_fused_allreduce_rmsnorm which performs:
    1. AllReduce on input
    2. Add residual
    3. RMSNorm
    """
    from .trtllm_mnnvl_ar import trtllm_mnnvl_fused_allreduce_rmsnorm

    # Validate required parameters for RMS fusion
    if residual_in is None:
        raise ValueError("MNNVL AllReduce+RMS fusion requires residual_in")
    if residual_out is None:
        raise ValueError(
            "MNNVL AllReduce+RMS fusion requires residual_out (prenorm_output)"
        )
    if norm_out is None:
        raise ValueError("MNNVL AllReduce+RMS fusion requires norm_out (normed_output)")
    if rms_gamma is None:
        raise ValueError("MNNVL AllReduce+RMS fusion requires rms_gamma")

    # Call the MNNVL fusion function
    trtllm_mnnvl_fused_allreduce_rmsnorm(
        prenorm_output=residual_out,
        normed_output=norm_out,
        shard_input=input,
        multicast_buffer_ptr=workspace.multicast_buffer_ptr,
        buffer_ptrs_dev=workspace.buffer_ptrs_dev,
        unicast_ptr=workspace.unicast_ptr,
        buffer_M=workspace.buffer_M,
        buffer_flags_mnnvl=workspace.buffer_flags,
        nranks=workspace.world_size,
        rank=workspace.rank,
        gamma=rms_gamma,
        epsilon=rms_eps,
        residual=residual_in,
        launch_with_pdl=launch_with_pdl,
    )

    return norm_out


# ============================================================================
# CONTEXT MANAGER
# ============================================================================


class AllReduceFusionContext:
    """
    Context manager with automatic workspace management.

    This provides a convenient high-level API that handles workspace
    creation and cleanup automatically.

    Example:
        >>> with AllReduceFusionContext(
        ...     backend="auto",
        ...     world_size=8,
        ...     rank=0,
        ...     max_token_num=2048,
        ...     hidden_dim=4096,
        ...     dtype=torch.bfloat16,
        ...     topology="single_node"
        ... ) as ctx:
        ...     for batch in training_loop:
        ...         prenorm = torch.empty_like(batch.hidden_states)
        ...         normed = torch.empty_like(batch.hidden_states)
        ...
        ...         output = ctx.allreduce_fusion(
        ...             input=batch.hidden_states,
        ...             residual_out=prenorm,
        ...             norm_out=normed,
        ...             residual_in=batch.residual,
        ...             rms_gamma=model.norm_weight,
        ...             launch_with_pdl=True
        ...         )
        >>> # Workspace automatically cleaned up
    """

    def __init__(
        self,
        backend: Literal["trtllm", "mnnvl", "auto"] = "auto",
        world_size: int = None,
        rank: int = None,
        max_token_num: int = None,
        hidden_dim: int = None,
        dtype: torch.dtype = None,
        device: Optional[torch.device] = None,
        topology: str = "single_node",
        **kwargs,
    ):
        """
        Initialize context manager.

        Args:
            backend: Backend to use ("trtllm", "mnnvl", or "auto")
            world_size: Number of ranks
            rank: Current rank
            max_token_num: Maximum tokens to support
            hidden_dim: Hidden dimension
            dtype: Data type
            device: CUDA device
            topology: Network topology ("single_node" or "multi_node")
            **kwargs: Additional backend-specific arguments
        """
        # Workspace creation does all the selection logic via decorator
        self.workspace = create_allreduce_fusion_workspace(
            backend=backend,
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=device,
            topology=topology,
            **kwargs,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        destroy_allreduce_fusion_workspace(self.workspace)

    def allreduce_fusion(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Call allreduce_fusion with the managed workspace.

        Args:
            input: Input tensor
            **kwargs: Additional arguments passed to allreduce_fusion()

        Returns:
            Output tensor
        """
        return allreduce_fusion(input=input, workspace=self.workspace, **kwargs)
