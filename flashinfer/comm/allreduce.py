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
        >>> workspace.destroy()
"""

from typing import Union, Literal, Optional, Tuple, List, cast, Any
from .workspace_base import AllReduceFusionWorkspace

import torch

from .trtllm_ar import trtllm_allreduce_fusion
from .trtllm_ar import trtllm_create_ipc_workspace_for_all_reduce_fusion
from .trtllm_ar import check_trtllm_allreduce_fusion_workspace_metadata

from .mapping import Mapping

from .mnnvl import CommBackend, SymmDeviceMemory

# Note: AllReduceFusionPattern and QuantizationSFLayout are pseudo-types (classes with int constants)
# Import them for runtime use but type hint as int for mypy compatibility
from .trtllm_ar import AllReduceFusionPattern
from .trtllm_mnnvl_ar import MNNVLAllReduceFusionWorkspace
from .trtllm_mnnvl_ar import MNNVLAllreduceFusionStrategy
from .trtllm_mnnvl_ar import trtllm_mnnvl_allreduce
from .trtllm_mnnvl_ar import trtllm_mnnvl_fused_allreduce_add_rmsnorm

# ============================================================================
# WORKSPACE IMPLEMENTATIONS
# ============================================================================
#
# Workspace classes wrap the underlying backend workspace implementations:
# - TRTLLMAllReduceFusionWorkspace: Wraps trtllm_create_ipc_workspace_for_all_reduce_fusion
# - MNNVLAllReduceFusionWorkspace: Wraps MNNVL workspace (see trtllm_mnnvl_ar.py)
#
# Each workspace:
# 1. Calls the backend-specific workspace creation function in __init__
# 2. Stores the internal workspace as _internal_workspace
# 3. Exposes essential attributes for the unified API
# 4. Can be destroyed using workspace.destroy()
# ============================================================================


class TRTLLMAllReduceFusionWorkspace(AllReduceFusionWorkspace):
    """TensorRT-LLM workspace for AllReduce fusion."""

    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.float16,
        comm_backend: Optional[CommBackend] = None,
    ):
        """
        Create TensorRT-LLM AllReduce fusion workspace.

        Args:
            tp_size: Tensor parallel size (world size)
            tp_rank: Tensor parallel rank
            max_token_num: Maximum number of tokens
            hidden_dim: Hidden dimension size
            dtype: Data type
            comm_backend: Communication backend
            **kwargs: Additional arguments for workspace creation
        """
        super().__init__(tp_size, tp_rank)

        # Call the actual workspace creation function
        self._internal_workspace = trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank=tp_rank,
            tp_size=tp_size,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            comm_backend=comm_backend,
            create_metadata=True,
            use_fp32_lamport=dtype == torch.float32,
            use_symm_dev_mem=True,
        )

        # Store essential attributes for easy access
        # Cast to 3-tuple to make linter happy, since we always call with create_metadata=True
        workspace_tuple = cast(
            Tuple[List[List[int]], torch.Tensor, List[SymmDeviceMemory], dict],
            self._internal_workspace,
        )
        self.ipc_handles = workspace_tuple[0]
        self.workspace_tensor = workspace_tuple[1]
        self.mem_handles = workspace_tuple[2]
        self.metadata = workspace_tuple[3]

    @property
    def backend(self) -> str:
        return "trtllm"

    def __getattr__(self, name):
        """Delegate attribute access to internal workspace if not found."""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._internal_workspace, name)

    def is_buffer_size_sufficient(
        self,
        tp_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: Optional[Any] = None,
    ) -> bool:
        try:
            check_trtllm_allreduce_fusion_workspace_metadata(
                num_tokens, hidden_dim, tp_size, dtype, self.metadata
            )
            return True
        except ValueError as e:
            print(f"Workspace is insufficient for problem size. {e}")
            return False

    def destroy(self) -> None:
        """Destroy workspace and free resources."""
        if getattr(self, "_destroyed", False):
            return  # Already destroyed, nothing to do

        del self.ipc_handles
        del self.workspace_tensor
        del self.mem_handles
        del self.metadata
        self._destroyed = True


# ============================================================================
# BACKEND CHECKS - Hard requirements for backend selection
# ============================================================================


def _trtllm_workspace_check(
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> bool:
    """
    Check if trtllm backend CAN be used for workspace creation.

    Hard requirements:
    - Up to 16 ranks supported.
    """
    return world_size <= 16


def _mnnvl_workspace_check(
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> bool:
    """
    Check if mnnvl backend CAN be used for workspace creation.

    """

    return True


# ============================================================================
# HEURISTIC - Performance-based backend selection
# ============================================================================


def _workspace_creation_heuristic(
    suitable_backends: list[str],
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
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
        **kwargs: Additional arguments

    Note that at this point, the backend selection does not take "runtime parameters" into account, such as layout_code, and fusion pattern.

    Returns:
        List containing the selected backend (single element)
    """
    if not suitable_backends:
        return []

    if len(suitable_backends) == 1:
        return suitable_backends

    # Decision tree based on benchmark data

    # Single-node scenarios
    # From benchmarking data, we can see that MNNVL is either on par (smaller problem sizes) or significantly faster than TRTLLM (larger problem sizes such as hidden_dim=8192, token_num=64 for TP=4), for single-node scenarios.
    # However, trtllm has a larger support surface (more fusion patterns, more quantization support, etc.)
    if "mnnvl" in suitable_backends:
        return ["mnnvl"]
    else:
        return [suitable_backends[0]]


# ============================================================================
# WORKSPACE CREATION
# ============================================================================


def create_allreduce_fusion_workspace(
    backend: Literal["trtllm", "mnnvl", "auto"] = "auto",
    world_size: int = None,
    rank: int = None,
    max_token_num: int = None,
    hidden_dim: int = None,
    dtype: torch.dtype = None,
    gpus_per_node: int = None,
    comm_backend: Optional[CommBackend] = None,
    force_oneshot_support: bool = False,
) -> AllReduceFusionWorkspace:
    """
    Create workspace for AllReduce fusion operations.

    Backend selection uses topology-based checks and heuristics.

    **Important: Workspace Reusability**
    The workspace is allocated based on the total size (max_token_num * hidden_dim * dtype_size).
    You can reuse the same workspace with different shapes as long as the total size fits:

    - Workspace(max_token_num=2048, hidden_dim=4096) can handle:
      - (token_num=2048, hidden_dim=4096) ✓
      - (token_num=1024, hidden_dim=4096) ✓
      - (token_num=4096, hidden_dim=2048) ✓ (same total size)
      - (token_num=1024, hidden_dim=8192) ✓ (same total size)
      - (token_num=4096, hidden_dim=4096) ✗ (too large)

    Use `workspace.is_buffer_size_sufficient(token_num, hidden_dim, dtype)` to check before use.

    Args:
        backend: Backend to use ("trtllm", "mnnvl", or "auto")
                 "auto" uses heuristic to select best backend
        world_size: Number of ranks in the process group
        rank: Current rank ID
        max_token_num: Maximum number of tokens to support
        hidden_dim: Hidden dimension size
        dtype: Data type for communication tensors
        gpus_per_node: Number of GPUs per node (for multi-node topology).
        comm_backend: Communication backend to use.
        force_oneshot_support: Allocate workspace for oneshot strategy vs twoshot
                    True: Allocate workspace for oneshot strategy up to the largest problem size requested
                    False: Allocate workspace for twoshot strategy for all problem sizes, and for oneshot strategy up to the heuristic threshold.
                    Note that only the workspace for MNNVL backend needs to be initialized with the correct strategy.
                    The trtllm backend will be sufficient for both strategies.

    Returns:
        Workspace object (TRTLLMAllReduceFusionWorkspace or MNNVLAllReduceFusionWorkspace)
        The workspace type determines which backend will be used in allreduce_fusion()

    Raises:
        BackendSupportedError: If no suitable backend available for the configuration
        ValueError: If problem size not supported for the specified backend

    Examples:
        >>> # Auto-select best backend
        >>> workspace = create_allreduce_fusion_workspace(
        ...     backend="auto",
        ...     world_size=8,
        ...     rank=0,
        ...     max_token_num=2048,
        ...     hidden_dim=4096,
        ...     dtype=torch.bfloat16,
        ... )
        >>> print(workspace.backend)  # "trtllm"
        >>> print(workspace.get_workspace_capacity())  # 8388608 elements

        >>> # Check if workspace can handle different problem sizes
        >>> workspace.is_buffer_size_sufficient(1024, 4096, 8, torch.bfloat16)  # True
        >>> workspace.is_buffer_size_sufficient(4096, 2048, 8, torch.bfloat16)  # True (same total)

        >>> # Explicit backend selection
        >>> workspace = create_allreduce_fusion_workspace(
        ...     backend="mnnvl",
        ...     world_size=16,
        ...     rank=0,
        ...     max_token_num=2048,
        ...     hidden_dim=4096,
        ...     dtype=torch.bfloat16,
        ... )
        >>> print(workspace.backend)  # "mnnvl"
    """
    if gpus_per_node is None:
        gpus_per_node = min(torch.cuda.device_count(), world_size)
    # Determine the actual backend to use
    if backend == "auto":
        # Find suitable backends (any compute capability check needs to be checked at kernel runtime, since there are no tensor available at this point)
        suitable_backends = []
        if _trtllm_workspace_check(
            backend=backend,
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
        ):
            suitable_backends.append("trtllm")
        if _mnnvl_workspace_check(
            backend=backend,
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
        ):
            suitable_backends.append("mnnvl")

        if not suitable_backends:
            raise ValueError("No suitable backend found. ")

        # Apply heuristic to select best backend
        selected = _workspace_creation_heuristic(
            suitable_backends=suitable_backends,
            backend=backend,
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
        actual_backend = selected[0]
    else:
        actual_backend = backend

    # Create workspace for selected backend using workspace constructors
    if actual_backend == "trtllm":
        return TRTLLMAllReduceFusionWorkspace(
            tp_size=world_size,
            tp_rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=comm_backend,
        )

    elif actual_backend == "mnnvl":
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=gpus_per_node,
            tp_size=world_size,
        )
        buffer_size_in_bytes = None
        if force_oneshot_support:
            buffer_size_in_bytes = (
                MNNVLAllReduceFusionWorkspace.get_required_buffer_size_bytes(
                    world_size,
                    max_token_num,
                    hidden_dim,
                    dtype,
                    MNNVLAllreduceFusionStrategy.ONESHOT,
                )
            )

        return MNNVLAllReduceFusionWorkspace(
            mapping=mapping,
            max_num_tokens=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=comm_backend,
            buffer_size_in_bytes=buffer_size_in_bytes,
        )
    else:
        raise RuntimeError(f"Unknown backend: {actual_backend}")


# ============================================================================
# MAIN API - NO backend parameter, infers from workspace type
# ============================================================================


def allreduce_fusion(
    input: torch.Tensor,
    workspace: AllReduceFusionWorkspace,
    pattern: int,
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
    use_oneshot: Optional[bool] = None,
    fp32_acc: bool = False,
) -> torch.Tensor:
    """
    AllReduce + RMSNorm fusion operation.

    Backend is automatically determined from workspace type. If you need another backend, create the workspace for the desired backend.

    Supports multiple fusion patterns:
    - AllReduce only
    - AllReduce + Residual + RMSNorm
    - AllReduce + Residual + RMSNorm + Quantization (FP8/FP4)

    **Note on Workspace Reusability:**
    You can reuse the same workspace with different (token_num, hidden_dim) combinations
    as long as `workspace.is_buffer_size_sufficient(token_num, hidden_dim, tp_size, dtype)` returns True.

    Args:
        input: Input tensor [token_num, hidden_dim]
        workspace: Workspace object (type determines backend, see create_allreduce_fusion_workspace)
        pattern: Fusion pattern (AllReduceFusionPattern constant, 0-5)
                 - kAllReduce = 0
                 - kARResidualRMSNorm = 1
                 - kARResidualRMSNormFP8Quant = 2
                 - kARResidualRMSNormFP4Quant = 3
                 - kARResidualRMSNormOutFP8Quant = 4
                 - kARResidualRMSNormOutFP4Quant = 5
                 Note: MNNVL only supports patterns 0 and 1
        launch_with_pdl: Use Persistent Dependency Launch

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
        use_oneshot: Use oneshot strategy vs twoshot
                     If None, uses internal heuristics.
                     Note: when explicitly set to True, the MNNVL backend needs to be initialized with a sufficiently large workspace.
        fp32_acc: [trtllm only] Use FP32 accumulation for AllReduce

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
        ...     pattern=AllReduceFusionPattern.kARResidualRMSNorm,
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
        ...     pattern=AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        ...     norm_out=normed,
        ...     quant_out=quant,
        ...     scale_out=scales,
        ...     residual_in=residual,
        ...     rms_gamma=norm_weight,
        ...     scale_factor=scale_tensor
        ... )
    """
    # Dispatch based on workspace type
    if isinstance(workspace, TRTLLMAllReduceFusionWorkspace):
        # TensorRT-LLM backend implementation
        # Extract shape from 2D input
        token_num, hidden_dim = input.shape

        # Allocate output if needed (keep 2D shape)
        if output is None:
            output = torch.empty_like(input)

        # Flatten all tensors to 1D for legacy trtllm_allreduce_fusion API
        # The legacy API expects flattened tensors and explicit token_num/hidden_dim
        # We require contiguous tensors so that view(-1) creates a view (not a copy),
        # ensuring writes to the flattened tensors are reflected in the original 2D tensors
        def _flatten_checked(t, name):
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            return t.view(-1)

        input_flat = _flatten_checked(input, "input")
        output_flat = _flatten_checked(output, "output")
        residual_in_flat = (
            _flatten_checked(residual_in, "residual_in")
            if residual_in is not None
            else None
        )
        residual_out_flat = (
            _flatten_checked(residual_out, "residual_out")
            if residual_out is not None
            else None
        )
        norm_out_flat = (
            _flatten_checked(norm_out, "norm_out") if norm_out is not None else None
        )
        quant_out_flat = (
            _flatten_checked(quant_out, "quant_out") if quant_out is not None else None
        )

        # Call legacy API with flattened tensors
        # Note: pattern and layout_code are ints but legacy API uses pseudo-type hints
        trtllm_allreduce_fusion(
            allreduce_in=input_flat,
            world_size=workspace.world_size,
            world_rank=workspace.rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=workspace.workspace_tensor,
            launch_with_pdl=launch_with_pdl,
            trigger_completion_at_end=launch_with_pdl,  # Same meaning
            fp32_acc=fp32_acc,
            pattern_code=pattern,  # type: ignore[arg-type]
            use_oneshot=use_oneshot,
            allreduce_out=output_flat,
            residual_in=residual_in_flat,
            residual_out=residual_out_flat,
            norm_out=norm_out_flat,
            quant_out=quant_out_flat,
            scale_out=scale_out,  # scale_out is not reshaped
            rms_gamma=rms_gamma,  # 1D tensor, no reshape needed
            rms_eps=rms_eps,
            scale_factor=scale_factor,
            layout_code=layout_code,  # type: ignore[arg-type]
            metadata=workspace.metadata,
        )

        # Return the most downstream output (already in 2D shape from input views)
        if norm_out is not None:
            return norm_out
        elif quant_out is not None:
            return quant_out
        else:
            return output

    elif isinstance(workspace, MNNVLAllReduceFusionWorkspace):
        if (
            pattern != AllReduceFusionPattern.kARResidualRMSNorm
            and pattern != AllReduceFusionPattern.kAllReduce
        ):
            raise ValueError(
                f"MNNVL AllReduce+RMS fusion does not support pattern {pattern}. Please try the TRTLLM backend instead."
            )

        if layout_code is not None:
            raise ValueError(
                "MNNVL AllReduce does not support quantization fusion and thus no layout_code"
            )

        # MNNVL backend implementation
        if pattern == AllReduceFusionPattern.kAllReduce:
            # AllReduce only
            if output is None:
                output = torch.empty_like(input)
            trtllm_mnnvl_allreduce(
                input=input,
                workspace=workspace,
                launch_with_pdl=launch_with_pdl,
                output=output,
            )
            return output

        elif pattern == AllReduceFusionPattern.kARResidualRMSNorm:
            # AllReduce + Residual + RMSNorm fusion
            # Validate required parameters
            if residual_in is None:
                raise ValueError("MNNVL AllReduce+RMS fusion requires residual_in")
            if rms_gamma is None:
                raise ValueError("MNNVL AllReduce+RMS fusion requires rms_gamma")

            # Allocate output tensors if not provided
            if norm_out is None:
                norm_out = torch.empty_like(input)
            if residual_out is None:
                residual_out = torch.empty_like(input)

            # Call the MNNVL fusion function
            norm_result, residual_result = trtllm_mnnvl_fused_allreduce_add_rmsnorm(
                input=input,
                residual_in=residual_in,
                gamma=rms_gamma,
                workspace=workspace,
                epsilon=rms_eps,
                output=norm_out,
                residual_out=residual_out,
                launch_with_pdl=launch_with_pdl,
            )
            return norm_result

        else:
            raise ValueError(f"Unsupported pattern for MNNVL backend: {pattern}")

    else:
        raise TypeError(
            f"Unknown workspace type: {type(workspace)}. "
            f"Expected TRTLLMAllReduceFusionWorkspace or MNNVLAllReduceFusionWorkspace"
        )
