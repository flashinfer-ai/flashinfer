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

import logging

logger = logging.getLogger(__name__)

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
from torch.distributed import ProcessGroup

from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.comm import allreduce_fusion_trace
from flashinfer.utils import is_confidential_compute

from .trtllm_ar import trtllm_allreduce_fusion
from .trtllm_ar import trtllm_create_ipc_workspace_for_all_reduce_fusion
from .trtllm_ar import _initialize_allreduce_fusion_protocol
from .trtllm_ar import check_trtllm_allreduce_fusion_workspace_metadata
from .trtllm_ar import trtllm_moe_allreduce_fusion
from .trtllm_ar import trtllm_moe_finalize_allreduce_fusion

from .mapping import Mapping

from .mnnvl import (
    CommBackend,
    SymmDeviceMemory,
    all_ranks_support_mnnvl,
    is_multicast_supported,
)

# Note: AllReduceFusionPattern and QuantizationSFLayout are pseudo-types (classes with int constants)
# Import them for runtime use but type hint as int for mypy compatibility
from .trtllm_ar import AllReduceFusionPattern
from .trtllm_ar import QuantizationSFLayout
from .trtllm_mnnvl_ar import MNNVLAllReduceFusionWorkspace
from .trtllm_mnnvl_ar import MNNVLAllreduceFusionStrategy
from .trtllm_mnnvl_ar import MNNVLQuantType
from .trtllm_mnnvl_ar import trtllm_mnnvl_allreduce
from .trtllm_mnnvl_ar import trtllm_mnnvl_fused_allreduce_add_rmsnorm
from .trtllm_mnnvl_ar import trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant

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
        group: Optional[ProcessGroup] = None,
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
            group: Process group for symmetric memory rendezvous. Defaults to torch.distributed.group.WORLD.
        """
        super().__init__(tp_size, tp_rank)

        # NVIDIA Confidential Computing requires multicast-free IPC workspaces so needs to disable symmetric device memory
        use_symm_dev_mem = not is_confidential_compute()

        self._internal_workspace = trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank=tp_rank,
            tp_size=tp_size,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            group=group,
            comm_backend=comm_backend,
            create_metadata=True,
            use_fp32_lamport=dtype == torch.float32,
            use_symm_dev_mem=use_symm_dev_mem,
        )

        # Store essential attributes for easy access
        # Cast to 3-tuple to make linter happy, since we always call with create_metadata=True
        if use_symm_dev_mem:
            # use_symm_dev_mem=True: (ipc_handles, workspace_tensor, mem_handles, metadata)
            symm_workspace_tuple = cast(
                Tuple[List[List[int]], torch.Tensor, List[SymmDeviceMemory], dict],
                self._internal_workspace,
            )
            self.ipc_handles = symm_workspace_tuple[0]
            self.workspace_tensor = symm_workspace_tuple[1]
            self.mem_handles = symm_workspace_tuple[2]
            self.metadata = symm_workspace_tuple[3]
        else:
            # use_symm_dev_mem=False: (ipc_handles, workspace_tensor, metadata)
            ipc_workspace_tuple = cast(
                Tuple[List[List[int]], torch.Tensor, dict],
                self._internal_workspace,
            )
            self.ipc_handles = ipc_workspace_tuple[0]
            self.workspace_tensor = ipc_workspace_tuple[1]
            self.metadata = ipc_workspace_tuple[2]
            # No symmetric-memory handles for the multicast-free IPC path.
            self.mem_handles = []

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
            logger.warning("Workspace is insufficient for problem size. %s", e)
            return False

    @flashinfer_api
    def checkpoint_prepare(self) -> None:
        """Detach physical backing; repeated successful calls are no-ops."""
        if not self.mem_handles or not all(
            isinstance(handle, SymmDeviceMemory) for handle in self.mem_handles
        ):
            raise NotImplementedError(
                "Stable-VA checkpointing is unavailable for workspaces backed "
                "by torch symmetric memory"
            )

        mapped = [handle.mapped for handle in self.mem_handles]
        if not any(mapped):
            return
        if not all(mapped):
            raise RuntimeError("TRT-LLM symmetric-memory handle state is inconsistent")

        for handle in self.mem_handles:
            handle._unmap_and_release_handles()
        # Do not return until every rank has released all workspace handles.
        self.mem_handles[0].comm_backend.barrier()

    @flashinfer_api
    def checkpoint_restore(self, comm_backend: CommBackend) -> None:
        """Restore physical backing; repeated successful calls are no-ops.

        Parameters
        ----------
        comm_backend : CommBackend
            Communication backend used to recreate and exchange workspace
            memory handles. It must have the same rank and world size as the
            original allocation.
        """
        if not self.mem_handles or not all(
            isinstance(handle, SymmDeviceMemory) for handle in self.mem_handles
        ):
            raise NotImplementedError(
                "Stable-VA checkpointing is unavailable for workspaces backed "
                "by torch symmetric memory"
            )

        mapped = [handle.mapped for handle in self.mem_handles]
        if all(mapped):
            return
        if any(mapped):
            raise RuntimeError("TRT-LLM symmetric-memory handle state is inconsistent")
        for handle in self.mem_handles:
            handle._create_and_map_handles(comm_backend)

        _initialize_allreduce_fusion_protocol(
            ipc_handles=self.ipc_handles,
            tp_rank=self.rank,
            flag_size=self.metadata["flag_size"],
            lamport_buffer_size=self.metadata["lamport_buffer_size"],
            lamport_comm_size=self.metadata["lamport_comm_size"],
            use_fp32_lamport=self.metadata["use_fp32_lamport"],
            control_flag_ptr=self.metadata["control_flag_ptr"],
        )
        torch.cuda.synchronize()
        comm_backend.barrier()

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
    # TRTLLM still has the larger specialized-fusion surface, such as MoE
    # patterns and packed group FP8 quantization.
    if "mnnvl" in suitable_backends:
        return ["mnnvl"]
    else:
        return [suitable_backends[0]]


# ============================================================================
# WORKSPACE CREATION
# ============================================================================


@flashinfer_api
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
    group: Optional[ProcessGroup] = None,
) -> AllReduceFusionWorkspace:
    r"""Create workspace for AllReduce fusion operations.

    Backend selection uses topology-based checks and heuristics.

    **Important: Workspace Reusability**
    The workspace is allocated based on the total size
    (``max_token_num * hidden_dim * dtype_size``). You can reuse the same
    workspace with different shapes as long as the total size fits.

    Use ``workspace.is_buffer_size_sufficient(tp_size, num_tokens, hidden_dim, dtype)``
    to check before reusing.

    Parameters
    ----------
    backend : Literal["trtllm", "mnnvl", "auto"]
        Backend to use. ``"auto"`` uses a topology-based heuristic to pick
        between ``"trtllm"`` and ``"mnnvl"``.
    world_size : int
        Number of ranks in the process group.
    rank : int
        Current rank id.
    max_token_num : int
        Maximum number of tokens the workspace must support.
    hidden_dim : int
        Hidden dimension size.
    dtype : torch.dtype
        Element dtype of the communication tensors.
    gpus_per_node : int, optional
        Number of GPUs per node (used for multi-node topology decisions).
        Defaults to ``min(torch.cuda.device_count(), world_size)``.
    comm_backend : Optional[CommBackend]
        Communication backend to use for rendezvous. Defaults to the
        process-group's default.
    force_oneshot_support : bool
        If ``True``, allocate workspace for the oneshot strategy up to the
        largest problem size requested. If ``False`` (default), allocate
        workspace for the twoshot strategy across all problem sizes and for
        the oneshot strategy up to the heuristic threshold. Only the MNNVL
        backend needs to be initialized with the correct strategy; the
        TRT-LLM backend works for both.
    group : Optional[ProcessGroup]
        Process group used for symmetric-memory rendezvous (TRT-LLM backend
        only). Defaults to ``torch.distributed.group.WORLD``.

    Returns
    -------
    AllReduceFusionWorkspace
        Either a ``TRTLLMAllReduceFusionWorkspace`` or
        ``MNNVLAllReduceFusionWorkspace``. The workspace type determines
        which backend :func:`allreduce_fusion` will dispatch to.

    Raises
    ------
    ValueError
        If no suitable backend is available for the requested configuration,
        or if the problem size is not supported by the chosen backend.
    RuntimeError
        If an explicit ``backend`` argument is passed that does not match
        any known backend implementation.

    Examples
    --------

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

    >>> # Explicit backend selection
    >>> workspace = create_allreduce_fusion_workspace(
    ...     backend="mnnvl",
    ...     world_size=16,
    ...     rank=0,
    ...     max_token_num=2048,
    ...     hidden_dim=4096,
    ...     dtype=torch.bfloat16,
    ... )

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
        local_mnnvl_supported = is_multicast_supported(torch.cuda.current_device())
        if all_ranks_support_mnnvl(
            local_mnnvl_supported, world_size, comm_backend, group
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
            group=group,
        )

    elif actual_backend == "mnnvl":
        if is_confidential_compute():
            raise ValueError(
                "NVIDIA Confidential Computing is not supported by the mnnvl AllReduce fusion backend "
                "since mnnvl backend requires NVLink multicast, which is unavailable under Confidential Computing. "
                "Use backend='trtllm' instead."
            )
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


@flashinfer_api(trace=allreduce_fusion_trace)
def allreduce_fusion(
    input: torch.Tensor,
    workspace: AllReduceFusionWorkspace,
    pattern: int,
    launch_with_pdl: bool = False,
    trigger_completion_at_end: bool = True,
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
    # ===== MOE Reduction parameters (pattern=kMoEReductionARResidualRMSNorm) =====
    moe_reduction_device_num_experts: Optional[int] = None,
    moe_reduction_scale_input: Optional[torch.Tensor] = None,
    moe_reduction_active_experts_token_input: Optional[torch.Tensor] = None,
    moe_reduction_token_input: Optional[torch.Tensor] = None,
    # ===== MOE Finalize parameters (pattern=kMoEFinalizeARResidualRMSNorm) =====
    expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
    expert_scale_factor: Optional[torch.Tensor] = None,
    shared_expert_output: Optional[torch.Tensor] = None,
    # ===== Group quant parameters =====
    block_quant_group_size: Optional[int] = None,
    # ===== RMSNorm variant =====
    weight_bias: float = 0.0,
) -> torch.Tensor:
    r"""AllReduce + RMSNorm fusion operation, with optional FP8/NVFP4
    quantization for supported backends.

    Backend is automatically determined from workspace type. If you need a
    different backend, create the workspace for that backend.

    Supports multiple fusion patterns:

    * AllReduce only
    * AllReduce + Residual + RMSNorm
    * AllReduce + Residual + RMSNorm + Quantization (FP8 / NVFP4)

    .. note::

        You can reuse the same workspace with different
        ``(num_tokens, hidden_dim)`` combinations as long as
        ``workspace.is_buffer_size_sufficient(tp_size, num_tokens, hidden_dim, dtype)``
        returns ``True``.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``[token_num, hidden_dim]`` for standard
        allreduce patterns. For ``kMoEFinalizeARResidualRMSNorm``, this is
        the permuted/padded MoE expert output of shape
        ``[num_permuted_rows, hidden_dim]``; the token output shape is
        determined by ``residual_in``.
    workspace : AllReduceFusionWorkspace
        Workspace object created by :func:`create_allreduce_fusion_workspace`.
        Its concrete type (TRT-LLM vs MNNVL) determines the backend.
    pattern : int
        Fusion pattern (``AllReduceFusionPattern`` constant):

        * ``kAllReduce = 0``
        * ``kARResidualRMSNorm = 1``
        * ``kARResidualRMSNormFP8Quant = 2``
        * ``kARResidualRMSNormFP4Quant = 3``
        * ``kARResidualRMSNormOutFP8Quant = 4``
        * ``kARResidualRMSNormOutFP4Quant = 5``
        * ``kMoEReductionARResidualRMSNorm = 6`` (TRT-LLM only)
        * ``kMoEFinalizeARResidualRMSNorm = 7`` (TRT-LLM only)
        * ``kARResidualRMSNormPerTokenGroupFP8PackedQuant = 8`` (TRT-LLM only)
        * ``kARResidualRMSNormOutPerTokenGroupFP8PackedQuant = 9`` (TRT-LLM only)
        * ``kARResidualRMSNormDynamicFP8Quant = 10``
        * ``kARResidualRMSNormOutDynamicFP8Quant = 11``

        MNNVL supports the standard FP8/NVFP4 quant patterns (2-5) and
        dynamic FP8 patterns (10-11). MoE and packed group quant patterns
        remain TRT-LLM only.

        ``kMoEFinalizeARResidualRMSNorm`` is an explicit TRT-LLM fused
        implementation of MoE finalize + AllReduce + RMSNorm and does not
        use the MNNVL backend. Fusion is not always the fastest path for every
        workload; benchmark this pattern on the target hardware, model,
        tensor-parallel size, and serving mode before enabling it by default.
    launch_with_pdl : bool
        Use Programmatic Dependent Launch.
    trigger_completion_at_end : bool
        TRT-LLM only. Controls when PDL completion is signaled. ``True``
        (default) signals after the kernel finishes (safe, no overlap).
        ``False`` signals early, allowing the next PDL-aware kernel to
        overlap with this one. Only safe when the next kernel also calls
        ``cudaGridDependencySynchronize()``. Ignored by the MNNVL backend.
    output : Optional[torch.Tensor]
        Pre-allocated AllReduce output buffer, shape
        ``[token_num, hidden_dim]``.
    residual_out : Optional[torch.Tensor]
        Pre-allocated pre-norm output (after residual add, before norm),
        shape ``[token_num, hidden_dim]``.
    norm_out : Optional[torch.Tensor]
        Pre-allocated normalized output, shape ``[token_num, hidden_dim]``.
    quant_out : Optional[torch.Tensor]
        Pre-allocated quantized output. FP8 uses shape
        ``[token_num, hidden_dim]`` and NVFP4 uses shape
        ``[token_num, hidden_dim / 2]``.
    scale_out : Optional[torch.Tensor]
        Pre-allocated quantization scale-factor buffer. Dynamic FP8 uses
        shape ``[token_num, 1]`` and dtype ``torch.float32``. NVFP4 uses
        the layout selected by ``layout_code``. Per-tensor FP8 does not use
        ``scale_out``.
    residual_in : Optional[torch.Tensor]
        Residual tensor to add, shape ``[token_num, hidden_dim]``.
    rms_gamma : Optional[torch.Tensor]
        RMSNorm weight, shape ``[hidden_dim]``.
    rms_eps : float
        RMSNorm epsilon for numerical stability.
    scale_factor : Optional[Union[torch.Tensor, float]]
        Output scale used by FP8/NVFP4 quantization.
    layout_code : Optional[int]
        NVFP4 scale-factor layout (``QuantizationSFLayout``). MNNVL
        supports ``SWIZZLED_128x4`` and ``LINEAR``; ``SWIZZLED_8x4``
        remains TRT-LLM only.
    use_oneshot : Optional[bool]
        ``True``/``False`` forces the oneshot/twoshot strategy; ``None``
        (default) uses internal heuristics. When set to ``True`` for
        MNNVL, the workspace must have been allocated with a sufficiently
        large size.
    fp32_acc : bool
        TRT-LLM only. Use FP32 accumulation for AllReduce.
    moe_reduction_device_num_experts : Optional[int]
        Number of local experts on this device, required for
        ``pattern=kMoEReductionARResidualRMSNorm``.
    moe_reduction_scale_input : Optional[torch.Tensor]
        Per-token-per-expert scales, shape ``[token_num, num_experts]``.
    moe_reduction_active_experts_token_input : Optional[torch.Tensor]
        Per-token-per-expert outputs, shape
        ``[token_num * num_experts, hidden_dim]``.
    moe_reduction_token_input : Optional[torch.Tensor]
        Per-token input (e.g. FC2 output), shape
        ``[token_num, hidden_dim]``.
    expanded_idx_to_permuted_idx : Optional[torch.Tensor]
        Mapping from ``(token, topk_idx)`` to permuted expert output row.
        Shape ``[token_num, top_k]``, dtype ``int32``. Required for
        ``pattern=kMoEFinalizeARResidualRMSNorm``.
    expert_scale_factor : Optional[torch.Tensor]
        Router weights for each selected expert, shape
        ``[token_num, top_k]``.
    shared_expert_output : Optional[torch.Tensor]
        Optional shared-expert output to add, shape
        ``[token_num, hidden_dim]``.
    block_quant_group_size : Optional[int]
        Group size for per-token-group FP8 packed quantization patterns
        (TRT-LLM only).
    weight_bias : float
        Bias added to ``rms_gamma`` before scaling.

        * ``0.0`` (default): standard RMSNorm
          (``out = gamma * x * rsqrt(...)``).
        * ``1.0``: Gemma / Qwen3.5 RMSNorm
          (``out = (1 + gamma) * x * rsqrt(...)``).

        Supported by both TRT-LLM and MNNVL backends for standard RMSNorm
        and quant patterns (1-5), and by TRT-LLM for MoE RMSNorm
        variants. Ignored for ``kAllReduce``.

    Returns
    -------
    torch.Tensor
        Output tensor for the selected pattern. Quant patterns return
        ``quant_out``, RMSNorm patterns return ``norm_out``, and
        ``kAllReduce`` returns ``output``.

    Examples
    --------

    >>> # Basic AllReduce + Residual + RMSNorm
    >>> workspace = create_allreduce_fusion_workspace(
    ...     backend="auto", world_size=8, rank=0,
    ...     max_token_num=2048, hidden_dim=4096, dtype=torch.bfloat16,
    ... )
    >>> prenorm = torch.empty_like(hidden_states)
    >>> normed = torch.empty_like(hidden_states)
    >>> output = allreduce_fusion(
    ...     input=hidden_states,
    ...     workspace=workspace,
    ...     pattern=AllReduceFusionPattern.kARResidualRMSNorm,
    ...     launch_with_pdl=True,
    ...     residual_out=prenorm,
    ...     norm_out=normed,
    ...     residual_in=residual,
    ...     rms_gamma=norm_weight,
    ... )
    """
    # Dispatch based on workspace type
    if isinstance(workspace, TRTLLMAllReduceFusionWorkspace):
        # TensorRT-LLM backend implementation
        if any(
            isinstance(handle, SymmDeviceMemory) and not handle.mapped
            for handle in workspace.mem_handles
        ):
            raise RuntimeError("TRT-LLM symmetric-memory handles are not attached")

        # ---- MOE Reduction pattern ----
        if pattern == AllReduceFusionPattern.kMoEReductionARResidualRMSNorm:
            if moe_reduction_device_num_experts is None:
                raise ValueError(
                    "moe_reduction_device_num_experts is required for "
                    "kMoEReductionARResidualRMSNorm pattern"
                )
            if moe_reduction_scale_input is None:
                raise ValueError(
                    "moe_reduction_scale_input is required for "
                    "kMoEReductionARResidualRMSNorm pattern"
                )
            if moe_reduction_active_experts_token_input is None:
                raise ValueError(
                    "moe_reduction_active_experts_token_input is required for "
                    "kMoEReductionARResidualRMSNorm pattern"
                )
            if moe_reduction_token_input is None:
                raise ValueError(
                    "moe_reduction_token_input is required for "
                    "kMoEReductionARResidualRMSNorm pattern"
                )
            if residual_in is None:
                raise ValueError(
                    "residual_in is required for kMoEReductionARResidualRMSNorm pattern"
                )
            if rms_gamma is None:
                raise ValueError(
                    "rms_gamma is required for kMoEReductionARResidualRMSNorm pattern"
                )

            token_num = residual_in.shape[0]
            hidden_dim = residual_in.shape[-1]

            trtllm_moe_allreduce_fusion(
                world_size=workspace.world_size,
                world_rank=workspace.rank,
                token_num=token_num,
                hidden_dim=hidden_dim,
                workspace_ptrs=workspace.workspace_tensor,
                launch_with_pdl=launch_with_pdl,
                residual_in=residual_in,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                scale_factor=(
                    scale_factor if isinstance(scale_factor, (int, float)) else 1.0
                ),
                moe_reduction_device_num_experts=moe_reduction_device_num_experts,
                moe_reduction_scale_input=moe_reduction_scale_input,
                moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                moe_reduction_token_input=moe_reduction_token_input,
                layout_code=layout_code,  # type: ignore[arg-type]
                moe_allreduce_out=output,
                residual_out=residual_out,
                norm_out=norm_out,
                quant_out=quant_out,
                scale_out=scale_out,
                weight_bias=weight_bias,
            )

            if norm_out is not None:
                return norm_out
            elif quant_out is not None:
                return quant_out
            elif residual_out is not None:
                return residual_out
            elif output is not None:
                return output
            else:
                return residual_in

        # ---- MOE Finalize pattern ----
        if pattern == AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm:
            if expanded_idx_to_permuted_idx is None:
                raise ValueError(
                    "expanded_idx_to_permuted_idx is required for "
                    "kMoEFinalizeARResidualRMSNorm pattern"
                )
            if residual_in is None:
                raise ValueError(
                    "residual_in is required for kMoEFinalizeARResidualRMSNorm pattern"
                )
            if rms_gamma is None:
                raise ValueError(
                    "rms_gamma is required for kMoEFinalizeARResidualRMSNorm pattern"
                )
            if norm_out is None:
                norm_out = torch.empty_like(residual_in)
            if residual_out is None:
                residual_out = torch.empty_like(residual_in)

            trtllm_moe_finalize_allreduce_fusion(
                allreduce_in=input,
                residual_in=residual_in,
                norm_weight=rms_gamma,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                norm_out=norm_out,
                residual_out=residual_out,
                quant_out=quant_out,
                scale_out=scale_out,
                workspace_ptrs=workspace.workspace_tensor,
                launch_with_pdl=launch_with_pdl,
                world_rank=workspace.rank,
                world_size=workspace.world_size,
                eps=rms_eps,
                shared_expert_output=shared_expert_output,
                expert_scale_factor=expert_scale_factor,
                routed_scaling_factor=None,
                weight_bias=weight_bias,
            )

            return norm_out

        # Extract shape from 2D input for the standard TRT-LLM fusion patterns.
        token_num, hidden_dim = input.shape

        if pattern in [
            AllReduceFusionPattern.kARResidualRMSNormPerTokenGroupFP8PackedQuant,
            AllReduceFusionPattern.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant,
        ]:
            if block_quant_group_size is None:
                raise ValueError(
                    f"block_quant_group_size is required for pattern: {pattern}"
                )
            if block_quant_group_size <= 0:
                raise ValueError(
                    f"block_quant_group_size must be > 0, got {block_quant_group_size}"
                )
            if scale_out is None:
                raise ValueError(f"scale_out is required for pattern: {pattern}")
            if hidden_dim % block_quant_group_size != 0:
                raise ValueError(
                    f"hidden_dim must be divisible by block_quant_group_size, got {hidden_dim} and {block_quant_group_size}"
                )

            groups_per_row = hidden_dim // block_quant_group_size
            k_num_packed = (groups_per_row + 3) // 4
            tma_aligned_mn = ((token_num + 3) // 4) * 4
            expected_shape = (token_num, k_num_packed)
            expected_stride = (1, tma_aligned_mn)
            if scale_out.shape != expected_shape:
                raise ValueError(
                    f"scale_out shape must be {expected_shape}, got {tuple(scale_out.shape)}"
                )
            if scale_out.stride() != expected_stride:
                raise ValueError(
                    f"scale_out stride must be {expected_stride}, got {scale_out.stride()}"
                )
            if scale_out.dtype != torch.int32:
                raise ValueError(
                    f"scale_out dtype must be torch.int32, got {scale_out.dtype}"
                )

        dynamic_fp8_patterns = (
            AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant,
            AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant,
        )
        if pattern in dynamic_fp8_patterns:
            if residual_in is None:
                raise ValueError("residual_in is required for dynamic FP8 patterns")
            if residual_out is None:
                raise ValueError("residual_out is required for dynamic FP8 patterns")
            if rms_gamma is None:
                raise ValueError("rms_gamma is required for dynamic FP8 patterns")
            if quant_out is None:
                raise ValueError("quant_out is required for dynamic FP8 patterns")
            if scale_out is None:
                raise ValueError("scale_out is required for dynamic FP8 patterns")
            if quant_out.shape != input.shape:
                raise ValueError(
                    "quant_out must have shape [token_num, hidden_dim] for dynamic FP8 patterns"
                )
            if quant_out.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    "quant_out must have dtype torch.float8_e4m3fn for dynamic FP8 patterns"
                )
            if scale_out.dtype != torch.float32:
                raise ValueError(
                    "scale_out must have dtype torch.float32 for dynamic FP8 patterns"
                )
            if not scale_out.is_contiguous():
                raise ValueError(
                    "scale_out must be contiguous for dynamic FP8 patterns"
                )
            if scale_out.shape != (token_num, 1):
                raise ValueError(
                    "scale_out must have shape [token_num, 1] for dynamic FP8 patterns"
                )
            if (
                pattern == AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant
                and norm_out is None
            ):
                raise ValueError(
                    "norm_out is required for kARResidualRMSNormOutDynamicFP8Quant"
                )

        # Dynamic FP8 patterns do not materialize allreduce_out, so avoid
        # allocating an unused tensor. This keeps the preallocated dynamic path
        # compatible with CUDA Graph capture.
        if output is None and pattern not in dynamic_fp8_patterns:
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
        output_flat = _flatten_checked(output, "output") if output is not None else None
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
            trigger_completion_at_end=trigger_completion_at_end,
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
            weight_bias=weight_bias,
            scale_factor=scale_factor,
            layout_code=layout_code,  # type: ignore[arg-type]
            metadata=workspace.metadata,
            block_quant_group_size=block_quant_group_size,
        )

        # Return the most downstream output (already in 2D shape from input views)
        if norm_out is not None:
            return norm_out
        elif quant_out is not None:
            return quant_out
        else:
            return output

    elif isinstance(workspace, MNNVLAllReduceFusionWorkspace):
        strategy = (
            MNNVLAllreduceFusionStrategy.AUTO
            if use_oneshot is None
            else (
                MNNVLAllreduceFusionStrategy.ONESHOT
                if use_oneshot
                else MNNVLAllreduceFusionStrategy.TWOSHOT
            )
        )
        mnnvl_quant_patterns = {
            AllReduceFusionPattern.kARResidualRMSNormFP8Quant: (
                MNNVLQuantType.FP8,
                False,
            ),
            AllReduceFusionPattern.kARResidualRMSNormFP4Quant: (
                MNNVLQuantType.NVFP4,
                False,
            ),
            AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant: (
                MNNVLQuantType.FP8,
                True,
            ),
            AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant: (
                MNNVLQuantType.NVFP4,
                True,
            ),
            AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant: (
                MNNVLQuantType.DYNAMIC_FP8,
                False,
            ),
            AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant: (
                MNNVLQuantType.DYNAMIC_FP8,
                True,
            ),
        }
        if pattern not in (
            AllReduceFusionPattern.kAllReduce,
            AllReduceFusionPattern.kARResidualRMSNorm,
            *mnnvl_quant_patterns.keys(),
        ):
            raise ValueError(
                f"MNNVL AllReduce+RMS fusion does not support pattern {pattern}. Please try the TRTLLM backend instead."
            )

        mnnvl_layout_code = (
            QuantizationSFLayout.SWIZZLED_128x4 if layout_code is None else layout_code
        )
        if (
            pattern in mnnvl_quant_patterns
            and mnnvl_layout_code == QuantizationSFLayout.SWIZZLED_8x4
        ):
            raise ValueError(
                "MNNVL quantization fusion supports SWIZZLED_128x4 or LINEAR scale layouts, not SWIZZLED_8x4"
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
                strategy=strategy,
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
                strategy=strategy,
                weight_bias=weight_bias,
            )
            return norm_result

        elif pattern in mnnvl_quant_patterns:
            if residual_in is None:
                raise ValueError(
                    "MNNVL quantized AllReduce+RMS fusion requires residual_in"
                )
            if rms_gamma is None:
                raise ValueError(
                    "MNNVL quantized AllReduce+RMS fusion requires rms_gamma"
                )

            quant_type, has_norm_out = mnnvl_quant_patterns[pattern]
            is_dynamic_fp8 = quant_type == MNNVLQuantType.DYNAMIC_FP8
            if is_dynamic_fp8:
                if residual_out is None:
                    raise ValueError(
                        "residual_out is required for MNNVL dynamic FP8 patterns"
                    )
                if quant_out is None:
                    raise ValueError(
                        "quant_out is required for MNNVL dynamic FP8 patterns"
                    )
                if scale_out is None:
                    raise ValueError(
                        "scale_out is required for MNNVL dynamic FP8 patterns"
                    )
                if has_norm_out and norm_out is None:
                    raise ValueError(
                        "norm_out is required for MNNVL dynamic FP8 norm-out pattern"
                    )
            elif has_norm_out and norm_out is None:
                norm_out = torch.empty_like(input)
            if residual_out is None:
                residual_out = torch.empty_like(input)

            quant_result, _, _, _ = trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant(
                input=input,
                residual_in=residual_in,
                gamma=rms_gamma,
                workspace=workspace,
                epsilon=rms_eps,
                output=norm_out if has_norm_out else None,
                residual_out=residual_out,
                quant_out=quant_out,
                scale_out=scale_out,
                output_scale=scale_factor,
                layout_code=mnnvl_layout_code,
                quant_type=quant_type,
                launch_with_pdl=launch_with_pdl,
                strategy=strategy,
                weight_bias=weight_bias,
            )
            return quant_result

        else:
            raise ValueError(f"Unsupported pattern for MNNVL backend: {pattern}")

    else:
        raise TypeError(
            f"Unknown workspace type: {type(workspace)}. "
            f"Expected TRTLLMAllReduceFusionWorkspace or MNNVLAllReduceFusionWorkspace"
        )
