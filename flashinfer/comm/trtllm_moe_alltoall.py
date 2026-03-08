"""
MoE All-to-All Operations (Throughput Backend)

This module provides the throughput-optimized all-to-all backend for MoE expert parallelism,
supporting multiple payloads per collective operation.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import functools

from ..api_logging import flashinfer_api

from .mnnvl import MnnvlMemory, MnnvlConfig
from .mapping import Mapping
from ..jit.comm import gen_moe_alltoall_module
from ..utils import register_custom_op


@dataclass
class _A2AState:
    """Internal state tracking for MoeAlltoAll operations."""

    phase: str = "idle"  # idle | dispatched
    local_num_tokens: Optional[int] = None
    combine_payload_offset: Optional[int] = None


@functools.cache
def get_moe_alltoall_module():
    """Get or build the MOE A2A JIT module."""
    module = gen_moe_alltoall_module().build_and_load()

    @register_custom_op(
        "flashinfer::moe_a2a_initialize",
        mutates_args=("workspace",),
    )
    def moe_a2a_initialize(
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        max_num_tokens: int,
    ):
        return module.moe_a2a_initialize(workspace, ep_rank, ep_size, max_num_tokens)

    @register_custom_op(
        "flashinfer::moe_a2a_dispatch",
        mutates_args=("workspace",),
    )
    def moe_a2a_dispatch(
        token_selected_experts: torch.Tensor,
        input_payloads: list[torch.Tensor],
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        num_experts: int,
    ):
        """
        Dispatch tokens and payloads to expert ranks.

        Args:
            token_selected_experts: [local_num_tokens, top_k] int32 tensor
            input_payloads: List of [local_num_tokens, *] tensors to dispatch
            workspace: [ep_size, size_per_rank] workspace tensor
            metainfo: Metadata tensor from initialize
            runtime_max_tokens_per_rank: Max tokens per rank in this batch
            ep_rank: Current expert parallel rank
            ep_size: Total expert parallel size
            top_k: Number of experts per token
            num_experts: Total number of experts

        Returns:
            recv_offsets: List of offsets for each payload in the workspace
            recv_sizes: List of sizes for each payload in the workspace
            combine_payload_offset: Offset for combine payload region
        """
        return module.moe_a2a_dispatch(
            token_selected_experts,
            input_payloads,
            workspace,
            metainfo,
            runtime_max_tokens_per_rank,
            ep_rank,
            ep_size,
            top_k,
            num_experts,
        )

    @register_custom_op(
        "flashinfer::moe_a2a_combine",
        mutates_args=("workspace",),
    )
    def moe_a2a_combine(
        payload: torch.Tensor,
        local_num_tokens: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        combine_payload_offset: int,
        payload_in_workspace: bool = False,
    ) -> torch.Tensor:
        """
        Combine expert outputs back to originating tokens.

        Args:
            payload: [ep_size, max_tokens, elements_per_token] tensor
            local_num_tokens: Number of tokens on this rank
            workspace: [ep_size, size_per_rank] workspace tensor
            metainfo: Metadata tensor from initialize
            runtime_max_tokens_per_rank: Max tokens per rank in this batch
            ep_rank: Current expert parallel rank
            ep_size: Total expert parallel size
            top_k: Number of experts per token
            combine_payload_offset: Offset from dispatch
            payload_in_workspace: If True, payload is workspace-backed

        Returns:
            output: [local_num_tokens, elements_per_token] tensor
        """
        return module.moe_a2a_combine(
            payload,
            local_num_tokens,
            workspace,
            metainfo,
            runtime_max_tokens_per_rank,
            ep_rank,
            ep_size,
            top_k,
            combine_payload_offset,
            payload_in_workspace,
        )

    @register_custom_op(
        "flashinfer::moe_a2a_sanitize_expert_ids",
        mutates_args=("expert_ids",),
    )
    def moe_a2a_sanitize_expert_ids(
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        ep_rank: int,
        invalid_expert_id: int,
    ):
        return module.moe_a2a_sanitize_expert_ids(
            expert_ids, workspace, metainfo, ep_rank, invalid_expert_id
        )

    @register_custom_op(
        "flashinfer::moe_a2a_get_metainfo_index_pairs",
        mutates_args=[],
    )
    def moe_a2a_get_metainfo_index_pairs():
        """
        Get all metainfo index constants from C++.

        Returns:
            Tuple of (names, values) where names is a list of constant names
            and values is a list of their corresponding integer values
        """
        return module.moe_a2a_get_metainfo_index_pairs()

    @register_custom_op(
        "flashinfer::moe_a2a_get_aux_data_size",
        mutates_args=[],
    )
    def moe_a2a_get_aux_data_size(
        ep_size: int,
        max_num_tokens: int,
    ):
        """
        Get the auxilary datasize per rank for the MoeAlltoAll operation.

        Args:
            ep_size: Total expert parallel size
            max_num_tokens: Maximum number of tokens across all ranks

        Returns:
            aux_data_size: Size of the auxilary data per rank in bytes
        """
        return module.moe_a2a_get_aux_data_size(ep_size, max_num_tokens)

    return SimpleNamespace(
        moe_a2a_initialize=moe_a2a_initialize,
        moe_a2a_dispatch=moe_a2a_dispatch,
        moe_a2a_combine=moe_a2a_combine,
        moe_a2a_sanitize_expert_ids=moe_a2a_sanitize_expert_ids,
        moe_a2a_get_metainfo_index_pairs=moe_a2a_get_metainfo_index_pairs,
        moe_a2a_get_aux_data_size=moe_a2a_get_aux_data_size,
    )


@flashinfer_api
def moe_a2a_initialize(
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    max_num_tokens: int,
):
    return get_moe_alltoall_module().moe_a2a_initialize(
        workspace, ep_rank, ep_size, max_num_tokens
    )


@flashinfer_api
def moe_a2a_wrap_payload_tensor_in_workspace(
    workspace: torch.Tensor,
    leading_shape: list[int],
    slice_start: int,
    slice_end: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Wrap an offset in the workspace into a tensor.

    Args:
        workspace: [ep_size, size_per_rank] or [size_per_rank] workspace tensor
        leading_shape: The leading shape to wrap the tensor with
        slice_start: The start of the slice in the workspace
        slice_end: The end of the slice in the workspace
        dtype: Data type for the output tensor

    Returns:
        tensor: [leading_shape, *] workspace-backed tensor
    """
    if workspace.ndim == 1:
        workspace = workspace.unsqueeze(0)
    workspace_base = workspace.view(dtype=torch.uint8)
    assert workspace.ndim == 2, "workspace must be shape [ep_size, size_per_rank]"
    assert slice_end - slice_start <= workspace_base.shape[1], (
        "slice_end - slice_start must belong to a single rank"
    )
    slice_rank = slice_start // workspace_base.stride(0)
    local_slice_start = slice_start % workspace_base.stride(0)
    slice_length = slice_end - slice_start
    local_slice_end = local_slice_start + slice_length
    assert local_slice_end <= workspace_base.shape[1], (
        "slice must fall within the workspace size per rank"
    )
    result = (
        workspace_base[slice_rank, local_slice_start:local_slice_end]
        .view(dtype=dtype)
        .view(*leading_shape, -1)
    )
    return result


@flashinfer_api
def moe_a2a_dispatch(
    token_selected_experts: torch.Tensor,
    input_payloads: list[torch.Tensor],
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    ep_rank: int,
    ep_size: int,
    top_k: int,
    num_experts: int,
):
    """
    Dispatch tokens and payloads to expert ranks.

    Args:
        token_selected_experts: [local_num_tokens, top_k] int32 tensor
        input_payloads: List of [local_num_tokens, *] tensors to dispatch
        workspace: [ep_size, size_per_rank] workspace tensor
        metainfo: Metadata tensor from initialize
        runtime_max_tokens_per_rank: Max tokens per rank in this batch
        ep_rank: Current expert parallel rank
        ep_size: Total expert parallel size
        top_k: Number of experts per token
        num_experts: Total number of experts

    Returns:
        output_payloads: List of payloads for this rank, backed by data in the workspace
        combine_payload_offset: The offset to place the combine payload in the workspace
    """
    recv_offsets, recv_sizes, combine_payload_offset = (
        get_moe_alltoall_module().moe_a2a_dispatch(
            token_selected_experts,
            input_payloads,
            workspace,
            metainfo,
            runtime_max_tokens_per_rank,
            ep_rank,
            ep_size,
            top_k,
            num_experts,
        )
    )

    output_payloads = []
    for input_payload, offset, size in zip(
        input_payloads, recv_offsets, recv_sizes, strict=True
    ):
        # This uses absolute offsets in the workspace, so skip indexing into the workspace
        output_payloads.append(
            moe_a2a_wrap_payload_tensor_in_workspace(
                workspace,
                [ep_size, runtime_max_tokens_per_rank],
                offset,
                offset + size,
                input_payload.dtype,
            )
        )

    return output_payloads, combine_payload_offset


@flashinfer_api
def moe_a2a_combine(
    payload: torch.Tensor,
    local_num_tokens: int,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    ep_rank: int,
    ep_size: int,
    top_k: int,
    combine_payload_offset: int,
    payload_in_workspace: bool = False,
) -> torch.Tensor:
    return get_moe_alltoall_module().moe_a2a_combine(
        payload,
        local_num_tokens,
        workspace,
        metainfo,
        runtime_max_tokens_per_rank,
        ep_rank,
        ep_size,
        top_k,
        combine_payload_offset,
        payload_in_workspace,
    )


@flashinfer_api
def moe_a2a_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    invalid_expert_id: int,
):
    return get_moe_alltoall_module().moe_a2a_sanitize_expert_ids(
        expert_ids, workspace, metainfo, ep_rank, invalid_expert_id
    )


@flashinfer_api
def moe_a2a_get_workspace_size_per_rank(
    ep_size: int,
    max_num_tokens: int,
    total_dispatch_payload_size_per_token: int,
    combine_payload_size_per_token: int,
):
    """
    Get the workspace size per rank for the MoeAlltoAll operation.

    Args:
        ep_size: Total expert parallel size
        max_num_tokens: Maximum number of tokens across all ranks
        total_dispatch_payload_size_per_token: The size of the payload per token in the dispatch phase. This should be the sum of all payloads.
        combine_payload_size_per_token: The size of the payload per token in the combine phase.

    Returns:
        workspace_size_per_rank: Size of the workspace per rank in bytes
    """
    aux_data_size = get_moe_alltoall_module().moe_a2a_get_aux_data_size(
        ep_size,
        max_num_tokens,
    )

    def pad_up(x, y):
        return ((x + y - 1) // y) * y

    # Pad to 128 bytes to ensure alignment. This matches the implementation of C++ torch OP code.
    return (
        pad_up(aux_data_size, 128)
        + pad_up(ep_size * max_num_tokens * total_dispatch_payload_size_per_token, 128)
        + pad_up(ep_size * max_num_tokens * combine_payload_size_per_token, 128)
    )


class MoeAlltoAll:
    """
    Manages MoE All-to-All operations with proper workspace allocation and synchronization.

    This class provides the throughput-optimized backend that supports multiple payloads
    per collective operation, explicit dispatch/combine phases, and workspace-backed tensors.

    Example:
        >>> moe_a2a = MoeAlltoAll(mapping, max_num_tokens=2048, top_k=2, num_experts=8)
        >>> recv = moe_a2a.dispatch(experts, [hidden, ids, scales], batch_size)
        >>> output = moe_a2a.combine(processed, batch_size)
    """

    # Single shared workspace across the process
    # _WORKSPACE: Optional[dict] = None
    _WORKSPACE_CACHE: dict[tuple[int, int, int, int], dict] = {}

    @classmethod
    def get_workspace(
        cls,
        workspace_size_per_rank: int,
        ep_rank: int,
        ep_size: int,
        max_num_tokens: int,
        mapping: Mapping,
    ) -> dict:
        key = (workspace_size_per_rank, ep_rank, ep_size, max_num_tokens)
        if key in cls._WORKSPACE_CACHE:
            return cls._WORKSPACE_CACHE[key]
        else:
            mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = moe_a2a_initialize(
                workspace,
                ep_rank,
                ep_size,
                max_num_tokens,
            )
            cls._WORKSPACE_CACHE[key] = {
                "workspace_size_per_rank": workspace_size_per_rank,
                "max_num_tokens": max_num_tokens,
                "ep_rank": ep_rank,
                "ep_size": ep_size,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
            }
            return cls._WORKSPACE_CACHE[key]

    @staticmethod
    @flashinfer_api
    def get_moe_workspace_size_per_rank(
        ep_size: int,
        top_k: int,
        max_num_tokens: int,
        hidden_size: int,
        extra_payload_bytes_per_token: int = 0,
    ) -> int:
        """
        Convenience wrapper to calculate the workspace size per rank for the MoeAlltoAll operation. Automatically calculates the size of the dispatch and combine payloads when using default values.
        This allocates space assuming 16-bit float, which may overallocate for quantized models. For a tighter bound, use the base function `moe_a2a_get_workspace_size_per_rank` directly.

        Args:
            ep_size: Total expert parallel size
            top_k: Number of experts per token
            max_num_tokens: Maximum number of tokens across all ranks
            hidden_size: Hidden dimension size
            extra_payload_bytes_per_token: Extra size per token in the payload
        Returns:
            workspace_size_per_rank: Size of the workspace per rank in bytes
        """
        # Default to 16-bit hidden states which should work in all cases.
        element_size = 2

        # Dispatch needs workspace for hidden states, token_selected_experts, token_final_scales
        total_dispatch_payload_size_per_token = (
            int(hidden_size * element_size)  # (Unquantized) token hidden states
            + top_k * 4  # token_selected_experts
            + top_k * 4  # token_final_scales
            + extra_payload_bytes_per_token  # extra payload bytes per token
        )

        # Requires space for hidden states
        combine_payload_size_per_token = int(hidden_size * element_size)

        return moe_a2a_get_workspace_size_per_rank(
            ep_size,
            max_num_tokens,
            total_dispatch_payload_size_per_token,
            combine_payload_size_per_token,
        )

    # Metainfo index constants (loaded dynamically from C++)
    # These offsets allow accessing internal workspace data for testing/debugging
    _METAINFO_INDEX: Optional[dict] = None

    @classmethod
    def _init_constants(cls):
        """Initialize constants from C++ if not already done."""
        if cls._METAINFO_INDEX is None:
            module = get_moe_alltoall_module()
            names, values = module.moe_a2a_get_metainfo_index_pairs()

            # Convert TVM arrays to Python and build dictionary
            # Strip "MOE_A2A_" prefix from names for cleaner API
            cls._METAINFO_INDEX = {}
            for name, value in zip(names, values, strict=True):
                # Convert from "MOE_A2A_SEND_COUNTERS_OFFSET_INDEX" to "SEND_COUNTERS_OFFSET_INDEX"
                clean_name = (
                    name.replace("MOE_A2A_", "")
                    if name.startswith("MOE_A2A_")
                    else name
                )
                cls._METAINFO_INDEX[clean_name] = int(value)

    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        workspace_size_per_rank: int = None,
        hidden_size: int = None,
        mnnvl_config: Optional[MnnvlConfig] = None,
    ):
        """
        Initialize MoeAlltoAll with workspace allocation.

        Args:
            mapping: Mapping object containing rank information
            max_num_tokens: Maximum number of tokens supported
            top_k: Number of experts per token
            num_experts: Total number of experts
            workspace_size_per_rank: Size of workspace per rank in bytes, if None hidden_size must be provided
            hidden_size: Hidden dimension size used when calculating the workspace size, if workspace_size_per_rank is not provided
            mnnvl_config: Used to configure the communication backend for the MNNVL memory object
        """
        # Initialize constants from C++
        self._init_constants()

        if workspace_size_per_rank is None:
            assert hidden_size is not None, (
                "hidden_size must be provided if workspace_size_per_rank is not provided"
            )
            workspace_size_per_rank = self.get_moe_workspace_size_per_rank(
                mapping.moe_ep_size, top_k, max_num_tokens, hidden_size
            )

        # Initialize MNNVL memory system
        MnnvlMemory.initialize()
        if mnnvl_config:
            MnnvlMemory.set_comm_from_config(mapping, mnnvl_config)  # type: ignore[attr-defined]

        self.workspace_size_per_rank = workspace_size_per_rank
        self.max_num_tokens = max_num_tokens
        self.ep_size = mapping.moe_ep_size
        self.ep_rank = mapping.moe_ep_rank
        self.top_k = top_k
        self.num_experts = num_experts

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive int")
        if not isinstance(self.num_experts, int) or self.num_experts <= 0:
            raise ValueError("num_experts must be a positive int")

        # Allocate or reuse workspace
        self._WORKSPACE = self.get_workspace(
            workspace_size_per_rank,
            self.ep_rank,
            self.ep_size,
            self.max_num_tokens,
            mapping,
        )
        # Validate workspace compatibility
        assert self._WORKSPACE["workspace_size_per_rank"] == workspace_size_per_rank, (
            "Workspace size mismatch"
        )
        assert self._WORKSPACE["max_num_tokens"] == self.max_num_tokens, (
            "Max tokens mismatch"
        )
        assert self._WORKSPACE["ep_rank"] == self.ep_rank, "EP rank mismatch"
        assert self._WORKSPACE["ep_size"] == self.ep_size, "EP size mismatch"

        self.mnnvl_mem = self._WORKSPACE["mnnvl_mem"]
        self.workspace = self._WORKSPACE["workspace"]
        self.metainfo = self._WORKSPACE["metainfo"]
        self._state = _A2AState()

    def _reset_workspace(self):
        """Reset the workspace to free up its state. This is mainly used for testing. Use this with caution. This object is no longer usable after this."""
        torch.cuda.synchronize()
        del self._WORKSPACE
        del self._WORKSPACE_CACHE[
            (
                self.workspace_size_per_rank,
                self.ep_rank,
                self.ep_size,
                self.max_num_tokens,
            )
        ]
        self._state.phase = "deleted"

    @flashinfer_api
    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        input_payloads: list[torch.Tensor],
        runtime_max_tokens_per_rank: int,
        invalid_token_expert_id: Optional[int] = None,
        expert_id_payload_index: Optional[int] = None,
    ) -> list[torch.Tensor]:
        """
        Perform MoE all-to-all dispatch operation.

        Args:
            token_selected_experts: [local_num_tokens, top_k] expert indices
            input_payloads: List of [local_num_tokens, *] tensors to dispatch
            runtime_max_tokens_per_rank: Max tokens per rank in this batch
            invalid_token_expert_id: If set, sanitize invalid tokens to this ID
            expert_id_payload_index: Index of expert IDs in input_payloads (required if invalid_token_expert_id is set)

        Returns:
            recv_tensors: List of [ep_size, max_tokens, *] tensors
        """
        assert self._state.phase == "idle", "dispatch called twice without combine"
        assert runtime_max_tokens_per_rank <= self.max_num_tokens, (
            "runtime_max_tokens_per_rank exceeds max_num_tokens"
        )

        recv_tensors, combine_payload_offset = moe_a2a_dispatch(
            token_selected_experts,
            input_payloads,
            self.workspace,
            self.metainfo,
            runtime_max_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            self.num_experts,
        )

        # Update state
        self._state.local_num_tokens = token_selected_experts.size(0)
        self._state.combine_payload_offset = combine_payload_offset
        self._state.phase = "dispatched"

        # Sanitize invalid tokens if requested
        if invalid_token_expert_id is not None:
            assert expert_id_payload_index is not None, (
                "expert_id_payload_index required when invalid_token_expert_id is set"
            )
            recv_expert_ids = recv_tensors[expert_id_payload_index]
            moe_a2a_sanitize_expert_ids(
                recv_expert_ids,
                self.workspace,
                self.metainfo,
                self.ep_rank,
                invalid_token_expert_id,
            )

        return recv_tensors

    @flashinfer_api
    def combine(
        self,
        payload: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        payload_in_workspace: bool = False,
    ) -> torch.Tensor:
        """
        Perform MoE all-to-all combine operation.

        Args:
            payload: [ep_size, max_tokens, elements_per_token] tensor
            runtime_max_tokens_per_rank: Max tokens per rank in this batch
            payload_in_workspace: If True, payload is workspace-backed (skip staging)

        Returns:
            output: [local_num_tokens, elements_per_token] tensor
        """
        assert self._state.phase == "dispatched", (
            "combine called before successful dispatch"
        )
        assert runtime_max_tokens_per_rank <= self.max_num_tokens, (
            "runtime_max_tokens_per_rank exceeds max_num_tokens"
        )

        output = moe_a2a_combine(
            payload,
            self._state.local_num_tokens,
            self.workspace,
            self.metainfo,
            runtime_max_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.top_k,
            self._state.combine_payload_offset,
            payload_in_workspace,
        )

        # Reset state for next round
        self._state = _A2AState()

        return output

    @flashinfer_api
    def get_combine_payload_tensor_in_workspace(
        self,
        runtime_max_tokens_per_rank: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Get combine payload tensor backed by workspace (zero-copy).

        This tensor can be written to directly by expert processing, avoiding
        a staging copy in the combine operation.

        Args:
            runtime_max_tokens_per_rank: Max tokens per rank in this batch
            hidden_size: Hidden dimension size
            dtype: Data type for the tensor

        Returns:
            tensor: [ep_size, max_tokens, hidden_size] workspace-backed tensor
        """
        if self._state.phase != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before successful dispatch"
            )

        element_size = torch.tensor([], dtype=dtype).element_size()
        return moe_a2a_wrap_payload_tensor_in_workspace(
            self.workspace[self.ep_rank, :],
            [self.ep_size, runtime_max_tokens_per_rank],
            self._state.combine_payload_offset,
            self._state.combine_payload_offset
            + self.ep_size * runtime_max_tokens_per_rank * hidden_size * element_size,
            dtype,
        )


__all__ = [
    "MoeAlltoAll",
    "moe_a2a_combine",
    "moe_a2a_dispatch",
    "moe_a2a_get_workspace_size_per_rank",
    "moe_a2a_initialize",
    "moe_a2a_sanitize_expert_ids",
    "moe_a2a_wrap_payload_tensor_in_workspace",
]
