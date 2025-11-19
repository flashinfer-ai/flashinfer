"""
MoE All-to-All Operations (Throughput Backend)

This module provides the throughput-optimized all-to-all backend for MoE expert parallelism,
supporting multiple payloads per collective operation.
"""

# TODO Review

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import functools

from .mnnvl import MnnvlMemory
from .mapping import Mapping
from ..jit.comm import gen_mnnvl_a2a_module
from ..utils import register_custom_op


@dataclass
class _A2AState:
    """Internal state tracking for MoeAlltoAll operations."""

    phase: str = "idle"  # idle | dispatched
    local_num_tokens: Optional[int] = None
    combine_payload_offset: Optional[int] = None


@functools.cache
def get_mnnvl_a2a_module():
    """Get or build the MNNVL A2A JIT module."""
    module = gen_mnnvl_a2a_module().build_and_load()

    @register_custom_op(
        "flashinfer::moe_a2a_initialize",
        mutates_args=[],
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
        mutates_args=[],
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
            recv_tensors: List of [ep_size, max_tokens, *] tensors
            combine_payload_offset: Offset for combine payload region
        """
        recv_offsets, recv_sizes, combine_payload_offset = module.moe_a2a_dispatch(
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
        workspace_base = workspace.flatten().view(dtype=torch.uint8)
        output_payloads = []
        for input_payload, offset, size in zip(
            input_payloads, recv_offsets, recv_sizes, strict=True
        ):
            output_payload = (
                workspace_base[offset : offset + size]
                .view([ep_size, runtime_max_tokens_per_rank, -1])
                .view(dtype=input_payload.dtype)
            )
            output_payloads.append(output_payload)

        return output_payloads, combine_payload_offset

    @register_custom_op(
        "flashinfer::moe_a2a_combine",
        mutates_args=[],
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
        mutates_args=[],
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
        "flashinfer::moe_a2a_get_combine_payload_tensor",
        mutates_args=[],
    )
    def moe_a2a_get_combine_payload_tensor(
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        runtime_max_tokens_per_rank: int,
        combine_payload_offset: int,
        dtype: torch.dtype,
        hidden_size: int,
    ) -> torch.Tensor:
        """
        Get combine payload tensor backed by workspace (zero-copy).

        Args:
            workspace: [ep_size, size_per_rank] workspace tensor
            ep_rank: Current expert parallel rank
            ep_size: Total expert parallel size
            runtime_max_tokens_per_rank: Max tokens per rank in this batch
            combine_payload_offset: Offset from dispatch
            dtype: Data type for the tensor
            hidden_size: Hidden dimension size

        Returns:
            tensor: [ep_size * max_tokens, hidden_size] workspace-backed tensor
        """
        return module.moe_a2a_get_combine_payload_tensor(
            workspace,
            ep_rank,
            ep_size,
            runtime_max_tokens_per_rank,
            combine_payload_offset,
            dtype,
            hidden_size,
        )

    return SimpleNamespace(
        moe_a2a_initialize=moe_a2a_initialize,
        moe_a2a_dispatch=moe_a2a_dispatch,
        moe_a2a_combine=moe_a2a_combine,
        moe_a2a_get_combine_payload_tensor=moe_a2a_get_combine_payload_tensor,
    )


def moe_a2a_initialize(
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    max_num_tokens: int,
):
    return get_mnnvl_a2a_module().moe_a2a_initialize(
        workspace, ep_rank, ep_size, max_num_tokens
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
    return get_mnnvl_a2a_module().moe_a2a_dispatch(
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
    return get_mnnvl_a2a_module().moe_a2a_combine(
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


def moe_a2a_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    invalid_expert_id: int,
):
    return get_mnnvl_a2a_module().moe_a2a_sanitize_expert_ids(
        expert_ids, workspace, metainfo, ep_rank, invalid_expert_id
    )


def moe_a2a_get_combine_payload_tensor(
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    runtime_max_tokens_per_rank: int,
    combine_payload_offset: int,
    dtype: torch.dtype,
    hidden_size: int,
) -> torch.Tensor:
    return get_mnnvl_a2a_module().moe_a2a_get_combine_payload_tensor(
        workspace,
        ep_rank,
        ep_size,
        runtime_max_tokens_per_rank,
        combine_payload_offset,
        dtype,
        hidden_size,
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
    _WORKSPACE: Optional[dict] = None

    def __init__(
        self,
        mapping: Mapping,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        workspace_size_per_rank: int = 512 * 1024 * 1024,
    ):
        """
        Initialize MoeAlltoAll with workspace allocation.

        Args:
            mapping: Mapping object containing rank information
            max_num_tokens: Maximum number of tokens supported
            top_k: Number of experts per token
            num_experts: Total number of experts
            workspace_size_per_rank: Size of workspace per rank in bytes (default: 512MB)
        """
        # Initialize MNNVL memory system
        MnnvlMemory.initialize()

        self.workspace_size_per_rank = workspace_size_per_rank
        self.max_num_tokens = max_num_tokens
        self.ep_size = mapping.tp_size
        self.ep_rank = mapping.tp_rank
        self.top_k = top_k
        self.num_experts = num_experts

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive int")
        if not isinstance(self.num_experts, int) or self.num_experts <= 0:
            raise ValueError("num_experts must be a positive int")

        # Allocate or reuse workspace
        if self._WORKSPACE is None:
            mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = moe_a2a_initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                self.max_num_tokens,
            )
            MoeAlltoAll._WORKSPACE = {
                "workspace_size_per_rank": workspace_size_per_rank,
                "max_num_tokens": self.max_num_tokens,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
            }
        else:
            # Validate workspace compatibility
            assert (
                self._WORKSPACE["workspace_size_per_rank"] == workspace_size_per_rank
            ), "Workspace size mismatch"
            assert self._WORKSPACE["max_num_tokens"] == self.max_num_tokens, (
                "Max tokens mismatch"
            )
            assert self._WORKSPACE["ep_rank"] == self.ep_rank, "EP rank mismatch"
            assert self._WORKSPACE["ep_size"] == self.ep_size, "EP size mismatch"

        self.mnnvl_mem = self._WORKSPACE["mnnvl_mem"]
        self.workspace = self._WORKSPACE["workspace"]
        self.metainfo = self._WORKSPACE["metainfo"]
        self._state = _A2AState()

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
            tensor: [ep_size * max_tokens, hidden_size] workspace-backed tensor
        """
        if self._state.phase != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before successful dispatch"
            )

        return moe_a2a_get_combine_payload_tensor(
            self.workspace,
            self.ep_rank,
            self.ep_size,
            runtime_max_tokens_per_rank,
            self._state.combine_payload_offset,
            dtype,
            hidden_size,
        )


__all__ = [
    "MoeAlltoAll",
    "moe_a2a_initialize",
    "moe_a2a_dispatch",
    "moe_a2a_combine",
    "moe_a2a_sanitize_expert_ids",
    "moe_a2a_get_combine_payload_tensor",
]
