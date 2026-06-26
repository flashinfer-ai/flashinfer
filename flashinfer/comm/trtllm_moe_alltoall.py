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
from ..tllm_enums import SfLayout


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
        output_dtype: Optional[torch.dtype] = None,
        output_scales: Optional[torch.Tensor] = None,
        sf_layout: SfLayout = SfLayout.layout_linear,
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
            output_dtype: Optional output data type
                currently supports [torch.bfloat16, torch.float8_e4m3fn]
            output_scales: Optional output scale tensor for quantized outputs
                currently support ue8m0 (packed in torch.uint8) with vector size of 32
            sf_layout: Output swizzle layout
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
            output_dtype,
            output_scales,
            sf_layout.value,
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
    r"""Initialize the MoE all-to-all workspace and return a metainfo tensor.

    The metainfo tensor encodes per-rank offsets and bookkeeping required by
    :func:`moe_a2a_dispatch` and :func:`moe_a2a_combine`; it must be passed
    back into those routines for the same workspace.  ``moe_a2a_initialize``
    is idempotent and must be called once per workspace allocation before any
    dispatch/combine.

    Parameters
    ----------
    workspace : torch.Tensor
        ``[ep_size, size_per_rank]`` shared workspace tensor.
    ep_rank : int
        Current expert-parallel rank.
    ep_size : int
        Total expert-parallel world size.
    max_num_tokens : int
        Maximum number of tokens any rank may dispatch in a single call;
        used to size the metainfo allocation.

    Returns
    -------
    torch.Tensor
        Metainfo tensor opaque to callers; pass it to subsequent
        ``moe_a2a_*`` calls.
    """
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
    r"""Wrap a slice of the shared workspace as a typed tensor view.

    Parameters
    ----------
    workspace : torch.Tensor
        ``[ep_size, size_per_rank]`` (or ``[size_per_rank]``) workspace
        tensor.
    leading_shape : list[int]
        Leading shape of the resulting view.  The trailing dimension is
        inferred from ``slice_end - slice_start`` and ``dtype``.
    slice_start : int
        Start offset (in bytes from the beginning of the workspace) of the
        slice to wrap.
    slice_end : int
        End offset (in bytes) of the slice.  Must lie within a single rank.
    dtype : torch.dtype
        Element dtype of the resulting view.

    Returns
    -------
    torch.Tensor
        A workspace-backed tensor of shape ``leading_shape + [-1]``.
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
    r"""Dispatch tokens and payloads to their target expert ranks.

    Parameters
    ----------
    token_selected_experts : torch.Tensor
        ``[local_num_tokens, top_k]`` ``int32`` tensor of expert assignments.
    input_payloads : list[torch.Tensor]
        Per-token payload tensors, each shaped ``[local_num_tokens, *]``.
    workspace : torch.Tensor
        ``[ep_size, size_per_rank]`` shared workspace.
    metainfo : torch.Tensor
        Metainfo tensor returned by :func:`moe_a2a_initialize`.
    runtime_max_tokens_per_rank : int
        Maximum tokens per rank for this batch (must be ``<=`` the
        ``max_num_tokens`` used at initialize time).
    ep_rank : int
        Current expert-parallel rank.
    ep_size : int
        Total expert-parallel world size.
    top_k : int
        Number of experts assigned per token.
    num_experts : int
        Total number of experts.

    Returns
    -------
    Tuple[list[torch.Tensor], int]
        ``(output_payloads, combine_payload_offset)``.  ``output_payloads``
        is a list of workspace-backed views, one per ``input_payloads``
        entry, that contains the data routed to this rank.  ``combine_payload_offset``
        is the workspace offset reserved for the matching
        :func:`moe_a2a_combine` call.
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
    output_dtype: Optional[torch.dtype] = None,
    output_scales: Optional[torch.Tensor] = None,
    sf_layout: SfLayout = SfLayout.layout_linear,
) -> torch.Tensor:
    r"""Combine per-expert outputs back to the originating ranks.

    Inverse of :func:`moe_a2a_dispatch`: scatters the rank-local expert
    output rows back to the ranks that supplied the original tokens.

    Parameters
    ----------
    payload : torch.Tensor
        Output payload to send back to the source ranks.  Shape
        ``[ep_size, runtime_max_tokens_per_rank, *]`` regardless of
        ``payload_in_workspace``: in both cases the payload holds the
        per-expert-rank outputs to be combined back to the source ranks.
        Only the backing memory differs (caller-supplied vs. workspace-backed
        view produced by :meth:`MoeAlltoAll.get_combine_payload_tensor_in_workspace`).
    local_num_tokens : int
        Number of tokens originally dispatched from this rank.
    workspace : torch.Tensor
        Shared workspace tensor (same one passed to dispatch).
    metainfo : torch.Tensor
        Metainfo tensor returned by :func:`moe_a2a_initialize`.
    runtime_max_tokens_per_rank : int
        Same value passed to :func:`moe_a2a_dispatch`.
    ep_rank : int
        Current expert-parallel rank.
    ep_size : int
        Total expert-parallel world size.
    top_k : int
        Number of experts assigned per token.
    combine_payload_offset : int
        Offset returned by :func:`moe_a2a_dispatch`.
    payload_in_workspace : bool
        ``True`` if ``payload`` is already a workspace-backed view (skips
        the staging copy).  Defaults to ``False``.
    output_dtype : Optional[torch.dtype]
        Optional output data type.  Currently supports ``torch.bfloat16``
        and ``torch.float8_e4m3fn``.
    output_scales : Optional[torch.Tensor]
        Optional output scale tensor for quantized outputs.  Currently
        supports UE8M0 (packed in ``torch.uint8``) with vector size 32.
    sf_layout : SfLayout
        Output swizzle layout.  Defaults to ``SfLayout.layout_linear``.

    Returns
    -------
    torch.Tensor
        ``[local_num_tokens, *]`` tensor with the combined outputs.
    """
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
        output_dtype,
        output_scales,
        sf_layout,
    )


@flashinfer_api
def moe_a2a_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    invalid_expert_id: int,
):
    r"""Replace expert IDs not owned by this rank with ``invalid_expert_id``.

    Parameters
    ----------
    expert_ids : torch.Tensor
        ``[local_num_tokens, top_k]`` ``int32`` tensor of expert assignments
        (mutated in place).
    workspace : torch.Tensor
        Shared workspace tensor.
    metainfo : torch.Tensor
        Metainfo tensor returned by :func:`moe_a2a_initialize`.
    ep_rank : int
        Current expert-parallel rank.
    invalid_expert_id : int
        Value to write where the original expert lies outside this rank's
        local range.
    """
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
    r"""Compute the per-rank workspace size for the MoE all-to-all primitive.

    Parameters
    ----------
    ep_size : int
        Total expert-parallel world size.
    max_num_tokens : int
        Maximum number of tokens across all ranks.
    total_dispatch_payload_size_per_token : int
        Sum (in bytes) of all per-token payloads sent during the dispatch
        phase.
    combine_payload_size_per_token : int
        Per-token payload size (in bytes) sent back during the combine
        phase.

    Returns
    -------
    int
        Required workspace size per rank, in bytes.
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
        r"""Compute the per-rank workspace size for the MoE all-to-all primitive.

        Convenience wrapper around :func:`moe_a2a_get_workspace_size_per_rank`
        that derives the dispatch / combine payload sizes from
        ``hidden_size`` and ``top_k`` assuming 16-bit hidden states.  For a
        tighter bound on quantized models use
        :func:`moe_a2a_get_workspace_size_per_rank` directly.

        Parameters
        ----------
        ep_size : int
            Total expert-parallel world size.
        top_k : int
            Number of experts assigned per token.
        max_num_tokens : int
            Maximum number of tokens across all ranks.
        hidden_size : int
            Hidden dimension size.
        extra_payload_bytes_per_token : int
            Extra payload bytes per token to reserve (e.g. for quantization
            scales).  Defaults to ``0``.

        Returns
        -------
        int
            Required workspace size per rank, in bytes.
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
        r"""Initialize :class:`MoeAlltoAll` and allocate the shared workspace.

        Parameters
        ----------
        mapping : Mapping
            Mapping object describing the parallel layout (must expose
            ``moe_ep_rank`` and ``moe_ep_size``).
        max_num_tokens : int
            Maximum number of tokens this rank will dispatch in any single
            call.
        top_k : int
            Number of experts assigned per token.
        num_experts : int
            Total number of experts (across all ranks).
        workspace_size_per_rank : int, optional
            Pre-computed workspace size in bytes per rank.  When ``None``,
            ``hidden_size`` must be provided and the workspace is sized via
            :meth:`get_moe_workspace_size_per_rank`.
        hidden_size : int, optional
            Hidden dimension size, used to derive
            ``workspace_size_per_rank`` when the latter is omitted.
        mnnvl_config : MnnvlConfig, optional
            Optional configuration for the underlying MNNVL communication
            backend.
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
        r"""Run the MoE all-to-all dispatch phase.

        Parameters
        ----------
        token_selected_experts : torch.Tensor
            ``[local_num_tokens, top_k]`` ``int32`` tensor of expert
            assignments.
        input_payloads : list[torch.Tensor]
            Per-token payload tensors, each shaped ``[local_num_tokens, *]``.
        runtime_max_tokens_per_rank : int
            Maximum tokens per rank in this batch.  Must be ``<=``
            ``max_num_tokens`` used at construction.
        invalid_token_expert_id : int, optional
            If supplied, expert IDs not owned by the current rank are
            rewritten to this value.  Requires ``expert_id_payload_index``.
        expert_id_payload_index : int, optional
            Index into ``input_payloads`` that holds the expert IDs to
            sanitize.  Required when ``invalid_token_expert_id`` is set.

        Returns
        -------
        list[torch.Tensor]
            Workspace-backed receive tensors, one per ``input_payloads``
            entry, each shaped ``[ep_size, runtime_max_tokens_per_rank, *]``.
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
        output_dtype: Optional[torch.dtype] = None,
        output_scales: Optional[torch.Tensor] = None,
        sf_layout: SfLayout = SfLayout.layout_linear,
    ) -> torch.Tensor:
        r"""Run the MoE all-to-all combine phase.

        Parameters
        ----------
        payload : torch.Tensor
            ``[ep_size, runtime_max_tokens_per_rank, elements_per_token]``
            output payload to scatter back to source ranks.
        runtime_max_tokens_per_rank : int
            Maximum tokens per rank in this batch (same value passed to
            :meth:`dispatch`).
        payload_in_workspace : bool
            ``True`` if ``payload`` is already a workspace-backed view (skips
            the staging copy).  Defaults to ``False``.
        output_dtype : Optional[torch.dtype]
            Optional output data type.  Currently supports ``torch.bfloat16``
            and ``torch.float8_e4m3fn``.
        output_scales : Optional[torch.Tensor]
            Optional output scale tensor for quantized outputs.  Currently
            supports UE8M0 (packed in ``torch.uint8``) with vector size 32.
        sf_layout : SfLayout
            Output swizzle layout.  Defaults to ``SfLayout.layout_linear``.

        Returns
        -------
        torch.Tensor
            ``[local_num_tokens, elements_per_token]`` combined tensor.
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
            output_dtype,
            output_scales,
            sf_layout,
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
        r"""Return a workspace-backed view to use as the combine payload.

        Zero-copy variant of :meth:`combine`: experts can write directly
        into the returned tensor and call :meth:`combine` with
        ``payload_in_workspace=True``.  Must be called after a successful
        :meth:`dispatch` and before :meth:`combine`.

        Parameters
        ----------
        runtime_max_tokens_per_rank : int
            Maximum tokens per rank in this batch.
        hidden_size : int
            Hidden dimension size.
        dtype : torch.dtype
            Element dtype of the resulting view.

        Returns
        -------
        torch.Tensor
            ``[ep_size, runtime_max_tokens_per_rank, hidden_size]``
            workspace-backed tensor.

        Raises
        ------
        RuntimeError
            If called before a successful :meth:`dispatch`.
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
