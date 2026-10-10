"""
MoE All-to-All Operations (Throughput Backend)

This module provides the throughput-optimized all-to-all backend for MoE expert parallelism,
supporting multiple payloads per collective operation.

The TensorRT-LLM implementation remains the default on every architecture.
Pass ``backend="cake"`` explicitly to use the generated Blackwell implementation.
All ranks in a collective must select the same backend, and use it consistently
for workspace sizing, initialization, dispatch, combine, and sanitization.
:class:`flashinfer.moe_ep.CakeAlltoAll` drives these ops with ``backend="cake"``;
:class:`flashinfer.moe_ep.NVLinkOneSidedAlltoAll` runs newer one-sided kernels.
"""

from types import SimpleNamespace
from typing import Literal, Optional, Sequence

import torch
import functools

from ..api_logging import flashinfer_api

from ..jit.comm import gen_moe_alltoall_module
from ..utils import register_custom_op, device_support_pdl
from ..tllm_enums import SfLayout

# Number of uint64 words in the active_rank_mask ABI (must match kRankMaskWords in
# csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h, which is
# derived from kMaxRanks there). A single word covers up to 64 ranks.
MOE_A2A_RANK_MASK_WORDS = 1
MoeAlltoAllTarget = Literal["legacy", "sm100a", "sm103a"]
MoeAlltoAllBackend = Literal["trtllm", "cake"]


def moe_a2a_active_rank_mask(active_ranks: Sequence[int], ep_size: int) -> torch.Tensor:
    r"""Build a CPU ``uint64`` active-rank bitmask for :func:`moe_a2a_dispatch` /
    :func:`moe_a2a_combine`'s ``active_rank_mask`` argument.

    Parameters
    ----------
    active_ranks : Sequence[int]
        Ranks that are alive and should participate in the collective, e.g. ``[0, 1, 3]``
        for a 4-rank job where rank 2 has failed.  Also accepts a 1D ``torch.Tensor`` of
        rank indices.
    ep_size : int
        Total expert-parallel world size (all ranks outside ``[0, ep_size)`` are ignored).

    Returns
    -------
    torch.Tensor
        ``[MOE_A2A_RANK_MASK_WORDS]`` ``uint64`` CPU tensor with bit ``i`` set for each
        active rank ``i``.
    """
    mask = 0
    for rank in active_ranks:
        rank = int(rank)
        assert 0 <= rank < ep_size, f"rank {rank} out of range [0, {ep_size})"
        mask |= 1 << rank
    words = [
        (mask >> (64 * word)) & 0xFFFFFFFFFFFFFFFF
        for word in range(MOE_A2A_RANK_MASK_WORDS)
    ]
    return torch.tensor(words, dtype=torch.uint64, device="cpu")


@functools.cache
def _moe_alltoall_target(device_index: int) -> MoeAlltoAllTarget:
    capability = torch.cuda.get_device_capability(device_index)
    if capability == (10, 0):
        return "sm100a"
    if capability == (10, 3):
        return "sm103a"
    raise ValueError(
        f'backend="cake" requires compute capability 10.0 or 10.3, got {capability}'
    )


@functools.cache
def _get_moe_alltoall_module_for_target(target: MoeAlltoAllTarget):
    """Build or load the legacy or exact-architecture all-to-all module."""
    module = gen_moe_alltoall_module(target).build_and_load()
    # Keep legacy names stable, but do not let loading an opt-in module replace
    # the default backend's custom-op implementations (or captured graphs).
    op_suffix = "" if target == "legacy" else f"_{target}"

    @register_custom_op(
        f"flashinfer::moe_a2a_initialize{op_suffix}",
        mutates_args=("workspace",),
    )
    def moe_a2a_initialize(
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        max_num_tokens: int,
        eplb_stats_num_experts: int = 0,
    ):
        return module.moe_a2a_initialize(
            workspace, ep_rank, ep_size, max_num_tokens, eplb_stats_num_experts
        )

    @register_custom_op(
        f"flashinfer::moe_a2a_dispatch{op_suffix}",
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
        enable_pdl: bool,
        eplb_local_stats: Optional[torch.Tensor] = None,
        enable_rank_mask: bool = False,
        active_rank_mask: Optional[torch.Tensor] = None,
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
            enable_pdl: Whether to use programmatic dependent launch
            eplb_local_stats: Optional [eplb_stats_num_experts] int32 tensor of
                this rank's local EPLB statistics to all-gather during dispatch
            enable_rank_mask: Whether to instantiate the kernel variant that checks
                active_rank_mask at all. False (default) compiles out every rank-mask
                check for the common no-fault-tolerance case and requires
                active_rank_mask to be omitted.
            active_rank_mask: Optional CPU uint64 tensor of shape [MOE_A2A_RANK_MASK_WORDS]
                (see :func:`moe_a2a_active_rank_mask`). Bit i set means rank i is alive and
                participates in this collective; tokens routed to a masked-off rank are
                dropped. Requires enable_rank_mask=True.

        Returns:
            recv_offsets: List of offsets for each payload in the workspace
            recv_sizes: List of sizes for each payload in the workspace
            combine_payload_offset: Offset for combine payload region
            eplb_gathered_stats_offset: Offset for the gathered EPLB stats
                region, or -1 when EPLB is disabled
            eplb_stats_num_experts: Number of experts in the EPLB stats, or 0
                when EPLB is disabled
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
            enable_pdl,
            eplb_local_stats,
            enable_rank_mask,
            active_rank_mask,
        )

    @register_custom_op(
        f"flashinfer::moe_a2a_combine{op_suffix}",
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
        output_scalar_scale: float = 1.0,
        sf_layout: Optional[SfLayout] = None,
        use_low_precision: bool = False,
        enable_pdl: bool = True,
        enable_rank_mask: bool = False,
        active_rank_mask: Optional[torch.Tensor] = None,
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
            output_dtype: Optional output data type. Supported types:
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.uint8 (packed fp4)
            output_scales: Optional output scale tensor for quantized outputs. Support types:
                torch.uint8 (packed ue8m0), vector size of 32
                torch.float8_e4m3fn, vector size of 16
            output_scalar_scale: Per-tensor global scale applied before FP4 block scaling
                (NVFP4 SFScaleVal). Defaults to 1.0; ignored by MXFP8/MXFP4 paths.
            sf_layout: Output swizzle layout. Defaults to linear.
            use_low_precision: If True, quantize payload to FP8 before combine
            enable_pdl: Whether to use programmatic dependent launch
            enable_rank_mask: Whether to instantiate the kernel variant that checks
                active_rank_mask at all. False (default) compiles out the rank-mask checks
                in peer synchronization for the common no-fault-tolerance case and requires
                active_rank_mask to be omitted.
            active_rank_mask: Optional CPU uint64 tensor of shape [MOE_A2A_RANK_MASK_WORDS]
                (see :func:`moe_a2a_active_rank_mask`). Should match the mask passed to the
                corresponding :func:`moe_a2a_dispatch` call (or be omitted from both). Requires
                enable_rank_mask=True.
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
            output_scalar_scale,
            sf_layout.value if sf_layout is not None else SfLayout.layout_linear.value,
            use_low_precision,
            enable_pdl,
            enable_rank_mask,
            active_rank_mask,
        )

    @register_custom_op(
        f"flashinfer::moe_a2a_combine_into{op_suffix}",
        mutates_args=("workspace", "output"),
    )
    def moe_a2a_combine_into(
        payload: torch.Tensor,
        local_num_tokens: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        combine_payload_offset: int,
        payload_in_workspace: bool,
        output_dtype: Optional[torch.dtype],
        output_scales: Optional[torch.Tensor],
        output_scalar_scale: float,
        sf_layout: Optional[SfLayout],
        use_low_precision: bool,
        enable_pdl: bool,
        enable_rank_mask: bool,
        active_rank_mask: Optional[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        module.moe_a2a_combine_into(
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
            output_scalar_scale,
            sf_layout.value if sf_layout is not None else SfLayout.layout_linear.value,
            use_low_precision,
            enable_pdl,
            enable_rank_mask,
            active_rank_mask,
            output,
        )

    @register_custom_op(
        f"flashinfer::moe_a2a_sanitize_expert_ids{op_suffix}",
        mutates_args=("expert_ids",),
    )
    def moe_a2a_sanitize_expert_ids(
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        ep_rank: int,
        invalid_expert_id: int,
        enable_pdl: bool,
    ):
        return module.moe_a2a_sanitize_expert_ids(
            expert_ids, workspace, metainfo, ep_rank, invalid_expert_id, enable_pdl
        )

    @register_custom_op(
        f"flashinfer::moe_a2a_get_metainfo_index_pairs{op_suffix}",
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
        f"flashinfer::moe_a2a_get_aux_data_size{op_suffix}",
        mutates_args=[],
    )
    def moe_a2a_get_aux_data_size(
        ep_size: int,
        max_num_tokens: int,
        eplb_stats_num_experts: int = 0,
    ):
        """
        Get the auxilary datasize per rank of the MoE all-to-all workspace.

        Args:
            ep_size: Total expert parallel size
            max_num_tokens: Maximum number of tokens across all ranks
            eplb_stats_num_experts: Number of experts reserved for EPLB stats
                (0 disables the EPLB region)

        Returns:
            aux_data_size: Size of the auxilary data per rank in bytes
        """
        return module.moe_a2a_get_aux_data_size(
            ep_size, max_num_tokens, eplb_stats_num_experts
        )

    return SimpleNamespace(
        moe_a2a_initialize=moe_a2a_initialize,
        moe_a2a_dispatch=moe_a2a_dispatch,
        moe_a2a_combine=moe_a2a_combine,
        moe_a2a_combine_into=moe_a2a_combine_into,
        moe_a2a_sanitize_expert_ids=moe_a2a_sanitize_expert_ids,
        moe_a2a_get_metainfo_index_pairs=moe_a2a_get_metainfo_index_pairs,
        moe_a2a_get_aux_data_size=moe_a2a_get_aux_data_size,
    )


def get_moe_alltoall_module(backend: MoeAlltoAllBackend = "trtllm"):
    """Use TRT-LLM by default; exact-architecture kernels require explicit opt-in."""
    if backend == "trtllm":
        return _get_moe_alltoall_module_for_target("legacy")
    if backend != "cake":
        raise ValueError(
            f"Unknown MoE all-to-all backend {backend!r}; expected 'trtllm' or 'cake'"
        )
    device_index = torch.cuda.current_device()
    return _get_moe_alltoall_module_for_target(_moe_alltoall_target(device_index))


def _clear_moe_alltoall_module_cache() -> None:
    _moe_alltoall_target.cache_clear()
    _get_moe_alltoall_module_for_target.cache_clear()


get_moe_alltoall_module.cache_clear = (  # type: ignore[attr-defined]
    _clear_moe_alltoall_module_cache
)


@flashinfer_api
def moe_a2a_initialize(
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    max_num_tokens: int,
    eplb_stats_num_experts: int = 0,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
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
    eplb_stats_num_experts : int
        Number of experts to reserve space for in the EPLB gathered-stats
        region.  ``0`` (default) disables the EPLB region; when non-zero it
        must match the ``eplb_local_stats`` length passed to
        :func:`moe_a2a_dispatch`.
    backend : {"trtllm", "cake"}
        Defaults to ``"trtllm"`` on all architectures. ``"cake"`` explicitly
        opts in to generated Blackwell kernels (CC 10.0/10.3). Use the same
        backend for all operations on this workspace and on all ranks.

    Returns
    -------
    torch.Tensor
        Metainfo tensor opaque to callers; pass it to subsequent
        ``moe_a2a_*`` calls.
    """
    return get_moe_alltoall_module(backend).moe_a2a_initialize(
        workspace, ep_rank, ep_size, max_num_tokens, eplb_stats_num_experts
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
    enable_pdl: Optional[bool] = None,
    eplb_local_stats: Optional[torch.Tensor] = None,
    enable_rank_mask: bool = False,
    active_rank_mask: Optional[torch.Tensor] = None,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
    recv_view_cache: Optional[dict] = None,
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
    enable_pdl : Optional[bool]
        Whether to use programmatic dependent launch.  ``None`` auto-detects
        from the device.
    eplb_local_stats : Optional[torch.Tensor]
        Optional ``[eplb_stats_num_experts]`` ``int32`` tensor of this rank's
        local EPLB statistics.  When provided, the dispatch all-gathers it
        across ranks and returns the result as ``eplb_gathered_stats``.  The
        length must match the ``eplb_stats_num_experts`` passed to
        :func:`moe_a2a_initialize`.
    enable_rank_mask : bool
        Whether to instantiate the kernel variant that checks ``active_rank_mask`` at
        all.  ``False`` (default) compiles out every rank-mask check for the common
        no-fault-tolerance case and requires ``active_rank_mask`` to be omitted.
    active_rank_mask : Optional[torch.Tensor]
        Optional CPU ``uint64`` tensor of shape ``[MOE_A2A_RANK_MASK_WORDS]`` (see
        :func:`moe_a2a_active_rank_mask`).  Bit ``i`` set means rank ``i`` is alive and
        participates in this collective; tokens routed to a masked-off rank are dropped
        instead of hanging the collective.  Requires ``enable_rank_mask=True``; the local
        ``ep_rank``'s own bit must always be set.
        Masking a peer does not advance that peer's transport epoch. Before a
        skipped peer rejoins, all ranks must quiesce and coordinate workspace
        reinitialization; merely restoring its mask bit is insufficient.
    backend : {"trtllm", "cake"}
        Defaults to ``"trtllm"``. Pass ``"cake"`` to opt in on CC 10.0/10.3;
        must match workspace initialization, combine, and all peer ranks.
    recv_view_cache : dict, optional
        Opaque cache of receive views. Pass an empty dictionary to enable
        caching; ``None`` (default) disables it. The cache retains a reference
        to ``workspace`` and is cleared when a different workspace tensor is
        passed. Keep the backing allocation alive and do not change workspace
        or cached-view shape, strides, or storage in place. Clear the dictionary
        to release its references; otherwise, leave its contents unchanged.

    Returns
    -------
    Tuple[list[torch.Tensor], int, Optional[torch.Tensor]]
        ``(output_payloads, combine_payload_offset, eplb_gathered_stats)``.
        ``output_payloads`` is a list of workspace-backed views, one per
        ``input_payloads`` entry, that contains the data routed to this rank.
        ``combine_payload_offset`` is the workspace offset reserved for the
        matching :func:`moe_a2a_combine` call.  ``eplb_gathered_stats`` is a
        workspace-backed ``[ep_size, eplb_stats_num_experts]`` ``int32`` view
        (row ``r`` holds rank ``r``'s ``eplb_local_stats``) when
        ``eplb_local_stats`` was provided, else ``None``.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(token_selected_experts.device)
    (
        recv_offsets,
        recv_sizes,
        combine_payload_offset,
        eplb_gathered_stats_offset,
        eplb_stats_num_experts,
    ) = get_moe_alltoall_module(backend).moe_a2a_dispatch(
        token_selected_experts,
        input_payloads,
        workspace,
        metainfo,
        runtime_max_tokens_per_rank,
        ep_rank,
        ep_size,
        top_k,
        num_experts,
        enable_pdl,
        eplb_local_stats,
        enable_rank_mask,
        active_rank_mask,
    )

    # Bind once per dispatch without pointer/device queries in the hot path.
    if (
        recv_view_cache is not None
        and recv_view_cache.get("_workspace") is not workspace
    ):
        recv_view_cache.clear()
        recv_view_cache["_workspace"] = workspace

    output_payloads = []
    for input_payload, offset, size in zip(
        input_payloads, recv_offsets, recv_sizes, strict=True
    ):
        # This uses absolute offsets in the workspace, so skip indexing into the workspace
        # Reuse views to avoid four tensor constructions per payload. Cached views
        # still observe fresh writes to the workspace bound above.
        # ``int()`` guards the key rather than fixing an observed miss: tvm-ffi
        # materializes an ``Array<int64_t>`` element as a real Python ``int``
        # today, so these already hash by value. Coercing keeps that property
        # local to this function instead of resting on an FFI implementation
        # detail -- an int-like that hashed by identity would miss on every
        # lookup, silently disabling the cache while the dict still grew. Same
        # idiom as the ``int()`` on FFI scalars in ``_init_constants`` below.
        key = (
            ep_size,
            runtime_max_tokens_per_rank,
            int(offset),
            int(size),
            input_payload.dtype,
        )
        payload_view = None if recv_view_cache is None else recv_view_cache.get(key)
        if payload_view is None:
            payload_view = moe_a2a_wrap_payload_tensor_in_workspace(
                workspace,
                [ep_size, runtime_max_tokens_per_rank],
                offset,
                offset + size,
                input_payload.dtype,
            )
            if recv_view_cache is not None:
                recv_view_cache[key] = payload_view
        output_payloads.append(payload_view)

    eplb_gathered_stats = None
    if eplb_gathered_stats_offset >= 0:
        eplb_gathered_stats = moe_a2a_wrap_payload_tensor_in_workspace(
            workspace,
            [ep_size],
            eplb_gathered_stats_offset,
            eplb_gathered_stats_offset + ep_size * eplb_stats_num_experts * 4,
            torch.int32,
        )

    return output_payloads, combine_payload_offset, eplb_gathered_stats


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
    output_scalar_scale: float = 1.0,
    sf_layout: SfLayout = SfLayout.layout_linear,
    output: Optional[torch.Tensor] = None,
    *,
    use_low_precision: bool = False,
    enable_pdl: Optional[bool] = None,
    enable_rank_mask: bool = False,
    active_rank_mask: Optional[torch.Tensor] = None,
    backend: MoeAlltoAllBackend = "trtllm",
) -> torch.Tensor:
    r"""Combine per-expert outputs back to the originating ranks.

    Inverse of :func:`moe_a2a_dispatch`: scatters the rank-local expert
    output rows back to the ranks that supplied the original tokens.

    ``backend="trtllm"`` is the default, including on Blackwell. Opt in with
    ``backend="cake"`` only for workspaces initialized and dispatched with
    that backend; all peer ranks must use the same selection.

    Parameters
    ----------
    payload : torch.Tensor
        Output payload to send back to the source ranks.  Shape
        ``[ep_size, runtime_max_tokens_per_rank, *]`` regardless of
        ``payload_in_workspace``: in both cases the payload holds the
        per-expert-rank outputs to be combined back to the source ranks.
        Only the backing memory differs (caller-supplied vs. workspace-backed
        view produced by :func:`moe_a2a_wrap_payload_tensor_in_workspace`).
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
        Optional output data type.  Currently supports ``torch.bfloat16``,
        ``torch.float8_e4m3fn``, and ``torch.uint8`` (packed fp4).
    output_scales : Optional[torch.Tensor]
        Contiguous CUDA scale tensor for quantized outputs.  MXFP8 and MXFP4 use
        UE8M0 scales packed in ``torch.uint8`` with vector size 32; NVFP4 uses
        UE4M3 scales in ``torch.float8_e4m3fn`` with vector size 16.  Its extent
        must exactly match ``sf_layout``, including layout padding.
    output_scalar_scale : float
        Per-tensor global scale applied before FP4 block scaling
        (NVFP4 SFScaleVal).  Defaults to ``1.0``; ignored by MXFP8/MXFP4
        paths.
    sf_layout : SfLayout
        Output swizzle layout.  Defaults to ``SfLayout.layout_linear``.
    output : Optional[torch.Tensor]
        Caller-provided contiguous output tensor. Its shape and dtype must
        match the requested combine output, and it must be on the same device
        as ``payload``.
    use_low_precision : bool
        If ``True``, quantize the recv-buffer payload to FP8 (e4m3) before
        accumulating; the combine upcasts to a bf16 output.
    enable_pdl : Optional[bool]
        Whether to use programmatic dependent launch.  ``None`` auto-detects
        from the device.
    enable_rank_mask : bool
        Whether to instantiate the kernel variant that checks ``active_rank_mask`` at
        all.  ``False`` (default) compiles out the rank-mask checks in peer
        synchronization for the common no-fault-tolerance case and requires
        ``active_rank_mask`` to be omitted.
    active_rank_mask : Optional[torch.Tensor]
        Optional CPU ``uint64`` tensor of shape ``[MOE_A2A_RANK_MASK_WORDS]`` (see
        :func:`moe_a2a_active_rank_mask`).  Should match the mask passed to the
        corresponding :func:`moe_a2a_dispatch` call (or be omitted from both).  Requires
        ``enable_rank_mask=True``.
    backend : MoeAlltoAllBackend
        Communication backend. Defaults to ``"trtllm"``; use ``"cake"`` only
        with a workspace initialized by the Cake backend.

    Returns
    -------
    torch.Tensor
        ``[local_num_tokens, *]`` tensor with the combined outputs.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(payload.device)
    if output is not None:
        if not output.is_cuda:
            raise ValueError(
                f"output must be a CUDA tensor, got device={output.device}"
            )
        if not output.is_contiguous():
            raise ValueError(f"output must be contiguous, got stride={output.stride()}")

    module = get_moe_alltoall_module(backend)
    args = (
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
        output_scalar_scale,
        sf_layout,
        use_low_precision,
        enable_pdl,
        enable_rank_mask,
        active_rank_mask,
    )
    if output is None:
        return module.moe_a2a_combine(*args)
    module.moe_a2a_combine_into(*args, output)
    return output


@flashinfer_api
def moe_a2a_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    invalid_expert_id: int,
    enable_pdl: Optional[bool] = None,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
):
    r"""Sanitize invalid slots that contain no token routed to this rank by
    setting their expert IDs to ``invalid_expert_id``.

    ``backend="trtllm"`` is the default. For a workspace initialized with
    ``backend="cake"``, explicitly pass ``backend="cake"`` here as well.

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
        Value to write into slots that received no token (per
        ``recv_counters``, e.g. padding beyond a source rank's valid count).
    enable_pdl : Optional[bool]
        Whether to use programmatic dependent launch.  ``None`` auto-detects
        from the device.
    backend : MoeAlltoAllBackend
        Communication backend. Defaults to ``"trtllm"``; use ``"cake"`` only
        with a workspace initialized by the Cake backend.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(expert_ids.device)
    return get_moe_alltoall_module(backend).moe_a2a_sanitize_expert_ids(
        expert_ids, workspace, metainfo, ep_rank, invalid_expert_id, enable_pdl
    )


@flashinfer_api
def moe_a2a_get_workspace_size_per_rank(
    ep_size: int,
    max_num_tokens: int,
    total_dispatch_payload_size_per_token: int,
    combine_payload_size_per_token: int,
    eplb_stats_num_experts: int = 0,
    *,
    backend: MoeAlltoAllBackend = "trtllm",
):
    r"""Compute the per-rank workspace size for the MoE all-to-all primitive.

    ``backend="trtllm"`` is the default. Pass ``backend="cake"`` to size a
    workspace for the explicitly selected Blackwell implementation.

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
    eplb_stats_num_experts : int
        Number of experts reserved for the EPLB gathered-stats region
        (``0`` disables it).
    backend : MoeAlltoAllBackend
        Communication backend used to size the workspace. Defaults to
        ``"trtllm"``; pass ``"cake"`` for the Cake backend.

    Returns
    -------
    int
        Required workspace size per rank, in bytes.
    """
    aux_data_size = get_moe_alltoall_module(backend).moe_a2a_get_aux_data_size(
        ep_size,
        max_num_tokens,
        eplb_stats_num_experts,
    )

    def pad_up(x, y):
        return ((x + y - 1) // y) * y

    # Pad to 128 bytes to ensure alignment. This matches the implementation of C++ torch OP code.
    return (
        pad_up(aux_data_size, 128)
        + pad_up(ep_size * max_num_tokens * total_dispatch_payload_size_per_token, 128)
        + pad_up(ep_size * max_num_tokens * combine_payload_size_per_token, 128)
    )


__all__ = [
    "moe_a2a_active_rank_mask",
    "moe_a2a_combine",
    "moe_a2a_dispatch",
    "moe_a2a_get_workspace_size_per_rank",
    "moe_a2a_initialize",
    "moe_a2a_sanitize_expert_ids",
    "moe_a2a_wrap_payload_tensor_in_workspace",
]
