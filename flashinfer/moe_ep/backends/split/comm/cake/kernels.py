"""Kernel-level operations of the generated Cake MoE all-to-all kernels.

These wrap the JIT module behind :class:`flashinfer.moe_ep.CakeAlltoAll`: they
size and initialize a workspace and run one dispatch / combine round on it.
The kernels are generated for compute capability 10.0 (``sm100a``) and 10.3
(``sm103a``); each architecture has its own module.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Sequence

import torch

from ......tllm_enums import SfLayout
from ......utils import device_support_pdl

# Number of uint64 words in the active_rank_mask ABI (must match kRankMaskWords in
# csrc/cake_moe_alltoall_dispatch.cuh, which is derived from kMaxRanks there). A
# single word covers up to 64 ranks.
RANK_MASK_WORDS = 1

_TARGETS = {(10, 0): "sm100a", (10, 3): "sm103a"}


@functools.cache
def get_cake_module(target: str) -> Any:
    """Build (on first use) and load the module of one architecture."""
    from ......jit.comm import gen_moe_ep_cake_module

    return gen_moe_ep_cake_module(target).build_and_load()  # type: ignore[arg-type]


@functools.cache
def _device_target(device_index: int) -> str:
    capability = torch.cuda.get_device_capability(device_index)
    target = _TARGETS.get(capability)
    if target is None:
        raise ValueError(
            "The Cake MoE all-to-all kernels require compute capability 10.0 or "
            f"10.3, got {capability}; use flashinfer.moe_ep.NVLinkOneSidedAlltoAll "
            "on other GPUs."
        )
    return target


def current_module() -> Any:
    """The module of the current CUDA device's architecture."""
    return get_cake_module(_device_target(torch.cuda.current_device()))


def metainfo_index() -> dict[str, int]:
    """Metainfo field indices, keyed by name without the ``MOE_A2A_`` prefix."""
    names, values = current_module().moe_a2a_get_metainfo_index_pairs()
    return {
        str(name).removeprefix("MOE_A2A_"): int(value)
        for name, value in zip(names, values, strict=True)
    }


def active_rank_mask(active_ranks: Sequence[int], ep_size: int) -> torch.Tensor:
    """Build the CPU ``uint64`` ``active_rank_mask`` of :func:`dispatch` and
    :func:`combine`.

    ``active_ranks`` lists the ranks that are alive and participate in the
    collective, e.g. ``[0, 1, 3]`` for a 4-rank job whose rank 2 failed; a 1D
    tensor of rank indices works too. Returns a ``[RANK_MASK_WORDS]`` tensor
    with bit ``i`` set for each active rank ``i``.
    """
    mask = 0
    for rank in active_ranks:
        rank = int(rank)
        if not 0 <= rank < ep_size:
            raise ValueError(f"rank {rank} out of range [0, {ep_size})")
        mask |= 1 << rank
    words = [
        (mask >> (64 * word)) & 0xFFFFFFFFFFFFFFFF for word in range(RANK_MASK_WORDS)
    ]
    return torch.tensor(words, dtype=torch.uint64, device="cpu")


def workspace_size_per_rank(
    ep_size: int,
    max_num_tokens: int,
    dispatch_bytes_per_token: int,
    combine_bytes_per_token: int,
    eplb_stats_num_experts: int = 0,
) -> int:
    """Workspace bytes per rank for rounds of up to ``max_num_tokens`` tokens per
    rank whose dispatch payloads total ``dispatch_bytes_per_token`` and whose
    combine payload is ``combine_bytes_per_token``. ``eplb_stats_num_experts``
    reserves the gathered EPLB statistics region (0 disables it)."""
    aux_data_size = current_module().moe_a2a_get_aux_data_size(
        ep_size, max_num_tokens, eplb_stats_num_experts
    )

    def pad_up(x: int) -> int:
        return (x + 127) // 128 * 128

    # Each region is padded to 128 bytes, as the module lays them out.
    return (
        pad_up(aux_data_size)
        + pad_up(ep_size * max_num_tokens * dispatch_bytes_per_token)
        + pad_up(ep_size * max_num_tokens * combine_bytes_per_token)
    )


def initialize(
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    max_num_tokens: int,
    eplb_stats_num_experts: int = 0,
) -> torch.Tensor:
    """Clear this rank's slice of the ``[ep_size, bytes_per_rank]`` workspace
    and return its metainfo tensor.

    Call it once per workspace allocation; no peer may dispatch into the
    workspace until every rank has initialized its slice.
    ``eplb_stats_num_experts`` must match the ``eplb_local_stats`` length later
    passed to :func:`dispatch` (0 disables EPLB statistics).
    """
    return current_module().moe_a2a_initialize(
        workspace, ep_rank, ep_size, max_num_tokens, eplb_stats_num_experts
    )


def wrap_workspace(
    workspace: torch.Tensor,
    leading_shape: Sequence[int],
    slice_start: int,
    slice_end: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """View the workspace bytes ``[slice_start, slice_end)`` as a ``dtype``
    tensor of shape ``leading_shape + [-1]``.

    The offsets count from the start of the ``[ep_size, bytes_per_rank]`` (or
    ``[bytes_per_rank]``) workspace and must fall within a single rank.
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
    local_slice_end = local_slice_start + slice_end - slice_start
    assert local_slice_end <= workspace_base.shape[1], (
        "slice must fall within the workspace size per rank"
    )
    return (
        workspace_base[slice_rank, local_slice_start:local_slice_end]
        .view(dtype=dtype)
        .view(*leading_shape, -1)
    )


def dispatch(
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
    recv_view_cache: Optional[dict] = None,
) -> tuple[list[torch.Tensor], int, Optional[torch.Tensor]]:
    """Send each token's payloads to the ranks that own its experts.

    ``token_selected_experts`` is the ``[local_num_tokens, top_k]`` ``int32``
    routing and ``input_payloads`` the ``[local_num_tokens, *]`` tensors to
    send. ``runtime_max_tokens_per_rank`` bounds every rank's token count in
    this round and must not exceed the ``max_num_tokens`` of
    :func:`initialize`. ``enable_pdl=None`` uses programmatic dependent launch
    where the device supports it.

    ``eplb_local_stats``, an ``[eplb_stats_num_experts]`` ``int32`` tensor, is
    all-gathered across the ranks during the dispatch.

    ``enable_rank_mask`` selects the kernel variant that honors
    ``active_rank_mask`` (see :func:`active_rank_mask`); without it the mask
    checks are compiled out and the mask must be omitted. Tokens routed to a
    masked-off rank are dropped instead of hanging the collective, and the
    local rank's bit must be set. Masking a peer does not advance its
    transport epoch: before a skipped peer rejoins, all ranks must quiesce and
    re-initialize the workspace.

    ``recv_view_cache``, an initially empty dict owned by the caller, reuses
    the receive views across rounds on the same workspace tensor. It holds a
    reference to the workspace and is cleared when another workspace is
    passed.

    Returns ``(recv_payloads, combine_payload_offset, eplb_gathered_stats)``:
    one ``[ep_size, runtime_max_tokens_per_rank, *]`` workspace view per input
    payload holding the rows routed to this rank, the workspace offset of the
    matching :func:`combine` payload, and the ``[ep_size,
    eplb_stats_num_experts]`` gathered statistics (``None`` without
    ``eplb_local_stats``).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(token_selected_experts.device)
    (
        recv_offsets,
        recv_sizes,
        combine_payload_offset,
        eplb_gathered_stats_offset,
        eplb_stats_num_experts,
    ) = current_module().moe_a2a_dispatch(
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

    if (
        recv_view_cache is not None
        and recv_view_cache.get("_workspace") is not workspace
    ):
        recv_view_cache.clear()
        recv_view_cache["_workspace"] = workspace

    recv_payloads = []
    for input_payload, offset, size in zip(
        input_payloads, recv_offsets, recv_sizes, strict=True
    ):
        # The offsets index the whole workspace. ``int()`` keeps the key hashing
        # by value whatever integer type the FFI returns.
        key = (
            ep_size,
            runtime_max_tokens_per_rank,
            int(offset),
            int(size),
            input_payload.dtype,
        )
        view = None if recv_view_cache is None else recv_view_cache.get(key)
        if view is None:
            view = wrap_workspace(
                workspace,
                [ep_size, runtime_max_tokens_per_rank],
                offset,
                offset + size,
                input_payload.dtype,
            )
            if recv_view_cache is not None:
                recv_view_cache[key] = view
        recv_payloads.append(view)

    eplb_gathered_stats = None
    if eplb_gathered_stats_offset >= 0:
        eplb_gathered_stats = wrap_workspace(
            workspace,
            [ep_size],
            eplb_gathered_stats_offset,
            eplb_gathered_stats_offset + ep_size * eplb_stats_num_experts * 4,
            torch.int32,
        )
    return recv_payloads, combine_payload_offset, eplb_gathered_stats


def combine(
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
) -> torch.Tensor:
    """Return each source token's expert outputs, summed over its top-k slots.

    ``payload`` is the ``[ep_size, runtime_max_tokens_per_rank, *]`` expert
    output of the preceding :func:`dispatch` round on this rank, whose
    ``combine_payload_offset`` and ``runtime_max_tokens_per_rank`` it takes.
    ``payload_in_workspace`` marks a payload already written to that offset,
    e.g. through :func:`wrap_workspace`, which skips the staging copy.

    ``output_dtype`` quantizes the ``[local_num_tokens, *]`` result: bfloat16,
    ``float8_e4m3fn`` or packed FP4 ``uint8``. Quantized outputs write their
    block scales to the contiguous ``output_scales``: UE8M0 ``uint8`` scales
    per 32 elements for MXFP8 / MXFP4, UE4M3 ``float8_e4m3fn`` scales per 16
    for NVFP4, laid out by ``sf_layout`` including its padding. NVFP4 applies
    ``output_scalar_scale`` before block scaling. ``use_low_precision``
    transfers the payload as FP8 (e4m3) and accumulates into a bfloat16
    result. ``output``, a contiguous CUDA tensor of the result's shape and
    dtype, receives the result in place.

    ``active_rank_mask`` must match the one of the dispatch.
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
    module = current_module()
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
        sf_layout.value,
        use_low_precision,
        enable_pdl,
        enable_rank_mask,
        active_rank_mask,
    )
    if output is None:
        return module.moe_a2a_combine(*args)
    module.moe_a2a_combine_into(*args, output)
    return output


def sanitize_expert_ids(
    expert_ids: torch.Tensor,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    invalid_expert_id: int,
    enable_pdl: Optional[bool] = None,
) -> None:
    """Set the expert ids of the received rows beyond each source rank's token
    count to ``invalid_expert_id``, in place; ``expert_ids`` is the expert-id
    receive payload of the preceding :func:`dispatch`."""
    if enable_pdl is None:
        enable_pdl = device_support_pdl(expert_ids.device)
    current_module().moe_a2a_sanitize_expert_ids(
        expert_ids, workspace, metainfo, ep_rank, invalid_expert_id, enable_pdl
    )
