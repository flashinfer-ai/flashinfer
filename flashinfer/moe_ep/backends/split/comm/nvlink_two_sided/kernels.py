"""Kernel-level operations of the NVLink two-sided MoE all-to-all.

These wrap the JIT module behind :class:`flashinfer.moe_ep.NVLinkTwoSidedAlltoAll`.
:func:`prepare` exchanges the routing and computes the send / receive
indices of a round over the prepare workspace; :func:`alltoallv` then moves
rows along those indices through the FIFO channels of the communication
workspace.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch


@functools.cache
def get_nvlink_two_sided_alltoall_module() -> Any:
    """Build (on first use) and load the JIT module backing ``NVLinkTwoSidedAlltoAll``."""
    from ......jit.comm import gen_nvlink_two_sided_alltoall_module

    return gen_nvlink_two_sided_alltoall_module().build_and_load()


def set_max_usable_sm_count(max_sm_count: int) -> None:
    """Cap the SMs :func:`alltoallv` spreads its FIFO channels over, e.g. so
    that several ranks can share one GPU. Must precede the first
    :func:`comm_workspace_size_per_rank` or :func:`alltoallv`, which fix the
    channel count."""
    get_nvlink_two_sided_alltoall_module().set_moe_max_usable_sm_count(max_sm_count)


def comm_workspace_size_per_rank(ep_size: int) -> int:
    """Bytes per rank of the FIFO workspace used by :func:`alltoallv`."""
    return get_nvlink_two_sided_alltoall_module().get_moe_commworkspace_size_per_rank(
        ep_size
    )


def prepare_workspace_size_per_rank(ep_size: int) -> int:
    """Bytes per rank of the workspace used by :func:`prepare`."""
    return (
        get_nvlink_two_sided_alltoall_module().get_moe_prepare_workspace_size_per_rank(
            ep_size
        )
    )


def alltoallv(
    input: torch.Tensor,
    send_rank_cum_sum: torch.Tensor,
    send_indices: torch.Tensor,
    output: torch.Tensor,
    recv_rank_cum_sum: torch.Tensor,
    recv_indices: torch.Tensor,
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
) -> None:
    """Send the rows of the 2D ``input`` listed in ``send_indices`` to the
    peer ranks and write the rows received from them to the rows of
    ``output`` listed in ``recv_indices``.

    The ``int32`` cumulative per-rank counts split the index lists by peer.
    ``workspace`` is the ``[ep_size, words_per_rank]`` ``uint64`` FIFO
    workspace of :func:`comm_workspace_size_per_rank` bytes per rank.
    """
    get_nvlink_two_sided_alltoall_module().moe_comm(
        input,
        send_rank_cum_sum,
        send_indices,
        output,
        recv_rank_cum_sum,
        recv_indices,
        workspace,
        ep_rank,
        ep_size,
    )


def prepare(
    expert_ids: torch.Tensor,
    scales: Optional[torch.Tensor],
    expert_statistics: Optional[torch.Tensor],
    workspace: torch.Tensor,
    max_token_count_per_rank: int,
    ep_rank: int,
    ep_size: int,
    expert_count: int,
    slot_count: int,
    top_k: int,
) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """Exchange the ``[num_tokens, top_k]`` ``int32`` routing ``expert_ids``,
    their optional FP32 ``scales`` and the optional per-expert
    ``expert_statistics`` across the ranks, and compute the indices of the
    round.

    ``workspace`` is the ``[ep_size, words_per_rank]`` ``uint64`` workspace of
    :func:`prepare_workspace_size_per_rank` bytes per rank. ``expert_count``
    and ``slot_count`` must be divisible by 4.

    Returns ``(recv_expert_ids, recv_scales, send_rank_cum_sum, send_indices,
    recv_rank_cum_sum, recv_indices, backward_recv_indices,
    gathered_statistics)``. The received routing has ``ep_size *
    max_token_count_per_rank`` rank-major rows; rows past a source rank's
    token count carry ``slot_count`` as their expert id. The index tensors
    drive :func:`alltoallv` forward (send / recv) and back (recv /
    backward recv), and ``gathered_statistics`` is ``[ep_size,
    expert_count]``.
    """
    attrs = {"dtype": torch.int32, "device": expert_ids.device}
    recv_rows = max_token_count_per_rank * ep_size
    send_rows = max_token_count_per_rank * max(ep_size, top_k)

    recv_expert_ids = torch.empty((recv_rows, top_k), **attrs)
    send_rank_cum_sum = torch.empty((ep_size,), **attrs)
    recv_rank_cum_sum = torch.empty((ep_size,), **attrs)
    gather_recv_indices = torch.empty((recv_rows,), **attrs)
    recv_indices = torch.empty((recv_rows,), **attrs)
    gather_backward_recv_indices = torch.empty((send_rows,), **attrs)
    backward_recv_indices = torch.empty((send_rows,), **attrs)
    gather_send_indices = torch.empty((send_rows,), **attrs)
    send_indices = torch.empty((send_rows,), **attrs)
    recv_scales = (
        None
        if scales is None
        else torch.empty(
            (recv_rows, top_k), dtype=torch.float32, device=expert_ids.device
        )
    )
    gathered_statistics = (
        None
        if expert_statistics is None
        else torch.empty((ep_size, expert_count), **attrs)
    )

    get_nvlink_two_sided_alltoall_module().moe_prepare(
        expert_ids,
        scales,
        expert_statistics,
        workspace,
        recv_expert_ids,
        send_rank_cum_sum,
        recv_rank_cum_sum,
        gather_recv_indices,
        recv_indices,
        gather_backward_recv_indices,
        backward_recv_indices,
        gather_send_indices,
        send_indices,
        recv_scales,
        gathered_statistics,
        max_token_count_per_rank,
        ep_rank,
        ep_size,
        expert_count,
        slot_count,
        top_k,
    )
    return (
        recv_expert_ids,
        recv_scales,
        send_rank_cum_sum,
        gather_send_indices,
        recv_rank_cum_sum,
        gather_recv_indices,
        gather_backward_recv_indices,
        gathered_statistics,
    )
