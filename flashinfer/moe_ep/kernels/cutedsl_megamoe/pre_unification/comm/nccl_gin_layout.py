"""Host-side NCCL GIN window layout helpers for internode MegaMoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


@dataclass(frozen=True)
class RelayMessageLayout:
    """Byte layout for one Stage 1 dispatch relay message."""

    src_metadata_offset: int
    route_words_offset: int
    topk_weights_offset: int
    token_data_offset: int
    sf_data_offset: int
    message_bytes: int


@dataclass(frozen=True)
class NcclGinWindowLayout:
    """Symmetric NCCL GIN window layout shared by all ranks."""

    internode_send_count_offset: int
    internode_send_count_shape: Tuple[int, int]
    internode_recv_splits_offset: int
    internode_recv_splits_shape: Tuple[int, int]
    relay_send_buffer_offset: Optional[int]
    relay_send_buffer_shape: Optional[Tuple[int, int, int]]
    relay_recv_buffer_offset: Optional[int]
    relay_recv_buffer_shape: Optional[Tuple[int, int, int]]
    relay_message: Optional[RelayMessageLayout]
    intra_node_reduce_buf_offset: Optional[int]
    intra_node_reduce_buf_shape: Optional[Tuple[int, int, int]]
    internode_combine_buffer_offset: Optional[int]
    internode_combine_buffer_shape: Optional[Tuple[int, int, int, int]]
    internode_splits_signal_base: int
    internode_barrier_signal_base: int
    relay_node_signal_base: int
    icombine_arrival_signal_base: int
    total_signal_count: int
    total_bytes: int

    @property
    def metadata_signal_count(self) -> int:
        return self.relay_node_signal_base

    @property
    def relay_node_signal_count(self) -> int:
        return self.icombine_arrival_signal_base - self.relay_node_signal_base

    @property
    def combine_signal_count(self) -> int:
        return self.total_signal_count - self.icombine_arrival_signal_base


def build_relay_message_layout(
    *,
    hidden_bytes: int,
    sf_uint32_per_token: int,
    num_topk: int,
    alignment: int = 128,
) -> RelayMessageLayout:
    """Return the byte layout for one node-deduplicated dispatch relay message."""
    _require_positive("hidden_bytes", hidden_bytes)
    _require_positive("sf_uint32_per_token", sf_uint32_per_token)
    _require_positive("num_topk", num_topk)
    _require_positive("alignment", alignment)

    cursor = 0
    src_metadata_offset = cursor
    # TokenSrcMetadata itself is one packed 8B record and needs only 8B
    # alignment.  Reserve a 16B wire slot so route_words, topk_weights, and
    # the following payload planes all start 16B-aligned.  At target topk=4,
    # each route/weight plane is exactly one 16B vector, allowing a single
    # 128-bit load/store and keeping future vectorized or cp.async-based
    # pack/fan-out implementations simple.  Bytes [src_metadata_offset + 8,
    # src_metadata_offset + 16) are padding, not metadata fields (Invariant
    # I6).  This padding is an implementation/performance choice, not a GIN
    # correctness requirement; for the target shape the complete message
    # rounds to the same 4096B transfer stride with or without these 8 bytes.
    # With 16B rather than 128B message-stride alignment, the target message
    # would remain 4080B instead of rounding to 4096B, saving 16B (~0.4%) per
    # put but losing the power-of-two stride.  Keeping 4096B also matches the
    # existing 128B token-pool alignment: Stage 2 whole-message copies and
    # Stage 3 token-plane vector copies are naturally aligned. Neither path
    # uses TMA; the ISA minimum for their b128 operations remains 16B.
    cursor += 16
    route_words_offset = cursor
    cursor += _round_up(num_topk * 4, 16)
    topk_weights_offset = cursor
    cursor += _round_up(num_topk * 4, 16)
    token_data_offset = cursor
    cursor += hidden_bytes
    sf_data_offset = cursor
    cursor += sf_uint32_per_token * 4
    message_bytes = _round_up(cursor, alignment)

    return RelayMessageLayout(
        src_metadata_offset=src_metadata_offset,
        route_words_offset=route_words_offset,
        topk_weights_offset=topk_weights_offset,
        token_data_offset=token_data_offset,
        sf_data_offset=sf_data_offset,
        message_bytes=message_bytes,
    )


def build_megamoe_gin_window_layout(
    *,
    world_size: int,
    ranks_per_node: int,
    num_experts_per_rank: int,
    max_relay_tokens_per_node: Optional[int] = None,
    hidden_bytes: Optional[int] = None,
    sf_uint32_per_token: Optional[int] = None,
    num_topk: Optional[int] = None,
    combine_hidden: Optional[int] = None,
    alignment: int = 128,
) -> NcclGinWindowLayout:
    """Compute symmetric GIN window offsets for metadata and optional relay payloads."""
    _require_positive("world_size", world_size)
    _require_positive("ranks_per_node", ranks_per_node)
    _require_positive("num_experts_per_rank", num_experts_per_rank)
    _require_positive("alignment", alignment)
    if world_size % ranks_per_node != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"ranks_per_node ({ranks_per_node})."
        )
    num_nodes = world_size // ranks_per_node
    remote_slots = max(1, num_nodes - 1)
    node_experts = ranks_per_node * num_experts_per_rank

    cursor = 0
    internode_send_count_offset = cursor
    # The final int32 in every row is the node-deduplicated relay-message
    # count. Keeping it in the same vector as the R*E expert-hit counts lets
    # one signal-carrying put establish completion for the entire Stage 1 wire
    # contract; a second put would otherwise rely on undocumented cross-call
    # QP ordering.
    metadata_row_entries = node_experts + 1
    cursor += remote_slots * metadata_row_entries * 4
    cursor = _round_up(cursor, alignment)
    internode_recv_splits_offset = cursor
    cursor += remote_slots * metadata_row_entries * 4
    cursor = _round_up(cursor, alignment)

    has_payload = any(
        value is not None
        for value in (
            max_relay_tokens_per_node,
            hidden_bytes,
            sf_uint32_per_token,
            num_topk,
        )
    )
    relay_send_buffer_offset = None
    relay_send_buffer_shape = None
    relay_recv_buffer_offset = None
    relay_recv_buffer_shape = None
    relay_message = None
    intra_node_reduce_buf_offset = None
    intra_node_reduce_buf_shape = None
    internode_combine_buffer_offset = None
    internode_combine_buffer_shape = None
    if has_payload:
        if None in (
            max_relay_tokens_per_node,
            hidden_bytes,
            sf_uint32_per_token,
            num_topk,
        ):
            raise ValueError(
                "relay payload layout requires max_relay_tokens_per_node, "
                "hidden_bytes, sf_uint32_per_token, and num_topk."
            )
        _require_positive("max_relay_tokens_per_node", max_relay_tokens_per_node)
        relay_message = build_relay_message_layout(
            hidden_bytes=hidden_bytes,
            sf_uint32_per_token=sf_uint32_per_token,
            num_topk=num_topk,
            alignment=alignment,
        )
        relay_send_buffer_offset = cursor
        relay_send_buffer_shape = (
            remote_slots,
            max_relay_tokens_per_node,
            relay_message.message_bytes,
        )
        cursor += remote_slots * max_relay_tokens_per_node * relay_message.message_bytes
        cursor = _round_up(cursor, alignment)
        relay_recv_buffer_offset = cursor
        relay_recv_buffer_shape = relay_send_buffer_shape
        cursor += remote_slots * max_relay_tokens_per_node * relay_message.message_bytes
        cursor = _round_up(cursor, alignment)

        if combine_hidden is not None:
            _require_positive("combine_hidden", combine_hidden)
            # Stage 4 is BF16-only: the first region is local send staging for
            # the within-node FP32 gather-reduce, and the second is the
            # double-buffered RDMA landing plane consumed by TopkReduce.
            intra_node_reduce_buf_offset = cursor
            intra_node_reduce_buf_shape = (
                remote_slots,
                max_relay_tokens_per_node,
                combine_hidden,
            )
            cursor += (
                remote_slots * max_relay_tokens_per_node
                * combine_hidden * 2
            )
            cursor = _round_up(cursor, alignment)
            internode_combine_buffer_offset = cursor
            internode_combine_buffer_shape = (
                2,
                remote_slots,
                max_relay_tokens_per_node,
                combine_hidden,
            )
            cursor += (
                2 * remote_slots * max_relay_tokens_per_node
                * combine_hidden * 2
            )
            cursor = _round_up(cursor, alignment)

    internode_splits_signal_base = 0
    internode_barrier_signal_base = internode_splits_signal_base + remote_slots
    relay_node_signal_base = internode_barrier_signal_base + 2
    relay_node_signal_count = remote_slots if has_payload else 0
    icombine_arrival_signal_base = relay_node_signal_base + relay_node_signal_count
    combine_signal_count = remote_slots if has_payload else 0
    total_signal_count = max(1, icombine_arrival_signal_base + combine_signal_count)

    return NcclGinWindowLayout(
        internode_send_count_offset=internode_send_count_offset,
        internode_send_count_shape=(remote_slots, metadata_row_entries),
        internode_recv_splits_offset=internode_recv_splits_offset,
        internode_recv_splits_shape=(remote_slots, metadata_row_entries),
        relay_send_buffer_offset=relay_send_buffer_offset,
        relay_send_buffer_shape=relay_send_buffer_shape,
        relay_recv_buffer_offset=relay_recv_buffer_offset,
        relay_recv_buffer_shape=relay_recv_buffer_shape,
        relay_message=relay_message,
        intra_node_reduce_buf_offset=intra_node_reduce_buf_offset,
        intra_node_reduce_buf_shape=intra_node_reduce_buf_shape,
        internode_combine_buffer_offset=internode_combine_buffer_offset,
        internode_combine_buffer_shape=internode_combine_buffer_shape,
        internode_splits_signal_base=internode_splits_signal_base,
        internode_barrier_signal_base=internode_barrier_signal_base,
        relay_node_signal_base=relay_node_signal_base,
        icombine_arrival_signal_base=icombine_arrival_signal_base,
        total_signal_count=total_signal_count,
        total_bytes=_round_up(cursor, alignment),
    )
