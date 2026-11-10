"""Host-side state for the SM120 GPU-heap hybrid transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .heuristic import MegaMoEKernelConfig


COMBINE_PLAN_ROW_OFFSET_WORDS = 32


def combine_plan_record_words(max_rows_per_peer: int) -> int:
    """Return payload plus the aligned uint64 commit for one peer plan."""

    payload_words = COMBINE_PLAN_ROW_OFFSET_WORDS + max_rows_per_peer
    signal_word_offset = (payload_words + 1) // 2 * 2
    return signal_word_offset + 2


@dataclass
class HybridTransportState:
    """Own the buffers, streams and compiled launchers for one EP rank.

    The runner owns model inputs and compute kernels; this object owns only
    communication state.  The transport protocol itself is intentionally
    fixed to GPU-built routes, chunked owner plans, sparse dispatch, and a
    ready-queue/direct-staging combine ring.
    """

    dispatch_ring_channels: int
    combine_ring_channels: int
    dispatch_ring_slots: int
    combine_ring_slots: int

    activation_rx: Any = None
    sf_rx: Any = None
    topk_weight_rx: Any = None
    max_pair_routes: int = 0

    dispatch_send_token_indices: Any = None
    dispatch_send_counts: Any = None
    dispatch_owner_token_counts: Any = None
    dispatch_owner_token_bases: Any = None
    dispatch_owner_token_cursors: Any = None
    dispatch_owner_route_output_bases: Any = None
    dispatch_ready: Any = None
    dispatch_epoch: int = 0
    dispatch_send_stream: Any = None
    dispatch_recv_stream: Any = None
    dispatch_send_done: Any = None
    dispatch_credit_done: Any = None
    dispatch_recv_done: Any = None
    dispatch_send_work_queue: Any = None
    dispatch_recv_work_queue: Any = None
    dispatch_tx_handoff: Any = None
    dispatch_ring_phase: Any = None

    dispatch_ring_rx: Any = None
    dispatch_ring_tx: Any = None
    dispatch_ring_ack: Any = None
    dispatch_ring_ack_tx: Any = None
    dispatch_control_rx: Any = None
    dispatch_control_tx: Any = None
    dispatch_ring_sender_kernel: Any = None
    dispatch_ring_receiver_kernel: Any = None
    dispatch_ring_sender_runtime_args: Any = None
    dispatch_ring_ack_runtime_args: Any = None
    dispatch_ring_receiver_runtime_args: Any = None
    dispatch_route_builder_kernel: Any = None
    dispatch_route_builder_runtime_args: Any = None

    route_stream: Any = None
    expert_pool_bases: Any = None
    combine_source_expert_bases: Any = None
    owner_plan_count_rx: Any = None
    owner_plan_count_signal_rx: Any = None
    owner_plan_count_signal_tx: Any = None
    owner_plan_count_tx: Any = None
    owner_plan_route_rx: Any = None
    owner_plan_route_signal_rx: Any = None
    owner_plan_route_signal_tx: Any = None
    owner_plan_route_tx: Any = None
    owner_plan_route_bases_tx: Any = None
    owner_plan_route_cursors_tx: Any = None
    owner_plan_route_bases_rx: Any = None
    owner_plan_route_capacity: int = 0
    owner_plan_ready: Any = None
    owner_plan_push_kernel: Any = None
    owner_route_builder_kernel: Any = None
    owner_plan_count_push_runtime_args: Any = None
    owner_plan_route_push_runtime_args: Any = None
    owner_count_builder_runtime_args: Any = None
    owner_metadata_builder_runtime_args: Any = None
    owner_plan_count_push_done: Any = None
    owner_plan_route_push_done: Any = None
    owner_metadata_done: Any = None

    ring_rx: Any = None
    ring_tx: Any = None
    ring_ack: Any = None
    ring_ack_tx: Any = None
    ring_epoch: int = 0
    ring_stream: Any = None
    ring_recv_stream: Any = None
    owner_plan_stream: Any = None
    owner_metadata_stream: Any = None
    ring_done: Any = None
    ring_recv_done: Any = None
    ring_credit_done: Any = None
    ring_phase: Any = None
    ring_sender_kernel: Any = None
    ring_receiver_kernel: Any = None
    ring_sender_runtime_args: Any = None
    ring_ack_runtime_args: Any = None
    ring_receiver_runtime_args: Any = None

    ring_send_rows: Any = None
    ring_send_output_rows: Any = None
    combine_recv_output_rows: Any = None
    ring_send_ready_slots: Any = None
    ring_send_ready_values: Any = None
    ring_send_counts: Any = None
    combine_pool_to_stage_rows: Any = None
    combine_ready_queue_done: Any = None
    combine_ready_queue_probe: Any = None
    combine_ready_queue_selected: Any = None
    combine_send_work_queue: Any = None
    combine_recv_work_queue: Any = None

    k1_sm_count: int = 0

    compiled_dispatch_route_builder: Any = None
    compiled_owner_plan_count_push: Any = None
    compiled_owner_plan_route_push: Any = None
    compiled_owner_count_builder: Any = None
    compiled_owner_metadata_builder: Any = None
    compiled_dispatch_ring_sender: Any = None
    compiled_dispatch_ring_ack_waiter: Any = None
    compiled_dispatch_ring_receiver: Any = None
    compiled_ring_sender: Any = None
    compiled_ring_ack_waiter: Any = None
    compiled_ring_receiver: Any = None
    kernel_objects: list[Any] = field(default_factory=list)

    @classmethod
    def from_config(
        cls, config: "MegaMoEKernelConfig"
    ) -> "HybridTransportState":
        """Create transport state from the already-resolved pure config."""
        dispatch_channels = config.dispatch_channels
        combine_channels = config.combine_channels
        dispatch_slots = config.dispatch_slots
        combine_slots = config.combine_slots
        if min(
            dispatch_channels,
            combine_channels,
            dispatch_slots,
            combine_slots,
        ) <= 0:
            raise ValueError(
                "hybrid transport channels and ring slots must be positive"
            )
        if dispatch_channels != combine_channels:
            raise ValueError(
                "per-channel TX handoff requires equal dispatch and combine "
                "channel counts"
            )
        return cls(
            dispatch_ring_channels=dispatch_channels,
            combine_ring_channels=combine_channels,
            dispatch_ring_slots=dispatch_slots,
            combine_ring_slots=combine_slots,
        )
