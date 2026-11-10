"""Bounded GIN/LSA rings with strong network commits, system-ordered P2P stores, cyclic ACKs and terminal reuse credits."""

import cutlass
from . import gin_bridge
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64
from cuda.bindings import driver as cuda


from ..moe_utils import _nanosleep, spin_wait
from .hybrid_transport import (
    COMBINE_PLAN_ROW_OFFSET_WORDS,
    combine_plan_record_words,
)
from ..sm120_ptx_helpers import (
    ldg_b32_cv_raw,
    spin_wait_i32_ge_cv_raw,
)


# A control record keeps the count in word zero and an aligned uint64 commit
# signal at byte eight.  The count is transferred and remotely completed
# before the commit is published, so an observer cannot consume a stale count
# from the prior epoch.
_CONTROL_RECORD_BYTES = 128
_CONTROL_SIGNAL_BYTE_OFFSET = 8


@cute.jit
def _peer_uses_ibgda(peer, *, force_ibgda_mask):
    return (
        Int32(force_ibgda_mask) & (Int32(1) << peer)
    ) != Int32(0)


@cute.jit
def _prepare_hybrid_peer(peer, own_rank, lane, *, force_ibgda_mask):
    # Routing is explicit in the GIN/LSA bridge; no library state mutation.
    pass



@cute.jit
def _publish_tx_handoff_epoch(tx_handoff, index, epoch):
    """Publish this channel's current epoch with release semantics."""
    cute.arch.atomic_exch(
        tx_handoff.iterator + index,
        val=epoch,
        sem="release",
        scope="sys",
    )


class GinGpuRingCombineSender:
    """Gather K2-ready row batches and issue block-level put-with-signal."""

    def __init__(
        self,
        world_size: int,
        payload_bytes: int,
        max_rows_per_peer: int,
        *,
        chunk_rows: int = 8,
        channels: int = 4,
        slots: int = 2,
        hybrid_force_ibgda_mask: int,
        worker_ctas: int,
        num_experts_per_rank: int,
        blocking_put: bool,
        progress_windows: bool,
    ) -> None:
        gin_bridge.bind(self, 'return_tx')
        self.world_size = world_size
        self.hybrid_force_ibgda_mask = hybrid_force_ibgda_mask
        self.payload_bytes = payload_bytes
        self.payload_i32 = payload_bytes // 4
        self.payload_u128 = payload_bytes // 16
        self.chunk_rows = chunk_rows
        if chunk_rows > 128:
            raise ValueError("combine chunk_rows must fit in one 128-thread CTA")
        self.row_id_chunk_bytes = 4 * chunk_rows
        self.chunk_payload_bytes = (
            self.row_id_chunk_bytes + payload_bytes * chunk_rows
        )
        # The GIN commit uses an aligned uint64 signal variable.
        # Keep it outside the data range and align every cyclic slot to 16 B.
        self.signal_offset_bytes = (
            (self.chunk_payload_bytes + 7) // 8
        ) * 8
        self.slot_bytes = (
            (self.signal_offset_bytes + 8 + 15) // 16
        ) * 16
        self.slot_i32 = self.slot_bytes // 4
        self.max_rows_per_peer = max_rows_per_peer
        self.max_chunks_per_peer = (
            max_rows_per_peer + chunk_rows - 1
        ) // chunk_rows
        self.channels = channels
        self.slots = slots
        self.worker_ctas = min(worker_ctas, world_size * channels)
        self.num_experts_per_rank = num_experts_per_rank
        self.blocking_put = blocking_put
        self.progress_windows = progress_windows
        if self.worker_ctas <= 0:
            raise ValueError("combine sender worker_ctas must be positive")
        if self.worker_ctas % world_size:
            raise ValueError(
                "combine sender requires an integral CTA stripe per peer"
            )
        if self.worker_ctas != world_size * channels:
            raise ValueError(
                "combine sender requires one owner CTA per peer/channel"
            )
        self.vectorized_scatter = (
            payload_bytes % 16 == 0 and self.row_id_chunk_bytes % 16 == 0
        )
        # One CTA owns each (peer, channel).  Its four warps probe four
        # candidate chunks in parallel and publish ready candidates into a
        # tiny per-peer/channel queue.  Keeping this equal to the CTA's warp
        # count avoids a second producer kernel whose polling CTAs could
        # occupy the SMs needed by K2.
        self.ready_queue_depth = 4
        self.signal_stride = (
            (self.max_chunks_per_peer + channels - 1) // channels + 1
        )

    @cute.kernel
    def kernel(
        self,
        rx_slots: cute.Tensor,
        tx_slots: cute.Tensor,
        ack_signals: cute.Tensor,
        fc2_workspace: cute.Tensor,
        send_output_rows: cute.Tensor,
        send_ready_slots: cute.Tensor,
        send_ready_values: cute.Tensor,
        send_counts: cute.Tensor,
        combine_output: cute.Tensor,
        fc2_done_counter: cute.Tensor,
        ready_queue_done: cute.Tensor,
        ready_queue_probe: cute.Tensor,
        ready_queue_selected: cute.Tensor,
        tx_handoff: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        peer, channel_index, _ = cute.arch.block_idx()
        lane_idx, _, _ = cute.arch.thread_idx()

        if lane_idx == Int32(0):
            spin_wait_i32_ge_cv_raw(
                tx_handoff.iterator.toint()
                + Int64(peer * Int32(self.channels) + channel_index) * Int64(4),
                epoch,
            )
        cute.arch.barrier(barrier_id=0, number_of_threads=128)

        _prepare_hybrid_peer(
            peer,
            own_rank,
            lane_idx,
            force_ibgda_mask=self.hybrid_force_ibgda_mask,
        )


        epoch_base = epoch * Int32(self.signal_stride)
        tx_i32 = cute.recast_tensor(tx_slots, cutlass.Int32)
        ack_i32 = cute.recast_tensor(ack_signals, cutlass.Int32)
        fc2_i32 = cute.recast_tensor(fc2_workspace, cutlass.Int32)
        output_i32 = cute.recast_tensor(combine_output, cutlass.Int32)
        fc2_u128 = cute.recast_tensor(fc2_workspace, cutlass.Uint128)
        output_u128 = cute.recast_tensor(
            combine_output, cutlass.Uint128
        )
        count = send_counts[peer]
        plan_epoch_base = epoch * Int32(self.num_experts_per_rank)
        num_chunks = (
            count + Int32(self.chunk_rows) - Int32(1)
        ) // Int32(self.chunk_rows)
        channel_chunks = Int32(0)
        if channel_index < num_chunks:
            channel_chunks = (
                (num_chunks - Int32(1) - channel_index)
                // Int32(self.channels)
                + Int32(1)
            )
        sequence = Int32(1)
        scan_base = Int32(0)
        while sequence <= channel_chunks:
            chunk_idx = (
                channel_index
                + (sequence - Int32(1)) * Int32(self.channels)
            )
            if peer != own_rank:
                selected = Int32(-1)
                scan_round = Int32(0)
                warp_idx = lane_idx // Int32(32)
                lane_in_warp = lane_idx % Int32(32)
                queue_base = (
                    (peer * Int32(self.channels) + channel_index)
                    * Int32(self.ready_queue_depth)
                )
                selected_index = (
                    peer * Int32(self.channels) + channel_index
                )
                while selected < Int32(0):
                    ordinal = (
                        scan_base
                        + scan_round * Int32(self.ready_queue_depth)
                        + warp_idx
                    ) % channel_chunks
                    candidate = (
                        channel_index
                        + ordinal * Int32(self.channels)
                    )
                    done_index = (
                        peer * Int32(self.max_chunks_per_peer)
                        + candidate
                    )
                    candidate_done = ldg_b32_cv_raw(
                        ready_queue_done.iterator.toint()
                        + Int64(done_index) * Int64(4)
                    )
                    row_ready = candidate_done == Int32(0)
                    candidate_row_base = (
                        candidate * Int32(self.chunk_rows)
                    )
                    row_in_candidate = lane_in_warp
                    while row_in_candidate < Int32(self.chunk_rows):
                        if candidate_row_base + row_in_candidate < count:
                            metadata_index = (
                                peer * Int32(self.max_rows_per_peer)
                                + candidate_row_base
                                + row_in_candidate
                            )
                            packed_ready_slot = ldg_b32_cv_raw(
                                send_ready_slots.iterator.toint()
                                + Int64(metadata_index) * Int64(4)
                            )
                            if packed_ready_slot < plan_epoch_base:
                                row_ready = False
                            else:
                                # Metadata is published concurrently with this poll.
                                # Acquire its epoch before consuming the associated
                                # threshold; force a fresh load on every retry.
                                cute.arch.fence_acq_rel_sys()
                                ready_slot = (
                                    packed_ready_slot - plan_epoch_base
                                )
                                ready_value = ldg_b32_cv_raw(
                                    send_ready_values.iterator.toint()
                                    + Int64(metadata_index) * Int64(4)
                                )
                                observed = ldg_b32_cv_raw(
                                    fc2_done_counter.iterator.toint()
                                    + Int64(ready_slot) * Int64(4)
                                )
                                if observed < ready_value:
                                    row_ready = False
                        row_in_candidate = row_in_candidate + Int32(32)
                    candidate_ready = cute.arch.vote_all_sync(row_ready)
                    # The receiver derives the valid row count from the
                    # sequence number.  Therefore the one partial tail
                    # chunk, if present, must remain the channel's final
                    # wire sequence even though full chunks may reorder.
                    if (
                        candidate == num_chunks - Int32(1)
                        and sequence != channel_chunks
                    ):
                        candidate_ready = False
                    if lane_in_warp == Int32(0):
                        queued_candidate = Int32(-1)
                        if candidate_ready:
                            queued_candidate = candidate
                        ready_queue_probe[
                            queue_base + warp_idx
                        ] = queued_candidate
                    cute.arch.barrier(
                        barrier_id=0, number_of_threads=128
                    )
                    if lane_idx == Int32(0):
                        selected_candidate = Int32(-1)
                        queue_slot = Int32(0)
                        while queue_slot < Int32(self.ready_queue_depth):
                            queued = ready_queue_probe[
                                queue_base + queue_slot
                            ]
                            if (
                                selected_candidate < Int32(0)
                                and queued >= Int32(0)
                            ):
                                selected_candidate = queued
                            queue_slot = queue_slot + Int32(1)
                        if selected_candidate >= Int32(0):
                            ready_queue_done[
                                peer * Int32(self.max_chunks_per_peer)
                                + selected_candidate
                            ] = Int32(1)
                        ready_queue_selected[
                            selected_index
                        ] = selected_candidate
                    cute.arch.barrier(
                        barrier_id=0, number_of_threads=128
                    )
                    selected = ready_queue_selected[selected_index]
                    if selected < Int32(0):
                        _nanosleep(500)
                        scan_round = scan_round + Int32(1)
                chunk_idx = selected
                scan_base = (
                    (chunk_idx - channel_index)
                    // Int32(self.channels)
                    + Int32(1)
                ) % channel_chunks
            row_base = chunk_idx * Int32(self.chunk_rows)
            # Readiness used to be polled serially by lane zero.  A chunk can
            # contain outputs from unrelated experts/K2 CTAs, so that made
            # every later row pay the polling latency of all earlier rows.
            # Assign one lane to each row and release the pack only after the
            # whole chunk is ready.
            if (
                lane_idx < Int32(self.chunk_rows)
                and row_base + lane_idx < count
            ):
                metadata_index = (
                    peer * Int32(self.max_rows_per_peer)
                    + row_base
                    + lane_idx
                )
                spin_wait_i32_ge_cv_raw(
                    send_ready_slots.iterator.toint()
                    + Int64(metadata_index) * Int64(4),
                    plan_epoch_base,
                )
                cute.arch.fence_acq_rel_sys()
                ready_slot = (
                    send_ready_slots[metadata_index] - plan_epoch_base
                )
                ready_value = send_ready_values[metadata_index]
                spin_wait(
                    fc2_done_counter.iterator + ready_slot,
                    lambda value: value >= ready_value,
                    fail_sleep_cycles=500,
                )
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            if peer != own_rank:
                slot = (sequence - Int32(1)) % Int32(self.slots)
                signal_index = (
                    (peer * Int32(self.channels) + channel_index)
                    * Int32(self.slots)
                    + slot
                )
                if sequence > Int32(self.slots):
                    # Before reusing the first slot in a window, complete the
                    # previous window's nonblocking data/commit puts.  This is
                    # the producer-side progress point: it cannot depend on an
                    # ACK that the receiver can only create after observing an
                    # otherwise still-pending NBI commit.
                    if (
                        cutlass.const_expr(self.progress_windows)
                        and cutlass.const_expr(not self.blocking_put)
                        and (sequence - Int32(1)) % Int32(self.slots)
                        == Int32(0)
                    ):
                        gin_bridge.quiet(self)
                        cute.arch.barrier(
                            barrier_id=0, number_of_threads=128
                        )
                    if lane_idx == Int32(0):
                        # Ack records are cache-line sized; word zero carries
                        # the monotonically increasing epoch/sequence value.
                        spin_wait_i32_ge_cv_raw(
                            ack_i32.iterator.toint()
                            + Int64(signal_index) * Int64(128),
                            epoch_base + sequence - Int32(self.slots),
                        )
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)
            self._send_one_chunk(
                peer,
                channel_index,
                sequence,
                chunk_idx,
                count,
                rx_slots,
                tx_slots,
                fc2_workspace,
                send_output_rows,
                combine_output,
                own_rank,
                epoch,
                lane_idx,
            )
            sequence = sequence + Int32(1)

        if peer != own_rank and channel_chunks > Int32(0):
            # Retire the final, possibly partial, send window.  Earlier
            # windows are retired immediately before their slots are reused.
            gin_bridge.quiet(self)
            cute.arch.barrier(barrier_id=0, number_of_threads=128)

    @cute.jit
    def _send_one_chunk(
        self,
        peer,
        channel_index,
        sequence,
        chunk_idx,
        count,
        rx_slots,
        tx_slots,
        fc2_workspace,
        send_output_rows,
        combine_output,
        own_rank,
        epoch,
        lane_idx,
    ):
        """Submit one already-ready combine chunk without waiting."""
        row_base = chunk_idx * Int32(self.chunk_rows)
        fc2_i32 = cute.recast_tensor(fc2_workspace, cutlass.Int32)
        output_i32 = cute.recast_tensor(combine_output, cutlass.Int32)
        fc2_u128 = cute.recast_tensor(fc2_workspace, cutlass.Uint128)
        output_u128 = cute.recast_tensor(combine_output, cutlass.Uint128)
        if peer == own_rank:
            warp_idx = lane_idx // Int32(32)
            lane_in_warp = lane_idx % Int32(32)
            row_in_chunk = warp_idx
            while (
                row_in_chunk < Int32(self.chunk_rows)
                and row_base + row_in_chunk < count
            ):
                metadata_index = (
                    own_rank * Int32(self.max_rows_per_peer)
                    + row_base
                    + row_in_chunk
                )
                source_row = (
                    own_rank * Int32(self.max_rows_per_peer)
                    + row_base
                    + row_in_chunk
                )
                output_row = send_output_rows[metadata_index]
                if cutlass.const_expr(self.vectorized_scatter):
                    src_base_u128 = source_row * Int64(self.payload_u128)
                    dst_base_u128 = output_row * Int64(self.payload_u128)
                    vector = lane_in_warp
                    while vector < Int32(self.payload_u128):
                        output_u128[dst_base_u128 + Int64(vector)] = fc2_u128[
                            src_base_u128 + Int64(vector)
                        ]
                        vector = vector + Int32(32)
                else:
                    src_base = source_row * Int64(self.payload_i32)
                    dst_base = output_row * Int64(self.payload_i32)
                    word = lane_in_warp
                    while word < Int32(self.payload_i32):
                        output_i32[dst_base + Int64(word)] = fc2_i32[
                            src_base + Int64(word)
                        ]
                        word = word + Int32(32)
                row_in_chunk = row_in_chunk + Int32(4)
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
        else:
            tx_i32 = cute.recast_tensor(tx_slots, cutlass.Int32)
            epoch_base = epoch * Int32(self.signal_stride)
            slot = (sequence - Int32(1)) % Int32(self.slots)
            signal_index = (
                (peer * Int32(self.channels) + channel_index)
                * Int32(self.slots)
                + slot
            )
            stage_row_base = peer * Int32(self.max_rows_per_peer) + row_base
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            remote_slot_index = (
                (own_rank * Int32(self.channels) + channel_index)
                * Int32(self.slots)
                + slot
            )
            local_payload = cute.make_tensor(
                fc2_workspace.iterator
                + Int64(stage_row_base) * Int64(self.payload_bytes),
                cute.make_layout(self.chunk_rows * self.payload_bytes),
            )
            remote_payload = cute.make_tensor(
                rx_slots.iterator
                + Int64(remote_slot_index * Int32(self.slot_bytes))
                + Int64(self.row_id_chunk_bytes),
                cute.make_layout(self.chunk_rows * self.payload_bytes),
            )
            remote_signal = cute.make_tensor(
                rx_slots.iterator
                + Int64(remote_slot_index * Int32(self.slot_bytes))
                + Int64(self.signal_offset_bytes),
                cute.make_layout(8),
            )
            local_signal = cute.make_tensor(
                tx_slots.iterator
                + Int64(signal_index * Int32(self.slot_bytes))
                + Int64(self.signal_offset_bytes),
                cute.make_layout(8),
            )
            if lane_idx == Int32(0):
                tx_i32[
                    signal_index * Int32(self.slot_i32)
                    + Int32(self.signal_offset_bytes // 4)
                ] = epoch_base + sequence
                tx_i32[
                    signal_index * Int32(self.slot_i32)
                    + Int32(self.signal_offset_bytes // 4 + 1)
                ] = chunk_idx
            if cutlass.const_expr(self.gin_enabled):
                if (Int32(self.hybrid_force_ibgda_mask) & (Int32(1) << peer)) != Int32(0):
                    gin_bridge.put_fused(self, remote_payload, local_payload, remote_signal, local_signal, peer)
                else:
                    if cutlass.const_expr(self.blocking_put):
                        gin_bridge.put_block(self, remote_payload, local_payload, peer)
                        cute.arch.barrier(barrier_id=0, number_of_threads=128)
                        gin_bridge.fence()
                        cute.arch.barrier(barrier_id=0, number_of_threads=128)
                        gin_bridge.put_block(self, remote_signal, local_signal, peer)
                        cute.arch.barrier(barrier_id=0, number_of_threads=128)
                    else:
                        gin_bridge.put_nbi_block(self,
                            remote_payload, local_payload, peer
                        )
                        cute.arch.barrier(barrier_id=0, number_of_threads=128)
                        gin_bridge.fence()
                        cute.arch.barrier(barrier_id=0, number_of_threads=128)
                        cute.arch.fence_acq_rel_sys()
                        cute.arch.barrier(barrier_id=0, number_of_threads=128)
                        num_chunks = (
                            count + Int32(self.chunk_rows) - Int32(1)
                        ) // Int32(self.chunk_rows)
                        channel_chunks = (
                            (num_chunks - Int32(1) - channel_index)
                            // Int32(self.channels)
                            + Int32(1)
                        )
                        # The final data commit is the peer/channel producer
                        # completion record.  Complete it remotely so stream
                        # completion never relies on a terminal quiet; earlier
                        # commits remain nonblocking and overlap K2 staging.
                        if (
                            cutlass.const_expr(self.progress_windows)
                            and sequence == channel_chunks
                        ):
                            gin_bridge.put_block(self,
                                remote_signal, local_signal, peer
                            )
                        else:
                            gin_bridge.put_nbi_block(self,
                                remote_signal, local_signal, peer
                            )
                        cute.arch.barrier(barrier_id=0, number_of_threads=128)
            else:
                if cutlass.const_expr(self.blocking_put):
                    gin_bridge.put_block(self, remote_payload, local_payload, peer)
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)
                    gin_bridge.fence()
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)
                    gin_bridge.put_block(self, remote_signal, local_signal, peer)
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)
                else:
                    gin_bridge.put_nbi_block(self,
                        remote_payload, local_payload, peer
                    )
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)
                    gin_bridge.fence()
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)
                    num_chunks = (
                        count + Int32(self.chunk_rows) - Int32(1)
                    ) // Int32(self.chunk_rows)
                    channel_chunks = (
                        (num_chunks - Int32(1) - channel_index)
                        // Int32(self.channels)
                        + Int32(1)
                    )
                    # The final data commit is the peer/channel producer
                    # completion record.  Complete it remotely so stream
                    # completion never relies on a terminal quiet; earlier
                    # commits remain nonblocking and overlap K2 staging.
                    if (
                        cutlass.const_expr(self.progress_windows)
                        and sequence == channel_chunks
                    ):
                        gin_bridge.put_block(self,
                            remote_signal, local_signal, peer
                        )
                    else:
                        gin_bridge.put_nbi_block(self,
                            remote_signal, local_signal, peer
                        )
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)

    @cute.jit
    def launcher(
        self,
        rx_slots: cute.Tensor,
        tx_slots: cute.Tensor,
        ack_signals: cute.Tensor,
        fc2_workspace: cute.Tensor,
        send_output_rows: cute.Tensor,
        send_ready_slots: cute.Tensor,
        send_ready_values: cute.Tensor,
        send_counts: cute.Tensor,
        combine_output: cute.Tensor,
        fc2_done_counter: cute.Tensor,
        ready_queue_done: cute.Tensor,
        ready_queue_probe: cute.Tensor,
        ready_queue_selected: cute.Tensor,
        tx_handoff: cute.Tensor,
        work_queue: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        # One CTA has exclusive lifetime ownership of one (peer, channel).
        # Its four warps select ready chunks within that channel; no global
        # claimant/completion counter participates in the production path.
        self.kernel(
            rx_slots,
            tx_slots,
            ack_signals,
            fc2_workspace,
            send_output_rows,
            send_ready_slots,
            send_ready_values,
            send_counts,
            combine_output,
            fc2_done_counter,
            ready_queue_done,
            ready_queue_probe,
            ready_queue_selected,
            tx_handoff,
            own_rank,
            epoch,
        ).launch(
            grid=[self.world_size, self.channels, 1],
            block=[128, 1, 1],
            stream=stream,
            min_blocks_per_mp=(1 if self.progress_windows else 2),
        )

    @cute.kernel
    def ack_kernel(
        self,
        ack_signals: cute.Tensor,
        send_counts: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        """Retire terminal credits after combine data submission.

        K3 only depends on receiver completion. Keeping terminal ownership
        recovery in the ordered TX stream preserves replay safety without
        putting the final network ACK on the current replay's critical path.
        """
        tidx, _, _ = cute.arch.thread_idx()
        ack_i32 = cute.recast_tensor(ack_signals, cutlass.Int32)
        epoch_base = epoch * Int32(self.signal_stride)
        # Every lane executes the same non-blocking scan.  A single stalled
        # peer/channel must not hold a resident polling CTA for that channel
        # and starve the receiver CTA that would produce its terminal ACK.
        if tidx >= Int32(0):
            finished = Int32(0)
            while finished == Int32(0):
                all_ready = Int32(1)
                channel_index = Int32(0)
                while channel_index < Int32(self.channels):
                    peer = Int32(0)
                    while peer < Int32(self.world_size):
                        count = send_counts[peer]
                        if peer != own_rank:
                            num_chunks = (
                                count + Int32(self.chunk_rows) - Int32(1)
                            ) // Int32(self.chunk_rows)
                            if channel_index < num_chunks:
                                last_sequence = (
                                    (
                                        num_chunks
                                        - Int32(1)
                                        - channel_index
                                    )
                                    // Int32(self.channels)
                                    + Int32(1)
                                )
                                last_slot = (
                                    last_sequence - Int32(1)
                                ) % Int32(self.slots)
                                last_signal_index = (
                                    (
                                        peer * Int32(self.channels)
                                        + channel_index
                                    )
                                    * Int32(self.slots)
                                    + last_slot
                                )
                                observed = ldg_b32_cv_raw(
                                    ack_i32.iterator.toint()
                                    + Int64(last_signal_index) * Int64(128)
                                )
                                if observed < epoch_base + last_sequence:
                                    all_ready = Int32(0)
                        peer = peer + Int32(1)
                    channel_index = channel_index + Int32(1)
                finished = all_ready
                if finished == Int32(0):
                    _nanosleep(500)

    @cute.jit
    def ack_launcher(
        self,
        ack_signals: cute.Tensor,
        send_counts: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.ack_kernel(
            ack_signals,
            send_counts,
            own_rank,
            epoch,
        ).launch(
            grid=[1, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )

class GinGpuRingCombineReceiver:
    """Wait for ready row batches, scatter them, and return slot ownership."""

    def __init__(
        self,
        world_size: int,
        payload_bytes: int,
        max_rows_per_peer: int,
        *,
        chunk_rows: int = 8,
        channels: int = 4,
        slots: int = 2,
        hybrid_force_ibgda_mask: int,
        worker_ctas: int,
        num_experts_per_rank: int,
        count_block_elements: int,
        plan_slots: int,
        progress_windows: bool,
    ) -> None:
        gin_bridge.bind(self, 'return_rx')
        self.world_size = world_size
        self.hybrid_force_ibgda_mask = hybrid_force_ibgda_mask
        self.payload_i32 = payload_bytes // 4
        self.payload_u128 = payload_bytes // 16
        self.chunk_rows = chunk_rows
        self.row_id_chunk_bytes = 4 * chunk_rows
        self.chunk_payload_bytes = (
            self.row_id_chunk_bytes + payload_bytes * chunk_rows
        )
        self.signal_offset_bytes = (
            (self.chunk_payload_bytes + 7) // 8
        ) * 8
        self.slot_bytes = (
            (self.signal_offset_bytes + 8 + 15) // 16
        ) * 16
        self.slot_i32 = self.slot_bytes // 4
        self.max_rows_per_peer = max_rows_per_peer
        self.max_chunks_per_peer = (
            max_rows_per_peer + chunk_rows - 1
        ) // chunk_rows
        self.channels = channels
        self.slots = slots
        self.worker_ctas = min(worker_ctas, world_size * channels)
        if self.worker_ctas <= 0:
            raise ValueError("combine receiver worker_ctas must be positive")
        if self.worker_ctas % world_size:
            raise ValueError(
                "combine receiver requires an integral CTA stripe per peer"
            )
        if self.worker_ctas != world_size * channels:
            raise ValueError(
                "combine receiver requires one owner CTA per peer/channel"
            )
        self.num_experts_per_rank = num_experts_per_rank
        self.count_block_elements = count_block_elements
        self.plan_slots = plan_slots
        self.progress_windows = progress_windows
        self.vectorized_scatter = (
            payload_bytes % 16 == 0 and self.row_id_chunk_bytes % 16 == 0
        )
        self.plan_row_offset_words = COMBINE_PLAN_ROW_OFFSET_WORDS
        self.plan_record_words = combine_plan_record_words(max_rows_per_peer)
        self.plan_signal_word_offset = self.plan_record_words - 2
        self.signal_stride = (
            (self.max_chunks_per_peer + channels - 1) // channels + 1
        )

    @cute.jit
    def _receive_one_chunk(
        self,
        peer,
        channel_index,
        sequence,
        count,
        rx_slots,
        ack_rx,
        ack_tx,
        recv_output_rows,
        combine_output,
        own_rank,
        epoch,
        lane_idx,
    ):
        """Scatter one arrived combine chunk and return its slot credit."""
        epoch_base = epoch * Int32(self.signal_stride)
        slot = (sequence - Int32(1)) % Int32(self.slots)
        signal_index = (
            (peer * Int32(self.channels) + channel_index)
            * Int32(self.slots)
            + slot
        )
        rx_i32 = cute.recast_tensor(rx_slots, cutlass.Int32)
        ack_tx_i32 = cute.recast_tensor(ack_tx, cutlass.Int32)
        output_i32 = cute.recast_tensor(combine_output, cutlass.Int32)
        rx_u128 = cute.recast_tensor(rx_slots, cutlass.Uint128)
        output_u128 = cute.recast_tensor(combine_output, cutlass.Uint128)
        src_base = Int64(signal_index * Int32(self.slot_i32))
        source_chunk_idx = gin_bridge.read_commit_chunk(self, peer, rx_i32.iterator.toint()
            + (
                src_base + Int64(self.signal_offset_bytes // 4)
            )
            * Int64(4))
        row_base = source_chunk_idx * Int32(self.chunk_rows)
        warp_idx = lane_idx // Int32(32)
        lane_in_warp = lane_idx % Int32(32)
        row_in_chunk = warp_idx
        while (
            row_in_chunk < Int32(self.chunk_rows)
            and row_base + row_in_chunk < count
        ):
            output_row = recv_output_rows[
                peer * Int32(self.max_rows_per_peer)
                + row_base
                + row_in_chunk
            ]
            if cutlass.const_expr(self.vectorized_scatter):
                src_row_base_u128 = (
                    Int64(signal_index * Int32(self.slot_bytes // 16))
                    + Int64(self.row_id_chunk_bytes // 16)
                    + Int64(row_in_chunk * Int32(self.payload_u128))
                )
                dst_base_u128 = output_row * Int64(self.payload_u128)
                vector = lane_in_warp
                while vector < Int32(self.payload_u128):
                    output_u128[dst_base_u128 + Int64(vector)] = rx_u128[
                        src_row_base_u128 + Int64(vector)
                    ]
                    vector = vector + Int32(32)
            else:
                src_row_base = (
                    src_base
                    + Int64(self.row_id_chunk_bytes // 4)
                    + Int64(row_in_chunk * Int32(self.payload_i32))
                )
                dst_base = output_row * Int64(self.payload_i32)
                word = lane_in_warp
                while word < Int32(self.payload_i32):
                    output_i32[dst_base + Int64(word)] = rx_i32[
                        src_row_base + Int64(word)
                    ]
                    word = word + Int32(32)
            row_in_chunk = row_in_chunk + Int32(4)
        cute.arch.barrier(barrier_id=0, number_of_threads=128)
        cute.arch.fence_acq_rel_sys()
        local_ack_index = (
            (peer * Int32(self.channels) + channel_index)
            * Int32(self.slots)
            + slot
        )
        if lane_idx == Int32(0):
            ack_tx_i32[local_ack_index * Int32(32)] = epoch_base + sequence
        cute.arch.barrier(barrier_id=0, number_of_threads=128)
        cute.arch.fence_acq_rel_sys()
        remote_ack_index = (
            (own_rank * Int32(self.channels) + channel_index)
            * Int32(self.slots)
            + slot
        )
        local_ack = cute.make_tensor(
            ack_tx.iterator + local_ack_index * Int32(128),
            cute.make_layout(128),
        )
        remote_ack = cute.make_tensor(
            ack_rx.iterator + remote_ack_index * Int32(128),
            cute.make_layout(128),
        )
        gin_bridge.put_block(self, remote_ack, local_ack, peer)
        cute.arch.barrier(barrier_id=0, number_of_threads=128)

    @cute.kernel
    def owned_channel_kernel(
        self,
        rx_slots: cute.Tensor,
        ack_rx: cute.Tensor,
        ack_tx: cute.Tensor,
        count_records: cute.Tensor,
        recv_output_rows: cute.Tensor,
        combine_output: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        """Receive exactly one peer/channel with deterministic CTA ownership."""
        task, _, _ = cute.arch.block_idx()
        lane_idx, _, _ = cute.arch.thread_idx()
        peer = task // Int32(self.channels)
        channel_index = task % Int32(self.channels)
        epoch_base = epoch * Int32(self.signal_stride)
        rx_i32 = cute.recast_tensor(rx_slots, cutlass.Int32)

        _prepare_hybrid_peer(
            peer,
            own_rank,
            lane_idx,
            force_ibgda_mask=self.hybrid_force_ibgda_mask,
        )
        count = Int32(0)
        if peer != own_rank:
            count_base = (
                (epoch % Int32(self.plan_slots))
                * Int32(self.world_size * self.count_block_elements)
                + peer * Int32(self.count_block_elements)
            )
            for expert in cutlass.range_constexpr(
                0, self.num_experts_per_rank, 1
            ):
                count = count + Int32(
                    count_records[count_base + Int32(expert)]
                )

        num_chunks = (
            count + Int32(self.chunk_rows) - Int32(1)
        ) // Int32(self.chunk_rows)
        channel_chunks = Int32(0)
        if channel_index < num_chunks:
            channel_chunks = (
                (num_chunks - Int32(1) - channel_index)
                // Int32(self.channels)
                + Int32(1)
            )

        sequence = Int32(1)
        while sequence <= channel_chunks:
            if (
                cutlass.const_expr(self.progress_windows)
                and peer != own_rank
                and sequence > Int32(self.slots)
                and (sequence - Int32(1)) % Int32(self.slots)
                == Int32(0)
            ):
                # Complete the preceding ACK window before the sender can
                # need its first credit to reuse a cyclic slot.
                gin_bridge.quiet(self)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
            slot = (sequence - Int32(1)) % Int32(self.slots)
            signal_index = (
                (peer * Int32(self.channels) + channel_index)
                * Int32(self.slots)
                + slot
            )
            if lane_idx == Int32(0):
                gin_bridge.wait_commit32(self, peer,
                    rx_i32.iterator.toint()
                    + (
                        Int64(signal_index * Int32(self.slot_i32))
                        + Int64(self.signal_offset_bytes // 4)
                    )
                    * Int64(4),
                    epoch_base + sequence,
                )
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            cute.arch.fence_acq_rel_sys()
            self._receive_one_chunk(
                peer,
                channel_index,
                sequence,
                count,
                rx_slots,
                ack_rx,
                ack_tx,
                recv_output_rows,
                combine_output,
                own_rank,
                epoch,
                lane_idx,
            )
            sequence = sequence + Int32(1)

        if (
            cutlass.const_expr(self.progress_windows)
            and peer != own_rank
            and channel_chunks > Int32(0)
        ):
            gin_bridge.quiet(self)
            cute.arch.barrier(barrier_id=0, number_of_threads=128)

    @cute.jit
    def launcher(
        self,
        rx_slots: cute.Tensor,
        ack_rx: cute.Tensor,
        ack_tx: cute.Tensor,
        count_records: cute.Tensor,
        recv_output_rows: cute.Tensor,
        combine_output: cute.Tensor,
        work_queue: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.owned_channel_kernel(
            rx_slots,
            ack_rx,
            ack_tx,
            count_records,
            recv_output_rows,
            combine_output,
            own_rank,
            epoch,
        ).launch(
            grid=[self.worker_ctas, 1, 1],
            block=[128, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )


class GinGpuRingDispatchSender:
    """Pack only tokens routed to each expert owner into bounded slots."""

    def __init__(
        self,
        world_size: int,
        num_tokens: int,
        activation_bytes_per_token: int,
        sf_bytes_per_token: int,
        weight_bytes_per_token: int,
        *,
        chunk_tokens: int = 64,
        channels: int = 4,
        slots: int = 2,
        local_handoff_windows: int,
        remote_handoff_windows: int,
        hybrid_force_ibgda_mask: int,
        worker_ctas: int,
    ) -> None:
        gin_bridge.bind(self, 'dispatch_tx')
        self.world_size = world_size
        self.hybrid_force_ibgda_mask = hybrid_force_ibgda_mask
        self.num_tokens = num_tokens
        self.chunk_tokens = chunk_tokens
        self.num_chunks = (num_tokens + chunk_tokens - 1) // chunk_tokens
        self.act_bpt = activation_bytes_per_token
        self.sf_bpt = sf_bytes_per_token
        self.weight_bpt = weight_bytes_per_token
        if (
            activation_bytes_per_token % 4
            or sf_bytes_per_token % 4
            or weight_bytes_per_token % 4
        ):
            raise ValueError("sparse dispatch fields must be 4-byte aligned")
        self.token_id_chunk_bytes = chunk_tokens * 4
        self.act_chunk_bytes = chunk_tokens * activation_bytes_per_token
        self.sf_chunk_bytes = chunk_tokens * sf_bytes_per_token
        self.weight_chunk_bytes = chunk_tokens * weight_bytes_per_token
        self.payload_bytes = (
            self.token_id_chunk_bytes
            + self.act_chunk_bytes
            + self.sf_chunk_bytes
            + self.weight_chunk_bytes
        )
        self.signal_offset_bytes = (self.payload_bytes + 7) // 8 * 8
        self.slot_bytes = (
            self.signal_offset_bytes + 8 + 15
        ) // 16 * 16
        self.channels = channels
        self.slots = slots
        self.worker_ctas = min(worker_ctas, world_size * channels)
        if self.worker_ctas <= 0:
            raise ValueError("dispatch worker_ctas must be positive")
        if self.worker_ctas != world_size * channels:
            raise ValueError(
                "dispatch sender requires one owner CTA per peer/channel"
            )
        if local_handoff_windows <= 0 or remote_handoff_windows <= 0:
            raise ValueError("dispatch handoff windows must be positive")
        self.local_handoff_windows = local_handoff_windows
        self.remote_handoff_windows = remote_handoff_windows
        self.signal_stride = (
            (self.num_chunks + channels - 1) // channels + 1
        )

    @cute.kernel
    def kernel(
        self,
        rx_slots: cute.Tensor,
        tx_slots: cute.Tensor,
        ack_payload: cute.Tensor,
        control_rx: cute.Tensor,
        control_tx: cute.Tensor,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        topk_weights: cute.Tensor,
        send_token_indices: cute.Tensor,
        send_counts: cute.Tensor,
        tx_handoff: cute.Tensor,
        work_queue: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        worker_index, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        # Use the same fixed strided ownership as dispatch RX.  Matching the
        # first resident waves prevents bounded-slot TX from waiting on ACKs
        # owned by receiver CTAs that have not been scheduled yet.
        task = worker_index
        while task < Int32(self.world_size * self.channels):
            channel_index = task // Int32(self.world_size)
            peer = task % Int32(self.world_size)
            _prepare_hybrid_peer(
                peer,
                own_rank,
                tidx,
                force_ibgda_mask=self.hybrid_force_ibgda_mask,
            )
            count = send_counts[peer]
            if peer == own_rank:
                count = Int32(0)
            tx_i32 = cute.recast_tensor(tx_slots, cutlass.Int32)
            ack_i32 = cute.recast_tensor(ack_payload, cutlass.Int32)
            control_rx_i32 = cute.recast_tensor(control_rx, cutlass.Int32)
            control_tx_i32 = cute.recast_tensor(control_tx, cutlass.Int32)
            act_i32 = cute.recast_tensor(activation, cutlass.Int32)
            sf_i32 = cute.recast_tensor(activation_sf, cutlass.Int32)
            weight_i32 = cute.recast_tensor(topk_weights, cutlass.Int32)
            if channel_index == Int32(0) and peer != own_rank:
                control_base = peer * Int32(32)
                if tidx == Int32(0):
                    control_tx_i32[control_base] = count
                    control_tx_i32[
                        control_base
                        + Int32(_CONTROL_SIGNAL_BYTE_OFFSET // 4)
                    ] = epoch
                    control_tx_i32[
                        control_base
                        + Int32(_CONTROL_SIGNAL_BYTE_OFFSET // 4 + 1)
                    ] = Int32(0)
                cute.arch.fence_acq_rel_sys()
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                local_count = cute.make_tensor(
                    control_tx.iterator
                    + peer * Int32(_CONTROL_RECORD_BYTES),
                    cute.make_layout(4),
                )
                remote_count = cute.make_tensor(
                    control_rx.iterator
                    + own_rank * Int32(_CONTROL_RECORD_BYTES),
                    cute.make_layout(4),
                )
                remote_signal = cute.make_tensor(
                    control_rx.iterator
                    + own_rank * Int32(_CONTROL_RECORD_BYTES)
                    + Int32(_CONTROL_SIGNAL_BYTE_OFFSET),
                    cute.make_layout(8),
                )
                local_signal = cute.make_tensor(
                    control_tx.iterator
                    + peer * Int32(_CONTROL_RECORD_BYTES)
                    + Int32(_CONTROL_SIGNAL_BYTE_OFFSET),
                    cute.make_layout(8),
                )
                gin_bridge.put_block(self, remote_count, local_count, peer)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                gin_bridge.fence()
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                cute.arch.fence_acq_rel_sys()
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                if cutlass.const_expr(self.num_tokens <= 128):
                    gin_bridge.put_block(self,
                        remote_signal, local_signal, peer
                    )
                else:
                    gin_bridge.put_nbi_block(self,
                        remote_signal, local_signal, peer
                    )
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                if cutlass.const_expr(self.num_tokens <= 128):
                    # A zero-count peer has no later data operation to drive
                    # this admission commit to remote completion.
                    gin_bridge.quiet(self)
                    cute.arch.barrier(
                        barrier_id=0, number_of_threads=128
                    )
            epoch_base = epoch * Int32(self.signal_stride)
            payload_i32 = self.payload_bytes // 4
            slot_i32 = self.slot_bytes // 4
            num_chunks = (
                count + Int32(self.chunk_tokens) - Int32(1)
            ) // Int32(self.chunk_tokens)
            channel_chunks = Int32(0)
            if channel_index < num_chunks:
                channel_chunks = (
                    (num_chunks - Int32(1) - channel_index)
                    // Int32(self.channels)
                    + Int32(1)
                )
            handoff_windows = Int32(self.remote_handoff_windows)
            if cutlass.const_expr(self.hybrid_force_ibgda_mask > 0):
                if not _peer_uses_ibgda(
                    peer,
                    force_ibgda_mask=self.hybrid_force_ibgda_mask,
                ):
                    handoff_windows = Int32(self.local_handoff_windows)
            if channel_chunks == Int32(0) and tidx == Int32(0):
                _publish_tx_handoff_epoch(
                    tx_handoff,
                    peer * Int32(self.channels) + channel_index,
                    epoch,
                )
            chunk_idx = channel_index
            while chunk_idx * Int32(self.chunk_tokens) < count:
                channel = channel_index
                sequence = chunk_idx // Int32(self.channels) + Int32(1)
                slot = (sequence - Int32(1)) % Int32(self.slots)
                signal_index = (
                    (peer * Int32(self.channels) + channel) * Int32(self.slots)
                    + slot
                )
                if sequence > Int32(self.slots):
                    flush_remote_window = peer != own_rank
                    if cutlass.const_expr(self.hybrid_force_ibgda_mask > 0):
                        flush_remote_window = _peer_uses_ibgda(
                            peer,
                            force_ibgda_mask=self.hybrid_force_ibgda_mask,
                        )
                    if cutlass.const_expr(self.act_bpt >= 6144):
                        if (
                            flush_remote_window
                            and (sequence - Int32(1)) % Int32(self.slots)
                            == Int32(0)
                        ):
                            # Wide rows need an explicit producer progress
                            # point before their cyclic credit wait; smaller
                            # rows make better progress with pure NBI overlap.
                            gin_bridge.quiet(self)
                            cute.arch.barrier(
                                barrier_id=0, number_of_threads=128
                            )
                    if tidx == Int32(0):
                        spin_wait_i32_ge_cv_raw(
                            ack_i32.iterator.toint()
                            + Int64(signal_index) * Int64(128),
                            epoch_base + sequence - Int32(self.slots),
                        )
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)

                tx_base = Int64(signal_index * Int32(slot_i32))
                row_base = chunk_idx * Int32(self.chunk_tokens)
                act_words_per_token = self.act_bpt // 4
                sf_words_per_token = self.sf_bpt // 4
                weight_words_per_token = self.weight_bpt // 4
                token_id_words = self.chunk_tokens
                act_chunk_words = self.act_chunk_bytes // 4
                sf_chunk_words = self.sf_chunk_bytes // 4
                row = Int32(0)
                while row < Int32(self.chunk_tokens):
                    list_pos = row_base + row
                    token_idx = Int32(-1)
                    if list_pos < count:
                        token_idx = send_token_indices[
                            peer * Int32(self.num_tokens) + list_pos
                        ]
                    if tidx == Int32(0):
                        tx_i32[tx_base + Int64(row)] = token_idx
                    if list_pos < count:
                        word = tidx
                        while word < Int32(act_words_per_token):
                            tx_i32[
                                tx_base
                                + Int64(token_id_words)
                                + Int64(row * Int32(act_words_per_token) + word)
                            ] = act_i32[
                                token_idx * Int32(act_words_per_token) + word
                            ]
                            word = word + Int32(128)
                        word = tidx
                        while word < Int32(sf_words_per_token):
                            tx_i32[
                                tx_base
                                + Int64(token_id_words + act_chunk_words)
                                + Int64(row * Int32(sf_words_per_token) + word)
                            ] = sf_i32[
                                token_idx * Int32(sf_words_per_token) + word
                            ]
                            word = word + Int32(128)
                        word = tidx
                        while word < Int32(weight_words_per_token):
                            tx_i32[
                                tx_base
                                + Int64(
                                    token_id_words
                                    + act_chunk_words
                                    + sf_chunk_words
                                )
                                + Int64(
                                    row * Int32(weight_words_per_token) + word
                                )
                            ] = weight_i32[
                                token_idx * Int32(weight_words_per_token) + word
                            ]
                            word = word + Int32(128)
                    row = row + Int32(1)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                cute.arch.fence_acq_rel_sys()
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                remote_index = (
                    (own_rank * Int32(self.channels) + channel)
                    * Int32(self.slots)
                    + slot
                )
                remote_payload = cute.make_tensor(
                    rx_slots.iterator + remote_index * Int32(self.slot_bytes),
                    cute.make_layout(self.payload_bytes),
                )
                local_payload = cute.make_tensor(
                    tx_slots.iterator + signal_index * Int32(self.slot_bytes),
                    cute.make_layout(self.payload_bytes),
                )
                remote_signal = cute.make_tensor(
                    rx_slots.iterator
                    + Int64(remote_index * Int32(self.slot_bytes))
                    + Int64(self.signal_offset_bytes),
                    cute.make_layout(8),
                )
                local_signal = cute.make_tensor(
                    tx_slots.iterator
                    + Int64(signal_index * Int32(self.slot_bytes))
                    + Int64(self.signal_offset_bytes),
                    cute.make_layout(8),
                )
                if tidx == Int32(0):
                    tx_i32[
                        signal_index * Int32(slot_i32)
                        + Int32(self.signal_offset_bytes // 4)
                    ] = epoch_base + sequence
                    tx_i32[
                        signal_index * Int32(slot_i32)
                        + Int32(self.signal_offset_bytes // 4 + 1)
                    ] = Int32(0)
                if cutlass.const_expr(self.num_tokens <= 128):
                    gin_bridge.put_block(self,
                        remote_payload, local_payload, peer
                    )
                else:
                    gin_bridge.put_nbi_block(self,
                        remote_payload, local_payload, peer
                    )
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                gin_bridge.fence()
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                cute.arch.fence_acq_rel_sys()
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                # Preserve payload-before-commit ordering while allowing
                # intermediate commits to overlap the next chunk.  The final
                # commit is blocking: once this CTA exits, its peer/channel
                # receiver must not depend on a later communication operation to
                # flush the last NBI signal.
                if cutlass.const_expr(self.num_tokens <= 128):
                    gin_bridge.put_block(self,
                        remote_signal, local_signal, peer
                    )
                else:
                    gin_bridge.put_nbi_block(self,
                        remote_signal, local_signal, peer
                    )
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                if (
                    sequence == Int32(self.slots) * handoff_windows
                    and tidx == Int32(0)
                ):
                    _publish_tx_handoff_epoch(
                        tx_handoff,
                        peer * Int32(self.channels) + channel_index,
                        epoch,
                    )
                chunk_idx = chunk_idx + Int32(self.channels)

            if (
                cutlass.const_expr(self.num_tokens <= 128)
                and peer != own_rank
                and channel_chunks > Int32(0)
            ):
                gin_bridge.quiet(self)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)

            if (
                channel_chunks > Int32(0)
                and channel_chunks
                < Int32(self.slots) * handoff_windows
                and tidx == Int32(0)
            ):
                _publish_tx_handoff_epoch(
                    tx_handoff,
                    peer * Int32(self.channels) + channel_index,
                    epoch,
                )

            task = task + Int32(self.worker_ctas)

    @cute.kernel
    def ack_kernel(
        self,
        ack_payload: cute.Tensor,
        send_counts: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        """Wait for terminal slot ownership after payload submission.

        Keeping this separate from the producer lets combine use the TX SM
        partition once every dispatch payload has been issued, while this
        lightweight waiter still prevents the next replay from recycling a
        slot before its peer has consumed it.
        """
        tidx, _, _ = cute.arch.thread_idx()
        ack_i32 = cute.recast_tensor(ack_payload, cutlass.Int32)
        epoch_base = epoch * Int32(self.signal_stride)
        if tidx >= Int32(0):
            finished = Int32(0)
            while finished == Int32(0):
                all_ready = Int32(1)
                channel_index = Int32(0)
                while channel_index < Int32(self.channels):
                    peer = Int32(0)
                    while peer < Int32(self.world_size):
                        count = send_counts[peer]
                        if peer != own_rank:
                            num_chunks = (
                                count + Int32(self.chunk_tokens) - Int32(1)
                            ) // Int32(self.chunk_tokens)
                            if channel_index < num_chunks:
                                last_sequence = (
                                    (num_chunks - Int32(1) - channel_index)
                                    // Int32(self.channels)
                                    + Int32(1)
                                )
                                last_slot = (
                                    last_sequence - Int32(1)
                                ) % Int32(self.slots)
                                last_signal_index = (
                                    (
                                        peer * Int32(self.channels)
                                        + channel_index
                                    )
                                    * Int32(self.slots)
                                    + last_slot
                                )
                                observed = ldg_b32_cv_raw(
                                    ack_i32.iterator.toint()
                                    + Int64(last_signal_index) * Int64(128)
                                )
                                if observed < epoch_base + last_sequence:
                                    all_ready = Int32(0)
                        peer = peer + Int32(1)
                    channel_index = channel_index + Int32(1)
                finished = all_ready
                if finished == Int32(0):
                    _nanosleep(500)

    @cute.jit
    def launcher(
        self,
        rx_slots: cute.Tensor,
        tx_slots: cute.Tensor,
        ack_payload: cute.Tensor,
        control_rx: cute.Tensor,
        control_tx: cute.Tensor,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        topk_weights: cute.Tensor,
        send_token_indices: cute.Tensor,
        send_counts: cute.Tensor,
        tx_handoff: cute.Tensor,
        work_queue: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            rx_slots,
            tx_slots,
            ack_payload,
            control_rx,
            control_tx,
            activation,
            activation_sf,
            topk_weights,
            send_token_indices,
            send_counts,
            tx_handoff,
            work_queue,
            own_rank,
            epoch,
        ).launch(
            grid=[self.worker_ctas, 1, 1],
            block=[128, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def ack_launcher(
        self,
        ack_payload: cute.Tensor,
        send_counts: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.ack_kernel(
            ack_payload,
            send_counts,
            own_rank,
            epoch,
        ).launch(
            grid=[1, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )


class GinGpuRingDispatchReceiver:
    """Unpack sparse token chunks and publish per-token K1 readiness."""

    def __init__(
        self,
        sender: GinGpuRingDispatchSender,
        *,
        worker_ctas: int,
    ) -> None:
        gin_bridge.bind(self, 'dispatch_rx')
        self.world_size = sender.world_size
        self.hybrid_force_ibgda_mask = sender.hybrid_force_ibgda_mask
        self.num_tokens = sender.num_tokens
        self.chunk_tokens = sender.chunk_tokens
        self.num_chunks = sender.num_chunks
        self.act_bpt = sender.act_bpt
        self.sf_bpt = sender.sf_bpt
        self.weight_bpt = sender.weight_bpt
        self.token_id_chunk_bytes = sender.token_id_chunk_bytes
        self.act_chunk_bytes = sender.act_chunk_bytes
        self.sf_chunk_bytes = sender.sf_chunk_bytes
        self.weight_chunk_bytes = sender.weight_chunk_bytes
        self.payload_bytes = sender.payload_bytes
        self.signal_offset_bytes = sender.signal_offset_bytes
        self.slot_bytes = sender.slot_bytes
        self.channels = sender.channels
        self.slots = sender.slots
        self.worker_ctas = min(
            worker_ctas, self.world_size * self.channels
        )
        if self.worker_ctas <= 0:
            raise ValueError("dispatch receiver worker_ctas must be positive")
        if self.worker_ctas != self.world_size * self.channels:
            raise ValueError(
                "dispatch receiver requires one owner CTA per peer/channel"
            )
        self.signal_stride = sender.signal_stride

    @cute.kernel
    def kernel(
        self,
        rx_slots: cute.Tensor,
        ack_rx: cute.Tensor,
        ack_tx: cute.Tensor,
        control_rx: cute.Tensor,
        work_queue: cute.Tensor,
        activation_rx: cute.Tensor,
        sf_rx: cute.Tensor,
        weight_rx: cute.Tensor,
        dispatch_ready: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        worker_index, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        # Keep TX/RX channel ownership identical across the resident waves.
        task = worker_index
        while task < Int32(self.world_size * self.channels):
            channel_index = task // Int32(self.world_size)
            peer = task % Int32(self.world_size)
            _prepare_hybrid_peer(
                peer,
                own_rank,
                tidx,
                force_ibgda_mask=self.hybrid_force_ibgda_mask,
            )
            control_i32 = cute.recast_tensor(control_rx, cutlass.Int32)
            count = Int32(0)
            if peer != own_rank:
                if tidx == Int32(0):
                    gin_bridge.wait_commit32(self, peer,
                        control_i32.iterator.toint()
                        + Int64(
                            peer * Int32(_CONTROL_RECORD_BYTES)
                            + Int32(_CONTROL_SIGNAL_BYTE_OFFSET)
                        ),
                        epoch,
                    )
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                cute.arch.fence_acq_rel_sys()
                count = ldg_b32_cv_raw(
                    control_i32.iterator.toint()
                    + Int64(peer * Int32(128))
                )
            rx_i32 = cute.recast_tensor(rx_slots, cutlass.Int32)
            ack_tx_i32 = cute.recast_tensor(ack_tx, cutlass.Int32)
            act_i32 = cute.recast_tensor(activation_rx, cutlass.Int32)
            sf_i32 = cute.recast_tensor(sf_rx, cutlass.Int32)
            weight_i32 = cute.recast_tensor(weight_rx, cutlass.Int32)
            epoch_base = epoch * Int32(self.signal_stride)
            payload_i32 = self.payload_bytes // 4
            slot_i32 = self.slot_bytes // 4
            num_chunks = (
                count + Int32(self.chunk_tokens) - Int32(1)
            ) // Int32(self.chunk_tokens)
            channel_chunks = Int32(0)
            if channel_index < num_chunks:
                channel_chunks = (
                    (num_chunks - Int32(1) - channel_index)
                    // Int32(self.channels)
                    + Int32(1)
                )
            chunk_idx = channel_index
            while chunk_idx * Int32(self.chunk_tokens) < count:
                channel = channel_index
                sequence = chunk_idx // Int32(self.channels) + Int32(1)
                slot = (sequence - Int32(1)) % Int32(self.slots)
                signal_index = (
                    (peer * Int32(self.channels) + channel) * Int32(self.slots)
                    + slot
                )
                rx_base = Int64(signal_index * Int32(slot_i32))
                if (
                    cutlass.const_expr(self.num_tokens <= 128)
                    and sequence > Int32(self.slots)
                    and (sequence - Int32(1)) % Int32(self.slots)
                    == Int32(0)
                ):
                    gin_bridge.quiet(self)
                    cute.arch.barrier(
                        barrier_id=0, number_of_threads=128
                    )
                if tidx == Int32(0):
                    gin_bridge.wait_commit32(self, peer,
                        rx_i32.iterator.toint()
                        + (
                            rx_base + Int64(self.signal_offset_bytes // 4)
                        )
                        * Int64(4),
                        epoch_base + sequence,
                    )
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                cute.arch.fence_acq_rel_sys()
                row_base = chunk_idx * Int32(self.chunk_tokens)
                act_words_per_token = self.act_bpt // 4
                sf_words_per_token = self.sf_bpt // 4
                weight_words_per_token = self.weight_bpt // 4
                token_id_words = self.chunk_tokens
                act_chunk_words = self.act_chunk_bytes // 4
                sf_chunk_words = self.sf_chunk_bytes // 4
                peer_act_base = Int64(peer) * Int64(
                    self.num_tokens * act_words_per_token
                )
                peer_sf_base = Int64(peer) * Int64(
                    self.num_tokens * sf_words_per_token
                )
                peer_weight_base = Int64(peer) * Int64(
                    self.num_tokens * weight_words_per_token
                )
                row = Int32(0)
                while row < Int32(self.chunk_tokens):
                    list_pos = row_base + row
                    if list_pos < count:
                        token_idx = ldg_b32_cv_raw(
                            rx_i32.iterator.toint()
                            + (rx_base + Int64(row)) * Int64(4)
                        )
                        word = tidx
                        while word < Int32(act_words_per_token):
                            act_i32[
                                peer_act_base
                                + Int64(token_idx * Int32(act_words_per_token) + word)
                            ] = ldg_b32_cv_raw(
                                rx_i32.iterator.toint()
                                + (
                                    rx_base
                                    + Int64(token_id_words)
                                    + Int64(
                                        row * Int32(act_words_per_token) + word
                                    )
                                )
                                * Int64(4)
                            )
                            word = word + Int32(128)
                        word = tidx
                        while word < Int32(sf_words_per_token):
                            sf_i32[
                                peer_sf_base
                                + Int64(token_idx * Int32(sf_words_per_token) + word)
                            ] = ldg_b32_cv_raw(
                                rx_i32.iterator.toint()
                                + (
                                    rx_base
                                    + Int64(token_id_words + act_chunk_words)
                                    + Int64(
                                        row * Int32(sf_words_per_token) + word
                                    )
                                )
                                * Int64(4)
                            )
                            word = word + Int32(128)
                        word = tidx
                        while word < Int32(weight_words_per_token):
                            weight_i32[
                                peer_weight_base
                                + Int64(
                                    token_idx * Int32(weight_words_per_token) + word
                                )
                            ] = ldg_b32_cv_raw(
                                rx_i32.iterator.toint()
                                + (
                                    rx_base
                                    + Int64(
                                        token_id_words
                                        + act_chunk_words
                                        + sf_chunk_words
                                    )
                                    + Int64(
                                        row * Int32(weight_words_per_token) + word
                                    )
                                )
                                * Int64(4)
                            )
                            word = word + Int32(128)
                    row = row + Int32(1)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                cute.arch.fence_acq_rel_sys()
                ack_index = signal_index
                ready_row = tidx
                while (
                    ready_row < Int32(self.chunk_tokens)
                    and row_base + ready_row < count
                ):
                    token_idx = ldg_b32_cv_raw(
                        rx_i32.iterator.toint()
                        + (rx_base + Int64(ready_row)) * Int64(4)
                    )
                    dispatch_ready[
                        Int32(1)
                        + peer * Int32(self.num_tokens)
                        + token_idx
                    ] = epoch
                    ready_row = ready_row + Int32(128)
                if tidx == Int32(0):
                    ack_tx_i32[ack_index * Int32(32)] = epoch_base + sequence
                cute.arch.fence_acq_rel_sys()
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                local_ack = cute.make_tensor(
                    ack_tx.iterator + ack_index * Int32(128),
                    cute.make_layout(128),
                )
                remote_ack_index = (
                    (own_rank * Int32(self.channels) + channel)
                    * Int32(self.slots)
                    + slot
                )
                remote_ack = cute.make_tensor(
                    ack_rx.iterator + remote_ack_index * Int32(128),
                    cute.make_layout(128),
                )
                # The final credit is the peer/channel completion commit.  Make
                # only that ACK blocking so the receiver event never depends on
                # a terminal quiet, while earlier credits remain overlapped.
                if (
                    cutlass.const_expr(self.num_tokens <= 128)
                    and sequence == channel_chunks
                ):
                    gin_bridge.put_block(self, remote_ack, local_ack, peer)
                else:
                    gin_bridge.put_nbi_block(self, remote_ack, local_ack, peer)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                chunk_idx = chunk_idx + Int32(self.channels)

            if peer != own_rank and channel_chunks > Int32(0):
                gin_bridge.quiet(self)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)

            task = task + Int32(self.worker_ctas)

    @cute.jit
    def launcher(
        self,
        rx_slots: cute.Tensor,
        ack_rx: cute.Tensor,
        ack_tx: cute.Tensor,
        control_rx: cute.Tensor,
        work_queue: cute.Tensor,
        activation_rx: cute.Tensor,
        sf_rx: cute.Tensor,
        weight_rx: cute.Tensor,
        dispatch_ready: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            rx_slots,
            ack_rx,
            ack_tx,
            control_rx,
            work_queue,
            activation_rx,
            sf_rx,
            weight_rx,
            dispatch_ready,
            own_rank,
            epoch,
        ).launch(
            grid=[self.worker_ctas, 1, 1],
            block=[128, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )
