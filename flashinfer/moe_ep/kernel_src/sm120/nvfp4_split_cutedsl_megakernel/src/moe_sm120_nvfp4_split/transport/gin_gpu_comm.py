"""GIN/LSA count-first and expert-chunked owner-plan publication."""

import cutlass.cute as cute
import cutlass
from . import gin_bridge
from cutlass.cutlass_dsl import Int32, Int64




@cute.jit
def _prepare_hybrid_peer(peer, own_rank, lane, *, force_ibgda_mask):
    # Routing is explicit in the GIN/LSA bridge; no library state mutation.
    pass



class GinGpuOwnerPlanPush:
    """Publish count-first, expert-chunked owner plans.

    Data uses aligned 256-int64 blocks. One CTA owns each (peer, expert), so a
    short expert route can commit without waiting behind every other expert of
    the same source. Each CTA publishes a disjoint epoch commit after a device
    fence, mirroring the data-ring payload/commit protocol.
    """

    def __init__(
        self,
        world_size: int,
        num_experts_per_rank: int,
        route_capacity: int,
        *,
        block_elements: int = 256,
        plan_slots: int = 2,
        hybrid_force_ibgda_mask: int,
        completion_credit: bool,
    ) -> None:
        gin_bridge.bind(self, 'dispatch_plan')
        if block_elements != 256:
            raise ValueError("owner-plan block geometry must be 256 int64")
        if num_experts_per_rank >= block_elements:
            raise ValueError("count record does not fit in one owner-plan block")
        self.world_size = world_size
        self.hybrid_force_ibgda_mask = hybrid_force_ibgda_mask
        self.num_experts_per_rank = num_experts_per_rank
        self.route_capacity = route_capacity
        self.block_elements = block_elements
        self.payload_elements = block_elements - 1
        self.commit_elements = 1
        self.plan_slots = plan_slots
        self.completion_credit = completion_credit

    @cute.kernel
    def count_kernel(
        self,
        count_rx: cute.Tensor,
        count_signal_rx: cute.Tensor,
        count_signal_tx: cute.Tensor,
        count_tx: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        """Publish the small count record before either data-plane starts."""
        peer, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        _prepare_hybrid_peer(
            peer,
            own_rank,
            tid,
            force_ibgda_mask=self.hybrid_force_ibgda_mask,
        )
        plan_slot = epoch % Int32(self.plan_slots)
        count_slot_stride = Int32(
            self.world_size * self.block_elements
        )
        count_local_base = (
            plan_slot * count_slot_stride
            + peer * Int32(self.block_elements)
        )
        count_remote_base = (
            plan_slot * count_slot_stride
            + own_rank * Int32(self.block_elements)
        )
        signal_slot_stride = Int32(
            self.world_size * self.commit_elements
        )
        signal_local_base = (
            plan_slot * signal_slot_stride
            + peer * Int32(self.commit_elements)
        )
        signal_remote_base = (
            plan_slot * signal_slot_stride
            + own_rank * Int32(self.commit_elements)
        )
        if peer == own_rank:
            word = tid
            while word < Int32(self.block_elements):
                count_rx[count_remote_base + word] = count_tx[
                    count_local_base + word
                ]
                word = word + Int32(128)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            if tid == Int32(0):
                count_signal_rx[signal_remote_base] = Int64(epoch)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
        else:
            local_count = cute.make_tensor(
                count_tx.iterator + count_local_base,
                cute.make_layout(self.block_elements),
            )
            remote_count = cute.make_tensor(
                count_rx.iterator + count_remote_base,
                cute.make_layout(self.block_elements),
            )
            if tid == Int32(0):
                count_signal_tx[signal_local_base] = Int64(epoch)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            gin_bridge.put_nbi_block(self, remote_count, local_count, peer)
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            gin_bridge.fence()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            local_commit = cute.make_tensor(
                count_signal_tx.iterator + signal_local_base,
                cute.make_layout(self.commit_elements),
            )
            remote_commit = cute.make_tensor(
                count_signal_rx.iterator + signal_remote_base,
                cute.make_layout(self.commit_elements),
            )
            gin_bridge.put_nbi_block(self, remote_commit, local_commit, peer)
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            # This event is also the source-buffer reuse credit for the next
            # replay.  For GIN retire local count/commit sources here; receivers acquire
            # the strong commit before reading.
            # Receive/credit joins protect the next epoch.
            if cutlass.const_expr(self.completion_credit):
                gin_bridge.quiet(self)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)

    @cute.kernel
    def route_kernel(
        self,
        count_tx: cute.Tensor,
        route_rx: cute.Tensor,
        route_signal_rx: cute.Tensor,
        route_signal_tx: cute.Tensor,
        route_tx: cute.Tensor,
        route_bases: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        """Publish expert-major route blocks with one commit per expert."""
        peer, expert, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        _prepare_hybrid_peer(
            peer,
            own_rank,
            tid,
            force_ibgda_mask=self.hybrid_force_ibgda_mask,
        )
        plan_slot = epoch % Int32(self.plan_slots)
        count_slot_stride = Int32(
            self.world_size * self.block_elements
        )
        route_slot_stride = Int32(
            self.world_size * self.route_capacity
        )
        route_signal_slot_stride = Int32(
            self.world_size
            * self.num_experts_per_rank
            * self.commit_elements
        )
        count_local_base = (
            plan_slot * count_slot_stride
            + peer * Int32(self.block_elements)
        )
        count = Int32(count_tx[count_local_base + expert])
        base_index = (
            peer * Int32(self.num_experts_per_rank) + expert
        )
        expert_base = route_bases[base_index]
        chunks = (
            count + Int32(self.payload_elements) - Int32(1)
        ) // Int32(self.payload_elements)
        chunk = Int32(0)
        while chunk < chunks:
            route_offset = (
                expert_base + chunk * Int32(self.block_elements)
            )
            local_base = (
                plan_slot * route_slot_stride
                + peer * Int32(self.route_capacity)
                + route_offset
            )
            remote_base = (
                plan_slot * route_slot_stride
                + own_rank * Int32(self.route_capacity)
                + route_offset
            )
            if peer == own_rank:
                word = tid
                while word < Int32(self.block_elements):
                    route_rx[remote_base + word] = route_tx[
                        local_base + word
                    ]
                    word = word + Int32(128)
                cute.arch.fence_acq_rel_sys()
                cute.arch.barrier(
                    barrier_id=0,
                    number_of_threads=128,
                )
            else:
                local_route = cute.make_tensor(
                    route_tx.iterator + local_base,
                    cute.make_layout(self.block_elements),
                )
                remote_route = cute.make_tensor(
                    route_rx.iterator + remote_base,
                    cute.make_layout(self.block_elements),
                )
                gin_bridge.put_nbi_block(self,
                    remote_route,
                    local_route,
                    peer,
                )
            chunk = chunk + Int32(1)
        signal_local_base = (
            plan_slot * route_signal_slot_stride
            + (
                peer * Int32(self.num_experts_per_rank) + expert
            )
            * Int32(self.commit_elements)
        )
        signal_remote_base = (
            plan_slot * route_signal_slot_stride
            + (
                own_rank * Int32(self.num_experts_per_rank) + expert
            )
            * Int32(self.commit_elements)
        )
        if peer == own_rank:
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(
                barrier_id=0,
                number_of_threads=128,
            )
            if tid == Int32(0):
                route_signal_rx[signal_remote_base] = Int64(epoch)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(
                barrier_id=0,
                number_of_threads=128,
            )
        else:
            if tid == Int32(0):
                route_signal_tx[signal_local_base] = Int64(epoch)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(
                barrier_id=0,
                number_of_threads=128,
            )
            gin_bridge.fence()
            cute.arch.barrier(
                barrier_id=0,
                number_of_threads=128,
            )
            local_commit = cute.make_tensor(
                route_signal_tx.iterator + signal_local_base,
                cute.make_layout(self.commit_elements),
            )
            remote_commit = cute.make_tensor(
                route_signal_rx.iterator + signal_remote_base,
                cute.make_layout(self.commit_elements),
            )
            gin_bridge.put_nbi_block(self,
                remote_commit,
                local_commit,
                peer,
            )
            cute.arch.barrier(
                barrier_id=0,
                number_of_threads=128,
            )

        if (
            cutlass.const_expr(self.completion_credit)
            and peer != own_rank
        ):
            # Preserve per-expert streaming above, but make kernel completion
            # an explicit credit for reusing this replay's route sideband
            # buffers.  GIN source flush and receive/credit joins jointly retire the epoch;
            # CUDA stream completion alone does not retire NBI RMA.
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            gin_bridge.quiet(self)
            cute.arch.barrier(barrier_id=0, number_of_threads=128)

    @cute.jit
    def count_launcher(
        self,
        count_rx: cute.Tensor,
        count_signal_rx: cute.Tensor,
        count_signal_tx: cute.Tensor,
        count_tx: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream,
    ):
        self.count_kernel(
            count_rx,
            count_signal_rx,
            count_signal_tx,
            count_tx,
            own_rank,
            epoch,
        ).launch(
            grid=[self.world_size, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )

    @cute.jit
    def route_launcher(
        self,
        count_tx: cute.Tensor,
        route_rx: cute.Tensor,
        route_signal_rx: cute.Tensor,
        route_signal_tx: cute.Tensor,
        route_tx: cute.Tensor,
        route_bases: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream,
    ):
        self.route_kernel(
            count_tx,
            route_rx,
            route_signal_rx,
            route_signal_tx,
            route_tx,
            route_bases,
            own_rank,
            epoch,
        ).launch(
            grid=[self.world_size, self.num_experts_per_rank, 1],
            block=[128, 1, 1],
            stream=stream,
        )
