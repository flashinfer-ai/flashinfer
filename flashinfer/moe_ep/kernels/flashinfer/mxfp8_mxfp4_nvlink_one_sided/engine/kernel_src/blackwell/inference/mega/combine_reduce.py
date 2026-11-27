# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""End-of-round kernel: Sentinel top-k reduce, landing reset and round advance.

Peers push each route's BF16 FC2 partial into this rank's combine slot
``(token, topk, hidden)`` of the round's set. One thread owns 16 bytes of one
token's row and sums every valid route's slot in FP32 in ascending top-k order
with one BF16 rounding. Under the Sentinel combine the slots are armed with
0xFFFFFFFF words, which no pushed word takes: the thread issues all of its loads,
reloads only the vectors still holding the sentinel, and resets the slots it
consumed. Under the counter combine the main kernel's tail barrier has already
seen every peer finish, so the thread waits for the main grid and loads once.
Peers rewrite a valid slot before it is next read, and an invalid one is never
read, so with ``discard_slots`` a warp drops each 128-byte slot line it has
consumed from L2: once the slots outgrow L2, the pushed partials' write-back to
memory would otherwise take bandwidth from the reduce's reads, which bound it.

The kernel is a programmatic dependent of the main kernel, which triggers at
entry, so its CTAs start as soon as SMs free up. They take one of two roles:

  poller CTAs        the reduce above; under the Sentinel combine they never wait
                     for the main grid, so a poller finishes as soon as its
                     slots have landed
  housekeeping CTAs  the highest block indices; they wait for the main grid,
                     which has itself drained the communication grid, so every
                     local FC1 read of this round's dispatch landing rows and
                     every planner read of its top-k inbox is done. Under the
                     Sentinel dispatch each of their warps resets one
                     ``(source rank, source token)`` landing row if that token
                     routed here, and that token's top-k inbox words in any
                     case; every thread clears its share of the local and
                     set-shared counters the round used

A housekeeping warp reads its first row's inbox ids before it waits: an id word
holds either the sentinel or this round's final value, so only a sentinel needs
reading again afterwards, and the wait is followed by stores alone.

Under the in-kernel top-k reduce (``reduce_slots`` off) the main kernel's FC2
pushes have already summed every route into the output and no combine slots
exist, so the grid is housekeeping CTAs alone.

With a shared expert (``shared_slot``) every token has one more combine slot,
``topk``, which this rank's own shared FC2 tiles fill for every token of the
capacity; the sum adds it after the routes, in the same FP32 order, and treats
it as a valid route in every other respect (polled, reset and discarded
alike). With ``shared_bf16_rounding`` the routes' FP32 sum is rounded to BF16
before the shared slot is added.

With ``weight_routes`` (routing weights applied after FC2) the slots hold the
routes' unweighted BF16 partials, and the sum multiplies each route's partial by
its routing weight from this rank's own ``topk_scores`` in FP32 before adding it;
the shared slot adds with weight one.

Every CTA reads the round epoch at entry and then adds one to an entry ticket
with release semantics; the ticket is spread over several lines so that the
adds, which arrive together, do not queue on one address. The last housekeeping
CTA, which resets no landing row unless the grid is capped, waits until every
CTA of the grid has entered, then for the main grid, then clears the ticket and
advances the epoch, which moves the next round to the other set. Its spin runs
while the main grid is still running, so usually only stores follow the wait,
and it can trap only once the main grid is done. No CTA waits for the reduce
work of another, so the grid ends with whichever of the two roles ends last
rather than with a ticket taken after both.
"""

from typing import ClassVar, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32

from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.dsl_helpers import mark_alignment
from .....helpers.ptx_helpers import red_async_add_release_gpu_s32
from .....helpers.software_sync import NvlinkBarrier
from ......comm import ptx as sentinel_ptx


class SetNvlinkBarrier(NvlinkBarrier):
    """All-rank barrier whose signal words live in the round's landing set.

    The shared workspace alternates between two sets by round, so each set keeps
    its own phase counter; every rank is on the same set in the same round, so
    each set's signals and phase stay a self-consistent sense-reversing barrier.
    """

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        super(NvlinkBarrier, self).register_device_workspace(workspace)
        workspace.register(
            self.phase_counter_region,
            cutlass.Int32,
            (2,),
            buffer_space="local",
            byte_alignment=16,
            reset="zero_on_first_allocate",
        )
        workspace.register(
            self.signal_region,
            cutlass.Int32,
            (2,),
            buffer_space="shared",
            byte_alignment=16,
            reset="zero_on_first_allocate",
        )

    @cute.jit
    def assign_device_members(
        self, workspace: DeviceWorkspace, peer_rank_ptr_mapper, round_set: Int32
    ) -> None:
        # Spelled out rather than delegated: the DSL rewrites jit methods, which
        # breaks zero-argument super() inside the base class's own jit methods.
        self._grid_counter = workspace.ptr(self.grid_counter_region)
        self._phase_counter = workspace.ptr(self.phase_counter_region) + round_set
        self._signal = workspace.ptr(self.signal_region)
        self._peer_rank_ptr_mapper = peer_rank_ptr_mapper


class CombineTopkReduce:
    """BF16 Sentinel combine reduce for a fixed ``(hidden, topk, experts)``."""

    threads: ClassVar[int] = 128
    hidden_per_thread: ClassVar[int] = 8
    # The grid is persistent: at most this many CTAs per SM, each striding over
    # its role's work. The launch bounds keep this many resident per SM, so the
    # whole grid can run at once when the main kernel has left the SMs; progress
    # does not depend on it, since no CTA waits for another to enter.
    ctas_per_sm: ClassVar[int] = 8

    def __init__(
        self,
        token_comm,
        hidden: int,
        num_topk: int,
        num_experts: int,
        world_size: int,
        capacity: int,
        sm_count: int,
        tail_reset_bytes: int,
        poll_slots: bool = True,
        reset_landing: bool = True,
        discard_slots: bool = False,
        reduce_slots: bool = True,
        shared_slot: bool = False,
        shared_bf16_rounding: bool = False,
        weight_routes: bool = False,
    ):
        self.token_comm = token_comm
        self.shared_slot = bool(shared_slot)
        self.shared_bf16_rounding = bool(shared_bf16_rounding)
        if self.shared_bf16_rounding and not self.shared_slot:
            raise ValueError(
                "Rounding the routes' sum before the shared slot needs the shared slot."
            )
        self.poll_slots = bool(poll_slots)
        self.reset_landing = bool(reset_landing)
        self.discard_slots = bool(discard_slots)
        self.reduce_slots = bool(reduce_slots)
        self.weight_routes = bool(weight_routes)
        if self.weight_routes and not self.reduce_slots:
            raise ValueError(
                "Weighting the routes in the reduce needs its combine slots."
            )
        if self.discard_slots and self.poll_slots:
            raise ValueError(
                "Discarding consumed combine slots requires the counter combine."
            )
        if not self.reduce_slots and (self.poll_slots or self.discard_slots):
            raise ValueError(
                "Without combine slots there is nothing to poll or discard."
            )
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        # Combine slots per token: the routes', then the shared expert's.
        self.slots_per_token = self.num_topk + (1 if self.shared_slot else 0)
        self.num_experts = int(num_experts)
        self.world_size = int(world_size)
        self.capacity = int(capacity)
        self.experts_per_rank = self.num_experts // self.world_size
        # Workers index (token, 16-byte tile) flatly, so a row needs whole tiles.
        if self.hidden % self.hidden_per_thread != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a multiple of {self.hidden_per_thread}."
            )
        if self.discard_slots and self.hidden % (32 * self.hidden_per_thread) != 0:
            raise ValueError(
                f"Slot discards need a warp to hold 32 tiles of one token, so hidden ({self.hidden}) "
                f"must be a multiple of {32 * self.hidden_per_thread}."
            )
        if self.reset_landing:
            data_bytes = self.hidden * int(token_comm.activation_dtype.width) // 8
            sf_bytes = (
                token_comm.hidden_sf * int(token_comm.activation_sf_dtype.width) // 8
            )
            if data_bytes % (32 * 16) != 0 or sf_bytes % 16 != 0 or sf_bytes > 32 * 16:
                raise ValueError(
                    "Landing reset stores a row's data in runs of 32 lanes x 16 bytes and its scale factors "
                    f"one 16-byte vector per lane; hidden ({self.hidden}) gives {data_bytes} and {sf_bytes} bytes."
                )
        if self.num_topk > 32:
            raise ValueError(
                "Landing reset checks one route per lane, so topk must be at most 32."
            )
        self.hidden_tiles = self.hidden // self.hidden_per_thread
        self.workers = self.capacity * self.hidden_tiles
        self.landing_rows = self.world_size * self.capacity
        poller_ctas = (
            (self.workers + self.threads - 1) // self.threads
            if self.reduce_slots
            else 0
        )
        # One warp per landing row under the Sentinel dispatch, plus the CTA that
        # advances the round; otherwise one pass over the larger of the two tail
        # spans, whose last CTA also advances the round.
        if self.reset_landing:
            housekeeping_ctas = (
                self.landing_rows * 32 + self.threads - 1
            ) // self.threads + 1
        else:
            housekeeping_ctas = max(
                1, (int(tail_reset_bytes) // 16 + self.threads - 1) // self.threads
            )
        grid_limit = int(sm_count) * self.ctas_per_sm
        if poller_ctas + housekeeping_ctas > grid_limit:
            # Both roles stride over their work; split the persistent grid in
            # proportion to the CTAs each would take on its own.
            housekeeping_ctas = max(
                1, grid_limit * housekeeping_ctas // (poller_ctas + housekeeping_ctas)
            )
            poller_ctas = grid_limit - housekeeping_ctas
        self.poller_ctas = poller_ctas
        self.housekeeping_ctas = housekeeping_ctas
        self.grid_ctas = poller_ctas + housekeeping_ctas

    @cute.jit
    def __call__(
        self,
        combine_output: cute.Tensor,  # (capacity, slots per token, hidden) BF16, set-0 address
        reduced_output: cute.Tensor,  # (capacity, hidden) BF16
        topk_indices: cute.Tensor,  # (capacity, topk) Int32
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
        topk_scores: Optional[
            cute.Tensor
        ] = None,  # (capacity, topk) FP32, with weight_routes
    ) -> None:
        if cutlass.const_expr(self.weight_routes != (topk_scores is not None)):
            raise ValueError(
                "The routing weights are passed exactly when the reduce weights the routes."
            )
        args = (
            combine_output,
            reduced_output,
            topk_indices,
            local_rank,
            local_workspace,
            shared_workspace,
        )
        if cutlass.const_expr(self.weight_routes):
            kernel = self._weighted_kernel(*args, device_workspace, topk_scores)
        else:
            kernel = self._kernel(*args, device_workspace)
        kernel.launch(
            grid=[self.grid_ctas, 1, 1],
            block=[self.threads, 1, 1],
            min_blocks_per_mp=self.ctas_per_sm,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def _kernel(
        self,
        combine_output: cute.Tensor,
        reduced_output: cute.Tensor,
        topk_indices: cute.Tensor,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        device_workspace: DeviceWorkspace,
    ) -> None:
        self._run(
            combine_output,
            reduced_output,
            topk_indices,
            local_rank,
            local_workspace,
            shared_workspace,
            device_workspace,
        )

    @cute.kernel
    def _weighted_kernel(
        self,
        combine_output: cute.Tensor,
        reduced_output: cute.Tensor,
        topk_indices: cute.Tensor,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        device_workspace: DeviceWorkspace,
        topk_scores: cute.Tensor,
    ) -> None:
        self._run(
            combine_output,
            reduced_output,
            topk_indices,
            local_rank,
            local_workspace,
            shared_workspace,
            device_workspace,
            topk_scores,
        )

    @cute.jit
    def _run(
        self,
        combine_output: cute.Tensor,
        reduced_output: cute.Tensor,
        topk_indices: cute.Tensor,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        device_workspace: DeviceWorkspace,
        topk_scores: Optional[cute.Tensor] = None,
    ) -> None:
        comm = self.token_comm
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        round_set = comm.select_round_set(
            device_workspace, local_workspace, shared_workspace
        )
        epoch_word = device_workspace.ptr(comm.sentinel_control_region).toint() + Int64(
            comm.sentinel_epoch_word * 4
        )
        tickets = device_workspace.ptr(comm.sentinel_ticket_region)
        advances = block_idx == Int32(self.grid_ctas - 1)
        epoch = Uint32(0)
        if advances:
            if thread_idx == Int32(0):
                epoch = sentinel_ptx.ld_volatile_u32(epoch_word)
        # Every thread has read the epoch; the release orders those reads before
        # the round advance, which waits for the whole grid's tickets. The memory
        # system enforces it, so the issuing warp does not wait.
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                tickets + (block_idx % Int32(comm.sentinel_ticket_lines)) * Int32(32),
                Int32(1),
            )

        if block_idx < Int32(self.poller_ctas):
            combine_base = combine_output.iterator.toint() + Int64(round_set) * Int64(
                comm.shared_set_bytes
            )
            if cutlass.const_expr(not self.poll_slots):
                cute.arch.griddepcontrol_wait()
            item = block_idx * Int32(self.threads) + thread_idx
            while item < Int32(self.workers):
                self._reduce_worker(
                    item, combine_base, reduced_output, topk_indices, topk_scores
                )
                item += Int32(self.poller_ctas * self.threads)
        else:
            worker = (block_idx - Int32(self.poller_ctas)) * Int32(
                self.threads
            ) + thread_idx
            housekeeping_threads = self.housekeeping_ctas * self.threads
            lane = thread_idx % Int32(32)
            landing_row = worker // Int32(32)
            inbox_id = Uint32(sentinel_ptx.WORD_SENTINEL)
            if cutlass.const_expr(self.reset_landing):
                inbox_id = self._load_inbox_id(landing_row, lane, device_workspace)
            entered = Uint32(0)
            if advances:
                if thread_idx == Int32(0):
                    entered = self._wait_entered_before_main(tickets)
            cute.arch.griddepcontrol_wait()
            if advances:
                if thread_idx == Int32(0):
                    self._advance_round(tickets, epoch_word, epoch, entered)
            if cutlass.const_expr(self.reset_landing):
                while landing_row < Int32(self.landing_rows):
                    self._reset_landing_row(
                        landing_row, lane, local_rank, device_workspace, inbox_id
                    )
                    landing_row += Int32(housekeeping_threads // 32)
                    inbox_id = Uint32(sentinel_ptx.WORD_SENTINEL)
            device_workspace.reset_tail_space("local", worker, housekeeping_threads)
            # Peer-incremented counters of this round's set; peers next write this set
            # two rounds later, after this rank's next top-k publication.
            device_workspace.reset_tail_space("shared", worker, housekeeping_threads)

    @cute.jit
    def _wait_entered_before_main(self, tickets: cute.Pointer) -> Uint32:
        """Spin on the entry ticket while the main grid runs; return the last count.

        A CTA adds its ticket once all of its threads have read the epoch. The
        lines are read relaxed, all in flight at once, and the fence after the
        read that sums to the grid makes it an acquire. The main grid may still be
        waiting on a late peer while CTAs of this grid wait for an SM, so the spin
        gives up without a trap after ``POLL_SWEEP_LIMIT`` polls and resumes after
        ``griddepcontrol.wait`` (``_advance_round``).
        """
        entered = self._count_entered(tickets)
        polls = Int32(0)
        while (entered != Uint32(self.grid_ctas)) & (
            polls < Int32(sentinel_ptx.POLL_SWEEP_LIMIT)
        ):
            entered = self._count_entered(tickets)
            polls += Int32(1)
        if entered == Uint32(self.grid_ctas):
            sentinel_ptx.fence_acq_rel_gpu()
        return entered

    @cute.jit
    def _advance_round(
        self, tickets: cute.Pointer, epoch_word: Int64, epoch: Uint32, entered: Uint32
    ) -> None:
        """Clear the entry ticket and store ``epoch + 1`` once every CTA has entered.

        Runs after ``griddepcontrol.wait``: the main and communication grids,
        which read the epoch at entry, are done. ``entered`` is the count the spin
        before the wait ended on; if that spin gave up, it resumes here, where a
        CTA that has not entered yet waits only for an SM, so a ticket that never
        arrives traps once the poll budget is spent.
        """
        if entered != Uint32(self.grid_ctas):
            count = entered
            polls = Int32(0)
            while count != Uint32(self.grid_ctas):
                count = self._count_entered(tickets)
                polls += Int32(1)
                if polls == Int32(sentinel_ptx.POLL_SWEEP_LIMIT):
                    cute.printf(
                        "combine reduce: %d of %d CTAs entered\n",
                        Int32(count),
                        Int32(self.grid_ctas),
                    )
                    sentinel_ptx.trap()
            sentinel_ptx.fence_acq_rel_gpu()
        for line in cutlass.range_constexpr(self.token_comm.sentinel_ticket_lines):
            sentinel_ptx.st_volatile_u32(
                (tickets + Int32(line * 32)).toint(), Uint32(0)
            )
        sentinel_ptx.st_volatile_u32(epoch_word, epoch + Uint32(1))

    @cute.jit
    def _count_entered(self, tickets: cute.Pointer) -> Uint32:
        """Sum of the entry ticket's lines, each read relaxed."""
        entered = Uint32(0)
        for line in cutlass.range_constexpr(self.token_comm.sentinel_ticket_lines):
            entered = entered + sentinel_ptx.ld_volatile_u32(
                (tickets + Int32(line * 32)).toint()
            )
        return entered

    @cute.jit
    def _reduce_worker(
        self,
        worker: Int32,
        combine_base: Int64,
        reduced_output: cute.Tensor,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor] = None,
    ) -> None:
        token = worker // Int32(self.hidden_tiles)
        tile = worker % Int32(self.hidden_tiles)
        if token < Int32(self.capacity):
            topk = self.slots_per_token
            words = cute.make_rmem_tensor((topk * 4,), Uint32)
            addresses = cute.make_rmem_tensor((topk,), Int64)
            for k in cutlass.range_constexpr(topk):
                if cutlass.const_expr(k < self.num_topk):
                    expert = topk_indices[token, k]
                    addresses[k] = Int64(0)
                    if (expert >= Int32(0)) & (expert < Int32(self.num_experts)):
                        element = (Int64(token) * Int64(topk) + Int64(k)) * Int64(
                            self.hidden
                        ) + Int64(tile) * Int64(self.hidden_per_thread)
                        addresses[k] = combine_base + element * Int64(2)
                else:
                    # The shared expert's slot: every token of the capacity has one.
                    element = (Int64(token) * Int64(topk) + Int64(k)) * Int64(
                        self.hidden
                    ) + Int64(tile) * Int64(self.hidden_per_thread)
                    addresses[k] = combine_base + element * Int64(2)
            if cutlass.const_expr(topk_scores is not None):
                # Each route's routing weight, loaded with the slots it scales.
                weights = cute.make_rmem_tensor((self.num_topk,), Float32)
                for k in cutlass.range_constexpr(self.num_topk):
                    weights[k] = Float32(0.0)
                    if addresses[k] != Int64(0):
                        weights[k] = topk_scores[token, k]
            if cutlass.const_expr(self.discard_slots):
                # Bit k: route k is valid. It outlives the addresses, which the sum is the
                # last to need.
                valid_routes = Int32(0)
                for k in cutlass.range_constexpr(topk):
                    if addresses[k] != Int64(0):
                        valid_routes = valid_routes | Int32(1 << k)
            for k in cutlass.range_constexpr(topk):
                for word in cutlass.range_constexpr(4):
                    words[k * 4 + word] = Uint32(0)
                if addresses[k] != Int64(0):
                    if cutlass.const_expr(self.poll_slots):
                        loaded = sentinel_ptx.ld_volatile_v4(addresses[k])
                    else:
                        # The main grid has completed, and its tail barrier saw every
                        # peer's pushes land, so a plain load observes them.
                        loaded = sentinel_ptx.ld_global_v4(addresses[k])
                    for word in cutlass.range_constexpr(4):
                        words[k * 4 + word] = loaded[word]
            pending = Int32(1 if self.poll_slots else 0)
            sweeps = Int32(0)
            while pending != Int32(0):
                pending = Int32(0)
                for k in cutlass.range_constexpr(topk):
                    missing = (
                        (words[k * 4] == Uint32(sentinel_ptx.WORD_SENTINEL))
                        | (words[k * 4 + 1] == Uint32(sentinel_ptx.WORD_SENTINEL))
                        | (words[k * 4 + 2] == Uint32(sentinel_ptx.WORD_SENTINEL))
                        | (words[k * 4 + 3] == Uint32(sentinel_ptx.WORD_SENTINEL))
                    )
                    if missing:
                        loaded = sentinel_ptx.ld_volatile_v4(addresses[k])
                        for word in cutlass.range_constexpr(4):
                            words[k * 4 + word] = loaded[word]
                        pending = Int32(1)
                sweeps += Int32(1)
                if sweeps == Int32(sentinel_ptx.POLL_SWEEP_LIMIT):
                    cute.printf(
                        "sentinel combine: token %d hidden tile %d never received all routes\n",
                        token,
                        tile,
                    )
                    sentinel_ptx.trap()

            # BF16 -> FP32 is a 16-bit shift of each lane into the high half.
            accumulator = cute.make_rmem_tensor((self.hidden_per_thread,), Float32)
            for element in cutlass.range_constexpr(self.hidden_per_thread):
                accumulator[element] = Float32(0.0)
            lanes = cute.make_rmem_tensor((self.hidden_per_thread,), Uint32)
            values = cute.recast_tensor(lanes, Float32)
            for k in cutlass.range_constexpr(topk):
                if cutlass.const_expr(self.shared_bf16_rounding and k == self.num_topk):
                    # The routes' sum, rounded to BF16 before the shared slot joins it.
                    accumulator.store(
                        accumulator.load().to(cutlass.BFloat16).to(Float32)
                    )
                if addresses[k] != Int64(0):
                    for word in cutlass.range_constexpr(4):
                        lanes[word * 2] = words[k * 4 + word] << Uint32(16)
                        lanes[word * 2 + 1] = words[k * 4 + word] & Uint32(0xFFFF0000)
                    if cutlass.const_expr(
                        topk_scores is not None and k < self.num_topk
                    ):
                        for element in cutlass.range_constexpr(self.hidden_per_thread):
                            accumulator[element] = (
                                accumulator[element] + values[element] * weights[k]
                            )
                    else:
                        for element in cutlass.range_constexpr(self.hidden_per_thread):
                            accumulator[element] = (
                                accumulator[element] + values[element]
                            )

            out = cute.make_rmem_tensor(
                (self.hidden_per_thread,), reduced_output.element_type
            )
            out.store(accumulator.load().to(reduced_output.element_type))
            destination = cute.zipped_divide(
                reduced_output[token, None], (self.hidden_per_thread,)
            )[(None,), (tile,)]
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    reduced_output.element_type,
                    num_bits_per_copy=128,
                ),
                out,
                mark_alignment(
                    destination,
                    self.hidden_per_thread * reduced_output.element_type.width // 8,
                ),
            )
            if cutlass.const_expr(self.discard_slots):
                # A warp holds 32 consecutive 16-byte tiles of one token, since a row is
                # a multiple of 32 tiles and the stride a multiple of the warp: once every
                # lane has consumed its tiles, the lane at the start of each 128-byte line
                # drops that line of every valid slot.
                cute.arch.sync_warp()
                if tile % Int32(128 // 16) == Int32(0):
                    line = combine_base + (
                        Int64(token) * Int64(topk * self.hidden)
                        + Int64(tile) * Int64(self.hidden_per_thread)
                    ) * Int64(2)
                    for k in cutlass.range_constexpr(topk):
                        if ((valid_routes >> Int32(k)) & Int32(1)) != Int32(0):
                            sentinel_ptx.discard_global_l2(
                                line + Int64(k * self.hidden * 2)
                            )
            for k in cutlass.range_constexpr(topk if self.poll_slots else 0):
                if addresses[k] != Int64(0):
                    sentinel_ptx.st_volatile_v4_words(
                        addresses[k],
                        Uint32(sentinel_ptx.WORD_SENTINEL),
                        Uint32(sentinel_ptx.WORD_SENTINEL),
                        Uint32(sentinel_ptx.WORD_SENTINEL),
                        Uint32(sentinel_ptx.WORD_SENTINEL),
                    )

    @cute.jit
    def _inbox_id_address(
        self, global_warp: Int32, lane: Int32, device_workspace: DeviceWorkspace
    ) -> Int64:
        """Address of lane ``lane``'s top-k inbox id of landing row ``global_warp``."""
        comm = self.token_comm
        source = global_warp // Int32(self.capacity)
        token = global_warp % Int32(self.capacity)
        inbox_word = (
            source * Int32(comm.topk_slab_elements)
            + token * Int32(self.num_topk)
            + lane
        ) * Int32(4)
        return device_workspace.ptr(comm.topk_index_inbox_region).toint() + Int64(
            inbox_word
        )

    @cute.jit
    def _load_inbox_id(
        self, global_warp: Int32, lane: Int32, device_workspace: DeviceWorkspace
    ) -> Uint32:
        """Lane ``lane``'s top-k inbox id of landing row ``global_warp``, or the sentinel past the plane."""
        inbox_id = Uint32(sentinel_ptx.WORD_SENTINEL)
        if (global_warp < Int32(self.landing_rows)) & (lane < Int32(self.num_topk)):
            inbox_id = sentinel_ptx.ld_volatile_u32(
                self._inbox_id_address(global_warp, lane, device_workspace)
            )
        return inbox_id

    @cute.jit
    def _reset_landing_row(
        self,
        global_warp: Int32,
        lane: Int32,
        local_rank: Int32,
        device_workspace: DeviceWorkspace,
        inbox_id: Uint32,
    ) -> None:
        """Reset one ``(source rank, source token)`` landing row and its top-k inbox words.

        Lane ``k`` passes the token's ``k``-th inbox id as it last read it, possibly
        before the communication grid completed. An id word holds the sentinel
        until this round's value lands and keeps that value until this reset, so
        only a sentinel is read again here. The whole plane carries the id
        (``expert_count`` for a route no rank owns), and the row is reset when
        any of them is local. Every inbox id and score word of the token is reset
        afterwards, whatever it held, by the lane that read it.
        """
        comm = self.token_comm
        if global_warp < Int32(self.world_size * self.capacity):
            source = global_warp // Int32(self.capacity)
            token = global_warp % Int32(self.capacity)
            inbox_word = (
                source * Int32(comm.topk_slab_elements)
                + token * Int32(self.num_topk)
                + lane
            ) * Int32(4)
            routes_here = cutlass.Boolean(False)
            if lane < Int32(self.num_topk):
                if inbox_id == Uint32(sentinel_ptx.WORD_SENTINEL):
                    inbox_id = sentinel_ptx.ld_volatile_u32(
                        self._inbox_id_address(global_warp, lane, device_workspace)
                    )
                local_expert = Int32(inbox_id) - local_rank * Int32(
                    self.experts_per_rank
                )
                routes_here = cutlass.Uint32(local_expert) < cutlass.Uint32(
                    self.experts_per_rank
                )
            if Int32(cute.arch.vote_ballot_sync(routes_here)) != Int32(0):
                row = Int64(source) * Int64(self.capacity) + Int64(token)
                data_row_bytes = (
                    comm.plain_activation_row_elements
                    * int(comm.activation_dtype.width)
                    // 8
                )
                sf_row_bytes = (
                    comm.plain_activation_sf_row_elements
                    * int(comm.activation_sf_dtype.width)
                    // 8
                )
                data_vectors = (
                    comm.hidden_size * int(comm.activation_dtype.width) // 8 // 16
                )
                sf_vectors = (
                    comm.hidden_sf * int(comm.activation_sf_dtype.width) // 8 // 16
                )
                data_row = device_workspace.ptr(
                    comm.plain_activation_region
                ).toint() + row * Int64(data_row_bytes)
                sf_row = device_workspace.ptr(
                    comm.plain_activation_sf_region
                ).toint() + row * Int64(sf_row_bytes)
                sentinel = Uint32(sentinel_ptx.BYTE_SENTINEL_WORD)
                for vector in cutlass.range_constexpr(data_vectors // 32):
                    sentinel_ptx.st_volatile_v4_words(
                        data_row + Int64(lane + Int32(vector * 32)) * Int64(16),
                        sentinel,
                        sentinel,
                        sentinel,
                        sentinel,
                    )
                if lane < Int32(sf_vectors):
                    sentinel_ptx.st_volatile_v4_words(
                        sf_row + Int64(lane) * Int64(16),
                        sentinel,
                        sentinel,
                        sentinel,
                        sentinel,
                    )
            if lane < Int32(self.num_topk):
                inbox_sentinel = Uint32(sentinel_ptx.WORD_SENTINEL)
                sentinel_ptx.st_volatile_u32(
                    device_workspace.ptr(comm.topk_index_inbox_region).toint()
                    + Int64(inbox_word),
                    inbox_sentinel,
                )
                if cutlass.const_expr(comm.apply_topk_at_fc1):
                    sentinel_ptx.st_volatile_u32(
                        device_workspace.ptr(comm.topk_score_inbox_region).toint()
                        + Int64(inbox_word),
                        inbox_sentinel,
                    )


__all__ = ["CombineTopkReduce"]
