"""Shared GIN transport allocation, compilation and launch operations.

Both the standalone validation driver and the serving adapter use this protocol.
Memory ownership is supplied by ``bridge``; computation/input generation and
numerical reference validation remain outside this module.
"""
from __future__ import annotations
import math
import torch
from cuda.bindings import driver as cuda
from ..runtime.low_precision_dlpack import wrap as wrap_dlpack
from . import gin_bridge

_IBGDA_OWNER_PLAN_BLOCK_ELEMENTS = 256
_IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS = 255
_IBGDA_OWNER_PLAN_SLOTS = 2

class NetworkTransportRuntime:
    bridge = gin_bridge
    device_epoch = None

    def _epoch_kwargs(self):
        # The legacy standalone runner continues to pass scalar epochs. Serving
        # graphs read a workspace-owned device counter on every replay.
        return {} if self.device_epoch is None else {"epoch_ptr": self.device_epoch_cute}

    def _publish_dispatch_epoch(self, destination):
        if self.device_epoch is None:
            destination.fill_(self._transport.dispatch_epoch)
        else:
            destination.copy_(self.device_epoch.expand_as(destination))

    @staticmethod
    def _byte_workspace_view(
        workspace: torch.Tensor,
        byte_offset: int,
        dtype: torch.dtype,
        shape,
    ) -> torch.Tensor:
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        return workspace.narrow(0, byte_offset, nbytes).view(dtype).view(*shape)


    def allocate_workspaces(self) -> None:
        """Allocate the two opaque byte buffers the kernel partitions internally.

        ``local_workspace`` normally lives in plain CUDA device memory.  The
        GIN path places it in an NCCL-registered GPU allocation so the
        peer-major K2 output staging region has a registered lkey and can be
        used directly as an RMA source; counters / FC1 output remain local
        accesses even though their backing allocation is registered.
        ``shared_workspace`` lives in the NCCL LSA peer-mapped arena so the
        dispatch warps' ``peer_rank_ptr_mapper.map`` writes can reach peer ranks.
        ``(symmetric_base, peer_offsets_list)`` is computed once from the
        P2P arena base address -- the allocator preserves the same
        delta works for every sub-allocation in the heap, so the same
        table also covers ``my_activation`` / ``my_activation_sf`` /
        ``my_topk_weights`` / ``combine_output``.  ``run_kernel`` passes
        these ints through ``SymBufferHost`` so kernel.__call__ can pack
        them into SymBuffer{world_size}.

        Caller invariant: ``self._kernel`` must already be instantiated
        (``run_kernel`` is responsible for that step before invoking us).
        """
        if self._kernel is None:
            raise RuntimeError(
                "allocate_workspaces called before self._kernel was "
                "instantiated; run_kernel must create the kernel first."
            )

        local_ws_bytes, shared_ws_bytes = self._kernel.get_workspace_sizes()

        if self.comm_backend == "gin":
            self.bridge.prepare_network_arena(self, local_ws_bytes, _IBGDA_OWNER_PLAN_SLOTS, _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS, _IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS)
            self.local_workspace = self.bridge.sym_zeros(
                (local_ws_bytes,), torch.uint8
            )
        else:
            self.local_workspace = torch.zeros(
                (local_ws_bytes,), dtype=torch.uint8, device="cuda",
            )
        self.shared_workspace = self.bridge.arena_zeros((shared_ws_bytes,), torch.uint8)
        if self.rank_local_combine:
            self.rank_combine_ready = self.bridge.arena_zeros(
                (self.problem.num_tokens_per_rank + 1, self.world_size),
                torch.int32,
            )
        if self.comm_backend == "gin":
            tokens = self.problem.num_tokens_per_rank
            max_routes_per_source = tokens * self.problem.num_topk
            # Fixed-capacity GPU control buffers; their logical lengths
            # are rebuilt from the current topk_idx every replay.
            self._transport.max_pair_routes = max_routes_per_source
            self._transport.dispatch_send_token_indices = torch.full(
                (self.world_size, tokens),
                -1,
                dtype=torch.int32,
                device="cuda",
            )
            self._transport.dispatch_send_counts = torch.zeros(
                (self.world_size,), dtype=torch.int32, device="cuda"
            )
            owner_token_shape = (
                self.world_size,
                self.problem.num_experts_per_rank,
            )
            self._transport.dispatch_owner_token_counts = torch.zeros(
                owner_token_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.dispatch_owner_token_bases = torch.zeros(
                owner_token_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.dispatch_owner_token_cursors = torch.zeros(
                owner_token_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.expert_pool_bases = torch.zeros(
                (self.problem.num_experts_per_rank,),
                dtype=torch.int32,
                device="cuda",
            )
            self._transport.combine_source_expert_bases = torch.zeros(
                (self.world_size, self.problem.num_experts_per_rank),
                dtype=torch.int32,
                device="cuda",
            )
            activation_rx_shape = (
                self.world_size * tokens,
                *self.my_activation.shape[1:],
            )
            self._transport.activation_rx = torch.zeros(
                activation_rx_shape, dtype=torch.uint8, device="cuda"
            ).view(self.my_activation.dtype)
            sf_rx_shape = (
                self.world_size * tokens,
                *self.my_activation_sf.shape[1:],
            )
            self._transport.sf_rx = torch.zeros(
                sf_rx_shape, dtype=torch.uint8, device="cuda"
            ).view(self.my_activation_sf.dtype)
            self._transport.topk_weight_rx = torch.zeros(
                (self.world_size * tokens, self.problem.num_topk),
                dtype=torch.float32,
                device="cuda",
            )
            # Each source builds one compact expert-grouped record per owner.
            # Aligned 256-int64 payload blocks use the same nonblocking
            # payload/fence/epoch protocol as the data ring. The count record
            # has an independent commit so stable extents install first.
            max_route_blocks = (
                max_routes_per_source
                + _IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS
                - 1
            ) // _IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS
            self._transport.owner_plan_route_capacity = (
                max_route_blocks
                + self.problem.num_experts_per_rank
            ) * _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS
            count_shape = (
                _IBGDA_OWNER_PLAN_SLOTS,
                self.world_size,
                _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS,
            )
            route_shape = (
                _IBGDA_OWNER_PLAN_SLOTS,
                self.world_size,
                self._transport.owner_plan_route_capacity,
            )
            self._transport.owner_plan_count_rx = self.bridge.sym_zeros(
                count_shape, torch.int64
            )
            self._transport.owner_plan_count_signal_rx = self.bridge.sym_zeros(
                (_IBGDA_OWNER_PLAN_SLOTS, self.world_size),
                torch.int64,
            )
            self._transport.owner_plan_count_signal_tx = self.bridge.sym_zeros(
                (_IBGDA_OWNER_PLAN_SLOTS, self.world_size),
                torch.int64,
            )
            self._transport.owner_plan_count_tx = self.bridge.sym_zeros(
                count_shape, torch.int64
            )
            self._transport.owner_plan_route_rx = self.bridge.sym_zeros(
                route_shape, torch.int64
            )
            self._transport.owner_plan_route_signal_rx = self.bridge.sym_zeros(
                (
                    _IBGDA_OWNER_PLAN_SLOTS,
                    self.world_size,
                    self.problem.num_experts_per_rank,
                ),
                torch.int64,
            )
            self._transport.owner_plan_route_signal_tx = self.bridge.sym_zeros(
                (
                    _IBGDA_OWNER_PLAN_SLOTS,
                    self.world_size,
                    self.problem.num_experts_per_rank,
                ),
                torch.int64,
            )
            self._transport.owner_plan_route_tx = self.bridge.sym_zeros(
                route_shape, torch.int64
            )
            plan_shape = (
                self.world_size,
                self.problem.num_experts_per_rank,
            )
            self._transport.owner_plan_route_bases_tx = torch.zeros(
                plan_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.dispatch_owner_route_output_bases = torch.zeros(
                plan_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.owner_plan_route_cursors_tx = torch.zeros(
                plan_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.owner_plan_route_bases_rx = torch.zeros(
                plan_shape, dtype=torch.int32, device="cuda"
            )
            channels = self._transport.dispatch_ring_channels
            slots = self._transport.dispatch_ring_slots
            chunk_tokens = self._kernel.token_comm.ibgda_dispatch_chunk_tokens
            act_bpt = self.my_activation[0].numel() * self.my_activation.element_size()
            sf_bpt = self.my_activation_sf[0].numel() * self.my_activation_sf.element_size()
            weight_bpt = self.my_topk_weights[0].numel() * self.my_topk_weights.element_size()
            dispatch_payload_bytes = chunk_tokens * (
                4 + act_bpt + sf_bpt + weight_bpt
            )
            dispatch_slot_bytes = (
                (dispatch_payload_bytes + 8 + 15) // 16
            ) * 16
            self._transport.dispatch_ring_rx = self.bridge.sym_zeros(
                (
                    self.world_size,
                    channels,
                    slots,
                    dispatch_slot_bytes,
                ),
                torch.uint8,
            )
            self._transport.dispatch_ring_tx = self.bridge.sym_zeros(
                (
                    self.world_size,
                    channels,
                    slots,
                    dispatch_slot_bytes,
                ),
                torch.uint8,
            )
            self._transport.dispatch_ring_ack = self.bridge.sym_zeros(
                (self.world_size, channels, slots, 128), torch.uint8
            )
            self._transport.dispatch_ring_ack_tx = self.bridge.sym_zeros(
                (self.world_size, channels, slots, 128), torch.uint8
            )
            self._transport.dispatch_control_rx = self.bridge.sym_zeros(
                (self.world_size, 128), torch.uint8
            )
            self._transport.dispatch_control_tx = self.bridge.sym_zeros(
                (self.world_size, 128), torch.uint8
            )
            dispatch_tasks = self.world_size * channels
            self._transport.dispatch_send_work_queue = torch.zeros(
                (2 * dispatch_tasks + self.kernel_config.tx_sms + 1,),
                dtype=torch.int32,
                device="cuda",
            )
            self._transport.dispatch_recv_work_queue = torch.zeros(
                (2 * dispatch_tasks + self.kernel_config.rx_sms + 1,),
                dtype=torch.int32,
                device="cuda",
            )
            # Transport epochs start at one and are incremented before launch.
            # Initialize the publication matrix to the same predecessor epoch
            # so each channel can CAS epoch-1 -> epoch exactly once.
            self._transport.dispatch_tx_handoff = torch.full(
                (self.world_size, channels),
                1,
                dtype=torch.int32,
                device="cuda",
            )
            ready_words = self._kernel.token_comm.ibgda_dispatch_ready_words
            ready_off = self._kernel._local_offsets["ibgda_sf_staging"]
            self._transport.dispatch_ready = self.local_workspace[
                ready_off:ready_off + ready_words * 4
            ].view(torch.int32)
            plan_ready_words = (
                self.world_size * self.problem.num_experts_per_rank
            )
            plan_ready_off = ready_off + ready_words * 4
            self._transport.owner_plan_ready = self.local_workspace[
                plan_ready_off:plan_ready_off + plan_ready_words * 4
            ].view(torch.int32)
            # The bounded ring slots below are the only combine payload
            # storage; no full routed-output replica is allocated.
            channels = self._transport.combine_ring_channels
            slots = self._transport.combine_ring_slots
            chunk_rows = self.kernel_config.combine_chunk_rows
            if chunk_rows <= 0:
                raise ValueError(
                    "combine_chunk_rows must be positive"
                )
            payload_bytes = (
                self.problem.hidden * self.combine_output.element_size()
            )
            slot_bytes = (
                (chunk_rows * (4 + payload_bytes) + 8 + 15) // 16
            ) * 16
            self._transport.ring_rx = self.bridge.sym_zeros(
                (self.world_size, channels, slots, slot_bytes),
                torch.uint8,
            )
            self._transport.ring_tx = self.bridge.sym_zeros(
                (self.world_size, channels, slots, slot_bytes),
                torch.uint8,
            )
            self._transport.ring_ack = self.bridge.sym_zeros(
                (self.world_size, channels, slots, 128), torch.uint8
            )
            self._transport.ring_ack_tx = self.bridge.sym_zeros(
                (self.world_size, channels, slots, 128), torch.uint8
            )
            shape = (self.world_size, self._transport.max_pair_routes)
            self._transport.ring_send_rows = torch.zeros(
                shape, dtype=torch.int64, device="cuda"
            )
            self._transport.ring_send_ready_slots = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )
            self._transport.ring_send_ready_values = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )
            self._transport.ring_send_output_rows = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )
            self._transport.combine_recv_output_rows = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )
            self._transport.ring_send_counts = torch.zeros(
                (self.world_size,), dtype=torch.int32, device="cuda"
            )
            self.symmetric_base = int(self.shared_workspace.data_ptr())
            self.peer_offsets_list = tuple(0 for _ in range(self.world_size))
        else:
            self.symmetric_base, self.peer_offsets_list = self.bridge.peer_offsets(
                self.shared_workspace, self.world_size,
            )
            self.bridge.configure(self)


    def _compile_ibgda_gpu_push_kernels(self, cute_module) -> None:
        self.bridge.configure(self)
        import cuda.bindings.driver as cuda
        from cutlass.cute.runtime import from_dlpack
        from moe_sm120_nvfp4_split.transport.gin_gpu_comm import (
            GinGpuOwnerPlanPush,
        )
        from moe_sm120_nvfp4_split.transport.gin_gpu_ring import (
            GinGpuRingCombineReceiver,
            GinGpuRingCombineSender,
            GinGpuRingDispatchReceiver,
            GinGpuRingDispatchSender,
        )
        from moe_sm120_nvfp4_split.transport.gpu_runtime_route import (
            GpuChunkedOwnerRoutePlanBuilder,
            GpuDispatchRouteBuilder,
        )

        def comm_to_cute(tensor):
            return from_dlpack(wrap_dlpack(tensor)).mark_layout_dynamic()

        # communication progress must be established before the long-lived
        # receiver polling CTA is admitted.  A high-priority producer
        # stream prevents the scheduler from choosing the waiter first.
        if self._transport.dispatch_send_stream is None:
            self._transport.dispatch_send_stream = torch.cuda.Stream(
                priority=-1
            )
            self._transport.dispatch_recv_stream = torch.cuda.Stream(
                priority=-1
            )
        if self._transport.owner_plan_stream is None:
            self._transport.owner_plan_stream = torch.cuda.Stream(priority=-1)
        if self._transport.owner_metadata_stream is None:
            self._transport.owner_metadata_stream = torch.cuda.Stream(priority=0)
        chunk_tokens = self._kernel.token_comm.ibgda_dispatch_chunk_tokens
        act_bpt = (
            self.my_activation[0].numel()
            * self.my_activation.element_size()
        )
        sf_bpt = (
            self.my_activation_sf[0].numel()
            * self.my_activation_sf.element_size()
        )
        weight_bpt = (
            self.my_topk_weights[0].numel()
            * self.my_topk_weights.element_size()
        )
        # Ownership is a protocol property, not an SM-count heuristic: every
        # (peer, channel) has one lifetime owner CTA.  When an explicit tuning
        # override selects four channels, two lightweight owner CTAs may share
        # each of the 16 direction-specific communication SMs.
        dispatch_tasks = (
            self.world_size * self._transport.dispatch_ring_channels
        )
        dispatch_tx_workers = dispatch_tasks
        dispatch_rx_workers = dispatch_tasks
        self._transport.dispatch_ring_sender_kernel = (
            GinGpuRingDispatchSender(
                self.world_size,
                self.problem.num_tokens_per_rank,
                act_bpt,
                sf_bpt,
                weight_bpt,
                chunk_tokens=chunk_tokens,
                channels=self._transport.dispatch_ring_channels,
                slots=self._transport.dispatch_ring_slots,
                local_handoff_windows=(
                    self.kernel_config.dispatch_local_handoff_windows
                ),
                remote_handoff_windows=(
                    self.kernel_config.dispatch_remote_handoff_windows
                ),
                hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
                worker_ctas=dispatch_tx_workers,
            )
        )
        self._transport.dispatch_ring_receiver_kernel = (
            GinGpuRingDispatchReceiver(
                self._transport.dispatch_ring_sender_kernel,
                worker_ctas=dispatch_rx_workers,
            )
        )
        self._transport.dispatch_epoch = 1
        dispatch_sender_args = (
            comm_to_cute(self._transport.dispatch_ring_rx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_ring_tx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_ring_ack.reshape(-1)),
            comm_to_cute(self._transport.dispatch_control_rx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_control_tx.reshape(-1)),
            comm_to_cute(self.my_activation.view(torch.uint8).reshape(-1)),
            comm_to_cute(
                self.my_activation_sf.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(
                self.my_topk_weights.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_send_token_indices.reshape(-1)
            ),
            comm_to_cute(self._transport.dispatch_send_counts),
            comm_to_cute(self._transport.dispatch_tx_handoff.reshape(-1)),
            comm_to_cute(self._transport.dispatch_send_work_queue),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(self._transport.dispatch_send_stream.cuda_stream),
        )
        dispatch_receiver_args = (
            comm_to_cute(self._transport.dispatch_ring_rx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_ring_ack.reshape(-1)),
            comm_to_cute(
                self._transport.dispatch_ring_ack_tx.reshape(-1)
            ),
            comm_to_cute(self._transport.dispatch_control_rx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_recv_work_queue),
            comm_to_cute(
                self._transport.activation_rx.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(
                self._transport.sf_rx.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(
                self._transport.topk_weight_rx.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(self._transport.dispatch_ready),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(self._transport.dispatch_recv_stream.cuda_stream),
        )
        dispatch_ack_args = (
            comm_to_cute(self._transport.dispatch_ring_ack.reshape(-1)),
            comm_to_cute(self._transport.dispatch_send_counts),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(self._transport.dispatch_send_stream.cuda_stream),
        )
        self._transport.owner_plan_push_kernel = (
            GinGpuOwnerPlanPush(
                self.world_size,
                self.problem.num_experts_per_rank,
                self._transport.owner_plan_route_capacity,
                block_elements=(
                    _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS
                ),
                plan_slots=_IBGDA_OWNER_PLAN_SLOTS,
                hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
                completion_credit=(
                    self.problem.num_tokens_per_rank <= 128
                ),
            )
        )
        plan_stream = self._transport.owner_plan_stream
        owner_plan_count_push_args = (
            comm_to_cute(
                self._transport.owner_plan_count_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_count_signal_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_count_signal_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_count_tx.reshape(-1)
            ),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(plan_stream.cuda_stream),
        )
        owner_plan_route_push_args = (
            comm_to_cute(
                self._transport.owner_plan_count_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_signal_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_signal_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_bases_tx.reshape(-1)
            ),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(plan_stream.cuda_stream),
        )

        if self._transport.ring_stream is None:
            self._transport.ring_stream = torch.cuda.Stream(priority=0)
            self._transport.ring_recv_stream = torch.cuda.Stream(priority=0)
        payload_bytes = (
            self.problem.hidden * self.combine_output.element_size()
        )
        chunk_rows = self.kernel_config.combine_chunk_rows
        # Keep the worker grid resident in its direction-specific Green
        # Context.  Workers own disjoint peer/channel stripes, avoiding both
        # head-of-line blocking and an oversubscribed polling grid.
        combine_tasks = (
            self.world_size * self._transport.combine_ring_channels
        )
        combine_tx_workers = combine_tasks
        combine_rx_workers = combine_tasks
        self._transport.ring_sender_kernel = GinGpuRingCombineSender(
            self.world_size,
            payload_bytes,
            self._transport.max_pair_routes,
            chunk_rows=chunk_rows,
            channels=self._transport.combine_ring_channels,
            slots=self._transport.combine_ring_slots,
            hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
            worker_ctas=combine_tx_workers,
            num_experts_per_rank=self.problem.num_experts_per_rank,
            blocking_put=(
                self.kernel_config.combine_blocking_put
                or self.problem.num_tokens_per_rank <= 128
            ),
            progress_windows=(self.problem.num_tokens_per_rank <= 128),
        )
        self._transport.ring_receiver_kernel = GinGpuRingCombineReceiver(
            self.world_size,
            payload_bytes,
            self._transport.max_pair_routes,
            chunk_rows=chunk_rows,
            channels=self._transport.combine_ring_channels,
            slots=self._transport.combine_ring_slots,
            hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
            worker_ctas=combine_rx_workers,
            num_experts_per_rank=self.problem.num_experts_per_rank,
            count_block_elements=_IBGDA_OWNER_PLAN_BLOCK_ELEMENTS,
            plan_slots=_IBGDA_OWNER_PLAN_SLOTS,
            progress_windows=(self.problem.num_tokens_per_rank <= 128),
        )
        max_chunks_per_peer = (
            self._transport.max_pair_routes + chunk_rows - 1
        ) // chunk_rows
        self._transport.combine_ready_queue_done = torch.zeros(
            (self.world_size * max_chunks_per_peer,),
            dtype=torch.int32,
            device="cuda",
        )
        self._transport.combine_ready_queue_probe = torch.full(
            (
                self.world_size
                * self._transport.combine_ring_channels
                * 4,
            ),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self._transport.combine_ready_queue_selected = torch.full(
            (
                self.world_size
                * self._transport.combine_ring_channels,
            ),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        combine_tasks = (
            self.world_size * self._transport.combine_ring_channels
        )
        self._transport.combine_send_work_queue = torch.zeros(
            (
                2 * combine_tasks
                + self._transport.ring_sender_kernel.worker_ctas
                + 2,
            ),
            dtype=torch.int32,
            device="cuda",
        )
        self._transport.combine_recv_work_queue = torch.zeros(
            (
                combine_tasks
                + self._transport.ring_receiver_kernel.worker_ctas
                + 1,
            ),
            dtype=torch.int32,
            device="cuda",
        )
        kernel = self._kernel
        fc2_spec = kernel._local_region_by_name["fc2_output_workspace"]
        fc2_done_spec = kernel._local_region_by_name["fc2_done_counter"]
        fc2_ws_full = self.local_workspace[
            kernel._local_offsets["fc2_output_workspace"]:
            kernel._local_offsets["fc2_output_workspace"] + fc2_spec.nbytes
        ]
        # The full worst-case pool can exceed the signed 32-bit memref
        # length accepted by the current CuTe runtime.  Ring rows address
        # a dynamically populated prefix.  Keep the descriptor below
        # INT32_MAX bytes; device indexing is pointer based and the
        # backing allocation remains the kernel's full worst-case pool.
        row_bytes = self.problem.hidden * self.combine_output.element_size()
        descriptor_row_limit = max(1, (2**31 - 1) // row_bytes)
        used_pool_rows = min(
            self._kernel.pool_token_capacity,
            descriptor_row_limit,
        )
        fc2_used_bytes = (
            used_pool_rows
            * self.problem.hidden
            * self.combine_output.element_size()
        )
        fc2_ws = fc2_ws_full.narrow(0, 0, fc2_used_bytes)
        fc2_done = self.local_workspace[
            kernel._local_offsets["fc2_done_counter"]:
            kernel._local_offsets["fc2_done_counter"] + fc2_done_spec.nbytes
        ].view(torch.int32)
        direct_map_byte_offset = (
            kernel._local_offsets["ibgda_sf_staging"]
            + kernel.ibgda_direct_stage_map_offset_i32 * 4
        )
        self._transport.combine_pool_to_stage_rows = self.local_workspace[
            direct_map_byte_offset:
            direct_map_byte_offset + kernel.pool_token_capacity * 4
        ].view(torch.int32)
        self._transport.ring_epoch = 1
        ring_sender_args = (
            comm_to_cute(self._transport.ring_rx.reshape(-1)),
            comm_to_cute(self._transport.ring_tx.reshape(-1)),
            comm_to_cute(self._transport.ring_ack.reshape(-1)),
            comm_to_cute(fc2_ws),
            comm_to_cute(
                self._transport.ring_send_output_rows.reshape(-1)
            ),
            comm_to_cute(
                self._transport.ring_send_ready_slots.reshape(-1)
            ),
            comm_to_cute(
                self._transport.ring_send_ready_values.reshape(-1)
            ),
            comm_to_cute(self._transport.ring_send_counts),
            # Keep the host memref in BF16. Flattening the full output to
            # bytes can exceed CuTe's signed 32-bit layout-size limit.
            # The ring kernel performs its own vector recasts and Int64 indexing.
            comm_to_cute(self.combine_output.reshape(-1)),
            comm_to_cute(fc2_done),
            comm_to_cute(self._transport.combine_ready_queue_done),
            comm_to_cute(self._transport.combine_ready_queue_probe),
            comm_to_cute(self._transport.combine_ready_queue_selected),
            comm_to_cute(self._transport.dispatch_tx_handoff.reshape(-1)),
            comm_to_cute(self._transport.combine_send_work_queue),
            self.rank,
            self._transport.ring_epoch,
            cuda.CUstream(self._transport.ring_stream.cuda_stream),
        )
        ring_ack_args = (
            comm_to_cute(self._transport.ring_ack.reshape(-1)),
            comm_to_cute(self._transport.ring_send_counts),
            self.rank,
            self._transport.ring_epoch,
            cuda.CUstream(self._transport.ring_stream.cuda_stream),
        )
        ring_receiver_args = (
            comm_to_cute(self._transport.ring_rx.reshape(-1)),
            comm_to_cute(self._transport.ring_ack.reshape(-1)),
            comm_to_cute(self._transport.ring_ack_tx.reshape(-1)),
            comm_to_cute(
                self._transport.owner_plan_count_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.combine_recv_output_rows.reshape(-1)
            ),
            comm_to_cute(self.combine_output.reshape(-1)),
            comm_to_cute(self._transport.combine_recv_work_queue),
            self.rank,
            self._transport.ring_epoch,
            cuda.CUstream(self._transport.ring_recv_stream.cuda_stream),
        )


        kernel = self._kernel
        meta = self._byte_workspace_view(
            self.shared_workspace,
            kernel._shared_offsets["src_token_topk_idx"],
            torch.int32,
            kernel._shared_region_by_name["src_token_topk_idx"].shape,
        )
        counts = self._byte_workspace_view(
            self.shared_workspace,
            kernel._shared_offsets["expert_recv_count"],
            torch.int64,
            kernel._shared_region_by_name["expert_recv_count"].shape,
        )
        sums = self._byte_workspace_view(
            self.shared_workspace,
            kernel._shared_offsets["expert_recv_count_sum"],
            torch.int64,
            kernel._shared_region_by_name["expert_recv_count_sum"].shape,
        )
        if self._transport.route_stream is None:
            self._transport.route_stream = torch.cuda.Stream(priority=0)
        route_stream = cuda.CUstream(self._transport.route_stream.cuda_stream)
        self._transport.dispatch_route_builder_kernel = GpuDispatchRouteBuilder(
            self.world_size,
            self.problem.num_tokens_per_rank,
            self.problem.num_topk,
            self.problem.num_experts_per_rank,
            owner_route_chunk_elements=_IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS,
            owner_route_capacity=self._transport.owner_plan_route_capacity,
            owner_plan_slots=_IBGDA_OWNER_PLAN_SLOTS,
        )
        dispatch_route_builder_args = (
            comm_to_cute(self.my_topk_idx.reshape(-1)),
            comm_to_cute(
                self._transport.dispatch_send_token_indices.reshape(-1)
            ),
            comm_to_cute(self._transport.dispatch_send_counts),
            comm_to_cute(self._transport.owner_plan_count_tx.reshape(-1)),
            comm_to_cute(
                self._transport.owner_plan_route_bases_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_owner_route_output_bases.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_cursors_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_owner_token_counts.reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_owner_token_bases.reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_owner_token_cursors.reshape(-1)
            ),
            comm_to_cute(self._transport.owner_plan_route_tx.reshape(-1)),
            comm_to_cute(
                self._transport.combine_recv_output_rows.reshape(-1)
            ),
            self._transport.dispatch_epoch,
            route_stream,
        )
        self._transport.owner_route_builder_kernel = (
            GpuChunkedOwnerRoutePlanBuilder(
                self.world_size,
                self.problem.num_tokens_per_rank,
                self.problem.num_topk,
                self.problem.num_experts_per_rank,
                kernel.token_padding_block,
                self._kernel_k2.k2_token_tile_tokens,
                (
                    self._kernel_k2.token_comm
                    .fc2_publishes_per_token_cluster_tile
                ),
                self._transport.k1_sm_count,
                self._transport.owner_plan_route_capacity,
                route_block_elements=_IBGDA_OWNER_PLAN_BLOCK_ELEMENTS,
                plan_slots=_IBGDA_OWNER_PLAN_SLOTS,
                hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
            )
        )
        owner_count_builder_args = (
            comm_to_cute(self._transport.owner_plan_count_rx.reshape(-1)),
            comm_to_cute(
                self._transport.owner_plan_count_signal_rx.reshape(-1)
            ),
            comm_to_cute(counts.reshape(-1)),
            comm_to_cute(sums.reshape(-1)),
            comm_to_cute(self._transport.expert_pool_bases),
            comm_to_cute(
                self._transport.combine_source_expert_bases.reshape(-1)
            ),
            comm_to_cute(self._transport.ring_send_counts),
            comm_to_cute(
                self._transport.owner_plan_route_bases_rx.reshape(-1)
            ),
            self.rank,
            self._transport.dispatch_epoch,
            route_stream,
        )
        metadata_stream = cuda.CUstream(
            self._transport.owner_metadata_stream.cuda_stream
        )
        owner_metadata_builder_args = (
            comm_to_cute(self._transport.owner_plan_route_rx.reshape(-1)),
            comm_to_cute(
                self._transport.owner_plan_route_signal_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_bases_rx.reshape(-1)
            ),
            comm_to_cute(meta.reshape(-1)),
            comm_to_cute(counts.reshape(-1)),
            comm_to_cute(sums.reshape(-1)),
            comm_to_cute(self._transport.expert_pool_bases),
            comm_to_cute(
                self._transport.combine_source_expert_bases.reshape(-1)
            ),
            comm_to_cute(self._transport.ring_send_rows.reshape(-1)),
            comm_to_cute(
                self._transport.ring_send_output_rows.reshape(-1)
            ),
            comm_to_cute(
                self._transport.ring_send_ready_slots.reshape(-1)
            ),
            comm_to_cute(
                self._transport.ring_send_ready_values.reshape(-1)
            ),
            comm_to_cute(
                self._transport.combine_pool_to_stage_rows.reshape(-1)
            ),
            comm_to_cute(self._transport.owner_plan_ready),
            self._transport.dispatch_epoch,
            metadata_stream,
        )
        for attribute, role in (
            ("owner_plan_push_kernel", "dispatch_plan"),
            ("owner_route_builder_kernel", "dispatch_plan"),
            ("dispatch_ring_sender_kernel", "dispatch_tx"),
            ("dispatch_ring_receiver_kernel", "dispatch_rx"),
            ("ring_sender_kernel", "return_tx"),
            ("ring_receiver_kernel", "return_rx"),
        ):
            self.bridge.bind(getattr(self._transport, attribute), role)
        link_options = f"--link-libraries={self.bridge.device_bitcode()}"
        self._transport.compiled_owner_plan_count_push = cute_module.compile(
            self._transport.owner_plan_push_kernel.count_launcher,
            *owner_plan_count_push_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_owner_plan_route_push = cute_module.compile(
            self._transport.owner_plan_push_kernel.route_launcher,
            *owner_plan_route_push_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_dispatch_ring_sender = cute_module.compile(
            self._transport.dispatch_ring_sender_kernel.launcher,
            *dispatch_sender_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_dispatch_ring_ack_waiter = cute_module.compile(
            self._transport.dispatch_ring_sender_kernel.ack_launcher,
            *dispatch_ack_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_dispatch_ring_receiver = cute_module.compile(
            self._transport.dispatch_ring_receiver_kernel.launcher,
            *dispatch_receiver_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_ring_sender = cute_module.compile(
            self._transport.ring_sender_kernel.launcher,
            *ring_sender_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_ring_ack_waiter = cute_module.compile(
            self._transport.ring_sender_kernel.ack_launcher,
            *ring_ack_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_ring_receiver = cute_module.compile(
            self._transport.ring_receiver_kernel.launcher,
            *ring_receiver_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_dispatch_route_builder = cute_module.compile(
            self._transport.dispatch_route_builder_kernel.launcher,
            *dispatch_route_builder_args,
            **self._epoch_kwargs(),
        )
        self._transport.compiled_owner_count_builder = cute_module.compile(
            self._transport.owner_route_builder_kernel.count_launcher,
            *owner_count_builder_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.compiled_owner_metadata_builder = cute_module.compile(
            self._transport.owner_route_builder_kernel.metadata_launcher,
            *owner_metadata_builder_args,
            **self._epoch_kwargs(),
            options=link_options,
        )
        self._transport.dispatch_ring_sender_runtime_args = dispatch_sender_args
        self._transport.dispatch_ring_ack_runtime_args = dispatch_ack_args
        self._transport.dispatch_ring_receiver_runtime_args = dispatch_receiver_args
        self._transport.ring_sender_runtime_args = ring_sender_args
        self._transport.ring_ack_runtime_args = ring_ack_args
        self._transport.ring_receiver_runtime_args = ring_receiver_args
        self._transport.dispatch_route_builder_runtime_args = (
            dispatch_route_builder_args
        )
        self._transport.owner_plan_count_push_runtime_args = (
            owner_plan_count_push_args
        )
        self._transport.owner_plan_route_push_runtime_args = (
            owner_plan_route_push_args
        )
        self._transport.owner_count_builder_runtime_args = (
            owner_count_builder_args
        )
        self._transport.owner_metadata_builder_runtime_args = (
            owner_metadata_builder_args
        )
        # Materialize every communication function before starting persistent
        # receivers. A first-use module load may synchronize the context and
        # deadlock behind a receiver that is waiting for that very sender.
        # CUDA_MODULE_LOADING=EAGER alone is insufficient for CuTe CUlibrary.
        def checked(result):
            if int(result[0]) != 0:
                raise RuntimeError(f'Communication function preload failed: {result}')
            return result[1] if len(result) == 2 else result[1:]
        loaded = set()
        for name, compiled in vars(self._transport).items():
            if not name.startswith('compiled_') or compiled is None:
                continue
            library = int(compiled.library)
            if library in loaded:
                continue
            loaded.add(library)
            module = checked(cuda.cuLibraryGetModule(cuda.CUlibrary(library)))
            count = checked(cuda.cuModuleGetFunctionCount(module))
            for function in checked(cuda.cuModuleEnumerateFunctions(count, module)):
                checked(cuda.cuFuncLoad(function))


    def _reset_local_counters(self, *, fused: bool = False) -> None:
        """Zero only the local flag/counter regions between back-to-back
        launches.  The caller must first drain the preceding replay because
        these regions are also consumed by persistent dispatch/combine CTAs.
        The kernel does not reset its accumulating local counters
        (``fc1_done_counter`` / ``fc2_done_counter`` spin thresholds,
        ``load_balance_counter`` work-stealing cursor, ``expert_send_count`` /
        ``l1_arrival_count`` / ``atomic_counter`` dispatch write-cursors, plus
        the phase-flip ``grid_sync_counter``), so a re-launch with stale values
        deadlocks or writes out of bounds. ``nvlink_barrier_counter`` and its
        peer-written signal slots are generation state and survive replay. Data
        buffers (token / act / sf / fc1_output / fc2_output_workspace / topk
        weights) are fully overwritten each launch and are deliberately NOT
        zeroed -- a large memset between launches desyncs the ranks and pollutes
        the steady-state entry-barrier timing. Peer-written control regions
        below are safe to reset only after the preceding replay has drained;
        K1's device-side entry barrier joins every rank after this launch-
        ordered reset before any peer can publish the next replay's counts.
        A memset is not a ``kernel_cutlass*`` event, so it stays out of that
        timing."""
        kernel = self._kernel
        if kernel is None or self.local_workspace is None:
            return
        # fused=True (graph-owned reset): collect the constant fills and issue
        # them as one kernel instead of one fill node per region.
        fused_ops = [] if fused else None

        def _fill(t, value):
            if fused_ops is not None and isinstance(value, int):
                fused_ops.append((t, value))
            else:
                t.fill_(value)
        for name in (
            "l1_arrival_count",
            "expert_send_count",
            "grid_sync_counter",
            "fc1_done_counter",
            "fc2_done_counter",
            "atomic_counter",
            "load_balance_counter",
            "k2_ready_queue_desc",
            "k2_ready_queue_ready",
            "k2_ready_queue_state",
            "dispatch_rank_cache_state",
            "rank_combine_tile_done",
        ):
            if name in kernel._local_offsets:
                off = kernel._local_offsets[name]
                nbytes = kernel._local_region_by_name[name].nbytes
                _fill(self.local_workspace[off:off + nbytes], 0)
        if "rank_combine_route_map" in kernel._local_offsets:
            off = kernel._local_offsets["rank_combine_route_map"]
            nbytes = kernel._local_region_by_name[
                "rank_combine_route_map"
            ].nbytes
            _fill(self.local_workspace[off:off + nbytes].view(torch.int32), -1)
        if self.rank_combine_ready is not None:
            if self.device_epoch is None:
                self._rank_combine_epoch += 1
                self.rank_combine_ready[-1, 0].fill_(self._rank_combine_epoch)
            else:
                self.rank_combine_ready[-1, 0].add_(1)
        # K2 uses the high 32 bits of expert_recv_count_sum as K1's
        # count-publication marker. The symmetric workspace survives CUDA
        # Graph replays, so leaving this marker intact lets K2 consume stale
        # expert extents before K1 has published the current replay. Reset
        # only this small control region; K1 still overwrites the data pools.
        if (
            self._kernel_k2_finalizer is None
            and self.shared_workspace is not None
            and "expert_recv_count_sum" in kernel._shared_offsets
        ):
            for name in (
                "expert_recv_count",
                "expert_recv_count_sum",
                "rank_combine_route_count",
            ):
                if name in kernel._shared_offsets:
                    off = kernel._shared_offsets[name]
                    nbytes = kernel._shared_region_by_name[name].nbytes
                    _fill(self.shared_workspace[off:off + nbytes], 0)
        if self.comm_backend == "gin":
            ready_bytes = (
                self._kernel.token_comm.ibgda_dispatch_ready_words
                + self._kernel.token_comm.ibgda_owner_plan_ready_words
            ) * 4
            ready_off = self._kernel._local_offsets["ibgda_sf_staging"]
            _fill(self.local_workspace[
                ready_off:ready_off + ready_bytes
            ], 0)
            counts = self._byte_workspace_view(
                self.shared_workspace,
                kernel._shared_offsets["expert_recv_count"],
                torch.int64,
                kernel._shared_region_by_name["expert_recv_count"].shape,
            )
            _fill(counts, 0)
            _fill(self._transport.dispatch_send_counts, 0)
            _fill(self._transport.dispatch_send_work_queue, 0)
            _fill(self._transport.dispatch_recv_work_queue, 0)
            _fill(self._transport.combine_send_work_queue, 0)
            _fill(self._transport.combine_recv_work_queue, 0)
            _fill(self._transport.dispatch_owner_token_counts, 0)
            _fill(self._transport.dispatch_owner_token_cursors, 0)
            _fill(self._transport.ring_send_counts, 0)
            _fill(self._transport.owner_plan_count_tx, 0)
            _fill(self._transport.owner_plan_route_cursors_tx, 0)
        if self._split_green_trace is not None:
            _fill(self._split_green_trace, 0)
        if self._transport.combine_ready_queue_done is not None:
            _fill(self._transport.combine_ready_queue_done, 0)
        if self._transport.combine_ready_queue_probe is not None:
            _fill(self._transport.combine_ready_queue_probe, -1)
        if self._transport.combine_ready_queue_selected is not None:
            _fill(self._transport.combine_ready_queue_selected, -1)
        if fused_ops:
            from moe_sm120_nvfp4_split.runtime.fused_fill import launch_fused_fill

            launch_fused_fill(fused_ops, torch.cuda.current_stream())


    def _ibgda_streaming_dispatch_begin(self) -> None:
        if not getattr(self, '_gin_receiver_preloaded', False):
            # Load/allocate receiver resources before any persistent ACK wait.
            # Epoch zero and the initialized zero-count control records make
            # this an empty receive: no payload, signal, ACK or GIN RMA.
            assert self._transport.dispatch_epoch == 1
            assert torch.count_nonzero(self._transport.dispatch_control_rx.view(torch.int32)[::32]).item() == 0
            prewarm_args = list(self._transport.dispatch_ring_receiver_runtime_args)
            prewarm_args[-2] = 0
            prewarm_args[-1] = cuda.CUstream(self._transport.dispatch_recv_stream.cuda_stream)
            self._transport.compiled_dispatch_ring_receiver(*prewarm_args, **self._epoch_kwargs())
            self._transport.dispatch_recv_stream.synchronize()
            self._gin_receiver_preloaded = True
        tokens = self.problem.num_tokens_per_rank
        own = self.rank
        activation_by_rank = self._transport.activation_rx.view(
            self.world_size, tokens, *self.my_activation.shape[1:]
        )
        sf_by_rank = self._transport.sf_rx.view(
            self.world_size, tokens, *self.my_activation_sf.shape[1:]
        )
        weight_by_rank = self._transport.topk_weight_rx.view(
            self.world_size, tokens, self.problem.num_topk
        )
        phase = self._phase_timing_begin("dispatch.streaming_ring")
        self._transport.dispatch_epoch += 1
        launch_stream = torch.cuda.current_stream()
        route_phase = self._phase_timing_begin("route.dispatch_builder")
        route_args = list(self._transport.dispatch_route_builder_runtime_args)
        route_args[-2] = self._transport.dispatch_epoch
        route_ready = torch.cuda.Event()
        route_ready.record(launch_stream)
        self._transport.route_stream.wait_event(route_ready)
        route_args[-1] = cuda.CUstream(self._transport.route_stream.cuda_stream)
        self._transport.compiled_dispatch_route_builder(*route_args, **self._epoch_kwargs())
        route_done = torch.cuda.Event()
        route_done.record(self._transport.route_stream)
        launch_stream.wait_event(route_done)
        self._phase_timing_end(route_phase)
        activation_by_rank[own].copy_(self.my_activation)
        sf_by_rank[own].copy_(self.my_activation_sf)
        weight_by_rank[own].copy_(self.my_topk_weights)
        self._publish_dispatch_epoch(self._transport.dispatch_ready[0:1])
        self._publish_dispatch_epoch(self._transport.dispatch_ready[
            1 + own * tokens:1 + (own + 1) * tokens
        ])

        sender_args = list(self._transport.dispatch_ring_sender_runtime_args)
        ack_args = list(self._transport.dispatch_ring_ack_runtime_args)
        receiver_args = list(self._transport.dispatch_ring_receiver_runtime_args)
        sender_args[-2] = self._transport.dispatch_epoch
        ack_args[-2] = self._transport.dispatch_epoch
        receiver_args[-2] = self._transport.dispatch_epoch
        ready = torch.cuda.Event()
        ready.record(launch_stream)
        for stream in (
            self._transport.dispatch_send_stream,
            self._transport.dispatch_recv_stream,
        ):
            stream.wait_event(ready)
        plan_stream = self._transport.owner_plan_stream
        plan_stream.wait_event(ready)
        with torch.cuda.stream(plan_stream):
            count_push_args = list(
                self._transport.owner_plan_count_push_runtime_args
            )
            route_push_args = list(
                self._transport.owner_plan_route_push_runtime_args
            )
            count_push_args[-2] = self._transport.dispatch_epoch
            route_push_args[-2] = self._transport.dispatch_epoch
            count_push_args[-1] = cuda.CUstream(plan_stream.cuda_stream)
            route_push_args[-1] = cuda.CUstream(plan_stream.cuda_stream)
            self._transport.compiled_owner_plan_count_push(*count_push_args, **self._epoch_kwargs())
            self._transport.owner_plan_count_push_done = torch.cuda.Event()
            self._transport.owner_plan_count_push_done.record(plan_stream)
            # Route blocks and activation payloads use independent TX streams;
            # owner metadata joins both at source/expert granularity.
            self._transport.compiled_owner_plan_route_push(*route_push_args, **self._epoch_kwargs())
            self._transport.owner_plan_route_push_done = torch.cuda.Event()
            self._transport.owner_plan_route_push_done.record(plan_stream)
        # Give the IBGDA producer a head start before enqueuing persistent
        # polling CTAs.  Launching the receiver first can occupy communication
        # progress resources while every CTA is waiting on its first signal.
        with torch.cuda.stream(self._transport.dispatch_send_stream):
            sender_phase = self._phase_timing_begin(
                "dispatch.tx_sender",
                stream=self._transport.dispatch_send_stream,
            )
            self._transport.compiled_dispatch_ring_sender(*sender_args, **self._epoch_kwargs())
            self._phase_timing_end(
                sender_phase,
                stream=self._transport.dispatch_send_stream,
            )
            # Publish TX-grid completion before joining terminal credits.
            # Combine may reuse the TX partition as soon as dispatch has
            # submitted all payloads; the ACK waiter remains ordered on this
            # stream and independently protects the next epoch's slot reuse.
            self._transport.dispatch_send_done = torch.cuda.Event()
            self._transport.dispatch_send_done.record(
                self._transport.dispatch_send_stream
            )
            self._transport.compiled_dispatch_ring_ack_waiter(*ack_args, **self._epoch_kwargs())
            self._transport.dispatch_credit_done = torch.cuda.Event()
            self._transport.dispatch_credit_done.record(
                self._transport.dispatch_send_stream
            )
        with torch.cuda.stream(self._transport.dispatch_recv_stream):
            receiver_phase = self._phase_timing_begin(
                "dispatch.rx_receiver",
                stream=self._transport.dispatch_recv_stream,
            )
            self._transport.compiled_dispatch_ring_receiver(*receiver_args, **self._epoch_kwargs())
            self._phase_timing_end(
                receiver_phase,
                stream=self._transport.dispatch_recv_stream,
            )
            self._transport.dispatch_recv_done = torch.cuda.Event()
            self._transport.dispatch_recv_done.record(
                self._transport.dispatch_recv_stream
            )
        # Record dispatch completion on the receiver stream itself.  Ending
        # this phase after K1/K2 joined would fold compute overlap into the
        # reported dispatch time.
        self._phase_timing_end(
            phase, stream=self._transport.dispatch_recv_stream
        )
        self._transport.dispatch_ring_phase = None


    def _ibgda_streaming_dispatch_end(self) -> None:
        current = torch.cuda.current_stream()
        current.wait_event(self._transport.dispatch_recv_done)
        current.wait_event(self._transport.dispatch_send_done)


    def _ibgda_build_runtime_owner_route_plan(self) -> None:
        """Install this replay's K1 and combine metadata entirely on GPU."""
        phase = self._phase_timing_begin("route.owner_control_builder")
        current = torch.cuda.current_stream()
        if self._transport.ring_credit_done is not None:
            # ring_send_counts is also the previous epoch's ACK snapshot.
            self._transport.route_stream.wait_event(
                self._transport.ring_credit_done
            )
        self._transport.route_stream.wait_event(
            self._transport.owner_plan_count_push_done
        )
        count_args = list(self._transport.owner_count_builder_runtime_args)
        count_args[-2] = self._transport.dispatch_epoch
        count_args[-1] = cuda.CUstream(self._transport.route_stream.cuda_stream)
        self._transport.compiled_owner_count_builder(*count_args, **self._epoch_kwargs())
        count_done = torch.cuda.Event()
        count_done.record(self._transport.route_stream)
        current.wait_event(count_done)
        # Metadata is produced incrementally after the compact count record.
        # K1 and combine TX consume per-source/per-expert ready epochs; the
        # complete metadata event is not an admission barrier.
        self._transport.owner_metadata_stream.wait_event(count_done)
        metadata_args = list(
            self._transport.owner_metadata_builder_runtime_args
        )
        metadata_args[-2] = self._transport.dispatch_epoch
        metadata_args[-1] = cuda.CUstream(
            self._transport.owner_metadata_stream.cuda_stream
        )
        self._transport.compiled_owner_metadata_builder(*metadata_args, **self._epoch_kwargs())
        self._transport.owner_metadata_done = torch.cuda.Event()
        self._transport.owner_metadata_done.record(
            self._transport.owner_metadata_stream
        )
        self._phase_timing_end(phase)


    def _ibgda_streaming_combine_begin(self) -> None:
        """Start per-channel combine RX/TX without a full-plan barrier."""
        self._transport.ring_done = None
        current = torch.cuda.current_stream()
        self._transport.ring_epoch += 1
        if self._transport.ring_epoch != self._transport.dispatch_epoch:
            raise RuntimeError("dispatch/combine TX handoff epochs diverged")
        sender_args = list(self._transport.ring_sender_runtime_args)
        ack_args = list(self._transport.ring_ack_runtime_args)
        receiver_args = list(self._transport.ring_receiver_runtime_args)
        sender_args[-2] = self._transport.ring_epoch
        ack_args[-2] = self._transport.ring_epoch
        receiver_args[-2] = self._transport.ring_epoch
        launch_ready = torch.cuda.Event()
        launch_ready.record(current)
        self._transport.ring_stream.wait_event(launch_ready)
        self._transport.ring_recv_stream.wait_event(launch_ready)
        # Admission is exclusively the device-side per-peer/channel epoch
        # handoff. Terminal ACK/credit recovery remains independently ordered
        # and protects cyclic slot reuse across replays.
        self._transport.ring_phase = self._phase_timing_begin(
            "combine.streaming_ring"
        )
        with torch.cuda.stream(self._transport.ring_recv_stream):
            receiver_phase = self._phase_timing_begin(
                "combine.rx_receiver",
                stream=self._transport.ring_recv_stream,
            )
            self._transport.compiled_ring_receiver(*receiver_args, **self._epoch_kwargs())
            self._phase_timing_end(
                receiver_phase,
                stream=self._transport.ring_recv_stream,
            )
            self._transport.ring_recv_done = torch.cuda.Event()
            self._transport.ring_recv_done.record(self._transport.ring_recv_stream)
        with torch.cuda.stream(self._transport.ring_stream):
            sender_phase = self._phase_timing_begin(
                "combine.tx_sender",
                stream=self._transport.ring_stream,
            )
            self._transport.compiled_ring_sender(*sender_args, **self._epoch_kwargs())
            self._phase_timing_end(
                sender_phase,
                stream=self._transport.ring_stream,
            )
            # K3 depends on submission/receive completion, not credit return.
            self._transport.ring_done = torch.cuda.Event()
            self._transport.ring_done.record(self._transport.ring_stream)
            self._transport.compiled_ring_ack_waiter(*ack_args, **self._epoch_kwargs())
            self._transport.ring_credit_done = torch.cuda.Event()
            self._transport.ring_credit_done.record(
                self._transport.ring_stream
            )


    def _ibgda_streaming_combine_end(self) -> None:
        """Join the GPU ring without a CPU synchronize or rank barrier."""
        current = torch.cuda.current_stream()
        current.wait_event(self._transport.ring_done)
        current.wait_event(self._transport.ring_recv_done)
        self._phase_timing_end(self._transport.ring_phase)


    def _wait_for_previous_streaming_replay(self) -> None:
        """Order replay-local resets after all prior transport consumers.

        This is the production dependency: terminal credits protect cyclic
        slots and work queues, while receive completion protects the local
        ready/control records.  Host synchronization and rank barriers remain
        a debug/stress aid, not part of the performance path.
        """
        current = torch.cuda.current_stream()
        for event in (
            self._transport.dispatch_credit_done,
            self._transport.dispatch_recv_done,
            self._transport.ring_credit_done,
            self._transport.ring_recv_done,
            self._transport.owner_plan_route_push_done,
            self._transport.owner_metadata_done,
        ):
            if event is not None:
                current.wait_event(event)
