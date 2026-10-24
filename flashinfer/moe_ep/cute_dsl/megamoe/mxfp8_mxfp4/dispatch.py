# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVLink MXFP8 token dispatch for the persistent MegaMoE.

The implementation is a thin, FlashInfer-owned launcher around the existing
CuTe-DSL ``TokenInPullTokenBackPush`` standalone device entry.  Even for one
rank we intentionally use the device protocol: it is the source of truth for
expert-major pool ordering, scale-factor atom swizzling, metadata packing, and
release publication to the FC1 scheduler.

Routing scores are copied into the compatibility pool because the shared
communication ABI requires it.  They are not applied by dispatch or FC1; the
authoritative FP32 scores remain on the home rank for the post-FC2 reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .device_workspace import DeviceWorkspacePartition
from .workspace import WorkspaceConfig


@dataclass(frozen=True)
class DeviceExpertPoolOffsets:
    """Device-side prefix contract derived from packed expert counts.

    ``expert_counts`` is the low-i32 strided view of
    ``expert_recv_count_sum``. Pool prefixes use expert index order, while
    the scheduler may visit a permutation of those experts. Materializing
    a second offsets array would add a kernel and an unnecessary synchronization
    point, so this object carries the exact derivation contract instead.
    """

    expert_counts: torch.Tensor
    token_padding_block: int
    sf_padding_block: int
    cluster_tile_tokens: int


@dataclass(frozen=True)
class DispatchProducts:
    """Bound dispatch launch and the device products consumed downstream."""

    launch: Callable[[], None]
    expert_recv_count_sum: torch.Tensor
    expert_pool_offsets: DeviceExpertPoolOffsets
    token_src_metadata: torch.Tensor
    fc1_ready_counters: torch.Tensor
    token_pool: torch.Tensor
    scale_pool: torch.Tensor
    compatibility_route_scores: torch.Tensor

    def __call__(self) -> None:
        self.launch()


@dataclass(frozen=True)
class InKernelDispatchProducts:
    """CuTe tensor products published by the in-kernel dispatch phase."""

    expert_recv_count_sum: Any
    token_src_metadata: Any
    fc1_ready_counters: Any
    token_pool: Any
    scale_pool: Any
    compatibility_route_scores: Any


class Mxfp8DispatchPhase:
    """Device-callable dispatch phase embedded in the persistent kernel.

    This object does not launch a kernel.  The assembler allocates
    :meth:`shared_storage_type` in its CTA shared storage, constructs the
    existing token-communication argument bundle with :meth:`make_args`, then
    calls this object from each dispatch warp.  The scheduler warp must call
    :meth:`scheduler_wait` before consuming expert counts. The MXFP8 override
    overwrites owner-side counts each launch and needs only one tail drain;
    other backends retain the vendored helper's reset protocol.
    """

    def __init__(
        self,
        config: WorkspaceConfig,
        *,
        cluster_shape_mn: tuple[int, int],
        dispatch_warp_start: int,
        num_other_warps: int,
        flag_batch: int = 1,
        rank: int = 0,
        cta_local_counts: bool = False,
    ) -> None:
        if not 1 <= config.world_size <= 16:
            raise ValueError("NVLink dispatch supports 1 <= world_size <= 16")
        if not 0 <= rank < config.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if len(cluster_shape_mn) != 2 or any(v <= 0 for v in cluster_shape_mn):
            raise ValueError("cluster_shape_mn must contain two positive values")
        if dispatch_warp_start < 0 or num_other_warps < 1:
            raise ValueError(
                "persistent dispatch requires non-negative dispatch_warp_start "
                "and at least one co-resident scheduler warp"
            )
        if not 1 <= flag_batch <= 32:
            raise ValueError("flag_batch must be in [1, 32]")
        self.config = config
        self.rank = rank
        self.cluster_shape_mn = cluster_shape_mn
        self.dispatch_warp_start = dispatch_warp_start
        self.num_other_warps = num_other_warps
        self.flag_batch = flag_batch
        self.cta_local_counts = cta_local_counts
        self._token_comm: Any | None = None
        self._tail_reduce: Any | None = None

    def bind_in_kernel_reduce(self, route_terms: Any, output: Any) -> None:
        """Bind device-region operands for the optional all-warp tail reduce."""

        from .fc2_epilogue import InKernelTopKReduction

        self._tail_reduce = InKernelTopKReduction(
            hidden=self.config.hidden,
            num_topk=self.config.num_topk,
            route_terms=route_terms,
            output=output,
        )

    @property
    def token_comm(self) -> Any:
        if self._token_comm is None:
            import cutlass

            from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src import (
                token_comm as token_comm_module,
            )

            from .tail import overwrite_count_token_comm_class

            comm_class = overwrite_count_token_comm_class(
                token_comm_module.TokenInPullTokenBackPush
            )
            from .token_comm import make_token_comm_binding

            binding = make_token_comm_binding(
                self.config,
                cluster_shape_mn=self.cluster_shape_mn,
                dispatch_warp_start=self.dispatch_warp_start,
                num_other_warps=self.num_other_warps,
                flag_batch=self.flag_batch,
            )
            self._token_comm = binding.instantiate(
                helper_type=comm_class,
                fc1_token_dtype=cutlass.Float8E4M3FN,
                combine_format=None,
            )
            self._token_comm.cta_local_counts = self.cta_local_counts
        return self._token_comm

    @property
    def num_dispatch_warps(self) -> int:
        return self.token_comm.num_dispatch_warps

    def shared_storage_type(self) -> type:
        """Return the CuTe struct the assembler must allocate once per CTA."""

        return self.token_comm.extra_smem_storage_class()

    def make_args(self, **kwargs: Any) -> Any:
        """Construct the vendored argument bundle without reinterpreting views."""

        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.token_comm import (
            TokenCommArgs,
        )

        return TokenCommArgs(
            world_size=self.config.world_size,
            local_rank=self.rank,
            num_total_experts=self.config.num_total_experts,
            num_experts_per_rank=self.config.num_experts_per_rank,
            num_topk=self.config.num_topk,
            hidden_bytes=self.config.hidden,
            sf_uint32_per_token=self.config.sf_uint32_per_token,
            token_padding_block=self.config.token_padding_block,
            sf_padding_block=self.config.sf_padding_block,
            **kwargs,
        )

    @staticmethod
    def products(token_comm_args: Any) -> InKernelDispatchProducts:
        """Expose the exact tensors published before FC1 consumes work."""

        return InKernelDispatchProducts(
            expert_recv_count_sum=token_comm_args.expert_recv_count_sum,
            token_src_metadata=token_comm_args.token_src_metadata,
            fc1_ready_counters=token_comm_args.fc1_ready_counter,
            token_pool=token_comm_args.fc1_input_token_buffer,
            scale_pool=token_comm_args.fc1_input_sf_buffer,
            compatibility_route_scores=(token_comm_args.fc1_input_topk_weights_buffer),
        )

    def __call__(
        self,
        token_comm_args: Any,
        token_comm_storage: Any,
        *,
        warp_idx: Any,
        lane_idx: Any,
        tidx: Any,
    ) -> None:
        """Run prep, barrier, pull, metadata, and readiness publication."""

        # The body's dispatch-to-scheduler named barrier publishes this CTA's
        # count stores even when startup omits the closing grid rendezvous.
        self.token_comm.dispatch_warp_body(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )

    def bind_local_counts(self, token_comm_args: Any, storage: Any) -> Any:
        """Reuse dead prep scratch for CTA-private counts, without growing SMEM."""
        import cutlass.cute as cute

        counts = cute.make_tensor(
            storage.smem_expert_count.data_ptr(),
            cute.make_layout((self.config.num_experts_per_rank,)),
        )
        # Keep the packed global ABI for diagnostics; only CTA 0 writes it.
        self.token_comm.global_count_sum = token_comm_args.expert_recv_count_sum
        token_comm_args.expert_recv_count_sum = counts
        return counts

    def scheduler_wait(self, token_comm_args: Any) -> None:
        """Rendezvous the scheduler with published expert counts."""

        self.token_comm.sched_warp_pre_init_wait(token_comm_args)

    def fc1_ready_counter_ptr(self, token_comm_args: Any) -> Any:
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

    def kernel_tail(
        self,
        token_comm_args: Any,
        *,
        warp_idx: Any,
        lane_idx: Any,
        tidx: Any,
    ) -> None:
        """Delegate the all-warp persistent-kernel tail rendezvous/reset."""

        # Every producer thread orders its peer route stores at system scope
        # before handing completion to the dispatch warps through the CTA
        # rendezvous. Their grid/NVLink release-acquire barriers then make those
        # stores visible to the source-rank reducer, either in the all-warp
        # tail callback or launched after kernel exit.
        if self.config.world_size > 1:
            import cutlass.cute as cute

            cute.arch.fence_acq_rel_sys()

        self.token_comm.kernel_tail(
            token_comm_args,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
            tail_work=self._tail_reduce,
        )


class SingleRankMxfp8DispatchPhase(Mxfp8DispatchPhase):
    """Compatibility name for the original standalone one-rank harness."""

    def __init__(self, config: WorkspaceConfig, **kwargs: Any) -> None:
        if config.world_size != 1:
            raise NotImplementedError(
                "SingleRankMxfp8DispatchPhase only supports world_size=1"
            )
        super().__init__(config, **kwargs)


class SingleRankMxfp8Dispatch:
    """Compile and bind the native one-rank MXFP8 dispatch kernel.

    Compilation is specialized to the workspace capacity.  Per-invocation
    token counts remain graph-safe because staging writes ``-1`` to every
    inactive route row; the device dispatcher skips those masked edges while
    preserving the order of live ``(token, top-k slot)`` pairs.
    """

    def __init__(
        self,
        config: WorkspaceConfig,
        *,
        num_sms: int | None = None,
        flag_batch: int = 1,
        cta_local_counts: bool = False,
    ) -> None:
        if config.world_size != 1:
            raise NotImplementedError(
                "SingleRankMxfp8Dispatch only supports world_size=1"
            )
        if not 1 <= flag_batch <= 32:
            raise ValueError("flag_batch must be in [1, 32]")
        if config.num_topk > 32:
            raise ValueError("num_topk must be at most 32")
        self.config = config
        self.num_sms = num_sms
        self.flag_batch = flag_batch
        self.cta_local_counts = cta_local_counts

    @staticmethod
    def _require_tensor(
        name: str,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        if tensor.dtype is not dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def _validate_inputs(
        self,
        partition: DeviceWorkspacePartition,
        activation: torch.Tensor,
        scales: torch.Tensor,
        route_ids: torch.Tensor,
        route_scores: torch.Tensor,
    ) -> None:
        if partition.plan.config != self.config:
            raise ValueError("dispatch and workspace configurations do not match")
        if not partition.shared:
            raise ValueError(
                "single-rank dispatch still requires the shared workspace regions"
            )
        device = partition.local_workspace.device
        if device.type != "cuda":
            raise ValueError("single-rank dispatch requires CUDA tensors")
        capacity = self.config.max_tokens_per_rank
        self._require_tensor(
            "activation",
            activation,
            dtype=torch.float8_e4m3fn,
            shape=(capacity, self.config.hidden),
            device=device,
        )
        self._require_tensor(
            "scales",
            scales,
            dtype=torch.float8_e8m0fnu,
            shape=(capacity, self.config.sf_uint32_per_token * 4),
            device=device,
        )
        self._require_tensor(
            "route_ids",
            route_ids,
            dtype=torch.int64,
            shape=(capacity, self.config.num_topk),
            device=device,
        )
        self._require_tensor(
            "route_scores",
            route_scores,
            dtype=torch.float32,
            shape=(capacity, self.config.num_topk),
            device=device,
        )

    def _make_launcher(self, *, num_sms: int) -> Any:
        # All CuTe imports remain behind compile(): importing the backend for
        # discovery on a CPU-only machine must not initialize the DSL/runtime.
        import cutlass
        import cutlass.cute as cute

        config = self.config
        phase = SingleRankMxfp8DispatchPhase(
            config,
            cluster_shape_mn=(1, 1),
            dispatch_warp_start=0,
            num_other_warps=1,
            flag_batch=self.flag_batch,
            cta_local_counts=self.cta_local_counts,
        )

        @cute.kernel
        def _phase_kernel(
            input_token_buffer,
            input_sf_buffer,
            input_topk_idx_buffer,
            input_topk_weights_buffer,
            expert_send_count,
            expert_recv_count,
            expert_recv_count_sum,
            src_token_topk_idx,
            token_src_metadata,
            l1_arrival_count,
            l1_token_buffer,
            l1_sf_buffer,
            l1_topk_weights_buffer,
            nvlink_barrier_signal,
            nvlink_barrier_counter,
            grid_sync_counter,
            local_zero_prefix,
            shared_zero_prefix,
            peer_rank_ptr_mapper,
        ):
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(phase.shared_storage_type())
            args = phase.make_args(
                input_token_buffer=input_token_buffer,
                input_sf_buffer=input_sf_buffer,
                topk_idx=input_topk_idx_buffer,
                input_topk_weights_buffer=input_topk_weights_buffer,
                expert_send_count=expert_send_count,
                expert_recv_count=expert_recv_count,
                expert_recv_count_sum=expert_recv_count_sum,
                src_token_topk_idx=src_token_topk_idx,
                fc1_input_token_buffer=l1_token_buffer,
                fc1_input_sf_buffer=l1_sf_buffer,
                fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
                fc1_ready_counter=l1_arrival_count,
                token_src_metadata=token_src_metadata,
                combine_output=l1_token_buffer,
                nvlink_barrier_signal=nvlink_barrier_signal,
                nvlink_barrier_counter=nvlink_barrier_counter,
                grid_sync_counter=grid_sync_counter,
                local_zero_prefix=local_zero_prefix,
                shared_zero_prefix=shared_zero_prefix,
                peer_rank_ptr_mapper=peer_rank_ptr_mapper,
                sm_count=num_sms,
            )
            if cutlass.const_expr(phase.cta_local_counts):
                phase.bind_local_counts(args, storage)
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = tidx // 32
            lane_idx = tidx % 32
            if warp_idx < phase.num_dispatch_warps:
                phase(
                    args,
                    storage,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    tidx=tidx,
                )
            elif warp_idx == phase.num_dispatch_warps:
                phase.scheduler_wait(args)

        class _Launcher:
            @cute.jit
            def __call__(
                self,
                input_token_buffer,
                input_sf_buffer,
                input_topk_idx_buffer,
                input_topk_weights_buffer,
                expert_send_count,
                expert_recv_count,
                expert_recv_count_sum,
                src_token_topk_idx,
                token_src_metadata,
                l1_arrival_count,
                l1_token_buffer,
                l1_sf_buffer,
                l1_topk_weights_buffer,
                nvlink_barrier_signal,
                nvlink_barrier_counter,
                grid_sync_counter,
                local_zero_prefix,
                shared_zero_prefix,
                peer_rank_ptr_mapper_host,
                stream,
            ):
                peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()
                _phase_kernel(
                    input_token_buffer,
                    input_sf_buffer,
                    input_topk_idx_buffer,
                    input_topk_weights_buffer,
                    expert_send_count,
                    expert_recv_count,
                    expert_recv_count_sum,
                    src_token_topk_idx,
                    token_src_metadata,
                    l1_arrival_count,
                    l1_token_buffer,
                    l1_sf_buffer,
                    l1_topk_weights_buffer,
                    nvlink_barrier_signal,
                    nvlink_barrier_counter,
                    grid_sync_counter,
                    local_zero_prefix,
                    shared_zero_prefix,
                    peer_rank_ptr_mapper,
                ).launch(
                    grid=[num_sms, 1, 1],
                    block=[160, 1, 1],
                    stream=stream,
                )

        return _Launcher()

    def compile(
        self,
        partition: DeviceWorkspacePartition,
        activation: torch.Tensor,
        scales: torch.Tensor,
        route_ids: torch.Tensor,
        route_scores: torch.Tensor,
        *,
        stream: int,
    ) -> DispatchProducts:
        """Compile once and return a zero-argument, graph-capturable launch."""

        self._validate_inputs(partition, activation, scales, route_ids, route_scores)

        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch

        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.sym_buffer import (
            SymBufferHost,
        )

        num_sms = self.num_sms
        if num_sms is None:
            properties = torch.cuda.get_device_properties(activation.device)
            num_sms = properties.multi_processor_count
        if num_sms <= 0:
            raise ValueError("num_sms must be positive")

        def to_cute(tensor: torch.Tensor, *, align: int = 16) -> Any:
            result = cutlass_torch.from_dlpack(tensor, assumed_align=align)
            leading_dim = cutlass_torch.get_leading_dim(tensor)
            return result.mark_layout_dynamic(leading_dim=leading_dim)

        local = partition.local
        shared = partition.shared
        local_zero_prefix = partition.local_workspace[
            : partition.plan.local.zero_prefix_bytes
        ].view(torch.int32)
        assert partition.shared_workspace is not None
        shared_zero_prefix = partition.shared_workspace[
            : partition.plan.shared.zero_prefix_bytes
        ].view(torch.int32)
        # Four E8M0 bytes form the uint32 atom copied by token_comm.  Re-viewing
        # is zero-copy and preserves the staged K32 byte order.
        input_sf_u32 = scales.view(torch.uint8).view(torch.int32).reshape(-1)
        runtime = {
            "input_token_buffer": to_cute(activation, align=16),
            "input_sf_buffer": to_cute(input_sf_u32, align=4),
            "input_topk_idx_buffer": to_cute(route_ids, align=8),
            "input_topk_weights_buffer": to_cute(route_scores, align=4),
            "expert_send_count": to_cute(local["expert_send_count"], align=8),
            "expert_recv_count": to_cute(shared["expert_recv_count"], align=8),
            "expert_recv_count_sum": to_cute(shared["expert_recv_count_sum"], align=8),
            "src_token_topk_idx": to_cute(shared["src_token_topk_idx"], align=4),
            "token_src_metadata": to_cute(local["token_src_metadata"], align=8),
            "l1_arrival_count": to_cute(local["l1_arrival_count"], align=4),
            "l1_token_buffer": to_cute(local["l1_token_buffer"], align=128),
            "l1_sf_buffer": to_cute(local["l1_sf_buffer"], align=16),
            "l1_topk_weights_buffer": to_cute(local["l1_topk_weights_buffer"], align=4),
            "nvlink_barrier_signal": to_cute(shared["nvlink_barrier_signal"], align=4),
            "nvlink_barrier_counter": to_cute(local["nvlink_barrier_counter"], align=4),
            "grid_sync_counter": to_cute(local["grid_sync_counter"], align=4),
            "local_zero_prefix": to_cute(local_zero_prefix, align=4),
            "shared_zero_prefix": to_cute(shared_zero_prefix, align=4),
            # For world=1 map(local, 0) == local.  base_addr is retained only
            # for the host ABI; the device mapper consumes the zero offset.
            "peer_rank_ptr_mapper_host": SymBufferHost(
                base_addr=activation.data_ptr(),
                offsets=(0,),
                rank_idx=0,
                num_max_ranks=1,
            ),
            "stream": cuda.CUstream(stream),
        }
        compiled = cute.compile(self._make_launcher(num_sms=num_sms), **runtime)
        launch_stream = torch.cuda.ExternalStream(stream, device=activation.device)

        # The two phase/signal regions live outside the counter prefixes and
        # must start from zero exactly once, then persist across launches.
        # Pool payloads do not need clearing, but whole-workspace initialization
        # here also makes the first launch safe when the allocator used empty().
        with torch.cuda.stream(launch_stream):
            partition.local_workspace.zero_()
            assert partition.shared_workspace is not None
            partition.shared_workspace.zero_()

        def launch() -> None:
            # The standalone entry has no persistent-kernel tail to reset its
            # accumulating counters.  Record the minimal prefix memsets on the
            # same stream immediately before dispatch; this remains graph-safe.
            with torch.cuda.stream(launch_stream):
                partition.zero_prefixes_()
                compiled(**runtime)

        packed_counts = shared["expert_recv_count_sum"]
        expert_counts_i32 = packed_counts.view(torch.int32)[::2]
        offsets = DeviceExpertPoolOffsets(
            expert_counts=expert_counts_i32,
            token_padding_block=self.config.token_padding_block,
            sf_padding_block=self.config.sf_padding_block,
            cluster_tile_tokens=self.config.cluster_tile_tokens,
        )
        return DispatchProducts(
            launch=launch,
            expert_recv_count_sum=packed_counts,
            expert_pool_offsets=offsets,
            token_src_metadata=local["token_src_metadata"],
            fc1_ready_counters=local["l1_arrival_count"],
            token_pool=local["l1_token_buffer"],
            scale_pool=local["l1_sf_buffer"],
            compatibility_route_scores=local["l1_topk_weights_buffer"],
        )


__all__ = [
    "DeviceExpertPoolOffsets",
    "DispatchProducts",
    "InKernelDispatchProducts",
    "Mxfp8DispatchPhase",
    "SingleRankMxfp8Dispatch",
    "SingleRankMxfp8DispatchPhase",
]
