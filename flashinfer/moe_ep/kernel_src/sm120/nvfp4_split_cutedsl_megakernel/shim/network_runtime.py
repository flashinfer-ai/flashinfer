"""Workspace-owned serving adapter for the shared GIN transport protocol."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from .comm import ensure_not_capturing


class SessionNetworkBridge:
    """Allocation/binding protocol without the standalone driver's globals."""

    def __init__(self, session):
        self.session = session
        self.arena = None
        self.expected = []
        self.index = 0
        self.cursor = 0

    def arena_zeros(self, shape, dtype):
        return self.session.zeros(shape, dtype)

    def prepare_network_arena(self, runtime, local_bytes, slots, block, payload):
        from moe_sm120_nvfp4_split.transport.gin_bridge import network_region_bytes

        if self.arena is not None:
            raise RuntimeError("Network arena already prepared")
        self.expected = network_region_bytes(
            runtime, local_bytes, slots, block, payload
        )
        capacity = sum(((n + 4095) // 4096 + 1) * 4096 for n in self.expected)
        self.arena = self.session.register_network_arena(capacity)

    def sym_zeros(self, shape, dtype):
        ensure_not_capturing("GIN bucket suballocation")
        size = math.prod(shape) * dtype.itemsize
        if self.index >= len(self.expected) or size != self.expected[self.index]:
            raise RuntimeError(f"GIN allocation order mismatch at {self.index}: {size}")
        aligned = (size + 4095) // 4096 * 4096
        start = self.cursor
        if self.arena is None or start + aligned + 4096 > self.arena.numel():
            raise MemoryError("GIN bucket arena capacity exceeded")
        self.cursor += aligned + 4096
        self.index += 1
        self.arena[start : start + aligned].zero_()
        return self.arena[start : start + size].view(dtype).reshape(shape)

    def configure(self, runtime):
        import torch.distributed as dist

        if self.index != len(self.expected):
            raise RuntimeError("Incomplete GIN network allocation plan")
        signature = (self.session.cursor, self.cursor, self.expected)
        signatures = [None] * self.session.world
        dist.all_gather_object(signatures, signature, group=self.session.control_group)
        if any(value != signature for value in signatures):
            raise RuntimeError("GIN allocation order differs across EP ranks")
        torch.cuda.synchronize()
        dist.barrier(group=self.session.control_group)
        runtime.network_backend = "gin"

    def bind(self, kernel, role):
        self.session.bind_network_kernel(kernel, role)

    @staticmethod
    def device_bitcode():
        from moe_sm120_nvfp4_split.transport.gin_runtime import device_bitcode

        return device_bitcode()


def create_network_runtime(workspace, bucket, spec, bundle, combine_output):
    # Lazy import preserves the same-NUMA path's import/initialization behavior.
    from moe_sm120_nvfp4_split.transport.network_runtime import NetworkTransportRuntime
    from moe_sm120_nvfp4_split.transport.hybrid_transport import HybridTransportState
    from moe_sm120_nvfp4_split.runtime.low_precision_dlpack import wrap
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda

    class ServingNetworkRuntime(NetworkTransportRuntime):
        def _phase_timing_begin(self, *args, **kwargs):
            return None

        def _phase_timing_end(self, *args, **kwargs):
            pass

        def begin(self):
            # Inside the reusable child graph. All old consumers are joined in
            # end(), including terminal credits, before another layer resets.
            # A previous child graph already joins its credits on the caller
            # stream. Its capture-local event cannot be reused in a different
            # layer's graph capture. The caller stream preserves this ordering.
            self._transport.ring_credit_done = None
            self.device_epoch.add_(1)
            self._reset_local_counters()
            self._ibgda_streaming_dispatch_begin()
            self._ibgda_build_runtime_owner_route_plan()
            self._ibgda_streaming_combine_begin()

        def end(self):
            self._ibgda_streaming_dispatch_end()
            self._ibgda_streaming_combine_end()
            self._wait_for_previous_streaming_replay()

        def compiled_roles(self):
            t = self._transport
            return {
                # Graph roles 0/1 are K1/K2, 2/3 are dedicated TX/RX.
                2: (
                    t.compiled_owner_plan_count_push,
                    t.compiled_owner_plan_route_push,
                    t.compiled_dispatch_ring_sender,
                    t.compiled_dispatch_ring_ack_waiter,
                    t.compiled_ring_sender,
                    t.compiled_ring_ack_waiter,
                ),
                3: (
                    t.compiled_dispatch_ring_receiver,
                    t.compiled_ring_receiver,
                    t.compiled_owner_count_builder,
                    t.compiled_owner_metadata_builder,
                ),
            }

    cfg = workspace.config
    runtime = ServingNetworkRuntime()
    runtime.bridge = SessionNetworkBridge(workspace._comm_session)
    runtime.rank, runtime.world_size = cfg.rank, cfg.world_size
    runtime.problem = SimpleNamespace(
        num_tokens_per_rank=bucket,
        num_topk=cfg.num_topk,
        hidden=cfg.hidden,
        num_experts_per_rank=cfg.local_experts,
    )
    runtime.kernel_config = spec.kernel
    runtime.comm_backend = "gin"
    runtime.rank_local_combine = spec.kernel.rank_local_combine
    runtime.hybrid_force_ibgda_mask = workspace._topology.force_ibgda_peer_mask
    runtime._kernel, runtime._kernel_k2 = bundle.k1, bundle.k2
    runtime._kernel_k2_finalizer = bundle.k2_finalizer
    runtime._transport = HybridTransportState.from_config(spec.kernel)
    runtime._transport.k1_sm_count = spec.kernel.k1_sms * spec.kernel.k1_ctas_per_sm
    runtime.my_activation = workspace.x[:bucket].view(torch.uint8)
    runtime.my_activation_sf = workspace.x_scale[:bucket]
    runtime.my_topk_idx = workspace.topk_ids[:bucket]
    runtime.my_topk_weights = workspace.topk_weights[:bucket]
    runtime.combine_output = combine_output
    runtime.rank_combine_ready = None
    runtime._rank_combine_epoch = 0
    runtime._split_green_trace = None
    runtime.local_workspace = runtime.shared_workspace = None
    runtime.device_epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
    runtime.device_epoch_cute = from_dlpack(
        wrap(runtime.device_epoch)
    ).mark_layout_dynamic()
    runtime.allocate_workspaces()
    # Ordinary streams capture all communication into a child graph. Captured
    # kernel nodes are rebound to the dedicated TX/RX Green Contexts afterwards.
    for name in (
        "dispatch_send_stream",
        "dispatch_recv_stream",
        "ring_stream",
        "ring_recv_stream",
        "owner_plan_stream",
        "route_stream",
        "owner_metadata_stream",
    ):
        if name not in workspace._transport_streams:
            workspace._transport_streams[name] = torch.cuda.Stream(
                priority=-1 if "send" in name else 0
            )
        setattr(runtime._transport, name, workspace._transport_streams[name])
    runtime._compile_ibgda_gpu_push_kernels(cute)
    t = runtime._transport
    # Allocate receiver resources before persistent waiters exist. Zero epoch
    # and zero control counts perform no transfer; this happens before capture.
    args = list(t.dispatch_ring_receiver_runtime_args)
    args[-2] = 0
    args[-1] = cuda.CUstream(t.dispatch_recv_stream.cuda_stream)
    t.compiled_dispatch_ring_receiver(*args, **runtime._epoch_kwargs())
    t.dispatch_recv_stream.synchronize()
    runtime._gin_receiver_preloaded = True
    runtime.device_epoch.fill_(1)
    torch.cuda.synchronize()
    return runtime
